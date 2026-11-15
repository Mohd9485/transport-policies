# Author: Mohammad Al-Jarrah
"""
Consistency Policy (CP; Prasad et al., 2024), state-based, with the settings of the released
configurations: an EDM teacher (Karras et al., 2022) on the 1-D U-Net, distilled into a
consistency trajectory model (CTM; Kim et al., 2024) student. Both stages train for the same
number of gradient steps.

At deployment the student maps Gaussian noise at the largest noise level directly to an action
chunk (CP 1-step); with chaining, the chunk is re-noised at two intermediate levels and mapped
again (CP 3-step, three network evaluations).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import ConditionalUnet1D, PowerEMA
from .dp import cosine_with_warmup
from .common import StateNoise


def _expand(x, ndim):
    """Append singleton dimensions to x until it has ndim dimensions."""
    return x[(...,) + (None,) * (ndim - x.ndim)]


class KarrasSchedule:
    """Noise levels and EDM parameterization of the teacher and the CTM student."""

    def __init__(self, t_min=0.02, t_max=80.0, rho=7.0, bins=80, data_std=0.5, ode_steps_max=1):
        self.t_min, self.t_max, self.rho, self.bins = t_min, t_max, rho, bins
        self.data_std, self.ode_steps_max = data_std, ode_steps_max

    def times(self, k):
        """Noise level of the discretization index k (0 = t_max, bins - 1 = t_min)."""
        r = 1 / self.rho
        t = self.t_max ** r + k / (self.bins - 1) * (self.t_min ** r - self.t_max ** r)
        return (t ** self.rho).clamp(self.t_min, self.t_max)

    def scalings(self, t):
        """Skip, output and input scalings of the consistency parameterization at noise level t."""
        c_skip = self.data_std ** 2 / ((t - self.t_min) ** 2 + self.data_std ** 2)
        c_out = (t - self.t_min) * self.data_std / (t ** 2 + self.data_std ** 2) ** 0.5
        c_in = 1 / (t ** 2 + self.data_std ** 2) ** 0.5
        return c_skip, c_out, c_in

    @staticmethod
    def rescale(t):
        """Noise level t mapped to the time input of the network."""
        return 1000 * 0.25 * torch.log(t + 1e-44)

    def denoise(self, net, x, t, cond, clamp=False):
        """Teacher's estimate of the clean chunk from x at noise level t."""
        c_skip, c_out, c_in = [_expand(c, x.ndim) for c in self.scalings(t)]
        out = net(x * c_in, self.rescale(t), cond) * c_out + x * c_skip
        return out.clamp(-1.0, 1.0) if clamp else out

    def jump(self, net, x, t, stop, cond, clamp=False):
        """Student's map from x at noise level t to noise level stop (the CTM parameterization)."""
        c_skip, c_out, c_in = [_expand(c, x.ndim) for c in self.scalings(t)]
        g = net(x * c_in, self.rescale(t), cond, stoptime=self.rescale(stop)) * c_out + x * c_skip
        ratio = _expand(stop / t, x.ndim)
        out = x * ratio + g * (1 - ratio)
        return out.clamp(-1.0, 1.0) if clamp else out

    def add_noise(self, x, t):
        """Perturb x with Gaussian noise of standard deviation t."""
        return x + torch.randn_like(x) * _expand(t, x.ndim)

    @torch.no_grad()
    def heun_step(self, net, x, t, t_next, cond):
        """One Heun step of the teacher's probability-flow ODE from t to t_next."""
        step = _expand(t_next - t, x.ndim)
        mask = (step == 0).float()
        d = (x - self.denoise(net, x, t, cond)) / (_expand(t, x.ndim) + mask)
        x_next = x + step * d
        d_next = (x_next - self.denoise(net, x_next, t_next, cond)) / (_expand(t_next, x.ndim) + mask)
        return (x + step * (d + d_next) / 2) * (1 - mask) + x * mask

    def log_normal_times(self, n, device, p_mean=-1.2, p_std=1.2):
        """Noise levels of the teacher training, log-normally distributed."""
        return (torch.randn((n,), device=device) * p_std + p_mean).exp()

    def dsm_times(self, n, device):
        """Noise levels on the Karras schedule, used for half of each student batch."""
        r = 1 / self.rho
        xi = torch.rand((n,), device=device) * 0.7
        return ((self.t_max ** r + xi * (self.t_min ** r - self.t_max ** r)) ** self.rho).clamp(self.t_min, self.t_max)

    def ctm_indices(self, n, device):
        """Indices t <= u <= s of the start, teacher, and stop levels of a CTM training triple."""
        t = torch.randint(0, self.bins, (n,), device=device)
        s = t + (torch.rand((n,), device=device) * (self.bins + 1 - t)).long()
        u = t + (torch.rand((n,), device=device) * (s - t + 1)).long()
        return t, s, torch.minimum(u, t + self.ode_steps_max)

    def karras_weights(self, t):
        """Loss weights of the teacher at noise level t."""
        return (t ** 2 + self.data_std ** 2) / ((t * self.data_std) ** 2)


def pseudo_huber(pred, target, weights=None):
    """Pseudo-Huber loss with delta = 0.00054 sqrt(dimension of the chunk)."""
    delta = math.sqrt(math.prod(pred.shape[1:])) * 0.00054
    loss = torch.sqrt(F.mse_loss(pred, target, reduction='none') ** 2 + delta ** 2) - delta
    if weights is not None:
        loss = loss * _expand(weights, loss.ndim)
    return loss.mean()


class CP(nn.Module):
    """
    Configuration keys: act_dim, step_embed_dim, down_dims, kernel_size, n_groups, dropout,
    lr, betas, weight_decay, warmup_steps, batch_size, iterations_teacher, iterations_student,
    lr_schedule_student (length of the student's cosine schedule), chaining_times,
    state_noise, state_noise_dims. `chaining` (False: CP 1-step, True: CP 3-step) is an option
    of the deployment only.
    """
    reference = 'gaussian'

    def __init__(self, cfg, dim_s, dim_a):
        super().__init__()
        self.cfg, self.act_dim = cfg, cfg['act_dim']
        self.h_p = dim_a // self.act_dim
        self.sched = KarrasSchedule()
        unet = dict(input_dim=self.act_dim, global_cond_dim=dim_s, step_embed_dim=cfg['step_embed_dim'],
                    down_dims=cfg['down_dims'], kernel_size=cfg['kernel_size'], n_groups=cfg['n_groups'],
                    mid_dropout_rate=cfg['dropout'])
        self.teacher = ConditionalUnet1D(**unet)
        self.student = ConditionalUnet1D(dropout_rate=cfg['dropout'], with_stop_time=True, **unet)
        self.chaining = False

    def _optimizer(self, net, schedule_steps):
        """AdamW and a cosine learning-rate schedule with warmup for the network net."""
        c = self.cfg
        opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], betas=tuple(c['betas']), eps=1e-8,
                                weight_decay=c['weight_decay'])
        return opt, cosine_with_warmup(opt, c['warmup_steps'], schedule_steps)

    def fit(self, S, A, log=print):
        """Train the teacher, then distill it into the student."""
        c, sched = self.cfg, self.sched
        A = A.reshape(A.shape[0], self.h_p, self.act_dim)
        noise = StateNoise(c.get('state_noise', 0.0), c.get('state_noise_dims'), S.shape[1], S.device)
        bs, dev = min(c['batch_size'], A.shape[0]), A.device

        # ---- stage 1: EDM teacher, with the EMA of its weights
        ema = PowerEMA(self.teacher)
        opt, lr_sched = self._optimizer(self.teacher, c['iterations_teacher'])
        self.teacher.train()
        for i in range(c['iterations_teacher']):
            idx = torch.randint(0, A.shape[0], (bs,), device=dev)
            a, s = A[idx], noise(S[idx])
            t = sched.log_normal_times(bs, dev)
            loss = pseudo_huber(sched.denoise(self.teacher, sched.add_noise(a, t), t, s), a, sched.karras_weights(t))
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            ema.step(self.teacher)
            if (i + 1) % max(1, c['iterations_teacher'] // 10) == 0:
                log('  teacher iteration %d/%d  loss %.4f' % (i + 1, c['iterations_teacher'], loss.item()))

        # ---- stage 2: CTM student, initialized from the teacher (zero weights for the stop time)
        dsed, sd = c['step_embed_dim'], {}
        for k, v in self.teacher.state_dict().items():
            if 'cond_encoder.1.weight' in k:
                v = torch.cat([v[:, :dsed], torch.zeros(v.shape[0], dsed, dtype=v.dtype, device=v.device),
                               v[:, dsed:]], dim=-1)
            sd[k] = v
        self.student.load_state_dict(sd)
        target = copy.deepcopy(self.student).requires_grad_(False)
        self.teacher.requires_grad_(False)
        self.student.train(); target.train(); self.teacher.train()
        opt, lr_sched = self._optimizer(self.student, c['lr_schedule_student'])
        params, params_target = list(self.student.parameters()), list(target.parameters())
        for i in range(c['iterations_student']):
            idx = torch.randint(0, A.shape[0], (bs,), device=dev)
            a, s = A[idx], noise(S[idx])
            # CTM loss: the student's jump from t to the stop level s matches the target network's
            # jump from the teacher's one-step Heun solution at u; both are mapped to t_min
            k_t, k_s, k_u = sched.ctm_indices(bs, dev)
            t, stop, u = sched.times(k_t), sched.times(k_s), sched.times(k_u)
            x_t = sched.add_noise(a, t)
            x_u = sched.heun_step(self.teacher, x_t, t, sched.times(torch.minimum(k_t + 1, k_u)), s)
            pred = sched.jump(self.student, x_t, t, stop, s)
            with torch.no_grad():
                tgt = sched.jump(target, x_u, u, stop, s)
            t_min = torch.full_like(t, sched.t_min)
            pred = sched.jump(target, pred, stop, t_min, s)
            with torch.no_grad():
                tgt = sched.jump(target, tgt, stop, t_min, s)
            loss = pseudo_huber(pred, tgt)
            # denoising (DSM) loss on the student's jump to t_min
            half = int(math.ceil(bs / 2))
            t = torch.cat((sched.dsm_times(half, dev), sched.log_normal_times(bs - half, dev)))
            t_min = torch.full_like(t, sched.t_min)
            pred = sched.jump(self.student, sched.add_noise(a, t), t, t_min, s)
            loss = loss + pseudo_huber(pred, a, sched.karras_weights(t))
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_sched.step()
            with torch.no_grad():
                torch._foreach_copy_(params_target, params)       # target network = student
            if (i + 1) % max(1, c['iterations_student'] // 10) == 0:
                log('  student iteration %d/%d  loss %.4f' % (i + 1, c['iterations_student'], loss.item()))
        self.teacher.load_state_dict(ema.averaged_model.state_dict())
        self.eval()

    @torch.no_grad()
    def sample(self, s, z=None):
        """Action chunks at states s (n, dim_s): one jump from t_max to t_min, or three with chaining."""
        sched, n = self.sched, s.shape[0]
        x = torch.randn((n, self.h_p, self.act_dim), device=s.device)
        t_min = torch.tensor([sched.t_min], device=s.device)
        out = sched.jump(self.student, x, torch.tensor([sched.t_max], device=s.device), t_min, s, clamp=True)
        if self.chaining:
            for level in self.cfg['chaining_times']:
                t = torch.tensor([float(level)], device=s.device)
                out = sched.jump(self.student, sched.add_noise(out, t), t, t_min, s, clamp=True)
        return out.reshape(n, -1)
