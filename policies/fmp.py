# Author: Mohammad Al-Jarrah
"""
Flow matching policy (FMP).

A velocity field v(x, tau, s) is trained by conditional flow matching on the linear interpolant
between a reference chunk z ~ P_A (a demonstrated chunk drawn independently of the pair) and the
demonstrated chunk a of the pair (s, a):

    x_tau = (1 - tau) z + tau a,      loss = E | v(x_tau, tau, s) - (a - z) |^2,   tau ~ U(0, 1).

At deployment, the action chunk for an observation history s is obtained by integrating
dx/dtau = v(x, tau, s) from tau = 0 to 1 with Heun's method, starting at z ~ P_A; the map
z -> x(1) is the second block of a triangular map with reference P_A.
"""
import torch
import torch.nn as nn

from .common import TimeConditionedMLP, cosine_lr, set_lr, EMA, StateNoise, draw_pairs


class FMP(nn.Module):
    """
    Configuration keys: width, n_blocks, time_embed_dim, time_scale, time_max_period, act,
    lr, batch_size, iterations, weight_decay, grad_clip, warmup_frac, ema_decay,
    n_steps (Heun steps at deployment), state_noise, state_noise_dims.
    """
    reference = 'data'

    def __init__(self, cfg, dim_s, dim_a):
        super().__init__()
        self.cfg = cfg
        self.net = TimeConditionedMLP(dim_a, dim_s, cfg['width'], cfg['n_blocks'], cfg['time_embed_dim'],
                                      cfg['time_scale'], cfg['time_max_period'], cfg['act'])

    def fit(self, S, A, log=print):
        """Train the velocity field on the pairs (S, A)."""
        c = self.cfg
        self.net.initialize()
        opt = torch.optim.AdamW(self.net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        ema = EMA(self.net, c['ema_decay']) if c.get('ema_decay', 0) > 0 else None
        noise = StateNoise(c.get('state_noise', 0.0), c.get('state_noise_dims'), S.shape[1], S.device)
        bs, n_it = min(c['batch_size'], A.shape[0]), c['iterations']
        self.train()
        for i in range(n_it):
            set_lr(opt, cosine_lr(i, c['lr'], n_it, c['warmup_frac']))
            a, s = draw_pairs(A, S, bs)
            z = A[torch.randint(0, A.shape[0], (bs,), device=A.device)]   # reference z ~ P_A
            s = noise(s)
            tau = torch.rand(bs, 1, device=A.device)
            x = (1 - tau) * z + tau * a
            loss = ((self.net(x, tau, s) - (a - z)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), c['grad_clip'])
            opt.step()
            if ema is not None:
                ema.update(self.net)
            if (i + 1) % max(1, n_it // 10) == 0:
                log('  iteration %d/%d  loss %.4f' % (i + 1, n_it, loss.item()))
        if ema is not None:
            ema.copy_to(self.net)
        self.eval()

    @torch.no_grad()
    def sample(self, s, z):
        """Integrate the flow from the reference samples z (n, dim_a) at states s (n, dim_s)."""
        n_steps, x = self.cfg['n_steps'], z.clone()
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t0 = torch.full((x.shape[0], 1), k * dt, device=x.device)
            t1 = torch.full((x.shape[0], 1), k * dt + dt, device=x.device)
            v0 = self.net(x, t0, s)
            v1 = self.net(x + dt * v0, t1, s)                     # Heun's method
            x = x + dt * 0.5 * (v0 + v1)
        return x
