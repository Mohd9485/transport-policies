# Author: Mohammad Al-Jarrah
"""
Diffusion policy with a residual network (DPR): the training objective of Diffusion Policy (a
denoising model with a Gaussian reference) on the residual MLP and the Heun integrator of FMP
and SIP, so that the effect of the reference can be separated from that of the network.

With the cosine schedule alpha_bar(tau) (tau = 0: pure noise, tau = 1: data) and
x_tau = sqrt(alpha_bar) a + sqrt(1 - alpha_bar) zeta, zeta ~ N(0, I), the network predicts zeta:

    loss = E | eps(x_tau, tau, s) - zeta |^2,   tau ~ U(0, 1).

At deployment, the probability-flow ODE of the DDIM sampler is integrated with Heun's method
from tau = 0 (x ~ N(0, I)) to tau = 1, with the predicted chunk clipped to [-1, 1] at every step.
"""
import math

import torch
import torch.nn as nn

from .common import TimeConditionedMLP, cosine_lr, set_lr, StateNoise, draw_pairs

COSINE_S, ABAR_MIN = 0.008, 1e-4           # offset of the cosine schedule; floor of alpha_bar


def alpha_bar(tau):
    """Cosine noise schedule (Nichol and Dhariwal, 2021), indexed by tau in [0, 1]."""
    f = torch.cos((1.0 - tau + COSINE_S) / (1 + COSINE_S) * math.pi / 2) ** 2
    f0 = math.cos(COSINE_S / (1 + COSINE_S) * math.pi / 2) ** 2
    return (f / f0).clamp(min=ABAR_MIN, max=1.0)


class DPR(nn.Module):
    """
    Configuration keys: width, n_blocks, time_embed_dim, time_scale, time_max_period, act,
    lr, batch_size, iterations, weight_decay, grad_clip, warmup_frac, n_steps,
    state_noise, state_noise_dims.
    """
    reference = 'gaussian'

    def __init__(self, cfg, dim_s, dim_a):
        super().__init__()
        self.cfg = cfg
        self.net = TimeConditionedMLP(dim_a, dim_s, cfg['width'], cfg['n_blocks'], cfg['time_embed_dim'],
                                      cfg['time_scale'], cfg['time_max_period'], cfg['act'])

    def fit(self, S, A, log=print):
        """Train the noise-prediction network on the pairs (S, A)."""
        c = self.cfg
        self.net.initialize()
        opt = torch.optim.AdamW(self.net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        noise = StateNoise(c.get('state_noise', 0.0), c.get('state_noise_dims'), S.shape[1], S.device)
        bs, n_it = min(c['batch_size'], A.shape[0]), c['iterations']
        self.train()
        for i in range(n_it):
            set_lr(opt, cosine_lr(i, c['lr'], n_it, c['warmup_frac']))
            a, s = draw_pairs(A, S, bs)
            s = noise(s)
            tau = torch.rand(bs, 1, device=A.device)
            ab = alpha_bar(tau)
            zeta = torch.randn_like(a)                             # the Gaussian reference
            x = ab.sqrt() * a + (1.0 - ab).clamp(min=0.0).sqrt() * zeta
            loss = ((self.net(x, tau, s) - zeta) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), c['grad_clip'])
            opt.step()
            if (i + 1) % max(1, n_it // 10) == 0:
                log('  iteration %d/%d  loss %.4f' % (i + 1, n_it, loss.item()))
        self.eval()

    @torch.no_grad()
    def sample(self, s, z=None):
        """Action chunks at states s (n, dim_s), integrated from Gaussian noise (z is not used)."""
        n, n_steps = s.shape[0], self.cfg['n_steps']
        x = torch.randn(n, self.net.heads[0].out_features, device=s.device)
        taus = torch.linspace(0.0, 1.0, n_steps + 1, device=s.device)

        def coeffs(t):
            """Signal and noise coefficients at time t."""
            ab = alpha_bar(torch.full((n, 1), float(t), device=s.device))
            return ab.sqrt(), (1.0 - ab).clamp(min=0.0).sqrt()

        def estimate(x_, t):
            """Predicted clean chunk (clipped to [-1, 1]) and the noise consistent with it."""
            alpha, sigma = coeffs(t)
            eps = self.net(x_, torch.full((n, 1), float(t), device=s.device), s)
            a_hat = ((x_ - sigma * eps) / alpha).clamp(-1.0, 1.0)
            return a_hat, (x_ - alpha * a_hat) / sigma.clamp(min=1e-8)

        for k in range(n_steps):
            t0, t1 = taus[k].item(), taus[k + 1].item()
            a0, s0 = coeffs(t0)
            a1, s1 = coeffs(t1)
            a_hat, z_hat = estimate(x, t0)
            if k < n_steps - 1:                                    # Heun step of the DDIM ODE
                a_hat2, z_hat2 = estimate(a1 * a_hat + s1 * z_hat, t1)
                z_bar = 0.5 * (z_hat + z_hat2)
                x = a1 * (x / a0 + (s1 / a1 - s0 / a0) * z_bar)
            else:                                                  # last step: DDIM (Euler)
                x = a1 * a_hat + s1 * z_hat
        return x
