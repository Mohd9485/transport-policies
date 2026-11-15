# Author: Mohammad Al-Jarrah
"""
Stochastic interpolant policy (SIP).

The stochastic interpolant (Albergo et al., 2023) between a reference chunk z ~ P_A and the
demonstrated chunk a of the pair (s, a),

    x_tau = (1 - tau) z + tau a + gamma(tau) zeta,   gamma(tau) = eps sqrt(tau (1 - tau)),   zeta ~ N(0, I),

defines a drift b and a denoiser eta, learned by one network with two heads:

    loss = E | b(x_tau, tau, s) - (a - z + gamma'(tau) zeta) |^2 + lambda E | eta(x_tau, tau, s) - zeta |^2.

At deployment, the action chunk for an observation history s is obtained by integrating the SDE

    dx = [ b(x, tau, s) - (gamma(tau) / 2) eta(x, tau, s) ] dtau + gamma(tau) dW

from tau = 0 to 1 with the stochastic Heun method, starting at z ~ P_A.
"""
import numpy as np
import torch
import torch.nn as nn

from .common import TimeConditionedMLP, cosine_lr, set_lr, EMA, StateNoise, draw_pairs

T_EPS = 1e-3                               # training times are drawn in [T_EPS, 1 - T_EPS]


class SIP(nn.Module):
    """
    Configuration keys: width, n_blocks, time_embed_dim, time_scale, time_max_period, act,
    noise_level (eps), denoiser_weight (lambda), lr, batch_size, iterations, weight_decay,
    grad_clip, warmup_frac, ema_decay, n_steps, state_noise, state_noise_dims.
    """
    reference = 'data'

    def __init__(self, cfg, dim_s, dim_a):
        super().__init__()
        self.cfg = cfg
        self.net = TimeConditionedMLP(dim_a, dim_s, cfg['width'], cfg['n_blocks'], cfg['time_embed_dim'],
                                      cfg['time_scale'], cfg['time_max_period'], cfg['act'], heads=2)

    def gamma(self, tau):
        """Noise level of the stochastic interpolant at time tau."""
        return self.cfg['noise_level'] * torch.sqrt(tau * (1.0 - tau))

    def gamma_dot(self, tau):
        """Time derivative of gamma."""
        return self.cfg['noise_level'] * (1.0 - 2.0 * tau) / (2.0 * (torch.sqrt(tau * (1.0 - tau)) + 1e-8))

    def fit(self, S, A, log=print):
        """Train the drift and denoiser on the pairs (S, A)."""
        c = self.cfg
        torch.randperm(A.shape[0], device=A.device)           # unused draw, kept so that a seed
        self.net.initialize()                                  # reproduces the paper's checkpoints
        opt = torch.optim.AdamW(self.net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        ema = EMA(self.net, c['ema_decay'], warmup=True) if c.get('ema_decay', 0) > 0 else None
        noise = StateNoise(c.get('state_noise', 0.0), c.get('state_noise_dims'), S.shape[1], S.device)
        bs, n_it = min(c['batch_size'], A.shape[0]), c['iterations']
        self.train()
        for i in range(n_it):
            set_lr(opt, cosine_lr(i, c['lr'], n_it, c['warmup_frac']))
            a, s = draw_pairs(A, S, bs)
            z = A[torch.randint(0, A.shape[0], (bs,), device=A.device)]   # reference z ~ P_A
            s = noise(s)
            tau = T_EPS + (1.0 - 2.0 * T_EPS) * torch.rand(bs, 1, device=A.device)
            zeta = torch.randn_like(a)
            x = (1 - tau) * z + tau * a + self.gamma(tau) * zeta
            b, eta = self.net(x, tau, s)
            loss = (((b - (a - z + self.gamma_dot(tau) * zeta)) ** 2).mean()
                    + c['denoiser_weight'] * ((eta - zeta) ** 2).mean())
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
        """Integrate the SDE from the reference samples z (n, dim_a) at states s (n, dim_s)."""
        n_steps, x = self.cfg['n_steps'], z.clone()
        dt = 1.0 / n_steps
        sqrt_dt = np.sqrt(dt)
        for k in range(n_steps):
            t0 = torch.full((x.shape[0], 1), k * dt, device=x.device)
            t1 = torch.full((x.shape[0], 1), k * dt + dt, device=x.device)
            b0, eta0 = self.net(x, t0, s)
            g0 = self.gamma(t0)
            drift0 = b0 - 0.5 * g0 * eta0
            xi = torch.randn_like(x)                              # Brownian increment / sqrt(dt)
            x_pred = x + drift0 * dt + g0 * sqrt_dt * xi
            b1, eta1 = self.net(x_pred, t1, s)                    # stochastic Heun
            drift1 = b1 - 0.5 * self.gamma(t1) * eta1
            x = x + 0.5 * (drift0 + drift1) * dt + g0 * sqrt_dt * xi
        return x
