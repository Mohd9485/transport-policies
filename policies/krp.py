# Author: Mohammad Al-Jarrah
"""
Knothe-Rosenblatt policy (KRP): the stochastic map filter of Spantini et al. (2022) as a policy.

A lower-triangular monotone map z = R(s, a), with components

    R_k(s, a_1..a_k) = c_k(s, a_<k) + int_0^{a_k} softplus(g_k(s, a_<k, t)) dt + eps * a_k,

is trained by maximum likelihood, so that R(s, a) ~ N(0, I) when (s, a) ~ P_{S,A}:

    loss = E [ 0.5 |R(s, a)|^2 - sum_k log dR_k/da_k ].

The integral is computed with Gauss-Legendre quadrature. At deployment (the "pairs" sampler of
the stochastic map filter), a demonstrated pair (s_j, a_j) is drawn, mapped to z = R(s_j, a_j),
and the action chunk for the observation history s is the solution a of R(s, a) = z, obtained
component by component (a_1, then a_2, ...) with a safeguarded Newton method.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import activation, cosine_lr, set_lr, EMA, StateNoise, draw_pairs


class BatchedLinear(nn.Module):
    """dim_a independent linear layers, one per component of the map, applied in one batched call."""

    def __init__(self, n_comp, n_in, n_out, fan_in, in_mask=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_comp, n_in, n_out))
        self.bias = nn.Parameter(torch.empty(n_comp, 1, n_out))
        with torch.no_grad():
            for k in range(n_comp):
                bound = 1.0 / math.sqrt(max(1, int(fan_in[k])))
                self.weight[k].uniform_(-bound, bound)
                self.bias[k].uniform_(-bound, bound)
            if in_mask is not None:                 # zero the inputs a component may not see
                self.weight.mul_(torch.as_tensor(in_mask, dtype=torch.float32).unsqueeze(-1))

    def forward(self, x):                           # x: (n_comp, B, n_in)
        return torch.baddbmm(self.bias, x, self.weight)

    def forward_one(self, k, x):                    # component k only, x: (B, n_in)
        """Output of the layer for component k only."""
        return torch.addmm(self.bias[k], x, self.weight[k])


class TriangularMonotoneMap(nn.Module):
    """The map R(s, a); component k sees the state s and the actions a_1, ..., a_{k-1}."""

    def __init__(self, dim_s, dim_a, width, depth, act, quad_points, eps=1e-3, residual=False):
        super().__init__()
        self.n, self.eps, self.residual = dim_a, eps, residual
        n_ctx = dim_s + dim_a
        mask = np.zeros((dim_a, n_ctx), dtype=np.float32)
        mask[:, :dim_s] = 1.0
        for k in range(dim_a):
            mask[k, dim_s:dim_s + k] = 1.0
        self.register_buffer('mask', torch.as_tensor(mask).unsqueeze(1))      # (dim_a, 1, n_ctx)
        ctx_dim = dim_s + np.arange(dim_a)
        self.act = activation(act)
        L, H = dim_a, width
        self.c_layers = nn.ModuleList([BatchedLinear(L, n_ctx, H, ctx_dim, mask)]
                                      + [BatchedLinear(L, H, H, np.full(L, H)) for _ in range(depth - 1)]
                                      + [BatchedLinear(L, H, 1, np.full(L, H))])
        self.g_in_ctx = BatchedLinear(L, n_ctx, H, ctx_dim + 1, mask)
        self.g_in_t = nn.Parameter(torch.empty(L, 1, H))
        with torch.no_grad():
            for k in range(L):
                bound = 1.0 / math.sqrt(int(ctx_dim[k]) + 1)
                self.g_in_t[k].uniform_(-bound, bound)
        self.g_layers = nn.ModuleList([BatchedLinear(L, H, H, np.full(L, H)) for _ in range(depth - 1)]
                                      + [BatchedLinear(L, H, 1, np.full(L, H))])
        nodes, weights = np.polynomial.legendre.leggauss(quad_points)           # on [-1, 1]
        nodes = np.concatenate([0.5 * (nodes + 1.0), [1.0]]).astype(np.float32)  # the extra node 1
        weights = np.concatenate([0.5 * weights, [0.0]]).astype(np.float32)      # gives d R_k / d a_k
        self.register_buffer('nodes', torch.as_tensor(nodes))
        self.register_buffer('weights', torch.as_tensor(weights))

    def _lay(self, layer, k, x):
        """Apply layer to all components (k is None) or to component k only."""
        return layer(x) if k is None else layer.forward_one(k, x)

    def _c_net(self, ctx, k):
        """Conditioner network: scalar context of the monotone map of each component."""
        c = self._lay(self.c_layers[0], k, ctx)
        for layer in self.c_layers[1:-1]:
            u = self._lay(layer, k, self.act(c))
            c = c + u if self.residual else u
        return self._lay(self.c_layers[-1], k, self.act(c)).squeeze(-1)

    def _g_net(self, h, k):
        """Network of the monotone map, applied to the hidden state h."""
        for layer in self.g_layers[:-1]:
            u = self._lay(layer, k, self.act(h))
            h = h + u if self.residual else u
        return self._lay(self.g_layers[-1], k, self.act(h))

    def forward(self, s, a):
        """z = R(s, a) (B, dim_a) and log dR_k/da_k (B, dim_a)."""
        ctx = torch.cat([s, a], dim=1).unsqueeze(0) * self.mask                # (L, B, n_ctx)
        L, B, Kp = ctx.shape[0], ctx.shape[1], self.nodes.shape[0]
        c = self._c_net(ctx, None)                                              # (L, B)
        ak = a.t().unsqueeze(-1)                                                # (L, B, 1)
        t = ak * self.nodes.view(1, 1, Kp)                                      # quadrature points
        h = self.g_in_ctx(ctx).unsqueeze(2) + t.unsqueeze(-1) * self.g_in_t.unsqueeze(1)
        h = self._g_net(h.reshape(L, B * Kp, -1), None)
        slope = F.softplus(h.reshape(L, B, Kp)) + self.eps
        z = (c + ak.squeeze(-1) * (slope * self.weights.view(1, 1, Kp)).sum(-1)).t()
        return z, torch.log(slope[..., -1]).t()

    def component_context(self, k, s, a):
        """Context of component k: the state and the preceding action coordinates."""
        ctx = torch.cat([s, a], dim=1) * self.mask[k]
        return self._c_net(ctx, k), self.g_in_ctx.forward_one(k, ctx)

    def component_value_and_slope(self, k, c_k, h0_k, ak):
        """R_k and dR_k/da_k at the values ak (R,) of the k-th action, for a fixed context."""
        R, Kp = ak.shape[0], self.nodes.shape[0]
        t = ak.unsqueeze(-1) * self.nodes.view(1, Kp)
        h = h0_k.unsqueeze(1) + t.unsqueeze(-1) * self.g_in_t[k].view(1, 1, -1)
        slope = F.softplus(self._g_net(h.reshape(R * Kp, -1), k).reshape(R, Kp)) + self.eps
        return c_k + ak * (slope * self.weights.view(1, Kp)).sum(-1), slope[:, -1]


@torch.no_grad()
def invert(model, s, z, n_iters=10, lo=-3.0, hi=3.0, n_expand=20, tol=1e-6):
    """
    Solve R(s, a) = z for a, one component at a time: a bracket [lo, hi] (expanded until it
    contains the root), then Newton steps that fall back to bisection when they leave the bracket.
    """
    R, L = z.shape
    a = torch.zeros((R, L), dtype=z.dtype, device=z.device)
    for k in range(L):
        c_k, h0_k = model.component_context(k, s, a)
        zk = z[:, k]
        lo_k = torch.full((R,), float(lo), dtype=z.dtype, device=z.device)
        hi_k = torch.full((R,), float(hi), dtype=z.dtype, device=z.device)
        S_lo, _ = model.component_value_and_slope(k, c_k, h0_k, lo_k)
        S_hi, _ = model.component_value_and_slope(k, c_k, h0_k, hi_k)
        for _ in range(n_expand):
            if bool(((S_lo <= zk) & (zk <= S_hi)).all()):
                break
            lo_k = torch.where(S_lo <= zk, lo_k, 2.0 * lo_k)
            hi_k = torch.where(zk <= S_hi, hi_k, 2.0 * hi_k)
            S_lo, _ = model.component_value_and_slope(k, c_k, h0_k, lo_k)
            S_hi, _ = model.component_value_and_slope(k, c_k, h0_k, hi_k)
        ak = 0.5 * (lo_k + hi_k)
        dx_old = hi_k - lo_k
        done = torch.zeros_like(ak, dtype=torch.bool)
        for _ in range(n_iters):
            Sk, dSk = model.component_value_and_slope(k, c_k, h0_k, ak)
            above = Sk > zk
            hi_k = torch.where(above, ak, hi_k)
            lo_k = torch.where(above, lo_k, ak)
            dx = (Sk - zk) / dSk
            step = ak - dx
            ok = (step >= lo_k) & (step <= hi_k) & (2.0 * dx.abs() <= dx_old.abs())
            newly = dx.abs() <= tol
            dx_old = torch.where(ok, dx, 0.5 * (hi_k - lo_k))
            ak = torch.where(done, ak, torch.where(ok | newly, step, 0.5 * (lo_k + hi_k)))
            done = done | newly
            if bool(done.all()):
                break
        a[:, k] = ak
    return a


class KRP(nn.Module):
    """
    Configuration keys: width, depth, act, quad_points, residual, lr, batch_size, iterations,
    weight_decay, grad_clip, warmup_frac, ema_decay, state_noise, state_noise_dims.
    """
    reference = 'gaussian'                     # deployment uses the pairs sampler, not P_A

    def __init__(self, cfg, dim_s, dim_a):
        super().__init__()
        self.cfg = cfg
        self.map = TriangularMonotoneMap(dim_s, dim_a, cfg['width'], cfg['depth'], cfg['act'],
                                         cfg['quad_points'], residual=cfg.get('residual', False))
        self.pairs = None                      # (S, A) of the demonstrations, for the sampler

    def set_pairs(self, S, A):
        """The normalized demonstration pairs that the pairs sampler maps to the reference."""
        self.pairs = (S, A)

    def fit(self, S, A, log=print):
        """Train the triangular map on the pairs (S, A)."""
        c = self.cfg
        opt = torch.optim.AdamW(self.map.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
        ema = EMA(self.map, c['ema_decay']) if c.get('ema_decay', 0) > 0 else None
        noise = StateNoise(c.get('state_noise', 0.0), c.get('state_noise_dims'), S.shape[1], S.device)
        bs, n_it = min(c['batch_size'], A.shape[0]), c['iterations']
        self.train()
        for i in range(n_it):
            set_lr(opt, cosine_lr(i, c['lr'], n_it, c['warmup_frac']))
            a, s = draw_pairs(A, S, bs)
            z, log_slope = self.map(noise(s), a)
            loss = (0.5 * (z ** 2).sum(dim=1) - log_slope.sum(dim=1)).mean()      # negative log-likelihood
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.map.parameters(), c['grad_clip'])
            opt.step()
            if ema is not None:
                ema.update(self.map)
            if (i + 1) % max(1, n_it // 10) == 0:
                log('  iteration %d/%d  negative log-likelihood %.4f' % (i + 1, n_it, loss.item()))
        if ema is not None:
            ema.copy_to(self.map)
        self.set_pairs(S, A)
        self.eval()

    @torch.no_grad()
    def sample(self, s, z=None):
        """Action chunks at states s (n, dim_s) with the pairs sampler (z is not used)."""
        S, A = self.pairs
        j = torch.randint(0, S.shape[0], (s.shape[0],), device=s.device)
        z, _ = self.map(S[j], A[j])
        return invert(self.map, s, z)
