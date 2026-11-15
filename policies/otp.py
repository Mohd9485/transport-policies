# Author: Mohammad Al-Jarrah
"""
Optimal transport policy (OTP).

The policy is the map T(s, z) of the conditional optimal transport problem between the
independent coupling P_S x P_A and the joint distribution P_{S,A} of the demonstrations, with
the quadratic cost on the action. It is computed by the max-min (semi-dual) formulation

    max_f min_T  E_{(s,z) ~ P_S x P_A} [ 0.5 |z - T(s, z)|^2 - f(s, T(s, z)) ] + E_{(s,a) ~ P_{S,A}} [ f(s, a) ]

where f is a critic and T a map, both residual MLPs (Al-Jarrah et al., 2024). For each critic
update, the map takes K_in gradient steps. The independent coupling is realized by pairing the
actions of a minibatch with a permutation of its states. At deployment, the action chunk for an
observation history s is T(s, z) with z a demonstrated chunk drawn uniformly (P_A): one network
evaluation per chunk.
"""
import torch
import torch.nn as nn

from .common import ResidualBlock, activation, init_linear, cosine_lr, set_lr, EMA, StateNoise


class ResidualMLP(nn.Module):
    """Residual MLP on the concatenation (x, s), with `out_dim` outputs."""

    def __init__(self, dim_a, dim_s, width, n_blocks, out_dim, act):
        super().__init__()
        self.activation = activation(act)
        self.layer_input = nn.Linear(dim_a + dim_s, width, bias=False)
        self.resblocks = nn.ModuleList([ResidualBlock(width, self.activation) for _ in range(n_blocks)])
        self.layer_out = nn.Linear(width, out_dim, bias=False)

    def forward(self, x, s):
        h = self.layer_input(torch.cat((x, s), dim=1))
        for block in self.resblocks:
            h = block(h)
        return self.layer_out(self.activation(h))


class TransportMap(ResidualMLP):
    """T(s, z) = z + Delta(z, s): the map predicts the displacement of the reference sample."""

    def forward(self, z, s):
        return z + super().forward(z, s)


class OTP(nn.Module):
    """
    Configuration keys (configs/<task>.py):
        width, n_blocks       [critic, map] widths and numbers of residual blocks
        lr                    [critic, map] learning rates (AdamW)
        batch_size, iterations, k_in, weight_decay, grad_clip, warmup_frac, act
        ema_decay             EMA of the map's weights (0: none)
        state_noise, state_noise_dims   optional noise on the conditioning states (Square)
    """
    reference = 'data'                         # deployment draws z from the demonstrated chunks

    def __init__(self, cfg, dim_s, dim_a):
        super().__init__()
        self.cfg = cfg
        self.critic = ResidualMLP(dim_a, dim_s, cfg['width'][0], cfg['n_blocks'][0], 1, cfg['act'])
        self.map = TransportMap(dim_a, dim_s, cfg['width'][1], cfg['n_blocks'][1], dim_a, cfg['act'])

    def fit(self, S, A, log=print):
        """Train on normalized pairs S (M, dim_s), A (M, dim_a), torch tensors on the device."""
        c = self.cfg
        for net in (self.critic, self.map):
            net.apply(init_linear)
            nn.init.zeros_(net.layer_out.weight)                  # start at f = 0, T = identity
        opt_f = torch.optim.AdamW(self.critic.parameters(), lr=c['lr'][0], weight_decay=c['weight_decay'])
        opt_T = torch.optim.AdamW(self.map.parameters(), lr=c['lr'][1], weight_decay=c['weight_decay'])
        ema = None                                             # EMA of the map, started after the first step
        noise = StateNoise(c.get('state_noise', 0.0), c.get('state_noise_dims'), S.shape[1], S.device)
        M, bs, n_it = A.shape[0], min(c['batch_size'], A.shape[0]), c['iterations']
        torch.randperm(M, device=A.device)                     # unused draw, kept so that a seed
        self.train()                                           # reproduces the paper's checkpoints
        for i in range(n_it):
            set_lr(opt_f, cosine_lr(i, c['lr'][0], n_it, c['warmup_frac']))
            set_lr(opt_T, cosine_lr(i, c['lr'][1], n_it, c['warmup_frac']))
            idx = torch.randperm(M, device=A.device)[:bs]
            a, s = A[idx], noise(S[idx])
            s_ind = s[torch.randperm(bs, device=A.device)]     # (s_ind, a) ~ P_S x P_A
            for _ in range(c['k_in']):                         # map: min over T
                Ta = self.map(a, s_ind)
                loss_T = 0.5 * ((a - Ta) ** 2).sum(dim=1).mean() - self.critic(Ta, s_ind).mean()
                opt_T.zero_grad()
                loss_T.backward()
                torch.nn.utils.clip_grad_norm_(self.map.parameters(), c['grad_clip'])
                opt_T.step()
            loss_f = self.critic(self.map(a, s_ind), s_ind).mean() - self.critic(a, s).mean()
            opt_f.zero_grad()                                  # critic: max over f
            loss_f.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c['grad_clip'])
            opt_f.step()
            if c.get('ema_decay', 0) > 0:
                if ema is None:
                    ema = EMA(self.map, c['ema_decay'])
                else:
                    ema.update(self.map)
            if (i + 1) % max(1, n_it // 10) == 0:
                log('  iteration %d/%d  map loss %.4f  critic loss %.4f' % (i + 1, n_it, loss_T.item(), loss_f.item()))
        if ema is not None:
            ema.copy_to(self.map)
        self.eval()

    @torch.no_grad()
    def sample(self, s, z):
        """Action chunks T(s, z) for states s (n, dim_s) and reference samples z (n, dim_a)."""
        return self.map(z, s)
