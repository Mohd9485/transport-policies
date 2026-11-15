# Author: Mohammad Al-Jarrah
"""
Building blocks shared by the policies: residual MLP blocks, the sinusoidal time embedding, the
learning-rate schedule, the exponential moving average of the weights, and the state noise.
"""
import math

import numpy as np
import torch
import torch.nn as nn


def activation(name):
    """The activation module named in a configuration."""
    return {'mish': nn.Mish(), 'relu': nn.ReLU(), 'silu': nn.SiLU(), 'tanh': nn.Tanh()}[name]


def init_linear(module):
    """Initialization of the residual MLPs: Xavier-uniform weights and biases equal to 0.1."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.1)


def timestep_embedding(t, dim, max_period):
    """Sinusoidal embedding (B, dim) of the scalars t (B, 1), as in transformer positional codes."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
    args = t.float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResidualBlock(nn.Module):
    """x -> act(x + W2 act(norm(W1 x))): the block of every residual MLP in the paper."""

    def __init__(self, width, act, use_layernorm=False):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width) if use_layernorm else None
        self.activation = act

    def forward(self, x):
        out = self.linear1(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.linear2(self.activation(out))
        return self.activation(out + x)


class TimeConditionedMLP(nn.Module):
    """
    Residual MLP f(x, tau, s) with a time input, used by FMP, SIP and DPR: the input is the
    concatenation of the transported chunk x, the sinusoidal embedding of tau, and the state s.
    `heads` output layers of width dim_a share the trunk (SIP uses two: velocity and denoiser).
    """

    def __init__(self, dim_a, dim_s, width, n_blocks, time_embed_dim, time_scale, time_max_period,
                 act='mish', use_layernorm=True, heads=1):
        super().__init__()
        self.activation = activation(act)
        self.time_embed_dim, self.time_scale, self.time_max_period = time_embed_dim, time_scale, time_max_period
        self.layer_input = nn.Linear(dim_a + time_embed_dim + dim_s, width, bias=False)
        self.resblocks = nn.ModuleList([ResidualBlock(width, self.activation, use_layernorm)
                                        for _ in range(n_blocks)])
        self.heads = nn.ModuleList([nn.Linear(width, dim_a, bias=False) for _ in range(heads)])

    def forward(self, x, tau, s):
        t_feat = timestep_embedding(tau * self.time_scale, self.time_embed_dim, self.time_max_period)
        h = self.layer_input(torch.cat([x, t_feat, s], dim=1))
        for block in self.resblocks:
            h = block(h)
        h = self.activation(h)
        out = [head(h) for head in self.heads]
        return out[0] if len(out) == 1 else out

    def initialize(self):
        """Xavier initialization, with the output layers set to zero (the field starts at zero)."""
        self.apply(init_linear)
        for head in self.heads:
            nn.init.zeros_(head.weight)


def cosine_lr(i, lr, iterations, warmup_frac):
    """
    Learning rate at iteration i: a linear warm-up over the first warmup_frac * iterations steps,
    then a cosine decay from lr to lr / 1000 over the training budget.
    """
    n_warm = int(warmup_frac * iterations)
    if i < n_warm:
        return lr * (i + 1) / n_warm
    lr_min = lr * 1e-3
    return lr_min + (lr - lr_min) * (1 + math.cos(math.pi * (i - n_warm) / max(1, iterations))) / 2


def set_lr(optimizer, lr):
    """Set the learning rate of all parameter groups of the optimizer."""
    for group in optimizer.param_groups:
        group['lr'] = lr


class EMA:
    """
    Exponential moving average of a module's parameters.
    With warmup=True the decay of the k-th update is min(decay, (1 + k) / (10 + k)), so that the
    random initialization does not persist in the average of a long-memory EMA.
    """

    def __init__(self, module, decay, warmup=False):
        self.decay, self.warmup, self.k = decay, warmup, 0
        self.shadow = {n: p.detach().clone() for n, p in module.named_parameters()}

    @torch.no_grad()
    def update(self, module):
        """Move the averaged parameters towards the current parameters of the module."""
        d = min(self.decay, (1.0 + self.k) / (10.0 + self.k)) if self.warmup else self.decay
        self.k += 1
        for n, p in module.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def copy_to(self, module):
        """Replace the module's parameters by their average (done once, at the end of training)."""
        for n, p in module.named_parameters():
            p.copy_(self.shadow[n])


class StateNoise:
    """
    Gaussian noise added to the conditioning states of every training minibatch (the "state
    noise" of the Square experiments): standard deviation `std` (a scalar or one value per
    state dimension, in normalized units) on the dimensions `dims` only (None: all dimensions).
    """

    def __init__(self, std, dims, dim_s, device):
        std = np.broadcast_to(np.asarray(std, dtype=np.float32), (dim_s,)).copy()
        mask = np.ones(dim_s, np.float32) if dims is None else np.isin(np.arange(dim_s), dims).astype(np.float32)
        self.std = torch.as_tensor(std * mask, device=device)
        self.on = bool((self.std > 0).any())

    def __call__(self, s):
        return s + self.std * torch.randn_like(s) if self.on else s


def draw_pairs(A, S, batch_size):
    """A minibatch of (action chunk, state) pairs, drawn uniformly with replacement."""
    idx = torch.randint(0, A.shape[0], (batch_size,), device=A.device)
    return A[idx], S[idx]


def count_parameters(module):
    """Number of parameters of a module."""
    return sum(p.numel() for p in module.parameters())
