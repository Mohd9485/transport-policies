# Author: Mohammad Al-Jarrah
"""
The conditional 1-D temporal U-Net of Diffusion Policy (Chi et al., 2023), used by DP and CP.

The action chunk (B, h_p, act_dim) is processed as a sequence along time; the diffusion time and
the observation history enter every residual block through FiLM (a per-channel scale and bias).
CP's student additionally receives the stop time of the consistency trajectory model and uses
dropout in its residual blocks (Prasad et al., 2024).
"""
import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of the diffusion time."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=x.device) * -(math.log(10000) / (half - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    """Halve the temporal length with a strided convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    """Double the temporal length with a transposed convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, c_in, c_out, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                                   nn.GroupNorm(n_groups, c_out), nn.Mish())

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two convolution blocks with FiLM conditioning between them, and a residual connection."""

    def __init__(self, c_in, c_out, cond_dim, kernel_size, n_groups, dropout_rate=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(c_in, c_out, kernel_size, n_groups),
                                     Conv1dBlock(c_out, c_out, kernel_size, n_groups)])
        self.out_channels = c_out
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * c_out))     # scale and bias
        self.residual_conv = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.scale_dropout = nn.Dropout(dropout_rate)
        self.bias_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, cond):
        out = self.scale_dropout(self.blocks[0](x))
        film = self.cond_encoder(cond).reshape(x.shape[0], 2, self.out_channels, 1)
        out = self.blocks[1](film[:, 0] * out + film[:, 1])
        return self.bias_dropout(out + self.residual_conv(x))


class ConditionalUnet1D(nn.Module):
    """
    input_dim: action dimension; global_cond_dim: dimension of the flattened observation history;
    dropout_rate: dropout of the down- and up-sampling blocks; mid_dropout_rate: dropout of the two
    middle blocks (0.2 in Consistency Policy, for the teacher and the student alike);
    with_stop_time: CP's student, which also conditions on the stop time.
    """

    def __init__(self, input_dim, global_cond_dim, step_embed_dim=256, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8, dropout_rate=0.0, mid_dropout_rate=0.0, with_stop_time=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        self.with_stop_time = with_stop_time
        self.diffusion_step_encoder = nn.Sequential(SinusoidalPosEmb(step_embed_dim),
                                                    nn.Linear(step_embed_dim, 4 * step_embed_dim), nn.Mish(),
                                                    nn.Linear(4 * step_embed_dim, step_embed_dim))
        cond_dim = step_embed_dim * (2 if with_stop_time else 1) + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        def block(c_in, c_out, p=0.0):
            return ConditionalResidualBlock1D(c_in, c_out, cond_dim, kernel_size, n_groups, p)

        self.mid_modules = nn.ModuleList([block(all_dims[-1], all_dims[-1], mid_dropout_rate),
                                          block(all_dims[-1], all_dims[-1], mid_dropout_rate)])
        self.down_modules = nn.ModuleList([
            nn.ModuleList([block(c_in, c_out, dropout_rate), block(c_out, c_out, dropout_rate),
                           Downsample1d(c_out) if k < len(in_out) - 1 else nn.Identity()])
            for k, (c_in, c_out) in enumerate(in_out)])
        self.up_modules = nn.ModuleList([
            nn.ModuleList([block(2 * c_out, c_in, dropout_rate), block(c_in, c_in, dropout_rate),
                           Upsample1d(c_in) if k < len(in_out) - 1 else nn.Identity()])
            for k, (c_in, c_out) in enumerate(reversed(in_out[1:]))])
        self.final_conv = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
                                        nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, sample, timestep, global_cond, stoptime=None):
        """sample (B, h_p, input_dim), timestep (B,) or scalar, global_cond (B, global_cond_dim)."""
        B = sample.shape[0]
        x = sample.transpose(1, 2)
        timestep = torch.as_tensor(timestep, device=sample.device)
        features = [self.diffusion_step_encoder(timestep.reshape(-1).expand(B))]
        if self.with_stop_time:
            features.append(self.diffusion_step_encoder(stoptime.reshape(-1).expand(B)))
        cond = torch.cat(features + [global_cond], dim=-1)
        skips = []
        for res1, res2, down in self.down_modules:
            x = res2(res1(x, cond), cond)
            skips.append(x)
            x = down(x)
        for mid in self.mid_modules:
            x = mid(x, cond)
        for res1, res2, up in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = up(res2(res1(x, cond), cond))
        return self.final_conv(x).transpose(1, 2)


class PowerEMA:
    """
    The EMA of Diffusion Policy: decay 1 - (1 + step)^(-0.75), capped at 0.9999, applied to a
    copy of the network (the averaged network is the one deployed).
    """

    def __init__(self, model, power=0.75, max_value=0.9999):
        import copy
        self.averaged_model = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.max_value, self.step_count = power, max_value, 0

    @torch.no_grad()
    def step(self, model):
        """Update the averaged model with the current weights of model."""
        k = max(0, self.step_count - 1)
        decay = 0.0 if k <= 0 else max(0.0, min(1 - (1 + k) ** -self.power, self.max_value))
        for p, q in zip(model.parameters(), self.averaged_model.parameters()):
            q.mul_(decay).add_(p.data, alpha=1 - decay)
        self.step_count += 1
