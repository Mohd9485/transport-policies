# Author: Mohammad Al-Jarrah
"""
Deployment of a trained policy, and checkpoints.

An Agent turns a trained policy into the controller used in the rollouts: it normalizes the
observation history, draws the reference samples, calls the policy, and returns the action chunk
in the units of the environment. The sampling rule is the one of the paper:

    - the reference sample z is a demonstrated chunk drawn uniformly (P_A) for OTP, FMP and SIP;
      Gaussian-reference policies draw their own noise and ignore z;
    - a fresh reference sample is drawn for every chunk, from a generator seeded by the episode;
    - with n_samples > 1 ("mean of 8" on Square), n chunks are sampled for the same history,
      clipped to [-1, 1] in normalized units, and averaged.
"""
import numpy as np
import torch


class Agent:
    """Controller of the rollouts: normalization, reference draws, and sampling of action chunks."""
    def __init__(self, policy, stats, A_pool, act_dim, normalize, denormalize, n_samples=1):
        """
        policy       trained policy (policies.build_policy), on its device
        stats        {'S': (lo, hi), 'A': (lo, hi)} of the normalization used in training
        A_pool       (M, dim_a) demonstrated action chunks in environment units: the reference P_A
        act_dim      action dimension of the environment
        normalize, denormalize   the functions of the task module (envs/<task>/task.py)
        """
        self.policy, self.stats, self.act_dim, self.n = policy, stats, act_dim, n_samples
        self.normalize, self.denormalize = normalize, denormalize
        self.A_pool = A_pool
        self.device = next(policy.parameters()).device
        self.rng = np.random.default_rng(0)

    def reset(self, episode_seed):
        """Seed the reference draws (numpy) and the sampling noise (torch) of one episode."""
        self.rng = np.random.default_rng(episode_seed)
        torch.manual_seed(episode_seed)

    @torch.no_grad()
    def __call__(self, obs_history):
        """Observation history (h_o, obs_dim) -> action chunk (h_p, act_dim) in environment units."""
        s = self.normalize(np.asarray(obs_history, np.float32).reshape(1, -1), self.stats['S'])
        z = self.A_pool[self.rng.integers(0, len(self.A_pool), size=self.n)]
        z = np.clip(self.normalize(z, self.stats['A']), -1.0, 1.0)
        s_t = torch.as_tensor(s, device=self.device).repeat(self.n, 1)
        z_t = torch.as_tensor(z, device=self.device)
        y = self.policy.sample(s_t, z_t).cpu().numpy()
        y = y[0] if self.n == 1 else np.clip(y, -1.0, 1.0).mean(axis=0)
        return self.denormalize(y[None], self.stats['A'])[0].reshape(-1, self.act_dim)


def save_checkpoint(path, name, task, cfg, policy, stats):
    """Weights, configuration and normalization of a trained policy, in one file."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({'name': name, 'task': task, 'cfg': cfg,
                'state_dict': {k: v.cpu() for k, v in policy.state_dict().items()},
                'stats': {k: (np.asarray(lo), np.asarray(hi)) for k, (lo, hi) in stats.items()}}, path)


def load_checkpoint(path, device):
    """(policy in eval mode on `device`, checkpoint dict)."""
    from . import build_policy
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    dim_s, dim_a = len(ckpt['stats']['S'][0]), len(ckpt['stats']['A'][0])
    policy = build_policy(ckpt['name'], ckpt['cfg'], dim_s, dim_a)
    policy.load_state_dict(ckpt['state_dict'])
    return policy.to(device).eval(), ckpt
