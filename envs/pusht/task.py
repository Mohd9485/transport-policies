# Author: Mohammad Al-Jarrah
"""
Push-T task: demonstrations, (observation history, action chunk) pairs, normalization, and
closed-loop rollouts with the evaluation protocol of the paper.

The environment is the state-based Push-T task of Diffusion Policy (Chi et al., 2023), vendored
in `envs/pusht/vendored/` (MIT license) with two compatibility edits: `gym` is imported as
`gymnasium`, and `matplotlib.cm.get_cmap` is replaced by `pyplot.get_cmap`. The dynamics, the
keypoints, the reward, and the success threshold are those of the original implementation.

    observation   9 block keypoints (x, y) and the end-effector position   OBS_DIM = 20
    action        target position of the end effector                       ACT_DIM = 2
    episode       at most 300 steps; success when the coverage reaches 0.95

Score of an episode (the metric of the paper and of Diffusion Policy): the maximum over the
episode of the reward clip(coverage / 0.95, 0, 1); a score of 1 is a completed task.
"""
import os

import numpy as np

from .vendored.pusht_keypoints_env import PushTKeypointsEnv

OBS_DIM   = 20
ACT_DIM   = 2
MAX_STEPS = 300
ACT_LO    = np.array([12.0, 25.0])          # range of the demonstrated actions; executed
ACT_HI    = np.array([511.0, 511.0])        # actions are clipped to it

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_PATH = os.path.join(ROOT, 'data', 'pusht', 'pusht_cchi_v7_replay.zarr')

# Episode seeds: the paper reports held-out episodes 10000-10499; tuning used episodes 0-49.
HELD_OUT_SEEDS = list(range(10000, 10500))
TUNING_SEEDS   = list(range(0, 50))


def make_env(render_size=96):
    """The state-based Push-T environment (old gym API: reset() -> obs, step() -> 4-tuple)."""
    return PushTKeypointsEnv(legacy=False, agent_keypoints=False, render_size=render_size,
                             keypoint_visible_rate=1.0)


def load_demos(path=DEMO_PATH):
    """
    The 206 proficient-human demonstrations released with Diffusion Policy.

    Returns a dict with 'obs' (N, OBS_DIM), 'action' (N, ACT_DIM) and 'episode_ends' (E,), the
    exclusive end index of each demonstration. The observation is assembled as in Diffusion
    Policy: the nine keypoints, flattened, followed by the end-effector position.
    """
    import zarr
    z = zarr.open(str(path), 'r')
    keypoint = np.asarray(z['data/keypoint'])                       # (N, 9, 2)
    state    = np.asarray(z['data/state'])                          # (N, 5), first two = end effector
    obs = np.concatenate([keypoint.reshape(len(keypoint), -1), state[:, :2]], axis=1)
    return {'obs': obs.astype(np.float32),
            'action': np.asarray(z['data/action']).astype(np.float32),
            'episode_ends': np.asarray(z['meta/episode_ends'])}


def make_pairs(demos, h_o, h_p):
    """
    Training pairs (S_k, A_k) of the conditional sampling problem, one per time index k of every
    demonstration:

        S_k = (o_{k-h_o+1}, ..., o_k)        observation history, flattened   (h_o * OBS_DIM,)
        A_k = (a_k, ..., a_{k+h_p-1})        action chunk, flattened          (h_p * ACT_DIM,)

    At the start of a demonstration the first observation is repeated, and at its end the last
    action is repeated, as in Diffusion Policy; pairs never span two demonstrations.
    """
    S, A = [], []
    starts = np.concatenate([[0], demos['episode_ends'][:-1]])
    for start, end in zip(starts, demos['episode_ends']):
        obs, act = demos['obs'][start:end], demos['action'][start:end]
        n = len(obs)
        for k in range(n):
            S.append(obs[np.clip(np.arange(k - h_o + 1, k + 1), 0, n - 1)].reshape(-1))
            A.append(act[np.clip(np.arange(k, k + h_p), 0, n - 1)].reshape(-1))
    return np.asarray(S, np.float32), np.asarray(A, np.float32)


def fit_normalizer(X, eps=1e-8):
    """Per-dimension (lo, hi) of X, for the affine map onto [-1, 1] used by every policy."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    hi = np.where(hi - lo < eps, lo + eps, hi)
    return lo.astype(np.float32), hi.astype(np.float32)


def normalize(X, stats):
    """Map X from [lo, hi] to [-1, 1] per dimension."""
    lo, hi = stats
    return (2.0 * (X - lo) / (hi - lo) - 1.0).astype(np.float32)


def denormalize(X, stats):
    """Inverse of normalize: map X from [-1, 1] back to [lo, hi]."""
    lo, hi = stats
    return ((X + 1.0) / 2.0 * (hi - lo) + lo).astype(np.float32)


def rollout(env, policy, seed, h_o, h_a, render=False):
    """
    One closed-loop episode under the receding-horizon protocol of the paper: the policy receives
    the last h_o observations and returns an action chunk, whose first h_a actions are executed
    (clipped to the demonstrated range) before the policy is queried again.

    Parameters
    ----------
    env    : environment from make_env()
    policy : callable, observation history (h_o, OBS_DIM) -> action chunk (>= h_a, ACT_DIM)
    seed   : episode seed; it fixes the initial condition
    render : also return the rendered frames

    Returns
    -------
    dict with 'score' (the paper's metric), 'reward' (per step), 'success' (coverage reached
    0.95), 'steps', and 'frames' (empty unless render=True).
    """
    env.seed(seed)
    obs = np.asarray(env.reset(), dtype=np.float64)[:OBS_DIM]     # drop the visibility mask
    history = [obs] * h_o
    rewards, frames = [], []
    success, steps = False, 0
    while steps < MAX_STEPS and not success:
        chunk = np.asarray(policy(np.stack(history[-h_o:])), dtype=np.float64).reshape(-1, ACT_DIM)
        for a in chunk[:h_a]:
            raw, reward, done, _ = env.step(np.clip(a, ACT_LO, ACT_HI))
            history.append(np.asarray(raw, dtype=np.float64)[:OBS_DIM])
            rewards.append(float(reward))
            if render:
                frames.append(env.render(mode='rgb_array'))
            steps += 1
            success = bool(done)
            if success or steps >= MAX_STEPS:
                break
    rewards = np.asarray(rewards)
    return {'score': float(rewards.max()) if len(rewards) else 0.0, 'reward': rewards,
            'success': success, 'steps': steps, 'frames': frames}
