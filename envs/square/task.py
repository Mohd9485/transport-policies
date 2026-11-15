# Author: Mohammad Al-Jarrah
"""
Square task (robomimic NutAssemblySquare, proficient-human demonstrations, low-dimensional
observations, absolute actions): demonstrations, (observation history, action chunk) pairs,
normalization, and closed-loop rollouts with the evaluation protocol of the paper.

    observation   object pose (14), end-effector position (3) and quaternion (4),
                  gripper joint positions (2)                                      OBS_DIM = 23
    action        end-effector position (3), rotation in the 6-D representation of Zhou et al.
                  (2019) (6), gripper command (1)                                   ACT_DIM = 10
    episode       at most 400 steps; success when the nut is placed on the square peg

The policies predict the 10-D action; it is converted to the 7-D action of the robosuite
controller (position, axis-angle rotation, gripper) before execution, as in Diffusion Policy.
Requires robosuite 1.2.0 and robomimic 0.2.0 (see README.md).
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, 'data', 'square', 'ph', 'low_dim_abs.hdf5')

OBS_KEYS  = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
OBS_DIM   = 23
ACT_DIM   = 10
MAX_STEPS = 400

# State dimensions that carry the object (nut) pose, in the observation history of h_o = 2
# steps; the state noise of the paper is added to these dimensions only.
OBJECT_DIMS = list(range(14, 23)) + list(range(37, 46))

HELD_OUT_SEEDS = list(range(10000, 10500))
TUNING_SEEDS   = list(range(0, 50))


# ---------------------------------------------------------------- rotation representations
def axis_angle_to_6d(rotvec):
    """Axis-angle (N, 3) -> first two rows of the rotation matrix, flattened (N, 6)."""
    return Rotation.from_rotvec(rotvec).as_matrix()[:, :2, :].reshape(-1, 6)


def rot6d_to_axis_angle(r6):
    """6-D rotation (N, 6) -> axis-angle (N, 3), with Gram-Schmidt orthonormalization."""
    a1, a2 = r6[:, :3], r6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    return Rotation.from_matrix(np.stack([b1, b2, np.cross(b1, b2)], axis=1)).as_rotvec()


def from_env_action(a7):
    """Controller action (N, 7) -> policy action (N, 10)."""
    a7 = np.asarray(a7, dtype=np.float64).reshape(-1, 7)
    return np.concatenate([a7[:, :3], axis_angle_to_6d(a7[:, 3:6]), a7[:, 6:]], axis=1)


def to_env_action(a10):
    """Policy action (N, 10) -> controller action (N, 7)."""
    a10 = np.asarray(a10, dtype=np.float64).reshape(-1, ACT_DIM)
    return np.concatenate([a10[:, :3], rot6d_to_axis_angle(a10[:, 3:9]), a10[:, 9:]], axis=1)


# ---------------------------------------------------------------- data
def load_demos(path=DATA_PATH):
    """The 200 proficient-human demonstrations: 'obs' (N, 23), 'action' (N, 10), 'episode_ends'."""
    import h5py
    obs, act, ends = [], [], []
    with h5py.File(path, 'r') as f:
        for k in sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1])):
            g = f['data'][k]
            obs.append(np.concatenate([g['obs'][key][:] for key in OBS_KEYS], axis=1))
            act.append(from_env_action(g['actions'][:]))
            ends.append((ends[-1] if ends else 0) + len(act[-1]))
    return {'obs': np.concatenate(obs).astype(np.float32),
            'action': np.concatenate(act).astype(np.float32),
            'episode_ends': np.asarray(ends)}


def make_pairs(demos, h_o, h_p):
    """Training pairs (S_k, A_k), constructed as in envs/pusht/task.py."""
    S, A = [], []
    starts = np.concatenate([[0], demos['episode_ends'][:-1]])
    for start, end in zip(starts, demos['episode_ends']):
        obs, act = demos['obs'][start:end], demos['action'][start:end]
        n = len(obs)
        for k in range(n):
            S.append(obs[np.clip(np.arange(k - h_o + 1, k + 1), 0, n - 1)].reshape(-1))
            A.append(act[np.clip(np.arange(k, k + h_p), 0, n - 1)].reshape(-1))
    return np.asarray(S, np.float32), np.asarray(A, np.float32)


def fit_normalizer(X, which='A', scheme='minmax', eps=1e-8):
    """
    Per-dimension (lo, hi) of the affine map onto [-1, 1].

    scheme 'minmax' : the range of every dimension (all transport policies, and DPR).
    scheme 'dp'     : the normalization of Diffusion Policy's low-dimensional Square configuration,
                      used for DP and CP: the states are divided by their largest absolute value;
                      the positions of the actions are scaled by their range and the rotation and
                      gripper entries are left unchanged.
    """
    lo, hi = X.min(axis=0), X.max(axis=0)
    hi = np.where(hi - lo < eps, lo + eps, hi)
    if scheme == 'minmax':
        return lo.astype(np.float32), hi.astype(np.float32)
    d = X.shape[1]
    if which == 'S':
        m = np.float32(np.abs(X).max())
        return -m * np.ones(d, np.float32), m * np.ones(d, np.float32)
    pos = (np.arange(d) % ACT_DIM) < 3
    out_lo, out_hi = -np.ones(d, np.float32), np.ones(d, np.float32)
    out_lo[pos], out_hi[pos] = lo[pos], hi[pos]
    return out_lo, out_hi


def normalize(X, stats):
    """Map X from [lo, hi] to [-1, 1] per dimension."""
    lo, hi = stats
    return (2.0 * (X - lo) / (hi - lo) - 1.0).astype(np.float32)


def denormalize(X, stats):
    """Inverse of normalize: map X from [-1, 1] back to [lo, hi]."""
    lo, hi = stats
    return ((X + 1.0) / 2.0 * (hi - lo) + lo).astype(np.float32)


# ---------------------------------------------------------------- environment
def make_env(render=False):
    """The robomimic Square environment with the absolute-pose controller of the data set."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': OBS_KEYS})
    meta = FileUtils.get_env_metadata_from_dataset(DATA_PATH)
    meta['env_kwargs']['controller_configs']['control_delta'] = False
    return EnvUtils.create_env_from_metadata(env_meta=meta, render=False, render_offscreen=render,
                                             use_image_obs=False)


def get_obs(raw):
    """Flat state vector of a robosuite observation (the keys OBS_KEYS, in order)."""
    return np.concatenate([np.asarray(raw[k], dtype=np.float32).reshape(-1) for k in OBS_KEYS])


def rollout(env, policy, seed, h_o, h_a, bounds):
    """
    One closed-loop episode: the policy receives the last h_o observations and returns an action
    chunk, whose first h_a actions are executed (clipped to the demonstrated range `bounds`)
    before the policy is queried again. The episode ends at success or after MAX_STEPS steps.

    Returns a dict with 'success', 'score' (1.0 or 0.0), 'steps', and the first and last
    observations 'start_obs' and 'end_obs'.
    """
    np.random.seed(seed)                                    # robosuite draws its initial state from numpy
    obs = get_obs(env.reset())
    start_obs, history = obs, [obs] * h_o
    success, steps = False, 0
    while steps < MAX_STEPS and not success:
        chunk = np.asarray(policy(np.stack(history[-h_o:])), dtype=np.float64).reshape(-1, ACT_DIM)
        for a in chunk[:h_a]:
            raw, _, _, _ = env.step(to_env_action(np.clip(a, bounds[0], bounds[1]))[0])
            history.append(get_obs(raw))
            steps += 1
            success = bool(env.is_success()['task'])
            if success or steps >= MAX_STEPS:
                break
    return {'success': success, 'score': float(success), 'steps': steps,
            'start_obs': start_obs, 'end_obs': history[-1]}
