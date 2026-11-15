# Author: Mohammad Al-Jarrah
"""
Training and evaluation of one policy on one task: the functions behind scripts/train.py,
scripts/evaluate.py, scripts/tune.py and the notebooks.

    from experiment import train, evaluate
    path = train('pusht', 'otp', seed=0)                       # trains and saves a checkpoint
    res  = evaluate('pusht', 'otp', seed=0, episodes=range(10000, 10050))

Checkpoints: checkpoints/<task>/<policy>[_noise]_seed<k>.pt
Results:     results/<task>/<policy>[_noise][_mean8][_3step]_seed<k>.json
"""
import os
import copy
import json
import time
import importlib

import numpy as np
import torch

from policies import build_policy, Agent, save_checkpoint, load_checkpoint, NAMES

ROOT = os.path.dirname(os.path.abspath(__file__))
TASKS = ('pusht', 'square')


def task_module(task):
    """The task's data, normalization and rollout functions (envs/<task>/task.py)."""
    return importlib.import_module('envs.%s.task' % task)


def get_config(task, policy, state_noise=False):
    """
    The configuration of `policy` on `task` (configs/<task>.py), with the state noise of the
    Square experiments when state_noise=True.
    """
    module = importlib.import_module('configs.%s' % task)
    cfg = copy.deepcopy(module.CONFIGS[policy])
    if state_noise:
        if task != 'square':
            raise ValueError('the state noise is a modification of the Square experiments')
        cfg['state_noise'] = module.STATE_NOISE
        cfg['state_noise_dims'] = task_module(task).OBJECT_DIMS
    return cfg


def default_device():
    """The first GPU if one is available, else the CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tag(policy, seed, state_noise=False, mean_of=1, cp_steps=1):
    """File name of a checkpoint (mean_of, cp_steps = 1) or of an evaluation."""
    name = policy + ('_noise' if state_noise else '') + ('_mean%d' % mean_of if mean_of > 1 else '')
    name += '_3step' if (policy == 'cp' and cp_steps == 3) else ''
    return '%s_seed%d' % (name, seed)


def checkpoint_path(task, policy, seed, state_noise=False):
    """Path of the checkpoint of a policy: checkpoints/<task>/<tag>.pt."""
    return os.path.join(ROOT, 'checkpoints', task, tag(policy, seed, state_noise) + '.pt')


def prepare_data(task, cfg):
    """
    Demonstration pairs (S, A) at the policy's horizons, and the normalization statistics:
    min-max to [-1, 1] for every dimension, or the 'dp' scheme of Diffusion Policy on Square.
    """
    T = task_module(task)
    S, A = T.make_pairs(T.load_demos(), cfg['h_o'], cfg['h_p'])
    if task == 'square':
        scheme = cfg.get('normalization', 'minmax')
        stats = {'S': T.fit_normalizer(S, 'S', scheme), 'A': T.fit_normalizer(A, 'A', scheme)}
    else:
        stats = {'S': T.fit_normalizer(S), 'A': T.fit_normalizer(A)}
    return S, A, stats


def _state_noise_in_units(task, cfg, S, stats):
    """The state noise is specified in min-max units; express it in the units of `stats`."""
    if cfg.get('state_noise', 0) and task == 'square' and cfg.get('normalization', 'minmax') != 'minmax':
        lo, hi = task_module(task).fit_normalizer(S, 'S', 'minmax')
        cfg['state_noise'] = (cfg['state_noise'] * (hi - lo) / (stats['S'][1] - stats['S'][0])).tolist()
    return cfg


def train(task, policy, seed=0, state_noise=False, device=None, cfg=None, save=True, log=print):
    """
    Train one policy with one seed and save the checkpoint. Returns (checkpoint path, seconds).
    `cfg` overrides the configuration of configs/<task>.py (used by the tuning script).
    """
    device = device or default_device()
    cfg = cfg or get_config(task, policy, state_noise)
    torch.manual_seed(seed)
    np.random.seed(seed)
    S, A, stats = prepare_data(task, cfg)
    T = task_module(task)
    fit_cfg = _state_noise_in_units(task, copy.deepcopy(cfg), S, stats)
    model = build_policy(policy, fit_cfg, S.shape[1], A.shape[1]).to(device)
    S_t = torch.as_tensor(T.normalize(S, stats['S']), device=device)
    A_t = torch.as_tensor(T.normalize(A, stats['A']), device=device)
    log('%s on %s, seed %d: %d pairs, dim(S) = %d, dim(A) = %d, device %s'
        % (NAMES[policy], task, seed, len(S), S.shape[1], A.shape[1], device))
    t0 = time.time()
    model.fit(S_t, A_t, log=log)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.time() - t0
    log('trained in %.1f min' % (seconds / 60))
    path = checkpoint_path(task, policy, seed, state_noise)
    if save:
        save_checkpoint(path, policy, task, fit_cfg, model, stats)
    return (path if save else model), seconds


def make_agent(task, policy, checkpoint, device=None, mean_of=1, cp_steps=1):
    """The controller of a saved policy: (Agent, configuration)."""
    device = device or default_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    cfg = ckpt['cfg']
    T = task_module(task)
    S, A = T.make_pairs(T.load_demos(), cfg['h_o'], cfg['h_p'])
    if policy == 'krp':                                    # the pairs sampler maps demonstrated pairs
        model.set_pairs(torch.as_tensor(T.normalize(S, ckpt['stats']['S']), device=device),
                        torch.as_tensor(T.normalize(A, ckpt['stats']['A']), device=device))
    if policy == 'cp':
        model.chaining = cp_steps == 3
    agent = Agent(model, ckpt['stats'], A, T.ACT_DIM, T.normalize, T.denormalize, n_samples=mean_of)
    return agent, cfg


def run_episodes(task, agent, cfg, episodes, render=False, progress=None):
    """Roll out `agent` on the episode seeds; returns per-episode scores, steps and seconds."""
    T = task_module(task)
    env = T.make_env()
    bounds = None
    if task == 'square':                                   # executed actions are clipped to this range
        actions = T.load_demos()['action']
        bounds = (actions.min(0), actions.max(0))
    out = {'episodes': [], 'score': [], 'success': [], 'steps': [], 'seconds': [], 'frames': [], 'rewards': []}
    for k, e in enumerate(episodes):
        agent.reset(e)
        t0 = time.time()
        if task == 'square':
            r = T.rollout(env, agent, seed=e, h_o=cfg['h_o'], h_a=cfg['h_a'], bounds=bounds)
        else:
            r = T.rollout(env, agent, seed=e, h_o=cfg['h_o'], h_a=cfg['h_a'], render=render)
        out['seconds'].append(time.time() - t0)
        out['episodes'].append(int(e))
        out['score'].append(float(r['score']))
        out['success'].append(bool(r['success']))
        out['steps'].append(int(r['steps']))
        if render:                                         # frames and per-step rewards, for animations
            out['frames'].append(r['frames'])
            out['rewards'].append(r['reward'])
        if progress:
            progress(k + 1, len(episodes), r['score'])
    return out


def summarize(scores):
    """Mean score, standard error, success rate (score 1) and failure rate (score < 0.5)."""
    s = np.asarray(scores, float)
    return {'mean': float(s.mean()), 'se': float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0,
            'success_rate': float((s >= 1.0).mean()), 'failure_rate': float((s < 0.5).mean()), 'n': int(len(s))}


def evaluate(task, policy, seed=0, episodes=None, state_noise=False, mean_of=1, cp_steps=1,
             device=None, save=True, log=print):
    """
    Evaluate a saved checkpoint on the given episode seeds (default: the 500 held-out episodes
    of the paper) and save the result as JSON. Returns the result dict.
    """
    T = task_module(task)
    episodes = list(T.HELD_OUT_SEEDS if episodes is None else episodes)
    agent, cfg = make_agent(task, policy, checkpoint_path(task, policy, seed, state_noise), device,
                            mean_of, cp_steps)
    res = run_episodes(task, agent, cfg, episodes,
                       progress=lambda k, n, s: log('  episode %d/%d  score %.3f' % (k, n, s)) if k % 50 == 0 else None)
    res.pop('frames'); res.pop('rewards')
    res.update(task=task, policy=policy, seed=seed, state_noise=state_noise, mean_of=mean_of, cp_steps=cp_steps,
               summary=summarize(res['score']), seconds_per_episode=float(np.mean(res['seconds'])))
    log('%s seed %d: score %.3f +- %.3f over %d episodes, %.2f s/episode'
        % (tag(policy, seed, state_noise, mean_of, cp_steps), res['summary']['mean'], res['summary']['se'],
           len(episodes), res['seconds_per_episode']))
    if save:
        path = os.path.join(ROOT, 'results', task, tag(policy, seed, state_noise, mean_of, cp_steps) + '.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(res, open(path, 'w'))
    return res
