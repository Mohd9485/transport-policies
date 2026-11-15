# Author: Mohammad Al-Jarrah
"""
Evaluate saved policies on held-out episodes (the protocol of the paper).

    python scripts/evaluate.py --task pusht --policy otp --seed 0 --n-episodes 50
    python scripts/evaluate.py --task pusht --policy cp --cp-steps 3 --seed 0 1 2
    python scripts/evaluate.py --task square --policy sip --seed 0 --state-noise --mean-of 8 --workers 12

By default the 500 held-out episodes 10000-10499 are used; --n-episodes N takes the first N.
Each result is saved to results/<task>/<name>_seed<k>.json, with the per-episode scores, and
summarized as the mean score, its standard error, the success rate (score 1), the failure rate
(score below 0.5) and the time per episode.

The episodes can be split over several worker processes (--workers), which helps on Square,
whose simulator runs on the CPU. The time per episode is measured inside each worker; use one
worker per GPU, and no other jobs, to measure it as in the paper.
"""
import os
import sys
import json
import time
import argparse
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


def _worker(job):
    """Load the agent in a fresh process and roll out one shard of episodes."""
    import torch
    torch.set_num_threads(1)
    import experiment as X
    task, policy, seed, noise, mean_of, cp_steps, device, episodes = job
    agent, cfg = X.make_agent(task, policy, X.checkpoint_path(task, policy, seed, noise),
                              torch.device(device), mean_of, cp_steps)
    out = X.run_episodes(task, agent, cfg, episodes)
    out.pop('frames'); out.pop('rewards')
    return out


def evaluate_parallel(task, policy, seed, episodes, noise, mean_of, cp_steps, devices, workers):
    """Evaluate one checkpoint on the episodes, split over workers and devices."""
    import experiment as X
    shards = [episodes[i::workers] for i in range(workers)]
    jobs = [(task, policy, seed, noise, mean_of, cp_steps, devices[i % len(devices)], sh)
            for i, sh in enumerate(shards) if sh]
    t0 = time.time()
    if workers == 1:
        parts = [_worker(jobs[0])]
    else:
        with mp.get_context('spawn').Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
    order = np.argsort(np.concatenate([p['episodes'] for p in parts]))
    res = {k: list(np.concatenate([p[k] for p in parts])[order]) for k in ('episodes', 'score', 'success', 'steps', 'seconds')}
    res = {k: [v.item() if hasattr(v, 'item') else v for v in vals] for k, vals in res.items()}
    res.update(task=task, policy=policy, seed=seed, state_noise=noise, mean_of=mean_of, cp_steps=cp_steps,
               summary=X.summarize(res['score']), seconds_per_episode=float(np.mean(res['seconds'])),
               wall_seconds=time.time() - t0, workers=workers)
    return res


def main():
    import experiment as X
    from policies import POLICIES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=X.TASKS, required=True)
    ap.add_argument('--policy', nargs='+', choices=sorted(POLICIES), required=True)
    ap.add_argument('--seed', nargs='+', type=int, default=[0])
    ap.add_argument('--n-episodes', type=int, default=500, help='first N of the held-out episodes 10000-10499')
    ap.add_argument('--state-noise', action='store_true', help='Square: the checkpoint trained with state noise')
    ap.add_argument('--mean-of', type=int, default=1, help='Square: execute the mean of N sampled chunks')
    ap.add_argument('--cp-steps', type=int, choices=[1, 3], default=1, help='CP: 1-step or 3-step sampling')
    ap.add_argument('--devices', nargs='+', default=['cuda:0'], help='GPUs used by the workers, round robin')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    episodes = X.task_module(args.task).HELD_OUT_SEEDS[:args.n_episodes]
    for policy in args.policy:
        for seed in args.seed:
            res = evaluate_parallel(args.task, policy, seed, episodes, args.state_noise, args.mean_of,
                                    args.cp_steps, args.devices, args.workers)
            name = X.tag(policy, seed, args.state_noise, args.mean_of, args.cp_steps)
            path = os.path.join(X.ROOT, 'results', args.task, name + '.json')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(res, open(path, 'w'))
            s = res['summary']
            print('%-22s score %.3f +- %.3f  success %.1f%%  failure %.1f%%  %.2f s/episode  (%d episodes) -> %s'
                  % (name, s['mean'], s['se'], 100 * s['success_rate'], 100 * s['failure_rate'],
                     res['seconds_per_episode'], s['n'], path), flush=True)


if __name__ == '__main__':
    main()
