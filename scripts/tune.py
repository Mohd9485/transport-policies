# Author: Mohammad Al-Jarrah
"""
Hyperparameter search with SMAC3 (Lindauer et al., 2022), as used to select the settings of
OTP, FMP, SIP, KRP and DPR in configs/<task>.py.

    python scripts/tune.py --task pusht --policy fmp                      # 4 hours (default)
    python scripts/tune.py --task square --policy sip --walltime-hours 8

A trial trains the policy with the sampled configuration and returns one minus its mean score on
the tuning episodes 0-49 (disjoint from the held-out episodes 10000-10499 of the evaluation).
The training seed is SMAC's instance seed, so a configuration can be re-evaluated with other
seeds. The search runs for a wall-clock budget (4 hours by default) and starts from the best
configuration found for the paper, i.e. the settings of configs/<task>.py, for both tasks; the
search spaces are defined in tuning/search_spaces.py. The incumbent is written to
results/<task>/tuning/<policy>_incumbent.json, and SMAC's run history to the same folder.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

import experiment as X
from tuning.search_spaces import SPACES, build_space, to_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=X.TASKS, required=True)
    ap.add_argument('--policy', choices=sorted(SPACES), required=True)
    ap.add_argument('--walltime-hours', type=float, default=4.0, help='wall-clock budget of the search')
    ap.add_argument('--n-trials', type=int, default=10000, help='upper bound on the number of trials')
    ap.add_argument('--n-episodes', type=int, default=50, help='tuning episodes 0..N-1')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    from smac import Scenario, HyperparameterOptimizationFacade as HPO

    base = X.get_config(args.task, args.policy)
    cs = build_space(args.task, args.policy, base)
    device = torch.device(args.device) if args.device else X.default_device()
    episodes = X.task_module(args.task).TUNING_SEEDS[:args.n_episodes]
    out_dir = os.path.join(X.ROOT, 'results', args.task, 'tuning')
    os.makedirs(out_dir, exist_ok=True)

    def objective(sample, seed: int = 0) -> float:
        """Train with the sampled configuration and return 1 - mean score on the tuning episodes."""
        cfg = to_config(args.task, args.policy, base, sample)
        try:
            model, _ = X.train(args.task, args.policy, seed=seed, device=device, cfg=cfg, save=False,
                               log=lambda *a: None)
            ckpt = os.path.join(out_dir, '_trial.pt')
            S, A, stats = X.prepare_data(args.task, cfg)
            from policies import save_checkpoint
            save_checkpoint(ckpt, args.policy, args.task, model.cfg, model, stats)
            agent, cfg_a = X.make_agent(args.task, args.policy, ckpt, device)
            score = float(np.mean(X.run_episodes(args.task, agent, cfg_a, episodes)['score']))
        except Exception as exc:                            # a failed trial scores zero
            print('trial failed: %s' % exc)
            score = 0.0
        print('score %.4f  %s' % (score, dict(sample)), flush=True)
        return 1.0 - score

    scenario = Scenario(cs, name='%s_%s' % (args.task, args.policy), output_directory=out_dir,
                        deterministic=False, n_trials=args.n_trials, use_default_config=True,
                        walltime_limit=args.walltime_hours * 3600)
    # 10 random configurations start the search, after the default (the paper's settings), which
    # is evaluated first so that every search is measured against it
    initial = HPO.get_initial_design(scenario, n_configs=10, max_ratio=1.0)
    smac = HPO(scenario, objective, initial_design=initial, overwrite=False)
    from smac.runhistory.dataclasses import TrialInfo, TrialValue
    default = cs.get_default_configuration()
    smac.tell(TrialInfo(config=default, seed=0), TrialValue(cost=objective(default, 0)))
    incumbent = smac.optimize()
    best = to_config(args.task, args.policy, base, incumbent)
    path = os.path.join(out_dir, '%s_incumbent.json' % args.policy)
    json.dump(best, open(path, 'w'), indent=1)
    print('incumbent -> %s\n%s' % (path, best))


if __name__ == '__main__':
    main()
