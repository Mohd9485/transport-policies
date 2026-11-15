# Author: Mohammad Al-Jarrah
"""
Tables of the paper from the evaluation files in results/<task>/.

    python scripts/summarize_results.py --task pusht                 # Table 2
    python scripts/summarize_results.py --task square                # Table 3
    python scripts/summarize_results.py --task pusht --bootstrap dp  # paired bootstrap against DP

Push-T, per policy (all available seeds pooled): mean score +- standard error, seed spread (the
population standard deviation of the per-seed mean scores), success rate (score 1), failure rate
(score below 0.5), time per episode, parameters, and training time.
Square, per policy: the success rate in the four settings of Table 3 (plain, mean of 8, state
noise, both).

--bootstrap REF: for every policy, a 95% interval of the difference of its mean score and that of
REF, from 10,000 paired bootstrap replicates; the initial conditions are resampled jointly for the
two policies and the training seeds independently.
"""
import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import experiment as X
from policies import NAMES

ORDER = ['otp', 'fmp', 'sip', 'krp', 'dp', 'dpr', ('cp', 1), ('cp', 3)]


def load(task, name):
    """{seed: result} for the evaluation files results/<task>/<name>_seed<k>.json."""
    out = {}
    for f in sorted(glob.glob(os.path.join(X.ROOT, 'results', task, name + '_seed*.json'))):
        r = json.load(open(f))
        if 'seed' in r:                                      # evaluation files of scripts/evaluate.py
            out[r['seed']] = r
    return out


def label(entry):
    """Name of a table entry as printed in the tables."""
    return '%s %d-step' % (NAMES['cp'], entry[1]) if isinstance(entry, tuple) else NAMES[entry]


def file_name(entry, noise=False, mean_of=1):
    """Prefix of the evaluation files of a table entry (without the seed)."""
    policy, steps = entry if isinstance(entry, tuple) else (entry, 1)
    return X.tag(policy, 0, noise, mean_of, steps)[:-len('_seed0')]


def parameters(task, policy):
    """Number of parameters used at deployment (OTP: the map; CP: the student)."""
    path = X.checkpoint_path(task, policy, 0)
    if not os.path.exists(path):
        return float('nan')
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    keep = {'otp': 'map.', 'cp': 'student.'}.get(policy, '')
    n = sum(v.numel() for k, v in sd.items() if k.startswith(keep) and k not in ('map.mask', 'map.nodes', 'map.weights'))
    if policy == 'krp':          # weights fixed at zero by the triangular structure are not parameters
        width = sd['map.c_layers.0.weight'].shape[-1] + sd['map.g_in_ctx.weight'].shape[-1]
        n -= int((sd['map.mask'] == 0).sum()) * width
    return n / 1e6


def pusht_table():
    """Print Table 2 (Push-T)."""
    times = {}
    tpath = os.path.join(X.ROOT, 'results', 'pusht', 'training_time.json')
    if os.path.exists(tpath):
        times = json.load(open(tpath))
    print('%-11s %15s %8s %9s %9s %8s %9s %10s' % ('Policy', 'Score', 'Spread', 'Success', 'Failure',
                                                  'Time[s]', 'Param[M]', 'Train[min]'))
    for entry in ORDER:
        runs = load('pusht', file_name(entry))
        if not runs:
            continue
        scores = np.concatenate([r['score'] for r in runs.values()])
        s = X.summarize(scores)
        spread = np.std([np.mean(r['score']) for r in runs.values()])
        policy = entry[0] if isinstance(entry, tuple) else entry
        train_min = np.mean([v for k, v in times.items() if k.startswith(policy + '_seed')] or [np.nan]) / 60
        print('%-11s %7.3f +- %.3f %8.3f %8.1f%% %8.1f%% %8.2f %9.2f %10.1f'
              % (label(entry), s['mean'], s['se'], spread, 100 * s['success_rate'], 100 * s['failure_rate'],
                 np.mean([r['seconds_per_episode'] for r in runs.values()]), parameters('pusht', policy), train_min))


def square_table():
    """Print Table 3 (Square)."""
    settings = [('Plain', False, 1), ('Mean of 8', False, 8), ('State noise', True, 1), ('Both', True, 8)]
    print('%-11s' % 'Policy' + ''.join('%14s' % s[0] for s in settings))
    for entry in ORDER:
        row = []
        for _, noise, mean_of in settings:
            runs = load('square', file_name(entry, noise, mean_of))
            row.append('%7.3f (%d)' % (np.mean(np.concatenate([r['score'] for r in runs.values()])), len(runs))
                       if runs else '%14s' % '-')
        print('%-11s' % label(entry) + ''.join('%14s' % c for c in row))
    print('(n) = number of training seeds pooled')


def bootstrap(task, ref, n_rep=10000, rng=np.random.default_rng(0)):
    """Paired bootstrap of the difference of mean scores against `ref` (initial conditions joint)."""
    def matrix(entry):
        """Scores of an entry as a (seeds, episodes) array."""
        runs = load(task, file_name(entry))
        return np.stack([runs[k]['score'] for k in sorted(runs)]) if runs else None     # (seeds, episodes)
    R = matrix(ref)
    for entry in ORDER:
        P = matrix(entry)
        if P is None or R is None or entry == ref or P.shape[1] != R.shape[1]:
            continue
        diffs = np.empty(n_rep)
        n = P.shape[1]
        for b in range(n_rep):
            e = rng.integers(0, n, n)                                    # the same episodes for both
            sp = rng.integers(0, P.shape[0], P.shape[0])                 # independent seeds
            sr = rng.integers(0, R.shape[0], R.shape[0])
            diffs[b] = P[sp][:, e].mean() - R[sr][:, e].mean()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print('%-11s - %-5s  %+.4f  [%+.4f, %+.4f]' % (label(entry), label(ref), P.mean() - R.mean(), lo, hi))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=X.TASKS, required=True)
    ap.add_argument('--bootstrap', default=None, help='reference policy of the paired bootstrap, e.g. dp')
    args = ap.parse_args()
    pusht_table() if args.task == 'pusht' else square_table()
    if args.bootstrap:
        print()
        bootstrap(args.task, args.bootstrap)


if __name__ == '__main__':
    main()
