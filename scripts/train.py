# Author: Mohammad Al-Jarrah
"""
Train one or more policies and save their checkpoints.

    python scripts/train.py --task pusht --policy otp --seed 0
    python scripts/train.py --task pusht --policy otp fmp sip --seed 0 1 2
    python scripts/train.py --task square --policy sip --seed 0 --state-noise

Checkpoints are written to checkpoints/<task>/<policy>[_noise]_seed<k>.pt, and the training
times to results/<task>/training_time.json. The hyperparameters are those of configs/<task>.py.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from experiment import train, tag, ROOT, TASKS
from policies import POLICIES


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', choices=TASKS, required=True)
    ap.add_argument('--policy', nargs='+', choices=sorted(POLICIES), required=True)
    ap.add_argument('--seed', nargs='+', type=int, default=[0])
    ap.add_argument('--state-noise', action='store_true', help='Square only: train with the state noise')
    ap.add_argument('--device', default=None, help="e.g. cuda:0 (default: cuda if available, else cpu)")
    args = ap.parse_args()
    device = torch.device(args.device) if args.device else None

    times_path = os.path.join(ROOT, 'results', args.task, 'training_time.json')
    os.makedirs(os.path.dirname(times_path), exist_ok=True)
    times = json.load(open(times_path)) if os.path.exists(times_path) else {}
    for policy in args.policy:
        for seed in args.seed:
            path, seconds = train(args.task, policy, seed, args.state_noise, device)
            times[tag(policy, seed, args.state_noise)] = seconds
            json.dump(times, open(times_path, 'w'), indent=1)
            print('saved %s\n' % path)


if __name__ == '__main__':
    main()
