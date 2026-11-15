#!/bin/bash
# Author: Mohammad Al-Jarrah
# The full protocol of the paper: every policy trained with 3 seeds and evaluated on the 500
# held-out episodes 10000-10499, then the tables.
#
#   bash scripts/reproduce_paper.sh pusht            # Table 2
#   bash scripts/reproduce_paper.sh square           # Table 3 (plain, mean of 8, state noise, both)
#
# Optional environment variables:
#   POLICIES="otp fmp"   policies to run (default: all)
#   SEEDS="0 1 2"        training seeds (default: 0 1 2)
#   DEVICE=cuda:0        GPU for training;  EVAL_DEVICES="cuda:0 cuda:1" GPUs for the evaluation
#   WORKERS=12           evaluation processes (default: 1 on Push-T, 12 on Square)
#
# Cost: training takes from minutes (OTP) to about a day (DP and CP on Square) per seed on one
# GPU; the evaluation of a seed takes from minutes to hours. Steps whose output already exists
# are skipped, so the script can be stopped and restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:?usage: bash scripts/reproduce_paper.sh pusht|square}
POLICIES=${POLICIES:-"otp fmp sip krp dp dpr cp"}
SEEDS=${SEEDS:-"0 1 2"}
DEVICE=${DEVICE:-cuda:0}
EVAL_DEVICES=${EVAL_DEVICES:-$DEVICE}
if [ "$TASK" = square ]; then WORKERS=${WORKERS:-12}; NOISE="no yes"; else WORKERS=${WORKERS:-1}; NOISE="no"; fi

for P in $POLICIES; do
  for N in $NOISE; do
    FLAG=""; SUFFIX=""
    if [ "$N" = yes ]; then FLAG="--state-noise"; SUFFIX="_noise"; fi
    # ---- training
    for S in $SEEDS; do
      if [ ! -f "checkpoints/$TASK/${P}${SUFFIX}_seed$S.pt" ]; then
        python scripts/train.py --task "$TASK" --policy "$P" --seed "$S" $FLAG --device "$DEVICE"
      fi
    done
    # ---- evaluation: one sample per chunk, and (Square) the mean of 8 samples
    MEANS="1"; [ "$TASK" = square ] && MEANS="1 8"
    STEPS="1"; [ "$P" = cp ] && STEPS="1 3"
    for M in $MEANS; do
      for K in $STEPS; do
        NAME="${P}${SUFFIX}"; [ "$M" -gt 1 ] && NAME="${NAME}_mean$M"; [ "$K" = 3 ] && NAME="${NAME}_3step"
        for S in $SEEDS; do
          if [ ! -f "results/$TASK/${NAME}_seed$S.json" ]; then
            python scripts/evaluate.py --task "$TASK" --policy "$P" --seed "$S" $FLAG --mean-of "$M" \
                   --cp-steps "$K" --devices $EVAL_DEVICES --workers "$WORKERS"
          fi
        done
      done
    done
  done
done
python scripts/summarize_results.py --task "$TASK"
