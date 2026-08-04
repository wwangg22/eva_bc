#!/usr/bin/env bash
# Stage C: the S3 arm comparison (docs/slot/EXP_BC_ARMS.md).
#
# Arms are matched on demo count and differ only in composition:
#   A  1024 nominal demos            (seeds 0-7)
#   B   512 nominal + 512 DART       (seeds 0-3, dart002 10-11, dart005 20-21)
# Both use --pool success, so failed DART episodes are excluded from BOTH arms and the only
# difference is what produced the successful ones. See EXP_BC_ARMS.md sections "Three flag
# corrections" for why neither inherited pool filter would do.
#
# Usage:  bash scripts/run_stage_c.sh <seed> [steps]
# Runs ONE seed of BOTH arms sequentially, so several seeds can be launched as separate
# background jobs (or not) without editing anything.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

SEED="${1:?usage: run_stage_c.sh <seed> [steps]}"
STEPS="${2:-100000}"

A=$(ls data/v2/nominal_s{0,1,2,3,4,5,6,7}.hdf5)
B=$(ls data/v2/nominal_s{0,1,2,3}.hdf5 data/v2/dart002_s1{0,1}.hdf5 data/v2/dart005_s2{0,1}.hdf5)

for ARM in A B; do
  case $ARM in A) DATA=$A ;; B) DATA=$B ;; esac
  OUT="runs/bc_arm${ARM}_seed${SEED}"
  echo "=== arm $ARM seed $SEED  ($STEPS steps) -> $OUT  $(date -Is) ==="
  python slot_act/train_flow.py --data $DATA --pool success \
    --out "$OUT" --seed "$SEED" --steps "$STEPS" --save-every 10000
done
echo "=== stage C seed $SEED done  $(date -Is) ==="
