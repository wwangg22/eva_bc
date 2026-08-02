#!/bin/bash
# DAgger round 2: collect from flow_dagger_v3 with the extended grasp table,
# 3 seeds for scale diversity (round-1 lesson: single seed pinned one object scale).
# Sequential — one GPU job at a time.
set -e
cd "$(dirname "$0")"
source /home/william/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
export REBOT_GRASP_TABLE="$PWD/grasp_table_extended.pt"
for SEED in 203 204 205; do
  echo "=== dagger r2 seed $SEED start $(date) ==="
  python collect_dagger.py --ckpt ../runs/flow_dagger_v3/ckpt_final.pt \
    --rollouts 80 --target-takeovers 40 --seed "$SEED" \
    --out "dagger_r2_s${SEED}.h5"
  echo "=== dagger r2 seed $SEED done $(date) ==="
done
echo "=== ALL ROUND-2 COLLECTIONS DONE $(date) ==="
