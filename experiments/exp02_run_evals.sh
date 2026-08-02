#!/bin/bash
# EXP02: open-loop execution-horizon ablation on the frozen v1 champion.
# No retraining — eval-time --n-action-steps override only.
# Baseline n=15 already exists: runs/flow_nominal_v1/eval_gate2_64ep_diag.json (59.4%).
# Identical settings to the baseline: 64 eps, 16 envs, seed 42, flush ON, 30 s episodes.
set -u
source /home/william/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/william/Desktop/isaacLab/reBot/reBot_ACT
for N in 8 4 2 1; do
  echo "=== $(date '+%F %T') exp02 n_action_steps=$N starting ==="
  python act/eval_act.py \
    --ckpt runs/flow_nominal_v1/ckpt_final.pt \
    --episodes 64 --num-envs 16 --seed 42 \
    --n-action-steps "$N" \
    --out "runs/flow_nominal_v1/eval_h${N}_64ep.json" \
    > "runs/flow_nominal_v1/eval_h${N}_64ep.log" 2>&1
  echo "=== $(date '+%F %T') exp02 n_action_steps=$N exit=$? ==="
done
echo "=== $(date '+%F %T') exp02 chain complete ==="
