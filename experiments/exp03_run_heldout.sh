#!/bin/bash
# EXP03 follow-up: held-out eval (seed 123) of champion candidates.
# Best-of-6 selection on the seed-42 suite overfits that suite; the champion must
# also lead on an unseen spawn set. Sequential, one GPU job at a time.
set -u
source /home/william/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/william/Desktop/isaacLab/reBot/reBot_ACT
for c in exp03_D1 exp03_N3 flow_nominal_v1 flow_dagger_v3; do
  d="runs/$c"
  echo "=== $(date '+%F %T') heldout eval $c start ==="
  python act/eval_act.py \
    --ckpt "$d/ckpt_final.pt" \
    --episodes 64 --num-envs 16 --seed 123 \
    --out "$d/eval_64ep_seed123.json" \
    > "$d/eval_64ep_seed123.log" 2>&1
  echo "=== $(date '+%F %T') heldout eval $c exit=$? ==="
done
echo "=== $(date '+%F %T') heldout chain complete ==="
