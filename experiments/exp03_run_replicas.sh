#!/bin/bash
# EXP03: 3 nominal + 3 nominal+dagger training replicas (seeds 1,2,3), each + 64-ep eval.
# Exact v1/v3 recipe: 100k steps, batch 64, lr 1e-4, chunk 50. Strictly sequential (one GPU).
# See EXP03_dagger_interference.md.
set -u
source /home/william/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/william/Desktop/isaacLab/reBot/reBot_ACT

run_one () {  # $1=name $2=pool $3=seed $4=extra-data
  local name=$1 pool=$2 seed=$3 extra=$4
  echo "=== $(date '+%F %T') exp03 $name train start ==="
  # shellcheck disable=SC2086
  python act/train_flow.py \
    --data expert/demos_nominal_s10*.h5 $extra \
    --pool "$pool" --seed "$seed" \
    --out "runs/exp03_${name}" \
    --steps 100000 --batch-size 64 --lr 1e-4 --save-every 100000 \
    > "runs/exp03_${name}_train.log" 2>&1
  echo "=== $(date '+%F %T') exp03 $name train exit=$? ==="
  echo "=== $(date '+%F %T') exp03 $name eval start ==="
  python act/eval_act.py \
    --ckpt "runs/exp03_${name}/ckpt_final.pt" \
    --episodes 64 --num-envs 16 --seed 42 \
    --out "runs/exp03_${name}/eval_64ep.json" \
    > "runs/exp03_${name}/eval_64ep.log" 2>&1
  echo "=== $(date '+%F %T') exp03 $name eval exit=$? ==="
}

for s in 1 2 3; do
  run_one "N${s}" nominal "$s" ""
done
for s in 1 2 3; do
  run_one "D${s}" nominal+dagger "$s" "expert/dagger_r1.h5"
done
echo "=== $(date '+%F %T') exp03 chain complete ==="
