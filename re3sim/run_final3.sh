#!/usr/bin/env bash
# Third pass: double the data again, train on the UNION of both collections.
#
# The scaling is steep and has not flattened: 1024 episodes / 60k steps -> 7.8 %, 2048 / 100k
# -> 20.1 %. `success` still tracks `lifted` almost exactly, so the carry and the place are
# learned and every point of the remaining gap is the grasp -- which is the part of the
# trajectory that most needs coverage of the spawn distribution.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc
RUN=re3sim/runs/final3
mkdir -p "$RUN"

echo "############ demos (fresh seeds, union with final2)  $(date -Is) ############"
BIAS_MAX=0.0 python -u re3sim/expert/collect_demos.py --headless \
  --num_envs 128 --batches 16 --seed 9 \
  --out "$RUN/demos.hdf5" > "$RUN/collect.log" 2>&1
echo "collect exit=$?"
grep -aE "demos\] wrote" "$RUN/collect.log" | tail -2

echo
echo "############ flow-matching BC on BOTH collections, 140k steps  $(date -Is) ############"
python -u re3sim/act/train_flow.py \
  --data re3sim/runs/final2/demos.hdf5 "$RUN/demos.hdf5" \
  --out "$RUN/bc" --steps 140000 --seed 9 > "$RUN/train.log" 2>&1
echo "train exit=$?"
head -2 "$RUN/train.log"; tail -2 "$RUN/train.log"

echo
echo "############ eval from the env's own reset  $(date -Is) ############"
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 --expert-rate 0.310 \
  --out "$RUN/bc_eval.json" > "$RUN/eval.log" 2>&1
echo "exit=$?"
grep -aE "horizon|seed 8|POOLED|near_miss|carried_astray|no_lift|retains|median " "$RUN/eval.log" | tail -14
echo "############ DONE $(date -Is) ############"
