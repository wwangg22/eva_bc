#!/usr/bin/env bash
# Second pass on the end-to-end task, with more data.
#
# The first pass scored 7.8 % from a cold reset against a 31.9 % expert. The taxonomy says the
# carry and the place are LEARNED -- `success` tracks `lifted` almost exactly -- and that all
# of the loss is the grasp. The most likely cause is not the optimiser but the sample budget:
# adding the transit doubled every episode (488 -> 1057 steps), so the same 327 demonstrations
# now have to cover twice as much behaviour. So: twice the demonstrations, and longer training.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN=re3sim/runs/final2
mkdir -p "$RUN"

echo "############ demos  $(date -Is) ############"
BIAS_MAX=0.0 python -u re3sim/expert/collect_demos.py --headless \
  --num_envs 128 --batches 16 --seed 5 \
  --out "$RUN/demos.hdf5" > "$RUN/collect.log" 2>&1
echo "collect exit=$?"
grep -aE "SUCCESS|demos\] wrote" "$RUN/collect.log" | tail -20

echo
echo "############ flow-matching BC, 100k steps  $(date -Is) ############"
python -u re3sim/act/train_flow.py --data "$RUN/demos.hdf5" --out "$RUN/bc" \
  --steps 100000 --seed 5 > "$RUN/train.log" 2>&1
echo "train exit=$?"
head -2 "$RUN/train.log"; tail -2 "$RUN/train.log"

EXP=$(grep -a "demos] wrote" "$RUN/collect.log" | grep -oE "\(([0-9.]+)%" | tr -d '(%')
echo
echo "############ eval from the env's own reset  $(date -Is) ############"
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 \
  --expert-rate "0.$(printf '%.0f' "${EXP:-32}")" \
  --out "$RUN/bc_eval.json" > "$RUN/eval.log" 2>&1
echo "exit=$?"
grep -aE "horizon|seed 8|POOLED|near_miss|carried_astray|no_lift|retains|median " "$RUN/eval.log" | tail -14
echo "############ DONE $(date -Is) ############"
