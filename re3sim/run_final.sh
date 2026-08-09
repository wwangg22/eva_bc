#!/usr/bin/env bash
# The full ladder against the FINAL photoreal env: reconstructed cube / tape roll / tape
# measure colliders, reconstructed desk (splats + fitted slab, yaw corrected), grip height
# re-measured for the real mesh (32 mm, not the analytic cube's 25 mm).
#
# Everything in re3sim/runs/prim/ was measured against PRIMITIVE colliders and a desk that was
# 90 degrees out. None of it transfers; this replaces it.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc
RUN=re3sim/runs/final
mkdir -p "$RUN"

echo "############ demos  $(date -Is) ############"
BIAS_MAX=0.0 python -u re3sim/expert/collect_demos.py --headless \
  --num_envs 128 --batches 8 --seed 3 \
  --out "$RUN/demos.hdf5" > "$RUN/collect.log" 2>&1
echo "collect exit=$?"
grep -aE "workstation\]|SUCCESS|demos\] wrote" "$RUN/collect.log" | tail -16

echo
echo "############ flow-matching BC  $(date -Is) ############"
python -u re3sim/act/train_flow.py --data "$RUN/demos.hdf5" --out "$RUN/bc" \
  --steps 60000 --seed 3 > "$RUN/train.log" 2>&1
echo "train exit=$?"
head -2 "$RUN/train.log"; tail -2 "$RUN/train.log"

EXP=$(grep -a "demos] wrote" "$RUN/collect.log" | grep -oE "\(([0-9.]+)%" | tr -d '(%' )
echo "expert rate for the retention figure: ${EXP:-unknown}%"

echo
# ONE evaluation now, and it is the deployable one. The demonstrations start from the env's
# own `reset()` -- the transit works -- so "matched protocol" and "cold reset" are the same
# thing, and the two-number hedge that §4a needed is gone.
echo "############ eval: from the env's own reset  $(date -Is) ############"
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 \
  --expert-rate "0.$(printf '%.0f' "${EXP:-40}")" \
  --out "$RUN/bc_eval.json" > "$RUN/eval.log" 2>&1
echo "exit=$?"
grep -aE "reset pose|seed 8|POOLED|near_miss|carried_astray|no_lift|retains|median " "$RUN/eval.log" | tail -14
echo "############ DONE $(date -Is) ############"
