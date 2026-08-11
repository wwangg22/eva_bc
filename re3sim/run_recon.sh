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
cd "$(dirname "${BASH_SOURCE[0]}")/.."
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
echo "############ eval: matched (demo-start) protocol  $(date -Is) ############"
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 --match-demo-start \
  --expert-rate "0.$(printf '%.0f' "${EXP:-30}")" \
  --out "$RUN/bc_eval_matched.json" > "$RUN/eval_matched.log" 2>&1
echo "exit=$?"
grep -aE "DEMO-START|seed 8|POOLED|near_miss|carried_astray|no_lift|retains|median " "$RUN/eval_matched.log" | tail -16

echo
echo "############ eval: cold reset  $(date -Is) ############"
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 \
  --out "$RUN/bc_eval_cold.json" > "$RUN/eval_cold.log" 2>&1
echo "exit=$?"
grep -aE "reset pose|seed 8|POOLED|no_lift" "$RUN/eval_cold.log" | tail -10
echo "############ DONE $(date -Is) ############"
