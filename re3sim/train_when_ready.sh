#!/usr/bin/env bash
# Wait for demo collection to finish, then train and evaluate. Chained so the whole
# expert -> BC -> eval leg completes unattended.
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc
RUN=re3sim/runs/prim

until grep -aq "\[demos\] wrote" "$RUN/collect.log" 2>/dev/null; do sleep 60; done
echo "############ demos ready $(date -Is) ############"
grep -aE "SUCCESS|demos\] wrote" "$RUN/collect.log" | tail -12

echo
echo "############ training $(date -Is) ############"
python -u re3sim/act/train_flow.py --data "$RUN/demos.hdf5" --out "$RUN/bc" \
  --steps 60000 --seed 1 > "$RUN/train.log" 2>&1
echo "train exit=$?  $(date -Is)"
tail -3 "$RUN/train.log"

echo
echo "############ evaluating $(date -Is) ############"
# Held-out spawn seeds: collection used seed*1000+batch = 1000..1006, so 88000+ cannot collide.
# --teleport-pregrasp is NOT available to the policy, so this measures the policy from the
# env's own reset -- i.e. it will read low, and that is the honest number for THIS checkpoint.
# (a) MATCHED protocol -- start where the demonstrations start. This is the number that says
#     whether BC learned the grasp-and-place manoeuvre. Expert reference: 32.4 % on the same
#     protocol (batches 0-1: 42/128 and 41/128).
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 --expert-rate 0.324 --match-demo-start \
  --out "$RUN/bc_eval_matched.json" > "$RUN/eval_matched.log" 2>&1
echo "matched-protocol eval exit=$?  $(date -Is)"
grep -aE "seed |POOLED|near_miss|carried|no_lift|retains|wrote|DEMO-START" "$RUN/eval_matched.log" | tail -20

echo
echo "############ evaluating from a COLD RESET $(date -Is) ############"
# (b) The deployable number. This checkpoint has never seen the reset pose, so it is expected
#     to read very low -- reported anyway, because the gap between (a) and (b) IS the transit
#     problem, quantified.
python -u re3sim/act/eval_flow.py --headless --ckpt "$RUN/bc/ckpt_final.pt" \
  --num_envs 128 --seeds 88000,88001,88002 \
  --out "$RUN/bc_eval_cold.json" > "$RUN/eval_cold.log" 2>&1
echo "cold-reset eval exit=$?  $(date -Is)"
grep -aE "seed |POOLED|near_miss|carried|no_lift|wrote|reset pose" "$RUN/eval_cold.log" | tail -20
