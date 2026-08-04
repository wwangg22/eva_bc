#!/usr/bin/env bash
# EXP_DEPTH probe A, round 2 -- replicate the horizon effect and test whether it saturates.
#
# Round 1 (scripts/run_horizon.sh) gave a clean DIFFERENTIAL: +11.5 points for the
# stalled-in-mouth checkpoint, -2.1 for the never-entered one. But at n = 96 the individual
# gains are p = 0.088 and p = 0.75. The direction is what was pre-registered; the significance
# is not there, and quoting +11.5 points as established would be exactly the mistake
# EXP_TIGHT.md §7 caught last session (a single-cell effect that did not replicate).
#
# Three cells:
#   * both subjects again at spawn seed 888 -- a different reset draw, baselines already on disk
#     (bc_armB_seed2 Tight s888 = 0.708, bc_armB_seed1 Tight s888 = 0.615). If the differential
#     holds across spawn seeds it is real; if it flips, round 1 was a draw.
#   * bc_armB_seed2 at 30 s -- does the gain keep coming, or does it saturate? "The policy is
#     slow" and "the policy needs exactly a bit more than 600 steps" are different claims, and
#     a third point on the curve separates them.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

run_one() {  # run, secs, seed
  local run="$1" secs="$2" seed="$3"
  local out="runs/$run/horizon_${secs}s_Tight_s${seed}.json"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  echo "=== $run  ${secs}s  s${seed}  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "runs/$run/ckpt_final.pt" \
    --task Rebot-PrecisionSlot-Tight-v0 --num-envs 32 --episodes 128 --seed "$seed" \
    --episode-length-s "$secs" --out "$out" 2>&1 | grep -E "eval_act" \
    || echo "!!! FAILED (continuing): $out"
}

run_one bc_armB_seed2 20 888   # stalled-in-mouth shape, second spawn draw
run_one bc_armB_seed1 20 888   # never-entered shape, second spawn draw
run_one bc_armB_seed2 30 777   # does the gain saturate?
echo "=== horizon probe round 2 done  $(date -Is) ==="
