#!/usr/bin/env bash
# EXP_DEPTH probe A -- is the 600-step episode horizon binding? (docs/slot/EXP_DEPTH.md)
#
# 82-84 % of failures at every clearance are depth failures. They split into two shapes, and the
# split is per-CHECKPOINT, not per-clearance -- visible in the median failure depth of the 36
# sweep cells:
#   * bc_armB_seed0 / seed2 : median failure depth +29 .. +35 mm  -> ENTERED and stopped short
#   * bc_armA_seed2 / seed1 : median failure depth -16 .. -41 mm  -> never crossed the mouth
# "Out of time" can only explain the first shape. So run both shapes at a longer horizon: if the
# stalled-in-mouth runs recover and the never-entered runs do not, the horizon is binding for one
# failure mode and irrelevant for the other -- which is a sharper statement than either alone.
#
# Subjects are chosen for HEADROOM (both ~0.62-0.72, so a real effect has room to show) and for
# opposite failure shapes. Their 12 s baselines already exist on disk as the sweep's own cells:
#   runs/bc_armB_seed2/eval_ckpt_final_Rebot-PrecisionSlot-Tight-v0_s777.json  0.625  (+29.4 mm)
#   runs/bc_armB_seed1/eval_ckpt_final_Rebot-PrecisionSlot-Tight-v0_s777.json  0.719  (-41.5 mm)
# so only the 20 s cells are new. Same seed, same task, same episode/env counts as those cells,
# which is what makes the comparison legitimate.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

run_one() {  # run, secs
  local run="$1" secs="$2"
  local out="runs/$run/horizon_${secs}s_Tight_s777.json"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  echo "=== $run  ${secs}s  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "runs/$run/ckpt_final.pt" \
    --task Rebot-PrecisionSlot-Tight-v0 --num-envs 32 --episodes 128 --seed 777 \
    --episode-length-s "$secs" --out "$out" 2>&1 | grep -E "eval_act" \
    || echo "!!! FAILED (continuing): $out"
}

run_one bc_armB_seed2 20   # stalled-in-mouth shape
run_one bc_armB_seed1 20   # never-entered shape
echo "=== horizon probe done  $(date -Is) ==="
