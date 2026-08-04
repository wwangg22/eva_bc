#!/usr/bin/env bash
# EXP_ROBUSTNESS -- perturbed evaluations of the champion (docs/slot/EXP_ROBUSTNESS.md).
# Gate first: the nominal cell must reproduce the sweep's -v0 s777 number, or the
# perturbation code path is not inert at zero and nothing after it is interpretable.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
run_one() {  # name, extra flags...
  local name="$1"; shift
  local out="$RUN/robust_${name}.json"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  echo "=== $name  $*  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs 32 --episodes 128 --seed 777 --out "$out" "$@" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $out"
}

run_one gate_nominal
run_one dx_p005 --slot-dx  0.005
run_one dx_p010 --slot-dx  0.010
run_one dx_p020 --slot-dx  0.020
run_one dx_m010 --slot-dx -0.010
run_one dy_p005 --slot-dy  0.005
run_one dy_p010 --slot-dy  0.010
run_one spawn15 --spawn-scale 1.5
run_one spawn20 --spawn-scale 2.0
echo "=== robustness sweep done  $(date -Is) ==="
