#!/usr/bin/env bash
# EXP_ROBUSTNESS round 3 -- REPLICATION on a second training seed (docs/slot/EXP_ROBUSTNESS.md §9).
#
# Rounds 1-2 are one checkpoint at one spawn seed. PLAN 5.28's single-seed caution says a margin
# under ~10 points on one checkpoint is not evidence, and the two dx effects are 13.5 and 9.4
# points -- right on that line. Training-seed variance on this project is 15-29 points, which is
# larger than either effect, so "bc_armB_seed0 dips at -10 mm" and "the policy dips at -10 mm"
# are different claims and only round 3 can tell them apart.
#
# Subject: bc_armA_seed0 -- a DIFFERENT training arm and seed, and the second-best run in the
# sweep (0.927 on -v0 s777), so it has enough headroom for a 10-point dip to be visible and
# enough ceiling for the dy collapse to be unambiguous.
#
# Only the cells that carry a claim are replicated. dx_p005 and dy_p010 are omitted: the first
# was a null (p = 1.0) and the second is a duplicate of a floor-zero cell.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armA_seed0}"
run_one() {  # name, extra flags...
  local name="$1"; shift
  local out="$RUN/robust_${name}.json"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  echo "=== $name  $*  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs 32 --episodes 128 --seed 777 --out "$out" "$@" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $out"
}

run_one gate_nominal            # must reproduce this run's own sweep cell (0.927) -- the gate
run_one dx_m010 --slot-dx -0.010   # claim: -13.5 pts
run_one dx_p010 --slot-dx  0.010   # claim: flat
run_one dx_p020 --slot-dx  0.020   # claim: -9.4 pts
run_one dy_p005 --slot-dy  0.005   # claim: floor zero
run_one spawn20 --spawn-scale 2.0  # claim: -42.7 pts, grasp starts failing
echo "=== robustness round 3 done  $(date -Is) ==="
