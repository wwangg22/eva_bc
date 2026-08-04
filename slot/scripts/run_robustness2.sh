#!/usr/bin/env bash
# EXP_ROBUSTNESS round 2 -- axes the env does not randomise AT ALL (docs/slot/EXP_ROBUSTNESS.md §7),
# plus the clearance-crossed dy ladder that tests WHY round 1's dy cell collapsed (§7d).
#
# Round 1 moved the goal (slot dx/dy) and widened the block spawn box. Both are axes the policy
# has at least SEEN vary, or can see vary through the slot-frame observation. Round 2 moves the
# things the challenge env holds perfectly fixed across all 2038 demos:
#   * the arm's start pose  -- identical every episode; zero coverage
#   * the observation       -- enable_corruption = False; the policy has never seen a noisy sensor
# plus the symmetric dy control, the dy ladder, and an all-at-once cell.
#
# Same subject as round 1 so every cell is comparable to robust_gate_nominal:
# champion ckpt_final, spawn seed 777, 128 episodes / 32 envs, -v0 unless a cell says otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
run_task() {  # name, task, extra flags...
  local name="$1" task="$2"; shift 2
  local out="$RUN/robust_${name}.json"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  echo "=== $name  $task  $*  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task "$task" \
    --num-envs 32 --episodes 128 --seed 777 --out "$out" "$@" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $out"
}
run_one() { local n="$1"; shift; run_task "$n" Rebot-PrecisionSlot-v0 "$@"; }

# ---- the dy ladder, crossed with clearance. THE decisive test of the geometric model in §7d:
# the block half-width is 15.0 mm and the mouth half-width is 15.0 + clearance, so a slot shifted
# by dy is enterable only if |dy| <= clearance. -v0 is 1.5 mm, -Loose-v0 is 3.0 mm. Same 3 mm
# shift, opposite predictions -- which no "the policy is fragile" story can produce.
run_one      dy_p001 --slot-dy 0.001
run_one      dy_p002 --slot-dy 0.002
run_one      dy_p003 --slot-dy 0.003
run_task loose_dy_p003 Rebot-PrecisionSlot-Loose-v0 --slot-dy 0.003
run_task loose_dy_p005 Rebot-PrecisionSlot-Loose-v0 --slot-dy 0.005
# dy symmetry control: lateral_error is UNSIGNED, so -1 mm and +1 mm are indistinguishable in
# that channel. If the policy is reading it, the two must score the same; if they differ, it is
# reading direction from the raw block pose instead.
run_one dy_m010 --slot-dy -0.010
# arm start pose: the one axis with literally zero training coverage. 0.10 rad is pick_place's own
# value (pick_place_v1_env_cfg.py:137), chosen there for the same reason.
run_one arm005  --arm-jitter 0.05
run_one arm010  --arm-jitter 0.10
# sensor noise, as a fraction of each channel's own training std
run_one noise05 --obs-noise 0.05
run_one noise20 --obs-noise 0.20
# everything at once, at levels each of which is individually survivable if round 1 held.
# dy is deliberately 0 here: round 1 showed a 5 mm dy is a floor-zero cell on its own, so
# including it would guarantee a zero and measure nothing about the other three axes.
run_one combo   --slot-dx 0.010 --spawn-scale 1.5 --arm-jitter 0.05
echo "=== robustness round 2 done  $(date -Is) ==="
