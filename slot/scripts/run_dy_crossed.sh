#!/usr/bin/env bash
# The sharp clearance-crossed dy cell that §7d should have asked for (EXP_ROBUSTNESS §7e).
#
# Measured on -v0 (1.5 mm clearance): dy = 1 mm -> 0.948, dy = 2 mm -> 0.000. The cliff is
# between 1 and 2 mm, exactly where the geometric model puts it.
#
# §7d's crossed cell was loose_dy_p003 -- a 3.0 mm shift against a 3.0 mm clearance, which sits
# EXACTLY on the boundary (mouth half-width 15 + 3 = 18 mm, block half-width 15 mm, so a 3 mm
# shift leaves precisely zero margin on one side). That makes it a coin-flip rather than a test.
# The sharp cell is 2 mm: a total floor on -v0 and comfortably inside the mouth on -Loose-v0.
# Same policy, same shift, opposite prediction, and no boundary case to argue about.
#
# Also runs the -Tight-v0 rung (0.5 mm clearance), where the model predicts the cliff moves IN:
# even 1 mm should be a floor there, while it costs 3 points on -v0.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
run_task() {  # name, task, flags...
  local name="$1" task="$2"; shift 2
  local out="$RUN/robust_${name}.json"
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  echo "=== $name  $task  $*  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task "$task" \
    --num-envs 32 --episodes 128 --seed 777 --out "$out" "$@" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $out"
}

run_task loose_dy_p002 Rebot-PrecisionSlot-Loose-v0 --slot-dy 0.002
run_task tight_dy_p001 Rebot-PrecisionSlot-Tight-v0 --slot-dy 0.001
# the clearance controls: same task, dy = 0, so the crossed cells have their own baselines
# instead of borrowing -v0's. Without these, "loose_dy_p002 = 0.9" could just mean Loose is easy.
run_task loose_dy_p000 Rebot-PrecisionSlot-Loose-v0
run_task tight_dy_p000 Rebot-PrecisionSlot-Tight-v0
echo "=== crossed dy cells done  $(date -Is) ==="
