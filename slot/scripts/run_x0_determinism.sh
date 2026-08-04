#!/usr/bin/env bash
# EXP_TIGHT step B -- the determinism test (docs/slot/EXP_TIGHT.md section 5).
#
# Central hypothesis: what fails on the tight channel is the policy's OWN x0 sampling noise
# eating a 0.5 mm budget, not a deficiency of the learned trajectory. --fixed-x0 zeros freezes
# the flow's integration noise at the distribution's mode, making each chunk a deterministic
# function of the observation. The controller has always supported this; only the flag was new.
#
# Uses bc_armA_seed0 as a FIXED reference rather than "the champion": its stochastic baseline
# already exists at all three clearances at spawn 777, so every comparison here is paired at the
# same spawn seed and can be read with analysis/paired_evals.py. Which arm wins the arm
# comparison is a separate question and does not change the mechanism being tested.
#
# Usage:  bash scripts/run_x0_determinism.sh [run_dir]
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armA_seed0}"

for TASK in Rebot-PrecisionSlot-Loose-v0 Rebot-PrecisionSlot-v0 Rebot-PrecisionSlot-Tight-v0; do
  OUT="$RUN/eval_x0zeros_${TASK}_s777.json"
  [ -f "$OUT" ] && { echo "SKIP $OUT (exists)"; continue; }
  echo "=== fixed-x0 zeros: $TASK  $(date -Is) ==="
  # Same --num-envs/--episodes/--seed as the sweep, so the cells are directly comparable and
  # the first-episode cohort is the same size. See run_eval_sweep.sh for why that matters.
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task "$TASK" \
    --num-envs 32 --episodes 128 --seed 777 --fixed-x0 zeros --out "$OUT" 2>&1 \
    | grep -E "eval_act|fixed x0" || echo "!!! EVAL FAILED (continuing): $OUT"
done
echo "=== x0 determinism test done  $(date -Is) ==="
