#!/usr/bin/env bash
# Is the block's post-traverse lag TRANSIENT SWING or a PERMANENT SLIP in the pads?
#
# ANGLED_SLOT.md 8a measured the block lagging the TCP along the slot axis by up to 8.8 mm
# after a long traverse, and the push preserves whatever offset it starts with, so the insert
# lands ~5 mm shallow. Two explanations with opposite fixes:
#
#   transient -- the block is still swinging on its 0.365 s pendulum when the push begins.
#                Holding longer at the end of the traverse fixes it.
#   slip      -- the block slid along the pad faces during the traverse and friction is now
#                holding it there. Holding longer changes NOTHING, and the fix has to be a
#                slower traverse or a compensated insert target.
#
# theta = -0.350 is the cell to test: 157 mm traverse, 0.703 seated, depth 42.0 mm, and the
# grip provably intact (128/128 held through the push), so nothing else is confounded.
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

OUT=logs/turn_settle
mkdir -p "$OUT"
for S in 25 60 120; do
  echo "=== turn_settle $S  $(date -Is) ==="
  python scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 128 \
      --slot_yaw -0.350 --turn_settle "$S" --out_dir "$OUT" 2>&1 | tee "$OUT/full_$S.log" \
    | grep -E "seated success|depth   mean|lateral mean|trajectory used|envs reset|failures:"
  [ -f "$OUT/expert_Rebot-PrecisionSlot-v0.json" ] && mv "$OUT/expert_Rebot-PrecisionSlot-v0.json" "$OUT/settle_$S.json"
done
