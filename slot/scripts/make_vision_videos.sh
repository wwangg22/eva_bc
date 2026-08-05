#!/usr/bin/env bash
# Drive make_vision_videos.py ONE SEED PER PROCESS until both outcomes are captured.
#
# The retry loop lives here rather than inside the python because Isaac Sim cannot tear down and
# rebuild a camera-bearing env within a single process: the second gym.make raises "Unable to
# retrieve replicator graph". The first version looped internally and died immediately after
# writing its first clip.
#
# At n = 1 the policy's 0.804 makes the outcome a draw, so the script keeps going until it has a
# genuine success AND a genuine failure -- each clip is named by what actually happened, never by
# what was wanted.
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

OUT="${OUT:-runs/vision_bc/v1/videos}"
CKPT="${CKPT:-runs/vision_bc/v1/ckpt_final.pt}"

have () { ls "$OUT"/vision_$1_s*.mp4 >/dev/null 2>&1; }

for SEED in 777 778 779 780 781 782 783 784 785 786 787 788; do
  have success && have failure && break
  echo "=== seed $SEED  $(date -Is) ==="
  python scripts/make_vision_videos.py --ckpt "$CKPT" --out "$OUT" --seed "$SEED" 2>&1 \
    | grep -E "\[video\]|Traceback|Error|replicator" || true
done

echo
if have success && have failure; then echo "both outcomes captured:"; else echo "INCOMPLETE:"; fi
ls -la "$OUT"/*.mp4 2>/dev/null || echo "  (none)"
