#!/usr/bin/env bash
# Does the block SLIDE along the pads or ROTATE about the grip point -- and does slowing the
# traverse reduce it?
#
# The in-hand displacement reaches -12.5 mm at theta = -0.5, but inertia alone cannot explain
# it: 43 N of squeeze at mu = 1.0 against a 0.04 kg block tolerates ~1000 m/s^2 before sliding.
# A block hanging at roll phi displaces its centre by 33*sin(phi) instead, so 12.5 mm is 22 deg
# of roll -- and lerp_path's own docstring records exactly this failure at 5 deg on the
# axis-aligned task. tilt_deg settles it.
#
# turn_per_wp 5 is the ceiling the 720-step budget allows at this angle (~704 used).
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
OUT=logs/roll; mkdir -p "$OUT"
for W in 3 5; do
  for TH in -0.500 0.000; do
    echo "=== theta $TH  turn_per_wp $W ==="
    python scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 128 --slot_yaw "$TH" \
        --turn_per_wp "$W" --out_dir "$OUT" >"$OUT/full_${TH}_w$W.log" 2>&1
    [ -f "$OUT/expert_Rebot-PrecisionSlot-v0.json" ] && mv "$OUT/expert_Rebot-PrecisionSlot-v0.json" "$OUT/th${TH}_w$W.json"
    grep -E "seated success|depth   mean|trajectory used" "$OUT/full_${TH}_w$W.log"
  done
done
