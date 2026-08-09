#!/usr/bin/env bash
# WHERE does the depth go at negative theta -- the ARM or the GRIP?
#
# tcp_err_mm = the arm's own shortfall against its commanded waypoint, along the slot axis.
# slip_mm    = the block's displacement inside the pads, same axis.
# A longer settle changed nothing (logs/turn_settle), so the deficit is static; these two
# numbers say which of the two static causes it is, and they have opposite fixes.
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
OUT=logs/slip; mkdir -p "$OUT"
for TH in 0.350 0.000 -0.175 -0.350 -0.500; do
  echo "=== theta $TH ==="
  python scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 128 --slot_yaw "$TH" \
      --out_dir "$OUT" >"$OUT/full_$TH.log" 2>&1
  [ -f "$OUT/expert_Rebot-PrecisionSlot-v0.json" ] && mv "$OUT/expert_Rebot-PrecisionSlot-v0.json" "$OUT/theta_$TH.json"
  grep -E "seated success|depth   mean" "$OUT/full_$TH.log"
done
