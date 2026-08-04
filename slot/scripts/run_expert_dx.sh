#!/usr/bin/env bash
# EXPERT CONTROL for EXP_ROBUSTNESS §6b (docs/slot/EXP_ROBUSTNESS.md §8).
#
# The learned policy is flat in slot x from 0 to +10 mm and loses ~10 points at -10 mm and
# +20 mm. Two explanations fit that equally well from policy data alone:
#   (a) the ROBOT is worse at those slot positions -- reach, wrist conditioning, push geometry
#   (b) the POLICY is worse there -- it is furthest from what it was trained on
# The scripted expert discriminates them, because it is TOLD the new slot position exactly
# (--slot_dx patches SLOT_CENTER *and* moves insert_x with it) and re-solves its own IK seed for
# the new target. Its trajectory is open-loop and near-optimal by construction, so whatever it
# loses is the robot's, not a policy's.
#
# If the expert is flat across all four positions, the dips are the policy's and Stage D has a
# target. If the expert dips at -10 and +20 too, the dips are the arm's and no policy change
# would recover them.
#
# --out_dir is per-cell ON PURPOSE: run_expert.py names its report expert_<task>.json, so four
# runs into one directory would silently overwrite each other and deliver one result while
# looking like four. (make_videos.sh already shipped that exact bug once.)
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

for CELL in "m010:-0.010" "p000:0.000" "p010:0.010" "p020:0.020"; do
  TAG="${CELL%%:*}"; DX="${CELL##*:}"
  OUT="logs/expert_dx/$TAG"
  [ -f "$OUT/expert_Rebot-PrecisionSlot-v0.json" ] && { echo "SKIP $OUT"; continue; }
  echo "=== expert dx=$DX -> $OUT  $(date -Is) ==="
  # per-cell seed_file: plan.py CACHES the solved IK seed keyed on insert_x and REWRITES the file
  # (plan.py:147,171). Sharing logs/expert/seed_q.json across four insert_x values would clobber
  # the canonical seed that demo collection uses. Correctness would survive (the cache key would
  # miss and it would re-solve) but a shared artifact would be quietly modified, which is not
  # mine to do.
  mkdir -p "$OUT"
  python scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 128 \
    --slot_dx "$DX" --out_dir "$OUT" --seed 777 --seed_file "$OUT/seed_q.json" 2>&1 \
    | grep -E "expert|seated|SLOT" || echo "!!! FAILED (continuing): $OUT"
done
echo "=== expert dx control done  $(date -Is) ==="
