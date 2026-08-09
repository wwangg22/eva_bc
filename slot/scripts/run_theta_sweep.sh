#!/usr/bin/env bash
# GATE B -- can the arm physically insert at each slot angle? This sets theta_max.
#
# Big Will's instruction: "ensure the task is solvable by the arm (check if the arm can
# actually reach the required position)". This is that check, run the only way that answers it
# honestly: the scripted expert, which is TOLD the angle, on real physics. An IK-convergence
# probe would say "reachable" for poses the gripper cannot actually hold a block through.
#
# theta_max = the largest |angle| that still holds >= 0.95 seated. Angles are pinned exactly
# (--slot_yaw takes one value), so a cell that fails names a specific angle rather than a range.
#
# One process per angle: the CEM seed is cached to logs/expert/seed_q.json and shared, so the
# per-process cost is env build only.
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

N="${N:-128}"
OUT=logs/theta_sweep
mkdir -p "$OUT"

for TH in 0.000 0.175 -0.175 0.350 -0.350 0.500 -0.500 0.700 -0.700 0.900 -0.900; do
  DEST="$OUT/theta_${TH}.json"
  [ -f "$DEST" ] && { echo "SKIP $DEST"; continue; }
  echo "=== theta $TH rad  $(date -Is) ==="
  python scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs "$N" \
      --slot_yaw "$TH" --out_dir "$OUT" 2>&1 \
    | grep -E "seated success|lateral mean|depth   mean|\|yaw\|   mean|failures:|plan converged|Traceback|Error" \
    || true
  # run_expert names its own file; give it the angle so cells cannot overwrite each other
  if [ -f "$OUT/expert_Rebot-PrecisionSlot-v0.json" ]; then
    mv "$OUT/expert_Rebot-PrecisionSlot-v0.json" "$DEST"
  else
    echo "!!! no JSON produced for theta $TH"
  fi
done

echo
echo "=== gate B summary  $(date -Is) ==="
python analysis/theta_sweep.py "$OUT"
