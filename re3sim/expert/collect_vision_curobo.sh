#!/usr/bin/env bash
# Collect the workstation vision dataset with the CUROBO expert (run_expert_ws.py) —
# the expert Big Will's team trained TO USE for collection (2026-08-11 directive).
#
#   bash re3sim/expert/collect_vision_curobo.sh <out-dir> <episodes> <seed> [extra args...]
#
# Single env, serial episodes (cuRobo plans per-env); shards stream to disk per episode,
# so a killed run keeps everything already written. The task must be a -Vision*/-VisionDR
# variant: those own the two 160x120 student cameras (and the -VisionDR reset events own
# the station-cam pose).
#
# RE3SIM_ARM_START_JITTER=0.15 is deliberate (see collect_vision.sh for the sweep numbers).
set -euo pipefail

OUT=${1:?out dir}
N=${2:-64}
SEED=${3:?seed}
shift 3 2>/dev/null || true
# The task decides whether DR renders at all — make the choice VISIBLE at launch.
TASK=${TASK:-Rebot-Workstation-PickPlace1-Vision-v0}
echo "[collect_vision_curobo] task=$TASK episodes=$N seed=$SEED out=$OUT" >&2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

RE3SIM_ARM_START_JITTER=${RE3SIM_ARM_START_JITTER:-0.15} \
python -u re3sim/expert/run_expert_ws.py --task "$TASK" \
    --episodes "$N" --seed "$SEED" --shards "$OUT" "$@"
