#!/usr/bin/env bash
# Collect the workstation vision dataset: two cameras, expert-driven, no privileged inputs.
#
#   bash re3sim/expert/collect_vision.sh <out-dir> <num_envs> <batches> <seed> [extra args...]
#
# RE3SIM_SPLATS_PER_ENV=1 is NOT optional and collect_demos.py refuses without it: the
# gaussian desk is a single world prim by default, so every env but one would render the arm
# floating on a bare ground plane and nothing downstream would notice.
#
# RE3SIM_ARM_START_JITTER=0.15 is deliberate. The arm-start sweep measured the cost -- 96.1 %
# -> 90.6 %, still above the bar -- and a policy whose training set contains exactly one
# initial arm pose has no reason to be robust to any other, which is the first thing real
# hardware violates. 0.30 rad costs too much (75.0 %).
set -euo pipefail

OUT=${1:?out dir}
N=${2:-128}
B=${3:-3}
SEED=${4:-21}
shift 4 2>/dev/null || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc

RE3SIM_SPLATS_PER_ENV=1 RE3SIM_ARM_START_JITTER=${RE3SIM_ARM_START_JITTER:-0.15} \
python -u re3sim/expert/collect_demos.py --headless \
    --num_envs "$N" --batches "$B" --seed "$SEED" \
    --shards "$OUT" --out "${OUT%/}_state.hdf5" "$@"
