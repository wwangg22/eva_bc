#!/usr/bin/env bash
# Does letting the arm ARRIVE before closing the gripper fix the grasp?
#
# The expert plans a grasp pose whose forward kinematics puts the TCP on the cube to within
# 1.5 mm (the gate enforces it), commands those joint angles, and then closes. MEASURED with
# `--chatty` at 64 envs:
#
#   [diag home ] TCP err vs plan: median  1.1 mm  p90  1.7 mm   <- after a 40-step settle
#   [diag grasp] TCP err vs plan: median 11.1 mm  p90 60.0 mm   <- NO settle at all
#   [diag lift ] gap -1.2 mm, cube z 27.2 mm                    <- fingers shut on air
#
# The gripper opens to 89 mm on a 53-58 mm cube, so the margin is ~17 mm a side. An 11 mm
# arrival error spends most of it; 60 mm is a clean miss. The transit leg got a settle hold
# when it was found to be lagging -- the grasp descent never did, so the fingers close while
# the arm is still travelling.
#
# GRASP_SETTLE=0 reproduces the shipped behaviour exactly and is the control. Paired: same
# seed, same layouts, end-to-end (NO --teleport-pregrasp -- see 03_EXPERT_AND_BC.md 5.8 for
# why a teleported sweep measures a different manoeuvre).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc
N=${N:-64}
OUT=re3sim/runs/settle
mkdir -p "$OUT"
for S in ${SS:-0 20 40 80 160}; do
  L="$OUT/s${S}.log"
  GRASP_SETTLE=$S BIAS_MAX=0.0 python -u re3sim/expert/collect_demos.py --headless --chatty \
    --num_envs "$N" --batches 1 --seed 11 --out "$OUT/s${S}.hdf5" > "$L" 2>&1
  printf "settle %-4s  %s\n" "$S" "$(grep -a 'SUCCESS' "$L" | tail -1)"
  printf "            %s\n" "$(grep -a 'diag grasp' "$L" | tail -1 | sed 's/.*TCP/TCP/')"
done
