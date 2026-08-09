#!/usr/bin/env bash
# Re-measure the grasp height against whatever collider the env currently loads.
#
# GRIP_Z = 25 mm is a MEASURED number, but it was measured against an analytic 56 mm cube
# (p01_grasp_feasibility.py). The reconstructed mesh is 53.3 x 58.0 x 54.5 mm with rounded
# corners and a convex-hull collider -- a different object to close on, and the expert
# measured 19.5 % on it against ~31 % on the primitive. A constant that was measured on a
# different object is an assumption, so it gets re-measured rather than re-used.
#
# Teleport protocol throughout: this isolates the GRASP from the transit, which is a separate
# open problem (03_EXPERT_AND_BC.md 3b).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc
N=${N:-64}
for Z in ${ZS:-0.032 0.040 0.048 0.056 0.064}; do
  L=re3sim/runs/gripz/z${Z}.log
  mkdir -p re3sim/runs/gripz
  # NO --teleport-pregrasp. The first sweep used it, and under the pre-fix waypoint ordering
  # `seg_pre` was effectively AT the cube, so it measured a grasp with no descent -- which is
  # why it liked 32 mm. With the transit corrected the arm actually descends ~140 mm onto the
  # object and sags on the way, so the height has to be re-measured end to end.
  GRIP_Z=$Z BIAS_MAX=0.0 python -u re3sim/expert/collect_demos.py --headless \
    --num_envs "$N" --batches 1 --seed 11 \
    --out "re3sim/runs/gripz/z${Z}.hdf5" > "$L" 2>&1
  printf "grip_z %-6s  %s\n" "$Z" "$(grep -a 'SUCCESS' "$L" | tail -1)"
done
