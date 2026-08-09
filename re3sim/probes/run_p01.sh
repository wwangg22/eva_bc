#!/usr/bin/env bash
# B1 gate sweep. Controls FIRST -- if control_24 does not reproduce its measured 100 %,
# the search is under-budgeted and every later row is void.
# One process per object: a collider cannot be resized after the scene is built (C8).
set -u
cd "$(dirname "$0")/../.."
for obj in control_24 control_56 rubixcube tapemeasure rolloftape_onedge rolloftape; do
  echo "=============================================================="
  echo "=== $obj"
  echo "=============================================================="
  stdbuf -oL -eL python -u re3sim/probes/p01_grasp_feasibility.py --object "$obj" --headless 2>&1 \
    | stdbuf -oL grep -E "^\s|^\[b1\]|^  " | tail -40
done
echo "=== P01 sweep finished"
