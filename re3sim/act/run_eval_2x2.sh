#!/usr/bin/env bash
# The 2x2 eval (HANDOFF section 7, Step E): BOTH students x {nominal, DR} Play tasks.
# 64 episodes per cell (2 seeds x 32 envs), ONE Isaac job at a time -- the four evals
# run strictly sequentially inside one detached unit:
#
#   systemd-run --user --collect -p MemoryMax=26G --unit=eval-2x2 bash -c \
#    'source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6 && \
#     exec bash ~/Desktop/isaacLab/reBot/reBot_ACT/re3sim/act/run_eval_2x2.sh'
#
# Per-cell json (failure taxonomy included -- read that FIRST, not just the rate) and
# log land in re3sim/runs/eval_2x2/<student>_<cell>.{json,log}.
set -u
cd "$(dirname "$0")/../.."   # reBot_ACT

OUT=re3sim/runs/eval_2x2
mkdir -p "$OUT"

declare -A TASKS=(
  [nom]=Rebot-Workstation-PickPlace1-Vision-Play-v0
  [dr]=Rebot-Workstation-PickPlace1-VisionDR-Play-v0
)

for student in vbc_base vbc_vdr; do
  ckpt=re3sim/runs/$student/ckpt_final.pt
  [ -f "$ckpt" ] || { echo "MISSING CKPT: $ckpt"; exit 1; }
  for cell in nom dr; do
    echo "=== $student x ${TASKS[$cell]} $(date +%H:%M:%S) ==="
    python -u re3sim/act/eval_flow_vision.py --ckpt "$ckpt" \
      --task "${TASKS[$cell]}" --num_envs 32 --seeds 88000,88001 \
      --out "$OUT/${student}_${cell}.json" > "$OUT/${student}_${cell}.log" 2>&1
    echo "=== $student $cell rc=$? ==="
  done
done
echo "EVAL-2x2-DONE"
