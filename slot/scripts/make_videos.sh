#!/usr/bin/env bash
# Record one video per task variant for a trained checkpoint.
#
# Big Will reviews all rendered output, so these are produced for review and never inspected
# here. Paths are printed at the end.
#
# Play-v0 is deliberately absent: it differs from -v0 only in scene.num_envs / env_spacing
# (both overridden by parse_env_cfg) and enable_corruption, which the base cfg already sets
# False. Recording it would produce a second copy of the -v0 video.
#
# Usage:  bash scripts/make_videos.sh runs/bc_armA_seed0/ckpt_final.pt [num_envs]
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

CKPT="${1:?usage: make_videos.sh <ckpt.pt> [num_envs]}"
ENVS="${2:-4}"

for TASK in Rebot-PrecisionSlot-Loose-v0 Rebot-PrecisionSlot-v0 Rebot-PrecisionSlot-Tight-v0; do
  echo "=== recording $TASK  $(date -Is) ==="
  # 620 steps covers one full 600-step episode plus the reset margin. Recording more would
  # splice a second episode into the same file.
  python slot_act/eval_act.py --ckpt "$CKPT" --task "$TASK" \
    --num-envs "$ENVS" --episodes "$ENVS" --seed 777 \
    --video --video-length 620 \
    --out "$(dirname "$CKPT")/video_eval_${TASK}.json" 2>&1 | grep -E "eval_act|Moviepy|video" || true
  # gymnasium's RecordVideo always names the file rl-video-step-0.mp4 and always writes it to
  # the same videos/ dir, so WITHOUT this rename each clearance silently overwrites the last
  # one and the script delivers a single video while claiming three. (It happened: the only
  # visible sign was a UserWarning about "Overwriting existing videos".) Rename immediately,
  # before the next task can clobber it.
  VID="$(dirname "$CKPT")/videos/rl-video-step-0.mp4"
  if [ -f "$VID" ]; then
    mv "$VID" "$(dirname "$CKPT")/videos/${TASK}.mp4"
  else
    echo "!!! no video produced for $TASK"
  fi
done

echo
echo "videos written under $(dirname "$CKPT")/videos/ :"
ls -la "$(dirname "$CKPT")/videos/" 2>/dev/null || echo "  (none found -- check the log above)"
echo
echo "NOTE: the video_eval_*.json files beside them are NOT measurements. With --episodes equal"
echo "      to --num-envs, every episode is an episode_index_in_env == 0 episode, so the whole"
echo "      file is the first-episode cohort -- the one carrying the PhysX warm-start bias"
echo "      (measured on this project between -2.1 and +18.7 points). They are named"
echo "      video_eval_* precisely so no glob for eval_ckpt_* picks them up. Do not quote them."
