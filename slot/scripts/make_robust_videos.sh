#!/usr/bin/env bash
# Record what the two ends of EXP_ROBUSTNESS actually look like.
#
# Big Will reviews all rendered output, so these are produced for review and never inspected
# here. Two clips, chosen because they are the two results that a table cannot convey:
#   dx_p010  -- slot moved 10 mm FURTHER away, policy scores 96/96. The block should visibly
#               end up 10 mm deeper in absolute terms; the policy is tracking, not replaying.
#   dy_p005  -- slot moved 5 mm SIDEWAYS, policy scores 0/96. This is the one to watch: the
#               numbers say the block hits the wall FACE (final x 145 mm vs 257 nominal,
#               max height 75 mm vs 67) rather than missing narrowly, and a clip either shows
#               that or shows that I have misread it.
#
# Usage:  bash scripts/make_robust_videos.sh [runs/bc_armB_seed0] [num_envs]
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
ENVS="${2:-4}"

shoot() {  # name, flags...
  local name="$1"; shift
  local vid="$RUN/videos/rl-video-step-0.mp4"
  local dest="$RUN/videos/robust_${name}.mp4"
  [ -f "$dest" ] && { echo "SKIP $dest"; return; }
  echo "=== recording $name  $*  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs "$ENVS" --episodes "$ENVS" --seed 777 --video --video-length 620 \
    --out "$RUN/video_robust_${name}.json" "$@" 2>&1 | grep -E "eval_act|Moviepy|video" || true
  # gymnasium's RecordVideo always writes rl-video-step-0.mp4 to the same folder, so rename
  # immediately or the next clip silently overwrites this one and the script delivers one video
  # while claiming two. This bug already shipped once in make_videos.sh.
  if [ -f "$vid" ]; then mv "$vid" "$dest"; else echo "!!! no video produced for $name"; fi
}

shoot dx_p010 --slot-dx 0.010
shoot dy_p005 --slot-dy 0.005

echo
echo "videos written under $RUN/videos/ :"
ls -la "$RUN/videos/" 2>/dev/null || echo "  (none found -- check the log above)"
echo
echo "NOTE: video_robust_*.json beside them are NOT measurements -- with --episodes equal to"
echo "      --num-envs every episode is a first-episode one, the cohort carrying the PhysX"
echo "      warm-start bias. Quote runs/bc_armB_seed0/robust_*.json instead."
