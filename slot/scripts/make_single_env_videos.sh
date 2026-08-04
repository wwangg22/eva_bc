#!/usr/bin/env bash
# Single-workstation clips of the champion, for Big Will's review.
#
# ONE env, ONE episode, camera framed on that one workstation -- not the default wide shot that
# shows the whole grid. Three clips, chosen so the pair of outcomes is visible rather than
# described:
#
#   success_clean      the champion doing the task. -v0, no perturbation. (later-cohort rate 0.979)
#   failure_act002     the SAME policy under 2 % actuation noise. (0.146) This is the failure the
#                      whole of session 7 is about: the block is carried to the staging waypoint
#                      at x = 0.166 m, held at carry height, dead on the slot axis -- and the push
#                      is simply never attempted. A table cannot show "it just stops".
#   success_act002_s4  the SAME noise, with the flow's integration latent held at a good fixed
#                      draw (--fixed-x0 4). (0.823) The recovery, same conditions as the clip
#                      above, so the two are watchable side by side.
#
# Outcomes are stochastic at n = 1, so each clip RETRIES over spawn seeds until the episode
# actually has the outcome the filename claims. A clip labelled "failure" that shows a success
# would be worse than no clip.
#
# Camera (world metres; the env frame has the table top at z = 0, robot base at the origin,
# slot centred at x = 0.245, block spawning near (0.22, -0.13)):
#   eye    (0.75, -0.55, 0.45)   3/4 view from the block-spawn side, ~0.83 m out, elevated
#   lookat (0.24,  -0.03, 0.05)  between the staging waypoint and the slot mouth
# Both are one-line overrides if the framing wants adjusting.
#
# Usage:  bash scripts/make_single_env_videos.sh [runs/bc_armB_seed0]
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
EYE=(0.75 -0.55 0.45)
LOOKAT=(0.24 -0.03 0.05)
SEEDS=(777 778 779 780 781 782 783 784 785 786 787 788)

shoot() {  # name, want_success(1|0), extra flags...
  local name="$1" want="$2"; shift 2
  local dest="$RUN/videos/single_${name}.mp4"
  [ -f "$dest" ] && { echo "SKIP $dest"; return 0; }
  local raw="$RUN/videos/rl-video-step-0.mp4"
  for SEED in "${SEEDS[@]}"; do
    local js="$RUN/video_single_${name}.json"
    rm -f "$raw" "$js"
    echo "=== $name  seed $SEED  (want success=$want)  $(date -Is) ==="
    python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
      --num-envs 1 --episodes 1 --seed "$SEED" --video --video-length 620 \
      --viewer-eye "${EYE[@]}" --viewer-lookat "${LOOKAT[@]}" \
      --out "$js" "$@" 2>&1 | grep -E "eval_act|Moviepy" || true
    local got
    got=$(python -c "import json,sys; d=json.load(open('$js')); print(int(d['per_episode'][0]['success']))" 2>/dev/null || echo x)
    echo "    -> success=$got (wanted $want)"
    if [ "$got" = "$want" ]; then
      # gymnasium's RecordVideo always writes rl-video-step-0.mp4 to the same folder, so rename
      # NOW or the next clip silently overwrites it. This bug has already shipped twice here.
      if [ -f "$raw" ]; then mv "$raw" "$dest"; echo "    kept -> $dest"; return 0
      else echo "!!! outcome matched but no video file was produced for $name"; return 1; fi
    fi
  done
  echo "!!! no seed in {${SEEDS[*]}} produced success=$want for $name"
  return 1
}

shoot success_clean      1
shoot failure_act002     0 --action-noise 0.02
shoot success_act002_s4  1 --action-noise 0.02 --fixed-x0 4

echo
echo "clips written under $RUN/videos/ :"
ls -la "$RUN/videos/"single_*.mp4 2>/dev/null || echo "  (none -- check the log above)"
echo
echo "NOTE: video_single_*.json beside them are NOT measurements. One episode, seed-selected for"
echo "      a specific outcome -- that is the whole point of the clip and the opposite of a"
echo "      sample. Quote runs/bc_armB_seed0/robust_*.json and x0probe_*.json instead."
