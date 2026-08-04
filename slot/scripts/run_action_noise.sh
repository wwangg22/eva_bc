#!/usr/bin/env bash
# EXP_ROBUSTNESS §10 -- actuation noise, the counterpart to --obs-noise.
#
# Every perturbation measured so far does its damage to the SAME phase. Slot dx = -10/+20 mm,
# spawn box x2.0 and 20 % sensor noise are three unrelated stresses, and all three end with the
# block short of full depth rather than off-axis. --obs-noise 0.20 is the sharpest case: median
# failure |lateral| 0.11 mm -- the lowest of any cell -- at a median depth of -43.8 mm. The block
# is delivered dead on axis and then stops.
#
# So the push is the fragile part, and an actuator error is the perturbation that acts on the
# push most directly. This is the one axis in the robustness sweep where I would predict a
# WORSE outcome than the sensing equivalent at the same nominal magnitude -- chunked control
# low-passes an observation error over 15 steps, but an action error goes straight to the joints
# every step with nothing in between.
#
# Same subject and protocol as every other robustness cell: champion, -v0, s777, 128 ep / 32 envs.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
for CELL in "act002:0.02" "act005:0.05" "act020:0.20"; do
  NAME="${CELL%%:*}"; MAG="${CELL##*:}"
  OUT="$RUN/robust_${NAME}.json"
  [ -f "$OUT" ] && { echo "SKIP $OUT"; continue; }
  echo "=== $NAME  --action-noise $MAG  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs 32 --episodes 128 --seed 777 --action-noise "$MAG" --out "$OUT" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $OUT"
done
echo "=== action-noise cells done  $(date -Is) ==="
