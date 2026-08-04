#!/usr/bin/env bash
# EXP_STEER §12 -- the path PPO would actually walk.
#
# x0-steering sets x0 = alpha*tanh(z) broadcast across all 50 chunk positions. So its reachable
# set is exactly the broadcast family, parameterised by how far z is from 0:
#
#   z = 0        -> x0 = zeros                 measured: 0.167
#   z -> large   -> x0 = b<something>          measured: b1 = 0.000, b2 = 0.000
#
# The steering policy is initialised at mu = 0, sigma ~ 0.3, i.e. right at the zeros end. This
# ladder fills in the middle so the claim "PPO would find every direction worse and converge
# back to z = 0" rests on a measured curve rather than on two endpoints and an assumption.
#
# Scales are chosen to land where the steerer actually lives: ||x0|| = 18.71 * k, and the
# initialisation sits at k ~ 0.30, its saturation ceiling at k = 0.76.
#
# Idempotent. ~9 min.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"

for K in 0.15 0.30 0.60; do
  OUT="$RUN/x0bcast_act002_b1_k${K/./}.json"
  [ -f "$OUT" ] && { echo "SKIP $OUT"; continue; }
  echo "=== broadcast b1 x $K + 2% action noise  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs 32 --episodes 128 --seed 777 --fixed-x0 b1 --x0-scale "$K" \
    --action-noise 0.02 --out "$OUT" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $OUT"
done

# Is the broadcast family broken by the NOISE, or broken full stop? If b1 also collapses with no
# perturbation at all, then the steering parameterisation inherited from EXP07 was never viable
# on this task in ANY condition -- including the clean control arm (arm B) -- and that is a
# different, stronger statement than "it cannot express the noise-robust latent".
OUT="$RUN/x0bcast_clean_b1.json"
if [ -f "$OUT" ]; then echo "SKIP $OUT"; else
  echo "=== broadcast b1 CLEAN  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs 32 --episodes 128 --seed 777 --fixed-x0 b1 --out "$OUT" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $OUT"
fi

echo "=== broadcast ladder done  $(date -Is) ==="
