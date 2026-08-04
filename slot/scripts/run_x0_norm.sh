#!/usr/bin/env bash
# EXP_STEER §14 -- the norm result, and the version of it that is NOT hardcoding a latent.
#
# The §10 ladder on seed 1 came back MONOTONE INCREASING in ||x0||:
#   k=0.30 -> 0.302   k=0.76 -> 0.573   k=1.00 -> 0.771   k=1.50 -> 0.875
# (against stochastic 0.146 and a clean ceiling of 0.979). The good region is not the prior's
# typical shell at 18.71 -- it is OUTSIDE it, and the ladder had not turned over by 29.29.
#
# Two things follow, and the second is the one that matters:
#
#   (1) EXTEND THE LADDER. k = 2.0 and 3.0 on the same latent, to find where it turns over. If it
#       never does, the "typical set" framing of §10b is simply the wrong picture.
#
#   (2) SCALE WITHOUT CHOOSING. --x0-scale now also applies to a FRESHLY DRAWN x0 at every
#       refill. That is not a hardcoded latent: no search, no per-checkpoint constant, nothing
#       selected on a validation set -- one scalar, applied to the sampling the policy already
#       does. If scaled sampling recovers most of the deficit, the deliverable is a one-line
#       inference change rather than "we found a magic seed", and it is a knob a learned steerer
#       could then move as a function of state.
#
# Clean cells guard the obvious risk: a bolder policy may be worse when there is nothing to be
# bold about. seed1 at k=1 already costs 14.6 points clean (0.833 vs 0.979), so this is real.
#
# Protocol: champion, -v0, s777, 128 ep / 32 envs. Idempotent. ~30 min.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
COMMON=(--task Rebot-PrecisionSlot-v0 --num-envs 32 --episodes 128 --seed 777)

cell() {  # name, out, extra args...
  local name="$1" out="$2"; shift 2
  [ -f "$out" ] && { echo "SKIP $out"; return 0; }
  echo "=== $name  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" "${COMMON[@]}" "$@" --out "$out" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $out"
}

# (2) first -- it is the one that decides whether any of this needs a latent search at all.
for K in 1.50 2.00 3.00; do
  cell "SAMPLED x0 x$K + 2% action noise" "$RUN/x0samp_act002_k${K/./}.json" \
    --x0-scale "$K" --action-noise 0.02
done
cell "SAMPLED x0 x2.00 CLEAN" "$RUN/x0samp_clean_k200.json" --x0-scale 2.00

# (1) extend the fixed-latent ladder past where it was still climbing
for K in 2.00 3.00; do
  cell "fixed seed1 x$K + 2% action noise" "$RUN/x0norm_act002_s1_k${K/./}.json" \
    --fixed-x0 1 --x0-scale "$K" --action-noise 0.02
done
cell "fixed seed1 x1.50 CLEAN" "$RUN/x0norm_clean_s1_k150.json" --fixed-x0 1 --x0-scale 1.50

echo "=== x0 norm sweep done  $(date -Is) ==="
