#!/usr/bin/env bash
# EXP_STEER §10 -- follow-up to the constant-x0 probe, launched because seed 1 scored 0.771
# under 2 % action noise against 0.146 stochastic and 0.167 for x0 = zeros.
#
# Two questions, and the first one decides how the PPO arms must be PARAMETERISED:
#
#   (1) DOES THE GOOD LATENT NEED PER-CHUNK-POSITION STRUCTURE?  `--fixed-x0 <seed>` is a
#       (50, 7) matrix -- a different x0 vector at each of the 50 chunk positions. x0-STEERING
#       CANNOT EXPRESS THAT: SteerCore.set_steer broadcasts one 7-vector across all 50
#       (steer_core.py:58). So if the 0.771 depends on the across-position structure, the
#       steering action space is the wrong shape and no amount of PPO will find it.
#       `--fixed-x0 b<seed>` is row 0 of the same matrix repeated 50 times: the exact family
#       steering can reach. b<seed> vs <seed> isolates structure and nothing else.
#
#   (2) HOW COMMON IS A GOOD LATENT?  Four more full draws. The stochastic policy takes a
#       FRESH x0 every 15-step refill, ~40 per episode; if good latents were common it would
#       hit them often and still scores 0.146, which would make the finding about COMMITMENT
#       rather than about which latent. `zeros` already shows commitment alone is not enough
#       (constant, and still 0.167), so the two effects are separable -- but the rate matters
#       for the write-up and n = 5 is too few to quote.
#
# Same protocol as everything else: champion, -v0, s777, 128 ep / 32 envs, later cohort.
# Idempotent. ~25 min.
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

# (1) broadcast form of each probed seed -- what steering could actually reach.
#     b1 first: it is the one that pairs with the 0.771 cell.
for S in 1 2 3 4; do
  cell "broadcast x0=b$S + 2% action noise" "$RUN/x0probe_act002_b$S.json" \
    --fixed-x0 "b$S" --action-noise 0.02
done

# (2) four more full draws, to estimate how often a draw is a good one.
for S in 5 6 7 8; do
  cell "full x0=seed$S + 2% action noise" "$RUN/x0probe_act002_s$S.json" \
    --fixed-x0 "$S" --action-noise 0.02
done

# (3) is seed 1 a NOISE-ROBUST latent, or just a better latent full stop? Clean it should be
#     indistinguishable from stochastic's 0.979 and zeros' 0.938 -- if instead it is clearly
#     best clean too, the finding is "some latents are better", not "some latents survive noise".
cell "full x0=seed1 CLEAN" "$RUN/x0probe_clean_s1.json" --fixed-x0 1

# (4) the harder condition. Stochastic scores 0/96 at 5 % action noise (EXP_ROBUSTNESS §10).
#     If committing still recovers ground there, the effect is not a 2 %-specific artefact.
cell "full x0=seed1 + 5% action noise" "$RUN/x0probe_act005_s1.json" \
  --fixed-x0 1 --action-noise 0.05

# (5) THE NORM LADDER -- why is `zeros` bad when every random draw is good?
#     A d=350 standard normal concentrates on a shell of radius sqrt(350) = 18.71. x0 = zeros
#     has norm 0: it is not "the average sample", it is nowhere near the typical set the flow
#     was trained to integrate from. That would explain zeros = 0.167 with no appeal to
#     "the mode is a bad mode" at all.
#     This matters for the PPO design, not just the write-up: x0 = alpha*tanh(z) with
#     alpha = 1 and clip_actions = 1 caps ||x0|| at 0.7616*18.71 = 14.25, and at
#     initialisation (mu = 0, sigma ~ 0.3) it sits at ~5.6. If success tracks ||x0||, the
#     steering policy STARTS in the bad regime and cannot reach the good one.
#     Scales: 0.30 = steering at init, 0.76 = steering's saturation ceiling, 1.50 = past the
#     shell in the other direction (a control -- if only small norms hurt, 1.50 stays good).
for K in 0.30 0.76 1.50; do
  cell "full x0=seed1 x$K + 2% action noise" "$RUN/x0norm_act002_s1_k${K/./}.json" \
    --fixed-x0 1 --x0-scale "$K" --action-noise 0.02
done

echo "=== x0 family done  $(date -Is) ==="
