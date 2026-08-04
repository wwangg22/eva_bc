#!/usr/bin/env bash
# EXP_STEER §11 -- HELD-OUT VALIDATION of the winning constant x0. This is not optional.
#
# The probe scored 5-9 candidate latents on the SAME 96 episodes (spawn seed 777) and then
# reported the maximum. That is selection on the evaluation set: with a spread of 0.031 to 0.771
# the winner is partly a winner because those particular spawns suited it. Quoting 0.771 as
# "what a good latent achieves" without re-measuring on unseen spawns is exactly the error this
# project retracted twice in session 6 (EXP_ROBUSTNESS §9d, EXP_DEPTH §8).
#
# So: take the argmax from seed 777 and re-run it at TWO fresh spawn seeds. The pre-registered
# reading is that a real effect keeps most of its margin (>0.55 at both) while a selection
# artefact regresses toward the population mean of the probed latents.
#
# The zeros and stochastic references are re-measured at the same fresh seeds, because the
# comparison has to move together -- a drop at seed 888 means nothing if 888 is simply harder.
#
# Usage:  bash scripts/run_x0_holdout.sh <BEST_SEED> [RUN]
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

BEST="${1:?usage: run_x0_holdout.sh <BEST_SEED> [RUN] -- pass the winning --fixed-x0 value}"
RUN="${2:-runs/bc_armB_seed0}"

cell() {  # name, out, extra args...
  local name="$1" out="$2"; shift 2
  [ -f "$out" ] && { echo "SKIP $out"; return 0; }
  echo "=== $name  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$RUN/ckpt_final.pt" --task Rebot-PrecisionSlot-v0 \
    --num-envs 32 --episodes 128 "$@" --out "$out" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $out"
}

# Characterise the winner at the ORIGINAL spawn seed first: committing to a good latent is only
# a deployable fix if it costs nothing when there is no noise. `zeros` already shows commitment
# is not free (0.938 clean vs stochastic 0.979), so this is a real question, not a formality.
cell "s777: x0=$BEST CLEAN" "$RUN/x0probe_clean_s$BEST.json" --seed 777 --fixed-x0 "$BEST"
cell "s777: x0=$BEST + 5% noise" "$RUN/x0probe_act005_s$BEST.json" \
  --seed 777 --fixed-x0 "$BEST" --action-noise 0.05

for SP in 888 999; do
  cell "holdout s$SP: x0=$BEST + 2% noise" "$RUN/holdout_act002_x0${BEST}_s$SP.json" \
    --seed "$SP" --fixed-x0 "$BEST" --action-noise 0.02
  cell "holdout s$SP: x0=zeros + 2% noise" "$RUN/holdout_act002_zeros_s$SP.json" \
    --seed "$SP" --fixed-x0 zeros --action-noise 0.02
  cell "holdout s$SP: stochastic + 2% noise" "$RUN/holdout_act002_stoch_s$SP.json" \
    --seed "$SP" --action-noise 0.02
done

echo "=== x0 holdout done  $(date -Is) ==="
