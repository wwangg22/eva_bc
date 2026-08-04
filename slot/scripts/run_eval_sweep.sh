#!/usr/bin/env bash
# Stage C evaluation sweep (docs/slot/EXP_BC_ARMS.md).
#
# Every arm/seed is evaluated with IDENTICAL --num-envs and --episodes, because the headline
# success rate carries a first-episode bias that moves with --num-envs (measured +18.7 points
# on a learned policy, +12.9 on the expert). Comparisons use success_rate_later.
#
# TWO spawn seeds, not one. Re-running an eval at the SAME seed is bit-reproducible on this
# task (measured: 0.438/0.562/0.375 twice over), so a repeat yields no information and an error
# bar built from repeats would be exactly zero. Spawn variance can only be seen by varying the
# seed. Neither 777 nor 888 is among the collection seeds (0-7, 10-13, 20-23).
#
# Usage:  bash scripts/run_eval_sweep.sh [ckpt_name]
#         ckpt_name defaults to ckpt_final.pt; pass e.g. ckpt_0050000.pt for a learning curve.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

CKPT_NAME="${1:-ckpt_final.pt}"
TAG="${CKPT_NAME%.pt}"

for ARM in A B; do
  for SEED in 0 1 2; do
    RUN="runs/bc_arm${ARM}_seed${SEED}"
    [ -f "$RUN/$CKPT_NAME" ] || { echo "SKIP $RUN/$CKPT_NAME (absent)"; continue; }
    # The difficulty ladder is Loose (3.0 mm) -> v0 (1.5 mm) -> Tight (0.5 mm). Tight is in the
    # deliverable and was not in the pre-registered arm comparison; it is evaluated anyway
    # because summarize_arms.py handles any task set and the marginal cost is one more pass.
    # Rebot-PrecisionSlot-Play-v0 is deliberately NOT here: it differs from -v0 only in
    # scene.num_envs and env_spacing, both of which parse_env_cfg overrides, and in
    # enable_corruption, which the base cfg already sets False. It would be a duplicate of -v0.
    for TASK in Rebot-PrecisionSlot-Loose-v0 Rebot-PrecisionSlot-v0 Rebot-PrecisionSlot-Tight-v0; do
      for SPAWN in 777 888; do
        OUT="$RUN/eval_${TAG}_${TASK}_s${SPAWN}.json"
        [ -f "$OUT" ] && { echo "SKIP $OUT (exists)"; continue; }
        echo "=== $RUN $CKPT_NAME $TASK spawn=$SPAWN  $(date -Is) ==="
        # `|| echo ...` is load-bearing, not defensive noise. Under `set -euo pipefail` a
        # failed eval makes grep match nothing, grep exits 1, pipefail propagates and `set -e`
        # kills the WHOLE sweep -- one bad cell would silently discard the 30+ that follow it
        # during an unattended 2.5 h run. The sweep is idempotent (it skips existing outputs),
        # so continuing past a failure and re-running later is strictly better than aborting.
        # The marker string is greppable so a failure cannot be mistaken for a skip.
        python slot_act/eval_act.py --ckpt "$RUN/$CKPT_NAME" --task "$TASK" \
          --num-envs 32 --episodes 128 --seed "$SPAWN" --out "$OUT" 2>&1 | grep "eval_act" \
          || echo "!!! EVAL FAILED (continuing): $OUT"
      done
    done
  done
done
echo "=== eval sweep ($CKPT_NAME) done  $(date -Is) ==="
