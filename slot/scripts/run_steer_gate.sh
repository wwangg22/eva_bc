#!/usr/bin/env bash
# EXP_STEER §8 -- the arm-C gate, plus the constant-x0 probe that decides whether PPO runs at all.
#
# Two things happen here and they are separable:
#
#   THE GATE (C0/C1/C2). eval_steer.py's noise path must be the same perturbation eval_act.py's
#   is, or no steered number is comparable to the 83-point deficit it is supposed to close. The
#   reference is x0-ZEROS under noise, NOT the 0.146 stochastic cell -- see EXP_STEER 8a for the
#   mistake that correction fixes. With fixed_x0 = zeros neither harness draws a randn for x0,
#   both draw one randn_like(action) per env step, and both step the env the same number of times
#   in the same order, so C1 and C2 should agree EPISODE-FOR-EPISODE, not just in rate.
#
#   THE PROBE (5 constant draws under noise). Belief 2's mechanism is mode collapse at staging.
#   If that is right, some x0 selects the push. A constant x0 is the weakest possible steerer --
#   it cannot condition on state -- so this is a LOWER bound on trained steering and a direct
#   test of the mechanism, at ~1 % of a PPO run's cost. Spread < 5 pts kills belief 2 and the
#   PPO arms do not launch. See EXP_STEER 8b for the full decision rule.
#
# Protocol is the standard robustness cell: champion, -v0, s777, 128 ep / 32 envs, later cohort.
# Idempotent -- existing outputs are skipped, so a killed run resumes by re-invoking.
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/rei/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab6

RUN="${1:-runs/bc_armB_seed0}"
CKPT="$RUN/ckpt_final.pt"
COMMON=(--task Rebot-PrecisionSlot-v0 --num-envs 32 --episodes 128 --seed 777)

run_act() {  # name, out, extra args...
  local name="$1" out="$2"; shift 2
  [ -f "$out" ] && { echo "SKIP $out"; return 0; }
  echo "=== $name  $(date -Is) ==="
  python slot_act/eval_act.py --ckpt "$CKPT" "${COMMON[@]}" "$@" --out "$out" 2>&1 \
    | grep -E "eval_act" || echo "!!! FAILED (continuing): $out"
}

# --- gate ---------------------------------------------------------------------------------
# C0: is `zeros` even a good draw absent noise? Without this, a low C1 is unattributable --
# it could be the noise or it could be that zeros is simply a bad latent.
run_act "C0 x0=zeros CLEAN" "$RUN/eval_x0zeros_Rebot-PrecisionSlot-v0_s777.json" --fixed-x0 zeros
# C1: the gate REFERENCE (and the `zeros` cell of the probe -- same run, both roles).
run_act "C1 x0=zeros + 2% action noise" "$RUN/x0probe_act002_zeros.json" \
  --fixed-x0 zeros --action-noise 0.02
# C2: the gate SUBJECT -- same condition through the steering code path, z held at 0.
OUT="$RUN/steergate_steerzero_act002.json"
if [ -f "$OUT" ]; then echo "SKIP $OUT"; else
  echo "=== C2 zero-z steering + 2% action noise  $(date -Is) ==="
  python slot_act/eval_steer.py --ckpt "$CKPT" "${COMMON[@]}" --action-noise 0.02 --out "$OUT" 2>&1 \
    | grep -E "eval_steer" || echo "!!! FAILED (continuing): $OUT"
fi

# --- probe: four more constant draws ------------------------------------------------------
for S in 1 2 3 4; do
  run_act "probe x0=seed$S + 2% action noise" "$RUN/x0probe_act002_s$S.json" \
    --fixed-x0 "$S" --action-noise 0.02
done

echo "=== steer gate + x0 probe done  $(date -Is) ==="
