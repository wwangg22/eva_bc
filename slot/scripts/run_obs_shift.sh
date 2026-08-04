#!/usr/bin/env bash
# EXP_STEER §8d -- WHY does 2 % action noise cost 83 points when 5 % sensor noise costs 4.2?
#
# The action is an ABSOLUTE joint-position target (JointPositionActionCfg, scale 0.5,
# use_default_offset=True), so per-step noise does not integrate into a position random walk --
# that was the first explanation and it is ruled out by the config. Hypothesis (belief 9): the
# drive filters target jitter out of POSITION and differentiates it into VELOCITY, so obs[8:16]
# = joint_vel_rel goes far out of distribution while everything else stays put. The policy is
# then being shown an observation unlike anything in its training set, and its nearest learned
# behaviour is the freeze at x = 0.166 m that EXP_ROBUSTNESS §13b measured.
#
# This is a dose-response design, not a single pair. Five magnitudes spanning free to fatal
# (EXP_ROBUSTNESS has 0.02 -> 83 pts lost and 0.05 -> 0/96, so the cliff is below 0.02 and
# unmapped), each carrying eval_act.py's per-channel obs_dist diagnostic. If the mechanism is
# right, success and joint_vel |z| move together across all five -- a correlation one pair
# cannot show.
#
# `shift_obs005` is the CONTROL, and it is the sharpest cell here: obs_dist accumulates on the
# SIMULATOR's observation, not on the noised copy fed to the policy, so a sensing perturbation
# that the policy shrugs off should leave the recorded distribution near nominal. If instead
# obs005 also shows a joint_vel blow-up, the hypothesis is wrong and the shift is incidental.
#
# Protocol: champion, -v0, s777, 128 ep / 32 envs, stochastic x0 (the headline configuration).
# Idempotent. ~20 min.
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

cell "clean"                  "$RUN/shift_clean.json"
cell "action noise 0.005"     "$RUN/shift_act0005.json" --action-noise 0.005
cell "action noise 0.01"      "$RUN/shift_act001.json"  --action-noise 0.01
cell "action noise 0.02"      "$RUN/shift_act002.json"  --action-noise 0.02
cell "action noise 0.05"      "$RUN/shift_act005.json"  --action-noise 0.05
cell "obs noise 0.05 CONTROL" "$RUN/shift_obs005.json"  --obs-noise 0.05

# THE CAUSAL CELL, added 17:59 after the constant-x0 probe. §8d as originally written can only
# show that some channel goes out of distribution under action noise -- correlation. This cell
# makes it a causal test: same 2 % perturbation, same physics, but the latent that scores 0.823
# instead of 0.146. If joint_vel is just as far out of distribution here as in shift_act002,
# then the shift is a CONSEQUENCE of the noise and not the reason the policy freezes -- belief 9
# would survive as a measurement and die as an explanation. If instead the good latent keeps
# joint_vel near nominal, the two findings are one finding.
cell "action noise 0.02, x0=seed4" "$RUN/shift_act002_s4.json" \
  --action-noise 0.02 --fixed-x0 4

echo "=== obs-shift dose-response done  $(date -Is) ==="
