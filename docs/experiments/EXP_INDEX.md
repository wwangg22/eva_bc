# Experiment Ladder — pre-residual-RL causal ablations

*Started 2026-08-01. Directive (Big Will): before more DAgger or committing to residual RL,
run focused experiments answering (a) can current observations identify grasp success,
(b) does execute-15 cause precision failures, (c) does recovery training reproducibly
damage nominal grip behavior. Document everything: what we did, what we found, what we
believe, and validation when data looks odd.*

Companion docs: POSTMORTEM.md (why we're here), PLAN.md (grand plan), JOURNAL.md (running log).
Baseline under test everywhere: `runs/flow_nominal_v1/ckpt_final.pt` (59.4%, 64 ep, seed 42,
deterministic; per-episode diag in `runs/flow_nominal_v1/eval_gate2_64ep_diag.json`).

## The eight points → experiments

| # | Directive point | Experiment | Status |
|---|---|---|---|
| 1 | Aliasing hypothesis: classifier predicting true grasp success from obs; single-frame vs 2–4-frame history; pseudo-tactile bit | [EXP01](EXP01_grasp_aliasing_probe.md) | **DONE — info present single-frame; finger-only probe perfect on policy miss states; history NOT the fix; grasp-bit justified** |
| 2 | Ablate open-loop horizon without retraining: n_action_steps ∈ {15, 8, 4, 2, 1}; then phase-dependent execution | [EXP02](EXP02_execution_horizon.md) | **DONE — inverted: 59.4/32.8/3.1/0/0% at n=15/8/4/2/1; commitment is load-bearing; keep n=15** |
| 3 | Is DAgger interference real? ≥3 nominal + ≥3 nominal+DAgger replicas; success, miss/freeze, mid-carry release, grip divergence | [EXP03](EXP03_dagger_interference.md) | **DONE — interference REFUTED; seed variance 32.8–59.4% within nominal arm; v1↔v3 churn = noise floor; v2 verdict void; NEW CHAMPION exp03_N3 @ 64.1% pooled (held-out selected)** |
| 4 | If DAgger useful: recovery loss weights (0.1/0.25/0.5) and/or frozen-base recovery adapter | EXP04 (doc on start) | **descoped** — its premise (interference) refuted by EXP03; revisit only if a future ≥3-seed A/B shows recovery data harming |
| 5 | Improve recovery teacher: executed-state grasp verification, lying-can/joint-limit fixes, independent RNG streams | EXP05 (doc on start) | still relevant if DAgger r2 runs (teacher ceiling 68% unchanged), no longer gated on EXP04 |
| 6 | Frozen-base residual RL: arm-joints-only residual, inputs = state/history + base action + chunk phase + grasp signal + relative poses, strong bounds | EXP06 (doc on start) | after 1–5 |
| 7 | ResiP-style PPO residual baseline first; RFS only after; compare off-policy residual (ResFiT) if sample efficiency matters | folded into EXP06 design | after 1–5 |
| 8 | Per-phase instrumentation (approach alignment, grasp, post-close decision, lift, transport, basket alignment, release, placement) | cross-cutting; taxonomy scripts in this dir, per-step diag eval planned with EXP03 | ongoing |

## Decision rules (pre-registered)

- **EXP01:** single-frame probe AUC ≥ ~0.95 → the information IS in the obs; the policy's
  failure is *using* it, and an explicit grasp-success input bit is a cheap, justified fix.
  History ≫ single-frame → add 2–4-frame history to the policy. Aperture-only probe ≈ full
  probe → the signal lives in the finger channels (pseudo-tactile idea validated).
- **EXP02:** any n < 15 gaining ≥ +5 pts (≥ 3 episodes beyond the 0-churn floor) → adopt;
  examine taxonomy shift (grasp fixes vs new failure modes) before phase-dependent stage.
- **EXP03:** interference is "real" if the nominal-replica and dagger-replica metric
  distributions separate beyond replica spread (3 v 3, report ranges; no p-theater at n=3).
  Reproduces → EXP04. Doesn't reproduce → v3's 17-broken was seed noise; re-read POSTMORTEM
  §5 claims against replica grip-divergence data.

## GPU discipline

One Isaac/CUDA training-or-eval job at a time, always sequential, background with durable
logs. CPU-side work (EXP01 probe, analysis scripts, doc writing) runs concurrently.

## Log

- 2026-08-01: ladder started. EXP02 eval chain launched (n = 8, 4, 2, 1; n = 15 baseline
  already on disk). EXP01 labeling/probe implementation begun. `train_flow.py` had NO RNG
  seeding — `--seed` flag added for EXP03 replicas.
- 2026-08-02: EXP01 done (grasp-bit justified; history refuted). EXP02 done (horizon
  collapse 59.4→0%; commitment load-bearing). EXP03 forensics retracted POSTMORTEM §5;
  replicas refuted interference and exposed 26.6-pt seed variance; held-out selection
  crowned **exp03_N3 (64.1% pooled)** as frozen base. EXP04 descoped. Next: EXP06
  residual-RL design doc, pending Big Will's review of the ladder.
- 2026-08-02 (evening): **EXP06 CLOSED — additive residual exactly flat** (55.5→55.5
  pooled, 26 fixed/26 broken symmetric churn, state-independent residual; two 0%
  runs traced to rl_games clip_actions=100 bug — clip_actions must be 1.0).
  **EXP07 (x0-steering, RFS-style) CLOSED — SUCCESS: 55.5% → 91.4% pooled
  (89.1/93.8), Gate 6 (90%) cleared, first pre-registered config, single seed
  decisive (+35.9 pts).** 51 fixed / 5 broken; never-lifted bucket collapsed
  (18/19); learned z state-dependent. Docs: EXP06_residual_rl.md,
  EXP07_x0_steering.md (verdict + belief scorecard). Champion = frozen exp03_N3
  base + `runs/exp07_steer/s1_seed1/nn/exp07_steer.pth` steering head.
