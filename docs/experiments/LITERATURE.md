# Literature notes for the experiment ladder

*Collected 2026-08-01 by research agent; exact arXiv IDs verified, no substitutions.
Feeds EXP02 (PACE), EXP04 (CR-DAgger/IWR/Sirius), EXP06 (ResiP/ResFiT/RFS).*

## PACE — arXiv:2606.00537 (2026) → EXP02 stage 2

Training-free, test-time execution-horizon selection for any chunked policy: compute the
predicted chunk's **speed profile**, cut execution at low-speed minima (phase boundaries:
approach→contact etc.), replan there. Long horizons in coherent free-space motion, short
near contact. RoboTwin2.0: 57.8→64.2%; real ALOHA/Franka success 50.7→70.4%. No
retraining, no policy internals. **Maps to us:** replan right before the grasp — where our
misses happen. Related: DEHP (arXiv:2606.11408, learned horizon head), arXiv:2602.21445.

## ResiP — arXiv:2407.16677 → EXP06

Frozen chunked BC base + per-timestep additive residual trained with PPO from sparse
reward. Residual obs = [state, base action for that step]. **Residual includes the gripper
channel.** Kept small via near-zero final-layer init + tuned exploration noise (exact clip
in Appendix J-C, not fetched). Gains: one_leg 54→98%, round_table 12→94%, lamp 7→97%.
**Maps to us:** closest template; near-zero residual init is the base-preservation
mechanism; run a with/without-gripper residual ablation given our interference data.

## CR-DAgger — arXiv:2506.16685 (NeurIPS 2025) → EXP04

Frozen base + tiny (~2 MB) additive residual head trained on human *delta-action*
corrections collected DURING execution (no takeover → data stays on-policy).
**Base-preservation recipe: explicit zero-residual supervision on all non-correction
frames** + 4× denser sampling in a short window after each intervention onset. Single
batch training, no iterative rounds. <50 correction eps: book flip 40→100%, +64% avg.
**Maps to us:** the direct alternative to retraining the base on mixed pools (our v3
regression). Zero-target-on-nominal is the anti-interference mechanism.

## IWR — arXiv:2012.06733, and Sirius — arXiv:2211.08416 → EXP04 weights

- IWR: sample intervention vs non-intervention data **50/50 per batch** (≈3× effective
  upweight at their ratios); no per-dataset tuning needed. Threading 87.3% vs HG-DAgger
  75.3%. Non-intervention data kept as regularization.
- Sirius: intervention samples resampled to **50% of every batch — performance peaks
  there and DEGRADES if pushed higher or lower**; **pre-intervention frames (~15 steps
  before takeover) weighted to ZERO** (they are the robot's own failing actions).
**Maps to us:** we already zero-mask ALL policy steps (train_mask[:takeover_t]=0 —
stricter than Sirius). Our v3 trained at the pool's natural ratio (~23% recovery); the
50%-peak result suggests the EXP04 sweep should test {natural, 0.5} and DOWN-weights
{0.1, 0.25} — the two literatures pull in opposite directions, so measure.

## ResFiT — arXiv:2509.19301 → EXP06 alternative

Same frozen-base additive per-step residual, but **off-policy** (DDPG-variant + Q
ensemble): ~200× more sample-efficient than PPO residual (200k vs 40M steps, sim).
Real-robot: 14→64% in 134 rollouts. Gripper included in residual. **Maps to us:** in
cheap parallel sim, PPO is fine; if rollouts get expensive (or hardware), go off-policy.

## RFS — arXiv:2602.01789 → EXP06, CRITICAL for flow bases

On a **flow-matching** base, jointly learn (a) steering of the flow's initial noise x0
(global: selects which mode the flow decodes) and (b) an additive residual (local
refinement). Sim averages: **RFS 0.861 vs plain additive residual 0.433** vs latent-only
0.483. Pick-and-place: 0.939 vs ~0.50 plain. **Maps to us:** our base IS flow-matching
with a seedable x0; when the base commits to a wrong mode (bad grasp), a bounded local
residual can't fix it — x0 steering can. Plan: ResiP-style baseline first (per Big Will's
point 7), but expect its ceiling; RFS is the documented fix and our seeded-x0
infrastructure already supports it.
