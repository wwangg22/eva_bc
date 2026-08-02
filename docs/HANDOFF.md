# HANDOFF — 2026-08-02 ~12:30 PDT (post-EXP06-additive-residual, pre-compaction)

For the next session. Read this + `experiments/EXP06_residual_rl.md` (the full residual
arc incl. FINAL VERDICT + belief scorecard) + `experiments/EXP_INDEX.md` + POSTMORTEM.md
(carries dated CORRECTION/UPDATE blocks — parts of the original are retracted).
Env: `source /home/william/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6`.
ONE GPU job at a time (`nvidia-smi` + `ps aux | grep "[t]rain_\|[e]val_"` — NOT `pgrep -f`).
Big Will reviews all rendered output — never view images/videos yourself. `rm` is
classifier-blocked — hand Big Will exact commands. Address him as Big Will. Short Bash
timeouts; long jobs = background chains with durable logs + Monitor on the log.

**RUNNING AT WRITE TIME:** nothing. GPU free (15 MiB). No monitors armed.
**NEXT ACTION (Big Will's explicit directive): implement x0-STEERING (§4).** The plain
additive residual line is CLOSED by his decision after the r3 flat verdict.

---

## 1. WHAT FINISHED (this session, 2026-08-02 morning→noon)

### a. POSTMORTEM/doc corrections (from Big Will's critique — he was RIGHT)
- **EXP01 D/G conflation fixed everywhere** (POSTMORTEM ×3, HANDOFF, memory, EXP01 doc):
  variant **D** (finger pos/vel + **last-grip**, 5 dims) owns AUC 0.968 + **0% FPR** on
  the 665 on-policy freeze states; variant **G** (4 physical joints) owns AUC 0.976 but
  **27.1% FPR**. Any grasp bit MUST include the commanded-grip channel.
- **N3 failure accounting reconciled** (taxonomy.py on both suites): 64.1% pooled = 46
  failures over 128 eps, exhaustively = **34 grasp-phase miss/freeze** (18 never-lifted
  + 16 placed-1-then-closed-on-air) + **12 carry/release** (9 lifted-never-placed + 3
  stuck-at-carry) + 0 drops-after-place. 74% of residue is grasp-alignment shaped.
- POSTMORTEM fully reconciled with the ladder: TL;DR update block, §4a/§4b updates,
  §6 table revisions (open-loop row now "backwards-protective"), §7 items 5–8, §8 plan
  update block.

### b. EXP06 plain additive residual — BUILT, DEBUGGED, MEASURED, CLOSED
Full narrative + belief scorecard in `experiments/EXP06_residual_rl.md`. Headlines:
- **Gate 1 (grasp bit):** hand aperture rule REFUTED (40% FPR — raw aperture
  distributions of hold-vs-air overlap almost entirely); D-probe MLP exported to
  `experiments/exp06_grasp_bit.pt` — **0% FPR / 94.5% acc** (settling-window caveat
  documented). Runtime: 5 dims (6,7,14,15,40), sigmoid>0.5 AND commanded-closed.
- **Gate 2a (wrapper correctness): bit-exact** vs the base eval (0 flips, identical
  lengths/heights), re-verified after the queue vectorization. The wrapper was never
  the problem at any point.
- **NEW FINDING — frozen x0 is a policy-level knob:** fixed-x0 sweep on the frozen N3
  base spans **14.1%–56.2%** success across draws (seed2 14.1, seed1 37.5, seeds 3/7
  51.6, zeros 56.2 on suite-42). **x0=zeros selected on pooled 128 eps: 55.5%**
  (56.2/54.7) — the deterministic-base baseline. Determinism costs ~9 pts vs the
  stochastic base (64.1% pooled).
- **r1/r2 = 0% collapse, root cause PROVEN (3-step chain):** (i) diag matrix
  (`act/diag_training_env.py`): zero/bias/σ0.08 all healthy (~+1650 raw, 55–61%
  success), σ0.37 degrades (11.5%) but still +519 — nothing an epoch-0 agent emits
  reproduces the observed ≈−30; (ii) tensorboard: Episode_Reward/placed = 0.0 from
  EPOCH 1 → broken before learning; (iii) rl_games source (a2c_common.py:1181-84):
  `preprocess_actions` clamps to [−1,1] then **RESCALES to action-space bounds** — our
  `clip_actions: 100.0` multiplied every action ×100 → saturated tanh → constant ±α
  shoves. **Fix: clip_actions=1.0 (mandatory, comment in yaml) + eval-side mu clamp.**
  Also fixed en route: rl_games mu_init touches only the WEIGHT (bias stays ±0.125 —
  measured harmless); σ_init −2.5 (σ0.37 would cost ~46 pts per diag).
- **r3 (healthy run): pooled EXACTLY flat 55.5%→55.5%, 26 fixed / 26 broken — CAUSAL**
  (both sides deterministic on identical spawns). Mechanism works where aimed (10/19
  never-lifted → SUCCESS, 8 stuck + 6 carry fixed) but breaks equal numbers by inducing
  NEW grasp misses; learned residual is state-INDEPENDENT (~0.0084 units everywhere,
  identical success-vs-fail) — no discrimination despite grasp-bit/rel-pose inputs.
  Train reward beat base (+1700 vs +1644) while success stayed flat — never trust
  train reward. Analysis tool: `experiments/exp06_analyze_r3.py`.
- Belief-4 rule fired (+0.0 < +5 pts) → x0-steering. r3_seed2/3 replicas CANCELLED
  mid-run by Big Will's pivot decision (additive-flat formally rests on 1 healthy seed;
  resurrect replicas only if x0-steering also stalls).

### c. Infrastructure built & verified (ALL carries over to x0-steering)
- `act/eval_act.py`: controller VECTORIZED (tensor queue `_buf (N,15,7)` + `_idx`;
  peek/pop/act; `fixed_x0` param) — bit-exact vs deques, 2048 envs at ~10–15k steps/s,
  ~7 GB VRAM, GPU 100%.
- `act/modeling_flow.py`: `predict_action_chunk(..., x0=)` — (chunk,7) tensor
  broadcast to batch. **x0-steering needs a per-env (B,chunk,7) variant — small edit.**
- `act/residual_core.py`: 64-D obs builder (layout in docstring), GraspBit runtime,
  §4.2 flush ported, target-can selection mirrors `objects_canonical` exactly,
  `compose()` α·tanh blend.
- `act/residual_wrapper.py` (rl_games vec wrapper — split from train script so diags
  can import it), `act/train_residual.py` (2048 envs, minibatch=batch/4, run_config
  dump), `act/eval_residual.py` (ladder-protocol eval, `--x0-mode/--x0-seed`,
  rl_games ckpt loader with training-matched mu clamp), `act/residual_ppo_cfg.yaml`
  (teacher lineage + hard-won comments), `act/diag_training_env.py` (fixed-action
  attribution harness — reuse for any "training reward looks wrong" mystery).
- Eval runtime ~5 min/64 eps. Training 600 epochs/29.5M steps ≈ 35 min at 2048 envs.

## 2. WHAT WE LEARNED (new this session; older lessons in EXP06 doc + POSTMORTEM stand)

1. **rl_games `clip_actions` is an ACTION SCALE, not a clip** (rescale to bounds after
   [−1,1] clamp). Every `env:` block value in an adapted RL config is semantically
   load-bearing. Cost us two full training runs.
2. **"Zero-init residual" needs THREE things**: zero mu weight (mu_init), awareness
   that the mu BIAS stays random (harmless here, but check), and SMALL initial σ —
   exploration noise is part of the starting condition (σ 0.37 ≈ 1°/joint/step costs
   46 pts on a mm-precision base).
3. **Fixed-action attribution matrices beat theorizing**: the diag harness pinpointed
   in one 10-min run what two training-log post-hocs misdiagnosed (my σ story was
   WRONG for r1 — refuted by r2's identical failure at small σ; STOP rules work).
4. **x0 of a flow policy is a high-leverage on-manifold control surface** (14→56%
   across draws; zeros/mode best) — the empirical foundation for x0-steering. Fixing
   x0 for determinism costs ~9 pts vs per-refill draws (re-rolling noise each chunk
   self-corrects bad draws).
5. **Additive action residuals wash out symmetrically on this base** (26/26) —
   off-manifold nudges help marginal misses and hurt marginal successes; PPO learned
   effort, not discrimination. Train-reward gains (+3.4%) can coexist with flat
   success — the placement-stream reward pays for earlier/longer placements.
6. Per-env python containers (deques) are the real scaling limit before VRAM —
   vectorize controller state; verify rewrites via bit-exactness gates (gate2a
   pattern: cheap, decisive).

## 3. CURRENT STATE

- **Frozen base:** `runs/exp03_N3/ckpt_final.pt`. Stochastic 64.1% pooled;
  **deterministic x0=zeros mode 55.5% pooled** (56.2 @42 / 54.7 @123) = the steering
  baseline. Failure anatomy in §1a.
- Task #23 (Stage 6) in_progress = x0-steering now. #21/#22 open at 64.1%. #19 parked
  (77.4% perturbed pillar). The §4-fork "grasp-bit BC retrain" path from the previous
  HANDOFF is PARKED (Big Will chose residual RL, then x0-steering) — still valid if
  steering stalls.
- **Cleanup pending Big Will** (all eval JSONs/logs preserved):
  - old ladder ckpts (712 MB): `rm runs/exp03_N1/ckpt_*.pt runs/exp03_N2/ckpt_*.pt runs/exp03_D2/ckpt_*.pt runs/exp03_D3/ckpt_*.pt`
  - failed/cancelled residual runs: `cd runs/exp06_residual && rm -r r0_smoke r0_smoke2048 r1_seed1 r2_smallsigma r3_seed2` (keep `r3_actionfix` as evidence)
- Close-up failure video still unrendered (render only on request). DAgger r2 staged,
  unrun. All parked items unchanged from previous handoff.

## 4. THE PLAN: x0-STEERING (Big Will's directive — "focus on this")

**Convention: write the pre-registered design + beliefs into
`experiments/EXP07_x0_steering.md` BEFORE coding.** Design worked out so far (sketch
at the end of EXP06 doc; refine then implement):

1. **Mechanism:** RL policy acts at CHUNK granularity. At each refill the policy sees
   the obs and outputs bounded z; the base integrates from x0 = α_x0·tanh(z) instead
   of zeros. Output is always an on-manifold chunk — the base's own decoder does the
   arbitration; RL picks the mode. Gripper leverage comes THROUGH the base (x0 has a
   grip column) without giving RL the raw channel.
2. **Parameterization v1:** z ∈ R^7 (per action dim), broadcast across chunk
   positions; α_x0 = 1.0 (≈1σ of the trained noise distribution). Richer bases
   (constant+ramp, 14-D; or per-position low-rank) ONLY if v1 is flat — pre-register
   the escalation rule.
3. **Wrapper rework (the main implementation task):** synchronous chunk windows —
   ALL envs refill together every 15 env steps; one RL step = one window (summed
   reward). Mid-window flush or reset → that env re-predicts using its CURRENT z
   (steering persists within the window). Obs at refill: the 64-D builder MINUS the
   queued-base-action (doesn't exist pre-refill) and chunk-index (always 0) → 56-D.
   Keep grasp bit, fingers, rel poses, 41-D obs.
4. **Plumbing edits:** `predict_action_chunk` x0 accepts (B,chunk,7); controller
   refill accepts per-env x0 override; new chunk-window wrapper (reuse
   ResidualRlGamesWrapper skeleton); PPO yaml: horizon 24 windows (=360 env steps),
   episodes = 100 windows, same lr/clip; **clip_actions: 1.0** (lesson #1);
   σ_init: exploration in z-space is SAFE by construction (on-manifold) — can start
   larger, e.g. σ≈0.3; pre-register the choice.
5. **Gates (ordered):** (i) z=0 through the chunk wrapper reproduces the x0-zeros
   base 55.5% pooled episode-for-episode (the gate2a pattern — synchronous windows
   change refill timing vs the free-running controller, so expect episode-outcome
   match, demand ≥ near-identical; investigate any gap before training);
   (ii) early-health: epoch-~100 reward ≈ +1650 raw (diag reference); (iii) result:
   pooled 128-ep vs 55.5% baseline, +5-pt rule at 60.5%, stochastic-base 64.1% as
   headline, Gate 6 at 90%; taxonomy diff mandatory (does never-lifted collapse
   WITHOUT symmetric breakage? that's the discrimination signature additive lacked).
6. **If steering also washes out:** resurrect (a) r3_seed2/3 replicas (additive,
   n=3 verdict), (b) the grasp-bit BC retrain fork (previous HANDOFF §4 step 1),
   (c) hybrid steering+small-additive (RFS's full recipe). Decision then goes to
   Big Will with all three costed.

## 5. Key files map
- **Residual arc:** `experiments/EXP06_residual_rl.md` (READ FIRST — full arc,
  scorecard, carries-over list), `act/{residual_core,residual_wrapper,train_residual,
  eval_residual,diag_training_env}.py`, `act/residual_ppo_cfg.yaml`,
  `experiments/{exp06_grasp_bit.py,exp06_analyze_r3.py}`, `runs/exp06_residual/`
  (gate2*, x0sweep_*, r3_best_* JSONs).
- Ladder: `experiments/EXP_INDEX.md`, EXP01/02/03 docs, `LITERATURE.md` (RFS
  2602.01789 = the steering precedent, 43%-vs-86%), `taxonomy.py`.
- Policy: `act/modeling_flow.py` (x0 param), `act/eval_act.py` (vectorized
  controller, peek/pop, fixed_x0), `act/train_flow.py` (--seed), `act/dataset.py`
  (41-D layout).
- Expert/data: unchanged (`expert/demos_nominal_s10{1..8}.h5`, `dagger_r1.h5`,
  `run_expert_v1.py` FROZEN).
- Checkpoints: `runs/exp03_N3/` (**CHAMPION BASE**), `runs/exp06_residual/r3_actionfix/`
  (additive-residual evidence), plus historical (`exp03_D1`, `flow_nominal_v1`,
  `flow_dagger_v3`).
