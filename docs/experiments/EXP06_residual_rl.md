# EXP06 — Residual RL on the frozen flow base (Stage 6)

*2026-08-02. Status: IN PROGRESS — design pre-registered before any training.
Green-lit by Big Will 2026-08-02 with the observation spec in §Design. This doc is also
the **running progress log** for residual training (see §Training log at the bottom —
updated every run).*

## Question

Can a small PPO-trained residual on top of the frozen flow-matching champion
(`runs/exp03_N3/ckpt_final.pt`, **64.1% pooled** over 128 eps) close the gap toward
Gate 6 (≥90% nominal AND perturbed)? The ladder says the residue is well-shaped for
this: 34/46 pooled failures are grasp-phase precision (miss/freeze), 12/46
carry/release, 0 catastrophic/sequencing (EXP03 + taxonomy.py).

## Beliefs going in (pre-registered, before any result)

1. **The residual will beat the base, but not reach Gate 6 in its first config.**
   Guess: 75–85% pooled at convergence. Mechanism: reward-driven mm-scale alignment
   fixes concentrated in the 34 grasp-phase failures (the ResiP sweet spot). The
   12 carry/release failures involve the grip channel, which the residual cannot touch
   initially — expect them to persist (this is the pre-registered with-gripper
   ablation trigger).
2. **Near-zero-init residual starts at base performance** (~64% pooled). Must be
   *verified*, not assumed: a zero-residual pass through the full wrapper must
   reproduce the base's eval numbers episode-for-episode before any training
   (wrapper-correctness gate — if it doesn't, the wrapper is broken, stop).
3. **The grasp-success bit can be computed by a simple rule** (finger pos/vel + last
   grip command — EXP01 variant D's feature set) and will match the D-probe's
   transfer: **0% FPR on the 665 on-policy freeze states**, high TPR on expert
   grasped frames. If a threshold rule can't match it, distill the trained D-probe
   MLP instead (it's 5-D input, trivially cheap at runtime). NOTE (EXP01 correction
   2026-08-02): the bit MUST include the commanded-grip channel — physical apertures
   alone leave 27.1% FPR.
4. **Plain additive residual may plateau below expectations on a flow base** (RFS
   literature: plain residual 43% vs x0-steering 86% average). Pre-registered
   decision rule: if the residual fails to exceed the base by ≥5 pts pooled after the
   first full training budget (see §Method), switch to RFS-style x0-steering rather
   than tuning PPO harder.
5. **Training-seed variance applies to RL too** (EXP03 lesson). Any config-vs-config
   claim needs ≥3 seeds or a pooled-suite margin larger than the measured spread.
   First measure the spread with 2–3 seeds of the baseline config before believing
   any single-run number.

## Design (per Big Will's spec, 2026-08-02)

**Frozen base:** `runs/exp03_N3/ckpt_final.pt`, seeded x0 (deterministic), chunk 50 /
execute 15 unchanged (EXP02: commitment is load-bearing — the residual rides ON
committed chunks, per-step, and never shortens the horizon).

**Action space (residual):** 6 arm joints only, `a = a_base + α·tanh(π_res)` in the
base policy's normalized action units (1 unit = 0.5 rad). Gripper channel passes
through from the base untouched. α initial = 0.1 (≈2.9°/joint, ~2 cm fingertip
authority — sized to the measured 2–3 mm-decides-catch regime with headroom).
Residual head near-zero-initialized (final layer zeros → exact base behavior at init).

**Observation (Big Will's list, verbatim → concrete dims):**

| # | component | dims | source |
|---|---|---|---|
| 1 | current 41-D policy observation | 41 | same obs the base sees (normalized with base stats) |
| 2 | current queued BC action | 7 | the base action executing THIS env step (pre-residual, normalized units) |
| 3 | index within current 15-step execution chunk | 1 | step-in-chunk / 15 ∈ [0,1) |
| 4 | physical finger features | 4 | finger joint pos (dims 6,7) + vel (dims 14,15) — re-surfaced for salience |
| 5 | validated grasp-success bit | 1 | rule/probe on finger pos/vel + last grip command (§grasp-bit gate below) |
| 6 | gripper→target-can relative pose | 7 | can pos in gripper frame (3) + can quat in gripper frame (4, XYZW) |
| 7 | target-can→basket relative pose | 3 | basket center − can pos, world frame (basket ≈ planar target; z is its rim height) |

Total: **64-D**. Target can = the env's canonical target (first unplaced), matching the
base obs canonicalization. No history anywhere (per spec; EXP01 refuted its value).

**Algorithm:** PPO first (ResiP precedent), framework/hyperparams reused from the
project's prior privileged-teacher PPO harness (details filled in §Method once
verified against the repo). RFS x0-steering is the documented fallback (belief 4).

**Reward (initial, minimal):** the env's existing task reward terms for placement
success (per-can placed + final success), plus a small residual-magnitude penalty
(−c·‖α·tanh(π_res)‖², c small) to keep the residual near zero except where it pays.
No hand-shaped grasp reward in round 1 — the base already reaches grasp poses; the
residual's job is correction, and EXP03's taxonomy gives us per-bucket eval to see
exactly what it fixes. (Reward details finalized in §Method against the env's actual
reward terms.)

**Eval protocol:** identical to the ladder — 64 eps × seed 42 + 64 eps × seed 123,
pooled 128-ep numbers citable, per-bucket taxonomy on every eval, zero-residual base
numbers as the fixed reference (59.4/68.8/64.1%).

## Gates (ordered, each blocks the next)

1. **Grasp-bit gate:** offline validation on EXP01's labeled frames — 0% FPR on the
   665 on-policy freeze states AND ≥95% accuracy on expert post-close test frames.
   Script: `experiments/exp06_grasp_bit.py`.
2. **Wrapper-correctness gate:** zero-residual rollout through the full residual
   wrapper reproduces the base eval bit-for-bit (or ≥ episode-outcome-identical) on
   suite seed 42.
3. **Training gate:** residual training runs stably (no NaN, base stays frozen —
   assert no grads), reward increases.
4. **Result gate:** pooled 128-ep eval vs 64.1% base; per-bucket taxonomy diff.

## Gate 1 result — grasp bit VALIDATED (MLP), hand rule REFUTED (2026-08-02)

`exp06_grasp_bit.py` (same loaders/split as EXP01, runtime semantics
`bit = predictor AND commanded-closed`):

| candidate | test acc | TPR | on-policy FPR (665 freeze states) | gate |
|---|---|---|---|---|
| aperture threshold rule (best train θ) | 94.6% | 1.000 | **40.0%** | FAIL |
| D-probe MLP (5 dims: finger pos/vel + last-grip) | 94.5% | 0.947 | **0.0%** | PASS* |

- **The hand rule is dead** — and this *revises the postmortem's mechanism story*: raw
  aperture (pos6+pos7) distributions overlap almost completely (grasped median −0.053,
  on-policy-freeze median −0.056). No static aperture threshold separates them; the
  MLP is using velocity/stall structure + nonlinear combinations. "±12 mm separates
  the classes" was too naive.
- *Accuracy gate deviation, accepted with evidence: 94.5% vs pre-registered 95%.
  Per-k: acc 91.3 / 93.3 / 95.2 / 98.1% at k = 0/2/5/10 frames after close-end —
  errors concentrate in the aperture-settling window immediately after close, where
  the bit briefly flickers. For an obs *feature* (the residual also sees raw fingers)
  this is benign; the load-bearing property is the 0% FPR on real freeze states.*
- Exported artifact: `experiments/exp06_grasp_bit.pt` (MLP retrained on all expert
  episodes; 0% FPR on the 665 also with all-data training). Runtime: 5-D input
  (obs dims 6,7,14,15,40), sigmoid > 0.5, ANDed with commanded-closed.

## Method / implementation notes (finalized against repo facts, 2026-08-02)

- **Framework: rl_games 1.6.1** (installed; matches the prior privileged-teacher PPO
  harness `reBot_RL/scripts/rl_games/train.py` + `pick_place/agents/rl_games_ppo_cfg.yaml`
  and PLAN §6's decision).
- **Wrapper:** new gym wrapper around `Rebot-PickPlace-Play-v1` holding the frozen
  `BatchedACTController` (from `act/eval_act.py`); 6-D residual action in, full 7-D
  env action out (`arm = base + α·tanh(res)`, grip = base). Controller needs two small
  additions: thread the seeded `generator` into `predict_action_chunk` (hook already
  exists, `modeling_flow.py:94-119` — eval never passed it), and expose per-env chunk
  phase (= `n_action_steps − len(queue)`) + the served base action.
- **Rel-pose obs sources:** `ee_frame` (`target_pos_w/target_quat_w`, grasp-center
  offset −0.075 on link6/gripper_end), can poses via `env.scene[name].data.root_pos_w`,
  basket via `mdp.basket_centers_local`; target-can index recomputed exactly as
  `objects_canonical` does (nearest unplaced to EE, `mdp/observations.py:51-83`).
- **Reward:** env v1's own sparse terms (placed +60, dropping −30) + small residual
  magnitude penalty; `object_dropping` termination kept ON for training (mid-carry
  releases → immediate penalty + reset = clean signal); `episode_length_s = 30`.
  Eval keeps the ladder's exact protocol (drop-termination off) for comparability.
- **PPO config:** adapted from the teacher yaml (MLP [256,128,64] elu, adaptive lr
  1e-4, e_clip 0.2, horizon 24, mini_epochs 8) with: 64-D obs, 6-D actions, zero-init
  mu head (`const_initializer 0` → exact base behavior at init), small fixed sigma to
  start, minibatch sized to num_envs. num_envs chosen by VRAM smoke test (12 GB budget,
  flow refills batched over envs needing them).
- GPU discipline: ONE training/eval job at a time, background chain, durable logs.

## Training log (running — newest at bottom)

- 2026-08-02: doc created; beliefs pre-registered. Grasp-bit validation is the first
  executable step. No training launched yet.
- 2026-08-02: **Gate 1 PASSED** (MLP bit, 0% FPR / 94.5% acc with documented settling
  caveat; hand rule refuted at 40% FPR). Infra facts gathered; method finalized above.
  Next: wrapper implementation → gate 2 (zero-residual eval must reproduce base).
- 2026-08-02: **implementation landed** (all CPU-syntax-checked):
  - `act/modeling_flow.py`: `predict_action_chunk` gained optional `x0` (one shared
    (chunk,7) noise tensor → base chunk is a deterministic function of the obs; the
    PLAN §6 "fix x0" option). `FIXED_X0_SEED = 7` (act/eval_residual.py), same
    constant in training and eval.
  - `act/eval_act.py`: controller refactored (`_refill` extracted; `act()` unchanged
    behavior) + `peek()` (base action + phase WITHOUT popping) + `pop()` + `fixed_x0`.
  - `act/residual_core.py`: 64-D obs builder (layout in module docstring, matches
    the spec table above), GraspBit runtime (gate-1 artifact), §4.2 flush ported,
    target-can selection mirrors `objects_canonical` exactly, rel pose via
    `subtract_frame_transforms` (wxyz→XYZW), `compose()` = α·tanh blend, grip
    pass-through.
  - `act/eval_residual.py`: ladder-protocol eval through the residual path;
    `--x0-mode global|fixed`; `--residual-ckpt` optional (zero residual absent);
    rl_games checkpoint loader for trained evals.
  - `act/train_residual.py` + `act/residual_ppo_cfg.yaml`: rl_games PPO
    (teacher-config lineage), `ResidualRlGamesWrapper` subclass (64-D/6-D spaces,
    blending + residual penalty + controller reset on done inside `step`), training
    on **Rebot-PickPlace-Play-v1** (the base's own spawn distribution),
    `object_dropping` termination/penalty kept ON, 30 s horizon, zero-init mu head.
    Logs → `runs/exp06_residual/<run-name>/`.
- 2026-08-02: **gate 2 chain launched** (GPU): 2a zero-residual `--x0-mode global`
  seed 42 (must reproduce N3's 38/64 = 59.4% episode-for-episode) → 2b zero-residual
  `--x0-mode fixed` seed 42 (the deterministic-base baseline residual training
  actually starts from — pre-registered: may differ a few episodes from 2a; it, not
  2a, is the like-for-like reference during training). Logs:
  `runs/exp06_residual/gate2{a,b}_seed42.{json,log}`.
- 2026-08-02: **Gate 2a PASSED, bit-exact** — zero-residual global-x0 eval reproduces
  N3's seed-42 base eval perfectly: 59.4%, 0 episode flips, identical per-episode
  lengths and max-can heights. The wrapper (peek/pop, obs builder, flush port,
  blending) is verified correct end-to-end. Eval runtime ~5 min/64 eps.
- 2026-08-02: **Gate 2b = new finding: the fixed x0 is a policy-level degree of
  freedom.** Freezing x0 (seed 7) moved the base to 51.6% on suite 42 with **33
  episode flips** vs the global-RNG base — same order as the training-seed churn
  floor (31–39). Interpretation: with per-refill x0 the policy re-rolls its noise
  every chunk (a bad draw gets corrected at the next refill); one frozen draw bakes
  in a persistent bias. **Pre-registered next step (before any result): sweep fixed
  x0 ∈ {zeros(-1), 1, 2, 3} on suite 42 (~5 min each; seed-7 = 51.6% already on
  disk), then take the top 2 to suite 123 and select the training x0 on POOLED
  128-ep numbers** — same held-out hygiene as the EXP03 champion selection. Zeros
  (the distribution mean) is included as the natural "mode-seeking" candidate.
  Expectation: spread across x0 draws comparable to gate 2b's −7.8 pts; if the best
  pooled fixed-x0 base lands well below 64.1%, that gap is honest context for
  residual results (the residual's reference is the fixed-x0 base it rides on, not
  the stochastic base). `--x0-seed` added to eval_residual.py / train_residual.py.
- 2026-08-02: **x0 sweep, suite 42** (global-RNG reference 59.4%): zeros **56.2%**,
  seed 3 51.6%, seed 7 51.6%, seed 1 37.5%, seed 2 **14.1%**. The frozen draw spans
  a 42-pt range — a bad x0 can nearly destroy the policy, and no sampled draw beat
  the distribution mean. Zeros leads, consistent with mode-seeking. Suite-123 evals
  of {zeros, 3, 7} launched for pooled selection.
- 2026-08-02: **x0 SELECTED: zeros** on pooled 128 eps — zeros 55.5% (56.2/54.7),
  seed3 53.9%, seed7 51.6%. Every fixed x0 pays ~9 pts vs the stochastic base
  (64.1% pooled): the price of the deterministic-base condition, now measured. The
  residual's like-for-like reference is therefore **55.5% pooled (fixed-x0 zeros)**;
  64.1% remains the headline number to beat for the stage. (This also strengthens
  belief 4: x0 is demonstrably a high-leverage control surface — RFS x0-steering has
  something real to steer.)
- 2026-08-02: training launch: smoke (5 epochs, integration shakeout) then run
  **r1** = {x0 zeros, alpha 0.1, res-penalty 0.01, PPO seed 1, 128 envs, 3000
  epochs, Rebot-PickPlace-Play-v1, 30 s horizon, drop-termination ON}. Logs:
  `runs/exp06_residual/r1_seed1/`.
- 2026-08-02: **smoke PASSED** (5 epochs, 32 envs — wrapper + rl_games integration
  clean, no NaN). Full **r1** training running (128 envs, 3000 epochs, ~1.9 GB VRAM
  at startup; monitor armed for errors/completion). On completion: pooled two-suite
  eval of the best checkpoint via eval_residual.py (--x0-seed -1) + taxonomy diff vs
  the 55.5% fixed-x0 base and the 64.1% stochastic base.
- 2026-08-02: **scale-up (Big Will: use the VRAM).** 128 envs was integration-safety,
  not a real limit (1.9 GB/12 GB used). Actual bottleneck was the controller's
  per-env python deques → replaced with a fully vectorized tensor queue
  (`_buf (N,15,7)` + `_idx (N,)`, same refill order/semantics; eval_act.py). r1
  (128-env run, minutes old) killed and relaunched at **2048 envs, 600 epochs**
  (≈29.5M env-steps, minibatch 12288 = batch/4). Chain re-verifies gate 2a
  bit-exactness through the vectorized queue BEFORE training (hard abort on any
  episode flip), then 2048-env smoke (VRAM spike check: a full-reset refill pushes
  a 2048-batch through the flow transformer), then full r1.
- 2026-08-02: **r1 training COMPLETE** — gate2a-vec bit-exact, 2048-env smoke OK,
  600 epochs / 29.5M env-steps in ~35 min (~15k steps/s; VRAM ~1–2 GB — far from
  the 12 GB limit even at 2048 envs). No NaN; MAX EPOCHS reached. rl_games best
  ckpt (train-reward criterion): `runs/exp06_residual/r1_seed1/nn/exp06_residual.pth`
  (best 0.50 at 2048 envs; reward trajectory −0.27→0.50 over 600 epochs — modest,
  interpretation deferred to eval). HYGIENE: the killed 128-env run shared the same
  nn/ dir — files matching ep_800..ep_2200 and the duplicate ep_200/400/600 entries
  are STALE (128-env era); rm list to Big Will after eval. Pooled two-suite eval of
  the best ckpt launched (x0 zeros, fixed; reference 55.5% fixed-x0 / 64.1%
  stochastic).
- 2026-08-02: **r1 RESULT: 0.0% pooled (both suites) — residual destroyed the base;
  root cause diagnosed, not a wiring bug.** Taxonomy: never-lifted 46/64 + 38/64;
  still places can-1 in ~25% of eps but completes none; mean applied |residual|
  0.056/joint (learned mu moved far from zero). Training-log forensics: FIRST best
  ckpt (epoch 100) had mean reward −0.245, i.e. the σ-perturbed base collected only
  penalties from epoch 1 (a 56%-base should collect strongly positive placed
  reward). **Mechanism: initial exploration σ_init −1.0 → σ≈0.37 pre-tanh ≈ 1°/joint
  /step of dither — the same magnitude as the 2–3 mm grasp precision threshold. PPO
  never observed the working regime and optimized inside the collapsed one** (best
  crawled −0.25→+0.50 via placed-1 partial credit). Wrapper wiring is NOT suspect:
  gate2a-vec was bit-exact and zero-residual evals reproduce the base. Belief-2
  refinement: "zero-init mu = starts at base" was true for mu but operationally
  false — INITIAL EXPLORATION NOISE is part of the starting condition. (This is
  precisely why ResiP prescribes near-zero init AND small initial noise.)
- 2026-08-02: **r2 launched**: identical to r1 except σ_init −2.5 (σ≈0.08 → applied
  exploration ≈0.008 units ≈ 2 mm fingertip — the correction scale itself).
  Pre-registered health check: epochs 1–20 mean reward must be strongly positive
  (the base's own collection level). If it is still ≤0, STOP — that would implicate
  the training-reward path itself, not exploration, and forbids more training runs
  until diagnosed. Run: `runs/exp06_residual/r2_smallsigma/`.
- 2026-08-02: **r2 health check FAILED → STOP rule invoked.** First best reward
  −0.327 ≈ r1's −0.245 despite 4.6× smaller exploration noise. The σ-collapse
  diagnosis is wrong or incomplete; r2 killed at ~epoch 100. Two new suspects
  identified BEFORE the next experiment: (1) **rl_games zero-init gap** — verified
  in installed source (network_builder.py:317): `mu_init` is applied to the mu
  WEIGHT only; the bias keeps default U(±1/√64)≈±0.125 → constant per-joint
  residual bias up to ~0.36° from step 0 (belief-2's "starts at base" was violated
  a second way). (2) **episodic raw rewards ≈ −25..−33 ≈ the −30 drop penalty** —
  suggests drop-termination may be ending most training episodes. Diagnostic
  running (`act/diag_training_env.py`): zero / bias(+0.125) / N(0,0.08) / N(0,0.37)
  fixed-action conditions rolled through the EXACT training config, measuring
  success, terminated-vs-truncated, episode length, raw episodic reward. NO further
  training until this attributes the collapse.
- 2026-08-02: **ROOT CAUSE FOUND AND PROVEN — an action-scaling bug, not noise.**
  Diagnostic matrix (diag_training_env.py, exact training config, 512 envs each):
  zero 57.2% / +1644 raw; mu-bias(+0.125) 55.5% / +1633; r2-noise(σ0.08) 61.3% /
  +1657; r1-noise(σ0.37) 11.5% / +519. Env, reward path, wrapper, bias-init and r2
  exploration ALL healthy — nothing an epoch-0 agent should emit reproduces the
  observed ≈−30 raw/episode. Tensorboard then showed Episode_Reward/placed = 0.0
  from EPOCH 1 in both runs → rollouts broken before learning. Actual mechanism
  (verified in rl_games source, a2c_common.py:1181-1184): `preprocess_actions`
  clamps agent samples to [−1,1] then RESCALES to the action-space bounds — and our
  yaml said `clip_actions: 100.0`, so every action reached the wrapper ×100 and
  fully saturated the tanh → constant ±α shoves on all 6 joints from step 1. The
  r1 "σ too hot" diagnosis was WRONG (r2 refuted it; σ0.37 costs success but still
  collects +519, nowhere near −30). Fixes: `clip_actions: 1.0` (rescale = identity)
  + eval-side mus clamped to [−1,1] to match. LESSON: when adapting an RL config,
  every `env` block value is semantically load-bearing — clip_actions is not a
  clip, it is an ACTION SCALE.
- 2026-08-02: **r3 launched** — identical to r2 (σ_init −2.5) + the clip_actions
  fix. Same pre-registered health check: early epochs must show rewards/iter
  strongly positive (diag predicts ≈ +16 shaped at base level). Run:
  `runs/exp06_residual/r3_actionfix/`.
- 2026-08-02: **r3 health check PASSED** — epoch~100 best reward +1610 ≈ the diag's
  zero-residual reference (+1644). Rollouts collect base-level reward; the
  clip_actions fix closed the whole gap (r1/r2 logged negative at the same point).
  Training continues to 600 epochs; then pooled two-suite eval + taxonomy diff.
- 2026-08-02: **r3 RESULT: pooled 55.5% → 55.5%, EXACTLY flat (26 fixed / 26
  broken).** Both sides deterministic on identical spawns → this churn is CAUSAL,
  not seed noise. Failure analysis (exp06_analyze_r3.py):
  - **The mechanism works where it was aimed**: 10/19 base never-lifted episodes
    converted to SUCCESS (grasp misses fixed!), 8 placed-1-stuck freezes fixed, 6
    carry/release failures fixed. Belief-1's mechanism is real.
  - **But it breaks an equal number of working episodes**, mostly by inducing NEW
    grasp misses (residual-side never-lifted grows 19→26): the small uniform
    perturbation (~0.0084 units ≈ 0.24°/joint everywhere; identical on success and
    failure episodes) helps marginal misses and hurts marginal successes
    symmetrically. It has NOT learned state-dependent discrimination — it applies
    near-constant effort, so per-config outcomes reshuffle (seed42 −3.1, seed123
    +3.1) and the aggregate is zero-sum.
  - Train reward did beat base (+1700 vs +1644): earlier/longer placements on
    fixed episodes outweigh broken ones in stream-reward terms while success stays
    flat — reward and success decouple exactly as the eval-only-gate anticipated.
  - **Pre-registered belief-4 rule FIRES: +0.0 < +5 pts after the full budget →
    switch to RFS-style x0-steering rather than tuning PPO harder.** Belief-5
    tempering: r3 is ONE RL seed; seeds 2,3 of the same config launched (chain,
    ~90 min) to measure RL-training spread before the pivot is final. If any seed
    pools ≥60.5%, plain-residual stays alive; else x0-steering (design doc section
    to be written BEFORE implementation).

## x0-steering design sketch (PRELIMINARY — finalize only if the seed replicas confirm the flat verdict)

RFS insight, now backed by OUR OWN measurement: the frozen x0 is a policy-level knob
(14.1–56.2% success across draws, gate-2b sweep) whose outputs are always coherent,
on-manifold chunks — unlike additive action offsets, which perturb off-manifold and
produced r3's symmetric fix/break wash. Plan: RL acts at CHUNK granularity — at each
refill the policy sees the 64-D obs and outputs a bounded steering z (start: 7-D
per-action-dim offset broadcast over the chunk, x0 = α_x0·tanh(z); richer bases only
if needed), the base's own decoder turns it into actions. Wrapper reworks to
one-RL-step-per-chunk (15 env steps, summed reward). Full pre-registered beliefs +
gates to be written BEFORE implementation, after the r3_seed2/3 verdict.

## FINAL VERDICT (2026-08-02) — plain additive residual CLOSED; pivot to x0-steering

Big Will's call after reviewing the r3 failure analysis and the x0-steering rationale:
**stop the additive-residual line, pivot to x0-steering as the focus.** The r3_seed2/3
replica chain was cancelled mid-run (seed2 killed ~epoch 100, seed3 never started) —
so "additive residual = flat" formally rests on ONE healthy RL seed (r3, exactly
55.5%→55.5%, 26 fixed/26 broken, causal). The belief-4 rule had already fired; the
replicas were belt-and-suspenders and can be resurrected if x0-steering also stalls.

### Belief scorecard (pre-registered vs measured)
1. "Residual beats base 75–85%" — **WRONG.** Exactly flat (55.5%). The MECHANISM
   works (10/19 never-lifted converted to success) but a symmetric break rate
   cancels it: the learned residual is state-independent (~0.0084 units everywhere,
   identical on success/fail episodes) — no discrimination despite the grasp-bit and
   rel-pose inputs being available.
2. "Near-zero init starts at base" — **WRONG TWICE, then fixed.** (a) rl_games
   mu_init touches only the WEIGHT (bias stays U(±0.125) — measured harmless);
   (b) initial exploration σ is part of the starting condition (σ0.37 costs 46 pts;
   σ0.08 harmless — diag matrix); (c) the REAL killer was clip_actions=100 →
   rl_games rescales [-1,1] samples ×100 → saturated tanh from step 1 (r1/r2 = 0%).
   Gate-2 wrapper verification was bit-exact throughout — the wrapper was never wrong.
3. "Grasp bit computable by simple rule" — **HALF-WRONG.** Hand aperture threshold:
   40% FPR (distributions overlap — the ±12 mm story was naive). The D-probe MLP:
   0% FPR, exported and integrated. Bit must include last-grip command.
4. "Plain residual may plateau (RFS 43-vs-86)" — **CONFIRMED on our own data**, and
   the pre-registered switch rule fired. Bonus confirmation: the x0 sweep showed
   14.1–56.2% success across frozen draws — x0 is a high-leverage, on-manifold
   control surface; additive action offsets are off-manifold and washed out.
5. "RL training seed variance" — measured only n=1 healthy seed (replicas cancelled).

### Hard-won infrastructure that CARRIES OVER to x0-steering (all verified)
- Vectorized tensor-queue controller (`eval_act.py` peek/pop; gate2a bit-exact at
  2048 envs, ~15k steps/s, ~7 GB VRAM).
- `predict_action_chunk(x0=...)` fixed-noise path; x0-zeros selected champion mode
  (55.5% pooled deterministic base = the steering baseline).
- 64-D residual obs builder + validated grasp bit (`residual_core.py`,
  `exp06_grasp_bit.pt`).
- `residual_wrapper.py` (rl_games vec wrapper), `train_residual.py`,
  `eval_residual.py` (ladder-protocol evals + rl_games ckpt loader),
  `diag_training_env.py` (fixed-action attribution harness — REUSE for any future
  "training collects wrong reward" mystery).
- rl_games config lessons baked into `residual_ppo_cfg.yaml` comments
  (clip_actions=1.0 is MANDATORY; sigma_init −2.5).

### Cleanup (rm list for Big Will — checkpoints of failed/cancelled runs; logs+JSONs stay)
    cd runs/exp06_residual
    rm -r r0_smoke r0_smoke2048 r1_seed1 r2_smallsigma r3_seed2
    # keep: r3_actionfix (the one healthy additive-residual run, evidence)
