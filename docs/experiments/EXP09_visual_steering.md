# EXP09 — x0-steering with visual data (DESIGN, pre-registration in progress)

Status: DESIGN ONLY (written 2026-08-04 while EXP08 v5 trained). Big Will's
directive: "I really want to get the steering working with visual DATA."
Nothing here is built yet; smoke tests S1–S3 must run and be logged before any
training code, per the pre-registration convention.

## 1. Goal

Port EXP07's x0-steering (55.5→91.4 on the state base) to the vision student:
a steering head that consumes ONLY student-visible signals (images / features
derived from images + 23-D proprio) and modulates the frozen vision flow
base's chunk generation via x0 = tanh(z). Original Gate E guess: +4–10 pts on
top of wherever DAgger tops out.

## 2. Hard constraints (all discovered/confirmed in EXP08 — read its log first)

- **C1 — no privileged info at deploy.** EXP07's steering obs56 contains can
  poses relative to EE, basket delta, placed mask = privileged. A vision
  steering head must reconstruct or replace these from pixels+proprio.
- **C2 — DLSS frame perturbation.** Extra GPU work inside a vision driving
  loop corrupts frame content: every-step matmuls → student 80→36%;
  boundary-only LIVE feature builds → 46.9% (EXP08 bisection). Consequences:
  - Naive PPO (grad updates interleaved with env steps) trains on frames that
    don't match deployment.
  - Even a separate small steering forward between steps changes the loop's
    GPU-work signature in sim. Fix: **integrate steering into the student's
    single forward pass** so collection/eval/deploy loops do IDENTICAL
    per-step work. (Real rig: hardware cameras, constraint vanishes.)
- **C3 — vision env is slow** (67–813 env-steps/s vs thousands for the state
  env). PPO sample budgets that were cheap in EXP07 are expensive here.

## 3. Design space

### 3a. Steering observation (what replaces obs56)

- **Option A (RECOMMENDED): perception head.** Small supervised head (on the
  frozen vision encoder's features + proprio) predicting exactly the
  privileged components of obs56: rel_pos(3), rel_quat_xyzw(4),
  basket_delta(3), grasp bit(1) [+ optionally placed count]. Then the EXP07
  steering machinery runs UNCHANGED on [proprio-slices ⊕ predicted features].
  - Training data ALREADY EXISTS: every DAgger-v2 shard stores boundary
    images + raw ground truth (obs41, ee pose, can poses, placed, centers);
    `obs56_from_raw` gives exact targets. ~600+ eps ≈ 15–40k boundary samples.
    Post-hoc supervised training = C2-safe by construction.
  - Gives a measurable intermediate: per-feature prediction error (S3) tells
    us if pixels at 160×90 carry enough signal BEFORE we spend on RL.
- **Option B: latent steering.** Steering head eats the frozen vision
  transformer's pooled encoder latent directly; trained end-to-end by RL.
  More general, but opaque (no error metric), and forces route R2 (below).
  Fallback if S3 shows the perception targets are unpredictable.

### 3b. Training route for the steering head (the C2/C3 problem)

- **Route R1 (RECOMMENDED): PPO in the FAST state env + sensor-noise model.**
  Train steering with EXP07's exact PPO setup (camera-free env, no rendering,
  no C2, thousands of steps/s) but corrupt the ground-truth features fed to
  the steering head with the perception head's MEASURED error distribution
  (from S3: per-feature, per-phase — approach/grasp/carry — quantile noise +
  occlusion-style dropout). Deploy: perception head output → steering head.
  The noise injection is the sim2sim bridge; mismatch risk is correlated
  errors (occlusions), mitigated by phase-conditioned noise and by S3's
  worst-case analysis.
- **Route R2: PPO in the vision env, update-phase bursts.** Standard PPO
  already alternates rollout (inference-only, C2-safe like our eval loop) and
  update phases; the open question is whether a gradient BURST between
  horizon chunks corrupts the frames that follow (S2 measures exactly this).
  Gated on S2; even if safe, C3 makes this ~10–50× slower than R1.
- **Route R3 (fallback, weakest): offline advantage-weighted regression** on
  recorded rollouts (no live env). Safest, likely smallest gain.

### 3c. The transfer question nobody should skip

EXP07 steering works because the STATE base's flow maps x0 regions to
distinct action modes. The VISION base was distilled from already-steered
champion outputs — its x0-response is UNKNOWN and may be weaker (the
distillation data had no mode diversity left to encode). S1 measures this
before anything else is built. If the vision base barely responds to x0,
steering as-designed is dead on arrival and the alternative is (a) retrain
the vision base with an x0-diversity term, or (b) residual-action RL instead
of x0-steering — both bigger conversations with Big Will.

## 4. Pre-registered smoke tests (run + log BEFORE building)

- **S1 — x0 responsiveness of the vision base.** On ~200 recorded boundary
  states: sweep x0 over a grid (per-dim ±2), measure chunk spread
  (max pairwise EE-space divergence over the executed 15 steps) for v_best
  vision base vs the EXP03 state base on matched states. PASS: vision-base
  spread ≥ ~50% of state-base spread. CPU/GPU offline, C2-safe.
- **S2 — burst perturbation.** Drive the fixed student (eval loop) but insert
  a 1–2 s dummy-matmul burst every 64 steps (PPO-update signature). Compare
  driving success vs clean baseline, same seed. PASS for R2: within ~3 pts.
- **S3 — perception head error.** Train Option-A head on existing shards
  (train: r1v2+r2+r3 dirs; held out: r4 + cold dirs). Report per-feature RMSE
  + 95th percentile by phase. PASS: rel_pos RMSE ≲ 2 cm near-grasp,
  basket_delta ≲ 3 cm (thresholds to sanity-check with Big Will — these are
  roughly the precision the state policy needed).

## 4.5 Running log

- 2026-08-04: **S1 PASSED decisively** (`experiments/exp09_s1_x0probe.py`,
  `runs/exp09/s1_x0probe.json`): on 305 matched states (Gate B BC shards,
  which store images + obs41 per step), 8 shared x0 draws, executed-15-step
  action-space spread: vision base (v4) 0.345 mean / 0.360 median vs state
  base 0.436 / 0.439 → **ratio median 0.805, p10 0.663** (bar: ≥0.5). The
  distilled vision base retained x0-responsiveness — steering is viable.
- 2026-08-04: **design simplification after reading the obs code.** The
  DAgger-v2 shards do NOT store raw ground truth (only images/proprio/labels)
  — §3a's "targets already exist" claim was wrong for DAgger shards; it's the
  BC shards that carry per-step obs41. Better targets anyway:
  `objects_canonical` (obs41[16:32]) is already target-first object poses IN
  ROBOT-ROOT FRAME + placed flags, and obs41[32:34] is basket_center_xy. So
  the perception head predicts **obs41[16:34] (18-D) directly**, and the
  steering obs becomes the reconstructed obs41 = [proprio(23) rearranged ⊕
  predicted 18-D]. No obs56 rebuild, no EE-frame math, no FK. R1 then trains
  a NEW steering head by PPO in the state env on obs41 with measured noise
  injected on [16:34] — EXP07's recipe, new input. S3 targets: train on BC
  seed42, validate cross-seed on seed123 (caveat: both are training-stream
  seeds; a small fresh-seed collection with obs41 saved is the true held-out
  check later).

- 2026-08-04: **S3 v1 — below the sanity bar; one architecture iteration
  before verdict** (`experiments/exp09_s3_perception.py`,
  `runs/exp09/s3_perception/s3_result.json`; 45.8k train / 47.5k cross-seed
  val, 20k steps): target-pos vec err 5.5 cm mean / 11.8 cm p95 (bar ~2 cm),
  basket 4.8 / 9.8 cm (bar ~3), quats RMSE ~0.38–0.40, placed_flags RMSE 0.27
  (bad for a binary). Read: the info is in the pixels (v4 ACTS on them at
  78.9%) but the v1 head is weak — a global-avg-pooled fresh resnet18 discards
  exactly the spatial precision coordinate regression needs, and MSE on flags
  underfits. **S3v2:** predict from the FROZEN v4 student's own encoder tokens
  (31 spatial tokens, already task-trained) + small head — doubles as the
  end-to-end validation of the integrated-forward deploy design; BCE on
  flags, per-group loss weights. Fallback reading if v2 also misses: R1 can
  still proceed with MEASURED noise injected — PPO-under-noise then tells us
  directly whether steering tolerates ~5 cm sensing error.

- 2026-08-04: **S3v2 (frozen v4 encoder tokens + MLP head,
  `experiments/exp09_s3v2_perception.py`): tgt-pos 3.5 cm mean / 8.8 p95 (v1:
  5.5/11.8), basket 4.4 cm, flags vec_mean 0.063.** Better everywhere, still
  ~1.5–2× the sanity bar. The v4 tokens demonstrably carry metric state — the
  residual gap is head/likelihood-level or DLSS pixel noise, and further
  polishing is NOT the critical path because:
- 2026-08-04: **DESIGN KILL — R1 (state-env PPO → deploy on vision base) is
  broken in principle, caught before building.** The vision base's x0→behavior
  geometry was constructed independently during distillation (rectified-flow
  training samples its own x0 ~ N(0,1) against champion chunks) — it shares NO
  correspondence with the state base's x0 geometry. A steering z learned
  against the state base is meaningless to the vision base. S1's spread ratio
  only shows the vision base RESPONDS to x0, not that the geometries align.
  ⇒ Steering must be trained AGAINST the frozen vision base itself. Revised
  routes: **R2' (preferred): PPO in the vision env, steering head on the
  frozen v4 encoder tokens (integrated forward — zero extra per-step loop
  work; S3v2 proves the tokens carry the state), updates between horizon
  chunks — gated on S2.** R3' (if S2 fails): C2-safe offline loop — collect
  vision-env rollouts with per-chunk RANDOM z (x0 arg replaces the internal
  randn, identical loop compute), post-hoc advantage-weighted regression
  tokens→z, iterate. The perception head (S3v2 ckpt) is shelved for now —
  useful later for diagnostics/real-rig cross-checks, not in the deploy path.

- 2026-08-04: **S2 FAILED — PPO-style bursts corrupt frames.** v4 seed-42 eval
  with a ~0.5 s matmul burst every 64 steps: **70.3%** (7,11,15,12) vs clean
  84.4% (13,14,14,13); −14.1 pts, far outside the ±3 bar; the burst even
  destroys the cold-pool round-1 gain (13→7). Live PPO in the vision env (R2')
  is DEAD. **Route locked: R3' offline AWR.** Collector = eval loop + per-15-
  step-boundary z ~ N(0,1)^7, x0 = tanh(z) expanded, passed via the x0 arg
  (replaces the internal randn — same loop compute, C2-safe); record images/
  proprio/z + per-window placed-delta and reward sum + episode outcome.
  Post-hoc: recompute frozen v4 tokens from recorded images → fit steering
  head by advantage-weighted regression (window placed-delta primary, reward
  shaping secondary); iterate collections with head-mean + exploration noise.
  Eval = deterministic head mean, integrated forward.

- 2026-08-04: **AWR iteration 0 pre-registration.** Collector
  `experiments/exp09_awr_collect.py` (one encode + manual Euler mirroring
  predict_action_chunk — SAME loop compute for random-z and head-driven
  iterations, C2-safe). Iteration 0: z~N(0,1)^7 shared per chunk, fresh seeds
  3001/3002, 64 eps each. **Guess: driving success 65–85%** (the base already
  runs from random per-element x0; chunk-shared tanh(z) is mildly
  out-of-distribution). If BELOW 60: chunk-shared tanh-x0 hurts the base —
  reconsider steering parametrization (e.g. z added to random x0, or
  first-k-steps-only) BEFORE fitting any head. Success signal for AWR later:
  per-window placed-delta windows are rare (~2/ep) — the reward-sum shaping
  carries most gradient; z-collapse watch = head output variance.

- 2026-08-04: **it0 α=1 BELOW floor (0.406/0.391 vs 0.60)** → per
  pre-registration, parametrization fixed before fitting. **α-sweep (32 eps,
  seed 3001): α=0.5 → 0.562, α=0.25 → 0.625** (base band 0.64–0.69).
  **DECISION: α=0.25** — near-transparent to the base, steering authority to
  be proven by AWR (if the fitted head moves nothing at 0.25, retry 0.5).
  it0b collection at α=0.25 (64 eps × seeds 3001/3002) launched
  (`runs/exp09/it0b_chain.sh`, data → `data/exp09_awr/it0b_a025_seed*`).
  α=1 data kept (off-distribution exploration; don't mix naively).
  **NEXT ACT: build `experiments/exp09_awr_train.py`** — SteerHead(tokens 31,
  512 → MLP → 7-D z, must match the collector's import), precompute frozen v4
  tokens from recorded images, advantage = normalized (placed_delta primary +
  win_reward shaping) per window, weights = exp(A/β) clipped, weighted MSE
  tokens→z/α... note z recorded is PRE-tanh; head should regress z where
  advantage-weighted. Then eval: collector path with --head-ckpt
  --explore-std 0 IS the deterministic eval loop (same compute); compare vs
  v4 78.9 pooled / 65.6 held-out on seeds 42/123/555.

- 2026-08-04: **AWR it1 fit + eval pre-registration** (before building the
  trainer). Data: it0b_a025_seed{3001,3002} (128 eps, ~12k windows, α=0.25,
  driving 0.656/0.672). Fit: frozen v4 tokens (31×512) → SteerHead MLP → 7-D
  z; advantage A = 1.0·(placed_after−placed_before) + 0.3·znorm(win_reward);
  weights = exp(A/β) clipped [0.1,10], β = std(A); weighted MSE to recorded
  PRE-tanh z; 20k steps AdamW 3e-4; 5% val split. **Gates:** (a) z-collapse
  watch — head output std across val windows must be > 0.1 (else it learned
  a constant and the eval is meaningless); (b) steered eval (collector
  --head-ckpt --explore-std 0 --x0-scale 0.25, seeds 42/123 64 eps each +
  555) must NOT regress > 3 pts vs v4's 78.9 pooled / 65.6 held-out.
  **Belief (guess): iteration-1 gains are small, −1 to +3 pooled** — one
  round of purely random exploration at near-transparent α rarely finds much;
  the iteration loop (collect around head mean → refit) is where gains
  compound. If it1 regresses > 3: check weight concentration (report
  effective sample size = (Σw)²/Σw²) before blaming the route.

- 2026-08-04: **it1 interim + it2 pre-registration.** Same-harness collector
  baselines (v4, --baseline randn x0, warm-only): s42 0.875 / s123 0.766 /
  s555 0.547 (pooled 42/123 = 0.820; NOTE higher than eval_flow_vision's
  0.789 — different episode accounting, first-ep-per-env discarded; compare
  within-harness only). it1 steered: s42 0.891; s123 28/32 = 0.875 at half;
  555 pending. Head diagnostics: no z-collapse (output std 0.89), ESS
  6687/12672, val_mse 1.77 (head memorizes train z — expected for elite
  selection; generalization judged by driving eval only). **it2 gate
  (pre-registered): launch iteration 2 iff pooled steered ≥ pooled base
  + 2 pts AND 555 steered ≥ 555 base − 3 pts.** it2 recipe: collect 64 eps ×
  fresh seeds 3003/3004 with --head-ckpt it1 --explore-std 0.7 --x0-scale
  0.25 (data → it1x_std07_seed*); refit on it0b + it1x (all α=0.25 data);
  eval seeds 42/123/555 as it1. **Beliefs: it1x driving ≥ it1 steered − 5
  (noise σ=0.7 costs a little); it2 eval ≥ it1 eval (more data near the
  good region); watch ESS — head-centered data narrows z spread, advantage
  weights may concentrate.**

- 2026-08-05: **it1 VERDICT — GATE PASSED, first visual-steering gain is
  real.** Same-harness (collector, warm-only), 64 eps/seed:
  s42 0.875→**0.891** (+1.6), s123 0.766→**0.828** (+6.3), held-out 555
  0.547→**0.609** (+6.3). Pooled 42/123: 0.820→**0.859** (+3.9 ≥ +2 gate;
  555 +6.3 ≥ −3 gate). Beat the pre-registered guess band (−1..+3). The
  held-out improvement matters most: the head generalizes to unseen spawn
  streams — genuine steering, not spawn memorization (contrast EXP08's
  spawn-overfitting trap). One round of purely random z at α=0.25 was enough
  to find signal despite val_mse ≈ z-variance (elite memorization + smooth
  interpolation seems to suffice). **it2 LAUNCHED per pre-registration**
  (collect σ=0.7 around it1 head, seeds 3003/3004 → refit on it0b+it1x →
  eval; `runs/exp09/it2_chain.sh`, ~2 h).

- 2026-08-05: **it2 VERDICT — FAILED, it1 remains champion.** Ladder
  (base → it1 → it2): s42 0.875 → 0.891 → 0.844; s123 0.766 → 0.828 → 0.781;
  held-out 555 0.547 → 0.609 → **0.516**. Pooled 0.820 → 0.859 → 0.813.
  Beliefs missed: it1x σ=0.7 collections drove 0.703/0.625 (below the
  ≥0.81 belief); "more data near the good region" reversed the gain. Head
  diagnostics healthy (ESS 13303/25344, z_std 0.931) → not a degenerate fit.
  Hypotheses: **H1 mixture conflict** (it0b random-z elites vs it1x
  head-centered z pull the weighted regression to a compromise; pooled
  reward-normalization across datasets with different baselines mislabels
  advantages), **H2 σ=0.7 too hot** (exploration data itself degraded).
  **it3a pre-registered (cheapest decisive test, NO new collection): refit
  on it1x_std07_seed{3003,3004} ONLY → eval 42/123/555.** If it3a ≥ it1:
  H1 confirmed → iterate with fit-on-latest-only + σ=0.35. If it3a ≈ it2 or
  worse: H2/data-quality → recollect at σ=0.35 around it1 head before any
  refit. Guess: it3a lands between it2 and it1 (mixture is real but σ=0.7
  data alone is also weaker than it0b's random-z elites).

- 2026-08-05: **it3a VERDICT — landed between it2 and it1 (as guessed):
  s42 0.828 / s123 0.813 / 555 0.563, pooled 0.820 (= baseline).** Reading:
  H1 (mixture) real but partial — dropping it0b recovered held-out vs it2
  (0.516→0.563) and s123 (0.781→0.813); H2 also real — σ=0.7 it1x data alone
  still ≤ it1's it0b-random fit (its 0.703/0.625 driving polluted the elite
  pool). Champion unchanged: **it1** (0.891/0.828/0.609, pooled 0.859).
  **it4 pre-registered (both remedies jointly): collect 64 eps × seeds
  3005/3006 with --head-ckpt it1 --explore-std 0.35 --x0-scale 0.25 →
  fit on that data ONLY → eval 42/123/555.** Beliefs: it4x driving
  0.75–0.85 (σ=0.35 gentle, centered on a good head); it4 eval ≥ it1 if
  the loop can compound at all. **STOP RULE: if it4 pooled < it1's 0.859,
  the simple AWR loop is declared non-compounding — stop iterating, keep
  it1 as champion, bring escalation options to Big Will (per-phase z,
  CEM elite reuse, value-baseline advantage, x0-diversity base retrain).**

## 5. Phases (after smoke tests, subject to change)

P0 = S1–S3 (~half a day, mostly reusing existing shards + eval loop).
P1 = perception head production training + noise-model fit.
P2 = R1 steering PPO in state env + integrated-forward deployment eval in the
     vision env (report 42/123 + held-out 555).
P3 = only if R1 transfer fails and S2 passed: R2 vision-env PPO.

## 6. Relationship to EXP08

EXP08's DAgger ladder continues until it stalls or hits target; EXP09 is the
finisher on top of the best vision base (currently v4 = 78.9 pooled / 65.6
held-out; v5 pending). Steering composes with any base checkpoint — work is
not wasted by further DAgger rounds.
