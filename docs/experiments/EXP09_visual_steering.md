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
