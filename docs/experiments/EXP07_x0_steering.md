# EXP07 — x0-steering RL on the frozen flow base (Stage 6, take 2)

*2026-08-02. Status: IN PROGRESS — design pre-registered BEFORE any code (project
convention). Successor to EXP06 (plain additive residual, CLOSED exactly flat
55.5%→55.5% with symmetric 26-fixed/26-broken churn). Pivot chosen by Big Will after
the r3 failure analysis. This doc is the running progress log for the steering work.*

## Question

Can chunk-level RL that steers the flow policy's integration noise x0 (RFS-style,
arXiv 2602.01789) beat the frozen base where the additive action residual could not?

## Why steering should work where additive failed (the evidence chain)

1. **x0 is a measured high-leverage control surface on OUR base** (EXP06 gate 2b +
   sweep): frozen x0 draws span **14.1%–56.2%** success; zeros (the mode) is best at
   **55.5% pooled**. Something real to steer.
2. **Steered outputs are always on-manifold**: the base's own decoder turns x0 into a
   coherent chunk. Additive offsets perturb off-manifold — EXP06 r3 showed they help
   marginal misses and break marginal successes symmetrically (26/26, causal).
3. **Chunk-granularity credit assignment**: one decision per 15 env steps (100
   decisions/episode instead of 1500) with the window's summed reward — 15× denser
   reward per decision.
4. **Gripper leverage without the raw channel**: x0 has a grip column; RL influences
   grip timing THROUGH the base's decoder (the additive residual had no grip access —
   its 12 carry/release failures were structurally out of reach).
5. RFS literature: plain residual 43% vs x0-steering 86% average across tasks.

## Beliefs going in (pre-registered, before any result)

1. **z=0 through the chunk-window wrapper reproduces the x0-zeros base bit-exactly**
   (55.5% pooled, episode-for-episode vs `x0sweep_s-1_seed{42,123}.json`). The
   wrapper leaves the controller free-running — z only enters through x0 at refill —
   so unlike a "synchronous truncation" design there is NO behavioral deviation to
   tolerate. If this gate is not bit-exact, the wrapper is wrong; stop.
2. **Exploration in z-space is NOT free** (r1 lesson: initial noise is part of the
   starting condition — but this time we MEASURE it before training instead of
   assuming). Broadcast z ~ N(0,σ) resampled per window will cost some success;
   guess: σ=0.3 costs ≤10 pts vs 55.5% (x0 sensitivity near the mode is locally
   flat — the sweep's bad draws were full-magnitude σ=1 iid draws). The σ_init
   choice is made FROM this measurement (gate S0), not from precedent.
3. **The learned z will be state-DEPENDENT** (the discrimination signature additive
   lacked): per-episode mean |tanh(z)| and across-window z variance will differ
   between success and failure episodes. Mechanism guess: with on-manifold outputs, a
   nonzero z in a bad-grasp state is not automatically punished by breaking good
   states — PPO can afford to specialize.
4. **Result guess: 62–72% pooled** at first-config convergence — beats the +5-pt rule
   (60.5%) and plausibly the stochastic headline (64.1%), below Gate 6 (90%).
   Grounds: steering can at best recover the ~9-pt determinism tax (→64%) plus some
   share of the 34 grasp-phase failures; the RFS 43→86 jump is on different tasks and
   NOT assumed transferable.
5. **Broadcast z (one offset per action dim, constant across chunk positions) may be
   too coarse.** The x0 sweep perturbed iid per (position, dim); a 7-D constant shift
   is a much smaller subspace. Pre-registered escalation rule: if the first full
   budget lands < 60.5% pooled (same +5 rule as EXP06 belief 4), try ONE richer
   parameterization — constant+linear-ramp per dim (14-D) — before any PPO tuning.
   If that is also flat, steering closes and the EXP06 §4.6 fallbacks go to Big Will.
6. **RL-seed variance applies** (EXP03/EXP06 lesson): a single-run margin below ~5
   pts pooled is not evidence. First run 1 seed; replicate ×2 only if the result is
   in the decision-relevant 55–65% band.

## Design

**Frozen base:** `runs/exp03_N3/ckpt_final.pt`, chunk 50 / execute 15 (EXP02:
commitment horizon untouched). External normalizer as always.

**Action (z-space):** z ∈ R^7 per window (6 arm + 1 grip column of x0).
`x0_steer = α_x0 · tanh(z)` broadcast across all 50 chunk positions; **α_x0 = 1.0**
(≈1σ of the noise distribution the base was trained under). rl_games samples are
clamped to [−1,1] (clip_actions 1.0 → rescale is identity — the EXP06 mandatory
lesson), so the effective per-dim range is ±tanh(1) ≈ ±0.76: well inside the trained
noise distribution by design. Zero-init mu head → z≈0 → x0≈zeros = the 55.5% base.
(mu BIAS stays U(±0.125) per rl_games — gate S0 measures a bias-sized z explicitly
instead of assuming EXP06's "harmless" carries over to x0-space.)

**Timing model (the core mechanism):** the controller stays FREE-RUNNING (each env
refills when its queue empties — identical to every ladder eval). The RL clock ticks
in windows of 15 env steps. One RL step:

1. policy sees obs56 (built at the window boundary) → z
2. wrapper stores per-env `x0_steer = tanh(z)`; any refill that occurs during the
   next 15 env steps uses that env's current x0_steer
3. wrapper steps the env 15 times (per-step §4.2 flush check → controller.act →
   env.step), summing reward
4. returns (obs56 at new boundary, summed reward, done, time_outs)

Alignment: training + eval protocol both use 30 s episodes = 1500 steps = exactly
100 windows, and (this experiment's protocol change, see Reward) NO mid-episode
terminations — so every env's refills land on window boundaries and every reset
lands on a window edge. The ONLY desync source is the §4.2 flush (env re-predicts
mid-window with its CURRENT z, then refills off-boundary for the rest of the
episode): z for such an env is applied up to 14 steps stale. Accepted and documented
(flushes are rare — ~1/episode scale); no truncation, no reward-carry bookkeeping.

**Observation (56-D)** — the EXP06 64-D builder MINUS queued-base-action (doesn't
exist pre-refill) and chunk-index (always 0 at boundaries):

| slice | component | dims |
|---|---|---|
| [0:41] | base policy's 41-D obs (raw env units) | 41 |
| [41:45] | physical finger features (pos 6,7 + vel 14,15) | 4 |
| [45:46] | validated grasp bit (EXP06 gate-1 MLP, unchanged) | 1 |
| [46:53] | target-can pose in gripper frame (pos 3 + quat XYZW 4) | 7 |
| [53:56] | target-can → basket delta (env-local) | 3 |

**Reward / training protocol (DELIBERATE change vs EXP06 — pre-registered):**
training env uses the LADDER EVAL protocol: `object_dropping` termination OFF and
`dropping_penalty` OFF (they are inseparable — the penalty is an
`is_terminated_term` on that termination, pick_place_env_cfg.py:347/373). Rationale:
mid-window terminations would break exact window alignment (one drop desyncs the env
for the rest of training). A dropped can is still implicitly punished — zero
placement reward for the rest of the episode. Remaining reward = the env's
`objects_placed` stream (+ nothing else; NO residual-magnitude penalty — z is
bounded by construction and on-manifold, there is no "stay small" prior to encode;
x0=0 is preferred only if it wins reward). Train/eval protocol identity is a bonus:
the first config with zero train/eval mismatch.

**PPO config (`act/steer_ppo_cfg.yaml`, EXP06 lineage):** obs 56 / act 7; zero-init
mu weight; **σ_init pre-registered −1.2 (σ≈0.30 in z-space) PENDING gate S0** — if
S0 shows σ=0.3 costs >10 pts, drop to σ≈0.15 (−1.9); horizon 24 windows (=360 env
steps); minibatch = batch/4; clip_actions 1.0 (MANDATORY); lr 1e-4 adaptive kl 0.01;
gamma 0.99. Budget: 2048 envs, **200 epochs** first pass = 24·2048·200 ≈ 9.8M
window-transitions ≈ 147M env steps (~2.5–3 h at ~15k steps/s). Note this is ~5×
r3's env-step budget but ⅓ of its TRANSITION count — chunk-level RL consumes
windows, not steps. Resume-capable; extend only on a rising curve.

## Gates (ordered, each blocks the next)

- **S0 — z-response diag (before training, ~25 min):** eval fixed-z conditions on
  suite 42 (64 eps each): (a) z=0 → must equal `x0sweep_s-1_seed42` (this doubles as
  gate 1 on suite 42); (b) constant per-dim bias z ~ U(±0.125) (the mu-bias
  condition); (c) z ~ N(0,0.3) resampled per window per env, 2 draws; (d) N(0,0.6),
  1 draw. Decision: σ_init keeps exploration-level success within ~10 pts of 56.2%
  (suite 42 reference). If even σ=0.15 collapses the base, STOP and rethink (that
  would refute the local-flatness belief and predict a hard exploration problem).
- **1 — z=0 bit-exactness, both suites:** episode-for-episode identical to
  `x0sweep_s-1_seed{42,123}.json` (55.5% pooled). Bit-exact expected (free-running
  controller, deterministic forward); ANY flip = wrapper bug, stop.
- **2 — training health:** shaped episodic reward at base level (≈ +16.4 raw/window
  ≈ +1644/episode, the EXP06 diag reference) from the EARLIEST epochs, and
  Episode_Reward/placed nonzero from epoch 1 in tensorboard (the r1/r2 failure was
  visible there from epoch 1 — check it FIRST this time, it is the cheapest look).
  STOP rule: if early reward ≤ 0, no more training until a diag attributes it.
- **3 — result:** pooled 128-ep eval (both suites) of the best checkpoint vs
  **55.5%** (like-for-like base), **60.5%** (+5 rule), **64.1%** (stochastic
  headline), **90%** (Gate 6). MANDATORY taxonomy diff (does never-lifted collapse
  WITHOUT symmetric breakage?) + state-dependence check on z (belief 3): per-episode
  mean |tanh(z)| and per-episode across-window z std, success vs failure episodes.

## Implementation map (all small deltas on EXP06 infra)

- `act/modeling_flow.py`: `predict_action_chunk` x0 accepts (chunk,7) OR (B,chunk,7).
- `act/eval_act.py`: controller gains `steer_x0` (N,chunk,7) attribute; `_refill`
  uses `steer_x0[empty]` when set (takes precedence over `fixed_x0`).
- `act/steer_core.py`: `SteerCore` — obs56 builder + per-step flush/act stepping;
  shares GraspBit + feature code with `residual_core.py` (extracted helper).
- `act/steer_wrapper.py`: `SteerRlGamesWrapper` — 56-D/7-D spaces, window step loop.
- `act/train_steer.py` + `act/steer_ppo_cfg.yaml`: rl_games PPO, logs →
  `runs/exp07_steer/<run>/`.
- `act/eval_steer.py`: ladder-protocol eval through the window path; `--steer-ckpt`
  optional (absent = fixed-z modes for S0/gate-1: `--z-sigma`, `--z-bias`,
  `--z-seed`); records per-episode mean |tanh(z)| + across-window z std.

## Training log (running — newest at bottom)

- 2026-08-02: doc created; beliefs pre-registered. Next: implementation, then gate
  S0 + gate 1 in one GPU chain.
- 2026-08-02: **implementation landed** (per the map above). CPU unit tests
  (scratchpad test_steer_plumbing.py) ALL PASS: (1) controller steer_x0=zeros is
  action-for-action identical to fixed_x0=zeros incl. a mid-run desynced refill;
  (2) per-env steer routing changes ONLY the steered env; (3) set_steer tanh bound +
  (N,chunk,7) broadcast correct; (4) real FlowMatchingPolicy on CPU: (chunk,7) vs
  (B,chunk,7) x0 paths bit-identical, per-env x0 differences isolated to their env.
  Repo fact locked in: dropping_penalty is an is_terminated_term on object_dropping
  (pick_place_env_cfg.py:347/373) — confirms the penalty and termination can only be
  disabled together, as the window-aligned training protocol requires.
- 2026-08-02: **gate chain launched** (GPU, ~35 min): gate1 z=0 seed 42 → bit-exact
  check vs x0sweep_s-1_seed42 (hard abort on flip) → same for seed 123 → S0
  conditions on suite 42: bias U(±0.125), σ0.3 ×2 draws, σ0.6. Logs:
  `runs/exp07_steer/{gate1_*,s0_*}.{json,log}` + chain.log.
- 2026-08-02: **Gate 1 PASSED, bit-exact on BOTH suites** — z=0 through the full
  chunk-window path reproduces the x0-zeros base episode-for-episode: 56.2% @42,
  54.7% @123 (= 55.5% pooled), zero flips, identical lengths/placed/max-can-z
  (exp07_check_match.py). Belief 1 confirmed: the free-running-controller design
  means the wrapper adds NO behavioral deviation at z=0. S0 conditions running.
- 2026-08-02: **Gate S0 PASSED — σ_init −1.2 (σ≈0.30) CONFIRMED from data.** Suite 42
  (z=0 reference 56.2%): bias U(±0.125) 60.9% (+4.7, noise-level — the random mu bias
  is harmless in x0-space, measured not assumed); σ0.3 draws 53.1% / 54.7% (−3.1/−1.5
  — well inside the ≤10-pt criterion); σ0.6 35.9% (−20.3 — too hot, and evidence the
  x0 surface has real slope beyond the locally-flat mode region: leverage exists
  exactly as belief 2 guessed). Epoch-0 training reference locked: exploration-level
  rollouts should collect ≈ base-level reward (+1600s raw/episode); ≤0 = STOP rule.
- 2026-08-02: launching training: **smoke** (5 epochs, 2048 envs — integration
  shakeout at scale, tensorboard Episode_Reward/placed must be nonzero from epoch 1)
  then **s1** = {α_x0 1.0, σ_init −1.2, PPO seed 1, 2048 envs, 200 epochs ≈ 9.8M
  windows ≈ 147M env steps, window-aligned eval-protocol env}. Logs:
  `runs/exp07_steer/{smoke2048,s1_seed1}/`.
- 2026-08-02: **smoke PASSED (with one logging caveat understood, not glossed):**
  5 epochs / 2048 envs clean, no NaN, ~740 windows/s (≈11k env-steps/s → 200 epochs
  ≈ 3.7 h, a bit over the pre-registered 2.5–3 h estimate; accepted). Tensorboard
  Episode_Reward/placed shows 0.0 at epochs 1–3 — NOT the r1/r2 signature: an
  episode is 100 windows, an epoch 24, so the FIRST completed episodes land in epoch
  ~5; the episodic logger has nothing before that. The decisive number: the epoch-5
  checkpoint (first completed episodes) carries mean episodic reward **+1450** ≈
  base level (zero-ref +1644, r3 healthy +1610; r1/r2 broken ≈ −30). Gate-2 rule
  refined for window-RL: judge placed/reward at the first epoch AFTER episode
  completion (epoch ≥ 5), not epoch 1. Full run s1_seed1 launched; hard health check
  at epoch 20 (placed must be >0 and episodic reward ≈ +1600s, else STOP + diag).
- 2026-08-02 ~14:10: **Gate 2 PASSED (tensorboard, epoch 50 of 200):** placed reward
  0.0 only for epochs 1–4 (no completed episodes yet), then 71.1 by epoch 50;
  rewards/iter +1450 (epoch 5, = base level) → **+2133** and climbing;
  **ever_both_placed 0.79** on the training distribution (vs 55.5% deterministic-base
  success) — a metric that stream-reward gaming can't easily fake; episode lengths
  exactly 100 windows (alignment holding). CAUTION (r3 lesson): training metrics
  under exploration noise ≠ held-out deterministic eval; verdict waits on the pooled
  two-suite eval. Run ETA ~17:00 (~50 epochs/h); periodic ckpts every 50 epochs make
  the run resumable via train_steer.py --checkpoint if the machine goes down.
- 2026-08-02 ~15:05: **run STOPPED at ~epoch 105/200 by Big Will (leaving; machine
  going down). NOT a failure — reward still climbing: +1450 (ep5) → +2133 (ep50) →
  +2320 (ep100).** GPU verified free. Resume state: `s1_seed1/nn/
  last_exp07_steer_ep_100_rew_2319.8398.pth` (periodic) + `exp07_steer.pth` (rolling
  best). RESUME COMMAND (env_isaaclab6, from act/):
  `python train_steer.py --ckpt ../runs/exp03_N3/ckpt_final.pt --run-name s1_seed1
  --seed 1 --num_envs 2048 --max_iterations 200
  --checkpoint ../runs/exp07_steer/s1_seed1/nn/last_exp07_steer_ep_100_rew_2319.8398.pth`
  (rl_games restores model+optimizer+epoch → continues 100→200; same run dir, new
  events file). Then gate 3: pooled two-suite eval + taxonomy diff + z
  state-dependence (commands in HANDOFF).
- 2026-08-02 ~18:20: **RESUMED s1_seed1 (pid 5344) with the verbatim command above.
  Continuity CONFIRMED from tensorboard (new events file, summaries/):** epoch
  counter restored at 101 (not 1); first completed-episode reward at epoch 105 =
  **+2331.6** vs +2319.8 at the ep_100 checkpoint — exact continuation, no cold
  start. Episode lengths exactly 100 windows (alignment still holding).
  `Episode_Reward/placed` read 0.0 for epochs 101–104 — the KNOWN window-RL logging
  artifact (fresh envs after relaunch, first episodes complete ~5 epochs in), then
  68.0 at epoch 105. Watchers: persistent monitor on process exit + new checkpoints;
  ETA epoch 200 ≈ 2 h. Next: gate 3 pooled eval on `nn/exp07_steer.pth`.
- 2026-08-02 ~19:25: **s1_seed1 TRAINING COMPLETE — 200/200 epochs ("MAX EPOCHS
  NUM!"), exited cleanly.** Reward trajectory: +1450 (ep5, base level) → +2133
  (ep50) → +2320 (ep100) → +2361 (ep150) → **+2416 (ep200, = rolling best
  `exp07_steer.pth`)**. Slope flattening over the resumed half (+96 over 100 epochs
  vs +870 over the first 95) — consistent with plateau; per the r3 lesson, training
  reward under exploration noise decides NOTHING. **Gate 3 launched:** deterministic
  pooled two-suite eval (seeds 42+123) of `exp07_steer.pth`, chain log
  `runs/exp07_steer/gate3_chain.log`, outputs `s1_best_seed{42,123}.json`.
- 2026-08-02 ~20:20: **GATE 3 RESULT — pooled 117/128 = 91.4%** (seed42 57/64 =
  89.1%, seed123 60/64 = 93.8%), deterministic z = clamp(mu), x0 = tanh(z), ladder
  suites. Analyzer: `experiments/exp07_analyze_s1.py` (paired, causal — both sides
  deterministic on identical spawns).

## VERDICT (2026-08-02): x0-STEERING WORKS — 55.5% → 91.4% pooled, Gate 6 (90%) CLEARED

**Headline:** +35.9 pts over the like-for-like fixed-x0-zeros base (55.5%), +27.3
over the stochastic headline (64.1%), and above the 90% Gate-6 bar — with the FIRST
pre-registered config (7-D broadcast z, α_x0=1.0, σ_init −1.2, 200 epochs, bare
placed-stream reward, zero train/eval mismatch). No PPO tuning, no escalation needed.

**Taxonomy diff (the discrimination signature EXP06 lacked, exactly as predicted):**
- fixed 51 / broken 5 (seed42: 24/3, seed123: 27/2) — vs r3's symmetric 26/26 churn.
- **never_lifted collapses:** 18 of 19 base never-lifted episodes now SUCCEED
  (12 @42 → 11 success; 7 @123 → 7 success). The grasp-phase failure mode additive
  couldn't touch is essentially solved by steering the decoder.
- Every base failure bucket transfers mass to SUCCESS: lifted_never_placed 10/11,
  placed1_stuck_lift 10/11, placed1_stuck_low 13/15 across suites.
- Remaining 11 failures: placed1_stuck_low 4, placed1_stuck_lift 3,
  lifted_never_placed 2, never_lifted 2 — no new failure mode invented.
- **z is state-DEPENDENT (belief 3 confirmed):** mean |z| 0.220 on success episodes
  vs 0.282 on failures; within-episode z_std 0.21 vs 0.26 — identical pattern on
  both suites (r3's additive residual sat at a state-independent 0.0084 everywhere).
  The policy modulates x0 harder and more variably exactly where the base struggles.

**Why 91.4% is credible (odd-number validation):** (a) the eval path was proven
bit-exact at z=0 against the base suites (gate 1) — only mu differs here; (b) paired
per-episode diffs show coherent bucket structure, not uniform trivial success;
(c) training-distribution ever_both_placed was already 0.79 at ep50 UNDER exploration
noise — a deterministic-mu eval above that is consistent; (d) mean_z_mag 0.22
confirms a nonzero learned policy, and suites are the untouched ladder spawn suites.

**Belief scorecard (6 pre-registered):**
1. z=0 bit-exact through wrapper — **CONFIRMED** (gate 1, both suites, zero flips).
2. Exploration not free; σ0.3 ≤10 pts — **CONFIRMED** (−2 pts @σ0.3, −20 @σ0.6;
   σ_init −1.2 chosen from data).
3. Learned z state-dependent — **CONFIRMED** (0.220 vs 0.282 |z|; z_std 0.21 vs 0.26).
4. Result guess 62–72% pooled — **WRONG, UNDERSHOT: actual 91.4%.** Steering
   recovered not just the determinism tax but nearly all grasp-phase AND
   carry/release failures. The RFS-scale jump (43→86) DID transfer.
5. Broadcast z too coarse — **REFUTED**: 7-D constant-per-chunk z sufficed;
   escalation (14-D ramp) never triggered.
6. Seed variance rule — margin +35.9 pts is far outside the 55–65% decision band
   and ~7× the ±5-pt noise floor; single seed is decisive per the pre-registered
   rule. (Replicas remain optional polish, not evidence.)

**Status: EXP07 CLOSED — SUCCESS.** Champion: `runs/exp07_steer/s1_seed1/nn/
exp07_steer.pth` (steering head) on frozen base `runs/exp03_N3/ckpt_final.pt`.
Follow-ups (not blocking closure): optional seed replicas; perturbed/robustness
composite for the full Gate-6 criterion (parked dynamics-diversity round); close-up
failure video of the 11 residual failures on request.
- 2026-08-03: post-closure infra: `eval_steer.py` gained `--video/--video-length/
  --video-folder/--viewer-eye/--viewer-lookat` (ported from eval_act.py) for
  close-up renders of the final policy; success-episode recording (seed 42,
  1 env, 3 eps) → `runs/exp07_steer/videos_closeup/` + `video_success_seed42.json`.
  Distilled mechanism write-up added as POSTMORTEM §9; JOURNAL updated with the
  full EXP06→EXP07 arc.
- 2026-08-03: second close-up batch (seed 123, 1 env, 3 eps): 3/3 success →
  `runs/exp07_steer/videos_closeup_s123/`. Totals across both batches: 5
  successes + 1 failure (batch-1 ep0, placed 1/2) on camera, awaiting Big Will's
  review.
