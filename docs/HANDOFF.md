# HANDOFF — 2026-08-02 ~20:30 PDT (EXP07 CLOSED — SUCCESS: 55.5% → 91.4% pooled, Gate 6 cleared)

> **STATUS UPDATE (evening session): §3 steps 1–5 are DONE.** s1_seed1 resumed and
> finished 200/200 epochs (reward +2416 final); gate 3 pooled eval = **117/128 =
> 91.4%** (89.1 @42 / 93.8 @123) vs 55.5% base; taxonomy 51 fixed / 5 broken,
> never-lifted collapsed 18/19; z state-dependent (0.220 succ vs 0.282 fail |z|).
> Verdict + belief scorecard written into `experiments/EXP07_x0_steering.md`;
> POSTMORTEM §8, EXP_INDEX, memory all updated. Analyzer:
> `experiments/exp07_analyze_s1.py`; eval JSONs `runs/exp07_steer/s1_best_seed{42,123}.json`.
> **Champion: frozen `runs/exp03_N3/ckpt_final.pt` + steering head
> `runs/exp07_steer/s1_seed1/nn/exp07_steer.pth`.**
> **STATUS UPDATE 2 (2026-08-03): §3 step 6 ALSO DONE — reBot_ACT is now the
> eva_bc working clone** (eva_bc git history adopted in place; docs live in
> `docs/`; runs//weights/assets/third_party local-only via .gitignore +
> .git/info/exclude; sync script retired in fc21c45). Deep write-ups landed:
> **POSTMORTEM §9** (why additive failed / steering worked — mode errors vs aim
> errors, RL as state-conditioned mode selector) + JOURNAL entries for the whole
> EXP06/EXP07 arc. eval_steer.py gained --video/--viewer-eye; close-up success
> videos of the final policy render to `runs/exp07_steer/videos_closeup/`.
> **Pending Big Will only:** `git push -u origin main` from reBot_ACT (new
> commits), then the rm lists (old-layout doc duplicates, old ckpts, failed-run
> dirs, the redundant ~/…/reBot/eva_bc staging clone after push).
> **Open follow-ups:** perturbed/robustness composite on the steered stack (the
> full Gate-6 criterion), optional steering seed replicas, review of the 11
> remaining failures. The sections below are the pre-resume snapshot, kept for
> context.

For the next session. Read this + `experiments/EXP07_x0_steering.md` (pre-registered
beliefs, gates, full log) + `experiments/EXP06_residual_rl.md` (the closed additive
arc + belief scorecard) + `experiments/EXP_INDEX.md`. POSTMORTEM.md carries dated
CORRECTION blocks — parts of the original are retracted.
Env: `source /home/william/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6`.
ONE GPU job at a time (`nvidia-smi` + `ps aux | grep "[t]rain_\|[e]val_"` — NOT `pgrep -f`).
Big Will reviews all rendered output — never view images/videos yourself. `rm` is
classifier-blocked — hand Big Will exact commands. Address him as Big Will. Short
Bash timeouts; long jobs = background chains with durable logs; log stdout buffers
heavily — trust tensorboard events + checkpoint filenames over the .log for progress.

**RUNNING AT WRITE TIME:** nothing. GPU free (15 MiB). No monitors armed.
**NEXT ACTION: resume s1_seed1 training from epoch 100 (§3 step 1 — exact command
below), let it finish 200 epochs (~2 h), then gate 3 (pooled eval + analyses).**

---

## 1. WHAT FINISHED TODAY (2026-08-02 afternoon session)

### a. EXP07 x0-steering — designed, pre-registered, built, ALL PRE-TRAINING GATES PASSED
Everything per the project convention: `experiments/EXP07_x0_steering.md` written
BEFORE any code (6 beliefs incl. a falsifiable 62–72%-pooled result guess and an
escalation rule; ordered gates S0/1/2/3).

**Design (implemented and verified):**
- RL acts at CHUNK granularity: z ∈ R^7 (6 arm + 1 grip column), one rl_games step =
  one 15-env-step window; x0 = α_x0·tanh(z) (α_x0=1.0) broadcast across all 50 chunk
  positions; base integrates from that x0 instead of zeros → outputs always
  on-manifold; effective per-dim x0 range ±tanh(1)≈±0.76 (clip_actions 1.0).
- **Controller stays FREE-RUNNING** (refills on empty queue exactly as in every
  ladder eval); z only enters via the x0 of refills that occur while it is held.
  This is why gate 1 could demand BIT-exactness (no truncation/re-sync hacks).
- **Window-aligned protocol, training = eval protocol:** object_dropping termination
  + dropping_penalty OFF in training too (they are inseparable: the penalty is an
  `is_terminated_term`, pick_place_env_cfg.py:347/373). 30 s = 1500 steps = exactly
  100 windows → every reset lands on a window boundary; only §4.2 flushes desync an
  env (z applied ≤14 steps stale; accepted, documented). Zero train/eval mismatch.
- Obs 56-D = the EXP06 64-D minus queued-base-action(7) minus chunk-index(1):
  [0:41] base obs | [41:45] fingers | [45:46] grasp bit | [46:53] can-in-gripper
  pose | [53:56] can→basket delta. Reward: bare `objects_placed` stream, NO
  residual-magnitude penalty (z bounded by construction).

**Gate results (all on disk, `runs/exp07_steer/`):**
- **Gate 1 PASS, BIT-EXACT both suites:** z=0 through the full window path ==
  `x0sweep_s-1_seed{42,123}.json` episode-for-episode (56.2% @42 / 54.7% @123 =
  55.5% pooled; zero flips; comparator `experiments/exp07_check_match.py`).
- **Gate S0 PASS — σ_init −1.2 (σ≈0.30) chosen FROM DATA:** suite 42 (ref 56.2%):
  bias U(±0.125) 60.9% (mu-bias harmless in x0-space, measured); σ0.3 draws
  53.1/54.7% (−2 pts — locally FLAT around the x0 mode); σ0.6 35.9% (−20 pts —
  real slope further out = leverage exists). JSONs: `s0_{bias,sig03a,sig03b,sig06}_seed42.json`.
- **Gate 2 PASS (training health):** smoke 5 epochs @2048 envs clean; full run
  s1_seed1 healthy at every checkpoint. KEY window-RL logging lesson: an episode is
  100 windows but an epoch is 24 → `Episode_Reward/placed` is 0.0 for epochs 1–4
  because NO EPISODE HAS COMPLETED YET — judge health at epoch ≥5, not epoch 1
  (r1/r2's genuine failure signature was ≈−30 episodic reward; healthy base level is
  ≈+1644).

**Training progress at stop (NOT a failure — Big Will had to leave):** epoch ~105/200,
killed cleanly, GPU freed. Reward trajectory **+1450 (ep5, base level) → +2133
(ep50) → +2320 (ep100), still climbing**; `ever_both_placed` 0.79 @ep50 on the
training distribution (vs 55.5% deterministic-base success) — hard to fake with
stream-reward gaming, but r3's lesson stands: ONLY the held-out pooled eval counts.

### b. Infrastructure added (all CPU-unit-tested and/or gate-verified)
- `act/modeling_flow.py`: `predict_action_chunk` x0 now accepts (chunk,7) OR
  per-env (B,chunk,7).
- `act/eval_act.py`: controller `steer_x0` attr — per-env x0 override in `_refill`
  (precedence over `fixed_x0`).
- `act/residual_core.py`: refactor only — extracted `task_features()` (15-D shared
  tail) + `flush_check()`; obs64 layout/behavior unchanged.
- `act/steer_core.py` (SteerCore: set_steer/build_obs, subclasses ResidualCore),
  `act/steer_wrapper.py` (SteerRlGamesWrapper: 56/7 spaces, window step loop),
  `act/train_steer.py`, `act/eval_steer.py` (ladder protocol + fixed-z modes
  `--z-sigma/--z-bias/--z-seed`, per-episode mean_z_mag + z_std recorded),
  `act/steer_ppo_cfg.yaml` (σ_init −1.2; clip_actions 1.0 MANDATORY comment).
- `act/eval_residual.py`: `load_residual_policy` parameterized (obs_dim/act_dim) —
  reused by eval_steer for the 56/7 network.
- CPU test (scratchpad, gone after reboot; trivially rewritable): steer-zeros ==
  fixed-zeros action-for-action; per-env routing isolated; tanh bound/broadcast;
  real-model (chunk,7)-vs-(B,chunk,7) bit-identical.

### c. eva_bc shared repo — PUSHED (subagent)
- `git@github.com:wwangg22/eva_bc.git` `main` commit `1c04eca`, 67 files; local
  clone `/home/william/Desktop/isaacLab/reBot/eva_bc`. Curated copy of the pipeline
  (act/, expert/, experiments/ code; all docs under docs/; README with stage-by-stage
  pipeline + big Lessons Learned section; `sync_from_source.sh` bridge script,
  tested idempotent, never writes to source, never auto-commits). Excluded: runs/,
  *.h5, *.pt/pth, logs, result JSONs, third_party/, assets/, reBot_RL package
  (README documents the bring-your-own-env mdp interface instead).
- **Big Will's explicit plan: the GITHUB REPO IS CANONICAL eventually** — after the
  EXP07 verdict, CUT reBot_ACT OVER to eva_bc (adopt its layout/git history, keep
  runs//h5/pt local via .gitignore, retire the sync script). Memory
  `eva-bc-shared-repo.md` records this. If he intended the repo private, he should
  check visibility on GitHub (full lab notebook is in there).

## 2. NEW LESSONS (beyond the EXP06/POSTMORTEM canon, which all still stands)

1. **Free-running controller + windowed RL clock** beats forced-synchronous designs:
   no truncation, no reward-carry bookkeeping, and the z=0 path is bit-identical to
   the base — so wrapper verification is absolute, not approximate.
2. **Window-RL episodic loggers are silent until the first episodes complete**
   (epoch ≈ episode_windows/horizon). Do not read `placed = 0.0` at epoch 1 as the
   r1/r2 collapse signature; judge at the first post-completion epoch.
3. **Measure the exploration response BEFORE training** (gate S0 pattern:
   `--z-sigma` evals): it picks σ_init from data AND provides the exact epoch-0
   reward reference for the health tripwire. The x0 surface is locally flat (σ0.3 ≈
   −2 pts) and steep at σ0.6 (−20 pts) — ideal exploration geometry.
4. Training protocol chosen for RL BOOKKEEPING (window alignment) can double as
   train/eval-mismatch elimination — check termination/penalty coupling in the env
   cfg first (`is_terminated_term` makes them inseparable).
5. Stdout .log files buffer for tens of minutes under rl_games — progress checks
   must use checkpoint mtimes + tensorboard events, not the log tail.

## 3. THE PLAN (in order; each step's success criterion inline)

1. **Resume s1_seed1** (env_isaaclab6, from `act/`, ONE GPU job):
   `python train_steer.py --ckpt ../runs/exp03_N3/ckpt_final.pt --run-name s1_seed1
   --seed 1 --num_envs 2048 --max_iterations 200
   --checkpoint ../runs/exp07_steer/s1_seed1/nn/last_exp07_steer_ep_100_rew_2319.8398.pth`
   rl_games restores model+optimizer+epoch → runs 100→200 (~2 h at ~50 epochs/h,
   ~740 windows/s). Sanity: first logged rewards should be ≈+2300 (continuity);
   watch tensorboard, not the .log.
2. **Gate 3 eval** (best ckpt = `runs/exp07_steer/s1_seed1/nn/exp07_steer.pth`):
   `python eval_steer.py --ckpt ../runs/exp03_N3/ckpt_final.pt
   --steer-ckpt ../runs/exp07_steer/s1_seed1/nn/exp07_steer.pth --seed 42
   --out ../runs/exp07_steer/s1_best_seed42.json` and the same with `--seed 123`.
   (~5 min each; deterministic: z = clamp(mu), x0 = tanh(z).)
3. **Analyses (both mandatory before any verdict):**
   - Taxonomy diff vs the x0-zeros base: adapt `experiments/exp06_analyze_r3.py`
     (PAIRS → `x0sweep_s-1_seed{42,123}.json` vs `s1_best_seed{42,123}.json`; it
     already prints transition matrices + fixed/broken lists). The DISCRIMINATION
     SIGNATURE additive lacked: never-lifted bucket collapses WITHOUT symmetric new
     breakage.
   - z state-dependence (belief 3): per-episode `mean_z_mag` and `z_std` are already
     in the eval JSONs — compare success vs failure episodes (r3's residual was
     state-independent ~0.0084 everywhere; steering should differ).
4. **Verdict thresholds (pre-registered):** vs **55.5%** fixed-x0 base (like-for-like),
   **60.5%** (+5-pt rule), **64.1%** stochastic headline, **90%** Gate 6.
   - ≥60.5% pooled → steering WORKS; consider replicas (seeds 2,3) per belief 6 if
     the margin is inside the 55–65% decision band; then push toward Gate 6
     (options: longer budget on a rising curve, richer z per belief 5).
   - <60.5% pooled after the full 200 epochs → belief-5 escalation: ONE richer
     parameterization (constant+ramp per dim, 14-D z) before any PPO tuning; if that
     is also flat → EXP06 §4.6 fallbacks (additive replicas / grasp-bit BC retrain /
     hybrid) go to Big Will with costs.
5. **Write the EXP07 verdict** into the doc (belief scorecard vs the 6 pre-registered
   beliefs), update POSTMORTEM §8 + memory `bc-flow-postmortem.md`.
6. **eva_bc cutover** (Big Will's directive, AFTER the verdict): make reBot_ACT match
   the GitHub repo (adopt eva_bc git history in-place or re-clone + carry local
   state; keep runs//h5/pt gitignored), verify stage scripts run from the new
   layout, retire `sync_from_source.sh`. Sync any EXP07-verdict doc updates to
   eva_bc first (run its sync script → review diff → commit + push).

## 4. CURRENT STATE / REFERENCE NUMBERS

- **Frozen base:** `runs/exp03_N3/ckpt_final.pt` — stochastic 64.1% pooled;
  deterministic x0-zeros **55.5% pooled** (56.2/54.7) = steering baseline.
  N3 failure anatomy (46/128): 34 grasp-phase (18 never-lifted + 16 closed-on-air)
  + 12 carry/release + 0 drops-after-place.
- **EXP06 additive residual: CLOSED** — exactly flat 55.5→55.5 (26 fixed/26 broken,
  causal, state-independent residual). One healthy seed only (replicas cancelled).
- **EXP07 files:** doc + code per §1; eval JSONs in `runs/exp07_steer/`
  (gate1_seed{42,123}, s0_*, plus .check files); training run `s1_seed1/`
  (ep_50/ep_100 + rolling best + tb events); `smoke2048/` (disposable).
- Tasks: #23 = this stage. #21/#22 open at 64.1%. #19 parked (77.4% perturbed).
- **Cleanup pending Big Will** (rm blocked for me; all logs/JSONs stay):
  - old ladder ckpts (712 MB): `rm runs/exp03_N1/ckpt_*.pt runs/exp03_N2/ckpt_*.pt runs/exp03_D2/ckpt_*.pt runs/exp03_D3/ckpt_*.pt`
  - failed EXP06 runs: `cd runs/exp06_residual && rm -r r0_smoke r0_smoke2048 r1_seed1 r2_smallsigma r3_seed2` (keep `r3_actionfix`)
  - EXP07 smoke: `rm -r runs/exp07_steer/smoke2048`
- Parked (unchanged): close-up failure video (render only on request), DAgger r2
  staged, full-v1 dynamics-diversity round for the Gate-6 perturbed composite,
  task #19 perturbed pillar.

## 5. Key files map
- **EXP07:** `experiments/EXP07_x0_steering.md` (READ FIRST), `act/{steer_core,
  steer_wrapper,train_steer,eval_steer}.py`, `act/steer_ppo_cfg.yaml`,
  `experiments/exp07_check_match.py`, `runs/exp07_steer/`.
- **EXP06 (closed):** `experiments/EXP06_residual_rl.md`, `act/{residual_core,
  residual_wrapper,train_residual,eval_residual,diag_training_env}.py`,
  `experiments/{exp06_grasp_bit.py,exp06_analyze_r3.py}`, `runs/exp06_residual/`.
- Ladder: `experiments/EXP_INDEX.md`, EXP01/02/03 docs, `LITERATURE.md` (RFS
  2602.01789), `taxonomy.py`. Policy: `act/modeling_flow.py`, `act/eval_act.py`
  (vectorized controller + steer_x0), `act/train_flow.py`, `act/dataset.py`.
- Expert/data: unchanged (`expert/demos_nominal_s10{1..8}.h5`, `dagger_r1.h5`).
- Shared repo: `/home/william/Desktop/isaacLab/reBot/eva_bc` (github canonical-to-be).
