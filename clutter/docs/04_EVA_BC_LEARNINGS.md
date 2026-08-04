# 04 — eva_bc institutional memory, distilled for the clutter task

*Everything below is measured on eva_bc's own 2-can pick-and-place task. It cost GPU-weeks.
Read before running anything. Numbers are quoted, not paraphrased.*

Sources: `docs/PLAN.md`, `docs/JOURNAL.md`, `docs/POSTMORTEM.md`, `docs/HANDOFF.md`,
`docs/experiments/EXP01,02,03,06,07 + EXP_INDEX + LITERATURE`.

---

## 0. The single most important structural fact

**EXP07 (x0-steering) was built, its first three gates PASSED, and it was never finished.**

| stage | state |
|---|---|
| implementation (`steer_core/steer_wrapper/train_steer/eval_steer` + yaml) | landed, CPU unit-tested, all pass |
| **Gate 1** — z = 0 reproduces the fixed-x0 base | **PASSED bit-exact, zero flips**, 56.2 % @ seed 42 / 54.7 % @ seed 123 = **55.5 % pooled** |
| **Gate S0** — exploration-noise tolerance | **PASSED**, σ_init = −1.2 (σ ≈ 0.30) chosen *from data* |
| **Gate 2** — training health | **PASSED at epoch 50/200**: placed-reward 71.1, reward +1450 → +2133, `ever_both_placed` 0.79 |
| **Gate 3** — the result | **NEVER REACHED.** No pooled eval, no taxonomy diff, no z-state-dependence check |

The run died at epoch 50 of 200 and no artifacts survive in this tree (`runs/` is gitignored,
and the source machine `/home/william/...` does not exist here). So the *recommended* method
in eva_bc's own README is **unvalidated on its home task**. That is a risk we inherit and must
name: we are not porting a proven recipe, we are porting a recipe that passed its plumbing
gates and was cut short.

**Consequence for our plan:** treat x0-steering as promising-but-unproven, and do not stake the
70 % target on it. The staged BC pipeline up to Stage 3 is the part with real evidence.

---

## 1. The rules that must govern our ladder

### Statistics — the most expensive lesson in the repo

- **Training-seed variance was 26.6 points.** Identical data, identical recipe, seeds 1/2/3:
  **32.8 % / 50.0 % / 59.4 %**. `train_flow.py` was entirely unseeded when the original
  comparisons were made.
- **This voided every single-run A/B in nine months of work** — "DAgger nets zero",
  "offline recovery data actively hurts", the whole "17 fixed / 17 broken" narrative. The
  v1↔v3 churn of 34 episodes sits exactly on the same-data-different-seed noise floor of
  **31–39**.
- **Standing rule: ≥3 seeds per arm; select the champion on a held-out spawn seed; quote
  pooled ≥128-episode numbers only. Single-run comparisons are void.**
- Suite-to-suite wobble at n = 64 is **±5–10 points** (N3 read +9.4 on one suite, v3 −9.4).
- Unpaired n = 32 suites carry **±8 points**. Always pass `--seed`.
- Inference variance is **zero** with a seeded x0 (re-running one checkpoint churns 0
  episodes) — so per-episode diffs *within* a checkpoint are meaningful; between two
  separately-trained checkpoints they are nearly meaningless without replicas.

### Config traps

- **`clip_actions` in rl_games is an action SCALE, not a clip.** `preprocess_actions`
  (`a2c_common.py:1181-84`) clamps to [−1, 1] then **rescales to the action-space bounds**.
  `clip_actions: 100.0` multiplied every action ×100 and destroyed **two full training runs**
  (0.0 % success). **It must be 1.0.** — and the clutter env's shipped yaml has **100.0**.
- **"Zero-init residual" needs three things**: zero the mu *weight* (rl_games `mu_init`
  touches only the weight); know the mu *bias* stays random at ≈±0.125 (measured harmless);
  and a **small initial σ**. σ ≈ 0.37 (≈1°/joint/step) cost **46 points** on an mm-precision
  base (57.2 % → 11.5 %); σ ≈ 0.08 was harmless.
- **Measure the frozen base's noise tolerance before training** — EXP07 turned this into
  gate S0 and picked σ from data.

### Diagnostics

- **Fixed-action attribution beats theorizing.** `diag_training_env.py` reproduces the exact
  training config and rolls zero / bias / small-noise / large-noise actions with no RL. One
  10-minute run found what two rounds of training-log analysis had misdiagnosed: nothing an
  epoch-0 agent could emit produced the observed reward, so the *config* was broken before
  learning began.
- **Check the cheapest signal first**: `Episode_Reward/<term> = 0.0 from epoch 1` was visible
  in tensorboard the entire time and would have caught both dead runs.
- **Adapt health-check timing to the RL clock.** With chunk-window RL an episode is 100
  windows and an epoch is 24, so per-episode loggers are legitimately empty until epoch ~5.
  Do not read that as pathology.
- **Never trust train reward.** EXP06 r3 beat the base on train reward (+1700 vs +1644) with
  **exactly flat** success. Dense shaping streams pay for earlier/longer partial progress.
- **Trace the physical quantity, not the commanded one.** The lying-can root cause only
  appeared from an FK audit of commanded-q (from recorded actions) vs achieved-q (from obs):
  commanded z was uniform while executed z spread **16 mm**.
- **Know when an instrumentation line runs.** A constant calibrated from a print that
  executed *post-lift* rather than *at close* was off by 80 mm and wasted a full GPU run.

### Wrapper discipline

- **Gate every wrapper on bit-exact reproduction of the previous stage** with the learned
  component zeroed, before any gradient. It is cheap, decisive, and permanently exonerates
  the wrapper in every later debugging session. Make it enforceable with a comparator that
  exits nonzero (`experiments/exp07_check_match.py`).
- **Vectorize per-env controller state.** Per-env python deques were the real scaling limit
  long before VRAM; a tensor queue `_buf (N, 15, 7)` + `_idx (N,)` took residual training
  from 128 to 2048 envs. Gate the rewrite on bit-exactness too.

### Chunked-policy specifics

- **Chunk commitment is load-bearing.** Shortening the execution horizon at eval time, with
  no retraining, collapsed success monotonically:
  **59.4 → 32.8 → 3.1 → 0.0 → 0.0 %** at `n_action_steps` = 15 / 8 / 4 / 2 / 1.
  The open-loop window is not a latency cost being paid — it is *why the policy works*.
  **Never shorten the horizon to fix precision.**
- **A flow policy's x0 is a policy-level knob.** Frozen draws span **14.1 %–56.2 %** success
  on identical weights; freezing x0 at a bad draw flipped 33 of 64 episodes. **Zeros (the
  distribution mean) was best.** Fixing x0 for determinism costs **~9 points** (55.5 % vs
  64.1 % pooled) because per-refill re-rolling self-corrects bad draws.
- **Steer on-manifold, don't add off-manifold.** The additive residual fixed 26 episodes and
  broke 26 — causally, since both sides were deterministic on identical spawns. The learned
  residual was **state-independent** (~0.0084 units everywhere, identical on successes and
  failures): PPO learned *effort*, not *discrimination*, despite having a grasp bit and
  relative poses in its observation.
- **Feed the loss mask through attention**, not just the loss — the decoder's
  `key_padding_mask` from `action_is_pad` stops censored positions leaking into valid ones.
  Verified bit-identical under garbage ×1e3.
- **Verify normalization ownership.** This LeRobot version normalizes *outside* the policy;
  deployment must normalize obs and unnormalize actions, and stats live in the checkpoint.
  Applying one model's stats to another silently corrupted a whole analysis.

### Representation

- **Information can be present in the obs but unused (a salience failure, not partial
  observability).** The grasped-vs-closed-on-air distinction — the dominant failure — is
  decodable from a *single frame*: a 5-dim probe (finger pos + finger vel + **last commanded
  grip**) hit **AUC 0.968 with 0 % FPR** on 665 real on-policy freeze states, while the same
  probe given all 41 dims mislabelled **53.5 %** of them.
- **The commanded-grip channel is mandatory.** Physical finger joints alone score higher AUC
  (0.976) but **27.1 % FPR** — the disambiguating fact is "commanded closed AND resulting
  aperture", not aperture alone.
- **History is refuted as the fix**: ≤ +0.01 AUC, and finger-history transfers *worse*
  (40.8 % FPR).
- **A hand-written aperture threshold does not work**: 94.6 % accuracy but **40 % FPR** —
  raw aperture distributions of grasped vs closed-on-air overlap almost completely (medians
  −0.053 vs −0.056). The working bit is a small nonlinear MLP.

### Data / BC pipeline

- **Never supervise a failure; supervise the fix.** The mask boundary is failure
  *detection*; post-detection recovery stays trainable. The expert-failure mask is
  unconditional.
- **BC trains only on expert-generated actions.** Student actions are never labels; failed
  rollouts never enter the pool.
- **Filter at attempt granularity, not episode granularity.**
- **Preserve multimodality deliberately** — sample from top-p feasible candidates by margin,
  not argmax.
- **Check the eval horizon against the data's episode-length distribution before believing a
  0 % result.** A 500-step eval horizon against demos with median length 677 produced a
  spurious 0/4; only 7.9 % of demos would have fit.

### Sim / planning gotchas

- **Planner-valid ≠ executable.** cuRobo returns collision-free plans whose end poses the
  PD-controlled arm cannot track; executed height spread **16 mm** across grasp-table row
  families for one planned target, and corrective nudges of 12 mm commanded moved the arm
  1–4 mm *or backwards* — the arm was clamped. Detection via an **executed-state check** is
  the only fix.
- **PhysX contact-free ≠ planner margin-free.** A learned policy parks the arm inside
  planner sphere margins, killing every plan from that state.
- **Exclude retry candidates by identity, not list position.** Excluding positions re-served
  the exact failed row on 4/10 retry sequences.
- **Give each subsystem its own RNG stream.** Perturb scheduling and candidate dropout shared
  one stream, so the first behavioural divergence reshuffled every later episode's
  perturbation (12/31 episodes drew different events) and silently broke paired A/B.
- **Verify frame conventions numerically both ways.** cuRobo is WXYZ, Isaac Lab 3.0 is XYZW.

### Ops

- One GPU job at a time; background chains with durable logs + a monitor on the log.
- `ps aux | grep "[t]rain_\|[e]val_"`, not `pgrep -f`.
- Snapshot fixed-name artifacts per seed inside chain scripts, and skip seeds whose output
  already exists so chains are resumable.
- Freeze the runner during a multi-seed generation chain — each seed reloads the module, so a
  mid-chain edit forks later seeds' behaviour.
- **One informed change per long run.** A "+4 fixes at once" run went 87.5 → 81.3 % and cost
  three more 32-episode runs to attribute the damage.

---

## 2. Numbers to calibrate against

### eva_bc's own ladder (2-can pick-and-place, 64-ep suites unless noted)

| checkpoint | seed 42 | seed 123 | pooled 128 |
|---|---|---|---|
| `flow_nominal_v1` | 59.4 % | 57.8 % | 58.6 % |
| `flow_dagger_v3` | 59.4 % | 50.0 % | 54.7 % |
| exp03 N1 / N2 / **N3** (nominal, seeds 1/2/3) | 32.8 / 50.0 / **59.4** | — / — / 68.8 | — / — / **64.1 %** ← champion |
| exp03 D1 / D2 / D3 (+dagger) | 60.9 / 53.1 / 56.2 | 57.8 | 59.4 |
| N3 with fixed **x0 = zeros** | 56.2 % | 54.7 % | **55.5 %** ← deterministic base |
| N3 with fixed x0 seed 3 / 7 / 1 / 2 | 51.6 / 51.6 / 37.5 / **14.1** | | |
| EXP06 r3 (additive residual, healthy) | 52.3 | 58.6 | **55.5 % — exactly flat** |
| EXP06 r1 / r2 (`clip_actions` bug) | 0.0 % | 0.0 % | 0.0 % |
| EXP07 s1_seed1 (x0-steering) | — | — | **never evaluated** |

### Expert (teacher) baselines

| metric | value |
|---|---|
| old scripted expert (the floor) | 48.8 % |
| expert nominal, best version (v7, 32 eps) | **96.9 %** |
| expert nominal, 8-seed production chain (504 eps) | **85.7 %** |
| expert perturbed / recovery (seeded, reproduced 3×) | **77.4 %** |
| DAgger takeover success | 68 % |
| clean-grasp rate | 80.3–92 % (never hit the 95 % bar) |

**The teacher ceiling is the pipeline ceiling.** A ≤77 %-reliable teacher supervising the
policy's hardest states caps everything downstream — this is why our Stage 1 target must be
set high and measured honestly.

### Champion failure anatomy (N3, 46 failures over 128 eps, exhaustive)

**34 grasp-phase miss/freeze** (18 never-lifted + 16 placed-1-then-closed-on-air)
+ **12 carry/release** (9 lifted-never-placed + 3 stuck-at-carry) + **0 drops-after-place**.
**74 % of the residue is grasp-alignment shaped.**

### Throughput and cost (on a 12 GB card — ours is 10 GiB, so re-measure)

| item | number |
|---|---|
| flow BC training | 100k steps, batch 64, lr 1e-4 → **~35–40 min at ~45 steps/s, 1.0 GB VRAM**; loss 0.478 → 0.06 |
| flow policy size | **703k params** |
| policy eval | **~5 min / 64 eps** at 16 envs |
| expert demo generation | ~28 min / 63 episodes; 504 eps in 3 h 47 m |
| DAgger collection | 208 rollouts / 100 takeovers in ~1 h 45 m |
| residual PPO | 2048 envs, 600 epochs ≈ 29.5 M env-steps in **~35 min (~15k steps/s)** |
| steering PPO | 2048 envs, **~740 windows/s ≈ 11k env-steps/s** → 200 epochs ≈ 3.7 h |
| 3-seed replica chain | 6 × (35 min train + 12 min eval) ≈ 4.7 h |

### Reward reference levels used as training-health tripwires

zero-action reference **+1644 raw/episode** · σ0.08 +1657 · σ0.37 **+519** ·
r3 healthy +1610 → +1700 · **broken (`clip_actions` bug) ≈ −30** ·
EXP07 epoch-5 +1450, epoch-50 +2133.

### Dataset scale that produced 64.1 %

504 expert episodes over 8 seeds → **292 nominal-clean + 140 recovery demos**,
**314k trainable samples**, mask 7.7 %. Probe corpus: 4,368 expert post-close frames +
665 on-policy closed-on-air frames.

---

## 3. Architecture reference (what we are porting)

**Flow-matching chunk policy** (`act/modeling_flow.py`): vendored LeRobot ACT transformer with
CVAE/KL deleted; conditioning is a single timestep of privileged state split into
`observation.state` (proprio) + `observation.environment_state`; flow head takes projected
noisy chunk `x_τ` + sinusoidal positions as decoder queries, with scalar flow-time τ embedded
and added to all tokens; rectified-flow loss `v = x1 − x0` masked by `action_is_pad`;
**10 Euler steps** at inference; seedable generator (bit-deterministic given x0).

**Chunking:** chunk **50**, execute **15**, temporal ensembling **OFF**, per-env queues.

**Normalization:** external mean/std — the controller normalizes obs and unnormalizes actions.

**Training:** 100k steps, lr 1e-4, batch 64.

**x0-steering (EXP07, the recommended RL stage):** one RL action per 15-step execution window;
`x0_steer = α_x0 · tanh(z)` with **z ∈ R⁷** broadcast across all chunk positions,
**α_x0 = 1.0**; the controller stays **free-running** (each env refills when its own queue
empties — this is what made gate 1 bit-exact, and it supersedes HANDOFF's synchronous-window
sketch); window-summed reward; obs = base obs + finger features + grasp bit + relative poses;
PPO with `clip_actions: 1.0`, `σ_init −1.2`, `horizon_length 24` windows.

---

## 4. What transfers to clutter, and what does not

### Transfers directly

- The **action encoding** — clutter uses the identical action terms (`scale=0.5`,
  `use_default_offset=True`, binary gripper), so `(q_des − q_default)/0.5` and `±1` port
  unchanged, and `q_default` is the same `_START_POSE`.
- **XYZW quaternion convention** — same.
- The **grasp-bit probe recipe** — finger pos + finger vel + last grip command, all present in
  clutter's 42-D obs at indices `(6, 7, 14, 15, 41)`.
- The whole **methodology**: gates, seeds, bit-exactness, per-term logging, taxonomy diffs.
- The **vectorized tensor-queue controller** and the flow model itself.

### Does not transfer

- **The cuRobo expert.** Not installed here, and the clutter motion is straight-line Cartesian
  servoing through a slot, not free-space planning.
- **The grasp table and carry waypoints** — built for 24 mm cylindrical cans at pick-place
  spawn radii, not 36 × 30 × 70 mm cuboids in a row at r = 0.25.
- **`objects_canonical`** and the two-object permutation logic — clutter has one target, so
  this entire class of bug disappears.
- **The obs layout constants** (41-D, 16/25 split) — must be re-derived for 42-D.
- **The perturbation suite** — clutter has no mid-episode events at all, so eva_bc's
  "perturbed expert caps DAgger" logic has nothing to run against until we build one.

### New in clutter, with no eva_bc precedent

- A **hard constraint termination** (`distractor_toppled`) that fires on a *bystander*, not
  the manipulated object. Nothing in eva_bc's task had this.
- **Deceptive shaping**: the reach gradient pulls the gripper toward a target flanked by
  things that end the episode on contact.
- **No `gripper_toggle` penalty**, so nothing suppresses grip chatter near the gap.
- **No domain randomisation**, so a clutter policy is less robustness-tested by construction.
</content>
