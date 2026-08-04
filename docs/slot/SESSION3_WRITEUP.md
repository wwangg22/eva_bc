# Session 3 writeup — demo collection, the `act/` port, and two wrong diagnoses

**2026-08-02/03.** Detailed record of what was built, what was measured, what was wrong and
why. Companion documents: `HANDOFF.md` (entry point), `EXP_NOISE_SWEEP.md` (the two
pre-registered noise experiments in full), `PORT_MAP.md` (the `act/` → `slot_act/` interface),
`PLAN.md` §7 (running verdict log).

Reading order if you only want the load-bearing parts: §1 (what changed), §4 (the noise
result), §6 (the non-determinism finding — this one changes how every future number must be
read), §7 (what I got wrong).

---

## 1. What was built

| file | purpose |
|---|---|
| `slot/expert/plan.py` | the validated trajectory, extracted from `run_expert.py`. Planner + execution schedule, the latter exposed as a **generator of per-env-step commands** so every consumer executes identical steps. |
| `slot/scripts/collect_demos.py` | batched demo collection (one demo per env) in eva_bc's HDF5 schema, with DART noise injection, phase-labelled segments, per-env outcomes, and loss censoring. |
| `slot/scripts/verify_demos.py` | structural **and semantic** verification of a demo file. No Isaac Sim needed. |
| `slot/scripts/check_port.py` | static consistency check of the 41-D → 34-D port. No GPU. |
| `slot/analysis/plan_determinism.py` | is the plan a pure function of the block pose? is execution reproducible? |
| `slot/slot_mdp.py` | the four-call `mdp` surface `act/` needs, so `eva_rl` is never edited. |
| `slot/slot_act/` | the ported BC + RL stack (18 files). |

Regression check on the extraction: `Rebot-PrecisionSlot-v0` at n=128 still reports
**128/128 seated** after `run_expert.py` was rewritten to consume `plan.py`. The refactor is
behaviour-preserving.

### 1a. The `retreat` phase

Added so the episode ends with the arm clear of the slot and the demo has a stable terminal
behaviour for the ~40 steps between the trajectory's end and the 600-step episode limit.

Verified rather than assumed — block pose immediately before and after the retreat:

```
RELEASE: block (256.8, -0.5, 55.0) mm, |yaw| 0.0046
RETREAT: block (256.8, -0.5, 55.0) mm, |yaw| 0.0046
```

Zero disturbance. Total trajectory 558/600 steps.

---

## 2. The verification that mattered most

A demo file can be perfectly well-*shaped* and still be wrong in the one way that silently
destroys behaviour cloning: an off-by-one between the observation and the action it labels.
Shape checks cannot see it.

This task's 34-D observation happens to carry `last_action` in its tail, which makes the
alignment checkable exactly:

> **`obs[t, 27:34]` must equal `actions[t-1]`.**

If the collector had recorded the observation *after* stepping — the natural mistake, since
`env.step` hands one back — the identity would instead read `obs[t, 27:34] == actions[t]`, and
the policy would be trained to predict an action it can already see in its own input.

Measured on the collected pools: **max absolute difference 0.000e+00.** Exact.

`verify_demos.py` also checks dtypes and shapes, that `obs[0]`'s last-action block is zero
(the env reset cleared it), that the gripper channel is exactly ±1, that the arm tracks its
own target at rest (worst 0.0019 rad), that censoring is contiguous, and that every demo has
the same length.

---

## 3. Loss censoring — the machinery was inert, and it took a prompt to notice

The ported `train_mask` builder censors any segment whose outcome is `"missed"` or `"lost"`.
Audit of the collected pools:

```
train_mask == 0 anywhere: 0 demos   (of successful: 0)
outcomes: reach:grasp:g0=grasped:512  lift=held:512  back=held:512  spin=held:512
          turn=held:512  push=held:512  release=seated:462  release=unseated:50
```

Every phase reports `held` on all 512 demos while 50 of them end unseated. **The detector was
purely grip-based, and on this task the grip never fails** — the finger gap sits at 29.96 mm
from grasp to release even in episodes that fail. So the only failure signal arrived at the
final segment, by which point there was nothing left to censor. The machinery was ported
faithfully and could not see this expert's actual failure mode.

An all-ones mask reads as "clean data" when it can equally mean "the detector is blind".
`verify_demos.py` now reports whether the censor separates seated from failed demos, so this
particular blindness cannot recur silently.

### 3a. The right detector: in-hand offset

The expert is **open-loop after the grasp** — it plans the whole trajectory from the post-reset
block pose and never looks again. So if the block shifts in the pads, every later frame pairs
an observation showing the *new* in-hand pose with an action computed for the *old* one. That
is a **wrong label, not a noisy one**, and training on it teaches the policy to ignore in-hand
error, which is precisely the error it needs to correct.

Tracking `block_pos − TCP` against its value at the end of the grasp catches both of this
task's failure modes with one number:

* the block **sliding** in the pads (the offset drifts), and
* the block **jamming** on a wall while the gripper keeps advancing (the offset drifts just as
  much, because the gripper moves and the block does not).

The property that decided the choice: it is **noise-agnostic**. Injected DART noise moves the
arm and the block it is holding *together*, leaving the offset unchanged. A naive "mask
wherever the arm is off-nominal" detector would delete exactly the corrective supervision the
DART pool exists to provide. The distinction the implementation draws:

| deviation | recorded action | verdict |
|---|---|---|
| injected by DART noise | the correct corrective command from that state | **keep** |
| the expert's own, uncorrected | just what happened | **censor** |

### 3b. The detector was wrong twice, and the guard caught it both times

**Version 1** measured the offset over the whole episode — including after the release, where
there is nothing in the hand. The gripper retreats ~90 mm during `retreat`, so it read a 92 mm
"slip" and censored **128/128 demos, successes included**.

**Version 2** fixed the span but kept a guessed 3.0 mm threshold. Measuring the profile instead
of guessing again showed why that also fails — these are the phase-end values from demos that
all **succeeded**:

```
in-hand slip [mm]   lift 2.23   back 1.81   spin 1.82   turn 3.41   push 4.73
```

Nominal drift reaches 4.7 mm by the end of the push. The block genuinely slides a few mm in the
pads while being dragged, and swings transiently because it hangs 33 mm below the grip point.
The threshold was below the *normal* behaviour, so it flagged everything.

What caught both was a guard added on principle rather than in response to a bug:

> a censor firing on more than half of all demos, **successes included**, is a broken detector,
> not dirty data.

That guard is permanent. It is the same shape as the `verify_demos.py` check that an all-ones
mask may mean "the detector is blind" — in both directions, a censor's output distribution is
itself evidence about the censor.

**Resolution: stop guessing the threshold.** The censor now defaults to **off** and the raw
per-step signal is stored as a `slip_mm` dataset in every demo (599 float32 — negligible).
`scripts/calibrate_slip.py` derives the threshold offline from pools that contain real
failures, so it can be re-derived without ever re-running the simulator.

The statistic it uses is **excess** slip, not raw slip:

```
excess(t) = slip(t) - median_over_successful_demos(slip(t))
```

Because nominal slip grows monotonically through the trajectory, a fixed threshold on the raw
value is really a threshold on *time*. Subtracting the per-timestep median over successful
demos removes that trend. The threshold is then set at a **low false-positive rate on
successful demos** — the discipline eva_bc used for the grasp bit's 0 % FPR gate — because
censoring good data is the expensive mistake when the pool is the product.

The script can also return the verdict *"excess slip does not separate outcomes, do not censor
on it"*. That is a real possible answer, not a failure: a censor with no signal only deletes
good data. It refuses to calibrate against a pool with no failures at all.

### 3c. The verdict: do not censor — and the reason is instructive

Calibrated on 384 DART demos (13 real failures), AUC = P(a random **failure** scores higher
than a random success), so 0.5 is no signal and **below 0.5 means the statistic is a SUCCESS
indicator**:

| window | AUC raw | AUC excess |
|---|---|---|
| whole gripped span | **0.301** | 0.518 |
| CARRY only (grasp → push) | 0.419 | **0.637** |
| PUSH only (push → release) | 0.298 | 0.451 |

Raw slip over the whole span is at **0.301** — strongly inverted. The mechanism is specific and
was predictable in hindsight: the expert drives the block to `insert_x = 0.2545`, which puts it
**against the back stop**. Once it bottoms out, the gripper keeps executing its remaining push
waypoints while the block physically cannot move, so the in-hand offset grows several mm. That
growth is the signature of a *fully seated* insert. My "failure detector" was measuring a
deliberate and desirable part of the successful behaviour, and censoring on it would have
deleted precisely the best demos.

Restricting to the **carry** window — where the block should ride rigidly and any drift really
is expert error — flips the sign and gives 0.637. That is a weak signal in the *right*
direction, and it sits just under the `min_auc = 0.65` gate registered before looking. The gate
holds: **no censoring is applied.** Nudging it to 0.63 to get the hoped-for answer is exactly
the move the pre-registration exists to prevent.

**Why this is the right answer and not a disappointment.** Identical actions from an identical
state already flip 23–25 % of outcomes on this task (§6b). If most failures are simulator chaos
rather than expert error, then no trajectory-derived detector *can* separate them — the
recorded labels are correct and the episode was merely unlucky. Masking would delete good
supervision to no benefit.

This refines the answer to the masking question rather than dismissing it. The principle —
never train a policy to reproduce the expert's mistakes or to predict stochasticity — is right,
and the machinery is now in place and calibrated. It simply does not bind on *this* expert,
whose actions are a deterministic function of the block spawn (§6, TEST 2: plans are
bit-identical). It will bind on DAgger data, where a policy-driven prefix genuinely is a
mistake to be censored, and `build_train_mask` + the `slip_mm` signal are ready for it.

---

## 4. The noise experiments — the main scientific result

Full detail in `EXP_NOISE_SWEEP.md`. Both experiments were pre-registered with beliefs and a
decision rule written before running. **Six of eight beliefs were wrong**, which is the
argument for writing them down first.

### S2-N: noise everywhere

| `noise_std` | seated (n=128) | resets | grip at `push` | lateral p90 |
|---|---|---|---|---|
| 0.00 | 100.0 % | 0 | 128/128 | 1.21 mm |
| 0.01 | 88.3 % | 13 | 118/128 | 31.06 mm |
| 0.02 | 57.0 % | 44 | 96/128 | 147.87 mm |
| 0.04 | 28.9 % | 68 | 89/128 | 151.41 mm |
| 0.08 | 7.8 % | 78 | 79/128 | 147.35 mm |

**Grip retention per phase is the whole result**: 128/128 in lift, back, spin and turn at every
level up to 0.04, while `push` collapses. Peak arm deviation says the same — free-space phases
barely move (lift 33 → 60 mrad) while `push` goes 74 → 530 mrad, which is the arm fighting the
walls.

### Why: DART's correctness argument has a domain

DART is valid here because the action is a joint **position target**, so the expert's command
is correct from any nearby state — commanding `q_t` pulls the arm to `q_t` wherever it started.
**That holds only in free space.** Inside a 1.5 mm per-side channel, commanding the nominal
target from a laterally perturbed state does not pull the block back to the centreline; it
drives it *harder into a wall*. The controller has stiffness 2000 and does not yield, so the
block levers out of the pads. The recorded label is then not merely noisy — it is **wrong**.

No value of `noise_std` fixes that. The constraint is not "how much noise does the arm
tolerate" but "**where is the position-target label still correct**".

### S2-N2: noise in free-space phases only

| `noise_std = 0.02` | seated | resets | grip at `push` | lateral p90 | `push` peak dev |
|---|---|---|---|---|---|
| all phases | 57.0 % | 44 | 96/128 | 147.87 mm | 350 mrad |
| **free space only** | **96.9 %** | **0** | **128/128** | **1.07 mm** | **74 mrad** |

Same seed, same 128 spawns, same magnitude. `push` deviation returns to its no-noise baseline
of 74 mrad *exactly*. Diagnosis confirmed.

Full curve: 0.02 → 96.9 %, 0.05 → 92.2 %, 0.10 → 75.0 %, 0.20 → 15.6 %.

### A third mechanism at high noise

Not grasp misses (a belief refuted twice). The 30 mid-episode terminations at 0.10 and 101 at
0.20 are the **carried block hitting the fixture**. `stage_x` is 0.165 precisely because the
carried block rides ~5 mm ahead of the TCP against wall faces at x = 0.210 — so "free space" is
only free within ~40 mm, and a 50 mrad perturbation across six joints closes that margin.
**Free space is bounded, and its bound caps the usable noise.**

### Deliverable for Stage D

`HANDOFF.md` §9 requires this task's action-noise tolerance to be measured rather than
inherited. Pick-place's *healthy* σ ≈ 0.08 sits between our 75 % and 15.6 % cells. `sigma_init`
should start at **≤ 0.05**, not the inherited −2.5.

---

## 5. The port

`act/` → `slot/slot_act/`, 18 files. Constants: `OBS_DIM 34`, `ENV_STATE_DIM 18`,
`BIT_DIMS[-1] 33`, `RES_OBS_DIM` **58**, `STEER_OBS_DIM` **50**.

Four things worth recording:

**5a. The package had to be renamed.** Keeping the name `act` is a trap — both directories are
importable as `act` and `sys.path` order decides. Measured:

```
cd /tmp && PYTHONPATH=.../eva_bc python -c "import act, act.dataset as D; print(D.OBS_DIM)"
41
```

No exception, no warning — the 41-D pick-place constants silently bound to 34-D slot data. My
first mitigation was a guard *inside* the copy, which is useless: in the failing case the copy
is never imported at all. Renaming removes the ambiguity instead of detecting it.

**5b. The edit a rename cannot make.** `residual_core.py` contained `obs41[:, 40]` — a **bare
integer** for the commanded-grip channel, invisible to any `obs41 → obs34` substitution and
out-of-bounds at 34. Now `obs34[:, BIT_DIMS[-1]]`, which cannot drift from `OBS_DIM`. That
channel is required: the four physical finger joints alone score a *better* AUC (0.976 vs
0.968) but **27.1 % false-positive rate**, against 0 % with the command included.

**5c. Two deliberate design departures.**
* The goal-delta tail is **4-D, not 3-D** (hence 58/50, not the 57/49 the port map predicted).
  The env's own `lateral_error` and `yaw_error` are **absolute values** — they tell the policy
  how wrong it is, not which way to correct. The signed versions are supplied, plus yaw,
  because a horizontal insertion into a 1.5 mm channel fails on yaw in a way a top-down drop
  into a basket never did.
* The flush rule's **z-drop half is removed**. It was a slip proxy sized against a 40 mm basket
  rim; here the block legitimately descends ~8 mm the instant the fingers open, so the
  pick-place rule would discard the committed action chunk at the most precision-critical
  moment. Chunk commitment is load-bearing — shortening the horizon collapsed success
  59.4 → 32.8 → 3.1 → 0 %.

**5d. `slot_mdp.placed_mask` wraps the seated predicate, not `is_inserted`.** `eval_act.py`
reduces this exact call into the headline success number, so this one line decides every
evaluation figure for the rest of the project. The env's bare predicate bounds block height
only from below and also passes for a block on the wall tops or dangling in a closed gripper.
`basket_centers_local` is deliberately **absent** so an unported call site raises rather than
silently receiving a constant.

`check_port.py` asserts the dimension identities, that `slot_act` did not resolve to the
tracked original, and that no pick-place task id survives. **It caught a stale
`Rebot-PickPlace-Play-v1` on its first run.**

---

## 6. The finding that changes how every number must be read

### 6a. The first executed episode in a process is systematically better — by 12.9 points

Same plan, bit-identical initial state (`|q_start − q_default| = 0.00e+00` on every run),
n=128, three executions in one process, **with run 0 genuinely the first episode executed**:

```
run 0  (FIRST in process): seated 128/128 = 100.0%   depth mean 46.74 mm  min 43.71
run 1                    : seated 114/128 =  89.1%   depth mean 45.41 mm  min 30.39
run 2                    : seated 109/128 =  85.2%   depth mean 44.89 mm  min 28.93

first execution 100.0%  vs  later mean 87.1%   ->  +12.9 pts,  binomial SE 3.0  ->  4.3 sigma
```

The first episode is not merely luckier on average — its depth distribution is *tighter*
(min 43.71 mm against a 40 mm threshold, versus 28.9–30.4 mm later). And the later runs' failure
sets are near-disjoint: 14 flips vs run 0, 19 flips vs run 0, but **31** between runs 1 and 2,
implying an overlap of about 1. So after the first episode, ~13 % of envs fail essentially at
random and independently each time.

This explains every "128/128" in this project. They were all first-episode measurements —
`run_expert.py` executes exactly one batch per process, as did rollout 0 of every collection
run. Nothing was faked; the number is real *for that condition* and does not generalise.

**Two operational consequences, and the second is the dangerous one:**

1. **Demo collection**: use one rollout per process. The first batch yields ~100 %.
2. **Evaluation**: a policy evaluated on its first batch of episodes will be
   **optimistically biased by roughly 13 points**. `eval_act.py` collects episodes until it has
   `--episodes` of them, with envs auto-resetting — so the first `num_envs` records are all
   first-episode records. An eval of 128 episodes on 128 envs is *entirely* first-episode and
   maximally biased; an eval of 128 episodes on 16 envs is mostly not. **Eval configuration
   silently changes the number.** Comparisons between arms stay valid only if every arm is
   evaluated with identical `num_envs` and `--episodes`, and absolute figures must be read as
   an upper bound.

The mechanism is presumably simulator state that `env.reset()` teleports bodies past without
flushing — PhysX contact manifolds and solver warm-start caches, populated by an episode full
of grasping and of pushing the block into the back stop. It is policy-independent, so it will
apply to trained policies exactly as it does to the scripted expert.

### 6b. Beyond the first episode, execution is not reproducible at all

Executing the **same plan** from a **bit-identical** initial state, three times in one process
(n=32, `|q_start − q_default| = 0.00e+00` on every run):

```
run 0: seated 29/32   depth mean 44.94 mm   min 30.70
run 1: seated 30/32   depth mean 45.69 mm   min 33.18
run 2: seated 24/32   depth mean 43.73 mm   min 31.49

run 1 vs run 0:  max |depth diff| 15.940 mm   mean 2.427 mm   outcome flips 5/32
run 2 vs run 0:  max |depth diff| 15.634 mm   mean 4.053 mm   outcome flips 9/32
run 2 vs run 1:  max |depth diff| 15.388 mm   mean 3.488 mm   outcome flips 8/32
```

**The simulation is not reproducible.** Identical actions and identical initial state put the
block up to 16 mm apart, and flip 15–28 % of outcomes. This is expected of GPU PhysX — parallel
contact solving is order-dependent — but its *magnitude* here is large because the task lives
on a 40 mm depth threshold that many spawns sit close to.

Consequences, all of which matter more than the rollout puzzle that led me here:

1. **The expert's "100 %" was one sample from a wide distribution.** Its honest rate is
   ~85–95 %; pooled over 4 rollouts × 128 it is **90.2 %** (nominal), 86.7 % (DART-0.02),
   82.4 % (DART-0.05).
2. **Any A/B comparison on this task needs pooling and repeats.** eva_bc's protocol already
   demanded ≥3 seeds and ≥128 pooled episodes; this is now a task-specific measurement of *why*,
   not an inherited rule.
3. **The action labels are still clean.** `plan_determinism.py` showed the plan is a **pure
   function of the block pose** — bit-identical (0.0000 mrad on all six phases) both
   back-to-back and after a full 599-step episode, and `fk()` is a pure function of its joint
   vector (0.000 µm). So the stochasticity is entirely in the *outcome*, not in the
   observation→action mapping the policy is trained on.
4. **Therefore success-filtering the nominal pool selects on luck, not on behaviour.** Since the
   actions are a deterministic function of the spawn and no spawn bin predicts failure, dropping
   the ~10 % failed demos removes them essentially at random rather than removing bad behaviour.
   This is a strong argument for the in-hand-slip censor of §3a, which detects the frames where
   the block actually went wrong irrespective of the final outcome.

---

## 7. What I got wrong, and how

Recorded in full because the failure mode is more useful than the conclusion.

### 7a. I claimed the planner was state-dependent. It is not.

Four rollouts in one process gave 128, 112, 108, 114 seated. Rollout 0 was better in all three
pools collected (nominal, DART-0.02, DART-0.05). Hunting the cause, I found that the
commanded joint target at the end of `push` — which is **spawn-independent by construction** —
had means that increased monotonically with rollout index:

```
j2:  1.855540 → 1.855945 → 1.878258 → 1.889880      "17 mrad drift"
```

I reported this as the diagnosis. **It is not significant.** The within-rollout spread of that
same quantity is 0.27 action units, so each mean carries a standard error of 0.024:

```
rollout3 − rollout0 = 0.034340 ± 0.030684 (SE)  ->  1.12 sigma
Welch t-test p = 0.266        ANOVA across the four rollouts p = 0.645
```

I compared four means that happened to be increasing without dividing the spread by √n —
**having printed that spread in the same table**. Four monotone points out of 4! = 24 orderings
is a 1-in-12 coincidence, not evidence.

`plan_determinism.py` then refuted it directly: the plan is bit-identical when computed twice
back-to-back *and* after a full episode has been executed.

**The lesson is specific:** a monotone sequence of four means is the single most seductive
pattern in a small table, and this project's own history is full of exactly this failure —
three of four hypotheses for the earlier grip-loss bug were about dynamics because I had
convinced myself the geometry was clear. *Compute the standard error before naming a drift.*

### 7b. Two of my own reported numbers contradicted themselves

Neither produced a crash; both were caught by noticing that two printed quantities could not
both be true.

* **Grip statistics counted envs that had already reset.** Isaac Lab hands a reset env a fresh
  scene, so its finger gap is measured on a *new* episode and reads as a healthy grip. At
  `noise_std = 0.10` this printed `grip held 126/128` directly beside `lateral p90 131 mm`.
  Fixed with `& ~resets`.
* **The headline success rate was conditional on not resetting.** `98.0 %` was printed where
  the expert's true score on the blocks it was given was **75.0 %**. Both rates are now printed,
  with the unconditional one labelled as the real score.

### 7c. Beliefs refuted across the two pre-registered noise experiments

| experiment | belief | outcome |
|---|---|---|
| S2-N | success monotone in noise | ✅ confirmed |
| S2-N | failure is precision, grip intact | ❌ it is contact; the grip is lost |
| S2-N | 0.02 tolerable at ≥ 95 % | ❌ 57.0 % |
| S2-N | achieved deviation ≪ commanded | ❌ mis-framed; the peak is dominated by systematic tracking lag and cannot resolve the noise at all |
| S2-N2 | ≥ 95 % at 0.05, resets ~0 | ~ 92.2 %, resets 2 (the reset collapse was the load-bearing half, and it held) |
| S2-N2 | high-noise failure becomes grasp misses | ❌ it is fixture collision while carrying |
| S2-N2 | deviation scales with `noise_std` | ✅ confirmed |

### 7d. I nearly missed the real effect by ordering my own test wrongly

The first version of `plan_determinism.py` ran its repeated-execution test **last**, after the
planning tests had already executed a full episode. Every one of its three "repeats" was
therefore a *later* episode, and it reported 114 / 109 / 109 seated — which I read as "run 0 is
not special, the rollout effect needs another explanation".

The hypothesis under test was literally *"the first executed episode in a process behaves
differently"*, and the test had no first episode in it. Moving the block to the top of `main()`
— changing nothing else — produced 128 / 114 / 109 and a 4.3 σ effect (§6a).

**The lesson:** when a test targets a *position-in-sequence* hypothesis, the position of the
test within the script is part of the experiment, not incidental setup. A null result from a
test that cannot express the hypothesis is not a null result.

### 7e. Resolved

The rollout-0 advantage is now explained and quantified: **+12.9 points, 4.3 σ**, caused by
being the first executed episode in a process, reproduced under an identical plan and a
bit-identical initial state. See §6a for the consequences — the eval-bias one matters more than
the collection one.

---

## 8. Data on disk

`slot/data/v2/` — one rollout per process, slip signal stored, censor calibrated offline.
Every file passes `scripts/verify_demos.py`.

| pool | demos | successful | rate |
|---|---|---|---|
| `nominal_s0..s3` | 512 | 512 | **100.0 %** |
| `dart002_s10..s13` | 512 | 492 | 96.1 % |
| `dart005_s20..s23` | 502 | 462 | 92.0 % |
| **total** | **1526** | **1466** | **96.1 %** |

**914,074 frames** (T = 599 each). Spawn coverage spans **100 % of the reset range** on all
three randomised axes: x [0.200, 0.240], y [−0.160, −0.100], yaw [−0.349, +0.350] rad.

Four further nominal pools (`nominal_s4..s7`) are collected to give arm A of `EXP_BC_ARMS.md`
1024 nominal demos, matching arm B's 1024 mixed — the arms differ in *composition* only, never
in volume.

Two results settle earlier confusion:

* **The nominal expert is 512/512 = 100 %** across four independent seeds. The 90.2 % reported
  mid-session was the first-episode bias contaminating a four-rollout process (§6a). Both
  numbers were real measurements of different things, and the ambiguity was invisible until the
  bias was found.
* **The noise sweep generalises across seeds.** DART rates at fresh seeds (96.1 % at 0.02,
  92.0 % at 0.05) track the sweep's single-cell predictions (96.9 %, 92.2 %), so the curve in
  §4 was not fitted to one spawn draw.

The superseded v1 pools (`slot/data/pool_*.hdf5`, collected 4 rollouts per process, no slip
signal) are kept only for the analyses in §7 and must not be trained on.

## 9. Next

1. Re-collect the three pools: one rollout per process, slip censoring on.
2. Validate `eval_act.py` end-to-end on a throwaway checkpoint **before** training — verify the
   instrument before generating what it measures.
3. Train flow-BC: Arm A (nominal only) vs Arm B (nominal + DART), ≥3 seeds each, champion on a
   held-out spawn seed, pooled ≥128-episode eval. Targets: ≥55 % Loose, ≥40 % v0.
4. Stage D (x0-steering) with `sigma_init ≤ 0.05` per §4.
5. Stage E: the env's Factory reward has never been run — a cheap PPO-from-scratch control.
