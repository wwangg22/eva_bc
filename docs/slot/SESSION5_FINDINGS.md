# Session 5 — upstream review, Stage C first results, and what they change

*2026-08-03, 01:00–onward. Written while the Stage C sweep runs. Every number here is
reproducible from a file in `slot/runs/` or `slot/data/v2/`; commands are given inline.*

This session opened with two instructions: pull and review the upstream changes in both
repos, and keep executing the plan. Both happened, and the second produced a result that
reorders the remaining work — so read §3 before §5.

---

## 0. Repo state after the pull

Both repos were behind. Both fast-forwarded cleanly; **no tracked file in either repo was
modified by me**, and my untracked `slot/` + `docs/slot/` came through untouched.

| repo | was | now | commits in |
|---|---|---|---|
| `eva_rl` | `05f0fb3` | `e56e7df` | 1 |
| `eva_bc` | `1c04eca` | `818391b` | 4 |

Verification (both clean, only my untracked dirs):

```
git -C /home/rei/Desktop/isaaclab/eva_rl status -sb
git -C /home/rei/Desktop/isaaclab/eva_bc status -sb
```

### 0a. `eva_rl` — wrist camera tilt (does not touch this task)

One commit, two files: a new `scripts/wrist_cam_tilt_sweep.py` and a 10-line change to
`source/reBot_RL/reBot_RL/tasks/manager_based/lift/camera_cfg.py` re-aiming the D405 wrist
mount by −30° about camera-local X.

I checked this before pulling rather than after, because a change to the robot asset would
have invalidated 2038 demos. It does not:

* the change is to a **`CameraCfg.OffsetCfg`** — a sensor pose. Cameras carry no mass,
  inertia, or collider, so no dynamics change.
* it lives in `lift/` (the pick-place task tree). The slot task is
  `challenge/precision_slot_env_cfg.py` and does not import it.
* this task's observation is 34-D privileged state. There is no camera in the loop at all.

**Conclusion: demos and checkpoints remain valid.** Recorded because "an unrelated repo
changed under me" is exactly the kind of thing that silently invalidates a result later.

### 0b. `eva_bc` — EXP07 closed, and it succeeded

Four commits, all pick-place: `EXP07_x0_steering.md`, `POSTMORTEM.md` §9,
`JOURNAL.md`, `experiments/exp07_analyze_s1.py`, video flags on `act/eval_steer.py`, and
the retirement of `sync_from_source.sh`. Reviewed in full — see §2.

---

## 1. Stage C is running; the first real numbers are in

Six runs (2 arms × 3 training seeds × 100k steps) launched at 00:27. Run 1 (arm A, seed 0)
finished at 01:11; arm B seed 0 is training now.

While it trains, a learning-curve chain evaluates arm A seed 0 at intermediate
checkpoints on `Rebot-PrecisionSlot-v0`, spawn seed 777, 128 episodes / 32 envs.
**All rates below are `success_rate_later`** — the first episode each env runs is excluded,
because PhysX contact/solver caches surviving `env.reset()` give it a measured bias.

| train steps | later k/n | later rate | first-ep rate | first-episode bias |
|---|---|---|---|---|
| 2 000 | — | 0.375 | 0.562 | +18.7 pts |
| 10 000 | 59/96 | 0.615 | 0.625 | +1.0 pts |
| 30 000 | 86/96 | 0.896 | 1.000 | +10.4 pts |
| 50 000 | 82/96 | 0.854 | 0.969 | +11.5 pts |
| 100 000 (`ckpt_final`) | 89/96 | **0.927** | 0.906 | **−2.1 pts** |

95 % Wilson intervals: 86/96 → [0.819, 0.942]; **89/96 → [0.856, 0.964]**.

**Longer training is better, not worse.** `ckpt_final` (92.7 %) beats `ckpt_0030000`
(89.6 %), so the overfitting concern §1b was written to guard against does not materialise,
and the pre-registered checkpoint happens to also be the best one — no tension between the
rule and the result. Both JSONs pass `check_eval_json.py` 14/14 (§6a).

**The first-episode bias went negative.** +18.7 pts on the 2000-step model, +1.0 at 10k,
+10.4 at 30k, **−2.1 at 100k**. It is not a constant, not monotone, and not always positive —
which is precisely why the cohort split must keep being *reported* rather than assumed small
and dropped. A single pooled `success_rate` would have been wrong by between −2 and +19 points
depending only on which checkpoint you happened to be looking at.

### 1a-i. The curve plateaus at 30k — established by pairing, not by eyeballing CIs

The rates wobble 89.6 → 85.4 → 92.7 over 30k/50k/100k and the Wilson intervals all overlap,
which on its own proves nothing. But these evals share spawn seed 777, so where the recorded
`spawn_pos` confirms the same spawn landed in the same episode slot, the comparison is *paired*
and McNemar applies — far more powerful than comparing two independent proportions, because
concordant episodes carry no information about the difference and get correctly discarded.

```
python analysis/paired_evals.py 'runs/bc_armA_seed0/eval_ckpt_*.json'
```

| comparison | agree | discordant (earlier / later) | McNemar χ² (1 df) | verdict |
|---|---|---|---|---|
| 10k vs 30k | 57 | 6 / 33 | 17.333 | real, **but see caveat** |
| 10k vs 50k | 51 | 11 / 34 | 10.756 | real, **but see caveat** |
| 10k vs 100k | 60 | 3 / 33 | 23.361 | real, **but see caveat** |
| **30k vs 50k** | 74 | 13 / 9 | 0.409 | p > 0.05 — noise |
| **30k vs 100k** | 81 | 6 / 9 | 0.267 | p > 0.05 — noise |
| **50k vs 100k** | 79 | 5 / 12 | 2.118 | p > 0.05 — noise |

**Caveat, caught by the tool and not by me.** I first ran this comparison ad hoc, and it
reported "spawns identical = True" for all six pairs. That was wrong: `eval_ckpt_0010000` was
written *before* `spawn_pos` was added to the eval schema, and my throwaway check treated a
missing field as agreement. `paired_evals.py` distinguishes *matching* from *unverifiable* and
flags the three 10k rows as **not verifiably paired**, so their χ² is not sound.

The conclusions survive the correction, by different routes:

* **10k → 30k is real** — not by McNemar, but because the Wilson intervals are disjoint:
  [0.515, 0.706] against [0.819, 0.942]. No pairing needed for a gap that size.
* **Nothing after 30k is measurable** — this uses only the 30k / 50k / 100k rows, which *are*
  verifiably paired, and it is the load-bearing claim. 30k and 100k agree on **81 of 96**
  episodes and disagree 6 one way, 9 the other.

The lesson is the one this project keeps relearning: a hand-rolled check written to confirm
something tends to confirm it. The instrument that distinguishes "verified" from "unknown"
earns its file.

Practical consequence for the *next* sweep, not this one: **30k steps per run would have been
enough**, turning ~35 min per run into ~12 and the 6-run sweep from ~3.5 h into ~1.2 h. This
sweep does not need changing — `--save-every 10000` already wrote `ckpt_0030000` for every run,
so the cheap-training question stays answerable from artifacts that exist.

A useful side result: the pairing held perfectly across four different checkpoints, which is
evidence that `summarize_arms.py`'s pairing check will find the arms paired too. It is still
*checked* there rather than assumed — different weights could in principle desync the reset
stream through the policy's own `torch.randn` draws for x0 — but with `flushes = 0` and every
env refilling in lockstep every 15 steps, the RNG consumption pattern is identical by
construction.

### 1a. The headline

> **⚠ SUPERSEDED BY §6d — READ THAT FIRST.** 92.7 % is `bc_armA_seed0`, the best of three
> training seeds, and training-seed variance turned out to be **15–29 points**. The honest
> pooled figures over 3 training seeds × 2 spawn seeds are **A 0.776 / B 0.792 on `-v0`**, and
> the objective is cleared on **all three clearances** — a weaker headline and a much stronger
> claim. The text below is left as written because the reasoning it triggered is what found the
> problem.

> **The 70 % target is cleared on `Rebot-PrecisionSlot-v0` by behaviour cloning alone —
> `92.7 %` at `ckpt_final`, and already `89.6 %` after 30 000 steps (≈12 minutes of GPU).
> No RL, no DAgger, no HG-DAgger.**

The pre-registered Stage C bar was 40 %. It was cleared at 10k steps. 92.7 % is more than
twice the bar and **22.7 points above the project target**, with the 95 % interval
[0.856, 0.964] entirely above it.

Two caveats stated up front, not buried:

1. **One spawn seed, one task, one training seed.** 777 / `-v0` / seed 0. The sweep
   is collecting `-Loose-v0`, `-v0` and `-Tight-v0` × spawn seeds {777, 888} × 6 runs
   precisely so this number can be checked rather than believed.
2. **This is not yet the arm comparison.** Arm A (nominal only) is the only arm with
   results. Arm B (nominal + DART) has not been evaluated. Nothing about the value of
   DART data can be read from the table above.

### 1b. Checkpoint selection — pre-registered NOW, before arm B is seen

The learning curve creates a trap: `ckpt_0030000` scoring 89.6 % invites a post-hoc
"pick the best checkpoint per arm", which inflates every subsequent number by the
selection it hides.

**Pre-registered, written before any arm-B result exists:**

* The **arm comparison uses `ckpt_final`** (100k steps), exactly as fixed in
  `EXP_BC_ARMS.md` before training started. It is not renegotiated.
* The learning curve is a **separate descriptive result** about arm A seed 0 alone.
* If `ckpt_final` scores materially below `ckpt_0030000`, that is a finding about
  **training length** (overfitting a 1977-demo pool), reported as its own result and
  confirmed on the held-out spawn seed 888 before it is acted on. It does not
  retroactively become the checkpoint-selection rule.

No run needs to be killed or shortened to honour this: `--save-every 10000` means every
run already keeps `ckpt_0030000` on disk, so the training-length question is answerable
later at zero extra training cost.

---

## 2. EXP07 (upstream): x0-steering works on pick-place — 55.5 % → 91.4 %

The sibling task's Stage-D-equivalent closed successfully while this session was running.
Because `slot/slot_act/` already carries the full steering stack, this is directly my
Stage D recipe, and it is now proven rather than speculative.

**What it did.** Freeze the flow base. Train PPO to emit `z ∈ R⁷` once per 15-step
execution window; the flow integration's starting noise becomes `x0 = α·tanh(z)`,
broadcast over all 50 chunk positions, `α = 1.0`. The base decodes from there.

**Why it beat the additive residual** (`POSTMORTEM.md` §9b–9c, distilled):

* the base's failures were **mode errors, not aim errors** — the decoder committing to
  the wrong *chunk family* for that state. A per-step additive offset translates a wrong
  plan; it cannot re-select it, so it fixed marginal misses and broke marginal successes
  symmetrically (26/26) and PPO correctly converged to doing nothing.
* steering is **on-manifold by construction**: every `z` yields a chunk the base itself
  would emit, so `z ≈ 0` (the best blind mode) stays available per state and PPO only
  departs where the expected gain is positive. Measured: **51 fixed / 5 broken**.
* credit assignment lands **where the failure happens** — one decision per window.

**Process facts worth copying, not just the result:** a bit-exactness gate (`z = 0` must
reproduce the frozen base episode-for-episode) before any training; an exploration-response
measurement (gate S0) that *chose* `σ_init` from data rather than precedent; train protocol
= eval protocol; and `clip_actions: 1.0`, which is mandatory because rl_games rescales
sampled actions to the action-space bounds (the default 100 destroyed two EXP06 runs).

### 2a. What transfers to slot — and the one big thing that does not

EXP07's single largest win was the **never-lifted bucket collapsing 18/19**: episodes where
the base committed to a bad grasp and never retried. §4 below shows that **this failure mode
does not exist on the slot task** — the expert grasps 2038/2038 and every failure is at
release. So the headline mechanism behind +35.9 pts has no counterpart here, and I should
**not** expect a comparable jump.

What *should* transfer is the part that matches this task's actual failure: re-choosing the
chunk family at the approach/push windows, where §5 shows the failures live.

---

## 3. The expert's failures are all one failure

Across all 16 pools, 2038 demos, the 61 failures have **exactly one** signature:

```
n= 61  reach:grasp:g0=grasped | lift=held | back=held | spin=held
       | turn=held | push=held | release=unseated
```

Not "mostly". All 61. **Zero** grasp failures, zero drops, zero topples, zero carry losses.

Reproduce:

```
python - <<'EOF'
import h5py, glob, json
from collections import Counter
c = Counter()
for p in sorted(glob.glob('slot/data/v2/*.hdf5')):
    g = h5py.File(p, 'r')['data']
    for k in g:
        d = g[k]
        if bool(d.attrs.get('success', False)):
            continue
        oc = json.loads(d.attrs['outcomes'])
        c[tuple((s, v.get('outcome')) for s, v in oc.items())] += 1
for sig, n in c.most_common():
    print(n, ' | '.join(f'{s}={o}' for s, o in sig))
EOF
```

**Consequences.**

1. The whole difficulty of this task is concentrated in the last ~135 control steps
   (`push` starts at t≈358, `release` at t≈465). Everything before that is solved.
2. A "did the grasp succeed" feature — the centrepiece of the ported steering observation —
   is measuring a quantity that is constant across the entire dataset.
3. The BC policy's failure distribution is **not** required to match the expert's. A cloned
   policy can fail in ways its demonstrator never did. §5 measures it directly rather than
   assuming; it turns out to agree.

---

## 4. The finger channels carry no grasp information (measured)

The ported grasp bit (`residual_core.GraspBit`) is an MLP over
`BIT_DIMS = [6, 7, 14, 15, 33]` — two finger positions, two finger velocities, and the
commanded grip channel. On pick-place it scored AUC 0.976 with 0 % false-positive rate.

On slot data the finger channels are **degenerate**. Over 707 802 frames where the block is
provably held (grip commanded closed *and* block z > 50 mm):

| cohort | n frames | mean Σ finger pos | sd |
|---|---|---|---|
| held (closed, block z > 50 mm) | 707 802 | −0.04889 | 0.00003 |
| not-held-ish (closed, block z < 40 mm) | 1 118 | −0.04887 | 0.00005 |

A **2 × 10⁻⁵ difference against a 3 × 10⁻⁵ spread**. The distributions are the same
distribution. Inspecting a single episode explains it: the finger joints are bimodal and
saturated — `+0.005` open, `−0.0246` closed — and they reach the closed value whether or
not a 30 mm block is between them. The fingers are not stopped by the block, so their
position cannot report the block.

(The "not-held" cohort is also weak — §3 shows there are no genuine closed-on-air frames in
the dataset to draw from. That weakness does not rescue the MLP: with no negative class in
2038 demos, the bit is not merely low-signal, it is **untrainable from this data**.)

### 4a. Decision: analytic grasp bit, and a pre-registered escalation

`experiments/exp06_grasp_bit.pt` **does not exist in either repo** — only the training
script `experiments/exp06_grasp_bit.py` was ever committed. Six call sites in `slot_act/`
point at the missing artifact, so `train_steer.py` would have died at line 70 on startup,
after paying Isaac Sim's boot cost. That is the blocker this section removes.

Given §4's measurement, retraining the MLP is not an option (no signal, no negative class).
The replacement is analytic and uses the env's **own authored predicate**:

```
grasp_bit = mdp.block_lifted(minimal_height=0.045, ee_max_dist=0.08) AND (obs[:, 33] < 0)
```

* `block_lifted` is `precision_slot_env_cfg`'s own reward term — block off the table *and*
  within 80 mm of the gripper. Using it means the feature cannot disagree with the env.
* the `obs[:,33] < 0` conjunct is EXP06's measured-decisive channel (it took false
  positives from 27.1 % to 0 % there). Verified for slot below.
* no artifact, no training, no new threshold invented by me.

**Sign conventions verified against real demo data, not assumed** (`data/v2/nominal_s0.hdf5`,
`demo_0`): `actions[:,6]` transitions at steps **40** and **464**; `−1 = closed`,
`+1 = open`; and `obs[t,33] == actions[t−1,6]` with max abs difference **0.0** over the
episode. Block z: 0.035 at spawn, 0.032 while being grasped, ~0.061 carried, 0.055 seated.

**Known property, stated rather than hidden:** between the gripper closing (t≈40) and the
block clearing 45 mm (t≈85) the bit reads 0 although the block *is* held — a ~45-step dead
zone. This is honest rather than harmful: "closed on the block but still on the table" is
genuinely not a completed grasp, and §3 shows the grasp phase is not where this task fails.

**Pre-registered escalation.** If Stage D underperforms, escalation #1 is to replace this
bit with the env's `engaged` predicate (`insertion_depth ≥ 10 mm ∧ lateral_error ≤ 30 mm`),
because §5 shows the failures are at the slot mouth, not the grasp. Registering it now means
that swap is a planned test rather than a post-hoc fit.

---

## 5. Why the BC policy fails: it jams at the slot mouth

Ten failures among the 96 later-episodes at `ckpt_0030000`. Sorted by insertion depth
(success needs ≥ 40 mm):

| ep | depth mm | lateral mm | yaw rad | env's raw `is_inserted` |
|---|---|---|---|---|
| 64 | −51.8 | 1.76 | 0.008 | 0 |
| 92 | −46.8 | 1.42 | 0.000 | 0 |
| 45 | −31.6 | 1.82 | 0.005 | 0 |
| 121 | −31.5 | **11.03** | 0.030 | 0 |
| 109 | −20.9 | 1.90 | 0.011 | 0 |
| 107 | 5.4 | **0.04** | 0.001 | 0 |
| 47 | 35.0 | 1.42 | 0.003 | 0 |
| 105 | 37.3 | 0.89 | 0.013 | 0 |
| 86 | 44.8 | 0.01 | 0.005 | **1** |
| 65 | 49.8 | 0.71 | 0.008 | **1** |

**Yaw is not a failure mode.** Every failure has |yaw| ≤ 0.030 rad against a 0.12 rad
tolerance. The `SUCCESS_YAW` guard is never the binding constraint.

**Lateral is, and it binds exactly at the clearance.** Successes: median 0.82 mm,
p90 1.37 mm, **max 1.50 mm**. The per-side clearance on `-v0` is **1.5 mm**. Five of the
seven aligned failures sit at 1.42–1.90 mm — i.e. just outside — and their depth is
strongly negative, meaning the block never entered at all. That is a **jam at the slot
mouth**: lateral error consumes the clearance, the block stops on the lip, and the push
phase drives it nowhere.

Two failures are a different mechanism and should not be lumped in: ep 107 is aligned to
**0.04 mm** and still only reached 5.4 mm depth (a push/timing failure, not an aiming one),
and ep 121 at 11.03 mm is a gross misalignment unlike anything else in the set.

Two more (86, 65) satisfy the env's bare `is_inserted` but are rejected by the seated-height
guard in `slot_mdp.placed_mask` — the guard added in Stage A after a probe cell scored
93.8 % with 13.28 mm mean lateral error. It is still earning its place: without it these
numbers would read 88/96 rather than 86/96.

### 5a-i. ⚠ The mouth-jam story is a `ckpt_0030000` story. It does not survive to `ckpt_final`

`analysis/failure_taxonomy.py` (written after §5, and the reason this correction exists) sorts
failures into mechanism buckets by a decision list. Run across the learning curve on `-v0`,
spawn 777, later cohort:

| checkpoint | fails | never_lifted | gross_miss | never_entered | stalled_in_mouth | seat_reject |
|---|---|---|---|---|---|---|
| 10 000 | 37 | 2 | 8 | 5 | 21 | 1 |
| 30 000 | 10 | 0 | 0 | 5 | 3 | 2 |
| 100 000 (`final`) | 7 | 0 | **3** | 1 | 3 | 0 |

`gross_miss` = the block ended more than one **block half-width** (15 mm, `mdp.BLOCK_HALF[1]`)
off the slot axis — a geometric threshold, not a tuned one. Splitting it out is what exposed the
error: conflated into `never_entered`, a bucket whose median |lateral| is 1.82 mm at 30k and
30.12 mm at 100k reads as one mechanism when it is two.

**What actually happens as training proceeds:**

* the **marginal-lateral jams vanish**. At 30k, five failures sat at 1.42–1.90 mm against a
  1.5 mm clearance with strongly negative depth — the mouth jam §5 describes. At `ckpt_final`
  there are **zero** such episodes.
* what survives is two unrelated populations: **3 gross transport failures**
  (|lateral| 23.0, 30.1, 73.4 mm — the block is nowhere near the slot) and **4 depth failures
  with fine lateral alignment** (|lateral| ≤ 1.16 mm, depth 23.0–40.0 mm, one of them at exactly
  the 40.0 mm threshold).
* `never_lifted` and `seat_reject` both empty out. The policy gets cleaner in every respect
  *except* a residual ~3 % of gross transport failures.

So **§5's "the failures are a jam at the slot mouth" is true of the 30k checkpoint and false of
the champion.** The correction matters because the two point at different fixes: a mouth jam
asks for sub-millimetre lateral centring; a gross miss asks why the block is being carried to
the wrong place at all; a depth shortfall asks about the push.

**Stated with its uncertainty:** this is 7 failures. A 3-of-7 split has an enormous interval and
I am not going to build on it. The sweep produces 36 evals (6 runs × 3 tasks × 2 spawn seeds),
which pooled with `failure_taxonomy.py --pool` gives a properly powered version of this table.
Treat the above as the hypothesis that pooling will test.

### 5a. The distribution says the clearance is doing the aligning

|lateral| over the 86 successes is close to uniform on [0, 1.5]:

| p25 | p50 | p75 | p90 | p95 | max |
|---|---|---|---|---|---|
| 0.540 | 0.825 | 1.157 | 1.370 | 1.425 | 1.500 |

A policy that actually *aimed* at the slot centre would pile up near 0 with a thin tail.
This fills the envelope right to the wall. The policy is not centring the block — it is
placing it somewhere inside a 3 mm-wide window and letting the geometry accept it.

### 5b. Pre-registered prediction for the clearance ladder

Written **before** the `-Loose-v0` / `-Tight-v0` evaluations in the sweep exist. Model: an
episode succeeds on a channel of half-clearance *c* if its final |lateral| < *c*.

| task | clearance | predicted later-rate |
|---|---|---|
| `-Loose-v0` | 3.0 mm | **0.90** |
| `-v0` | 1.5 mm | 0.885 *(measured 0.896 — the model is calibrated where it can be checked)* |
| `-Tight-v0` | 0.5 mm | **0.19** |

This is an **upper bound**, and the direction of the error is knowable in advance: on a
tighter channel the block that measures 1.2 mm off on `-v0` does not land at 1.2 mm, it
**jams earlier**, and the push phase then has less room to recover. So the true Tight number
should be **at or below 0.19**.

Falsification is what makes this worth writing: if Tight comes back near 0.9, the
lateral-envelope model is wrong and the policy is doing closed-loop correction I have not
detected — which would itself be the most important finding of the session.

### 5c. ⚠ A flaw in my own model, found before the data arrived — the measurement is censored

Kept in place rather than rewritten, because the flawed reasoning is the instructive part.

The |lateral| I am extrapolating from is measured **after the block is seated**, i.e. after the
channel walls have physically constrained it. It is a *censored* quantity, not a free one: the
distribution cannot exceed the clearance because the geometry forbids it. Its p100 being
exactly 1.500 mm on a 1.5 mm channel is not a coincidence — it is the censoring, visible.

The expert makes this unmistakable. It runs the *same open-loop trajectory* on all three
clearances, yet its measured lateral p90 falls with the channel:

| task | clearance | expert seated | expert lateral p90 |
|---|---|---|---|
| `-Loose-v0` | 3.0 mm | 128/128 | 1.38 mm |
| `-v0` | 1.5 mm | 128/128 | 1.11 mm |
| `-Tight-v0` | 0.5 mm | **128/128** | **0.48 mm** |

One trajectory, three different "errors". The number is reporting the *channel*, not the
policy. So §5b's CDF extrapolation is confounded, and the honest statement is weaker than what
I wrote: **the ~0.19 figure is not a calibrated prediction, it is a worst case under an
assumption I can now show is violated.** The real quantity that decides Tight is the
*pre-insertion* alignment at the slot mouth, which the eval does not currently record.

The prediction stays on the record and the sweep still tests it. But if Tight lands well above
0.19, the correct conclusion is "the censored-measurement model was wrong", **not** "the policy
turned out to be closed-loop" — those are different claims and §5b conflated them.

### 5c-ii. The censoring test, run properly — and a partial retraction of §5c

§5b extrapolated a Tight number from the `-v0` |lateral| CDF. §5c called that confounded because
the quantity is censored by the walls. **Both statements were reasoning; neither was a
measurement.** The measurement is available the moment the sweep's `-Loose-v0` cell lands: run
*the same policy* on two clearances and see whether the distribution moves.

If |lateral| were purely censored, doubling the channel 1.5 → 3.0 mm should roughly double every
quantile. `bc_armA_seed0/ckpt_final`, spawn 777, later cohort, successes:

| quantile | `-v0` (1.5 mm) | `-Loose-v0` (3.0 mm) | ratio | pure censoring predicts |
|---|---|---|---|---|
| p25 | 0.29 | 0.29 | 1.02× | 2.00× |
| p50 | 0.76 | 0.66 | 0.87× | 2.00× |
| p75 | 1.10 | 1.10 | 1.00× | 2.00× |
| p90 | 1.34 | 1.76 | 1.32× | 2.00× |
| max | 1.49 | 2.57 | — | — |

**The bulk of the distribution does not move.** Below p75 the quantiles are identical within
sampling noise; only the tail widens. So for *this policy* the lateral error is largely
**intrinsic**, not an artefact of the walls — which means §5c overcorrected. Censoring is real
but confined to the upper tail.

Why the expert looked different (p90 falling 1.38 → 1.11 → 0.48 across the same three rungs):
its intrinsic spread is tighter than the policy's, so for the expert the channel *is* the
binding constraint on where the block comes to rest more of the time. Both observations stand;
they are about two different distributions.

**The useful consequence.** `-Loose-v0` is the least-censored view available, so estimate the
Tight fit fraction from it rather than from `-v0`:

| channel | fraction of ALL later episodes with \|lateral\| < clearance |
|---|---|
| 0.5 mm (Tight) | **39/96 = 0.406** |
| 1.5 mm (v0) | 78/96 = 0.812 |
| 3.0 mm (Loose) | 95/96 = 0.990 |

**0.41, not 0.19.** This is a refinement of the *reasoning*, not a moved goalpost: `EXP_TIGHT.md`
belief 1 pre-registered **25–55 %** before any of this existed, and 0.41 sits inside it. The
pre-registered interval is not being revised to fit.

Still uncertain in **both** directions, and worth stating which way each pushes:
* *downward* — an episode 0.1 mm over budget on Tight does not merely "miss": it jams on the lip
  and the push phase then has less room to recover, so it can fail worse than the static
  geometry implies.
* *upward* — a narrower channel may **guide** a marginal block in as the walls contact it,
  which the static test cannot see at all.

### 5d. The consequence that actually matters: the expert already solves Tight

> **⚠ SUPERSEDED LATER THE SAME SESSION — read `EXP_TIGHT.md` §2b before acting on this.**
> The recommendation below ("collect Tight demos") is *mostly a no-op*, and checking
> `expert/plan.py` is what showed it: `ExpertParams` contains **no clearance term** at all, so
> the expert commands the **same trajectory** on Tight as on `-v0`. Tight demos would carry
> near-identical action labels to the 2038 already collected. The reasoning below is left intact
> because the step that killed it — *read the planner instead of assuming the expert adapts* —
> is the transferable part. The reframed plan is in `EXP_TIGHT.md`.

**128/128 on `-Tight-v0`.** Meanwhile **all 2038 demos in `data/v2/` were collected on
`Rebot-PrecisionSlot-v0`** — verified, not assumed:

```
python - <<'EOF'
import h5py, glob
from collections import Counter
c = Counter()
for p in sorted(glob.glob('slot/data/v2/*.hdf5')):
    g = h5py.File(p, 'r')['data']
    for k in g:
        c[g[k].attrs.get('task', '?')] += 1
print(c)          # Counter({'Rebot-PrecisionSlot-v0': 2038})
EOF
```

So if the `-v0`-trained policy underperforms on Tight, the first hypothesis is **train/test
clearance mismatch — a data problem** — and the fix is to collect Tight demos from an expert
that already scores 100 % there. That is one collection pass plus one training run on existing
tooling, with no new algorithm, no reward design, and no PPO.

**This re-orders Stage D.** The plan was "BC falls short on Tight → x0-steering". The evidence
says try the cheap thing first:

1. **Measure.** The sweep already evaluates every checkpoint on `-Tight-v0`. Get the number.
2. **If short: collect Tight demos and retrain.** Expert is 100 % there; harness exists.
3. **Only if that is also short: x0-steering.** Which is now unblocked and CPU-validated
   (§6a–6c) and will still be sitting there.

Reaching for the RL because the RL was the plan — when a 100 %-capable expert and an idle
collection harness are sitting right there — would be exactly the "mis-diagnosed the error
class" mistake that cost EXP06 two runs (`POSTMORTEM.md` §9b).

---

## 6. What this reorders

1. **Stage D (x0-steering) is no longer about reaching 70 %.** BC alone does that on `-v0`.
   Stage D's job is now **`-Tight-v0`**, where §5b predicts BC collapses to ~19 %, and the
   failure it must fix is sub-millimetre lateral centring at the slot mouth.
2. **EXP07's headline mechanism does not transfer** (§2a) — no never-lifted bucket exists
   here. Expect a smaller effect and say so before running, not after.
3. **The grasp bit was the wrong feature for this task** (§3, §4). It is now analytic and
   unblocking rather than load-bearing, with the `engaged` swap pre-registered as
   escalation #1.
4. **The `experiments/exp06_grasp_bit.pt` blocker is closed** — it would have crashed
   `train_steer.py` on startup after paying the Isaac Sim boot cost.

---

## 6a. The 89.6 % was audited before it was believed

An unexpectedly good number deserves the same scrutiny as an unexpectedly bad one, and this
project already has a case (Stage A: 93.8 % with 13.28 mm mean lateral error) where the
headline was arithmetic rather than robotics. `check_eval_json.py` re-derives every reported
rate from the per-episode records:

```
python scripts/check_eval_json.py \
    runs/bc_armA_seed0/eval_ckpt_0030000_Rebot-PrecisionSlot-v0_s777.json --expect-envs 32
```

**14/14 PASS.** The load-bearing ones: the two cohorts partition the 128 episodes with exactly
one first-episode per env; every reported rate equals the rate over its own records to 1e-9;
`success` is never true where `inserted_raw` is false; the seat guard rejected 2 episodes the
env's bare predicate accepted; and **every episode ran exactly 600 steps**, which is what rules
out an early termination silently re-randomising the scene and handing the policy a second
attempt.

### 6b. Zero flushes in 128 episodes — and why that matters for Stage D

`flushes 0 over 128 episodes (enabled: True)`. The §4.2 discontinuity flush is armed and never
fires on this task. Two consequences:

* the 30 mm position-jump trigger is correctly sized — it is not firing on legitimate motion,
  and the pick-place z-drop half was already removed for exactly that reason (it would have
  fired on the block's legitimate 8 mm settle at release, the single most precision-critical
  moment in the episode).
* **the steering window alignment will be exact.** In EXP07 the flush was the *only* source of
  window desync — an env that flushes re-predicts mid-window and then refills off-boundary for
  the rest of the episode, applying a z up to 14 steps stale. That was accepted there as rare
  (~1/episode scale). Here it is not rare, it is **absent**: 600 steps = 40 windows exactly,
  no mid-episode terminations, no flushes. Every refill lands on a window boundary.

## 6c. The steering path is CPU-validated (17/17) before any simulator boots

`POSTMORTEM.md` §9d attributes EXP07 reaching its result on the first configured run — against
EXP06 burning two on an rl_games default — to pre-registered discipline, of which the cheapest
item is a CPU plumbing test. `scripts/test_steer_cpu.py` is the slot equivalent: a real
`FlowMatchingPolicy` behind a real `BatchedACTController`, no Isaac Sim, ~20 s.

```
python scripts/test_steer_cpu.py
```

**17/17 PASS.** The one that earns the file:

> `steer_x0 = zeros` is **action-for-action identical** to `fixed_x0 = zeros`, max abs
> difference **0.000e+00**, *including across a deliberately desynced mid-window flush.*

That is EXP07's gate 1 — the property that makes any eventual steering result attributable to
steering rather than to a wrapper artefact — established on CPU in seconds instead of 35
minutes of GPU. Also covered: per-env routing (steering env 3 moves env 3 and provably nothing
else — the silent failure that trains against the wrong env's z), `steer_x0` taking precedence
over a conflicting `fixed_x0`, the 2-D vs 3-D `x0` paths being bit-identical, `set_steer`
applying `α·tanh(z)` broadcast constant across all 50 chunk positions with the bound holding
for |z| ≫ 1, and `SlotGraspBit`'s AND semantics, thresholds, sensor name and documented dead
zone.

---

## 6d. THE SWEEP LANDED — and it overturns most of this document

*36/36 evals, zero failures, completed 05:54:58. 6 checkpoints × 3 clearances × 2 spawn seeds ×
128 episodes = 4 608 episodes. Everything below is the pooled `success_rate_later` cohort
(1 152 later-episodes per arm per task).*

### 6d-i. ⚠ RETRACTION: the 92.7 % headline was the best of three training seeds

`bc_armA_seed0` is not representative. Per-training-seed later-rates, averaged over both spawn
seeds:

| task | arm | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|---|
| `-Loose-v0` | A | 0.911 | 0.734 | 0.688 | **0.224** |
| `-Loose-v0` | B | 0.938 | 0.708 | 0.750 | **0.229** |
| `-v0` | A | **0.927** | 0.745 | 0.656 | **0.271** |
| `-v0` | B | 0.969 | 0.688 | 0.719 | **0.281** |
| `-Tight-v0` | A | 0.703 | 0.786 | 0.635 | 0.151 |
| `-Tight-v0` | B | 0.958 | 0.667 | 0.667 | **0.292** |

**Training-seed variance is 15–29 points on identical data.** That is the pick-place pattern
reproducing exactly (32.8–59.4 % across training seeds there), and it is the single most
important number in this document. Everything I reported earlier in the session came from
seed 0, which is the *luckiest* seed for both arms.

**The honest headline, pooled over 3 training seeds × 2 spawn seeds:**

| task | clearance | arm A | arm B | vs 70 % objective |
|---|---|---|---|---|
| `-Loose-v0` | 3.0 mm | 0.778 [0.742, 0.810] | 0.799 [0.764, 0.829] | **both clear** |
| `-v0` | 1.5 mm | 0.776 [0.740, 0.808] | 0.792 [0.757, 0.823] | **both clear** |
| `-Tight-v0` | 0.5 mm | 0.708 [0.670, 0.744] | 0.764 [0.728, 0.797] | **both clear** |

This is a *weaker* headline than 92.7 % and a **much stronger claim**: the objective is met on
**all three clearances**, pooled across every training seed and both spawn seeds, rather than at
one cherry-picked cell. The pre-registered Stage C bars (0.55 Loose / 0.40 v0) are cleared by
both arms with room to spare.

### 6d-ii. My Tight predictions were both wrong — and the failure taxonomy says why

| source | predicted Tight | actual (pooled) |
|---|---|---|
| §5b, censored-CDF extrapolation | 0.19 | |
| §5c-ii, Loose-based fit fraction | 0.406 | |
| `EXP_TIGHT.md` belief 1 (pre-registered) | 0.25–0.55 | |
| **measured** | | **0.736** |

All three are refuted, and not narrowly. Belief 4 (yaw is not binding) survives: **1 yaw_reject
in 797 failures across all three clearances.**

The reason is visible in the pooled taxonomy, and it is not subtle:

| bucket | Loose 3.0 mm | v0 1.5 mm | Tight 0.5 mm |
|---|---|---|---|
| later-cohort success | 0.788 | 0.784 | 0.736 |
| never_lifted | 6 (2.5 %) | 3 (1.2 %) | 8 (2.6 %) |
| gross_miss | 27 (11.1 %) | 36 (14.5 %) | 39 (12.8 %) |
| **never_entered** | **102 (41.8 %)** | **95 (38.2 %)** | **136 (44.7 %)** |
| **stalled_in_mouth** | **102 (41.8 %)** | **109 (43.8 %)** | **116 (38.2 %)** |
| yaw_reject | 1 | 0 | 0 |
| seat_reject | 6 | 6 | 5 |

**The failure mix is the same at every clearance.** Tightening the channel 6× — 3.0 mm to
0.5 mm — moves the success rate by 5 points and barely touches the composition.

And the decisive detail: **`never_entered` failures have median |lateral| of 0.59 / 0.79 /
0.73 mm.** On the *Loose* channel that is 0.59 mm of lateral error inside a **3.0 mm** opening,
with depth −40.8 mm. The block is beautifully aligned and simply **stopped ~40 mm short of the
slot**. These are not precision failures at all.

> **`never_entered` + `stalled_in_mouth` = 82–84 % of all failures at every clearance, and both
> are DEPTH failures with lateral alignment well inside the budget. The binding constraint on
> this task is how far forward the policy drives the block in x — which is clearance-independent.
> That is why the ladder is flat.**

Per-run, the ladder is not even monotone: **4 of 6 runs have Tight within 5 points of `-v0` or
better**, and `armA_seed1` is *better* on Tight (0.786) than on `-v0` (0.745) or Loose (0.734).
Only `armA_seed0` shows a large drop (0.927 → 0.703) — the same seed that produced the
92.7 % headline. A one-run reading of the clearance ladder would have been wrong in both
directions.

The expert's own design note is the corroboration: `insert_x = 0.2545` "drives the block into
the **back stop**, which squares it and removes depth variance". The expert solves depth by
pushing into a hard stop. The cloned policy inherits the trajectory but not the guarantee.

### 6d-iii. DART: no measurable difference — and why McNemar disagrees

The pre-registered rule fires its middle branch:

* `-Loose-v0`: B − A = **+0.021**, max within-arm seed spread **0.229** → gap does not clear.
* `-v0`: B − A = **+0.016**, max within-arm seed spread **0.281** → gap does not clear.

> **DART made no measurable difference at this volume. Do not claim it helped.**

The pairing check confirms the arms faced **identical spawns in all 2 304 episode slots, 0
mismatched**, so McNemar applies — and reports χ² = 13.35, p < 0.05, B-only 386 vs A-only 290.
**That significance is an artefact and the pre-registered rule is right to override it.** The
2 304 episodes are not independent: they cluster into **3 training seeds**, and the seed effect
(22–28 points) is an order of magnitude larger than the arm effect (1.6–2.1 points). Treating
clustered observations as independent inflates n from 3 to 2 304 and makes a two-point
difference "significant".

At the correct unit of analysis the answer is plain:

| seed | Loose | v0 | Tight |
|---|---|---|---|
| 0 | B | B | B |
| 1 | **A** | **A** | **A** |
| 2 | B | B | B |

B wins seeds 0 and 2 and loses seed 1, **consistently across all three tasks** — the
task-to-task consistency is real, but the unit is the seed, and a sign test on 2 of 3 gives
p = 0.5. Three training seeds cannot resolve a two-point effect. Registering that as the
answer rather than reaching for the number that reads better.

*(One loose end for whoever picks this up: on `-Loose-v0` the successes' |lateral| reaches
**4.26 mm** against a nominal 3.0 mm per-side clearance, while `-v0` tops out at exactly 1.50
and Tight at exactly 0.50. The v0/Tight maxima are the clearance to the digit — clean
censoring — so the Loose overshoot means `lateral_error` is not a pure inside-the-channel
measure at that width. Flagged, not explained; it does not affect any success rate, which is
computed from the env's predicate rather than from this field.)*

## 7. Operational notes

* **Three-way GPU contention was allowed to stand.** At 01:12 the GPU carried arm-B
  training plus two evals. Memory is not the constraint (4 GB used of 62 GB system,
  ~4 GB of 11 GB VRAM); compute is — the GPU sits at 99–100 %. Measured cost: the
  `ckpt_0030000` eval took **>12 min of CPU time against a 2 m 11 s solo baseline**. I let
  it stand because killing mid-run discards completed work, but I queued nothing further.
  The standing rule remains **one GPU job at a time**.
* **Two duplicate background waiters were killed** (PIDs 14937, 15186) — three separate
  processes were polling the same file for the same string. Their task notifications
  reporting "failed with exit code 144" are those kills, not a real failure.
* `docs/slot/HANDOFF.md` §1a remains the authoritative "what is running right now".
