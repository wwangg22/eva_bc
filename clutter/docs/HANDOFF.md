# HANDOFF — ClutterExtract

**Rewritten 2026-08-03, after the reset.** The success criterion changed, the expert is being
started over, and everything written against the old criterion was deleted. This file is the
full record: what the task is, what happened, what is known, and what to do next.

Companion documents, in reading order:

| | |
|---|---|
| **`REFERENCE.md`** | ⭐ everything durable and metric-independent: env facts, action/obs layout, kinematics, toolchain traps, BC constraints, the rules, the commands. **Read this second.** |
| **`16_DISTURBANCE_ANATOMY.md`** | ⭐ why the old expert fails, measured four ways, and the plan that comes out of it |
| `15_STRICT_METRIC.md` | what the criterion is now, why, how it was calibrated, and how ~5 000 episodes missed the error |
| `06_EXPERT_DESIGN.md` | kept through the reset for §3 (the gripper is binary). Carries its own retraction banner |
| `00_ENVIRONMENT.md` | machine, software, repos |
| `03_ENV_FACTS.md` | env internals: gym registration, rl_games traps, launching |

---

## 1. STATUS

> **2026-08-04 — READ §10 FIRST.** The environment changed again: the row now spawns at a random
> heading and the target is any one of the five blocks. Every success number below §10 that is
> not labelled `-Fixed-v0` describes the *frozen-row* task. Full account:
> **`17_ROW_RANDOMISATION.md`**.

```
TASK      Extract the target from a row of five blocks and set it down in the goal zone,
          WITHOUT moving any of the four neighbours more than 2 mm from where it spawned.
          The row spawns at a random heading; the target is any one of the five blocks.
TARGET    ~70 % on random spawns.
NOW       3.0 %    (frozen pose_p33 expert, 768 held-out episodes, seeds 88000-88005)
          17.1 %   the same expert on -Fixed-v0, the frozen-row control
NEXT      Build the new expert. P40 (the drag gate) and task #26 (per-env pose solving)
          are the two candidate manoeuvres and #26 should be measured first -- it is
          cheaper and P41 part B suggests it may make P40 unnecessary.
```

The current expert is a **baseline to beat, not a solution to extend**. Two of its design
decisions now fail: it closes the jaw *in situ* between the neighbours (§4), and its whole
manoeuvre is a fixed joint chain planned for one target pose that `refine` cannot move far
enough (§10).

Both repos are committed and pushed: `eva_bc@9f9d6bb`, `eva_rl@d329ff3`.

---

## 2. THE TASK, EXACTLY

`Rebot-ClutterExtract-v0` / `-Play-v0` (16 envs) / `-Tight-v0` (6 mm gap) / `-Lenient-v0`
(the retired topple-only rule, for re-running old baselines only).

| | |
|---|---|
| row | five 36 × 30 × 70 mm blocks; distractors at y = ±42, ±84 mm; target y = 0; all x = 250 mm |
| free gap | **12 mm** |
| goal | (185, −185) mm — **the carry passes over `distractor_0` and `distractor_1`** |
| spawn jitter | target x ±12 mm and **yaw ±0.20 rad** (no y); distractors x ±10 mm, y ±5 mm |
| measured min free gap | 2.6–13.0 mm, median ~8 |
| masses | target 0.040 kg, distractors **0.025 kg** (lighter on purpose, so they tip) |
| episode | 14 s = 700 env steps; decimation 8, physics dt 2.5 ms → one action lasts 20 ms |
| success | in the goal circle, set down, **no neighbour toppled AND none moved > 2 mm** |
| terminations | `time_out`, `target_dropped`, `distractor_toppled`, **`distractor_disturbed`** |

Fuller numbers in `REFERENCE.md` §1.

### 2.1 The constraint that dominates everything

**The gripper action is binary.** `BinaryJointPositionAction` → `where(a[6] > 0, OPEN, CLOSE)`.
Two states, nothing between, no rate limit.

```
_GRIPPER_OPEN = 0.045 per finger        ->  90 mm of separation
target depth 36 mm, 41.2 mm at full yaw ->  48.8 mm excess, 24.4 mm PER FINGER
```

**Every grasp sweeps each finger 24 mm through a row whose gaps are 12 mm, and no policy —
scripted or learned — can choose otherwise.** P39 measured the cost at **17 points**.

**Big Will's decision, 2026-08-03: the 90 mm gripper stays.** The harder task is the more
interesting one. Those 17 points are deliberately left on the table. This is written into
`REFERENCE.md` §9 as a standing constraint so a later session cannot quietly "fix" it.

---

## 3. WHAT HAPPENED THIS SESSION

### 3.1 Big Will's two decisions

> "we need a strict threshold. The expert should not move the neighbors beyond 2mm. And feel
> free to edit the environment in ReBOT_RL. Just ensure to update the corresponding docs, and
> make sure you only touch the environment YOU ARE WORKING on."

The threshold is **2 mm** — the tightest of the three that had been instrumented — and the
standing "do not modify `challenge/`" rule is lifted **for the clutter env only**. Later:

> "Yes thats fine, we should work with this env instead. This is a better task to solve, more
> of a challenge."

= keep the 90 mm gripper (§2.1). And:

> "Lets start pushing changes to github, remember to pull first"

= pull, validate, update docs, commit, push — every change, both repos.

### 3.2 The environment change (`eva_rl@ceeb24c`)

Scoped to clutter-only files. `mdp/common.py`, `mdp/rewards.py`, `mdp/terminations.py` and the
other env cfgs are shared and were not touched.

**`challenge/mdp/clutter.py`**

```python
DISTURB_TOL = 0.002
distractor_displacements(env)     -> (N, 4)   per block, planar, from spawn
max_distractor_displacement(env)  -> (N,)     the MAX, not the sum
any_distractor_disturbed(env, tol=DISTURB_TOL)
target_at_goal(...)  gains  & ~any_distractor_disturbed(env, tol)
```

Two design points worth keeping:

* **The constraint is a `max`; the shaping reward stays a `sum`.** `distractors_disturbed`
  (weight −3.0) is the dense gradient toward gentleness and needs to be continuous. But "no
  neighbour was moved" must not be satisfiable by nudging four blocks 0.5 mm each, which a
  2 mm budget on a *sum* would allow.
* **The instantaneous read is safe** because displacement from spawn is very nearly monotone —
  a shoved block does not slide back — so evaluating per step and terminating is equivalent to
  latching, without carrying state.

**`challenge/clutter_env_cfg.py`**

```python
distractor_disturbed = DoneTerm(func=mdp.any_distractor_disturbed, params={"tol": mdp.DISTURB_TOL})
disturb_penalty      = RewTerm(func=mdp.is_terminated_term, weight=-40.0,
                               params={"term_keys": "distractor_disturbed"})
```

**`disturb_penalty` is not decoration and leaving it out would have been a real bug.** Adding a
termination to an MDP whose shaping terms are net-negative hands the agent a way to *profit* by
triggering it: ending the episode early stops the accumulation of `action_rate`, `joint_vel`
and `disturbance`. Without the matching −40, shoving a neighbour would have become the
highest-value action available on step one. It mirrors `topple_penalty`.

`RebotClutterExtractLenientEnvCfg` (`-Lenient-v0`) restores the old rule — `tol = inf`, both
new terms off. It exists so the retired baselines stay re-runnable: **a baseline that cannot be
re-run is a baseline that cannot be checked.** It is also load-bearing for probes (§3.5).

**`scripts/test_clutter_env.py`** gained V7 (the constraint is silent at rest, silent at 1 mm,
fires at 4 mm) and **negative control (d)**: target at the goal with an *upright* neighbour
dragged 85 mm with it must not be a success. **(d) is what the suite was missing**, and its
absence is why the error survived. The row above it — "a distractor shoved 30 mm but left
upright: **not** toppled" — was written as a *passing* check. It is a true statement about
`any_distractor_toppled` and it is silent on whether toppling was the right constraint.

Also fixed control (c), which restored a distractor to a *pre-reset* pose and would otherwise
have started passing for the wrong reason.

Env docs updated: `docs/envs/clutter-extract.md`, `docs/envs/README.md`,
`docs/CHALLENGE_SUITE.md`.

### 3.3 P35 — calibrating the threshold before trusting it

A threshold below the simulator's noise floor fails every episode regardless of what the policy
does, and the task reads 0 % forever. So, before believing anything: **show it does not fire
when nothing violates it.**

Null action for a full 700-step episode, 768 episodes, on `-Lenient-v0`.
**Registered prediction: max < 0.2 mm, 0/768 disturbed.**

```
   step   1:  0.0000 mm        p50  0.0005 mm
   step   5:  0.0003 mm        p90  0.0010 mm
   step  30:  0.0005 mm        p99  0.0010 mm
   step 325:  0.0010 mm        max  0.0010 mm
   step 700:  0.0010 mm        disturbed at 2 mm: 0/768 = 0.00 %
```

**Worst case 1 µm over a full episode. The threshold sits 2 097× above it.** Prediction held and
was conservative by 200×. Drift saturates by step ~325 rather than growing, so a constant
tolerance is the right shape. Bracketed on the other side by the 12 mm free gap.

### 3.4 The re-baseline — 16.4 %

Frozen expert (`pose_p33`, `--close 40 --holds-scale 0.25`, the shipping config), 768 held-out
episodes on the strict env.

```
seed     88000  88001  88002  88003  88004  88005     lenient    2 mm strict
strict   17.2   14.8   14.8   22.7   14.1   14.8       73.3 %       16.4 %
```

Three reasons to believe it rather than argue with it:

1. **Two independent code paths agree to 0.1 points.** Offline re-scoring on the *lenient* env
   (evaluator latches displacement, cut applied in Python afterwards) gave **16.3 %**; the
   env-native predicate on the *strict* env gives **16.4 %**. Different environments, different
   termination structure, different place the predicate is evaluated.
2. **The seed spread is exactly binomial.** sd(observed) = 3.26 pts vs sd(binomial, n=128,
   p=0.164) = 3.27. **Ratio 1.00.** Spawn batch contributes nothing beyond counting noise, so
   **16.4 % ± 2.6** (95 %, n = 768) is a clean estimate.
3. **P35** (§3.3) says the threshold cannot be firing on solver noise.

Two structural findings fell out:

* **`distractor_toppled` now reads 0.0 %**, on a manoeuvre that previously reported ~20 %
  topples. A block must slide before it can tip, so the disturbance term fires first in *every*
  episode that would have toppled. **Toppling was never a separate failure mode** — it was the
  tail of the disturbance distribution, and the old metric counted only the tail.
* **2 mm, 5 mm and 10 mm now select the identical 16.4 %**, and the median displacement among
  successes is **0.00 mm**. The population is sharply bimodal — an episode either touches
  nothing or it shoves — so there is no population of near-misses under the cliff and the exact
  threshold does not matter to the result. The apparent 42-point gap between thresholds in the
  offline re-scoring was an artefact of the *lenient* env's long tail.

### 3.5 P36 — the anatomy. Three predictions, two confirmed, and the refuted one did the work

768 episodes. Every distractor's displacement recorded at every step with its phase label.

Run on `-Lenient-v0` **deliberately**: the strict env terminates at the crossing and auto-resets
the scene from inside `env.step` (the R23 trap, `REFERENCE.md` §5), so it can report *that* the
threshold was crossed and nothing about what happened next. Same physics, termination off,
whole trajectory visible; crossing found offline.

| # | prediction | result | |
|---|---|---|---|
| 1 | > 80 % of first crossings in `close` | **81.1 %** close, 18.9 % carry | CONFIRMED |
| 2 | > 90 % of first crossings on the inner pair | **100.0 %** | CONFIRMED |
| 3 | displacement predominantly along **y** | **\|dx\| is 9.2× \|dy\|** | **REFUTED** |

```
|dx| (toward/away from robot)  median 4.050 mm   mean 5.500 mm
|dy| (across the row)          median 0.442 mm   mean 1.124 mm
y-dominant in 2.3 % of crossings
```

**The neighbour is hooked and carried, not shoved aside.** I had the mechanism backwards: I
expected the blade's perpendicular half-span to push the neighbour sideways out of its slot.
What actually happens is that something intrudes into the neighbour's footprint and then the
motion that follows is the motion the finger is already making — **along the opening axis**.

Three more facts from the same run:

* **It is a jump at one specific step.** `p01 = 160, median 160, p90 166, p99 174`. Mapped onto
  the schedule (approach 0–77, settle 78–79, descend 80–154, predwell 155–159, **close
  160–164**, carry 165–278, …), **step 160 is the first step of `close`.** ≥99 % of crossings
  are at or after the instant the fingers begin to shut, the whole p01–p99 range spans 14
  steps, and the block is already at **4.05 mm** when it first exceeds 2. **This kills "close
  more slowly" and "shorten the close hold" outright.**
* **d1 : d2 = 2.50 : 1, z = 10.8** against an even split. One side of the row is hit two and a
  half times more often than the other, which a symmetric sweep cannot produce. Unexplained,
  and worth exploiting. The **outer pair is never** the first to cross, in 768 episodes.
* **First crossing is in `close` (81 %); most of the *motion* is in `carry` (78.8 % of total
  block travel).** The jaw shuts, hooks the inner neighbour in the first control step, and then
  **carries it.** That is Big Will's "grabs another box as well", with a mechanism attached.
  `descend` and `approach` together contribute **2.5 mm across all 768 episodes**.
* Spawn tightness predicts it: disturbed spawns have a **9.37 mm** median free gap against
  **12.93 mm** for clean ones.

### 3.6 P37 — the yaw-matching sweep. +2.9 points, and the wrong model

From P36's refuted prediction I inferred a mechanism: yaw-matching the jaw swings a blade
corner into the neighbour's footprint (`47 · sin(11.4°) = 9.3 mm` against 7.8 mm of margin,
numbers from `target_axis()`'s docstring), and the closing travel then drags the block along x.
`ClutterExpert.target_axis(gain)` already parameterises exactly this and had never been swept.

```
yaw_gain     1.00    0.75    0.50    0.25    0.00   | no-match-yaw
strict      16.4 %  17.3 %  17.1 %  17.1 %  19.3 %  |   13.4 %
enclosure   17.2 %  15.6 %  14.8 %  17.2 %  23.4 %  |     --
```

**+2.9 points**, enclosure rises too (so the objectives are not opposed), roughly monotone —
but the curve is flat across 0.75–0.25 and all the movement is in the last step. Prior evidence
existed and had been dismissed: P16 measured that matching yaw *raised* close-phase contact
count from 65/128 to 93/128, and P19 found the square jaw outscoring the matched one 69.5 % vs
64.1 % — written off as noise, defensibly, since a 5.4-point gap sat inside a ±12-point
interval.

`--no-match-yaw` is **worse than baseline** (13.4 %). Dropping the orientation constraint is
not the same as pinning it to the nominal axis: `refine` is then free to pick any wrist roll
and evidently picks worse ones. A useful negative control — the gain is from *where the jaw
points*, not from *relaxing a constraint*.

**+2.9 of 56.9 is not a good enough return to keep refining a model built on unverified
docstring geometry.** Which is why P38 stopped inferring.

### 3.7 P38 — two ablations that assume no geometry at all

**Ablation B — the arm alone.** Identical commanded trajectory, `close` forced False at every
step. The arm visits exactly the same poses; only the finger motion is removed.

```
close as normal        disturbed  67.2 %   median 4.805 mm   p90 44.482 mm
gripper FORCED OPEN    disturbed   0.0 %   median 0.000 mm   p90  0.311 mm
```

**0.0 %. Not "low" — zero, on 384 episodes, p90 0.311 mm.** Prediction was < 10 %.

**100 % of the disturbance is the finger closing motion.** The whole approach, descent and
pre-grasp dwell — every pose the arm holds, at every point in the row — is completely clean.
**The arm can go anywhere in the row safely; the only unsafe act is closing the jaw inside it.**
This is the single most important measurement in the session and it took two minutes.

**Ablation P — widen the row until it stops.** Purely diagnostic; the shipping task keeps its
42 mm pitch.

```
   pitch  free gap   disturbed   median      p90    neighbour inner face
    42 mm   12.0 mm     67.2 %    4.805    44.482       27.0 mm
    48 mm   18.0 mm     17.4 %    0.000     8.954       33.0 mm
    54 mm   24.0 mm      0.0 %    0.000     0.354       39.0 mm
    60 mm   30.0 mm      0.0 %    0.000     0.370       45.0 mm
    70 mm   40.0 mm      0.0 %    0.000     0.000       55.0 mm
```

**The fouling reaches 33–39 mm from the target's centre.** Prediction was 25–28 mm from the
documented blade. **REFUTED — about 1.8× the documented geometry.** So the corner-intrusion
model P37 was built on is simply the wrong model, and **the "±19.2 mm perpendicular, ~47 mm
along the opening axis" figure quoted since Stage 1 is retracted** (the *conclusion* it
supported — straddle fore-and-aft, never enter a 12 mm row gap — survives on other evidence).

### 3.8 P39 — the gripper opening. +16.9 points, and declined

Following the ablations: if the disturbance is entirely the finger *closing motion* and the
fouling region is far wider than the gap, the quantity that matters is **how far each finger
travels while shut**. See §2.1 for the arithmetic — 24.4 mm of pure excess per finger.

```
separation  excess/finger   STRICT   enclosure  disturbed
   90 mm       24.4 mm      16.4 %    17.2 %     83.6 %   <- shipping
   70 mm       14.4 mm      16.7 %    14.8 %     83.3 %
   60 mm        9.4 mm      19.5 %    18.8 %     80.5 %
   52 mm        5.4 mm      27.3 %    26.6 %     72.7 %
   46 mm        2.4 mm      33.3 %    31.2 %     66.7 %
   42 mm        0.4 mm      39.1 %    35.2 %     60.9 %   <- below the yawed block's own
   38 mm        0.0 mm      41.9 %    35.9 %     58.1 %      depth (41.2 mm); DO NOT ADOPT
```

**+16.9 points at 46 mm**, which still leaves 2.4 mm of clearance over the widest a yawed
target can present. Monotone, and **enclosure rises with it** — the narrower opening grips
*better*, because the fingers spend less time ploughing. Not a trade.

**Declined by Big Will.** Reported, not adopted. The 42/38 mm rows are below the geometric
limit and are reported for curve shape only — at those openings the jaw cannot clear a fully
yawed target on approach and part of the gain is the gripper bulldozing the target.

### 3.9 The reset

Big Will: *"Let's clean up everything we have done that is NOT relevant… and start over from
figuring out an expert! Don't delete the expert writeup, but remove the old stuff."*

**Deleted** — 12 stage-result/planning docs (6 397 lines), probes P01–P34 and Q7, 153 JSON/logs
belonging to them, and the untracked binaries: **5.9 GB of BC checkpoints and 159 MB of demo
HDF5s**, both cloned from the manoeuvre being abandoned. `runs/` went **6.0 GB → 37 MB**, docs
**8 510 → 1 933 lines**.

**Kept** — `REFERENCE.md` (new; everything durable, extracted *before* the deletions),
`15_`, `16_`, `06_EXPERT_DESIGN.md` (per instruction, with a retraction banner), `00_`, `03_`,
`_kin.py`, P35/P36/P38, the whole `act/` BC pipeline, `expert/`, and the 16 videos that found
the metric bug.

Ten dangling doc references in `act/`, `expert/` and eva_rl were repointed rather than left
broken. Everything deleted is recoverable at `eva_bc@3e83f44`.

---

## 4. WHERE THE 53.6 POINTS ARE — the one-paragraph version

Under the 2 mm rule the failure taxonomy has **exactly one non-zero bucket**:
`distractor_disturbed` **83.6 %**, with `time_out`, `target_dropped` and `distractor_toppled`
all at **0.0 %**. The arm is innocent (0.0 % with the jaw held open). The damage happens in the
**first control step of the close**, to the **inner pair only**, **fore-and-aft** (|dx| 9.2×
|dy|), and the neighbour is then **carried** rather than left behind. The fouling reaches
**33–39 mm** against faces at 27 mm, because the jaw opens to **90 mm to grasp a 36 mm block**
and no policy can choose otherwise.

---

## 5. WHAT WORKED AND WHAT DIDN'T — methodology

Full rule list in `REFERENCE.md` §7. What this session specifically added or paid for:

### Worked

* **Run the ablation before the sweep.** P37 swept a five-point grid to find +2.9 points on an
  inferred mechanism. P38's ablation B — same trajectory, gripper forced open — took two
  minutes, returned **0.0 %**, and was worth more than the entire sweep: it proved the effect
  lives in one motion, which made the pitch sweep worth running and made the whole current plan
  visible. **A sweep assumes you already know which parameter; an ablation tells you.**
* **Registered predictions, written into the probe's docstring before the run.** Two were
  refuted and both refutations reordered the candidate fixes and killed two of them. The two
  *confirmed* predictions told me nothing I had not already assumed.
* **Measuring on `-Lenient-v0` to diagnose the strict env.** The strict env terminates on the
  quantity being measured and auto-resets from inside `env.step`. Keeping a same-physics,
  no-termination variant made the whole anatomy visible.
* **A null control before trusting a threshold** (P35). Two minutes; without it a threshold
  inside the noise floor is indistinguishable from a policy that cannot do the task.
* **Two independent paths to the same number** (16.3 offline / 16.4 env-native). A metric change
  that is also a measurement error is worthless, and this is what rules that out.
* **Extracting `schedule_utils.py`** so the diagnostic probes run *the manoeuvre that was
  measured* rather than a re-implementation. Verified behaviour-preserving: seed 88000 scored
  17.2 % before and after.
* **A local `clutter/.gitignore`** that overrides the root's bare `runs/` — `runs/*` re-excludes
  the contents including subdirectories, then `!runs/*.json` / `!runs/*.log` pull back the flat
  evidence. Keeps 10 MB of evidence, excludes GB of regenerable artifacts.

### Didn't work / cost time

* **Theorising from a docstring's geometry.** "±19.2 mm perpendicular, ~47 mm along" had been
  quoted since Stage 1 and used to justify grasp design. The measured fouling reach is 1.8×
  that. I built P37 on it and got 2.9 points for a five-cell sweep. **Docstring geometry is not
  measured geometry.**
* **My mechanism story was backwards twice.** First "the blade pushes the neighbour sideways"
  (refuted: |dx| is 9.2× |dy|), then "a yaw-swung corner is the intrusion" (refuted: the reach
  is far larger than any corner swing). What survived both was the *ablation*, not the model.
* **`put()` in the env smoke test broadcasts env 0's pose to all envs.** My restore silently
  displaced 15 of 16 envs, which under the new constraint is itself a failure — the test went
  red and it was my bug, not the env's. Fixed by keeping poses per-env.
* **Neither repo had a git identity**, so the first commit and a mid-rebase continue both failed
  and left a rebase in progress with a missing message file. Fixed with `git rebase --abort`,
  then setting `user.name`/`user.email` locally in both repos.

---

## 6. THE PLAN

**Subject to change** — in particular, step 1 is a gate and a negative result there re-plans
everything under it.

### 6.1 The idea

Two measured facts, and their conjunction is the whole plan:

* **An open jaw disturbs nothing, anywhere in the row.** 0.0 %, 384 episodes, p90 0.311 mm.
* **`DISTURB_TOL` binds the four distractors and says nothing about the target.** Sliding,
  dragging or tipping the *target* is entirely legal.

So the manoeuvre the constraints actually permit is:

```
1  reach in with the jaw OPEN                    measured safe: 0.0 % disturbance
2  pull the TARGET clear of the row (-x, toward the robot) with a fingertip, the closed
   jaw, or a partial close
3  close on it OUTSIDE the row, where the 24 mm finger sweep fouls nothing
4  carry to the goal
```

This is the extrinsic-dexterity solution the environment's own docstring describes — *"the
correct first action is often not a grasp"* — and **no expert in this project has ever tried
it.** Every one, from the first probe to `pose_p33`, closes the jaw between the neighbours. It
is also the only candidate that needs no benchmark change and attacks the whole 53.6 points
rather than a slice.

### 6.2 Ordered steps

**Reordered 2026-08-04.** Step 0 is new and comes before P40: **task #26, per-env pose solving.**
P41 part B measured an in-situ close at a *well-solved* pose disturbing neighbours by only
0.8–1.6 mm median, against the frozen expert's 4.8 mm. If that survives spawn jitter, the whole
extract-then-grasp plan below is solving a problem that a better pose already solves. It is
cheaper than P40 and it can retire it. Re-run P41 part B on jittered spawns first.

```
1  P40  GATE.  Measure step 2 alone.  Reach in with the jaw open, drag the target -x by
        30-40 mm, watch the neighbours.  Report the disturbance rate, and separately how
        far the target actually travels (it may not follow a fingertip cleanly).
        Registered prediction to write BEFORE running: near 0 %, since P38 says an open
        jaw is safe and the target is unconstrained.  Refutation: if dragging rakes the
        row, the plan is dead -- go to 6.4.
        CHEAP.  One probe, ~10 minutes.  NOTHING ELSE STARTS BEFORE THIS REPORTS.

2       DESIGN THE MANOEUVRE around whatever P40 says works:
          - what contacts the target (one fingertip? the closed jaw's outer face? a
            partial close?), and at what height -- the block is 70 mm tall and a low
            contact tips it, which is legal but changes the grasp that follows
          - how far to drag: far enough that the 24 mm sweep clears the row, i.e. the
            target's centre needs to end >= ~40 mm clear of the neighbours' inner faces
            (P38's pitch sweep says the fouling reaches 33-39 mm)
          - where the grasp pose sits once the target is out, and whether the target's
            yaw after dragging is still within what the jaw can accept

3       POSE + CHAIN.  Same machinery as before (ClutterExpert, _kin's FK-scored CEM),
        but a NEW pose family for the new manoeuvre.  Freeze by PERSISTING the artifact,
        never by re-deriving under a seed -- a CEM is not bit-reproducible
        (REFERENCE.md section 5).  Audit the seam in JOINT space wherever two solved
        paths meet: two IK branches at the same TCP differ by 1.90 rad and are invisible
        to every statistic in this codebase.

4       MEASURE.  seeds 88000-88005, 768 episodes, against the 16.4 % baseline, on the
        strict env, through collect_demos.py --arms appr so it shares the baseline's
        code path.  Report enclosure separately from success.

5       ONLY THEN: demos -> flow BC -> 3 seeds -> eval.  The act/ pipeline is unchanged
        and working.  REFERENCE.md section 6 has its constraints (n_action_steps=15 is
        not tunable; >=3 seeds; loss does not predict success; measure the N2 ambiguity
        floor offline first).  DO NOT TRAIN BEFORE 4 REPORTS -- ~50 min/seed and the
        ceiling is the expert's rate.

6       VIDEO IT EARLY.  record_video.py, one env per file, camera 0.67 m out.  Sixteen
        episodes found what ~5 000 scored ones did not.
```

### 6.3 Carry the +2.9 forward

`--yaw-gain 0` is free and measured. Whatever the new manoeuvre is, sweep the gain again on it
rather than assuming 0 transfers — the geometry of the close will have changed.

### 6.4 If P40 refutes the plan

Fallbacks, in order:

1. **Roll / clocking.** §3.5's **d1 : d2 = 2.50 : 1 (z = 10.8)** is unexplained. Something is
   chirally biased, and whatever makes `distractor_2` safer 2.5× more often is a property of
   the pose. P30 measured a roll worth +7.0 paired *against topple*; it has never been
   optimised against **disturbance**.
2. **Tip the target rather than translate it.** Also legal, also unconstrained, and it changes
   the footprint the jaw has to fit around. Untested.
3. **Re-open the gripper-opening question with Big Will** with the P39 curve in hand.

### 6.5 Deferred, deliberately

* ~~**Row orientation randomisation.**~~ **DONE 2026-08-04 — §10.** The deferral argument
  ("rotating the row before there is a manoeuvre to rotate would measure nothing") was wrong:
  the expert is being rebuilt anyway, so doing it *first* costs a day and doing it *after* costs
  the expert. Also note the old worry — "the row's 12 mm gap does not rotate with it" — was
  simply mistaken: a rigid transform rotates the blocks too, so the gap rotates with it and the
  clutter geometry is invariant.
* **A diverse, multi-mode expert** (#20). Now unblocked and more clearly motivated: mode-mixing
  hurts BC only when the mode is *unobservable*, and §10.1 says a single pose family cannot
  cover the slot × heading space — so multiple modes are not optional, and `clutter_obs` already
  supplies the observable that makes them learnable rather than merely ambiguous.
* **`-Tight-v0`** (6 mm gap) has never been measured, under either predicate.

---

## 10. 2026-08-04 — THE ROW MOVES NOW

Full account and every number: **`17_ROW_RANDOMISATION.md`**. The short version:

Big Will: *"I think its smarter to fix the env completely (make row randomize, and also can you
alternate the block of interest (right now it is always the middle one)"* — both were deferred
items #20/#21, and the redirection is right: the expert is being rebuilt from scratch anyway, so
building it against a row that never moves would bake in the assumptions the randomisation
exists to forbid, and it would have to be rebuilt a second time.

`mdp.reset_clutter_row` replaces the five per-asset reset terms and draws a rigid whole-row
heading `U(−0.30, +0.30)` rad, a ±10 mm rigid centre offset, **and which of the five slots holds
the target**. New `-Fixed-v0` (strict rule, frozen row) is the ablation control; `-Lenient-v0`
was pinned to the frozen row too, because "the old task" is the predicate *and* the spawn
distribution.

**The two changes are different in kind.** The row's pose is an *isometry* — gaps, faces and
bearings unchanged, so it constrains the arm, not the clutter. The slot is not: an end slot has
one adjacent neighbour instead of two. **Report success per slot from now on.**

```
frozen expert, 768 eps    -Fixed-v0  17.1 %   (vs 16.4 % pre-refactor: distributional
                          -v0         3.0 %    regression check, inside the +/-2.6 CI)
taxonomy on -v0           distractor_disturbed 97.0 %, time_out 0.0 %
P41 reachability          25/27 cells; BOTH worst corners at r = 0.3087 m solve with
                          pos_err 0.00 mm, o_align 1.000 -> NO WALL, ships as configured
P41 part B (close/slot)   end slots 0/9 disturbed, interior 3/14 -- suggestive, tiny n
smoke test                V8 added, V1 rewritten; passes
```

Three things this opened up, all in `17_` §4:

1. **The row centre's pose family does not continue to slot 0 at positive yaw.** Continuation
   misses there on *position* by 1.5–5.9 mm while a global search finds `pos_err 0.82 mm`,
   `o_align 1.000`. This is the two-IK-branches trap as a search boundary: **a single pose family
   will not cover the slot × heading space**, and the hole will not announce itself.
2. **Accepted poses live in a narrow wrist band, 17.2–23.8 mm.** Every pose that cleared the
   geometric gates and then shoved the row had `wrist_z ≥ 28 mm`. Cheap new gate, and it
   separates families `o_align` cannot.
3. **Closing in situ at a *well-solved* pose moved neighbours only 0.8–1.6 mm median** against
   the frozen expert's 4.8 mm / 44 mm p90. If that survives spawn jitter — and P41 had none,
   which flatters it exactly where the expert fails — then the deficit is **the pose, not the
   manoeuvre**, and P40 may be unnecessary. Task #26; settle it before building anything.

---

## 7. WHAT IS ON DISK

```
clutter/
  docs/       HANDOFF.md (this)  REFERENCE.md  SESSION_STATE.md
              15_STRICT_METRIC.md  16_DISTURBANCE_ANATOMY.md
              06_EXPERT_DESIGN.md  00_ENVIRONMENT.md  03_ENV_FACTS.md
  expert/     clutter_expert.py -- adaptation, schedule, pose machinery; reusable
              pose_p33.json     -- the frozen 16.4 % pose + chain + approach. THE BASELINE.
  probes/     _kin.py                    -- kinematics core (REFERENCE.md section 4)
              p35_disturb_calibration.py -- the null control for DISTURB_TOL
              p36_disturb_anatomy.py     -- when / which block / which way / how much
              p38_disturb_ablation.py    -- gripper-open ablation + row-pitch reach sweep
  act/        collect_demos.py  -- demo writer AND port gate, one code path.
                                   Flags: --arms --close --holds-scale --yaw-gain
                                   --no-match-yaw --grip-open --record --multimodal
              schedule_utils.py -- expand / approach_prefix, shared so probes cannot drift
              dataset.py train_flow.py eval_flow.py policy_runner.py
              record_video.py analyse_demos.py
  runs/       JSON + logs committed (37 MB); checkpoints, HDF5s, videos gitignored
```

eva_rl, clutter-only: `challenge/mdp/clutter.py`, `challenge/clutter_env_cfg.py`,
`challenge/__init__.py` (registry, additive), `scripts/test_clutter_env.py`,
`docs/envs/clutter-extract.md`, `docs/envs/README.md`, `docs/CHALLENGE_SUITE.md`.

---

## 8. STANDING CONSTRAINTS

- eva_bc work lives in **`eva_bc/clutter/`**.
- **eva_rl's `challenge/` may be edited — CLUTTER env only.** In scope: `mdp/clutter.py`,
  `clutter_env_cfg.py`, `scripts/test_clutter_env.py`, the clutter docs. **Out of scope**
  (shared): `mdp/common.py`, `mdp/rewards.py`, `mdp/terminations.py`, `mdp/observations.py`,
  the other env cfgs, and `lift/rebot_lift_env_cfg.py`.
- **Pull first, validate, update the docs, commit, push.** Every change, both repos. Both now
  have `user.name`/`user.email` set locally.
- **The 90 mm gripper stays.** It costs 17 measured points and it is the challenge.
- One GPU job at a time (10 GiB card). `python -u`, or output buffers and you see nothing.
- Address Big Will as Big Will.

---

## 9. COMMANDS

```bash
source /home/eva/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc
```

```bash
# THE BASELINE -- frozen expert on the strict task.  16.4 %.
python -u clutter/act/collect_demos.py --num_envs 128 --arms appr --close 40 \
    --holds-scale 0.25 --seeds 88000,88001,88002,88003,88004,88005 --headless \
    --json clutter/runs/strict2mm_expert.json

# the three live probes
python -u clutter/probes/p35_disturb_calibration.py --num_envs 128 --steps 700 --headless \
    --json clutter/runs/p35_disturb_calib.json
python -u clutter/probes/p36_disturb_anatomy.py --num_envs 128 --headless \
    --seeds 88000,88001,88002,88003,88004,88005 --json clutter/runs/p36_disturb_anatomy.json
python -u clutter/probes/p38_disturb_ablation.py --num_envs 128 --seeds 88000,88001,88002 \
    --headless --json clutter/runs/p38_ablation.json

# the env's own smoke test -- both constraints, four negative controls
cd /home/eva/Desktop/isaacLab/eva_rl && python -u scripts/test_clutter_env.py --headless
```

Levers all run through `collect_demos.py` so they share the baseline's code path:
`--yaw-gain 0` (+2.9 pts), `--grip-open 0.023` (46 mm — **measured, not adopted**),
`--close` / `--holds-scale` (hold durations, in physics steps).

BC pipeline and the rest: `REFERENCE.md` §8.
