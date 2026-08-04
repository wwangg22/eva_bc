# Big Will's feedback, 2026-08-03 — and the plan it changes

> ## ⚠ RESOLVED — see `15_STRICT_METRIC.md`
>
> **Item 1's open question is answered: the threshold is 2 mm**, and it is now enforced by the
> environment itself rather than by the evaluator. The expert scores **16.4 %**.
>
> **Item 3's constraint collision (§3) is lifted.** Big Will: *"feel free to edit the
> environment in ReBOT_RL... make sure you only touch the environment YOU ARE WORKING on."*
> The subclass-from-`clutter/` workaround planned below is no longer necessary; the row
> rotation goes into `clutter_env_cfg.py` behind a knob that defaults OFF.
>
> **§1.1's 5 mm and 10 mm columns are superseded.** They were measured on the lenient env,
> where a shoved episode plays on to the goal. With the termination in place, 2/5/10 mm select
> the *identical* 16.4 % and the median successful episode displaces neighbours by 0.00 mm.
> The apparent 42-point gap between thresholds was an artefact of the lenient env's tail.
> The 2 mm figure (16.3 %) stands and is confirmed independently at 16.4 %.

Three pieces of feedback, delivered after watching the 16 policy videos in `runs/videos/`.
Two of them invalidate things this effort had been treating as settled. They are recorded here
in full because the first one changes **every success number in Stages 0–2**.

---

## 1. The success criterion is too lenient — CONFIRMED IN THE CODE

> "in many of the success episodes, the robot actually grabs another box as well as the one of
> interest. It just so happened when the robot placed it down, the box didn't topple over. So
> clearly this should be considered a failure, if the boxes that aren't the one of interest get
> moved around."

**He is right, and it is verifiable without running anything.**
`challenge/mdp/clutter.py`:

```python
def target_at_goal(env, name="target"):
    p = common.object_pos_local(env, name)
    return (((p[:, :2] - goal).norm(dim=1) < GOAL_RADIUS)
            & (p[:, 2] < 0.055)
            & ~any_distractor_toppled(env))          # <-- TOPPLED, not MOVED
```

and `any_distractor_toppled` is `up_z < TOPPLE_DOT` with `TOPPLE_DOT = 0.75`, i.e. about
**41 degrees of tilt**. A neighbour dragged across the table and set down upright satisfies
the predicate completely.

The environment *does* track displacement — `mdp.distractors_disturbed` returns the summed
planar distance of all four distractors from their recorded spawn poses — but it is wired
**only to a reward term** (`RewardsCfg.disturbance`, weight −3.0). It is not in
`TerminationsCfg` and not in `clutter_success`. So the quantity exists, is computed every step,
and never reaches the metric.

### Why this was missed, which is the part worth learning from

The mistake was not failing to look. `HANDOFF.md` §11 has carried the rule *"score with the
env's own predicates, read at the end rather than enforced"* since Stage 0, and it was followed
exactly. The rule is right — inventing a private success criterion is how a project ends up
optimising something the benchmark does not reward. But it was applied without ever asking the
prior question: **does the benchmark's own predicate actually encode the task as stated?**

Here it does not. The task's premise, in `clutter.py`'s own module docstring, is "extract it
and set it down in the goal zone **without toppling any neighbour**" — and the predicate is a
faithful encoding of *that sentence*. It is the sentence that is too weak, and no amount of
care in reading the code would have caught that. **Watching sixteen videos did.**

A second contributor: `13_STAGE2_BC_RESULTS.md` §6.2 read the taxonomy as
"the policy has one failure mode and it is the expert's". That was true *of the buckets that
were being counted*. The bucket that mattered was not among them, so the taxonomy was complete
and uninformative at the same time. **A taxonomy only constrains what it enumerates.**

### What was done about it

`eval_flow.py` and `collect_demos.py` now latch, per episode and before any termination, the
**worst planar displacement of any non-target block from its spawn**, and report

```
strict success  =  target_at_goal  AND  max neighbour displacement < threshold
```

at thresholds **2 mm / 5 mm / 10 mm**, alongside the distribution of that displacement among
episodes the old predicate calls successful. Three thresholds rather than one because the right
cut is a judgement about the task, not a measurement — 2 mm is roughly "untouched", 10 mm is
roughly "nudged but still in its slot", and the row's free gap is 12 mm.

**RESULTS: see §1.1 below** (filled in from `runs/strict_*.json`).

### What it implies regardless of the exact number

* Every headline in `HANDOFF.md`, `11_STAGE2_RESULTS.md` and `13_STAGE2_BC_RESULTS.md` is an
  **upper bound**, including the expert's 73.3 % and the BC policy's 71.8 %.
* The *relative* comparisons are probably less affected than the absolute ones — expert and
  policy were measured under the same lenient predicate, on the same spawns, so the −1.5-point
  gap is likelier to survive than the levels are. **Likelier, not certainly**: if the policy
  drags neighbours more often than the expert does, the gap widens under the strict rule. The
  re-scoring measures exactly this and it must not be assumed.
* **Gate 1's 85 % and the ≈70 % mission target were both defined against the lenient
  predicate.** If the strict number is materially lower, those targets need restating — which
  is Big Will's call, not mine. My recommendation is to restate them against the strict
  predicate and treat the lenient numbers as historical.

### 1.1 The measured cost — it is not a correction, it is a different result

768 held-out episodes (seeds 88000-88005), expert and policy on identical spawns:

```
                     lenient    <2mm    <5mm   <10mm |  median      p90    >50mm
expert (v3 holds)      73.3 %  16.3 %  22.4 %  30.9 % |  13.7 mm  205.5 mm  25.1 %
BC v3 seed 1           72.0 %  19.3 %  26.3 %  34.0 % |  11.9 mm  206.2 mm  22.0 %
BC v3 seed 2           69.7 %  17.6 %  23.4 %  30.5 % |  13.2 mm  206.8 mm  22.1 %
BC v3 seed 3           73.7 %  18.9 %  25.4 %  33.6 % |  11.7 mm  206.3 mm  22.4 %
                                                ----
BC mean                71.8 %                  32.7 %       expert 30.9 %
```

`median` / `p90` / `>50mm` are the worst neighbour displacement **among the episodes the
benchmark calls successful**.

**At a 10 mm threshold the expert falls 73.3 % -> 30.9 %. That is 42.4 points.**

**And the ordering reverses.** Under the lenient predicate the policy sat 1.5 points *below*
its expert; under the strict one it sits **1.8 points above** it (32.7 % against 30.9 %, and
every individual seed is above the expert). The policy did not learn to be sloppier — it
inherited the demonstrations' sloppiness and smoothed it very slightly, which is what averaging
over a filtered demo pool would predict. It also means **the expert, not the policy, is the
thing that needs fixing**: BC is already reproducing it faithfully, and faithfully reproducing
a manoeuvre that shoves a neighbour 13.7 mm is the problem.

**The median "success" moves a neighbour 13.7 mm — more than the 12 mm free gap** the whole
task is built around. Half of the successes displace a block by more than the space between
blocks.

### 1.2 The distribution is bimodal, and the second mode is exactly what Big Will saw

`p90 ~ 206 mm` is not a nudge. Distances from each distractor's spawn to the goal zone:

```
distractor_0  (0.250, -0.084)  ->  120.1 mm
distractor_1  (0.250, -0.042)  ->  157.1 mm
distractor_2  (0.250, +0.042)  ->  236.1 mm
distractor_3  (0.250, +0.084)  ->  276.7 mm
```

The observed maxima are 213 / 213 / 256 mm. **A neighbour is being carried to the goal zone
along with the target**, and it happens in **22-25 % of all episodes** — a quarter of every run
this effort has ever scored. That is Big Will's observation, verbatim, with a number on it:

> "the robot actually grabs another box as well as the one of interest. It just so happened
> when the robot placed it down, the box didn't topple over."

So the population splits roughly three ways rather than two:

```
~31 %   target delivered, row essentially undisturbed      <- the real success rate
~42 %   target delivered, a neighbour shoved or carried    <- counted as success until now
~27 %   a neighbour toppled                                 <- always counted as failure
```

### 1.3 The evidence was already in the record, unconnected

This is the part that stings. Every Stage-1 and Stage-2 hazard table — P17, P22, P29 — reports
a **`close`-phase disturbance hazard of 71-76 %**, using a 1.5 mm displacement threshold. From
P29's own run this session:

```
hazards: close 76%, carry 45%
hazards: close 73%, carry 44%
hazards: close 73%, carry 40%
hazards: close 75%, carry 59%
```

Three quarters of episodes were **known** to disturb a neighbour, and it was reported in every
probe that measured hazards. It was read as "contact happens, the question is whether it
topples" — because the success predicate had already decided that toppling was the only thing
that counted. **The number was measured, published in the docs, and never allowed to reach the
metric.** Nobody had to look harder; somebody had to ask whether the metric matched the task.

---

## 2. The expert has exactly one grasp, at one angle

> "this robot really only uses ONE Way to pick up this object (at one specific angle too). Can
> we have a more diverse expert?"

**Also correct, and it is by construction rather than by accident.** The relevant history:

* `o_hat = x_hat`, `phi = 90 deg` — the orthogonal grasp — was chosen in P11 and never varied
  again. The fingers straddle the target fore-and-aft so they never enter the 12 mm row gaps.
  It is the single decision that made the task tractable at all.
* P28 found the pose *solver* produced **six structurally different clusters from eight draws**
  (wrist ±y × folded/extended arm × roll), and that was written up as a **defect** for BC:
  "mixing mirror-image joint trajectories for near-identical observations is P17's branch flip
  moved inside the network".
* P33 then ran a tournament over 8 candidates and froze the winner; P34 froze the whole 23-
  waypoint chain. Together those bought **+19 points** and made the data-generating process
  deterministic.

So the unimodality is a *result*, not an oversight — and Big Will is still right that it is a
liability. Three separate reasons:

1. **It caps what BC can represent.** A policy cloned from one manoeuvre cannot recover by
   trying a different one, which is precisely what a stuck or unlucky spawn needs.
2. **It is fragile to distribution shift.** Everything downstream — the frozen chain, the
   frozen approach, `pose_p33.json` — is tied to a row at one position and one orientation.
   Feedback item 3 breaks that assumption outright.
3. **It makes the +19-point pose gain suspicious as a *general* result.** It may be a gain
   specific to the one geometry the row has always had.

### The tension that has to be resolved, not ignored

P28/P33's evidence says mixing modes hurts BC. Big Will's request says one mode is not enough.
**Both can be true**, and the resolution is that mode-mixing hurts when the mode is
*unobservable* — the network sees near-identical observations with mirror-image actions and
must average them. It should not hurt when the mode is *implied by the observation*, e.g. when
the row's orientation determines which grasp family is correct.

That reframing makes feedback items 2 and 3 the same piece of work: **randomising the row's
orientation supplies the observable variable that makes a multi-mode expert learnable.**

The **unimodality control** (queued as task #17, `collect_demos.py --multimodal` is already
written and unrun) is the experiment that measures the cost of mode-mixing *without* the
observable, and it should be run first as the baseline for exactly this question.

---

## 3. The row is always broadside to the gripper

> "can our env also move the boxes to be lined up in an arbitrary orientation (i.e. right now
> the boxes are pretty much always horizontal to the robot hand, move the line of boxes
> around)"

Correct. `ClutterSceneCfg` puts the row at `ROW_X = 0.250` with distractors at
`y = ±0.042, ±0.084`, and `EventCfg` jitters each block by **±10 mm in x, ±5 mm in y** plus the
target's **±0.20 rad yaw**. The *row* never rotates and never translates as a unit. So the
approach azimuth is fixed, which is exactly what let a single frozen joint vector work.

### ⚠ A standing constraint this collides with

`HANDOFF.md` §2 and every session since: **do not modify eva_rl's `challenge/` package — it is
the benchmark under test; environment properties are findings to report, not defects to fix.**
That rule is Big Will's, and this request appears to cut against it.

**Plan, which is a judgement call and reversible.** Implement the randomisation as a **new env
variant registered from `clutter/`**, subclassing `RebotClutterExtractEnvCfg`, rather than
editing `challenge/`. That way:

* the benchmark stays pristine, so every number measured so far stays comparable;
* the harder variant is a separate, named task, and "does a policy trained on the rotated
  variant still solve the original?" becomes a measurable question instead of a lost baseline;
* it is trivially revertible if the answer is "no, change the benchmark itself".

**If Big Will wants the benchmark itself changed, say so and it is a small edit** — but it
should be a deliberate decision, because it retires every baseline in `07_`–`13_`.

### Two randomisations, materially different in difficulty

|  | what rotates | reachability | the manoeuvre |
|---|---|---|---|
| **(a) azimuth about the robot base** | the row's position on an arc at radius 0.25 | preserved by `joint1` symmetry | rotates rigidly; the frozen chain still works after a `joint1` offset |
| **(b) rotation about the row's own centre** | the row's heading, robot-relative | varies — the far end of the row moves toward/away | the approach direction relative to the row changes; the frozen chain does **not** transfer |

(a) is nearly free and tests translation invariance. **(b) is what the feedback is actually
asking for** — "the boxes are horizontal to the robot hand" describes the *heading*, not the
position. The honest plan is to implement both as independently switchable ranges, start with
(b) at a modest range (±30 deg) to keep the whole row reachable, and measure how the expert
degrades before deciding the full range.

**A geometric warning, from the numbers already in hand.** `target_axis()`'s docstring records
that the blade reaches ~47 mm along the opening axis against **7.8 mm of clearance**
perpendicular, so rotating the jaw by the target's full ±11.4 deg of spawn yaw already swings a
blade corner 9.3 mm — more than the margin. Rotating the *row* by ±30 deg is a much larger
version of the same geometry. **Expect the orthogonal grasp to stop working at some heading,
and expect that to be the finding rather than a bug.** The row's own 12 mm free gap does not
rotate with it: the fingers approach along a direction that no longer straddles the block's
36 mm faces.

---

## 4. The order this should be done in

Deliberately not the order the feedback was given in — item 1 changes the metric, and every
later measurement must be made against the corrected metric or it will have to be redone.

```
1  STRICT METRIC        re-score expert + all 3 BC seeds; restate every headline.   RUNNING
                        Decide the threshold WITH Big Will (2 / 5 / 10 mm), since it
                        is a task judgement and not a measurement.
2  UNIMODALITY CONTROL  task #17, already written and unrun.  Measures what mixing
                        pose modes costs BC when the mode is UNOBSERVABLE.  It is the
                        baseline against which "diverse expert" has to be judged, and
                        it is 3 training runs (~2.5 h) because of the 9.8-pt seed sd.
3  ROTATED ENV VARIANT  new task registered from clutter/, subclassing the benchmark
                        cfg.  Start with heading +/-30 deg.  Measure the FROZEN expert
                        on it first -- that number is the honest statement of how
                        specialised the current solution is, and it is cheap.
4  DIVERSE EXPERT       a pose FAMILY parameterised by row heading, plus wrist side and
                        roll.  Per-heading tournament (P33's method) rather than one
                        global pose.  This is the biggest piece of work and it should
                        not start until 2 and 3 have reported.
5  RETRAIN + RE-EVAL    3 seeds, strict metric, on both the original and rotated tasks.
```

**Subject to change**, as always — item 1's result may reorder everything below it. In
particular, if strict success turns out to be far below the lenient number, fixing *that* (a
gentler close, a different grasp height, a slower lift) may matter more than diversity.

---

## 5. What is now known to be wrong in the earlier documents

* **Every success number in `07_`–`13_` is an upper bound.** They are all `target_at_goal`,
  which ignores neighbour displacement. This includes 25 %, 57.7 %, 72.1 %, 73.3 % (expert) and
  68.1 % / 71.8 % (BC).
* **`13_STAGE2_BC_RESULTS.md` §6.2's "the policy has one failure mode" is withdrawn.** It had
  one failure mode *among the buckets being counted*. Dragging a neighbour was not counted.
* **The Stage-1 claim that the orthogonal grasp "stops the row's y-pitch being the binding
  constraint" is conditional on the row's heading**, which has never varied. It is not a
  general property of the task.
* **`06_EXPERT_DESIGN.md`'s single-pose recipe, and `pose_p33.json` itself, are specific to one
  row geometry.** They remain the right answer for `Rebot-ClutterExtract-v0` as shipped, and
  they are not a solution to the task Big Will is now describing.
