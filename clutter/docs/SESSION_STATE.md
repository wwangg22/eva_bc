# SESSION STATE — resume here

**Updated 2026-08-03, after Big Will set the threshold at 2 mm and lifted the `challenge/`
constraint for the clutter env.**

## ⚠ START HERE: `15_STRICT_METRIC.md`. THE TASK ITSELF CHANGED.

Big Will's decision: **2 mm**, and the clutter env may be edited. So this is no longer a
private re-scoring — **the environment's own success predicate now requires every neighbour to
stay within 2 mm of its spawn**, and a `distractor_disturbed` termination ends the episode the
moment one does not. Committed to `eva_rl` as `ceeb24c`.

```
                                       lenient      2 mm strict
frozen expert, 768 held-out episodes    73.3 %          16.4 %
```

**56.9 points. The expert retains 22.4 % of its measured rate. The mission target of ≈70 % is
53.6 points away, not 1.5.**

Three things make that number trustworthy, and they are the reason to believe it rather than
argue with it:

- **Two independent code paths agree to 0.1 pts.** Offline re-scoring on the lenient env gave
  16.3 %; the env-native predicate on the strict env gives 16.4 %.
- **The threshold is calibrated (P35).** Null action, full 700-step episode, 768 episodes:
  worst-case drift **1 µm**. The 2 mm cliff sits **2 097×** above the solver's own noise.
- **2 mm, 5 mm and 10 mm now select the identical 16.4 %.** With the termination in place the
  successes are *clean* — median neighbour displacement among successes is **0.00 mm**. There
  is no population of near-misses; the choice of threshold costs nothing in discrimination.

**The taxonomy collapsed to one bucket.** `time_out 0.0 %, target_dropped 0.0 %,
distractor_toppled 0.0 %, distractor_disturbed 83.6 %`. Toppling was never a separate failure
mode — a block must slide before it can tip, so the disturbance term fires first in every
episode that would have toppled. The old metric was counting only the tail of the disturbance
distribution.

**`DR2` no longer holds and the ladder must be re-entered a rung lower.** The pre-registered
rule was "expert clears 70 % → proceed to BC". It does not. **Fix the expert first** (§6 of
`15_`): all 56.9 points are one mechanism — the finger blades sweeping the neighbours during
the `close` phase, localised by P22 and reported at 71–76 % by every hazard table since P17.

---

## The prior headline (still true, now superseded in its numbers)

Big Will watched the 16 policy videos and found what ~5 000 measured episodes did not: the
benchmark's success predicate never checks whether a neighbouring block **moved**, only whether
it **toppled**. `mdp.target_at_goal` ends in `& ~any_distractor_toppled(env)`, and that is
`up_z < 0.75` — about 41 degrees of tilt. A block dragged across the table and set down upright
is a full success.

Re-scored on the same 768 held-out episodes, with a strict predicate that also requires every
neighbour to stay within N mm of its spawn:

```
                    lenient    <2mm    <5mm   <10mm |  median      p90    >50mm
expert (v3 holds)     73.3 %  16.3 %  22.4 %  30.9 % |  13.7 mm  205.5 mm  25.1 %
BC v3 seed 1          72.0 %  19.3 %  26.3 %  34.0 % |  11.9 mm  206.2 mm  22.0 %
BC v3 seed 2          69.7 %  17.6 %  23.4 %  30.5 % |  13.2 mm  206.8 mm  22.1 %
BC v3 seed 3          73.7 %  18.9 %  25.4 %  33.6 % |  11.7 mm  206.3 mm  22.4 %
                                                ----
BC mean               71.8 %                  32.7 %      expert 30.9 %
```

- **The expert falls 73.3 % → 30.9 % at 10 mm.** 42.4 points.
- The **median "success" displaces a neighbour 13.7 mm — more than the 12 mm free gap** the
  whole task is built around.
- **22–25 % of all episodes carry a neighbour to the goal zone** with the target. The p90 of
  ~206 mm matches the 120–277 mm row-to-goal distances almost exactly. This is Big Will's
  observation, with a number on it.
- **The ordering REVERSES under the strict metric.** Lenient: BC 71.8 % vs expert 73.3 %
  (−1.5). Strict: **BC 32.7 % vs expert 30.9 % (+1.8)**. The policy is *cleaner* than the
  demonstrations it was cloned from — it inherited the sloppiness rather than adding to it, and
  slightly smoothed it.

**The real success rate of everything built so far is about 31–33 %, not 73 %.**

The evidence was in the record all along: every hazard table since P17 reports a **`close`-phase
disturbance rate of 71–76 %**. It was measured, published in the docs, and never allowed to
reach the metric — because the predicate had already decided that only toppling counted.

## The three pieces of feedback, and the plan

Full detail, including the constraint tension on item 3, is `14_FEEDBACK_AND_NEXT.md`.

1. **Success must require neighbours to stay put.** Instrumentation DONE — `eval_flow.py` and
   `collect_demos.py` latch worst neighbour displacement and report strict success at
   2/5/10 mm. **Big Will must choose the threshold; it is a task judgement, not a
   measurement.** Gate 1's 85 % and the ≈70 % mission target were both set against the lenient
   predicate and need restating.
2. **The expert uses one grasp at one angle.** True, and by construction: P33's tournament and
   P34's freezing bought +19 points and made the dataset deliberately unimodal. It caps what BC
   can represent and ties everything to one row geometry.
3. **The row is always broadside to the gripper.** True — `ROW_X = 0.25`, fixed y-pitch, only
   per-block jitter (±10 mm x, ±5 mm y, target yaw ±0.20 rad). The row never rotates or
   translates as a unit.

**Items 2 and 3 are the same piece of work.** P28 showed mode-mixing hurts BC — but only when
the mode is *unobservable*. Randomising the row heading supplies the observable variable that
makes a multi-mode expert learnable.

## Where the 56.9 points are — P36/P37/P38/P39, `16_DISTURBANCE_ANATOMY.md`

The mechanism is now measured rather than assumed, and one measurement dominates:

> **Ablation B (P38): the same arm trajectory with the gripper forced open disturbs the row
> 0.0 % of the time** — 384 episodes, p90 0.311 mm, against 67.2 % when it closes normally.

So the arm can go anywhere in the row safely. **The only unsafe act is closing the jaw inside
the row.** Supporting facts:

- **When:** the 2 mm crossing is at the **first step of `close`** (p01 = 160, p99 = 174, and
  step 160 *is* that first step). It is a jump, not a creep — already 4.05 mm when it first
  exceeds 2. Kills "a slower close" and "shorten the close hold".
- **Which:** the inner pair, **100.0 %**, and **d1 : d2 = 2.50 : 1 (z = 10.8)** — an
  unexplained chiral bias worth exploiting.
- **Which way:** fore-aft. **|dx| is 9.2× |dy|.** The neighbour is *hooked and carried*, not
  shoved aside — `close` starts it (81 % of first crossings), `carry` accumulates it (79 % of
  total motion). Exactly Big Will's "grabs another box as well".
- **How far it reaches:** widening the row makes it vanish between 48 and 54 mm of pitch, so
  the fouling reaches **33–39 mm** from the target centre against faces at 27 mm — about
  **1.8× the blade geometry quoted in the docstrings since Stage 1**.

**Two fixes, one of which is Big Will's call:**

| | gain | note |
|---|---|---|
| jaw yaw-matching (`--yaw-gain 0`) | **+2.9 pts** | 16.4 → 19.3 %. Real, minor, keep it |
| **narrower gripper opening** | **+16.9 pts** | 16.4 → 33.3 %. ⚠ **an ENV change** — see below |
| **extract, then grasp** | untried | **first-ranked**, and needs no env change |

⚠ **The gripper opens to 90 mm to grasp a 36 mm block** (`_GRIPPER_OPEN = 0.045` per finger),
so each finger sweeps **24 mm of pure excess travel** through the row on every grasp — and
because `BinaryJointPositionAction` has exactly two states, **no policy can avoid it.** At
46 mm of separation (still 2.4 mm of clearance over the widest a yawed target presents) strict
success is **33.3 %** and enclosure *improves* from 17.2 % to 31.2 %. It costs 17 of the 57
points and it is imposed by the env, not chosen by the solution. **Changing it makes the task
easier, so it is Big Will's decision, not mine** (`16_` §2c.1).

**The fix that needs no env change (`16_` §5):** ablation B says the open jaw is safe anywhere
in the row, and **the target's own displacement is unconstrained** — `DISTURB_TOL` covers only
the four distractors. So: reach in with the jaw open, **pull the target clear of the row**,
close on it outside, carry. That is the extrinsic-dexterity solution the env's own docstring
describes (*"the correct first action is often not a grasp"*) and **no expert in this project
has ever tried it** — every one from P01 to `pose_p33` closes in situ. **P40 measures step 2
alone:** drag the target −x by 30–40 mm with the jaw open and watch the neighbours. Near 0 %
and the manoeuvre is viable; if dragging rakes the row it is dead and the opening is the only
lever left.

### Order (updated 2026-08-03 after the threshold decision)

```
1  STRICT METRIC        DONE AND SHIPPED.  2 mm, in the env, calibrated (P35), committed
                        (eva_rl ceeb24c), documented (15_STRICT_METRIC.md).  Gates restated.
2  FIX THE DISTURBANCE  <- THE WHOLE JOB NOW.  56.9 points sit in one mechanism and every
                        other failure bucket is empirically 0.0 %.  Levers never tried,
                        in order: (a) grip height vs blade sweep, (b) a straight-up lift
                        before any lateral motion, (c) roll/clocking optimised against
                        DISTURBANCE rather than topple, (d) a slower close.  All paired
                        physics probes on identical spawns -- minutes each, and immune to
                        the 9.8-pt BC seed noise.  Score on the env's own target_at_goal.
3  RE-MEASURE GATE 2    cheap: BC was +1.8 pts ABOVE its expert under the strict re-scoring,
                        so the port and the cloning are probably fine and the deficit is
                        entirely upstream.  One eval run per seed, no training.
4  ROW ROTATION         now a legitimate env edit (constraint lifted).  Plan: a cfg knob
                        defaulting OFF on -v0 so every baseline stays comparable, plus a
                        -Rot-v0 variant.  Measure the FROZEN expert on it first.
5  UNIMODALITY CONTROL  task #17, written and unrun (`collect_demos.py --multimodal`).
6  DIVERSE EXPERT       a pose FAMILY parameterised by row heading + wrist side + roll.
7  RETRAIN + RE-EVAL    3 seeds, strict metric, on both the original and rotated tasks.
```

**Do not train anything before 2 reports.** Cloning a 16.4 % expert costs ~50 min per seed and
the ceiling is the expert's rate.

**A geometric warning for item 4**, from numbers already in hand: `target_axis()`'s docstring
records that a blade reaches ~47 mm along the opening axis against **7.8 mm** of perpendicular
clearance, so the target's own ±11.4 deg of spawn yaw already swings a blade corner 9.3 mm —
more than the margin. Rotating the *row* by ±30 deg is a much larger version of that. **Expect
the orthogonal grasp to stop working at some heading, and expect that to be the finding rather
than a bug.**

## What was built this session (all verified, all compiles)

`clutter/act/` — new:

| file | what it does |
|---|---|
| `collect_demos.py` | demo generator **and** the Gate-2 instrument, one code path |
| `dataset.py` | 42-D HDF5 dataset (a deliberate copy of eva_bc's 41-D one; `13_` §1.1) |
| `train_flow.py` | rectified-flow BC over the vendored, unmodified `act/modeling_flow.py` |
| `policy_runner.py` | `ChunkController` + `load_checkpoint`, shared by eval and video |
| `eval_flow.py` | batched sim eval, failure taxonomy, strict-success re-scoring |
| `analyse_demos.py` | offline dataset audit; measures the N2 ambiguity with no simulator |
| `record_video.py` | one env, one episode per file, camera 0.67 m out, 60 deg lens |

Results, **all lenient unless stated**:

- **Gate 2a/2b (the `env.step` port): PASSED**, −0.2 / +0.6 / +0.6 points, flips at the ~8 %
  noise floor.
- **P29 refuted its own registered prediction twice.** A joint-space approach buries a finger
  4.85 mm in a neighbour (5.5 %); a *geometrically perfect* forward Cartesian solve scores
  **0.0 %** because it lands **1.90 rad** from the frozen chain on `joint6` — a different IK
  branch at the same tool point — with 100 % topple and 96 % of targets still delivered.
  Solving **backward** from the frozen pose makes the seam an identity: 73.0 %.
- **P27 is the best result of the stage.** The close hold is flat 560 → 40 physics steps with
  **100.0 % enclosure at every duration**, and shortening the other holds makes the expert
  **+3.9 points better** rather than trading against learnability.
- **`demos_v3`**: 774 demos, 325 steps, static 10.2 %, ambiguity floor 0.0607 (v1: 0.0989).
- **BC on v1**: 58.1 / 68.5 / 77.7 % (sd 9.8). **BC on v3**: 69.7 / 72.0 / 73.7 % (sd 2.0).
  The mean gain is **not** established at n=3 (Welch t = 0.64); the **variance collapse is**
  (F = 23.6, p ≈ 0.041) and the worst seed improved 11.6 points.
- **Training-seed variance dominated everything**: 9.8 pts, vs 1.6 binomial and 0.4 for the
  policy's own sampling. **The training loss does not predict success** — one seed had a lower
  final loss and scored 10.4 points worse. Select checkpoints in the simulator.

## Video

`clutter/runs/videos/` — 16 episodes of `bc_v3_s3`, one env each, 1280×720 at 25 fps
(half speed). 11 SUCCESS / 5 `distractor_toppled` under the lenient predicate. All five topples
terminate within a **six-step window (171–177)**, i.e. 6–12 env steps into the carry;
termination lags onset, so that is consistent with the close-phase attribution.

**These videos are what found the metric bug.** Watching beat reading the predicate.

## The three things most likely to bite next

- **Any last-write-wins metric is invalid under `env.step`.** Isaac Lab auto-resets a done env
  from *inside* the step call. Bit twice this session (R23). Latched quantities are safe.
- **Two IK branches at the same TCP are invisible to every statistic in this codebase** and
  differ by 1.90 rad. Check joint-space agreement at every join between solved paths.
- **`clip_actions`.** `|a|max = 4.63` on `joint4`; all six joints leave `[−1, 1]`.
  `clip_actions = 1.0` makes the manoeuvre unreachable for rl_games. BC is unaffected
  (`cfg.clip is None`, checked at startup).

## Hard-won rules

Stage 0: every negative result needs a positive control.
Stage 1: **pair everything.** Hazard rates, not raw counts. Verify segments, not waypoints.
Stage 2: compute the power before running the arm. A predictor mined from a dataset must be
tested on data that had no part in producing it. When nothing predicts the quantity, select on
it. "Frozen" means frozen — check which parts actually are.
Stage 2b: **audit the seam, not just the segment.** When a path must end at a known pose,
**solve backward from it.** Measure the irreducible loss before training — and do the
derivation. **Report the constraint metric separately from the task metric.**

**Stage 2c, the expensive one:** *"score with the env's own predicates"* is right, and it is
not sufficient. **Ask whether the predicate encodes the task.** Here it did not, for eight
stages and ~5 000 episodes. **A taxonomy only constrains what it enumerates** — §6.2 of `13_`
called the failure mode singular, and it was singular only among the buckets being counted.
**And watch the videos early.**

## Constraints still in force

- All eva_bc work stays in `eva_bc/clutter/`.
- **eva_rl's `challenge/` package may now be edited — for the CLUTTER env only.** Big Will,
  2026-08-03: *"feel free to edit the environment in ReBOT_RL. Just ensure to update the
  corresponding docs, and make sure you only touch the environment YOU ARE WORKING on."*
  So: `mdp/clutter.py`, `clutter_env_cfg.py`, `scripts/test_clutter_env.py` and the clutter
  docs are in scope. `mdp/common.py`, `mdp/rewards.py`, `mdp/terminations.py` and the other
  env cfgs are **shared** and remain out of scope.
- **Commit after validating and after updating the docs. Pull first.** Both repos are
  committed locally and **not pushed** — the standing rule is still that Big Will says when.
- ⚠ Before any push: the root `.gitignore`'s bare `runs/` swallows `clutter/runs/` — now
  ~1.5 GB. Evidence (JSON + logs) is ~8 MB and worth committing; checkpoints (~1.1 GB), the
  HDF5s and the 34 MB of video are regenerable. Recommendation: commit `runs/*.json` and
  `runs/*.log`, ignore `*.hdf5`, `*.pt`, `*.mp4`. **Big Will's call.**
- One GPU job at a time (10 GiB card). `python -u`, or output buffers and you see nothing.
