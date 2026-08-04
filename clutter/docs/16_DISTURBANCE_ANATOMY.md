# 16 — The anatomy of the disturbance (P36), and the fix it selects (P37)

**2026-08-03.** Under the 2 mm rule the expert scores **16.4 %**, and `distractor_disturbed`
is the only non-zero failure bucket — 83.6 %, with `time_out`, `target_dropped` and
`distractor_toppled` all at **0.0 %** (`15_STRICT_METRIC.md` §4). So the entire remaining task
is this one mechanism and **56.9 points depend on getting it right**, which is a good reason to
measure it before choosing a fix rather than after.

Eight stages of hazard tables had already said "it is the `close` phase" and stopped there. A
*rate* per phase is not a mechanism.

---

## 1. P36 — three predictions, two confirmed, and the third is the one that matters

`probes/p36_disturb_anatomy.py`. The shipping expert (frozen pose, frozen chain, frozen
approach, `--close 40 --holds-scale 0.25` — the exact configuration behind the 16.4 %) driven
through `env.step`, recording **every distractor's displacement at every step with its phase
label**. 768 episodes, seeds 88000–88005.

Run on **`-Lenient-v0` deliberately**: the strict env terminates at the crossing and auto-resets
the scene from inside `env.step` (R23), so it can report *that* the threshold was crossed and
nothing about what happened next. Same physics, termination off, whole trajectory visible; the
crossing is then found offline.

Predictions were written into the file's docstring before the run.

| # | prediction | result | |
|---|---|---|---|
| 1 | > 80 % of first crossings in `close` | **81.1 %** close, 18.9 % carry | **CONFIRMED** |
| 2 | > 90 % of first crossings on the inner pair | **100.0 %** | **CONFIRMED** |
| 3 | displacement predominantly along **y** | **\|dx\| is 9.2× \|dy\|** | **REFUTED** |

### 1.1 The refuted one selects the fix

```
   (3) WHICH DIRECTION, at the moment of crossing  [mm]
        |dx| (toward/away from robot)  median  4.050   mean  5.500
        |dy| (across the row)          median  0.442   mean  1.124
        y-dominant in 2.3% of crossings
```

**The neighbour is dragged fore-and-aft, not shoved sideways.** I had the mechanism backwards.
The reasoning behind prediction 3 was: the jaw opens and closes along x, the blade's
*perpendicular* half-span (±19.2 mm) is what comes near the neighbour faces (±27 mm), so the
neighbour gets pushed across the row. That is wrong about what happens *after* contact.

What the numbers say instead:

1. Yaw-matching swings a blade corner into the neighbour's footprint. The margin is 7.8 mm and
   the swing at the target's full ±11.4 deg of spawn yaw is **47 · sin(11.4°) = 9.3 mm**. This
   part of `target_axis()`'s docstring was right and had been sitting there unused.
2. Once the corner is *inside* the footprint, the motion that follows is the motion the finger
   is already making — **along the opening axis, i.e. x**. The blade does not push the
   neighbour out of its slot; it hooks it and takes it along.

That is why the fix is **jaw yaw-matching**, not grip height. Raising the grip does not change
the y-overlap that lets the corner in, and it does not change the direction of travel.

### 1.2 It is a jump at one specific step, not a creep

```
crossing step: p01 160   median 160   p90 166   p99 174
```

Mapped onto the schedule (approach 0–77, settle 78–79, descend 80–154, predwell 155–159,
**close 160–164**, carry 165–278, …):

**Step 160 is the first step of the `close` phase.** At least 99 % of crossings happen at or
after the instant the fingers begin to shut, and the whole p01–p99 range spans **14 steps**.
And the block is already at **4.05 mm** when it first exceeds 2 mm — it does not drift across
the threshold, it is displaced twice the threshold in a single control step.

This kills two candidate fixes outright. **A slower close cannot help** (P27's lever): the
damage is done in the first step, before duration is a variable. **Censoring or shortening the
`close` hold cannot help** either, for the same reason.

### 1.3 d1 : d2 = 2.50 : 1, and that is not noise

```
   distractor_0   outer      0    0.0%
   distractor_1   INNER    457   71.4%
   distractor_2   INNER    183   28.6%
   distractor_3   outer      0    0.0%
```

`z = 10.8` against an even split. **One side of the row is hit two and a half times more often
than the other**, which a symmetric blade sweep cannot produce. Something about the manoeuvre
is chirally biased — the wrist side, or the roll P30 measured at +7.0 paired, or the
approach's own asymmetry. It also means **there may be a strictly better clocking available**:
whatever makes d2 safer 2.5× more often is a property of the pose, not of the task.

The outer pair is **never** the first to cross, in 768 episodes. They sit 84 mm out, 57 mm
beyond the blade's perpendicular half-span, so this is a sanity check that passed: the geometry
is behaving as the geometry says it should.

### 1.4 First crossing is in `close`; most of the *motion* is in `carry`

```
   (4) HOW MUCH each phase contributes  [total mm of worst-block motion]
        carry      52116.7 mm   78.8%
        close      12118.4 mm   18.3%
        release     1546.0 mm    2.3%
        withdraw     294.2 mm    0.4%
        ...          <60 mm     <0.1%   (dwell, predwell, descend, approach, final, settle)
```

Read together with §1.2 this is the whole story, and it is **exactly what Big Will described
from the videos**:

> "the robot actually grabs another box as well as the one of interest"

The jaw shuts, hooks the inner neighbour in the first control step, and then **carries it**.
`close` starts it (81 % of first crossings) and `carry` accumulates it (79 % of total motion).
The neighbour is not knocked aside — it is picked up.

`descend` and `approach` together contribute **2.5 mm across all 768 episodes**, so the entire
pre-grasp trajectory is clean and P29's approach work is confirmed sound under the new metric.

### 1.5 Spawn tightness predicts failure

```
   (5) min free gap at spawn [mm]
        disturbed  median  9.37   mean  9.84
        clean      median 12.93   mean 12.82
        difference +2.98 mm
```

Clean episodes start with **3 mm more room**. Consistent with a fixed-size intrusion: a corner
that reaches 9.3 mm into a 7.8 mm margin fouls a tight spawn and clears a loose one. It also
means the residual 16.4 % is partly *luck of the draw*, and a fix that removes the intrusion
should benefit the tight spawns most.

---

## 2. P37 — the yaw-matching sweep

The lever exists already and has never been swept. `ClutterExpert.target_axis(gain)` rotates
the commanded opening axis `gain` of the way from the nominal axis toward the target's own:
`gain = 1.0` meets the 36 mm faces squarely and is what **every number in this project has
used**; `gain = 0.0` keeps the jaw on the nominal axis and eliminates the corner swing.

Exposed as `collect_demos.py --yaw-gain` / `--no-match-yaw` so the sweep runs through the same
code path as the 16.4 % baseline rather than a probe-local copy.

**There is prior evidence, and it was dismissed.** From `target_axis()`'s own docstring:

* **P16** measured that matching yaw *raised* the close-phase contact count from 65/128 to
  **93/128** while improving every grip statistic — "and it was not recognised at the time".
* **P19** then found the square jaw outscoring the matched one end to end, **69.5 % vs
  64.1 %**, "and it was written off as noise".

Under the lenient predicate a 5.4-point gap sat inside the ±12-point interval that P26 later
established, so writing it off was defensible at the time. Under the strict predicate the
quantity P16 measured — *contact count* — is no longer a curiosity; it is the metric.

Sweep: `gain ∈ {1.00, 0.75, 0.50, 0.25, 0.00}` plus `--no-match-yaw`, all six paired on seeds
88000–88005, on the **strict** env so `target_at_goal` is the 2 mm predicate.

**Registered prediction:** strict success rises monotonically as `gain` falls, with the largest
single step between 1.00 and 0.75, and enclosure stays at 100 % throughout (the target is
36 × 30 mm and an 11.4° jaw mismatch costs only 30/cos(11.4°) − 30 = 0.6 mm of effective
width, which the 12 mm stroke absorbs easily).

**What would refute it:** enclosure falling with `gain`. That would mean the square jaw cannot
grip a yawed target and the two objectives are genuinely opposed, in which case the fix is a
*partial* gain and the optimum is interior.

### 2.1 Results — the prediction half held, and the effect is small

```
yaw_gain     1.00    0.75    0.50    0.25    0.00   | no-match-yaw
strict      16.4 %  17.3 %  17.1 %  17.1 %  19.3 %  |   13.4 %
enclosure   17.2 %  15.6 %  14.8 %  17.2 %  23.4 %  |     --
```

**+2.9 points** from removing the yaw match entirely, and enclosure *rises* rather than
falling — so the two objectives are not opposed and my refutation criterion did not trigger.
The direction is right and the ordering is roughly monotone. But the prediction that the
largest step would be 1.00 → 0.75 was wrong: the curve is flat across 0.75–0.25 and all the
movement is in the last step to 0.00.

`--no-match-yaw` is **worse than the baseline** (13.4 %). Dropping the orientation constraint
entirely is not the same as pinning it to the nominal axis: `refine` is then free to pick any
wrist roll, and it evidently picks worse ones. This is a useful negative control — it shows
the gain is from *where the jaw points*, not from *relaxing a constraint*.

**+2.9 of the 56.9 points.** The mechanism I inferred from P36 is real and minor. That is not
a good enough return to keep refining a model built on unverified docstring geometry, which is
why P38 stops inferring and measures.

---

## 2b. P38 — two ablations that assume no geometry at all

`probes/p38_disturb_ablation.py`, 384 episodes per cell (seeds 88000–88002).

### 2b.1 Ablation B — the arm alone disturbs nothing

Identical commanded arm trajectory, `close` forced False at every step. The arm visits exactly
the same poses; only the finger motion is removed.

```
close as normal        disturbed  67.2 %   median 4.805 mm   p90 44.482 mm
gripper FORCED OPEN    disturbed   0.0 %   median 0.000 mm   p90  0.311 mm
```

**0.0 %. Not "low" — zero, on 384 episodes, with a p90 of 0.311 mm.**

Prediction 1 confirmed and then some (< 10 % predicted). **100 % of the disturbance is the
finger closing motion.** The whole approach, descent and pre-grasp dwell — every pose the arm
holds, at every point in the row — is completely clean. P22's eight-stage attribution to the
fingers is correct, and every finger-directed lever is aimed at the right body.

This is the control that should have been run before P37, and it costs two minutes.

### 2b.2 Ablation P — widen the row until it stops, and read off the reach

Sweep `ROW_PITCH` with everything else identical. Purely diagnostic — the shipping task keeps
its 42 mm pitch and 12 mm gap.

```
   pitch  free gap   disturbed   median      p90     neighbour inner face
    42 mm   12.0 mm     67.2 %    4.805    44.482        27.0 mm
    48 mm   18.0 mm     17.4 %    0.000     8.954        33.0 mm
    54 mm   24.0 mm      0.0 %    0.000     0.354        39.0 mm
    60 mm   30.0 mm      0.0 %    0.000     0.370        45.0 mm
    70 mm   40.0 mm      0.0 %    0.000     0.000        55.0 mm
```

**The fouling reaches 33–39 mm from the target's centre.** Prediction 2 said 25–28 mm, from
the documented blade half-span of 19.2 mm plus the 9.3 mm yaw swing. **REFUTED — the reach is
about 1.8× the documented blade.**

So the `target_axis()` docstring's geometry does not describe what is actually happening, and
the corner-intrusion model was the wrong model. It is not a marginal 9.3-into-7.8 mm clip that
better clocking could dial out; the finger sweeps a region far wider than the row's spacing.
That is why P37 bought only 2.9 points.

---

## 2c. P39 — the gripper opens to 90 mm to grasp a 36 mm block

Follow the ablations to their conclusion. If the disturbance is entirely the finger *closing
motion*, and the fouling region is far wider than the gap, then the quantity that matters is
**how far each finger travels while shut**.

```
_GRIPPER_OPEN = 0.045 per finger  ->  90 mm of separation
target depth  36 mm, effective 41.2 mm at the full 11.4 deg of spawn yaw
                                  ->  48.8 mm of excess, 24.4 mm PER FINGER
```

**Each finger sweeps 24 mm of pure excess travel through the row on every single grasp**, and
the policy has no say in it: `BinaryJointPositionAction` has exactly two states, so the
opening width is an environment constant, not an action.

Swept via `collect_demos.py --grip-open`, on the strict env, 768 episodes per cell:

```
separation  excess/finger   STRICT    enclosure   disturbed
   90 mm       24.4 mm      16.4 %     17.2 %      83.6 %     <- shipping
   70 mm       14.4 mm      16.7 %     14.8 %      83.3 %
   60 mm        9.4 mm      19.5 %     18.8 %      80.5 %
   52 mm        5.4 mm      27.3 %     26.6 %      72.7 %
   46 mm        2.4 mm      33.3 %     31.2 %      66.7 %
   42 mm        0.4 mm      39.1 %     35.2 %      60.9 %
   38 mm        0.0 mm      41.9 %     35.9 %      58.1 %   <- below the yawed block's own depth
```

**16.4 % → 33.3 %, +16.9 points**, at 46 mm — which still leaves 2.4 mm of clearance per finger
over the widest the target can present. Monotone, and **enclosure rises with it** (17.2 % →
31.2 %), so this is not a trade against grip quality: the narrower opening grips *better*
because the fingers spend less time ploughing.

⚠ **The 42 mm and 38 mm rows are below the geometric limit** (41.2 mm) and are reported for
the shape of the curve only. At those openings the jaw cannot clear a fully yawed target on
approach and part of the gain is the gripper bulldozing the target into place. **Do not adopt
them.** The defensible values are 46 mm (33.3 %) and 52 mm (27.3 %).

### 2c.1 This is an environment change, and it is Big Will's call

Unlike everything else in this document, the fix is not in the expert. Reducing the opening
changes what the robot can do in this task, so it changes the benchmark. Three things make it
a reasonable change, and one makes it worth asking about first:

* The gripper is **binary**. No policy — scripted or learned — can choose a narrower opening,
  so the 24 mm sweep is a floor imposed by the env on every possible solution.
* 90 mm of separation for a 36 mm block in a row with a **12 mm** gap is arguably a
  task-design oversight rather than a difficulty the task means to pose. The stated skill is
  "collision-aware approach and gentle contact", not "cope with an oversized jaw".
* It costs **17 points** of an achievable 57.
* But it does make the task easier, and the mission is to solve the task as posed. **Asking.**

The alternative that needs no env change is in §5.

---

## 5. The fix that needs no environment change: extract, *then* grasp

P38's ablation B is the key to this. **The arm can go anywhere in the row without disturbing
anything** — 0.0 % over 384 episodes. The only unsafe act is closing the jaw while it is
inside the row.

And there is an asymmetry the task hands us for free: **the target's own displacement is
unconstrained.** `DISTURB_TOL` applies to the four distractors. Nothing forbids sliding,
dragging or tipping the *target*.

So the manoeuvre the constraints actually permit is:

```
1  reach in with the jaw OPEN                        (measured safe: 0.0 % disturbance)
2  pull the TARGET clear of the row -- toward the robot, along -x, with a fingertip or
   the closed jaw, or by closing on it only partially
3  close on it OUTSIDE the row, where the 24 mm sweep fouls nothing
4  carry to the goal
```

This is the "extrinsic dexterity" solution the environment's own docstring describes —
*"the correct first action is often not a grasp"* — and **it has never been tried.** Every
expert this project has built, from P01 to the frozen `pose_p33`, closes the jaw in situ.

It is also what makes the P39 result interpretable rather than merely convenient: the reason a
90 mm jaw is survivable at all is that the task *intends* the grasp to happen somewhere else.

Ranked against the alternatives it is first, because it is the only candidate that (a) needs no
benchmark change, (b) attacks the whole 57 points rather than a slice, and (c) is directly
supported by a measured 0.0 %. It is also the largest piece of work: a new pose family, a new
chain, and P33's tournament re-run against it.

**Next probe (P40):** measure step 2 alone. Reach in, drag the target −x by 30–40 mm with the
jaw open, and measure the neighbours. If that is also near 0 % the manoeuvre is viable and the
rest is engineering; if dragging the target rakes the row, this is dead and the gripper opening
is the only lever left.

---

## 3. What this rules out, before spending time on it

| candidate fix | status | evidence |
|---|---|---|
| **extract then grasp** (§5) | **first-ranked, untried** | ablation B: the open jaw is 0.0 % unsafe anywhere in the row, and the target's own motion is unconstrained |
| **narrower gripper opening** (§2c) | **measured, +16.9 pts** — but it is an env change, asking | P39; enclosure improves too |
| roll / clocking | **live.** §1.3 has 2.5× of chiral asymmetry unexplained | d1 : d2 = 2.50 : 1, z = 10.8 |
| jaw yaw-matching | **done, +2.9 pts.** Real, minor, keep it | P37; `gain = 0` also improves enclosure |
| a straight-up lift before lateral motion | **demoted.** It cannot prevent the hook (that is `close`), and once 2 mm is crossed the magnitude no longer scores | §1.2, §1.4 |
| grip height | **dead.** The fouling reach is 33–39 mm, far wider than any clocking or height can dial out; and the blocks are 70 mm tall so no height avoids them | P38 pitch sweep |
| a slower close | **dead.** The crossing is in the *first* close step | §1.2 |
| shortening / censoring the `close` hold | **dead**, same reason | §1.2 |

Note how much of this ordering was produced by *refuted* predictions rather than confirmed
ones. P36's prediction 3 killed grip height and the gentle close; P38's prediction 2 killed the
corner-intrusion model that P37 had just been built on. The two confirmed predictions (phase,
block) told me nothing I had not already assumed.

---

## 4. Method notes worth keeping

**`act/schedule_utils.py`.** `expand` and `approach_prefix` were extracted from
`collect_demos.py` so P36 instruments *the manoeuvre that was measured* rather than a
re-implementation of it. Verified behaviour-preserving: seed 88000 scores **17.2 %** after the
extraction, identical to before. Same reasoning as `policy_runner.py`.

**Measure the direction, not just the rate.** Eight stages reported a close-phase hazard
*rate* of 71–76 % and every one of them was correct. None recorded which way the block went,
and that single number — `|dx|` vs `|dy|` — is what reordered the candidate fixes and killed
two of them.

**A hazard rate per phase is not an attribution.** Phases are not independent: `close` and
`carry` both have high hazard rates, but `close` is where the threshold is *first* crossed and
`carry` is where the *magnitude* accumulates. Those are different quantities with different
fixes, and only the first-crossing statistic separates them.

**Run the ablation before the sweep.** P37 swept a five-point parameter grid to find +2.9
points, on a mechanism inferred from geometry in a docstring. P38's ablation B — the same
trajectory with the gripper forced open — took two minutes, returned **0.0 %**, and was worth
more than the whole sweep: it established that the entire effect lives in one motion, which is
what made the pitch sweep worth running and what made §5 visible. **A parameter sweep assumes
you already know which parameter; an ablation tells you.**

**Docstring geometry is not measured geometry.** The blade's "±19.2 mm perpendicular, ~47 mm
along" has been quoted in this project since Stage 1 and used to justify the orthogonal grasp.
The pitch sweep puts the actual fouling reach at **33–39 mm**, about 1.8× that. The
docstring's *conclusion* (straddle fore-and-aft, do not enter the row gaps) survives; its
numbers do not, and a fix was designed on them.

**Look for the constraint that is absent.** The whole of §5 follows from noticing that
`DISTURB_TOL` covers the four distractors and says nothing about the target. Reading a spec for
what it *does not* forbid is a different act from reading it for what it requires, and it is
the same failure mode as `15_`: the metric was read for what it checked, never for what it
left out.
