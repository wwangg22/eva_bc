# 07 — STAGE 0 RESULTS: measurement, and six wrong answers on the way to a working grasp

> ## ⚠ RETRACTION 2026-08-02 — §11's "GATE 0 DOES NOT PASS" IS WITHDRAWN
>
> **Gate 0 passes.** Sections 11 and 13 below were written before the **orthogonal grasp** was
> tested, and they are wrong. Big Will asked whether the gripper had been tried rotated 90° —
> fingers straddling the target *fore and aft* instead of across the row — and it had not: the
> probe covering that family (`p02`) is marked **void** in §3 of this very document and was
> never re-run after the CEM was fixed.
>
> It works. **Nominal row: 100 % held, 0 % topple.** **256 random spawns: 100 % enclosed,
> 99 % target-at-goal, 25 % end-to-end success.** No env change is needed, and the
> recommendation in §13.2 to relax `ROW_PITCH` is withdrawn with the verdict.
>
> Read **§15 and §16** for the result. Everything in §§1–12 remains accurate *as measurements
> of the cross-row grasp* and is worth keeping — it is why the orthogonal grasp is now known to
> be necessary rather than merely sufficient. `HANDOFF.md` is the current summary.
>
> ### ⚠ FURTHER RETRACTIONS FROM STAGE 1 — see `08_STAGE1_RESULTS.md` §7
>
> Two more items in this document have since been superseded:
>
> - **The 25 % / 99 %-at-goal figures above were measured through a trajectory containing a
>   segment with a 100 % contact hazard** (`carry→place` deviated 108 mm from its Cartesian
>   line — an IK branch flip between independently-solved waypoints) **and a grasp pose whose
>   wrist sat inside a neighbour**. The expert now scores **57.7 % pooled over 768 episodes**.
> - **"Success rises monotonically with clearance" (§16) is withdrawn.** On 768 episodes
>   through a trajectory without those defects the relationship **inverts**: 82.4 % success at
>   0–4 mm of free gap against 52.1 % at 8–10 mm. Working explanation: a tightly packed row
>   supports itself, so a nudged block has nowhere to fall.
> - **P01's finger geometry is finally replaced** rather than merely retracted. The collision
>   meshes, read through instance proxies: blades **±19.2 mm perpendicular to the opening axis,
>   ~47 mm along it**. That number is what identified the real residual failure mode.

*Started 2026-08-02. This is the running record of Stage 0. Every probe, every failure, and
the evidence for each diagnosis. Beliefs pre-registered in `02_PLAN.md` are scored at the end.*

**Headline: the grasp primitive now works** (`rose 100 %`, `enclosed 98 %` on an isolated
target), but only after five successive diagnoses, four of which were defects in my own
instrument rather than facts about the task. That distinction is the whole point of the
control experiment and is why it was worth building before measuring anything.

---

## 0. What was run

| probe | question | status |
|---|---|---|
| `probes/_kin.py` | shared FK / CEM / waypoint core; seed of the Stage-1 expert | working |
| `p01_gripper_geometry.py` | finger collider extents, blade width, resolved friction | **run** |
| `p02_orientation_envelope.py` | attainable grasp orientations at the row | **run — result void, see §3** |
| `p03_grasp_control.py` | control grasp: solo vs clutter vs pre-singulated | **run** |
| `p04_reach_global.py` | global (non-local) sample of the attainable set | **run** |
| `p05_pinch_trace.py` | step-by-step commanded-vs-achieved trace of one grasp | **run** |

Environment smoke test (`eva_rl/scripts/test_clutter_env.py`, 64 envs) **PASSES** all rungs:
row geometry, topple constraint positive and negative, goal predicate, three negative
controls. Measured free gaps in one sample: **14.7 / 13.0 / 15.2 / 10.4 mm** — consistent with
the 7.0–18.8 mm distribution the write-up quotes.

---

## 1. P01 — finger geometry and friction

### 1.1 Two USD traversal traps, both silent

The first run reported `collision prims: 0` for both fingers and `0 material(s)` for the
table. Both were false. The arm USD splits `Geometry` and `Physics` scopes and the stock table
is `table_instanceable.usd`; **a plain `Usd.PrimRange` stops at the instance boundary.** The
result is not an error — it is a confident zero.

Fixed with `Usd.PrimRange(root, Usd.TraverseInstanceProxies())`. This matters beyond this
probe: any future code reading colliders or materials off this stage has the same trap.

### 1.2 The friction is not what the analysis assumed — Q5 answered

| surface | μ_s | μ_d | source |
|---|---|---|---|
| blocks (target and distractors) | 0.90 | 0.75 | authored in `clutter_env_cfg._block` |
| **table top** | **1.00** | **1.00** | `TableSurface` material inside the instanced USD |
| ground plane | 0.5 | 0.5 | — |

PhysX combines by **average** by default, so the effective block-on-table coefficient is
**μ ≈ 0.95**, not the 0.9 every earlier calculation used. Tip-vs-slide thresholds move:

| push axis | b | h_crit at μ=0.9 (assumed) | **h_crit at μ=0.95 (measured)** |
|---|---|---|---|
| across the row (y) | 30 mm | 16.67 mm | **15.79 mm** |
| along the row (x) | 36 mm | 20.00 mm | **18.95 mm** |

The correction goes the *wrong* way — higher friction means blocks tip more readily, not less.
Pushing is slightly harder than the plan assumed, not easier.

### 1.3 The finger collider cannot be read from its bounding box — Q1a/Q1b are NOT answered

P01's verdict section reported "outward finger thickness t = 19.76 mm ⇒ strategy A is dead".
**That number is not trustworthy and the conclusion drawn from it is withdrawn.** The evidence
against it is internal to the probe:

- The finger collider is a `Mesh` with `physics:approximation = 'convexDecomposition'` and
  **8 649 points** per finger.
- Its axis-aligned bound spans 38.44 mm along the opening axis *centred on the finger origin*,
  which would put the two inner faces 51.7 mm apart at `q = 0.045`.
- `CHALLENGE_SUITE` C3 **measured** the clear gap at that aperture as **89.07 mm**, fitted on
  30/45/70 mm gauges with a 0.035 mm residual.

Those cannot both describe the jaw. An AABB of a non-convex, non-axis-aligned mesh contains a
lot of air, and a convex *decomposition* is a union of hulls that the AABB does not
characterise. The honest conclusion is that **the outer width of the open gripper is not
recoverable from USD geometry** and must be measured by contact. P02 §2 was built to do that
(sweep the TCP across the row, see what moves) and is the right instrument; it has not yet
produced a number because of §3.

> **Retraction.** `HANDOFF.md` F2 and `06_EXPERT_DESIGN.md` §3 argue that every open-gripper
> placement encloses two blocks, using a zero-thickness-finger idealisation to derive two
> 6.94 mm alignment windows. The *arithmetic* is fine; what is now clear is that it rests on
> an unmeasured finger thickness, and the AABB-derived value that seemed to settle it does
> not. **Strategy A is neither confirmed dead nor alive.** Treat the two-window claim as an
> open question until the contact caliper runs.

---

## 2. P04 — the arm's actual geometry at the row

This is the measurement that should have come first, and it reframes everything.

### 2.1 The home pose, read rather than assumed

```
q_arm  = [0.0, -1.35, -0.3, -0.85, 0.0, 0.0]        (_START_POSE)
TCP    = (387.7,   0.0, 164.5) mm     env frame, table top = z 0
end    = (428.8,   0.0, 156.1) mm     gripper_end origin
a_hat  = (-0.980, 0.000, +0.199)      approach axis (gripper_end local -X)
o_hat  = ( 0.000, -1.000, 0.000)      opening axis
fingers at y = -45.07 and +45.07 mm, both at the TCP's x and z
```

**`a_hat · x̂ = −0.98`: this arm's fingers point back toward the robot.** `gripper_end` sits
41 mm *further from the base* than the TCP. So a grasp is made by putting the wrist **beyond**
the object and closing the jaw as it comes back in −x. Every intuition of the form "the wrist
is behind the object and the fingers reach forward onto it" is backwards for this arm, and I
had that backwards for four probes.

The finger positions also confirm the TCP convention end-to-end: the TCP is exactly midway
between the two finger body origins, 90.1 mm apart at `q = 0.045`. `TCP_OFFSET = (-0.0419,0,0)`
is right.

### 2.2 Global sampling is too weak to characterise a point

819 200 uniform joint draws put **3** samples within 10 mm of `(0.250, 0, 0.055)` — 0.0004 %.
That is not evidence of unreachability; it is evidence that uniform joint sampling has almost
no density in a 10 mm ball. `reachability_map.py` gets away with it because it quantises to
**2 cm voxels**.

So P04's headline verdict — *"the arm cannot point its fingers forward-and-down here"* — is
**correct but for a weaker reason than stated**: what it actually shows is that among the
orientations that *were* sampled, plus everything the CEM found over hundreds of restarts,
`a_hat · x̂` never exceeded −0.65. Combined with §2.1 that is coherent: the arm's only
orientation family at this point has the fingers pointing back and somewhat up.

---

## 3. P02 — void, and why

P02 reported **0 of 48 orientation cells attainable** and therefore ran no physics at all.
That verdict is **not valid**. Three defects, in order of discovery:

**(a) The cost let orientation outrank position.** `cost = |dp| + w·(axis terms)` with `|dp|`
in metres prices 1 mm at 0.001 against axis terms of order 0.25. The search walked **340–520 mm**
away from the target to perfect an axis it could not otherwise reach. Fixed by weighting.

**(b) Overcorrecting made it worse in the opposite direction.** At `w_pos = 20` position
dominated so completely that the axis terms became noise, and the search returned flipped
wrist poses. The two failures are symmetric and neither is a fact about the arm.

**(c) The floor penalty was silently inert.** `low_z` was the minimum over *all* bodies, and
`base_link` sits at z = 0, so `min ≡ 0` and the penalty never fired. The CEM happily returned
grasp poses with `gripper_end` at **z = 17 mm — inside the table**.

Final form, now in `_kin.ArmKin.cem`: position as a **hinge** (free within `pos_tol = 1 mm`,
then 0.2 per mm), floor as a hinge over the *gripper* bodies only, axis terms as bounded
tiebreakers, plus **multi-restart** from uniform draws. A weighted sum was the wrong model;
this is a constrained problem.

**P02 must be re-run under the corrected core before any of its claims are used.** Its
physical-caliper section (§2, the contact measurement of the gripper's outer width) is still
the right instrument and is still the outstanding answer to Q1a/Q1b.

---

## 4. P03 / P05 — the control experiment, and the two real findings

### 4.1 Why a control at all

P02 and P03 both returned negatives. A negative from an instrument that has never reproduced a
positive is uninterpretable. So: take the clutter away and grasp the target alone — something
`grasp_geometry.py` has already shown this gripper does. A failure there is a bug in my code;
a success licenses every later negative.

**The control failed three times.** Each failure was mine, and each was worth the run.

### 4.2 Finding — the standoff was in C9's unusable region

`p05_pinch_trace.py` prints commanded vs **achieved** state at every phase. The `axis`
approach (`grasp_geometry.py`'s recipe: stand off 70 mm along the z-flattened jaw axis) traced
like this:

```
after teleport    TCP (311.3, -33.8,  55.3) | blk (250.0,  0.0, 35.0) | qerr   0.00 mrad
settled at start  TCP (325.7, -34.0,  79.8) | blk (252.0, -0.6, 33.0) | qerr  97.13 mrad
waypoint 1        TCP (311.6, -19.7,  86.9) | blk (243.9,  4.3, 36.4) | qerr 148.10 mrad
waypoint 2        TCP (297.4,  -4.1,  95.9) | blk (237.0, 11.1, 38.8) | qerr 210.79 mrad
waypoint 3        TCP (255.5,   0.1,  59.5) | blk (174.2,  6.8, 15.0) | qerr  35.89 mrad
```

The standoff lands at **x ≈ 311 mm, z ≈ 55 mm**. `CHALLENGE_SUITE` C9 states plainly:
*"x = 0.30 — unusable below z ≈ 0.10 m (14–48 mm tracking error)"*. The arm cannot hold that
pose: it sags 24 mm on arrival, joint tracking error grows to **211 mrad (12°)**, and the
flailing jaw ploughs the block from `y = 0` to `x = 174 mm` — knocked over and 76 mm out of
place — before the fingers ever close.

This is not a subtle effect and it is fully predicted by a constraint already written down.
**Any waypoint chain for this task must be checked against the reliable band
(x ≈ 0.22–0.26, z ≈ 0.045–0.10) at planning time**, not discovered in execution.

**Fix: descend vertically.** Hold the wrist orientation fixed and come straight down from
`grip + (0,0,0.07)`. Because the jaw axis runs up-and-back, the wrist tracks down *behind* the
block while the two fingers come down either side of it. The traced result:

```
after teleport    TCP (250.2, -0.2, 125.6) | blk (250.0, 0.0, 35.0) | qerr  0.00 mrad
waypoint 1        TCP (250.0, -0.2, 100.9) | blk (250.0, 0.0, 32.0) | qerr 13.92 mrad
waypoint 2        TCP (249.3, -0.8,  78.5) | blk (250.0, 0.0, 32.0) | qerr 10.61 mrad
waypoint 3        TCP (249.2, -1.2,  54.6) | blk (250.0, 0.0, 32.0) | qerr 11.70 mrad
AT GRASP, open    TCP (250.1, -0.8,  54.5) | blk (250.0, 0.0, 32.0) | dL 42.8  dR 44.3
```

The block does not move at all, and both `|finger.y − block.y|` land at ~43 mm — the jaw is
around it. Tracking error stays under 14 mrad throughout.

### 4.3 Finding — I reproduced eva_rl's most expensive mistake verbatim

With the vertical approach the fingers closed to **`gap = 29.99 mm` on a 30 mm block** — a
textbook grasp. Then the lift dropped it and `gap → −1.23 mm`.

Cause, quoted from my own notes before the run (`06_EXPERT_DESIGN.md` rule 3, `02_PLAN.md`
rule 3):

> *"Finish every CEM search before closing the gripper. CEM uses `write_joint_state_to_sim`,
> which teleports the arm and re-opens the fingers hundreds of times; searching for a lift
> path after closing silently drops the object. eva_rl's handoff says this one cost the most
> time of anything in that effort."*

I planned the lift after the close. `cem() → fk() → write_joint_state_to_sim(q_fing = Q_OPEN)`,
several hundred times, with the block nominally in the jaw. Moving the lift solve above the
close phase fixed it in one edit.

The signature is worth recording because it is diagnostic: **a clean stall at the object width
followed by `gap → −1.2 mm` on the next phase is a re-opened-during-planning drop, not a grip
force problem.**

### 4.4 The control now PASSES

```
AT GRASP, open   gap 89.06  dL 37.6  dR 39.2   blk (250.0,  0.0,  32.0)
after close      gap 41.63                     blk (248.2, -1.8,  34.6)
after lift       gap 41.58                     blk (254.2, -2.2, 125.5)
rose 100 %   enclosed 98 %   median gap 41.65 mm
```

Block lifted from z = 32 mm to z = 125.5 mm and still enclosed. The convention chain — TCP
offset, approach axis, opening axis, action encoding, waypoint execution — is validated
end-to-end. **Negatives from here on are believable.**

One caveat carried forward: this pose had `o_align = 0.851`, i.e. the opening axis 32° off ŷ,
so the fingers grip on a diagonal and stall at **41.6 mm** rather than 30 mm. That only just
satisfies `|gap − 30| < 12`. A better-aligned solve (`o_align = 0.959`) stalled at exactly
**30.0 mm**. `w_o` has been raised to 0.60 so the search prefers alignment; grip quality should
be reported as `o_align` *and* stall gap, never as a binary.

### 4.5 An incidental measurement worth checking later

Blocks are written to `z = 0.035` (half height) and settle to **`z = 0.032`** every time —
a consistent 3 mm sink. Either the effective table top is at z ≈ −0.003 in the env frame, or
the blocks penetrate by 3 mm at rest (`contact_offset = 0.002`, `rest_offset = 0.0`). It is
systematic, not noise. **Every height threshold in the task is specified against z = 0**, so
this shifts `EXTRACT_Z`, the `z < 0.055` goal test and every `h_crit` by 3 mm. Worth pinning
down before the expert's height gates are tuned.

---

## 5. P03 full run — what the clutter actually costs

With the corrected primitive, all three conditions, 12 (grip z × approach tilt) cells each,
128 envs per cell:

| condition | best HELD | stall gap | topple |
|---|---|---|---|
| **solo** (distractors parked) | **100 %** in 11 of 12 cells | **30.0 mm** — exactly the block width | 0 % |
| **clutter** (nominal row) | **0 %** in 12 of 12 | −1.2 mm (shut on air) | **0 %** |
| **gap** (d1/d2 pushed 25 mm out) | **0 %** in 12 of 12 | −1.2 mm | **0 %** |

Two things in that table matter more than the headline.

**The topple rate is zero.** The gripper is not knocking the row over — it is being *stopped*
by it. That reframes the task: the binding constraint is clearance, not fragility, and the
`distractor_toppled` termination never even gets a chance to fire.

**25 mm of singulation is not enough.** That kills the comfortable reading that the fingers
merely clip the neighbours. It sent me back to P01's discarded AABB with a sharper question.

### 5.1 Partially un-retracting the P01 bound

§1.3 rejected the AABB outright. That was too broad. An AABB of a concave mesh overestimates
**inward** extent — it fills the jaw's mouth with air, which is exactly the 51.7-vs-89.07 mm
contradiction — but its **outer** extreme is attained by a real mesh vertex. Read that way it
predicts an open-gripper outer width of about **128.6 mm**, and the `gap` condition's 104 mm of
clearance is simply not enough room. P06 was written to test that prediction by contact.

## 6. P06 — the singulation threshold, measured

One env per push distance (0–70 mm in 0.55 mm steps), one shared grasp trajectory, full row.

```
push   free | TCP z @grasp | blocked | stall gap | HELD | topple
 0.0mm    54 |       87.8  |   YES   |   -1.25   |   .  |   -
11.0mm    76 |       88.0  |   YES   |   -1.25   |   .  |   -
22.0mm    98 |       87.8  |   YES   |    4.90   |   .  |   -
30.9mm   116 |       76.4  |   YES   |   30.07   | HELD |   -
39.7mm   133 |       61.2  |   YES   |   30.01   | HELD |   -
48.5mm   151 |       53.6  |    -    |   30.04   | HELD |   -
66.1mm   186 |       53.6  |    -    |   30.03   | HELD |   -
```

| quantity | value |
|---|---|
| first success | **push ≈ 31 mm** per side (115.7 mm clear between d1/d2 inner faces) |
| reliably clear descent | **push ≈ 48 mm** per side (151 mm clear) |
| success rate above threshold | 78 % (100 % above 48 mm) |
| **topple rate across the whole sweep** | **0 %** |

The contact diagnostic (achieved body positions at the grasp phase) shows the mechanism
plainly: at push = 0 the finger origins sit at y = −44.7 / +47.2 and **z = 88 mm**, resting on
the 70 mm-tall neighbours' tops, while the commanded height was 55 mm. As `push` grows the
stall height falls monotonically — 87.8 → 76.4 → 61.2 → 53.6 mm — until the gripper reaches
the commanded pose. The fingers are landing on the neighbours, exactly as the env intends.

**Note on pose sensitivity.** A first P06 run with a poorer grasp pose (`o_align = 0.902`)
reported the threshold at 67.8 mm; the run with `o_align = 0.998` puts it at 31 mm. The
opening-axis alignment is worth ~37 mm of required clearance. Any expert must gate on
`o_align`, not merely on TCP error — the two are not interchangeable and only the first
predicts whether the jaw fits.

## 7. The geometric obstruction that decides Stage 0

P06 says the grasp needs ~31–48 mm of lateral room per side. So: how is that room created?

**By a lateral push — it cannot be.** To push a distractor *outward* you must contact its
**inner** face. d1's inner face lives in the 12 mm gap between d1 and the target; d0's inner
face lives in the 12 mm gap between d0 and d1. The closed gripper is ~38.6 mm across. **No
inner face in the row is reachable.** Every outward-facing surface — d0's at y = −99, d3's at
y = +99 — can only be pushed *inward*.

That is a general statement about the row, not about one strategy:

> **A lateral push can compress this row. It can never spread it.**

Consequences, recorded against the pre-registered strategy list in `02_PLAN.md`:

- **B (singulate then grasp) — no contact surface.** Dead as originally conceived, not because
  pushing topples but because the push cannot be applied at all. Note this is a *different*
  reason than belief 1 predicted.
- **C (plow-back) — creates no lateral room.** Pushing on the exposed −x faces moves blocks
  along x. The row's pitch is unchanged, so the grasp is still blocked.
- **A (pair-capture)** — needs the finger to enter a gap, which P06 shows it cannot.
- **D (topple the target)** — the closed gripper is ~38.6 mm wide against a 30 mm target, so a
  frontal push contacts d1 and d2's front faces as well. Not obviously separable; unmeasured.

What remains is the mechanism the env's own docstring names — the fingers coming **down** the
gap as **wedges**, forcing the blocks apart as they descend. P06 only ever held the grasp pose
for 25 steps against a drive that had already conceded 33 mm of tracking error. `p07_wedge_press.py`
presses properly: command the TCP below the grip height, dwell, and measure the spread and the
topples. **That probe is the Gate 0 decision.**

## 8. P07 — wedging does not open the row

Press the open gripper down onto the nominal row, commanded TCP swept from 55 mm to −20 mm
(i.e. hard into the table), dwell 300 steps.

| quantity | value |
|---|---|
| deepest achieved TCP z | **69.8 mm** (commanded to −20 mm) |
| best single-side spread | d1 4.2 mm, d2 6.9 mm |
| **best symmetric spread** | **4.2 mm** — against the ~31 mm needed |
| topple rate | 0 % |
| min `up_z` at hardest press | **0.765** (threshold 0.75) |

The mechanism is visible in that last row. Pressing harder does not slide the neighbours
apart — it **tilts** them, `up_z` falling monotonically 0.934 → 0.765 as the press deepens,
arriving at the topple threshold without ever delivering useful lateral travel. That is
tip-vs-slide, observed directly: contact is at the blocks' top edges (z ≈ 70 mm) against
`h_crit = 15.8 mm`, so the blocks rotate rather than translate.

*Caveat recorded:* rows commanded below z ≈ 8 mm show erratic achieved heights (103–165 mm).
Those targets are unreachable and `refine`'s damped-least-squares step wanders. Those rows are
not evidence of anything and are excluded from the reading above.

## 9. P08 — strategy C's precondition HOLDS

§7 dismissed strategy C ("plow the neighbours back along x") for creating no lateral room.
**That was wrong**, and P04's own measurement says why: this arm's fingers point *back toward
the robot*, so the volume the gripper needs is not a y-slab through the row but a wedge leaning
toward the robot. Neighbours displaced in +x can fall outside it with the y-pitch untouched.

Teleporting all four distractors +x by a swept offset and running the validated grasp:

| offset | TCP z @grasp | blocked | stall gap | HELD |
|---|---|---|---|---|
| 0 – 62.4 mm | 84 → 93 mm | YES | −1.25 mm | . |
| **65.2 mm** | **54.4 mm** | – | **30.00 mm** | **HELD** |
| 65.2 – 87.9 mm | 54.4 mm | – | 30.0 mm | HELD (100 %) |

**Threshold: 65 mm of +x displacement**, then 100 % held, 0 % topple, with a sharp transition
(62.4 → 65.2 mm) that marks a real clearance boundary rather than a soft trend.

So the precondition is measured and satisfied. Everything then turns on whether the neighbours
can be *pushed* 65 mm — and unlike a lateral push, this one has a contact surface, because
every block's −x face is exposed.

## 10. P09 — the push topples. Belief 1 confirmed.

### 10.1 A measurement flaw that erased the thing being measured

P09's first run reported ±10 mm displacements with **random signs**, including d1 moving
*toward* the robot under a push directed away from it, and 0 % topple. That is not noise, it is
a reset: `env.step` runs the full MDP, so a block tipping past 41.4° fires `distractor_toppled`,
the scene resets, the block **re-spawns upright with fresh spawn jitter**, and the probe reads
`up_z = 1.0` plus a random offset drawn from the reset event's ±10 mm range.

**A contact probe must not step the MDP.** `_kin.ArmKin.hold_phys` / `run_phys` set the drive
target and step the simulator directly. Any future measurement of contact, disturbance or
toppling has to use them; `hold`/`run` are only for when the MDP itself is under study.

### 10.2 The result

Closed gripper driven +x through d1, 105 mm stroke at ~131 mm/s, contact height swept 44–90 mm:

| contact z | d1 Δx | d1 `up_z` | reading |
|---|---|---|---|
| 44.0 – 56.7 mm | **0.00 mm** | 1.000 | **no contact at all** |
| 58.5 – 80.2 mm | 13 – 166 mm | **0.000** | contact, and **always flat on the table** |
| 82.0 – 89.3 mm | ~0.2 mm | 1.000 | passes above the 67 mm block top |

- **The reachable contact window is z ≈ 58–80 mm.** Below 58 mm the gripper simply cannot
  reach the block: its fingers point up-and-back, so the fingertips sit *above* the TCP and
  nothing extends forward at low height. C1's no-top-down constraint is what forces this.
- **Every contact in that window topples the block.** `up_z → 0.000` is flat on the table, not
  a marginal tilt. Topple rate 54 % of all envs — the complement is the envs that never touched.
- Displacement does reach 54–166 mm, i.e. the push is easily strong enough. It is the *mode*
  that is wrong, not the magnitude.

This is the quasi-static prediction landing exactly: the only available contact heights are
**3–4× `h_crit(x) = 19.0 mm`**, and raising the push speed 5× (`--push_steps 8`) changed
nothing. Dynamics does not rescue it.

> **Belief 1, pre-registered in `02_PLAN.md`: "Q2 will come back unfavourable — a push at any
> reachable height topples the distractor more often than it slides it." Confidence
> moderate-high. → CONFIRMED**, and for a sharper reason than stated: not only does the push
> topple, but the sub-`h_crit` contact heights that would slide are *kinematically
> unreachable*.

## 11. GATE 0 — the verdict, and what it rests on

`02_PLAN.md` Gate 0: *"PASS if at least one of strategies A–D completes the task end-to-end in
≥50 % of a fixed 64-episode suite under privileged scripting."*

| strategy | status | evidence |
|---|---|---|
| **A** pair-capture | **fails** | the fingers cannot enter a gap; the descent stalls 33 mm high on the neighbours' tops (P06) |
| **B** singulate laterally | **fails — no contact surface** | outward pushing requires an *inner* face; every inner face is inside a 12 mm gap and the closed gripper is ~38.6 mm across (§7). A lateral push can only compress the row |
| — wedging (the env's own suggested mechanism) | **fails** | 4.2 mm symmetric spread against ~31 mm needed; pressing tilts toward topple instead of translating (P07) |
| **C** plow back along +x | **precondition holds, mechanism fails** | 65 mm of offset gives a clean 100 % grasp (P08), but every reachable push topples the block (P09) |
| **D** topple the target | **untested, and now the only candidate left** | the pusher spans ±19.3 mm about the target's centreline while d1/d2's inner faces sit at ±27 mm — so a frontal push on the target contacts **neither neighbour**, with 7.7 mm clearance each side |

**As measured, Gate 0 does not pass.** Under `DR1` that means: *stop and report to Big Will
with the evidence; do not weaken the env or proceed on hope.* This document is that report.

### 11.1 What would change the verdict

Three specific things, in descending order of promise. None is a reason to proceed to Stage 1
yet; all are cheap relative to a demo-generation chain.

1. **Strategy D, properly tested.** The clearance arithmetic above is favourable and the
   pusher geometry is already characterised. A toppled target satisfies `target_at_goal`'s
   `z < 0.055` and nothing penalises the *target* falling. The open question is not whether it
   can be toppled but whether it can then be moved to `(0.185, −0.185)`: after a backward
   topple it lies at x ≈ 0.285, *behind* the row, where the approach for a subsequent push is
   itself obstructed.
2. **Combined x/y clearance.** P06 (y) and P08 (x) each swept one axis with the other at
   nominal. The thresholds are 31–48 mm and 65 mm respectively; a *joint* offset may need far
   less of each, and the spawn jitter already supplies up to 22 mm of relative x offset for
   free. This is a 2-D version of a sweep already written.
3. **Pose sensitivity.** P06 measured the y-threshold at 68 mm with `o_align = 0.902` and
   31 mm with `o_align = 0.998` — the opening-axis alignment is worth ~37 mm of required
   clearance. No search over grasp poses has been run with `o_align` as the primary objective,
   and the P08 threshold rests on a single pose.

### 11.2 What is NOT in doubt

- The grasp primitive works: **100 % held on an isolated target, stall gap 30.0 mm**, across 11
  of 12 (grip z × tilt) cells.
- Given clearance, the clutter grasp works too: **100 % at ≥65 mm x-offset, 0 % topple.**
- The blocking is **clearance, not fragility** — the gripper is stopped by the row rather than
  knocking it over, and in every non-pushing experiment the topple rate was exactly 0 %.

That combination is worth stating plainly: this environment is not marginally hard, it is
**geometrically obstructed** for the gripper it ships with, and the obstruction is the same one
`CHALLENGE_SUITE` C1 identifies as the suite's defining constraint. eva_rl's own note that a
scripted extraction *"is still not verified"* now has a measured explanation behind it rather
than an absence of evidence.

## 11.3 P10 — the native rate on random spawns is 0 %

The one favourable-looking gap in the argument was that everything above was measured at the
**nominal** row, while the reset events supply up to 22 mm of relative x offset and a free gap
anywhere in the 7–19 mm range for free. So: reset normally, adapt the grasp to where the target
actually is (`refine`, achieved TCP re-read, median error 0.0–2.7 mm), execute the full
descend → close → lift → carry → place chain, score against the env's own predicates.

**256 episodes over 2 independent reset batches:**

| | |
|---|---|
| enclosed at close | 52.3 % |
| **target at goal** | **0.0 %** |
| **topple rate** | **100.0 %** |

Stratified by per-episode minimum free gap (0–8 / 8–10 / 10–12 mm): enclosure 52.6 / 51.1 /
53.1 %, success 0.0 % in every band. Stratified by the target's forward offset relative to its
neighbours (−100..−10 / −10..0 / 0..10 / 10..100 mm): enclosure 30.0 / 66.3 / 55.3 / 14.3 %,
success 0.0 % in every band. **No sub-population of spawns is solvable.**

Two readings, and the second matters more than the first:

- The 52 % "enclosed at close" is **not** 52 % of grasps. It is the `|gap − 30 mm| < 12 mm`
  criterion firing on a jam: this run's nominal pose came back with `o_align = 0.820` (the jaw
  35° off the block's faces), and a diagonal wedge against the row stalls the fingers in the
  same band as a real grasp. The enclosure check is necessary but not sufficient — it must be
  paired with `rose` before it means anything, and here `rose` never fires.
- The measured minimum free gap over 256 settled spawns is **2.4–12.3 mm, median ~8 mm** —
  meaningfully tighter than the 7.0–18.8 mm the write-up quotes. Whatever the source of that
  difference, the row this task actually presents is at the hard end.

### 11.4 A methodological correction that touches earlier probes

P10 reports **100 % topple** where P06/P08 reported 0 %. The difference is the instrument, not
the physics: P06/P08 stepped the MDP (`hold`/`run`), so any topple fired `distractor_toppled`,
reset the scene, and re-spawned the blocks upright — the same masking that made P09's first run
unreadable. P10 steps physics only, presses for ~2.2 s, and sees what really happens.

**Therefore: the 0 % topple rates quoted for P06, P07 and P08 are not trustworthy and are
withdrawn as topple measurements.** What survives is exactly what those probes were built to
measure and what does not depend on resets:

- P06's **clearance thresholds** (31 mm first success, 48 mm reliable) — the successful envs
  reached the commanded height with clean 30.0 mm stalls, which no reset would produce.
- P08's **65 mm x-threshold**, same reasoning, with a sharp 62.4 → 65.2 mm transition.
- P07's spread of 4.2 mm, whose minimum `up_z` was 0.765 — *above* the 0.75 threshold, so no
  reset could have fired there.

The lesson generalises and is now encoded in `_kin`: **any probe that measures contact,
disturbance or toppling must use `hold_phys`/`run_phys`.** This is the fifth instrument defect
in Stage 0 that produced a confident, plausible, wrong number.

## 12. Belief scorecard so far

| # | prediction | outcome |
|---|---|---|
| 1 | push topples rather than slides at all reachable heights | **CONFIRMED** (P09) — and sub-`h_crit` heights are unreachable |
| 2 | pair-capture 40–70 % nominal, ~0 % Tight | **wrong in kind** — 0 % nominal; the fingers never enter a gap at all |
| 4 | FK-CEM achieves ≤2 mm TCP error in the reliable band | **CONFIRMED** — 0.2–1.0 mm typical once the cost was constrained properly |
| 5 | 512–1024 envs is the working range on 10 GiB | not yet measured (Q7 outstanding) |
| 3, 6–12 | — | not yet reached |

## 13. Recommendation

Gate 0 does not pass, and under `DR1` the pre-registered response is to report rather than
proceed. Concretely, for Big Will:

**1. The task as shipped is geometrically obstructed for the gripper it ships with.** That is a
finding about the benchmark, and it is arguably the most valuable output of Stage 0 — it is the
answer to the open V4 rung that `docs/envs/clutter-extract.md:119` flags as *"still not
verified"*. Recommend reporting it upstream to eva_rl regardless of what we do next.

**2. The cheapest env change that would make the task solvable as designed is `ROW_PITCH`.**
The grasp is validated and reliable given clearance; it needs ~48 mm per side laterally, or
65 mm of depth offset. `ROW_PITCH = 0.042` gives 12 mm. A pitch that leaves ~50 mm of free gap
(`ROW_PITCH ≈ 0.080`) would make the nominal task a normal, solvable clutter problem while
keeping the topple constraint meaningful. **This is a recommendation to Big Will, not a change
to make unilaterally** — `HANDOFF.md` §1 is explicit that eva_rl's `challenge/` package is the
benchmark under test and that env changes are findings to report, not fixes to apply.

**3. Strategy D remains untested and is the only in-spec candidate left.** The pusher clears
both neighbours by 7.7 mm, so the target can be struck without touching them. What is unknown
is whether a toppled target can then be moved to `(0.185, −0.185)`; after a backward topple it
lies at x ≈ 0.30, behind the row and inside C9's unusable band. My honest estimate is that this
does not close, but it is a few hours to find out and it is the difference between "no path
found" and "no path exists".

**4. Do not start Stage 1.** An expert cannot be built on a manoeuvre that has no measured
success, and a demo-generation chain against a 0 %-success primitive is exactly the unbounded
risk `02_PLAN.md` put Gate 0 in front of.

## 14. Where this leaves Stage 0

**Answered:**
- Q5 (friction): μ_block 0.9, μ_table 1.0, effective ≈ 0.95. `h_crit` = 15.8 mm (y) / 19.0 mm (x).
- The arm's grasp orientation family at the row: fingers back-and-up, `a_hat·x̂ ≈ −0.65…−0.78`.
- The approach must be a **vertical descent**; jaw-axis standoffs leave the reliable band.
- A working, validated grasp primitive on an isolated target.

**Not answered, and blocking Gate 0:**
- Q1a/Q1b — the open gripper's true outer width, by contact. Needs P02 §2 re-run.
- Q2 — push/topple sweep. Not yet written.
- Q3/Q4 — pair-capture and target-topple feasibility.
- Q7 — throughput/VRAM at N envs.
- **The `clutter` and `gap` conditions of P03**, which are the actual Stage-0 measurement.

**Method note for the rest of this effort.** Four of the five diagnoses above were defects in
the measuring instrument, and every one of them produced a *plausible, confidently-worded
negative result* rather than an error. The control experiment is what separated them from
facts about the task. Nothing in Stages 1–4 should be trusted without an equivalent positive
control, and eva_bc's "gate every wrapper on bit-exact reproduction" rule is the same
principle — I now have my own reason to believe it rather than an inherited one.

---

# PART II — THE ORTHOGONAL GRASP (added 2026-08-02, after Big Will's question)

## 15. P11 — rotating the jaw 90° solves the approach

### 15.1 The gap in the record

Big Will asked whether the gripper had been tried **orthogonal to the block**, approaching from
above, clear of the neighbours, then descending. Checking honestly:

* **Top-down translation: tested.** The validated recipe *is* a vertical descent from
  z = 125 mm, above the 67 mm block tops.
* **Orthogonal orientation: never tested with a working instrument.** `p02` swept exactly this
  family (`o_des = x̂`, "G2") and returned 0 of 48 cells attainable — but that probe ran on the
  CEM *before* the three fixes in §3, and is marked **void** in this document. `p04` reported
  "G2: 0 samples", but only 3 of 819 200 draws landed near the grasp point at all, which I
  flagged at the time as a density artifact. Every probe that produced a number — P03, P05,
  P06, P08, P10 — used `o_des = ŷ`, i.e. fingers at ±44.5 mm **across** the row: precisely the
  configuration that lands them on d1/d2's tops.

The reason I set the family aside was a reasoning error, and the counter-example was already in
my own data. See §7.2 of `HANDOFF.md`: C1 constrains the approach **axis**, not the direction of
**travel**, and P05's working grasp descends vertically with `a_hat` far from vertical.

### 15.2 Attainability, with the corrected CEM

`o_des` swept from ŷ (φ=0) to x̂ (φ=90) at three grip heights, hinge cost, gripper-only floor,
12 restarts. **14 of 15 cells attainable, and every cell with φ ≥ 45° puts both fingers clear
of the row** — where φ=0 never does:

```
 z[mm]  phi | pos err | o_align |  low_z |        fL x,y |        fR x,y | clear row?
    55    0 |  0.85mm |   0.999 |  14.8mm | 251.4,  45.7 | 248.2, -44.4 |        no
    55   45 |  0.71mm |   0.997 |  41.0mm | 279.2,  34.0 | 220.2, -34.2 |       YES
    55   70 |  0.83mm |   0.987 |  15.6mm | 292.8,  13.5 | 208.3, -14.6 |       YES
    55   90 |  0.69mm |   0.998 |  29.0mm | 295.6,  -1.8 | 205.6,   2.4 |       YES
    65   90 |  0.91mm |   0.997 |  38.3mm | 204.8,  -2.2 | 294.7,   4.0 |       YES
```

The mechanism is legible in the finger coordinates. At φ=90 the two fingers sit at
**x ≈ 205 mm and 295 mm, both at |y| < 5 mm** — in front of and behind the row, on the target's
own centreline. They never enter a 12 mm gap. At φ=0 they sit at y = ±45 mm, inside d1 and d2.

### 15.3 Execution on the nominal row

| cell | TCP z at grasp | stall gap | encl | rose | HELD | topple |
|---|---|---|---|---|---|---|
| **z=65, φ=90** | **68.1 mm (reached)** | **36.40 mm** | 100 % | 100 % | **100 %** | **0 %** |
| z=55, φ=90 | 59.2 mm (blocked) | 36.49 mm | 100 % | 61 % | 61 % | 0 % |
| z=45, φ=90 | 84.9 mm (blocked) | −1.12 mm | 0 % | 0 % | 0 % | 0 % |
| z=55, φ=80 | 57.0 mm (reached) | 44.60 mm | 100 % | 100 % | 100 % | **100 %** |
| z=45, φ=70 | 71.3 mm (blocked) | 44.88 mm | 100 % | 100 % | 100 % | 75 % |

The stall gap of **36.40 mm** against the block's 36 mm x-faces is the ground truth that this is
a real grasp and not a jam. Note the scoring subtlety: an orthogonal jaw closes on the **36 mm**
x-faces, not the 30 mm y-faces. Using the wrong width would have marked every one of these as a
failure — the same class of error as the `o_align` trap.

`z = 0.065, φ = 90` is the operating point. Lower grip heights are blocked; φ=80 grasps but
topples.

## 16. P10 re-run — 25 % end-to-end on random spawns

### 16.1 Alignment is a gate, demonstrated twice

Two runs of the *same call*, differing only in the CEM's stochastic draw:

| `o_align` | enclosed | at goal | topple | success |
|---|---|---|---|---|
| 0.848 | 22.7 % | 0.0 % | 88.7 % | **0.0 %** |
| **0.991** | **100.0 %** | **99 %** | 75.0 % | **25.0 %** |

A jaw 32° off axis swings the fingers from |y| < 5 mm out to ±24 mm — straight back into the
gaps. This is the same effect P06 measured on the cross-row grasp, where alignment was worth
~37 mm of required clearance. **`o_align` is a selection criterion, not a soft cost term**, and
`best_pose()` in `p10` now maximises it subject to `pos_err < 1.5 mm`. Any expert must do the
same.

### 16.2 The result, 256 spawns over two independent reset batches

| metric | value |
|---|---|
| enclosed at close | **100.0 %** |
| target at goal | **99.2 % / 98.4 %** |
| topple rate | **75.0 %** |
| **SUCCESS** | **25.0 % / 25.0 %** |

**The manipulation is solved; the approach is not.** At-goal is 99 %, so every point of topple
reduction converts almost 1:1 into success. That is the whole of Stage 1's job.

Success rises monotonically with clearance — 21.4 / 25.5 / 28.6 / **55.6 %** across min-free-gap
bands 0–8 / 8–10 / 10–12 / 12–14 mm — which is the expected shape and a good sign.

**An anomaly worth understanding before tuning anything:** success *falls* as the target moves
forward of its neighbours — 29.4 / 28.0 / 23.9 / **13.3 %** across rel-dx bands −100..−10 /
−10..0 / 0..10 / 10..100 mm. That is backwards from the P08 clearance story, which said a
forward target should be *easier*. Something in the model is wrong, and it is cheap to find out.

## 17. Revised belief scorecard

| # | prediction | outcome |
|---|---|---|
| 1 | push topples rather than slides at all reachable heights | **CONFIRMED** — and sub-`h_crit` heights are kinematically unreachable |
| 2 | pair-capture 40–70 % nominal, ~0 % Tight | **wrong** — the cross-row grasp is 0 %, but the *orthogonal* grasp reaches 100 % nominal |
| 4 | FK-CEM ≤2 mm TCP error in the reliable band | **CONFIRMED** — 0.2–1.0 mm once the cost was constrained properly |
| 7 | Tight near-unsolvable | **open, and now doubtful** — the fingers no longer use the gaps. New prediction: Tight within 10 pts of nominal |
| 5, 3, 6, 8–12 | — | not yet measured |

## 18. The methodological lesson, restated

Six confidently-worded wrong answers in one stage: two USD traversal traps, three CEM/trajectory
defects, one measurement defect that erased topples, and one reasoning error that cost more than
all the others combined. **None of them crashed. Every one produced a plausible number.**

The five instrument defects were caught by a positive control — grasping an isolated block that
eva_rl had already shown to be graspable. The reasoning error was not caught by anything I did;
it was caught by Big Will asking whether I had tried the obvious alternative. Worth recording as
its own rule: **a control proves the instrument works, but it cannot tell you that you never
pointed it at the right hypothesis.** When a negative result rests on a family that was ruled
out by argument rather than measurement, that argument is the thing to re-examine first.
