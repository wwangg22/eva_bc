# ANGLED SLOT — implementation record

*Working record for the change planned in [`ANGLED_SLOT_PLAN.md`](ANGLED_SLOT_PLAN.md). That
file is the plan, written before any code; this one is what actually happened, including the
places the plan was wrong.*

**Status: gates A and C pass. Gate B (θmax) running.** Nothing deleted yet.

> **Every number in every other document in `docs/slot/` was measured on the AXIS-ALIGNED
> slot.** The champion's 0.979, all of `EXP_ROBUSTNESS`, all of `EXP_STEER`, the vision 0.804
> and the DAgger −31.3 describe a task that no longer exists. They are kept because they are
> true statements about that task and several of their *lessons* carry over, but none of them
> is a claim about the angled slot.

---

## 1. What changed, and the scoping rule it had to respect

Big Will lifted the "never edit eva_rl" rule for this change, with a constraint:

> *"Yes you can edit eva_rl, just make sure you only touch your assigned task (precision slot),
> and not any other task"*

`challenge/mdp/` is shared by **four** tasks. The scoping analysis in the plan (§0b) said the
five slot predicates have zero sibling users and `BLOCK_HALF` / `block_lifted` /
`block_dropped` / `block_pose_in_root` have two or three. That analysis was **checked, not
trusted** — `scripts/sibling_smoke.py` builds, resets and steps each sibling:

| task | obs | act | 20-step mean return | verdict |
|---|---|---|---|---|
| `Rebot-PreGrasp-v0` | (4, 32) | (4, 7) | +0.0489 | builds and steps |
| `Rebot-ClutterExtract-v0` | (4, 42) | (4, 7) | +0.0359 | builds and steps |
| `Rebot-DrawerOrder-v0` | (4, 33) | (4, 7) | +0.0457 | builds and steps |

The per-env slot yaw defaults to **zeros**, so a task that never installs the reset event sees
byte-identical geometry. That is what makes the sibling result a consequence of the design
rather than luck.

### Files touched in `eva_rl`

| file | change |
|---|---|
| `challenge/mdp/common.py` | `slot_yaw` / `slot_axis` / `slot_local_xy` / `slot_yaw_delta` added; `insertion_depth`, `lateral_error`, `yaw_error` rewritten in the slot frame |
| `challenge/mdp/slot.py` | **new** — `set_slot_yaw` + the `reset_slot_yaw` event term |
| `challenge/mdp/observations.py` | `slot_frame` 4-D → 6-D, signed, carrying the slot's orientation |
| `challenge/mdp/rewards.py` | `_keypoint_dist` goal position **and orientation** rotate with the slot |
| `challenge/mdp/__init__.py` | one import line |
| `precision_slot_env_cfg.py` | fixture parts static → kinematic; `slot_yaw_range` cfg; the reset event |

`BLOCK_HALF`, `block_lifted`, `block_dropped`, `block_pose_in_root`, `reach_block` — the shared
symbols — were not modified.

---

## 2. The three corrections to the plan

**2a. The plan's depth formula had the wrong sign.** `ANGLED_SLOT_PLAN.md` §2a says

```
insertion_depth = along - SLOT_DEPTH/2
```

but depth is measured **from the mouth**, and the mouth sits at `along = −SLOT_DEPTH/2`, so it
is `along + SLOT_DEPTH/2`. Taken literally the plan's version would have put the success
threshold 70 mm too deep — past the back stop, i.e. unreachable — and every episode would have
scored zero with no obvious cause. Caught by the θ = 0 regression, which is exactly what that
gate is for.

**2b. "Bit-for-bit at θ = 0" is not achievable and the gate was rewritten.** `(x − cx) + D/2`
and `x − (cx − D/2)` are the same real number and different floats. Measured deviation over 256
random poses: **1.49 × 10⁻⁸ m** — one float32 ULP at 0.2, and five orders below the 1.5 mm
clearance. The other two predicates *are* bit-exact, because the operations that could
re-associate were written not to:

* `lateral_error`: `−dx·0.0 + dy·1.0` collapses to `dy` exactly. **0.0 deviation.**
* `yaw_error`: the wrap is two `torch.where` conditionals, not `(d + π) mod 2π − π`. A modulo
  wrap would round every angle even when no wrapping is needed; the conditional form leaves the
  θ = 0 case untouched. **0.0 deviation.**

The predicate itself — the thing that decides success — agrees **exactly**: `is_inserted`
identical on 256/256 samples.

**2c. The observation had to grow, which the plan did not anticipate.** With a random slot
angle, `(depth, |lateral|, |yaw|)` no longer determines what the arm should do: the angle is
recoverable from them only up to a discrete ambiguity. A privileged teacher that cannot see
where the slot points cannot solve the task either, so the task would have become *impossible*
rather than *hard*. See §4.

---

## 3. Gate A — geometry and predicate agree

`slot/scripts/gate_a_slot_yaw.py`, n = 256. **14/14 checks pass.**

```
A1  theta = 0 reproduces the axis-aligned arithmetic
  [PASS] slot_yaw is exactly zero            max |theta| = 0.000e+00
  [PASS] insertion_depth matches             max dev 1.490e-08 m  (tolerance 1.5e-3 m)
  [PASS] lateral_error is bit-exact          max dev 0.000e+00 m
  [PASS] yaw_error is bit-exact              max dev 0.000e+00 rad
  [PASS] is_inserted identical               3/256 old, 3/256 new

A2  analytic placement at theta ~ U(-0.5, 0.5)
  [PASS] theta spans the range               [-0.499, +0.494] rad
  [PASS] seated          -> True             256/256
  [PASS] slot-frame coordinates read back    depth 1.01e-06 m, lateral 9.67e-07 m, yaw 5.96e-08 rad
  [PASS] too shallow     -> False            0/256
  [PASS] off centreline  -> False            0/256
  [PASS] off square      -> False            0/256

A3  the fixture bodies actually rotated
  [PASS] positions follow the rotation       max dev 9.54e-07 m
  [PASS] orientations follow theta           max dev 5.96e-08 rad
  [PASS] the fixture moved at all            max part displacement 19.3 mm

A4  the COLLIDER moved, not just the pose
  [PASS] block ejected by the rotated wall   median displacement 86.3 mm
```

Three of these deserve comment because they are the checks that could have been fake:

* **A1 is non-vacuous but thin** — only 3/256 random poses land inserted, so the boolean
  agreement is carried by 3 positives. The substantive comparison is the three continuous
  deviations, which are computed on all 256.
* **A3's third check exists because the first two would pass a rotation that does nothing.**
  Comparing "where the part is" against "where the closed-form says it should be" is satisfied
  by θ = 0 and no motion at all. The 19.3 mm displacement is what rules that out.
* **A4 is the one that tests physics rather than bookkeeping.** A3 reads a pose; A4 reads a
  consequence. The block is parked where the *rotated* +y wall now stands — a spot that is bare
  floor in the unrotated fixture — at θ = 1.2 rad, chosen because it separates the two cases by
  30 mm. Rotated collider ⇒ deep overlap ⇒ PhysX ejects it (86.3 mm). Unrotated collider ⇒
  resting contact on the floor ⇒ < 2 mm. A render-only rotation fails this.

**A.3 stills for Big Will:** `slot/runs/angled/stills/contact_sheet.png`, five workstations at
−40°, −20°, 0°, +20°, +40°. Rendered at 2× supersampling and averaged over 12 frames, because
`samples_per_pixel` and FXAA are no-ops in this build and the raytracer's per-frame jitter has a
measured temporal std of 35.24/255 on a static scene. Five simultaneous viewports at 4× ran the
10.5 GB card out of memory, hence 2× and the workspace camera only.

---

## 4. The observation: 34-D → 36-D

`slot_frame` was `(depth, |lateral|, |yaw|, inserted)`. It is now

```
(depth, across, dyaw, cos θ, sin θ, inserted)
```

Two changes, both forced:

**Signed `across` and `dyaw`.** With a fixed slot the sign was recoverable from the block pose
alone, so absolute values cost nothing. With a random angle it is not, and handing the network a
sign and then throwing it away is a pointless tax.

**`(cos θ, sin θ)`.** Without it the privileged teacher cannot see where the slot points. As a
unit vector rather than a raw angle, so the input has no branch cut. The **student never
receives these two dimensions** — recovering the slot's angle from pixels is precisely the
problem the visual policy exists to solve.

New privileged layout, 36-D:

```
[ 0: 8] joint_pos_rel      [ 8:16] joint_vel_rel      [16:23] block_pose_in_root
[23:29] slot_frame         [29:36] last_action
```

The **student contract is unchanged at 23-D** — `[0:16] ⊕ [29:36]`, encoders and its own last
command — which is the point of having defined it as a slice list in one file.

### Downstream consequences

Every checkpoint and every HDF5/shard pool from before this change is incompatible. `OBS_DIM`
went 34 → 36 in `slot_act/dataset.py`, so an old pool now fails the width check rather than
silently mis-slicing.

Four scripts each carried a hand-written `obs[:, 27:34]` for the last-action term. All four
would have silently fed the network the wrong seven numbers. They now call a new
`cameras.student_inputs(obs, wrist, works)`, so the layout is expressed **once**.

`slot_act/noise.py`'s `LAST_ACTION_SLICE` moved 27 → 29. That one matters more than it looks:
it is the slice whose perturbation sigma is forced to zero, so an un-updated copy would have
injected noise into two slot-geometry dimensions while leaving two action dimensions clean.

A hazard worth naming: axis-aligned and angled **vision shards are structurally identical** —
same keys, same widths, and `proprio` is 23-D in both because the student contract did not
change. Only the images differ. Concatenating them would produce a pool that is half a
different task, with nothing to notice it. `slot_yaw_range` is therefore recorded in the
per-shard collection contract, which `dataset_vision` already asserts is identical across a
pool.

---

## 5. The fixture: four kinematic bodies, not one compound prim

The plan preferred "one kinematic rigid body containing all four boxes", to stop the parts
desynchronising. **Implemented as four separate kinematic bodies instead**, for two reasons:

1. The desync risk the plan was avoiding does not exist. All four poses are closed-form
   functions of a single θ, written in the same call. There is no integration and no drift.
2. A compound prim needs either a custom spawner or a generated USD, and either one puts the
   fixture's dimensions in a *second* place. The current code derives the boxes from the same
   `mdp` constants the success predicate reads — the property that stops geometry and scoring
   drifting apart, which the v0 basket does not have.

Kinematic rather than dynamic: the fixture is bolted to the table as far as the task is
concerned, it must not be shoved by a mis-aimed block, and PhysX skips kinematic-vs-kinematic
contacts, so the four deliberately overlapping boxes cost nothing per step.

The reset event is declared **after** `reset_scene_to_default`, which writes every rigid body
back to its authored yaw-0 pose. Declared before it, the yaw would be silently overwritten every
episode — the same ordering trap `--arm-jitter` hit (`HANDOFF` §9).

Each part's offset from `SLOT_CENTER` is read back from its own `default_root_pose` rather than
recomputed in the event, so the geometry stays defined in exactly one place.

---

## 6. The expert, rewritten in the slot frame

`slot/expert/plan.py`. The phase structure survives unchanged
(reach → lift → back → spin → turn → push → release → retreat); only the waypoints moved.

* The two measured constants keep their world-x names and meanings, reinterpreted as distances
  along the slot axis: `stage_dist = SLOT_CENTER.x − stage_x = 0.080`,
  `insert_dist = insert_x − SLOT_CENTER.x = 0.0095`. Every number in `EXPERT_RESULTS.md` still
  means what it says.
* `p_align = SLOT_CENTER − stage_dist·û`, `p_ins = SLOT_CENTER + insert_dist·û`.
* `axis_slot` (the finger axis at insertion) is per-env: `sign · [−sin θ, cos θ, 0]`.
* The `spin` phase interpolates the finger axis from the block's yaw to **θ**, not to 0.
* **The grasp is untouched.** The block still spawns where it always did, so the arm picks it up
  exactly as before. Deliberate: it keeps the angle as the single new variable, and it means a
  blind policy can still grasp — so a blind score near zero is evidence about *insertion*.

`p_stage` (the retract before the traverse) stays a straight pull in −x. Checked rather than
assumed: the fixture's closest approach to the robot moves from x = 0.210 (mouth face, θ = 0) to
x = 0.205 (front corner, θ = 40°), so the measured 45 mm of clearance at `stage_x` = 0.165 is
unaffected by the angle.

**`polar_turn` was generalised, and it reduces exactly.** It used to sweep the base angle to
y = 0 at fixed x. It now traverses the straight segment joining two arbitrary points with the
base angle advancing by smootherstep. At θ = 0 the segment is `x = const`, whose line equation
gives `r = x / cos θ` — the old `(x, x·tan θ)` formula exactly. It falls back to an eased lerp
in the two degenerate cases (segment near the base axis; endpoints at the same azimuth).

**A trap that would have silently corrupted every demo.** Both `run_expert.py` and
`collect_demos.py` reset, read the block pose, plan, then reset *again* and teleport the block
back to the pose the plan was built for. That second reset now re-randomises the **slot yaw**
too. Without restoring it, every episode would have executed a plan solved for one angle against
a fixture at another — and it would have looked like an ordinary precision failure, not a bug.
Both call sites now restore θ alongside the block pose.

`run_expert.py`'s reported `|yaw|` and its yaw failure attribution were switched from the
block's **world** yaw to `mdp.yaw_error` (slot-relative). Left alone, every correctly-squared
insert at a rotated slot would have been scored as a yaw failure.

---

## 7. Gate C — regression at θ ≡ 0

`python slot/scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 128 --slot_yaw 0`

```
seated success        128/128 = 100.0%
depth   mean 46.6 mm  p10 45.9  (need >= 40.0)
lateral mean 0.53 mm  p90 1.08
|yaw|   mean 0.0051 rad  p90 0.0166  (need <= 0.12, slot-relative)
failures: too shallow 0, yawed 0, not seated 0, lost grip before release 0
trajectory used 558/600 env steps
```

100.0 %, matching the pre-change expert, with the same depth margin and the same 0.53 mm lateral
error. The rewrite did not change anything other than the angle.

---

## 8. Gate B — θmax  *(running)*

`slot/scripts/run_theta_sweep.sh`, 11 cells at pinned angles from 0 to ±0.9 rad, n = 128 each.
θmax is the largest |θ| holding ≥ 0.95 seated with nothing smaller failing.

This is Big Will's *"ensure the task is solvable by the arm"*, answered with the scripted expert
on real physics rather than with an IK-convergence probe — IK says "reachable" for plenty of
poses the gripper cannot actually hold a block through.

**Results so far** (n = 128 per cell; ±0.5, ±0.7, ±0.9 still running):

| θ | deg | seated | depth mean / p10 | lateral | steps used | traverse | block lead at end of turn |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −0.350 | −20.1° | **90/128 = 0.703** | 42.0 / **35.6** mm | 1.02 mm | **585**/600 | 157.0 mm | **−8.8 mm** |
| −0.175 | −10.0° | 128/128 | 47.3 / 47.0 | 0.97 | 573/600 | 143.4 mm | −2.9 mm |
| 0.000 | 0° | 128/128 | 46.6 / 45.9 | 0.53 | 558/600 | 129.5 mm | +2.9 mm |
| +0.175 | +10.0° | 128/128 | 46.9 / — | 0.46 | 543/600 | 115.6 mm | +7.0 mm |
| +0.350 | +20.1° | 128/128 | 47.3 / 46.9 | 0.63 | 531/600 | 102.2 mm | +4.6 mm |

### 8a. The failure is asymmetric, and it is MY planner's, not the robot's

−20° scores 0.703 while +20° scores 1.000. That asymmetry is the whole finding, and it has a
mechanism that the table above makes visible.

**The block spawns at y ≈ −0.13 and the slot's approach point swings with θ.**
`p_align = SLOT_CENTER − 0.080·û`, so at +20° it sits at y = −0.027 and at −20° at y = +0.027 —
a traverse of 102 mm one way and **157 mm** the other, a 55 % difference across the sign of θ.

The carried block hangs 33 mm below the grip point: a pendulum of period
2π√(0.033/9.81) = 0.365 s. A longer sideways traverse swings it more, and it is **still
swinging when the push starts**. The last column measures exactly that — the block's position
along the slot axis relative to the TCP at the end of the turn. It runs monotonically from
+7.0 mm to −8.8 mm as the traverse lengthens. The push then moves the TCP a fixed distance, so
whatever in-hand offset exists at the start of the push survives to the end of it:

* lead +4.6 mm at +20° → final depth **47.3 mm**
* lag **−8.8 mm** at −20° → final depth **42.0 mm**, against a 40 mm threshold, with p10 at
  **35.6 mm**. Roughly 30 % of episodes fall under the bar.

**This is not an envelope limit.** At −20° the arm holds the block 128/128 all the way through
the push, the IK converges 128/128, the lateral error is 1.02 mm and the yaw error 0.0026 rad.
Nothing is out of reach. The block simply arrives late because my trajectory does not wait for
it to stop swinging. So the answer to Big Will's *"is the task solvable by the arm"* at ±20° is
**yes** — what fails is the expert's timing.

**A prediction I made here was wrong, and checking it is what confirmed the mechanism.** Steps
used grows with the traverse (531 → 585 out of 600), so I expected the −0.5 cell to overflow the
budget and time out for reasons unrelated to physics. It did not: **594/600**, no timeouts, 0
resets, and all 128 envs still holding the block at the end of the push. The step count grows
more slowly than the Euclidean traverse because the turn's waypoint count scales with the
*azimuth* sweep, not the distance.

So the negative-θ failure is **purely the depth deficit**, and it is monotone in traverse length
with nothing else contaminating it:

| θ | traverse | depth mean | seated |
|---:|---:|---:|---:|
| −0.175 | 143.4 mm | 47.3 mm | 1.000 |
| −0.350 | 157.0 mm | 42.0 mm | 0.703 |
| −0.500 | 168.2 mm | **34.2 mm** | **0.250** |

≈ 0.5 mm of depth lost per mm of traverse past ~143 mm.

**…and then the horizon *does* bind, one cell further out.** −0.700 reports 0/128 with all 128
failures attributed to *yaw* — which would be the classic signature of the wrist running out of
travel, and is not what happened. It used **603 of 600 steps**: every env timed out, auto-reset,
and the final measurement was taken on a freshly randomised block. Read the phase trace instead
and the expert was fine right up to the release — held 128/128, gap 29.95 mm, yaw error
**0.0036 rad**. The cell is a pure horizon artifact and says nothing about reachability at −40°.

Two lessons, both worth keeping: an end-of-episode metric is meaningless in any env that reset,
so `resets` and `steps_used` must be read *before* any failure attribution; and my earlier guess
that the overflow would hit at −0.5 was wrong by one cell, because the turn's waypoint count
scales with the azimuth sweep rather than the distance.

### 8a-2. +0.5 rad fails, and NOT for the same reason

| θ | seated | resets mid-episode | held at push | finger gap at push | yaw err at push |
|---:|---:|---:|---:|---:|---:|
| +0.350 | 128/128 | 0 | **128**/128 | 29.97 mm | 0.0035 rad |
| +0.500 | 103/128 = 0.805 | **25** | **108**/128 | **25.08 mm** | **0.0918 rad** |

At +0.5 the traverse is 91.7 mm — the *shortest* in the sweep — so the pendulum story of §8a
cannot apply, and indeed the block's lead at the end of the turn is a healthy +2.0 mm. Through
the turn everything is nominal: held 128/128, gap 29.95 mm, yaw error 0.0047 rad, and the block
sits within 0.1 mm of the slot centreline.

**It goes wrong during the push.** The finger gap closes from 29.95 to 25.08 mm — the fingers
have shut past where the block was, which is what happens when the block is levered out of the
pads — and 25 episodes terminate early. This is the failure mode `EXP_NOISE_SWEEP` already
named on the axis-aligned task: inside the channel, the block gets forced out of the grasp.

I do not yet know *what* it hits. The two candidates are a wrist joint running out of travel
(the IK would then clamp and the commanded finger axis would deviate) and a collision between
the gripper and the fixture that only opens up past 20°. **The IK's per-waypoint position and
axis error would separate them, and the sweep script threw that output away** — it greps stdout
down to the summary lines. That is my instrumentation bug, not an unknown: the sweep will keep
the full per-cell log and the failing cells get re-run with `--trace push`.

Note `plan converged 128/128` at +0.5, so whatever happens is not IK failing to find a
solution — it is either a clamped solution that still reports converged, or contact.

### 8b. The horizon fix — done

The episode budget is now **720 steps (14.4 s)**, up from 600. It had become the binding
constraint at large negative θ (603 steps at −40°, 609 at −51.6°), and 14.4 s rather than 14.0 s
because the action chunking executes in 15-step windows and 700 is not divisible by 15.

### 8c. "The block is still swinging" — FALSIFIED

The obvious reading of §8a is that the block is mid-swing when the push begins, so holding
longer at the end of the traverse should fix it. **It does not.** θ = −0.350, n = 128, the only
variable being the hold at the end of the traverse:

| `turn_settle` | steps used | depth mean | depth p10 | lateral | seated |
|---:|---:|---:|---:|---:|---:|
| 25 (default) | 585/720 | 42.0 mm | 35.6 mm | 1.02 mm | 0.703 |
| 60 | 620/720 | **42.0 mm** | **35.6 mm** | 1.03 mm | 0.688 |
| 120 | 680/720 | **42.0 mm** | **35.6 mm** | 1.01 mm | 0.703 |

Identical to 0.1 mm across a 5× change in hold time. Nearly two extra seconds of standing still
recovers **nothing**, which means the block is already at rest at the default hold and the
deficit is a **static offset established during the traverse**, not residual motion.

*(The first attempt at this probe was worthless and looked meaningful: at 600 steps the 60 and
120 cells ran 620 and 680 steps, timed out, and reported 0/128 with identical post-reset garbage
metrics. Same trap as the −0.7 sweep cell. The horizon fix above had to land first.)*

That leaves two static causes, with opposite fixes:

* **the arm never reached its own commanded waypoint** — a clamped IK solution would produce a
  permanent shortfall that no amount of settling removes;
* **the block slid inside the pads during the traverse** and friction is holding it there.

**Do not read the IK table's "max pos err" as evidence here.** It shows 0.192 m for the turn at
θ = −0.35 — and 0.905 m at θ = 0, which scores 100 %. Whatever it measures, it is not the
tracking error at the end of the phase.

### 8d. ROOT CAUSE: the gripper rolls, and the block is clamped to it

`run_expert` now decomposes the deficit **along the slot axis**: `tcp_err_mm` (arm vs its own
commanded waypoint) and `slip_mm` (block vs TCP), plus the block's tilt off vertical.

| θ | rate | phase | tcp_err | block vs TCP | **tilt** | block dz | gap |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.000 | 1.000 | spin | −0.13 | −1.23 | 3.33° | −33.13 | 29.95 |
| | | turn | **−0.10** | +3.01 | 6.38° | −32.91 | 29.95 |
| −0.500 | 0.250 | spin | −0.25 | +1.49 | 2.47° | −33.14 | 29.95 |
| | | turn | **−0.12** | **−12.46** | **23.27°** | **−30.63** | 29.95 |

**The arm is exonerated.** `tcp_err` is −0.13 mm at the end of the traverse at *every* angle
from +0.35 to −0.5. The arm goes exactly where it is told.

**The block rotates; it does not slide.** A block clamped 33 mm below the grip point and pitched
by φ displaces its centre by 33·sin φ along the push axis and *rises* by 33·(1 − cos φ):

* predicted from 23.27°: **13.0 mm** displacement, **2.6 mm** rise
* measured: **12.46 mm** displacement, **2.50 mm** rise (−30.63 vs −33.13 nominal)

Two independent quantities, both matching. And the finger gap never moves off 29.95 mm, so the
pads never lose the block — it is pitched, not slipping.

**It is a static equilibrium, not a dynamic one.** Three separate falsifications:

| hypothesis | test | result |
|---|---|---|
| still swinging | hold 25 → 60 → 120 steps | depth 42.0 / 42.0 / 42.0 mm — **no effect** |
| inertial slip in the pads | traverse 67 % slower (`turn_per_wp` 3 → 5) | 32/128 both, depth 34.2 / 34.3 — **no effect** |
| arm shortfall | `tcp_err` along the slot axis | −0.13 mm at every θ — **not the arm** |

The per-phase numbers at `turn_per_wp` 3 and 5 are identical to two decimals (−12.46, 23.27°,
−30.63) despite the trajectory taking 594 vs 706 steps. Nothing about the *motion* matters. The
gripper simply ends up rolled, and it is rolled because **the expert's IK leaves rotation about
the finger axis free**:

> *"This is a 5-DOF task, not 6: position plus a direction. Pinning the full orientation was
> measured to over-constrain the arm… The leftover roll about the finger axis is genuinely free
> for this task, so it is steered by a nullspace bias toward `q_bias`."* — `expert/ik.py`

That was a sound decision on the axis-aligned task, where the free DOF lands at 6.4° and the
block stays near-upright. The commanded finger axis is horizontal by construction
(`[−sin θ, cos θ, 0]`), so the free DOF is precisely a **pitch of the block along the push
direction** — and at −28.6° the minimal-change solution lands at 23°, which pitches the block far
enough that it can no longer enter the channel squarely.

Note `solve_path` *already* biases each waypoint's nullspace toward the previous solution, so
the obvious fix — "keep the roll continuous along the path" — is in place and insufficient: the
roll drifts from 2.47° to 23.27° across the traverse anyway.

### 8e. Does a level-gripper solution EXIST? **Yes — and it is nearby**

`slot/scripts/probe_roll_exists.py`, 512 random restarts per pose. The solution set at a fixed
(position, finger-axis) target is a 1-D curve, so it is sampled by seeding the IK from 512
uniformly random joint configurations in parallel — one per env — and keeping only solutions
that hit the target to 1 mm and 1e-3 of axis error.

| θ | pose | expert lands at | best of 512 restarts | below 5° | nearest level solution |
|---:|---|---:|---:|---:|---:|
| 0.00 | align | 23.32° | 15.85° | 0.0 % | 13.7° @ Δq 0.30 |
| 0.00 | insert | 0.04° | 0.40° | 4.0 % | 0.0° @ Δq 0.00 |
| −0.35 | align | 38.01° | **1.66°** | 3.3 % | **0.1° @ Δq 0.76** |
| −0.35 | insert | 4.67° | **0.40°** | 7.3 % | **0.0° @ Δq 0.16** |
| −0.50 | align | 44.51° | **0.04°** | 2.4 % | **0.1° @ Δq 0.85** |
| −0.50 | insert | 8.18° | **0.58°** | 2.8 % | **0.1° @ Δq 0.20** |
| −0.70 | align | 51.91° | **1.15°** | 3.3 % | **0.1° @ Δq 0.88** |
| −0.70 | insert | 18.34° | **1.20°** | 3.2 % | **0.1° @ Δq 0.30** |

Two separate questions, both answered:

1. **Does a level solution exist anywhere?** Yes, at every pose out to −40°: the best restart is
   within 1.7° of level, and 2–7 % of all converged restarts are below 5°.
2. **Is it reachable from where the expert already is?** Yes. Seeding locally around the
   expert's own solution (Gaussian perturbations at σ = 0.1, 0.3, 0.6 rad) finds level solutions
   at **Δq ≈ 0.2–0.3 rad at the insert pose** and **Δq ≈ 0.8 rad at the align pose**. That
   second question matters independently: if the only level solutions lived in a different elbow
   branch, no soft bias could reach them while carrying a block, and the fix would need a
   replanned branch rather than a nudge.

**So the 23–52° roll is a CHOICE the null-space bias makes, not a kinematic necessity. θmax is
expert-limited and can be lifted.** The fix is to steer the free DOF toward level. Because the
level configuration at the align pose sits ~0.8 rad away in joint space, it has to be chosen
during *planning* — pick the level branch at the align pose and carry it through the push —
rather than applied as a small correction at execution time.

**Two caveats, stated because they bound the claim:**

* Existence at two isolated waypoints is not a continuous level path. The push is 46 waypoints
  from align to insert; both endpoints admit level solutions, so a level path is plausible, but
  it is **untested**.
* Some roll is clearly tolerable. At θ = 0 the expert sits at 23.32° at the align pose — and no
  restart out of 183 beat 15.85° there — yet θ = 0 scores 128/128, because the channel
  straightens the block during the push (block tilt 6.38° → 1.49°). What breaks the insert is
  the growth to 44–52°, not roll as such.

#### A metric that had to be fixed twice before it could be believed

The first version reduced the reference orientation to its single largest component and measured
"the angle of one body axis from vertical" — a quantity with an arbitrary offset. It reported the
canonical seed itself at **30.88°** when that pose is the zero by construction, and disagreed
with physics by 48°. The corrected metric uses the whole vector `v = R_ref^T ẑ` — the world
vertical written in the gripper's own frame, which is env-independent because a grasp differing
by a yaw about the vertical leaves `ẑ` unchanged.

The second fix was to the *validation*, not the metric. Demanding that commanded gripper roll
equal measured block tilt flagged a correct metric as broken. The grasp is compliant: gravity
pulls the block back toward vertical, so it follows only about half the gripper's roll —
14.01° → 6.38° (ratio 0.46) and 43.35° → 23.27° (ratio 0.54). Gripper roll is the causal
quantity and the one the IK controls; block tilt is its damped consequence.

**Rejected: rotating the block spawn region with the slot.** It equalises the traverse, and the
plan named it as the contingency, but it would make the block's spawn position a function of θ —
handing the vision policy a shortcut to the very quantity it is supposed to read off the
fixture. A task that can be solved by looking at the block instead of the slot is not the task.

*Instrument note:* the "lost grip before release" counter was reading the finger gap at
**episode end**, when the gripper is open (89 mm) in every env — so it was an alias for "number
of failures" and diagnosed nothing. It now samples the gap at the end of the push, the last
moment the block is held. Fixed mid-sweep, so cells from +0.500 onward carry the corrected
value; the `fail_grip` field in the five cells above equals their failure count and should be
ignored. No other field is affected.

---

## 9. Order of remaining work

1. **Gate B** → set `DEFAULT_SLOT_YAW_RANGE` from the measurement, not the guess.
2. Delete the axis-aligned data (§4 of the plan; Big Will: keep the small checkpoints, the
   multi-GB data folders are what he wants gone).
3. Re-collect state demos → flow-BC → state champion on the angled task.
4. Vision: cameras unchanged, **blind control first**. Blind scored 0.254 on the axis-aligned
   slot by aiming at the one place the slot was ever going to be; on the angled task it should
   collapse toward zero, and that collapse is the evidence that the task now requires
   perception.

Reuse deliberately, and do not repeat, the list in `ANGLED_SLOT_PLAN.md` §5 — in particular
**DAgger with a BC-clone teacher** (−31.3 pts) and **x0-steering as parameterised** (0.000/96
for every broadcast latent).
