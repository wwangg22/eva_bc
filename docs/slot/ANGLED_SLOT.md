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

**A second, purely artificial failure is about to bite.** Steps used grows with the traverse:
531 → 585 out of a 600-step budget across the range measured so far. The −0.5 and −0.7 cells
will overflow it, and every env in them will time out *regardless of the physics*. Any rate
below ~0.7 at large negative θ must therefore be read as a budget artifact until the horizon is
raised, not as a reachability result.

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

### 8b. The fix that follows from the diagnosis

1. **Damp the swing before the push.** `SETTLE["turn"]` is 25 steps = 0.50 s = 1.4 pendulum
   periods. Raise it so the block is actually still when the push begins.
2. **Raise the episode budget** from 600 steps (12.0 s) so the horizon stops being the binding
   constraint. It must stay divisible by the 15-step action window — 720 steps (14.4 s) does,
   600 → 700 would not.

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
