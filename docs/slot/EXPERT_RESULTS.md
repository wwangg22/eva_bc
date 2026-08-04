# Stage A/B — feasibility settled and a working scripted expert

**Session 2, 2026-08-02.** Supersedes parts of `HANDOFF.md` §3d and §7; see §1 for the
retractions. Everything here is measured, and every number names the script that produced it.

---

## 1. Retractions and corrections to session 1

Session 1 ended believing three things that are now known to be wrong. Each was wrong for an
instructive reason.

### 1a. RETRACTED — "the render-mesh bound is an over-estimate; physics contradicts it"

`HANDOFF.md` §3d warned in a block quote *"Do NOT use the render-mesh bound"* because
`gripper_envelope.py` reported a minimum wall-clearing grip height of 90.8 mm while a physics
stroke test appeared to clear at TCP z = 0.084. **The render bound was right.** The two tests
were measuring different geometries:

* Stage 2a stroked a **closed** hand (`q_fing = 0.0`). A shut gripper is narrow and slides
  down the middle of a 33 mm channel untouched, so it "clears" at any height.
* Stage 1 computed clearance for a hand **holding the block** (`q_fing = 0.016`, pads at
  |y| = 16 mm against wall inner faces at |y| = 16.5 mm). That is the configuration that
  matters, and it is the one the render bound described.

Read the right column and physics and geometry agree perfectly, monotonically:

| grip z | Stage-1 wall clearance (holding) | Stage 2b finger gap | outcome |
|---|---|---|---|
| 0.072 | −19.8 mm | 50.8 mm | pads never reach the block |
| 0.078 | −15.0 mm | 49.3 mm | same |
| 0.084 | −3.5 mm | 35.4 mm | pads jam on the wall tops, block flung 130 mm |
| 0.090 | **+0.4 mm** | 30.0 mm | **holds** |
| 0.096 | **+6.1 mm** | 30.1 mm | **holds** |

**Standing conclusion: a gripper holding the block clears the slot walls only at TCP
z ≥ 0.090.** This is the single tightest constraint in the task and it drives the whole
expert design (§3).

*Lesson:* "physics contradicts geometry" was really "I compared two different configurations."
Before believing a contradiction, check that both sides describe the same object.

### 1b. RETRACTED — "CEM jumps IK branch at grip z = 0.078"

Session 1 attributed a 158 mm tracking error to a CEM branch jump at z = 0.078. Re-running the
same script put the anomaly at **z = 0.090 instead**, with 0.084 and 0.096 clean. The
*specific cell* was noise; the *conclusion* — that CEM is not reproducible enough to generate
demonstrations — is strengthened, not weakened, by the anomaly moving. CEM is now used for
exactly one thing: choosing the elbow branch for a single seed pose. All accuracy comes from
DLS IK (`slot/expert/ik.py`).

### 1c. NEW — `mdp.is_inserted` is much weaker than it reads

`is_inserted` bounds block z only from **below** (`z > SLOT_FLOOR_Z − 0.005 = 0.015`, intended
to catch a block that fell off the table). There is no upper bound. So the predicate is
satisfied by a block that is **resting on top of the 30 mm walls**, or **still dangling in a
closed gripper** above the slot, as long as x, |y| and yaw are right.

Measured: an early `strategy_probe.py` cell scored **93.8 % on `is_inserted` with a mean
lateral error of 13.28 mm** — geometrically impossible *inside* a channel of 16.5 mm
half-width for a block of 15 mm half-width. Another cell scored 100 % while the finger gap was
33.86 mm (pads jammed on the walls, not gripping a 30 mm block) and the block sat at z = 67 mm,
12 mm above its seated height.

**Every number in this document is therefore reported against a strengthened predicate:**

```python
seated = mdp.is_inserted(env) & ((block_z - 0.055).abs() < 0.006)
```

measured **after the fingers open**, so a held block cannot count. The raw predicate is
reported alongside so the gap stays visible. The env is a shared tracked file and has not been
modified; this is an evaluation-side guard only.

---

## 2. Conventions, settled (`slot/analysis/conventions.py`)

Three things IK depends on, measured rather than reasoned about:

* **Quaternions are XYZW.** Writing XYZW reads back unchanged; on the *same* tensor
  `mdp.yaw_of` returns +0.3000 for a commanded +0.30 yaw and `quat_apply(+z).z` returns
  +1.0000. There is no contradiction in the repo — `common.object_quat`/`yaw_of` and
  `terminations.block_toppled`/`quat_apply` are both correct.
* **`root_physx_view.get_jacobians()` → `(N, 10, 6, 14)`, `is_fixed_base = False`.** 14 = 6
  floating-base DOFs + 8 joints, body dimension not offset. Session 1's check sliced columns
  without the +6 base offset, which is why it reported a 3.384 relative mismatch.
* **TCP is unambiguous**: `ee_frame`, a hand-rolled `gripper_end + quat_apply(TCP_OFFSET)`,
  and the shut-finger midpoint all agree to **0.04 mm**.

---

## 3. The expert (`slot/scripts/run_expert.py`, `slot/expert/ik.py`)

### 3a. Insertion strategy: horizontal drag, chosen by measurement

`strategy_probe.py` pre-registered and compared two strategies with the block already in the
gripper (so grasp error is excluded):

| strategy | seated success | why |
|---|---|---|
| Cartesian-straight +x drag at TCP z = 0.090 | **100 %** | block stays at z ≈ 55 mm the whole way |
| vertical lower-in, release above the seat | 46.9 % | 15 mm release drop costs ~3 mm of depth |

The vertical strategy's failure is **not** precision: success was flat (~40–50 %) across every
|dy| bin from 0 to 3 mm and every |dyaw| bin from 0 to 0.06 rad, while mean depth landed at
**39.9 mm against a 40 mm threshold**. It is a depth-margin failure. The drag also needs 30 mm
less reach, so it wins on both counts.

This also overturns the env write-up's premise that the block must be dropped in from above
and that "the arm cannot release from above". Neither is required.

### 3b. Why the original stroke scored 8–12 %

Session 1's Stage 2b measured 8.2–11.7 % with the block genuinely gripped (finger gap 30.0 mm)
and the TCP reaching its commanded 45 mm depth to under 1 mm, yet the block stopped at 29.5 mm.
The cause was **joint-space interpolation**: a joint-linear path does not move the TCP in a
straight line, and the bow dragged the block into a wall. In the first traced run the block
climbed from z = 55 mm to **69 mm**, riding up onto the wall tops.

With dense Cartesian waypoints through DLS IK, same geometry and same grip, the block stays at
z ≈ 55 mm, lag peaks at 2.65 mm mid-stroke and **recovers to 0.21 mm**, and the result is
100 %. Neither "slip" nor "jam" — the two hypotheses pre-registered in the probe — was correct.

### 3c. The IK

`slot/expert/ik.py` solves a **5-DOF task: TCP position + finger-axis direction**, not a full
6-DOF pose. Pinning the whole orientation over-constrains a 6R arm — measured, it left
**3.6 mm** of residual position error at the nominal pose and **15.6 mm** along a path, with
**0/64 envs converged**. The wrist roll about the finger axis is genuinely free for this task,
so it is steered by a **nullspace bias with a decaying gain** rather than constrained.

The decay matters. With Levenberg damping, `(I − J⁺J)` is only an approximate nullspace
projector, so a constant bias leaks into task space and fights convergence: a straight 74 mm
push had 23 % of its waypoints above 0.5 mm and 17/64 envs converged. Decaying the gain to
zero over the iteration budget keeps the branch guidance early, where it decides the elbow, and
leaves the endgame to the task term. After the fix **every waypoint of every phase converges**
(max position error 0.4 mm).

Warm-starting each waypoint from the previous solution gives branch continuity by construction
— the structural fix for §1b.

### 3d. Phase machine

`reach → grasp → lift → back → spin → turn → push → release`, all IK solved **before the block
is touched** (`write_joint_state_to_sim` teleports the arm and re-opens the fingers, so any
solve after the grasp destroys the grasp). Execution is then open-loop, which is why grasp
error propagates straight through to the insert — the quantity the script reports.

Geometry that forced each choice:

* **`grasp_h = 0.031`** (TCP above block centre). Bounded below by the wall clearance
  (TCP ≥ 0.090 with the block seated at 0.055 ⇒ ≥ 35 mm) and above by the block's own
  half-height (35 mm). Measured in-hand result: TCP ends **33.2 mm** above the block centre.
* **`carry_z = 0.095`.** At 0.090 the carried block's bottom sat at **20.9 mm** against a slot
  floor whose top is a 20 mm step at x = 0.210 — 0.9 mm of clearance. Raising to 0.095 also
  *increases* wall clearance. This alone took the expert from 53.1 % to 79.7 %.
* **`stage_x = 0.165`.** See §4 — this is the one that mattered most.
* **spin separated from traverse.** Rotating the wrist while also accelerating the block
  sideways is avoidable; `spin` in place holds 64/64 grips.
* **600-step budget is hard.** The episode is 600 steps and the policy gets the same. An early
  version ran 659 steps and every env timed out, reading as 0 % success. The script now counts
  its own steps and prints `trajectory used N/600`.

---

## 4. The turn-phase grip loss: three refuted hypotheses, then the answer

Worth recording in full, because three plausible explanations each predicted the wrong thing.

**Symptom.** During the y-traverse the finger gap grew from 29.95 mm to 36.4 mm and ~1/3 of
grips were lost. On Loose-v0 the channel is 36 mm wide, so a block presenting 36 mm jams
exactly at the mouth. Everything upstream was clean: `SPIN` held 64/64 at 29.95 mm.

| # | hypothesis | test | result |
|---|---|---|---|
| 1 | pendulum swing (block hangs 33 mm below the grip; traverse ≈ 4.4 pendulum periods) | smootherstep easing, zero end-point acceleration | **refuted** — 64.1 % → 65.6 %, gap still 36.4 mm |
| 2 | wrist tracking lag carrying the block with it | per-waypoint trace of the finger axis | **refuted** — `axis err ≤ 4e-4` throughout; the gripper holds its axis exactly, so the block really does rotate inside the pads |
| 3 | speed / dynamics | traverse 2.3× slower | **refuted** — gap only 36.4 → 33.3 mm, and onset stayed at the same *position* (cmd y ≈ −22 mm), not the same time |
| 4 | joint-space jerk near a singularity (`dθ₁/dy = x/(x²+y²)` grows 50 % as y → 0) | re-parametrise the traverse in polar coordinates so joint1 sweeps uniformly | **refuted** — 62.5 %, peak joint1 step got *worse* (0.0732 → 0.0881 rad) |

The one robust fact was position-dependence, so the decisive test was to run the identical
traverse 45 mm higher, lifting the block clear of the fixture:

| carry_z | block bottom | finger gap after traverse | grips held | final block yaw |
|---|---|---|---|---|
| 0.095 | 27 mm (below the 50 mm wall tops) | 29.95 → **36.60 mm** | 20/32 | 0.105 rad |
| 0.140 | 74 mm (clear of the walls) | 29.95 → **29.95 mm** | **32/32** | 0.0050 rad |

**It was a collision with the slot fixture the whole time.** My arithmetic had said "clear"
because I assumed the block sits at the TCP's x. The same output shows it does not: the
carried block sits **~5 mm ahead of the TCP in x** (block x = 185.1 at `stage_x` = 0.180), so
its nose reached ~205 mm against wall front faces at 210 mm, closing to ~3 mm once the block
yawed. Retracting to `stage_x = 0.165` costs nothing and removes the interference entirely.

*Lesson, and it is the same one as §1a and as gotcha 11 in `HANDOFF.md`:* the failure was in a
quantity I had asserted rather than measured. Three of the four hypotheses were about
**dynamics** because I had already convinced myself the geometry was clear.

---

## 5. Results

`slot/scripts/run_expert.py --stage_x 0.165 --num_envs 128`, judged by the **seated**
predicate with the fingers opened first. Logs: `slot/logs/ladder_*.log`.

| task | clearance | seated | lateral mean / p90 | \|yaw\| p90 | depth p10 | resets |
|---|---|---|---|---|---|---|
| Loose-v0 | 3.0 mm | **128/128 = 100 %** | 0.67 / 1.38 mm | 0.0198 rad | 46.2 mm | 0 |
| **v0 (the target)** | 1.5 mm | **128/128 = 100 %** | 0.55 / 1.11 mm | 0.0109 rad | 46.2 mm | 0 |
| Tight-v0 | 0.5 mm | **128/128 = 100 %** | 0.30 / 0.48 mm | 0.0037 rad | 46.2 mm | 0 |

Zero failures in every category (too shallow / yawed / not seated / lost grip), grip held
128/128 through every phase at a constant 29.95 mm gap, block z exactly 55.0 mm, 510/600 steps.

Two internal consistency checks worth stating, since 100 % on a 0.5 mm clearance invites
suspicion:

* **Lateral error falls as the slot narrows** (0.67 → 0.55 → 0.30 mm). The three variants
  really are different fixtures, and the narrower walls are physically constraining the block
  — the walls do the fine alignment, which is what the env's own docstring says they are for.
* **Depth p10 = 46.2 mm** against a maximum possible 47.5 mm (nose against the back stop). The
  block is being driven up to the back stop, which squares it and removes depth variance.

This clears `PLAN.md`'s Stage B expert bar (>= 85 % Loose, >= 75 % v0) with margin. eva_bc's
own history is the reason the bar was set there: a 94 %-nominal expert produced a 59-64 % BC
policy, so an expert below ~85 % makes the 70 % goal unreachable by training alone.

**Video** (v0, 4 envs tiled, 16/16 success, 10 s):
`eva_bc/slot/logs/expert/expert_Rebot-PrecisionSlot-v0.mp4`

### 5a. Caveat found while recording: the seed was batch-size dependent

Recording at n = 4 scored **50 %**, on the same task where n = 128 scored 100 %, with the
traverse ending at y = −12.9 mm instead of 0. Cause: CEM's population size *was* `num_envs`
(`elite = argsort()[:max(8, n // 20)]`), so a small batch produced a poorly-seeded elbow
branch and hence a different trajectory. The seed is a property of the robot, not of the
batch. It is now solved once at large n, validated, and cached to
`slot/logs/expert/seed_q.json` (keyed on `insert_x`/`carry_z`), which also makes the expert
**deterministic** — a requirement for reproducible demo collection. Re-run at n = 16 with the
cached seed: 16/16.

*This is the same failure mode as §1b*: a quantity that varied with something it should not
have depended on, silently.

---

## 6. Files

```
slot/expert/ik.py                 batched DLS IK, 5-DOF task, decaying nullspace bias
slot/scripts/run_expert.py        phase machine + end-to-end measurement
slot/analysis/conventions.py      quaternion order / Jacobian layout / TCP identity
slot/analysis/strategy_probe.py   horizontal vs vertical insertion + tolerance surface
slot/analysis/insertion_feasibility.py   session-1 geometry sweep (numbers superseded)
slot/logs/                        run logs and JSON evidence
```

Nothing tracked in either repo has been modified. No push has been made.

---

## 7. Next

1. **Ladder results** → confirm v0 (1.5 mm, the target) and Tight (0.5 mm).
2. **Demo collection**: HDF5 in eva_bc's schema, with phase labels and `train_mask`, from the
   full reset distribution. Target ≥ 85 % Loose / ≥ 75 % v0 (`PLAN.md` Stage B).
3. **Record a video** and give Big Will the path.
4. Stage C: port the flow-matching chunk BC to 34-D obs / 7-D action (`PORT_MAP.md`).

One design point to carry into Stage C: the expert is **open-loop after the grasp**, so its
demonstrations contain no corrective behaviour. eva_bc's postmortem attributes part of its BC
plateau to exactly this. DAgger or a deliberate injection of grasp perturbation will likely be
needed before RL.
