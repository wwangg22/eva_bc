# Porting the eva_bc pipeline to `Rebot-PrecisionSlot-*`

**Started 2026-08-02.** Working directory for all new code: `eva_bc/slot/`.
Docs for this port: `eva_bc/docs/slot/`.

> Status legend used throughout: **[TODO]** not started · **[RUN]** in flight ·
> **[DONE]** measured and written up · **[BLOCKED]** waiting on something · **[DEAD]** ruled out.

---

## 0. The objective

Train a policy that inserts the block into the slot from **randomized starts**, targeting
**70 % success** on `Rebot-PrecisionSlot-v0` (1.5 mm per-side clearance). The user's framing:
70 % is a target, not a hard requirement, "especially if the task requires a lot of precision".

The difficulty ladder gives three rungs, and they are the natural staging:

| gym id | per-side clearance | role here |
|---|---|---|
| `Rebot-PrecisionSlot-Loose-v0` | 3.0 mm | development rung — prove the pipeline end to end |
| `Rebot-PrecisionSlot-v0` | **1.5 mm** | **the target** |
| `Rebot-PrecisionSlot-Tight-v0` | 0.5 mm | stretch / difficulty-axis evidence |
| `Rebot-PrecisionSlot-Play-v0` | 1.5 mm, 16 envs | watching checkpoints, video |

---

## 1. The task, precisely

All coordinates env-local, metres, table top at z = 0, robot base at the origin.

**Block** — cuboid `45 x 30 x 70 mm` (half-extents `(0.0225, 0.0150, 0.0350)`), 0.04 kg,
spawns upright at `(0.220, -0.130, 0.035)`.
Reset randomization: `x +/- 0.02`, `y +/- 0.03`, **`yaw +/- 0.35 rad`**.
So the spawn set is `x in [0.200, 0.240]`, `y in [-0.160, -0.100]`, `yaw in [-20.1, +20.1] deg`.

**Slot** — centre `(0.245, 0)`, floor top `z = 0.020`, depth 0.070 along +x, walls 30 mm tall
(**tops at z = 0.050**) and 8 mm thick, inner half-width `0.0150 + clearance`, back stop at
`x = 0.280`. Derived: **mouth plane x = 0.210**.

**Success** (`mdp.is_inserted`), all four simultaneously:
- `depth = block_x - 0.210 >= 0.040` i.e. **block centre x >= 0.250**
- `|block_y| <= slot_half_width` (0.0165 at v0) — the walls enforce this once it is inside
- `|yaw| <= 0.12 rad` (6.9 deg)
- `block_z > 0.015`

**Actions, 7-D** — `arm_action` 6 joint position targets, `scale = 0.5`,
`use_default_offset = True`, so `action = (q_desired - q_default) / 0.5`; plus
`gripper_action` binary (`< 0` closes to 0.0, `>= 0` opens to 0.045 per finger).
**Identical in shape to the pick-place task eva_bc was built on** — the action plumbing ports
unchanged.

**Observations, 34-D** (pick-place was 41-D — this is the main dataset edit):

| slice | dim | term |
|---|---|---|
| `0:8` | 8 | `joint_pos` (`joint_pos_rel`, 6 arm + 2 finger) |
| `8:16` | 8 | `joint_vel` (`joint_vel_rel`) |
| `16:23` | 7 | `block_pose` in the robot root frame, pos + quat **XYZW** |
| `23:27` | 4 | `slot_error` = `(depth, lateral, yaw, inserted)` |
| `27:34` | 7 | `actions` (`last_action`) |

**Episode** — 12 s, `decimation = 8`, `dt = 1/400` -> **50 Hz policy, 600 steps**.
Terminations: `time_out`, `block_dropped` (z < -0.05), `block_toppled` (own +z axis has
world-z < 0.6).

**Finger drive override** — the env already splits the actuator group and sets the fingers to
`stiffness 2000 / damping 40`; the revolute joints keep `None` so the USD's validated gains
survive. Do not touch this.

---

## 2. Measured hardware constraints that bound every design decision

From `eva_rl/docs/CHALLENGE_SUITE.md`, all measured on this asset in this sim. These are the
constraints that make the task arm-specific.

| id | constraint | consequence for this port |
|---|---|---|
| **C1** | **0.00 %** of table-height voxels admit a finger axis within 26 deg of vertical; steepest available anywhere in the band is 42.3 deg off vertical | no top-down grasp, no vertical peg-in-hole. The insertion is a **horizontal slide along +x**, and the grasp is a **side grasp across the block's 30 mm width** |
| **C2** | USD authors the finger drive at 100 N/m -> ~1.7 N squeeze, ~0.05 kg payload | env already overrides to 2000/40. Block is 0.04 kg, comfortably inside |
| **C3** | `_GRIPPER_OPEN = 0.045` is a **per-finger joint value**; `separation = 2.000 * q` exactly; commanded opening **89.1 mm**, forced ~120 mm. Calibration `gap = 1.0035*(q_l+q_r) - 1.25 mm`, residual 0.035 mm | a 30 mm block sits at `q ≈ 0.0156` per finger when gripped. **The finger joint values are a direct, unfakeable grasp-success readout** — this is what the grasp-bit probe will use |
| **C9** | **TCP cannot go below ~44 mm above the table**; reliable band `x ≈ 0.22–0.26`, `z ≈ 0.045–0.10` | the grasp must take the block's **upper half** (block spans z 0 to 0.070 on the table). Grip height is the tightest design variable in the task |
| **C10** | TCP is at `(-0.0419, 0, 0)` from `gripper_end`, **not** -0.075 | use `mdp.TCP_OFFSET`. A 33 mm error here closes the fingers past the object every time |
| **C7** | 11 GB RTX 2080 Ti | budget 512–1024 envs for this contact-rich task; camera rendering above ~256 envs throws an illegal memory access |

**The unresolved one.** `docs/envs/precision-slot.md` **withdraws** the insertion probe's
achievability numbers: they were produced under the wrong TCP, and with the corrected value
"the fingers now clip the slot wall tops". So the load-bearing rung V4 (achievability) is
*unproven, not disproven*, and re-establishing it is step one of this port. The geometry is
brutally tight by construction: block centre sits at **z = 0.055** when inserted, wall tops
are at **z = 0.050**, and the TCP floor is **z = 0.044**.

---

## 3. What eva_bc gives us, and what has to be rebuilt

eva_bc is a staged pipeline with evaluation gates: scripted expert -> flow-matching chunk BC
-> batched sim eval -> DAgger -> validated grasp-success obs bit -> RL on the frozen base
(additive residual measured flat; **x0-steering** recommended).

| stage | eva_bc component | portability |
|---|---|---|
| 1 | `expert/run_expert_v1.py`, cuRobo planner | **REBUILD — cuRobo is not installed in `env_isaaclab6`.** See §4 |
| 2 | `act/train_flow.py` + `modeling_flow.py` + `dataset.py` | ports; the edit is the obs slice map 41-D -> 34-D |
| 3 | `act/eval_act.py` (vectorized controller, tensor queue, flush trigger) | ports; needs the new `mdp` touchpoints |
| 4 | `expert/collect_dagger.py` | ports once the expert exists |
| 5 | `experiments/exp06_grasp_bit.py` | retrain on our data; the finger channels are even cleaner here (C3) |
| 6 | `act/steer_*.py`, `train_steer.py`, `steer_ppo_cfg.yaml` | ports; obs builder needs the 34-D layout |
| 7 | `experiments/taxonomy.py`, `diag_training_env.py`, `exp07_check_match.py` | taxonomy buckets must be redefined for this task's failure modes |

### The `mdp` touchpoint contract eva_bc expects

`act/eval_act.py`, `residual_core.py`, `steer_core.py` call into the task package:

| eva_bc expects | slot equivalent |
|---|---|
| `mdp.placed_mask(env) -> (N,) bool` | `mdp.is_inserted(env)` — already exists |
| `mdp.object_pos_local(env, name) -> (N,3)` | exists in `challenge/mdp/common.py` |
| `mdp.basket_centers_local(env) -> (N,2)` | **not applicable** — the slot is fixed at `(0.245, 0)`. Drop it |
| `mdp.OBJECT_NAMES` | `("block",)` — single object |
| an `objects_canonical` obs term with target-first ordering | **not applicable** — one object, no ordering. Simplifies the obs builders |

A single shim module in `eva_bc/slot/` provides these, so nothing in `eva_rl` needs editing
for the pipeline to run.

---

## 4. The expert problem (cuRobo is missing)

`python -c "import curobo"` fails in `env_isaaclab6`. Installing cuRobo is a multi-hour
detour with its own risk surface, and eva_bc's own `PLAN.md` records the cuRobo path as
having cost days of onboarding. Three alternatives, ranked:

1. **Batched damped-least-squares IK inside the sim, verified by FK** *(chosen)*. Isaac Lab
   ships a batched Jacobian; we iterate DLS with `write_joint_state_to_sim` between steps
   (pure kinematics, no dynamics), then **measure the achieved TCP error and finger-axis
   alignment** and reject any env that did not converge. This is per-env, so it handles the
   randomized spawn natively, and it cannot silently return an unreachable pose.
2. **CEM over the 6 arm joints scored by FK in sim** — already proven on this arm in
   `eva_rl/scripts/challenge/slot_insertion_probe.py`. Downside: the population *is* the env
   batch, so it solves one target at a time. Use it to build a **waypoint lookup table** over
   the spawn grid (eva_bc's `grasp_table.pt` pattern) and as the fallback/verifier for (1).
3. Install cuRobo. Held in reserve.

Two rules carried in from the measured record, both of which have already burned this project:

- **Finish every CEM/IK search before placing the object.** These searches teleport the arm
  and reopen the fingers hundreds of times; running one after closing the gripper silently
  drops the object and reads as a slip.
- **Position-only IK is not a sufficient specification for this arm.** The finger-separation
  axis must be constrained explicitly, or the wrist arrives holding the block along its 45 mm
  length instead of across its 30 mm width. Measured: constraining it took the insert rate
  from **0 % to 55–81 %**.

---

## 5. Stage plan, with pre-registered targets

Each stage has a target that decides whether the next stage runs. Following eva_bc's own
convention: write the design and the decision rule down *before* running, and record verdicts
in place including retractions.

### Stage A — geometry and achievability **[RUN]**

The withdrawn V4 rung. Nothing else is worth building until this is settled.

| step | question | target |
|---|---|---|
| A1 | how far below the TCP does the hand hang? | a number, with the minimum wall-clearing grip height derived from it |
| A2 | is there a grip height that clears the wall tops (0.050) *and* the table (0.000) *and* sits above the TCP floor (0.044)? | a non-empty band, or the task needs a geometry change |
| A3 | can the arm hold a straight-line +x stroke from mouth to home at that height with the finger axis on world y? | TCP error < 4 mm and alignment error < 0.05 across the stroke |
| A4 | scripted grasp -> lift -> transport -> insert at the **nominal** spawn | **>= 80 % on Loose-v0** |

**Decision rule.** If A2 returns an empty band, the honest finding is that the fixture
geometry is infeasible as authored, and the fix is a documented geometry change (lower walls
or a taller block) rather than a policy. That would be a real result, and it gets written up
with the measurement rather than worked around silently.

### Stage B — scripted expert over the full reset distribution **[TODO]**

| target | value |
|---|---|
| expert success, Loose-v0, full randomization | **>= 85 %** |
| expert success, v0 (1.5 mm), full randomization | **>= 75 %** |
| first-attempt clean grasp rate | >= 90 % |

The expert's number **caps everything downstream** — eva_bc measured a 94 %-nominal expert
producing a 59–64 % BC policy. An expert below ~85 % makes the 70 % BC+RL target unreachable
and would force a redesign of the approach rather than more training.

Deliverable: HDF5 demos in eva_bc's schema with per-segment phase labels and a `train_mask`
that loss-censors the expert's own failed sub-attempts.

### Stage C — flow-matching chunk BC **[TODO]**

| target | value |
|---|---|
| pooled >= 128-episode held-out success on Loose-v0 | >= 55 % |
| pooled >= 128-episode held-out success on v0 | >= 40 % |

**Non-negotiable protocol, from eva_bc's most expensive lesson:** the same data + recipe
spanned **32.8 %–59.4 %** across training seeds, and same-data different-seed pairs flipped
31–39 of 64 episodes. So: **>= 3 training seeds per arm, champion selected on a held-out spawn
seed, pooled >= 128-episode numbers only. Single-run comparisons are void.**

Also carried in: **chunk commitment is load-bearing** — shortening the execution horizon at
eval time collapsed success monotonically 59.4 -> 32.8 -> 3.1 -> 0 -> 0 % at
`n_action_steps = 15/8/4/2/1`. Keep chunk 50 / execute 15. Never shorten the horizon; put RL
*on top of* committed chunks.

### Stage D — RL refinement on the frozen base **[TODO]**

Target: **70 % on `Rebot-PrecisionSlot-v0` random starts.**

x0-steering (RFS-style) rather than an additive action residual, on eva_bc's measured
evidence: the additive residual came out *exactly* flat (55.5 % -> 55.5 %, 26 episodes fixed /
26 broken, causal because both sides are deterministic), and the learned residual was
state-independent — PPO learned effort, not discrimination. Whereas fixed x0 draws alone span
**14.1 %–56.2 %** success on the same frozen base, which is the leverage available to steer.

Gates, in order, all inherited:
1. **wrapper bit-exactness** — z = 0 reproduces the frozen base episode-for-episode before any
   training. Cheap, decisive, permanently exonerates the wrapper.
2. **early health** — per-term episodic reward channels sane from epoch 1. A broken run is
   diagnosable there, not after 600 epochs.
3. **result** — pooled >= 128-episode held-out vs the deterministic-base baseline, with a
   pre-registered adoption rule, **plus a mandatory taxonomy diff**: did the targeted failure
   bucket collapse *without* symmetric new breakage?

`rl_games clip_actions` must be **1.0**. It is an action *scale*, not a clip —
`preprocess_actions` clamps to [-1, 1] and then rescales to the action-space bounds. This cost
eva_bc two full training runs.

### Stage E — a PPO-from-scratch control **[TODO, cheap]**

The env ships a full Factory-style multi-scale keypoint reward that **has never been run**
(`eva_rl/docs/HANDOFF.md` item 5). One short PPO run is the cheapest possible reading of
whether there is a gradient to climb, it is the V7 rung the suite defines, and it gives the
BC+RL result something honest to be compared against. Not the main line, but worth the GPU
hour once Stage A settles.

---

## 6. Standing rules for this port

Carried from eva_bc's lessons, all measured, all expensive:

- **Verify wrappers by bit-exact reproduction before any training.**
- **Never trust train reward** — the flat residual run *beat* the base on train reward
  (+1700 vs +1644) with zero success change.
- **Fixed-action attribution beats theorizing.** When training reward looks wrong, reproduce
  the exact training configuration and roll fixed action conditions through it (zero / bias /
  small noise / large noise), no RL.
- **"Zero-init residual" needs three things**: zero the mu weight, know the mu bias stays
  random, and use a **small initial sigma** — sigma ≈ 0.37 (about 1 deg/joint/step) destroyed
  a base that lives on mm-precision grasps.
- **Aggregate success rate hides real change** — always diff per-episode against a measured
  churn floor, and know that the floor for comparing two *training runs* is 31–39 episodes,
  not zero.
- **Information can be present in the obs but unused.** The grasp-success distinction was
  decodable from 5 raw dims at AUC 0.968 / 0 % FPR while the same probe on all 41 dims
  mislabelled 53.5 % — the fix is re-surfacing the signal as an explicit validated bit, not
  adding history (refuted: <= +0.01 AUC, transfers worse). The probe **needs the commanded
  grip channel**, not aperture alone.
- **A self-consistent harness proves nothing.** The withdrawn slot probe placed the block at
  its own computed TCP, so it looked right under any TCP offset. Validate against something
  that does not share the assumption.
- **One GPU job at a time** (11 GB). Long jobs run as background chains with durable logs.

---

## 7. Log

| date | stage | entry |
|---|---|---|
| 2026-08-02 | S0 | Both repos pulled, already up to date. `env_isaaclab6` verified: torch 2.11.0+cu128, CUDA available, rl_games / h5py / robomimic present. **cuRobo absent** -> expert must be rebuilt (§4). |
| 2026-08-02 | S0 | `scripts/test_precision_slot_env.py` **PASSES** on `Rebot-PrecisionSlot-v0`: 34-D obs, 11 reward terms, gap 33.000 mm vs 30.000 mm block = 1.500 mm per side, predicate fires 16/16 at the home pose, a block forced in at 0.60 rad relaxes to 0.220 rad. |
| 2026-08-02 | A1/A2 | `slot/analysis/gripper_envelope.py`: **reachability is NOT the constraint.** CEM hits every commanded TCP to **< 0.3 mm** with finger-axis alignment error < 0.004 across `x ∈ [0.18, 0.25]`, `z ∈ [0.060, 0.096]`, and at the block spawn for `z ≥ 0.052`. The binding constraint is the hand's vertical envelope: it hangs **40.8 mm below the TCP** (global min over all hand geometry), which would put the minimum wall-clearing grip height at **90.8 mm** — against a block whose top is at **90.0 mm** when seated in the slot. That bound is over-conservative on two counts (see next row). |
| 2026-08-02 | A1 | Two harness bugs found and fixed, both of which produced confident wrong numbers first. (i) `ComputeLocalBound` on a body prim returns the **whole subtree** bound, so `link5`'s box swallowed the downstream arm (254 mm across) and the posed result claimed the hand reached 60 mm *below the table*. Fixed by attributing each geometry prim to its nearest rigid-body ancestor via `ComputeRelativeBound`. (ii) The asset is authored with **payloads and instancing**, so a plain `Usd.PrimRange` found **zero** geometry — the traversal needs `Usd.TraverseInstanceProxies`. Without the guard added in the fix, that silently produced an empty envelope rather than an error. |
| 2026-08-02 | A2 | **The hand DOES fit — the render-mesh bound was wrong.** Physics collision test (stroke the empty closed gripper through the slot, measure TCP tracking; a jammed hand cannot track its command): grip z = 0.060 **FOULING** (8.3 mm err), 0.066 **FOULING** (11.7 mm), 0.072 clear (2.6 mm), 0.078 **158 mm err** (see below), 0.084 **clear (0.87 mm)**, 0.090 **clear (0.95 mm)**, 0.096 **clear (0.85 mm)**. So the usable grip band is **z ≈ 0.084–0.096**, against a block top at 0.090 when seated — gripping at 0.084 puts the TCP 29 mm above the block centre and 6 mm below its top, which is a real grasp on the upper half. The footprint-restricted render-mesh column had claimed −4 to −2 mm clearance there; it is an over-estimate and **must not be used as the bound**. Physics is the arbiter. |
| 2026-08-02 | A2 | **CEM is not reliable enough to be the expert's solver.** The z = 0.078 cell returned 158 mm tracking error with dz = +86 mm — not a collision, a different IK branch (its `pre` solve also had the worst alignment error of the sweep, 0.0258 vs < 0.003 elsewhere). A solver that silently jumps branch between two adjacent waypoints cannot generate demos. Consequence: the expert needs **seeded per-env DLS IK with a convergence check and branch continuity**, with CEM used only to find the nominal seed. |
| 2026-08-02 | A2/A3 | `slot/analysis/insertion_feasibility.py` written and launched — the redone V4 rung. Two corrections to the 90.8 mm bound: it measures clearance **restricted to the wall footprint** (only geometry standing inside `x ∈ [0.210,0.280]`, `|y| ∈ [half, half+0.008]` can hit a wall; the fingers press at `|y| = 0.015`, inboard of it), and it reports whether the geometry came from `CollisionAPI` or from render meshes (an over-estimate — the first run found no `CollisionAPI` prims). Then it runs the stroke **in physics**, starting the block at 6 mm depth so the measured stroke is a real 34 mm, not the withdrawn probe's ~1 mm. Retention is checked by **finger gap** (C3 calibration), which an untouched block cannot fake. |
| 2026-08-02 | S2 | **RETRACTION of the two A2 rows above.** The render-mesh bound was *right*; the physics test that appeared to refute it stroked a **closed** hand (`q_fing=0.0`), which slides down the middle of a 33 mm channel untouched — a different geometry from a hand *holding* a 30 mm block (pads at \|y\|=16 mm vs wall inner faces at 16.5 mm). Read the holding column and geometry and physics agree monotonically: clearance −3.5 mm at z=0.084 (physics: pads jam, gap 35.4 mm, block flung 130 mm), **+0.4 mm at 0.090** (holds, gap 30.0), +6.1 mm at 0.096 (holds). **Standing constraint: a loaded gripper clears the walls only at TCP z ≥ 0.090.** Separately, the CEM anomaly **moved from z=0.078 to z=0.090** between two runs of the same script — the cell was noise; the conclusion (CEM unfit for demos) stands and is strengthened. |
| 2026-08-02 | S2 | **`mdp.is_inserted` cannot be used bare for evaluation.** It bounds block z only from *below* (`z > 0.015`), so a block resting on the wall tops or dangling in a closed gripper passes. Measured: a probe cell scored 93.8 % with 13.28 mm mean lateral error (impossible inside a 16.5 mm half-width channel); another scored 100 % with a 33.86 mm finger gap and the block at z=67 mm. All numbers below use `is_inserted & (\|block_z − 0.055\| < 0.006)`, judged **after opening the fingers**. Env not modified. |
| 2026-08-02 | S2 | **The 8–12 % stroke rate was the harness, not the task.** Stage 2b interpolated in **joint space**, which bows the TCP path and dragged the block onto the wall tops (block z rose 55 → 69 mm). Dense **Cartesian** waypoints, same geometry and grip: **100 %**. Neither pre-registered hypothesis (slip / jam) was correct. |
| 2026-08-02 | S2 | `slot/expert/ik.py` — batched DLS IK replacing cuRobo. Solves a **5-DOF task** (TCP position + finger-axis direction); pinning full orientation over-constrains the 6R arm (3.6 mm residual at nominal, 15.6 mm along a path, 0/64 converged). Redundant roll steered by a nullspace bias with a **decaying** gain — with Levenberg damping `(I − J⁺J)` leaks into task space, which left 23 % of a straight push's waypoints above 0.5 mm. After the fix **every waypoint of every phase converges** (max 0.4 mm). |
| 2026-08-02 | S2 | `strategy_probe.py`: horizontal Cartesian drag **100 %** vs vertical lower-in **46.9 %**. The vertical failure is depth margin (mean 39.9 mm vs a 40 mm threshold), not precision — success was flat across every \|dy\| and \|dyaw\| bin. Overturns the env write-up's premise that the block must be dropped in from above. |
| 2026-08-02 | S2 | **Turn-phase grip loss: three refuted hypotheses.** Gap grew 29.95 → 36.4 mm mid-traverse, losing ~1/3 of grips. Refuted: pendulum swing (smootherstep easing, 64.1→65.6 %), wrist tracking lag (axis err ≤ 4e-4 throughout), speed (2.3× slower moved gap only to 33.3 mm and onset stayed at the same *position*), joint-space jerk (polar re-parametrisation made peak joint1 step *worse*, 0.0732→0.0881 rad). Decisive test — identical traverse 45 mm higher: gap **29.95 mm flat, 32/32 held**. **It was a collision.** The carried block rides **~5 mm ahead of the TCP in x**, so its nose reached ~205 mm against wall faces at 210 mm, closing to ~3 mm once yawed. `stage_x` 0.180 → **0.165** fixes it for free. |
| 2026-08-02 | **B** | **EXPERT DONE — 100 % seated on all three clearances at n=128.** Loose-v0 128/128 (lateral p90 1.38 mm), **v0 128/128** (p90 1.11 mm), Tight-v0 128/128 (p90 0.48 mm). Zero failures in every category, grip held through every phase at a constant 29.95 mm gap, block z exactly 55.0 mm, 510/600 steps, zero resets. Lateral error *falls* as the slot narrows (0.67/0.55/0.30 mm) — the walls do the fine alignment, a good consistency check on 100 % at 0.5 mm. **Clears the Stage B bar (≥85 % Loose, ≥75 % v0) with margin.** Video: `slot/logs/expert/expert_Rebot-PrecisionSlot-v0.mp4`. |
| 2026-08-02 | S2 | Two more forced parameters, each worth recording: `carry_z` 0.090→**0.095** (at 0.090 the carried block's bottom sat 0.9 mm above the 20 mm slot-floor step; **53.1 % → 79.7 %**), and the **600-step episode budget is hard** — an expert running 659 steps timed out every env and read as 0 % success. Also: CEM's population size *was* `num_envs`, so n=4 scored 50 % where n=128 scored 100 %; the seed is now solved once at large n and cached to `slot/logs/expert/seed_q.json`, which also makes the expert deterministic for demo collection. |
| 2026-08-02 | next | **Main risk to Stage C:** the expert plans everything before touching the block and runs **open-loop**, so its demos contain **no corrective behaviour**. eva_bc's postmortem blames exactly this for part of its BC plateau. Plan grasp-perturbation injection and/or HG-DAgger *before* collecting, using the tolerance surface from `strategy_probe.py` to set the perturbation scale. |

---

## Session 3 log (2026-08-02/03)

| # | what | verdict |
|---|---|---|
| 3.1 | **Extracted `slot/expert/plan.py`** from `run_expert.py` once demo collection became a second consumer. Execution schedule exposed as a generator of per-env-step commands so both consumers run *identical* steps. | Regression-checked: `Rebot-PrecisionSlot-v0` n=128 still **128/128 seated**. Refactor is behaviour-preserving. |
| 3.2 | **Added a `retreat` phase** — back the open gripper out of the slot after release, so the episode ends with the arm clear and the demo has a stable terminal behaviour. | Block pose identical before and after: (256.8, −0.5, 55.0) mm → (256.8, −0.5, 55.0) mm, \|yaw\| 0.0046 both. **Zero disturbance.** 558/600 steps. |
| 3.3 | **`slot/scripts/collect_demos.py`** — batched collection (one demo per env), eva_bc HDF5 schema, phase-labelled segments, per-env outcomes, `train_mask`. | Works. T=599 uniform, no auto-reset straddles the recording. |
| 3.4 | **`slot/scripts/verify_demos.py`** — the check that matters is the alignment identity `obs[t,27:34] == actions[t-1]`, exploiting `last_action` in the observation tail. Recording the obs *after* stepping is the natural mistake and would train the policy to predict an action it can already see; shape checks cannot see it. | **Max abs difference exactly 0.0.** Ten checks pass including arm-tracks-target (0.0009 rad at rest) and contiguous mask censoring. |
| 3.5 | **EXP S2-N: whole-trajectory DART noise.** Pre-registered. | 3 of 4 beliefs **wrong**. Success 100 → 88.3 → 57.0 → 28.9 → 7.8 % at `noise_std` 0/0.01/0.02/0.04/0.08. Failure is *contact*, not precision: grip 128/128 in every free-space phase at every level ≤0.04, all loss in `push` (128→79). |
| 3.6 | **Root cause, and it is principled.** DART is valid here because the action is a joint *position target* — correct from any nearby state. That holds **only in free space**. Inside a 1.5 mm channel, commanding the nominal target from a perturbed state drives the block *into* a wall. The label is not noisy, it is **wrong**. | `push` peak deviation 74 → 530 mrad while free-space phases stay flat (lift 33 → 60). No `noise_std` fixes this. |
| 3.7 | **EXP S2-N2: free-space-only noise.** Pre-registered before seeing the rest of S2-N. | At the same 0.02: **57.0 % → 96.9 %**, resets **44 → 0**, grip at `push` **96 → 128/128**, lateral p90 **147.87 → 1.07 mm**, `push` deviation back to its 74 mrad baseline exactly. |
| 3.8 | **A third failure mechanism at high noise.** Not grasp misses (belief 2 refuted again): 30 mid-episode terminations at 0.10 and 101 at 0.20 are the **carried block hitting the fixture**. `stage_x`=0.165 exists because the carried block rides ~5 mm ahead of the TCP against wall faces at x=0.210 — so "free space" is only free within ~40 mm, and that margin caps the usable noise. | **`noise_std = 0.05`** chosen by the pre-registered rule (≥90 % seated, ≤2 % resets → 92.2 %, 2). |
| 3.9 | **Two self-contradicting numbers caught in my own reporting.** (a) Grip stats counted envs that had already reset — Isaac Lab hands them a *fresh scene*, so the finger gap reads healthy; `grip held 126/128` sat next to `lateral p90 131 mm`. (b) The headline success rate was conditional on not resetting: 98.0 % printed where the true score was 75.0 %. | Both fixed: phase stats masked `& ~resets`; both rates printed with the unconditional one labelled the real score. Rows ≤0.05 unaffected (0–2 resets). |
| 3.10 | **`act/` → `slot/slot_act/`, package RENAMED.** Keeping the name `act` is a trap: both dirs are importable as `act` and `sys.path` order decides. Measured — `cd /tmp && PYTHONPATH=.../eva_bc python -c "import act.dataset as D; print(D.OBS_DIM)"` prints **41**, silently. A guard *inside* the copy cannot help (in the failing case the copy is never imported); my first attempt at this was wrong for exactly that reason. | Renamed, 35 import sites rewritten. Ambiguity removed rather than detected. |
| 3.11 | **Port constants.** `OBS_DIM 34`, `ENV_STATE_DIM 18`, `BIT_DIMS[-1] 33`, `RES_OBS_DIM` **58**, `STEER_OBS_DIM` **50**. | 58/50, **not** the 57/49 PORT_MAP predicted — the goal-delta tail is 4-D not 3-D, because the env's own `lateral_error`/`yaw_error` are **absolute values**: they say how wrong the policy is, not which way to correct. |
| 3.12 | **The edit a rename cannot make.** `residual_core.py` had `obs41[:, 40]` — a bare integer for the commanded-grip channel. Invisible to any `obs41→obs34` substitution, out-of-bounds at 34. | Now `obs34[:, BIT_DIMS[-1]]`, which cannot drift from `OBS_DIM`. That channel is required: finger joints alone score a better AUC (0.976 vs 0.968) but **27.1 % FPR** vs 0 %. |
| 3.13 | **Flush rule z-drop half removed** in both `residual_core.py` and `eval_act.py`. It was a slip proxy sized against a 40 mm basket rim; here the block legitimately descends ~8 mm the instant the fingers open, so the pick-place rule would flush the committed chunk at the most precision-critical moment. Chunk commitment is load-bearing (59.4 → 32.8 → 3.1 → 0 % when shortened). | Only `FLUSH_POS_JUMP = 0.03` kept. |
| 3.14 | **`slot/slot_mdp.py`** — the four-call surface `act/` needs, so `eva_rl` stays untouched. `placed_mask` wraps the **seated** predicate, not `is_inserted`; `basket_centers_local` deliberately absent so an unported call site raises rather than silently getting a constant. | `eval_act.py` reduces `placed_mask().all(dim=1)` into the headline number, so this one line decides every success figure for the rest of the project. |
| 3.15 | **`slot/scripts/check_port.py`** — static consistency check. The obs width is a constant in three files related only by arithmetic. Asserts the dim identities, that `slot_act` did not resolve to the tracked original, and that no pick-place task id survives. | **Caught a stale `Rebot-PickPlace-Play-v1` on its first run.** Now exits 0. |
| 3.16 | **BC chain integration test.** collector → HDF5 → `slot_act.dataset` → flow policy → checkpoint. | Checkpoint carries `env_state_dim: 18` and satisfies `eval_act.load_checkpoint`'s asserts. Pool filters port with **zero logic change** — DART episodes tag `episode_kind="dart"` and route to `recovery_pool_filter` automatically. |
| 3.17 | **Stage D deliverable (HANDOFF §9).** This task's action-noise tolerance, measured not inherited. | 0.02→96.9 %, 0.05→92.2 %, 0.10→75.0 %, 0.20→15.6 %. Pick-place's *healthy* σ≈0.08 sits between our 75 % and 15.6 % cells. `sigma_init` should start **≤0.05** (≈ −3.0), not the inherited −2.5. |
| 3.18 | **RETRACTION of 3.11's cause.** I diagnosed the rollout effect as planner state-dependence from a "17 mrad monotonic drift" in the spawn-independent push-end command. **Not significant**: within-rollout spread is 0.27 action units, so SE = 0.024 and the r3−r0 difference is **1.12 σ, Welch p = 0.266, ANOVA p = 0.645**. I compared four increasing means without dividing by √n, having printed the spread in the same table. | `plan_determinism.py` refuted it directly: `fk()` is a pure function of q (0.000 µm), and the plan is **bit-identical** (0.0000 mrad, all 6 phases) both back-to-back and after a full 599-step episode. |
| 3.19 | **THE simulation is not reproducible.** Same plan, bit-identical initial state (`|q_start − q_default| = 0.00e+00`), three executions in one process: seated 29/30/24 of 32; max \|depth diff\| **15.9 mm**; outcome flips 5/9/8 of 32. | Expected of GPU PhysX (order-dependent parallel contact solving) but large here because the task sits on a 40 mm depth threshold. **The expert's "100 %" was one sample from a wide distribution; its honest rate is 90.2 % (nominal), 86.7 % (DART-0.02), 82.4 % (DART-0.05).** |
| 3.20 | **Consequence: the action labels are still clean.** The plan is a pure function of the block pose, so the stochasticity is entirely in the *outcome*, not the observation→action map. Success-filtering the nominal pool therefore selects on **luck, not behaviour** — no spawn bin predicts failure, so it drops ~10 % of demos at random. | Strengthens the case for the in-hand-slip censor (3.21), which finds the frames where the block actually went wrong regardless of final outcome. |
| 3.21 | **Loss censoring was inert, and Big Will caught it.** Every phase reported `held` on all 512 demos while 50 ended unseated — the ported detector is grip-based and the grip *never* fails here (gap 29.96 mm from grasp to release even in failures). `train_mask == 0` on **0 demos**. | Replaced with an **in-hand offset** detector (`block_pos − TCP` vs its post-grasp value): catches sliding *and* jamming with one number, and is **noise-agnostic** so it does not delete DART's corrective labels. `verify_demos.py` now reports whether the censor separates seated from failed demos, so an all-ones mask can no longer read as "clean data". |
| 3.22 | **Session writeup** — `docs/slot/SESSION3_WRITEUP.md`, covering all of the above with evidence, retractions and the failure analysis. | Done. |
| 3.23 | **The slip censor was wrong twice.** v1 measured past the release (nothing in the hand) and read the gripper's 90 mm retreat as a 92 mm slip -> censored 128/128 including successes. v2 fixed the span but kept a guessed 3.0 mm threshold; measuring the profile showed nominal drift is already 2.2/1.8/1.8/3.4/4.7 mm at the ends of lift/back/spin/turn/push **in demos that all succeed**. The threshold sat below normal behaviour. | Both caught by a guard added on principle: *a censor firing on >50 % of demos, successes included, is a broken detector, not dirty data*. Same shape as `verify_demos.py`'s all-ones-mask check — a censor's output distribution is evidence about the censor. |
| 3.24 | **Resolution: the threshold is no longer guessed.** Censor defaults **off**; the raw per-step signal ships as a `slip_mm` dataset per demo. `scripts/calibrate_slip.py` derives it offline from pools containing real failures, using **excess** slip (`slip(t) − median over successful demos`) because nominal slip grows monotonically, so a fixed raw threshold is really a threshold on *time*. Threshold set at low FPR on successful demos (eva_bc's grasp-bit discipline). | The script can return *"does not separate outcomes, do not censor"* — a real answer, not a failure — and refuses to calibrate against a pool with no failures. Masks are re-derivable without re-running the simulator. |
| 3.25 | **One-rollout-per-process confirmed.** First v2 pool: **128/128 = 100 %** seated. | Confirms 3.19's operational fix. Pools re-collected as 12 single-rollout processes. |
| 3.26 | **Slip censor calibrated on the full set (1526 demos, 60 real failures).** Raw whole-span AUC **0.355** — strongly INVERTED, confirming the back-stop mechanism: the expert drives the block home against the stop, the gripper keeps advancing while the block cannot move, so high slip is the signature of a *fully seated* insert. Windowing to CARRY only (grasp→push) flips the sign to **excess AUC 0.697**, above the 0.65 gate registered before looking (the 3-pool subset gave 0.637 and the gate correctly withheld). | Threshold at 2 % FPR = `excess > 6.62 mm`; catches 15 % of failures, fires at median t=191. |
| 3.27 | **Honest magnitude of the censor.** `default_demo_filter` drops failed demos from training entirely, so censoring them does nothing. The censor's **only** real action is on the ~2 % of *successful* demos with a genuine mid-carry slip — about **1.3 % of training frames**. Correct on principle (those frames carry stale labels after the block diverged from plan) but not a meaningful win, and it cannot confound the A/B because both arms get identically censored data. | Applied once over the complete 16-pool set rather than twice over partial sets. |
| 3.28 | **The DART alignment check was a false alarm that became a better test.** `obs[t,27:34]` is `last_action` = the **executed** action; DART executes `nominal + noise` and labels with `nominal`, so the strict identity cannot hold and the 9.4e-02 residual **is** the noise. Checking that residual against the OU process it should be verifies alignment *and* noise injection together. | **[PASS] noise OFF from push onward: max \|eps\| = 9.6e-06, 0.05 % of noise_std** — direct independent confirmation that the phase restriction (the 57.0 %→96.9 % fix) actually held. Also: residual std 0.0150 vs declared 0.0200 (settle decay pulls it down), lag-1 autocorrelation **0.940** vs requested rho 0.95. Had I "fixed" the non-bug by labelling with the executed action, DART's entire corrective property would have been destroyed. |
| 3.29 | **v2 pools: all 12 pass verification.** nominal s0–s3 **512/512 = 100.0 %** (four independent seeds), dart002 96.1 %, dart005 90.4 %. | Settles the 100 %-vs-90.2 % confusion: the expert *is* 100 % when every episode is a first episode; 90.2 % was the first-episode bias contaminating a multi-rollout process. Both were real measurements of different things. DART rates match the sweep predictions across seeds (96.1 vs 96.9, 90.4 vs 92.2), so the sweep generalises rather than having been fitted to one seed. |

---

## Session 4 — freeze the data, validate the instruments, run Stage C

Detail and evidence in `docs/slot/SESSION4_WRITEUP.md`. Rows are verdicts, not intentions.

| # | what | verdict |
|---|---|---|
| 4.1 | **The queued collection chain had deadlocked on itself.** `while ps aux \| grep -q "[c]ollect_demos"` — the `[c]` trick stops *grep* matching itself but not `ps` matching the **waiter's own command line**, which contains the string four times. `nominal_s4..s7` never ran. | Invisible from the log, whose last line was a cleanly-completed pool. Caught by comparing wall-clock against the newest file's mtime. Killed and re-run directly. Gotcha added. |
| 4.2 | **Data frozen: 16 pools, 2038 demos, 1977 successful (97.0 %), T = 599 each.** All 16 pass `verify_demos.py`. | Nominal is **1023/1024 = 99.90 %**, not 100 % — `nominal_s5` lost one. The earlier "512/512 over four seeds" was true of those seeds and was never a claim that the expert cannot fail. |
| 4.3 | **`--episode-length-s` defaulted to 30 s, 2.5× this task's horizon.** Inherited from pick-place, whose demos ran ~1234 steps. Here control dt = decimation 8 / sim.dt 1/400 = **20 ms**, `episode_length_s` = 12.0, and every demo is exactly T = 599. | Would have rolled the policy **900 steps past** anything in its training distribution with `last_action` feeding back the whole way, at 2.5× the GPU cost. Default now 12.0. |
| 4.4 | **`--pool` defaulted to no filter, and the obvious alternative was worse.** `default` trains on the ~5 % of DART episodes that ended unseated; `nominal` requires `episode_kind == "nominal"` and would have **dropped arm B's entire DART half**, turning a 1024-demo arm into a 512-demo one — the exact volume confound the experiment exists to rule out. | Added `success_pool_filter` / `--pool success`; both arms use it. Verified: 255 demos → success 244, nominal 128, recovery 116, and the partition holds. |
| 4.5 | **`scripts/test_pipeline_cpu.py` — 17/17 on CPU in 40 s.** Validates everything in the training path except the sim rollout, before spending 2 arms × 3 seeds × 100 k steps. | The censor *does* reach the loss: garbage on censored steps leaves the loss **bit-identical** (6468.139160 both ways) while garbage on trainable steps moves it to 540827. Chunk commitment intact: exactly 3 forwards per 45 control steps, an absurd mid-window observation changes nothing, `reset` refills exactly the reset envs. Checkpoint round-trips to `0.00e+00`. |
| 4.6 | **Instrument test found a fourth wrong number.** `depth_mm` / `lateral_mm` / `yaw_rad` were read **inside `if done:`, after `env.step`** — and Isaac Lab resets done envs *inside* `step()`. They described the **next** episode's spawn. Reported `\|lateral\| median 126.79 mm` on episodes whose block was seated at y = 0.0001 ± 0.0008 m. | The headline was never wrong (success is sampled pre-step) — only the fields you would use to *explain* it. Arguably worse: a well-formed, entirely fictitious failure analysis. Fixed by hoisting the reads; median went 126.79 → **0.92 mm**. |
| 4.7 | **An eval at a fixed seed is bit-reproducible across processes** — 0.438/0.562/0.375 twice over. | Does **not** contradict 3.19: that was repeated episodes *inside* one process, where PhysX caches survive `env.reset()`. Consequence: repeating an eval yields **no** information and an error bar built from repeats is exactly zero. Vary `--seed`. The sweep uses two spawn seeds (777, 888). |
| 4.8 | **The first-episode bias reproduces on a learned policy:** +18.7 points (0.562 vs 0.375), alongside the +12.9 measured on the expert. | Rules out anything specific to the scripted plan. `success_rate_later` stands as the comparison statistic. |
| 4.9 | **SUPERSEDES 3.27 — the slip censor is calibrated but NOT applied.** Full-set calibration confirms the signal (CARRY-only excess AUC **0.693** on 2038 demos, vs 0.697 on 12 pools; raw AUC still < 0.5 in every window, so the back-stop inversion survives at full n). But both arms train with `--pool success`, so the **14.8 % of failed demos it catches are already excluded**, while the **2.0 % of successful demos it censors are not** — by construction, since the threshold *is* the 98th percentile of the success distribution. | On a success-only pool it is a pure subtraction of ≈16 000 frames from episodes that seated the block. Not applied; pools stay exactly as collected. The alternative (train on everything, let the censor truncate failures) needs **sensitivity**, and 14.8 % means 85 % of failed demos would train end-to-end with their bad endings — strictly worse than dropping them. |
| 4.10 | **Label-noise floor measured at 0.1 mrad** (`analysis/label_consistency.py`): among the 5 % most-similar cross-demo observation pairs, median action disagreement is 0.0002 units = **0.0 % of the action std**. | The feared 133 mrad null-space multimodality does **not** appear as label noise — the expert warm-starts each IK solve, so the branch is consistent across demos at the same spawn. The target function is clean, which is the quantitative explanation for 4.12. |
| 4.11 | **DART's value proposition, quantified.** Median cross-demo nearest-neighbour distance: nominal **0.232** std-units, DART σ=0.05 **0.625** — **2.7× wider coverage at an identical label-noise floor**. | Sharpens the arm comparison: if arm B loses it will *not* be because its labels are noisier. It would have to be that the extra coverage lies where the policy never goes. |
| 4.12 | **A throwaway 2000-step checkpoint (40 s, 256 demos) scores 43.8 % on `-v0`** (37.5 % on the unbiased `later` cohort). | The pre-registered Stage C bar (≥ 40 % v0) is close to uninformative; the question becomes how far *above* it the arms land, and whether 70 % is reachable without Stage D. Recorded before the real runs finished so it cannot be retrofitted. |
| 4.13 | **Failure taxonomy from that same checkpoint.** Of 27 failures: **13 ended at the slot mouth, at correct seated height, short of the 40 mm depth**; 6 short of the mouth; 8 never advanced (z 0.012–0.043 → dropped or never lifted); **0** deep-but-unseated. | The dominant failure is **depth**, not grasping and not alignment. If it survives into the trained models it is the obvious Stage D target. |
| 4.14 | **The open-loop-clock risk, and the test for it.** Every demo shares one phase schedule, so cross-demo nearest neighbours sit a median of 1–2 timesteps apart, and `last_action` occupies obs 27:34. A policy can integrate its own output and replay a time-indexed trajectory without ever re-reading the block. On this data that scores well. | `scripts/diag_feedback.py` re-randomises the block **mid-`reach`, before the fingers close**, drawing from the env's own reset range. A clock closes on nothing; feedback re-targets. Two controls: no-teleport, and a same-pose write through the identical code path (isolates the cost of the write). Not yet run. |
| 4.15 | **`Rebot-PrecisionSlot-Play-v0` is not a distinct evaluation.** It differs from `-v0` only in `scene.num_envs`, `env_spacing` (both overridden by `parse_env_cfg`) and `enable_corruption`, which the base `PolicyCfg` already sets `False`. There is no observation noise anywhere in this env. | The real ladder is **Loose 3.0 mm → v0 1.5 mm → Tight 0.5 mm**. The eval sweep covers all three; Play is skipped as a duplicate rather than burned on GPU. |
| 4.16 | **Stage C clears its own bar at 10 000 steps.** Arm A seed 0, `-v0`, `success_rate_later`: **0.615** (n=96) at 10 k, against a pre-registered bar of 0.40. The 2000-step throwaway gave 0.375. | The bar was inherited from pick-place and is not calibrated to this task. 10 k steps is **3.5 minutes** of training at the measured 47.4 steps/s. Makes it plausible that Stage C alone reaches the 70 % goal on `-v0`, with Stage D becoming a push-further rather than a rescue. |
| 4.17 | **The first-episode bias is not a constant.** +18.7 pts on the 2000-step model (0.562 vs 0.375), **+1.0 pt** at 10 k (0.625 vs 0.615). | Reads as the bias acting on *marginal* episodes: a warm-start advantage can only flip an outcome that was already close, and a weak policy has many such episodes while a strong one has few. **Does not license dropping the cohort split** — it is a property of policy × task, and `-Tight-v0` will produce many marginal episodes for any policy. |
| 4.18 | **I broke the "ONE GPU job at a time" rule and it cost real time.** Memory was never the constraint (910 MB training + 1.5 GB eval against 11 GB); *compute* was — the GPU sits at 96–99 % during training. A concurrent eval went from **2 m 11 s to over 10 minutes** while dragging training from 47.7 to 42 steps/s. | Total GPU work is conserved at best and slightly worse in practice (context switching, memory bandwidth). The only thing overlapping buys is *earlier* information, which is worth very little when the user is AFK. Run sequentially. |
| 4.19 | **Training loss is a poor convergence signal here.** It plateaus around step 20 k (0.053 → 0.050 → 0.047 → 0.040 across 20k–50k) with per-batch std 0.011–0.018, comparable to the change. | Flow-matching loss is noisy by construction — τ is drawn per sample. Success rate kept climbing well past the loss plateau (0.375 at 2 k → 0.615 at 10 k). Only eval success answers "was 100 k needed?"; do not shorten a run mid-sweep on loss evidence, as that breaks the matched protocol. |
| 4.20 | **`spawn_pos` now recorded per eval episode, so pairing is verified rather than assumed.** Two eval runs at the same seed *should* face identical spawns — which would make the arm comparison paired and admit McNemar — but the flow policy calls `torch.randn` for x0 on the same global generator the reset events draw from, so it only holds while every env refills in lockstep and none flushes. | `summarize_arms.py` checks spawn identity per episode slot and falls back to the unpaired analysis, saying so, if they differ. One cleanup owed: the early one-off arm-A-seed-0 eval predates the field and should be deleted so the sweep regenerates it. |

---

## Session 5 — upstream review, and Stage C clears the objective

Detail and evidence in `docs/slot/SESSION5_FINDINGS.md`. Rows are verdicts, not intentions.

| # | what | verdict |
|---|---|---|
| 5.1 | **Both repos pulled and reviewed.** `eva_rl` 05f0fb3 → e56e7df (1 commit: wrist-camera tilt); `eva_bc` 1c04eca → 818391b (4 commits: EXP07 closure). Both fast-forwarded; no tracked file modified; `slot/` + `docs/slot/` untouched. | The `eva_rl` change is a `CameraCfg.OffsetCfg` in the `lift/` tree — a sensor pose, no mass or collider, not imported by the slot task, and this task's obs is 34-D state with no camera in the loop. **Demos and checkpoints remain valid.** Checked *before* pulling, because a robot-asset change would have invalidated 2038 demos. |
| 5.2 | **THE OBJECTIVE IS MET on `-v0` by BC alone: 92.7 %** (89/96 later-cohort, 95 % CI [0.856, 0.964]), arm A seed 0 `ckpt_final`, spawn seed 777. 89.6 % already at 30 k steps. | No RL, no DAgger, no HG-DAgger — 22.7 points above the 70 % target with the entire confidence interval above it. Pending confirmation across the other two clearances and spawn seed 888 (the sweep). Audited 14/14 by `check_eval_json.py`, including *every episode exactly 600 steps*, which is what rules out an early termination handing the policy a second attempt. |
| 5.3 | **The learning curve plateaus at 30 k steps** — established by pairing, not by eyeballing overlapping CIs. 30k/50k/100k are mutually indistinguishable (McNemar χ² ≤ 2.12, 1 df); 30k and 100k agree on **81 of 96** episodes. | Next sweep can use **30 k steps per run**: ~12 min instead of ~35, turning a 6-run sweep from ~3.5 h into ~1.2 h. This sweep needs no change — `--save-every 10000` already wrote `ckpt_0030000` everywhere. New tool: `analysis/paired_evals.py`. |
| 5.4 | **My own ad-hoc pairing check was wrong, and the tool caught it.** It reported "spawns identical" for all six pairs; in fact `eval_ckpt_0010000` predates the `spawn_pos` field and my throwaway code read a *missing* field as agreement. | `paired_evals.py` separates **matching** from **unverifiable** and flags the three 10k rows as not-paired. Conclusions survive by different routes: 10k → 30k is real because the Wilson intervals are *disjoint* ([0.515,0.706] vs [0.819,0.942]); the "nothing after 30k" claim uses only verifiably-paired rows. A hand-rolled check written to confirm something tends to confirm it. |
| 5.5 | **The first-episode bias went NEGATIVE at 100 k** (−2.1 pts: 0.906 first vs 0.927 later), having been +18.7 at 2 k, +1.0 at 10 k, +10.4 at 30 k. | Not a constant, not monotone, **not always positive**. Supersedes any temptation from 4.17 to treat it as a small positive nuisance. A pooled `success_rate` would have been wrong by between −2 and +19 points depending only on which checkpoint you looked at. Keep reporting the split. |
| 5.6 | **All 61 expert failures share ONE signature: `release=unseated`.** Every one grasped, held through lift/back/spin/turn/push, and failed at release. **Zero** grasp failures, drops, topples or carry losses in 2038 demos. | The entire difficulty is in the last ~135 control steps (`push` t≈358, `release` t≈465). Directly kills the ported grasp-bit feature (5.7) and predicts that EXP07's headline mechanism will not transfer (5.9). |
| 5.7 | **`experiments/exp06_grasp_bit.pt` does not exist in either repo** — only the training script was ever committed. Six `slot_act/` call sites pointed at it, so `train_steer.py` would have died at line 70 **after** paying Isaac Sim's boot cost. | Blocker closed. Not fixable by retraining: the finger channels are **degenerate** on this robot — over 707 802 provably-held frames the summed finger position is −0.04889 ± 0.00003 against −0.04887 ± 0.00005 for closed-but-not-lifted, a 2e-5 difference against a 3e-5 spread. The fingers saturate identically with or without a 30 mm block between them. And per 5.6 there is no negative class to train on. Replaced by analytic `SlotGraspBit` = the env's own `block_lifted` **AND** commanded-closed. `check_port.py` 16/16. |
| 5.8 | **Steering path CPU-validated 17/17** (`scripts/test_steer_cpu.py`, ~20 s, no simulator). The gate that matters: `steer_x0 = zeros` is **action-for-action identical** to `fixed_x0 = zeros`, max abs diff **0.000e+00**, including across a deliberately desynced mid-window flush. | That is EXP07's gate 1 — the property making any steering result attributable to steering rather than a wrapper artefact — established in seconds instead of 35 min of GPU. Also covers per-env `z` routing (the silent failure that trains against the wrong env), 2-D vs 3-D `x0` equivalence, the `α·tanh` bound, and `SlotGraspBit` semantics. |
| 5.9 | **EXP07's headline mechanism will NOT transfer.** Its single largest win was the never-lifted bucket collapsing 18/19 — episodes that committed to a bad grasp and never retried. Per 5.6 that failure mode **does not exist here**. | Expect a materially smaller effect from x0-steering on slot, and say so before running rather than after. What *should* transfer is re-choosing the chunk family at the approach/push windows, which is where 5.10 puts the failures. |
| 5.10 | **BC's failures are a jam at the slot mouth.** Of 10 failures at `ckpt_0030000`: yaw is never binding (all \|yaw\| ≤ 0.030 rad against a 0.12 tolerance); five sit at 1.42–1.90 mm lateral with strongly negative depth (never entered); one is aligned to 0.04 mm and still only reached 5.4 mm depth (a *push* failure, not an aiming one); two passed the env's bare `is_inserted` and were correctly rejected by the seated-height guard. | Successes have max \|lateral\| = **1.50 mm** against a **1.5 mm** clearance, and the distribution is near-uniform on [0, 1.5] rather than piled up near zero. The clearance is doing the aligning, not the policy. |
| 5.11 | **⚠ RETRACTION IN PLACE — my clearance-ladder prediction is confounded.** I pre-registered Tight ≈ 0.19 from the \|lateral\| CDF. But that quantity is measured **after the walls constrain the block**: it is censored, which is why its p100 is exactly the clearance. | Proof it is censored: the expert runs the *same open-loop trajectory* on all three rungs, yet its lateral p90 falls 1.38 → 1.11 → **0.48 mm** as the channel narrows. One trajectory, three "errors" — the number reports the channel, not the policy. The prediction stays on the record and the sweep still tests it, but if Tight lands above 0.19 the conclusion is "the censored-measurement model was wrong", **not** "the policy is closed-loop". |
| 5.12 | **The expert already solves Tight: 128/128.** Meanwhile **all 2038 demos were collected on `-v0`** (verified from the `task` attr, not assumed). | So if BC underperforms on Tight the first hypothesis is **train/test clearance mismatch — a data problem**, and the fix is one collection pass with `--task Rebot-PrecisionSlot-Tight-v0` plus one training run on existing tooling. **Re-orders Stage D: measure → collect Tight demos → only then x0-steering.** Reaching for RL with a 100 %-capable expert and an idle collection harness sitting there would repeat the error-class misdiagnosis that cost EXP06 two runs. |
| 5.13 | **Zero flushes in 128 episodes** (`flush_enabled: True`). The §4.2 discontinuity flush is armed and never fires on this task. | Confirms the 30 mm position-jump threshold is correctly sized, and matters for Stage D: in EXP07 the flush was the *only* source of window desync (a flushed env applies a `z` up to 14 steps stale). Here it is not rare, it is **absent** — 600 steps = 40 windows exactly, no mid-episode terminations, no flushes, so every refill lands on a window boundary. |
| 5.14 | **`run_eval_sweep.sh` had a `set -euo pipefail` hazard.** A failed eval makes the `\| grep "eval_act"` match nothing, grep exits 1, pipefail propagates and `set -e` kills the **whole** sweep. | One bad cell would have silently discarded the 30+ evals after it during a 2.5 h unattended run. Fixed with an explicit `\|\| echo "!!! EVAL FAILED (continuing)"` marker — greppable, so a failure can never be mistaken for a skip. The sweep is idempotent, so continuing and re-running beats aborting. |
| 5.15 | **The "collect Tight demos" plan is mostly a no-op, and checking `expert/plan.py` is what showed it.** `ExpertParams` (`grasp_h`, `carry_z`, `stage_x`, `insert_x`, `turn_per_wp`) contains **no clearance term** — the string `clearance` does not appear in the planner, and its docstring says the defaults "measured 100 % on all three clearances". | The expert commands the **same trajectory** on Tight as on `-v0`; only the resulting observations differ, and only slightly (it seats 128/128 with minimal wall contact). Tight demos would supply near-identical action labels to the 2038 already collected. **Supersedes 5.12's recommendation**: demoted from "the plan" to a late fallback. Reframed hypothesis in `EXP_TIGHT.md`: what fails on Tight is the **policy's own x0 sampling noise** eating a clearance with no room for it — testable for free via `--fixed-x0 zeros`, a flag added this session over machinery `BatchedACTController` always had. |
| 5.16 | **`summarize_arms.py` would have mis-read the sweep, caught by exercising it on a synthetic sweep before the real one landed.** `bar = bar_v0 if t.endswith("Slot-v0") else bar_loose` is False for `Rebot-PrecisionSlot-Tight-v0`, so **Tight was being judged against the LOOSE bar (0.55)**. A Tight rate of ~0.23 would have printed `MISSES` and fired the verdict *"at least one arm missed the Stage C bar -- the bottleneck is not data composition"*, mis-framing the entire read. Second, silently letting a third task into a rule pre-registered as "B > A on **both** tasks" changes a pre-registered rule after seeing data. | Fixed by an explicit `PREREG = (Loose, v0)` tuple: Tight is **reported in full with Wilson CIs and excluded from both the bar check and the verdict**, and the verdict header now names the tasks it covers. Verified on a synthetic sweep built to the real filename/schema (6 runs × 3 tasks × 2 spawns) with Tight deliberately low, then deleted. The pre-registered rule is preserved exactly, not weakened to fit. |
| 5.17 | **Stage C training sweep complete: 6/6 runs, 100 000 steps each, 04:11:13.** Final-1k loss 0.030–0.038 across all runs; arm B marginally higher (0.0322–0.0384) than arm A (0.0295–0.0364), overlapping. | Throughput cleanly separates the contention cost: **48.0 steps/s** on the four uncontended runs vs **39.3 / 40.1** on the two that overlapped evals — an **18 % training-side tax**, on top of the concurrent eval itself going 2 m 11 s → 12 min. Both sides of the "one GPU job at a time" rule now measured rather than asserted. Arm B's slightly higher loss is consistent with DART data being harder to fit and is **not** evidence about generalisation; only the eval sweep answers that. |
| 5.18 | **⚠ CORRECTS 5.10 — the "jam at the slot mouth" story is a `ckpt_0030000` story and does not survive to `ckpt_final`.** New tool `analysis/failure_taxonomy.py` sorts failures by a decision list, splitting `gross_miss` (block ended > 15 mm off the slot axis, i.e. one block half-width, `mdp.BLOCK_HALF[1]`) out of `never_entered`. Across the curve on `-v0`/s777: 10k → 37 fails (2 never_lifted, 8 gross, 5 never_entered, 21 stalled, 1 seat_reject); 30k → 10 (0/0/5/3/2); **100k → 7 (0 / 3 gross / 1 / 3 / 0)**. | At 30k, five failures sat at **1.42–1.90 mm lateral against a 1.5 mm clearance** — the mouth jam. At `ckpt_final` there are **zero** such episodes; what survives is two unrelated populations, **3 gross transport failures** (\|lateral\| 23.0 / 30.1 / 73.4 mm) and **4 depth failures with fine lateral alignment** (\|lateral\| ≤ 1.16 mm, one at exactly the 40.0 mm threshold). `never_lifted` and `seat_reject` both empty out. Conflating gross with marginal hid this: one bucket whose median \|lateral\| is 1.82 mm at 30k and 30.12 mm at 100k is not one mechanism. **Held with its uncertainty — n = 7.** The sweep's 36 evals pooled via `--pool` give the powered version. |
| 5.19 | **The censoring question, settled by measurement instead of argument — and it partly RETRACTS 5.11.** Run the *same* policy on two clearances: if \|lateral\| were purely censored, doubling the channel 1.5 → 3.0 mm should double every quantile. Measured (armA_seed0 `ckpt_final`, s777, successes): p25 0.29 → 0.29 (1.02×), p50 0.76 → 0.66 (0.87×), p75 1.10 → 1.10 (1.00×), p90 1.34 → 1.76 (1.32×). | **The bulk does not move** — below p75 the quantiles are identical within noise, and only the tail widens. So the policy's lateral error is largely **intrinsic**, and 5.11 overcorrected: censoring is real but confined to the upper tail. The expert looked different (p90 1.38 → 1.11 → 0.48) because its intrinsic spread is tighter, so the channel binds more often for it — both observations stand, about two different distributions. **Useful output:** estimate Tight from the least-censored view (Loose), giving **39/96 = 0.406** of episodes inside a 0.5 mm channel, not 0.19. That is a refinement of *reasoning*, not a moved goalpost — `EXP_TIGHT.md` belief 1 pre-registered **25–55 %** and 0.41 sits inside it. |
| 5.20 | **`-Loose-v0` (3.0 mm) result: 94.8 %** (91/96, 95 % CI [0.884, 0.978]), armA_seed0 `ckpt_final`, spawn 777. Failure buckets: 2 never_entered, 2 stalled_in_mouth, 1 seat_reject; **zero** never_lifted, gross_miss or yaw_reject. | First cell of the 36-eval sweep. With `-v0` at 92.7 %, the two easier rungs are both comfortably above the 70 % objective for arm A seed 0 at one spawn seed. Pre-registered Loose bar was 0.55. |
| 5.21 | **Spawn-seed variance is large and immediately visible.** Same checkpoint (armA_seed0 `ckpt_final`), same task (`-Loose-v0`), two spawn seeds: **94.8 % (s777) vs 87.5 % (s888)** — a **7.3-point** spread, n=96 each. | Vindicates the pre-registered two-spawn-seed protocol (4.7: repeating a seed is bit-reproducible and yields *zero* information, so variance can only be seen by varying the seed). It also sets the scale for reading every single-cell number in this project, including the 92.7 % headline: a one-spawn-seed rate carries roughly ±7 points of spawn variance on top of its Wilson interval. **Do not compare two arms on one spawn seed.** |

### The sweep (36/36, zero failures, 05:54:58) — verdicts

| # | what | verdict |
|---|---|---|
| 5.22 | **⚠ RETRACTS the 92.7 % headline. Training-seed variance is 15–29 points on identical data.** Per-training-seed later-rates on `-v0`: arm A **0.927 / 0.745 / 0.656**, arm B **0.969 / 0.688 / 0.719**. `bc_armA_seed0` — the source of every number I reported mid-session — is the *luckiest* of three seeds for both arms. | Exactly the pick-place pattern (32.8–59.4 % across training seeds on identical data) reproducing here. **Honest pooled headline** over 3 training seeds × 2 spawn seeds, n=1152/arm/task: Loose **0.778 / 0.799**, v0 **0.776 / 0.792**, Tight **0.708 / 0.764**. Weaker number, far stronger claim: **the 70 % objective is cleared on all three clearances**, not at one cherry-picked cell. Pre-registered bars (0.55 Loose / 0.40 v0) cleared by both arms. |
| 5.23 | **⚠ REFUTES 5.11, 5.19 and `EXP_TIGHT` belief 1. Tight is barely harder than v0: 0.736 pooled.** Predictions were 0.19 (censored-CDF), 0.406 (Loose-based) and 0.25–0.55 (pre-registered). Actual **0.736**. Per-run the ladder is not even monotone — **4 of 6 runs have Tight within 5 pts of `-v0` or better**, and `armA_seed1` is *better* on Tight (0.786) than v0 (0.745). | Belief 4 survives handsomely: **1 yaw_reject in 797 failures** across all clearances. Only `armA_seed0` shows a large Tight drop (0.927 → 0.703) — the same seed behind the retracted headline. A one-run reading of the clearance ladder would have been wrong in both directions. |
| 5.24 | **THE REAL BOTTLENECK IS DEPTH, NOT PRECISION — and it is clearance-independent.** Pooled taxonomy is nearly identical at every clearance: `never_entered` 41.8 / 38.2 / 44.7 % and `stalled_in_mouth` 41.8 / 43.8 / 38.2 % (Loose / v0 / Tight) — together **82–84 % of all failures**. Decisively, `never_entered` failures have median \|lateral\| of **0.59 / 0.79 / 0.73 mm** — on Loose that is 0.59 mm of error inside a **3.0 mm** opening, with depth −40.8 mm. | The block is well aligned and simply **stops ~40 mm short of the slot**. These are not precision failures at all, which is why tightening the channel 6× moves success by only 5 points and barely touches the composition. Corroboration from the expert's own design note: `insert_x = 0.2545` "drives the block into the **back stop**, which squares it and removes depth variance" — the expert solves depth with a hard stop; the clone inherits the trajectory but not the guarantee. **Any further effort should target the push, not lateral centring.** |
| 5.25 | **DART: no measurable difference — and the pre-registered rule correctly overrides a "significant" McNemar.** Gap B−A = **+0.021** (Loose) and **+0.016** (v0) against within-arm seed spreads of **0.229 / 0.281**. The pairing check confirms the arms faced **identical spawns in all 2304 slots, 0 mismatched**, and McNemar reports χ² = 13.35, **p < 0.05**. | **That significance is a clustering artefact.** The 2304 episodes are not independent — they cluster into **3 training seeds**, and the seed effect (22–28 pts) is an order of magnitude larger than the arm effect (1.6–2.1 pts). Treating clustered observations as independent inflates n from 3 to 2304. At the correct unit: B wins seeds 0 and 2, loses seed 1, **consistently across all three tasks** — a sign test on 2/3 gives p = 0.5. Three training seeds cannot resolve a two-point effect. **Do not claim DART helped.** |
| 5.26 | **`EXP_TIGHT` beliefs 2 and 3 CONFIRMED — freezing the flow's x0 at the mode is worth +16.7 pts on Tight and costs nothing on the wide clearances.** armA_seed0 `ckpt_final`, s777, later cohort n=96: Loose 0.948 → 0.927 (−2.1, p=0.55), v0 0.927 → 0.896 (−3.1, p=0.45), **Tight 0.708 → 0.875 (+16.7, p=0.0045**, Wilson CIs non-overlapping: 0.790 vs 0.794). | **Contradicts the pick-place precedent exactly as pre-registered** (there, blind mode *resampling* was worth +8.6 pts). Mechanism: x0 sampling noise is a fixed spatial jitter — 3.0 mm and 1.5 mm channels absorb it, a 0.5 mm channel cannot. **Caveat: one training seed, and `armA_seed0` is the run with the largest Tight deficit** (v0 0.927 → Tight 0.703), i.e. the most room to help. Replication across the other five runs launched. |
| 5.27 | **`--fixed-x0` BREAKS spawn pairing — caught by `analysis/paired_evals.py`, which reported "spawns DIFFER in 96/96 slots, chi2 invalid".** Mechanism: with a frozen x0 the policy stops calling `torch.randn` at every refill, so global RNG consumption changes and the reset-event stream desynchronises. | **Corrects `EXP_TIGHT.md` §6's own pre-registered decision rule**, which specified paired McNemar. The comparison is still valid (same `--seed`, same spawn distribution) but **unpaired and less powerful**, so it must be read with two-proportion z-tests and Wilson intervals. A hand-rolled comparison would have silently reported a paired χ². **Any `--fixed-x0` vs stochastic comparison is unpaired by construction.** |
| 5.28 | **⚠ RETRACTS 5.26 — `EXP_TIGHT` belief 2 is REFUTED on replication.** `--fixed-x0 zeros` on `-Tight-v0`, s777, all six runs: armA_seed0 **+0.167**, armA_seed1 **−0.375**, armA_seed2 −0.187, armB_seed0 −0.010, armB_seed1 −0.052, armB_seed2 −0.115. **n=6, mean −0.095, sd 0.182. One run improved, one flat, four worse.** | The +16.7 pts at p=0.0045 was a **single-seed fluke on the seed I happened to test first** — and `armA_seed0` was precisely the run with the largest Tight deficit, i.e. the most room to help. Only the pre-registered replicate-across-seeds rule stopped this being written up as a result. Every number in the single-seed test was correct; the *inference* was not. Belief 3 (neutral at wide clearances) is unaffected. |
| 5.29 | **The finding that SURVIVES: the x0 choice alone moves success across a 54-point range** (−37.5 to +16.7) with policy weights, task, spawn seed and episode count all held fixed. | Reproduces EXP07's pick-place gate 2b almost exactly (frozen x0 draws spanned 14.1–56.2 % on one frozen base). The x0 → outcome map has enormous leverage on this architecture; what varies is **which** x0 is good, which is a property of the checkpoint and plausibly of the state — not a constant like "zeros is best". **This is the measured case for Stage D:** blind freezing is a coin flip with a 54-point spread, and x0-*steering* replaces the blind choice with a learned state-conditioned one. Belief 5's discount still applies (no never-lifted bucket here), so expect a modest gain — but the leverage being steered is no longer hypothetical. **Aim it at depth**, per 5.24. |
| 5.30 | **`make_videos.sh` silently delivered one video while claiming three.** gymnasium's `RecordVideo` always writes `rl-video-step-0.mp4` into the same `videos/` dir, so each clearance overwrote the previous one. The only visible sign was a `UserWarning: Overwriting existing videos` buried in the log; the script's closing `ls` listed a single file and looked plausible. | Fixed by renaming to `<task>.mp4` immediately after each recording, before the next task can clobber it, with an explicit `!!! no video produced` marker if the file is missing. Caught only by reading the `ls` output against the promise — three tasks, one file. Same family as gotcha 32: a loop that reports success per iteration while the artifacts collapse onto each other. |
