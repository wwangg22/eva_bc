# Rebuilding the Stage-1 expert without cuRobo

`python -c "import curobo"` fails in `env_isaaclab6`. This document records exactly what
`expert/run_expert_v1.py` does, which parts are cuRobo-specific, and what the replacement
has to provide. Companion: `PORT_MAP.md` (the `act/` side), `PLAN.md` (stage targets).

---

## 1. The good news: the cuRobo surface is tiny

The runner touches cuRobo in only five places, and **the only thing the rest of the runner
needs from the planner is a time-parameterised joint array at 50 Hz**:

```python
# run_expert_v1.py:248-264  -- the entire planner -> controller bridge
traj = getattr(result, f"{which}_interpolated_trajectory")
last = getattr(result, f"{which}_interpolated_last_tstep")   # bug #692: tail rows are stale
pos  = traj.position.reshape(-1, D).cpu().numpy()[: max(int(last), 2)]
t_src = arange(len(pos)) * 0.025          # cuRobo interpolation dt
t_dst = arange(0, t_src[-1], 0.02)        # 50 Hz env
out   = per-joint np.interp               # -> (T', 6)
```

So any replacement that yields `(T, 6)` joint waypoints slots in unchanged. The **minimal
interface** is five functions:

1. `plan_grasp(q6, target) -> {approach_qs, grasp_qs, lift_qs}` or a failure token
2. `plan_place(q6, target_xy, heights) -> qs` or failure
3. `plan_to_config(q6, q_goal) -> qs` or failure (the retreat)
4. `update_world(state)` — no-op if the path is analytic and collision-free by construction
5. `attach / detach` — no-op if the payload is not collision-checked

Everything else in the 811-line runner is generic and **ports as-is**: `build_train_mask`,
the env setup, all state readers, the perturbation primitives, `step_action` / `mark` /
`run_traj` / `hold`, the whole `fetch_one` retry-and-accounting control flow, `episode()`,
and the HDF5 writer.

### The action interface (generic, ports verbatim)

```python
# run_expert_v1.py:428-455
a = torch.zeros(1, 7, device="cuda")
a[0, :6] = (dq - q_default) / 0.5     # 1 action unit = 0.5 rad, relative to default joint pos
a[0, 6]  = grip                       # +1 = OPEN, -1 = CLOSE
obs = self.env.step(a)[0]
```

The `(o_t, a_t)` pair is recorded **before** the step, `last_obs` refreshed after — so the
obs/action alignment is causal. Keep that ordering.

---

## 2. The chosen replacement

**Batched damped-least-squares IK inside the sim, verified by FK**, with CEM as the
fallback and the verifier. Rationale in `PLAN.md` §4. Two structural reasons this fits
better here than it did for pick-place:

- The slot is at a **fixed pose**. Pick-place had a randomised basket *and* two objects, so
  it needed a goalset ranker over a 12,953-row grasp table. Here only the **block spawn**
  moves, over a small box: `x +/- 0.02`, `y +/- 0.03`, `yaw +/- 0.35 rad`. That is a local
  perturbation around one nominal, which is exactly the regime where DLS IK is
  well-conditioned.
- The insertion leg is a **straight line along +x holding a constant wrist orientation** —
  the easiest possible Cartesian path.

### Two rules that already burned this project

- **Finish every CEM/IK search before placing the object.** These searches score candidates
  with `write_joint_state_to_sim`, which teleports the arm and re-opens the fingers hundreds
  of times. Computing a lift path *after* closing the gripper drops the object and reads as
  a slip. Fixing the ordering alone took one control trial from 0 % to 100 %.
- **Position-only IK is not a sufficient specification for this arm.** The finger-separation
  axis (`gripper_left - gripper_right`) must be constrained explicitly, or the wrist arrives
  holding the block along its 45 mm length instead of across its 30 mm width and squirts it
  out of the slot. Measured: constraining it took the insert rate from **0 % to 55-81 %**.

Both are already implemented in the CEM cost used by
`eva_rl/scripts/challenge/slot_insertion_probe.py` and by
`eva_bc/slot/analysis/gripper_envelope.py`:
`cost = ||tcp - target|| + 0.25 * (1 - |sep . y_hat|)`.

### What we lose, and whether it matters

| lost with cuRobo | matters here? |
|---|---|
| goalset ranking over a proven-config table | **No** — the table existed because the pick-place wrist had a narrow feasibility manifold across a *wide* spawn annulus (`r` 0.16-0.36). Our spawn box is one small patch. |
| attached-object collision model | **Probably not** — the carried block's only collision partner is the slot, which is the target. |
| `plan_cspace` retreats | **No** — the runner already falls back to a 60-step linear joint interpolation at ladder rung 4. |
| collision-aware free-space planning | **Watch it** — the only obstacles are the table and the fixture. The approach must not sweep the hand through the slot walls. |

---

## 3. The phase machine, retargeted

Pick-place phases (`settle / approach / descend / close / lift / reopen / transport /
release / retreat`) map onto this task with **two new phases**:

| phase | pick-place | slot version |
|---|---|---|
| `settle` | 10 steps, gripper open | unchanged |
| `approach` | to a 10 cm standoff above the can | to a standoff **beside** the block, gripper open, wrist already yawed to match the block's yaw |
| `descend` | linear along world -z | linear along the grasp approach direction to the grip point |
| `close` | 6-step settle, then 15 steps commanded closed | unchanged; `close_disp` measured the same way |
| `lift` | 8 cm world-z lift | lift to carry height (the block must clear the **slot floor at z = 0.020**) |
| `reopen` | on a missed grasp | unchanged |
| `transport` | ladder of place plans to the basket | carry to the pre-insertion pose at the slot mouth |
| **`align`** | — | **NEW.** Null the 6-DoF error at the mouth: yaw to 0, lateral to 0, height to the grip band. This is where the 1.5 mm tolerance is won or lost. |
| **`insert`** | — | **NEW.** Straight-line +x stroke to depth >= 40 mm, wrist orientation held constant. |
| `release` | open, detach, settle | unchanged |
| `retreat` | plan home | linear interp to `Q_HOME` (rung-4 fallback pattern) |

New outcome vocabulary for the two new phases: `aligned` / `misaligned` / `inserted` /
`jammed`. **A jammed insertion followed by a retract-and-retry is structurally identical to
the `missed` -> `reopen` pattern**: mask the failed attempt, keep the recovery trainable.

### The coupling that pick-place never had

**The grasp pose and the insertion pose are not independent.** A can could be re-oriented
freely on the way to the basket; this block cannot — the wrist must arrive at the slot with
the finger axis on world y, and the block must be square. So the grasp candidate scorer needs
a **joint** feasibility check: the grasp pose AND its induced insertion pose must both be
reachable. Concretely, since the block spawns yawed by up to +/-0.35 rad and must end at yaw
0, the wrist has to sweep that yaw while holding the block. Verify that sweep is inside the
wrist's roll freedom before trusting any grasp candidate.

---

## 4. Success / failure predicates to re-derive

Every threshold below is sized against a 24 mm can and a 45 mm-tolerance basket. All of them
need re-deriving, and several change character entirely because our success predicate is
itself a **millimetre-scale** measurement.

| predicate | pick-place | slot |
|---|---|---|
| grasp succeeded | `can_z > 0.05` after lift | block rose clear of the table **and** finger gap ~= 30 mm |
| clean grasp | `close_disp < 0.005` | **tighten** — this defines the nominal BC pool, and a 5 mm shift is 3x the clearance |
| lost in transport | `can_z < 0.04` | block below carry height, or finger gap collapsed to ~0 |
| success | `\|dx\|,\|dy\| < 0.045 and z < 0.05` | `mdp.is_inserted`: depth >= 40 mm **and** lateral within half-width **and** \|yaw\| <= 0.12 rad |

**The strongest new signal available.** C3 measured `separation = 2.000 * q` exactly, with
`gap = 1.0035*(q_l+q_r) - 1.25 mm` at 0.035 mm residual. A 30 mm block therefore pins the
fingers at `q ~= 0.0156` each versus 0 on air — a **15.6 mm** separation in a channel the
observation already carries, against the +/-12 mm the pick-place task had to work with. The
grasp bit should be *easier* here, and finger gap is an unfakeable retention test (a
"is the object within 80 mm of the TCP" check once scored an untouched block on the table
at 100 %).

---

## 5. DAgger gates to re-derive

`collect_dagger.py`'s gates are calibrated to this exact task and **one of them degenerates**:

```python
MISS_STEPS    = 45     # consecutive closed-gripper steps with no rise -> air-close
MISS_RISE     = 0.02
STALL_STEPS   = 450    # steps without an increase in PLACED-CAN COUNT
TIMEOUT_STEPS = 900
DROP_Z, DROP_ABOVE = 0.02, 0.05
```

- `MISS_STEPS` **must exceed** close (21 steps) + ~10 lift steps. The v1 shakedown used 25
  and it tripped on **every successful** mid-lift grasp. Re-derive from our own phase lengths.
- **`STALL_STEPS` is defined on "increase in placed-can count" — with one object that is
  0 or 1 and the gate degenerates.** Redesign it around **insertion-depth progress**.
- `TIMEOUT_STEPS = 900` was for 1500-step episodes. Ours are **600**.
- The `over_basket` exemption (a legitimate >2 cm fall on release) has no analogue; the block
  is never released in free space.

Measured gate distribution on pick-place: **91 % miss / 9 % stall / 0 drop** over 146
takeovers, at 68 % takeover success — that 68 % is the teacher ceiling that capped the
whole pipeline.

**Also carried over:** the takeover must run a **scripted retreat before any planning**
(`collect_dagger.py:250-255`) — go *up* first, then interpolate to `Q_HOME`, *then* plan.
Root cause: policy poses sit inside the planner's collision margins; 3/3 failed takeovers in
one ablation were start-state collisions while the same scenes planned fine from home.
PhysX contact-free is not planner-margin-free. A blind joint-delta lift alone was not enough.

---

## 6. `taxonomy.py` buckets, redefined

The existing buckets are built entirely on `placed_final in {0,1,2}` and a 2-element
`max_can_z`, so the whole set has to be replaced. Proposed, exhaustive and ordered:

| bucket | condition |
|---|---|
| `success` | `is_inserted` at episode end |
| `inserted_then_lost` | reached depth >= 40 mm at some point, not at the end |
| `engaged_shallow` | max depth in [10, 40) mm |
| `aligned_never_engaged` | reached the mouth (within ~20 mm) but max depth < 10 mm |
| `carried_never_aligned` | block lifted clear of the table, never reached the mouth |
| `grasped_never_lifted` | fingers closed on the block, block never rose |
| `never_grasped` | fingers never closed on the block |
| `toppled` / `dropped` | the two terminations |

These need mm-scale observables logged per episode: max insertion depth, lateral error and
yaw error at max depth, max block z, and the finger gap at close. Budget time to calibrate
them the way `MISS_STEPS` was calibrated — the first version of a gate is usually wrong in a
way that looks like a policy failure.

---

## 7. The grasp-bit probe, retargeted

Architecture and protocol port unchanged: 5 raw obs dims -> `Linear(5,128) -> ReLU ->
Linear(128,128) -> ReLU -> Linear(128,1)`, Adam 1e-3, 60 fixed epochs, minibatch 256,
CPU-only, standardised with train-split `mu`/`sd`.

**Dims change**: `[6, 7, 14, 15, 40]` -> **`[6, 7, 14, 15, 33]`** (left/right finger joint
position, left/right finger joint velocity, last commanded grip). Dim 33 is the last element
of `last_action` in our 34-D observation.

**The gate**: **0 % false-positive rate** on on-policy closed-on-air freeze states AND
>= 95 % accuracy on expert post-close frames. Runtime rule: `sigmoid > 0.5` **AND**
commanded-closed (an open command forces the bit to 0).

**Do not drop the commanded-grip channel.** Measured on pick-place: the 5-dim variant scored
AUC 0.968 with **0 % FPR** on 665 real on-policy freeze states, while the 4 physical finger
joints alone scored a *better* AUC (0.976) but **27.1 % FPR**. The disambiguating fact is
"commanded closed AND resulting aperture", not aperture alone. Separately, a hand-tuned
aperture threshold rule was **refuted at 40 % FPR** — the raw aperture distributions of
hold-vs-air overlap almost entirely. And history is refuted as a fix (<= +0.01 AUC, transfers
*worse* at 40.8 % FPR).

Training data construction: positives/negatives are post-close frames from the lift window at
offsets `k in {0, 2, 5, 10}`, labelled by whether that attempt eventually lifted. On-policy
negatives come from miss-gated DAgger episodes at `j in {1,5,15,30,45}` steps before takeover
— **transfer set only, never trained on.** That asymmetry is what makes the result decisive.
Split is episode-level with a fixed seed, reproduced identically by the export script.

---

## 8. Geometry constants to re-point

| constant | pick-place | slot |
|---|---|---|
| object dims | `(0.024, 0.024, 0.036)` | `(0.045, 0.030, 0.070)` |
| object centre z at rest | `0.031` | `0.035` on the table, `0.055` in the slot |
| TCP offset | `0.075` along tool -X | **`0.0419`** (`mdp.TCP_OFFSET`) |
| pinch point along the finger axis | `0.048` | **remeasure by FK** |
| goal fixture | basket, `BASKET_CENTER` randomised | slot, fixed at `(0.245, 0)` |
| `Q_HOME` | `[0, -1.35, -0.3, -0.85, 0, 0]` | use the env's `_START_POSE` |
| lift success height | `0.05` | must clear the slot floor at `0.020` |
| episode length | 1500 steps (30 s) | **600 steps (12 s)** |

The table cuboid and `Q_HOME` are each duplicated across 4-5 files in the original; factor
them into one module here.

Object-orientation logic is **cylinder-specific and must be rewritten**: `can_axis()` reads
body Y because the YCB can is Y-up, and `lying = abs(axis[2]) < 0.5` is a binary
upright/lying test. Our block is authored Z-up with three distinct axes; the env's
`block_toppled` termination (own +z axis has world-z < 0.6) is the right primitive.

Also re-point every `/home/william/...` absolute path — the original repo was developed on a
different machine (grasp table, carry waypoints, robot yml asset root, and every `.sh`).
