# 01 — `Rebot-ClutterExtract-v0`: first-principles task analysis

*Written 2026-08-02 from the env source, the eva_rl write-ups, and the Isaac Lab manager
source. Every claim is either quoted from code or derived; the derived ones are marked
**[DERIVED — must verify in sim]** and become the Stage-0 probe list in `02_PLAN.md`.*

Sources read:
`challenge/clutter_env_cfg.py`, `challenge/mdp/clutter.py`, `challenge/mdp/common.py`,
`eva_rl/docs/envs/clutter-extract.md`, `eva_rl/docs/CHALLENGE_SUITE.md`,
`IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py`,
`IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py`.

---

## 1. What the task actually is

A blue **target** block sits at env-local `(0.250, 0.000, 0.035)`. Four lighter grey
**distractors** sit in the same row at `y = ∓0.042, ∓0.084`. All five blocks are
**36 × 30 × 70 mm** (`CL_BLOCK_HALF = (0.018, 0.015, 0.035)`); the target is 0.04 kg, each
distractor 0.025 kg.

Goal: get the target into a 45 mm-radius zone centred at `(0.185, −0.185)` and set it down,
**without any distractor's up-axis dropping below `TOPPLE_DOT = 0.75`**.

Success predicate (`clutter.target_at_goal`, `mdp/clutter.py:65`):

```python
((p[:, :2] - goal).norm(dim=1) < GOAL_RADIUS)   # within 45 mm in xy
& (p[:, 2] < 0.055)                             # set DOWN, not held high
& ~any_distractor_toppled(env)
```

### Three consequences of that predicate worth naming early

1. **The target does not have to end up upright.** Only `p_z < 0.055` is required. A 70 mm
   block lying on its side has its centre at 15 mm or 18 mm — comfortably under the bar. So
   "carry it over and drop it in the circle" is a *legal* solution. This is much weaker than
   "place it neatly", and it is the cheapest thing an expert can aim for.
2. **The no-topple constraint applies only to distractors**, and it is checked *at the moment
   of the success query* — but since `distractor_toppled` is also a termination, an episode
   that topples anything is over immediately anyway.
3. `clutter_success` is a **per-step** reward (`ManagerTermBase.__call__` returns
   `ok.float()`), so holding the goal state pays every step until timeout. Getting there
   *early* is worth far more than getting there at all.

## 2. Timing and reward scale — the −40 penalty is really −0.8

| quantity | value |
|---|---|
| `sim.dt` | 1/400 s |
| `decimation` | 8 |
| **control dt** | **0.02 s (50 Hz)** |
| `episode_length_s` | 14.0 |
| **steps per episode** | **700** |

Isaac Lab's `RewardManager.compute` (`reward_manager.py:150`) does

```python
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
```

so **every weight in `RewardsCfg` is a per-second rate**, not a per-step amount. Multiplying
through by `dt = 0.02`:

| term | weight | per-step max | episode budget |
|---|---|---|---|
| `reaching` | 2.0 | 0.040 | ≈ 28 if pinned at the target for all 700 steps |
| `lifting` | 8.0 | 0.160 | paid only while target > 55 mm **and** TCP within 80 mm |
| `extracted` | 15.0 | 0.300 | paid only while target > 90 mm and nothing toppled |
| `carrying` | 12.0 | 0.240 | gated on target > 70 mm |
| `success` | 60.0 | **1.200** | 1.2 × (steps spent in the goal zone) |
| `disturbance` | −3.0 | −0.06 per metre of summed distractor displacement | tiny in practice |
| **`topple_penalty`** | **−40.0** | **−0.800, once** | `is_terminated_term` returns 1.0 on the terminating step only |
| `action_rate` | −2e-2 | −4e-4 · ‖Δa‖² | |
| `joint_vel` | −5e-3 | −1e-4 · ‖q̇‖² | |

`is_terminated_term` (`isaaclab/envs/mdp/rewards.py:42`) returns the termination flag for the
named term, masked by `~time_outs`, i.e. **1.0 exactly once**. So the headline "−40 penalty"
is worth **−0.8** in accumulated-return units.

### The exploration landscape this creates

| behaviour | approximate return |
|---|---|
| hover near the target, never touch anything, time out | ≈ **20** (reach ≈ 0.7 sustained × 0.04 × 700) |
| reach in, topple a neighbour at t = 2 s | ≈ 4 − 0.8 = **3** |
| grasp, extract, carry, set down at t = 8 s, hold 6 s | ≈ 20 + 30 + **360** = **> 400** |

Two things follow, and they set the whole strategy:

- **The real cost of toppling is not the −0.8, it is the forfeited future.** Terminating at
  t = 2 s throws away up to ~400 points of remaining episode.
- **"Approach but never commit" is a strong local optimum at ≈ 20**, and it is *safe*, while
  every step toward the actual solution passes through states that risk a −0.8-and-reset.
  This is a hard-exploration problem with a deceptive shaped gradient. **It is exactly the
  structure that makes from-scratch PPO a bad first bet and a demonstration-initialised
  policy the right one** — which is fortunate, because eva_bc *is* a demonstration pipeline.

## 3. The geometry — where the difficulty actually lives

### 3.1 The approach must be near-horizontal (C1), so the fingers land in the gaps

`CHALLENGE_SUITE.md` C1, from 819,200 sampled joint configurations: in the table band
`z ∈ [0.00, 0.10)`, **0.00 %** of voxels admit a top-down grasp, and the steepest approach
attainable anywhere in that band is **42.3° off vertical**. There is no top-down grasp.

So the gripper comes in roughly along **+x** (from the robot toward the row), with the
fingers separating along **±y**. The blocks' 30 mm dimension is the y dimension —
`clutter.py:27` says so explicitly: *"30 mm across the fingers"*. And the row is spaced
along **y** at 42 mm pitch.

**Therefore the two fingers must descend into the two 12 mm gaps flanking the target.**
That is the whole task in one sentence, and it is a deliberate design: the approach axis and
the clutter axis are orthogonal, so there is no way to dodge the constraint by re-orienting.

### 3.2 The free gap, and how much of it the jitter eats

Nominal: pitch 42 mm − block width 30 mm = **12 mm** per side.
`Tight-v0`: pitch 36 mm → **6 mm** per side.

Reset jitter (`clutter_env_cfg.py:191–209`):

| asset | x | y | yaw |
|---|---|---|---|
| target | ±0.012 | — | **±0.20 rad** |
| distractor 0–3 | ±0.010 | ±0.005 | — |

The **target yaw jitter is the dominant gap-eater**. A 36 × 30 mm rectangle yawed by θ
presents a y-extent of `30·cos θ + 36·sin θ`; at θ = 0.20 rad that is
`29.40 + 7.15 = 36.55 mm`, i.e. **6.5 mm wider than nominal**, taking 3.3 mm off each gap
before the distractors' own ±5 mm y-jitter is counted. eva_rl's own measurement agrees:
`test_clutter_env.py` reports a post-reset free gap spanning **7.0 – 18.8 mm**.

**[DERIVED — must verify in sim]** The worst-case episode therefore presents a ~7 mm gap on
at least one side, and the target is yawed up to 11.5°, so the gripper must *also* yaw to
match it — which widens the gripper's own y-footprint at exactly the moment the gap is
narrowest.

### 3.3 The unmeasured number that decides everything: finger outer width

> **⚠ SUPERSEDED 2026-08-02 — read `06_EXPERT_DESIGN.md §3 instead.** The analysis below
> reasons about "opening the gripper to q ≈ 0.017 to admit a 30 mm block". **There is no such
> command.** `BinaryJointPositionActionCfg` offers exactly two apertures — `q = 0.045`
> (89.07 mm clear gap) and `q = 0.000`. There is no intermediate aperture, so the finger inner
> faces are fixed at **±44.53 mm** from the tool centre whenever the gripper is open, which
> puts them 17.5 mm *inside* both neighbouring distractors when centred on the target.
>
> The real feasibility result: there are exactly **two ≈6.94 mm alignment windows** where both
> fingers sit in free gaps, and **each encloses two blocks, never one**. A direct single-block
> grasp of the middle block is geometrically unavailable in the nominal spawn.
>
> The rest of this section is kept as the record of how the question was first framed. The
> finger-thickness measurement is still worth taking — it bounds those two windows — but it is
> no longer the number that decides feasibility.

The gripper's *clear opening* is well characterised (C3): `sep = 2.000·q` exactly, and
`gap = 1.0035·(q_L + q_R) − 1.25 mm`, so at the commanded `q = 0.045` the opening is
**89.1 mm**; forced, it reaches ~120 mm.

But threading into the row is not about the clear opening — it is about the **outer** width
of the two finger bodies:

```
outer_width(q) = clear_gap(q) + 2 · finger_thickness_y
```

To admit a 30 mm block (plus, say, 3 mm of approach clearance) needs `clear_gap ≈ 33 mm`,
i.e. `q ≈ 0.017`. The space available between the two neighbouring distractors' inner faces
is `30 (target) + 12 + 12 = 54 mm` nominal, **and only ~44 mm in the 7 mm-gap worst case.**

So the feasibility condition is:

```
33 + 2·t  ≤  54   →   t ≤ 10.5 mm   (nominal)
33 + 2·t  ≤  44   →   t ≤  5.5 mm   (worst-case jitter)
```

**`finger_thickness_y` is not recorded anywhere in either repo.** `gripper_stroke.py`
measures finger *body-origin separation* from the physics view, not the collider extents.
This single number decides whether the task is solved by *threading* or must be solved by
*singulation*, and it is Stage-0 probe #1.

### 3.4 Pushing a neighbour aside is probably not available — the tip-vs-slide bound

The task write-up says the policy "must either thread the fingers in precisely **or push a
neighbour aside first**". The second option looks unavailable on this arm.

A block of width `w` pushed horizontally at height `h` slides if `h < w/(2μ)` and **tips**
otherwise. With `w = 30 mm` and a block/table friction of `μ ≈ 0.7`
(the block authors `static_friction=0.9`; the Seattle Lab Table's material is the USD
default, and Isaac Lab's default combine mode averages them):

```
h_crit = 30 / (2 × 0.7) ≈ 21 mm
```

And C9 measures the **usable TCP floor at ~44 mm above the table** — commands below it land
at z = 0.039–0.055. So the lowest the tool centre point can push is roughly **twice the
tipping height**.

Worse, the block only has to tilt `atan(30/70) = 23.2°` for its centre of mass to pass the
tipping edge and fall, while `TOPPLE_DOT = 0.75` fires at `acos(0.75) = 41.4°`. There is no
"tip it a little and it comes back" band above 23.2°.

**[DERIVED — must verify in sim]** *A horizontal push anywhere the TCP can reach topples the
distractor rather than sliding it.* If this holds, the task has **exactly one solution
family: thread the fingers into the gaps**, and the "learn to push, VPG-style" framing in
the write-up is not achievable on this arm.

Three caveats that must be tested rather than argued, because each could rescue pushing:

1. **Pressing down while pushing** adds restoring moment and raises `h_crit`. A grasp-and-drag
   (close on the neighbour, translate, release) is a different mechanic entirely and may be
   the legitimate "singulation" primitive.
2. **The finger tips may reach below the TCP.** The 44 mm floor is where the *gripper body*
   bottoms out; if the fingertips extend downward past the TCP, contact could occur below 44 mm.
   Probe #1 measures this too.
3. **The real μ is a guess.** The table's material must be read off the stage, not assumed.

### 3.5 Grasp height, and why it is comfortable

The block is 70 mm tall, centre at z = 35 mm — *below* the 44 mm TCP floor. So a
centre-height grasp is impossible; the grasp must be **high on the block**, TCP at roughly
**z = 0.050–0.060**, i.e. 15–25 mm below the top face. C9's reliable band is
`x ≈ 0.22–0.26 m, z ≈ 0.045–0.10 m` and the row sits at `x = 0.250` — **the row was placed
inside the reliable band on purpose.** This part of the task is well-conditioned.

A high grasp does mean more tipping leverage on the *target* while extracting, but the target
is the heavy one (0.04 kg) and toppling it is not penalised — only `target_dropped`
(below −0.05 m) ends the episode.

### 3.6 Carry geometry

Goal `(0.185, −0.185)` is at radius `0.262 m` — inside C4's `r ≤ 0.32` envelope, and the
carry is ~196 mm of travel from the row. `carrying` only pays above z = 70 mm and
`extracted` above z = 90 mm, so the intended trajectory is: **grasp → lift clear (≥ 90 mm) →
translate → descend into the zone**.

Lift required: target centre from 35 mm to ≥ 90 mm = **55 mm of clean vertical extraction**
while the block is still flanked by neighbours for the first ~35 mm of it.

## 4. Observation and action spaces

### Actions — 7-D

```python
arm_action     = JointPositionActionCfg(joint_names=["joint[1-6]"], scale=0.5, use_default_offset=True)
gripper_action = BinaryJointPositionActionCfg(joint_names=["joint_left", "joint_right"], ...)
```

`use_default_offset=True` with `scale=0.5` means the arm command is an **absolute joint
target**: `q_target = q_default + 0.5 · a`, so `a ∈ [−1, 1]` spans ±0.5 rad about the start
pose. This matches eva_bc's encoding exactly — POSTMORTEM §2 describes "6 arm-joint position
targets (scaled `(dq − q_default)/0.5`) + 1 binary grip channel". **The action encoding ports
with zero change.**

### Observations — 42-D

| slice | term | dim | content |
|---|---|---|---|
| `0:8` | `joint_pos_rel` | 8 | 6 arm + 2 finger joint positions, relative to default |
| `8:16` | `joint_vel_rel` | 8 | matching velocities |
| `16:23` | `block_pose_in_root` (target) | 7 | position (3) + quaternion **XYZW** (4) |
| `23:35` | `clutter_obs` | 12 | per distractor: `(dx, dy)` relative to the target, and its `up_z` |
| `35:42` | `last_action` | 7 | previous action |

Things worth noticing:

- **The TCP pose is absent.** It is recoverable by forward kinematics from `joint_pos`, but
  a network must *learn* FK to do mm-precision work. eva_bc hit exactly this class of problem
  and its answer was obs surgery (`residual_core.py`'s 64-D builder adds relative poses).
  Adding TCP-relative-to-target features is the highest-value cheap change here.
- **The goal is absent — correctly**, because it is a constant `(0.185, −0.185)`.
- **The grasp-bit ingredients are all present**: finger positions `obs[6:8]`, finger
  velocities `obs[14:16]`, last grip command `obs[41]`. That is precisely eva_bc EXP01's
  *variant D* (5 dims: finger pos + finger vel + last commanded grip), the one that scored
  AUC 0.968 with **0 % FPR**. The grasp-bit probe ports directly, with new indices.
- **The distractor obs gives `up_z` but not tilt direction**, and gives xy relative to the
  target rather than to the TCP. For a policy that has to avoid touching them, TCP-relative
  distractor offsets would be the more useful framing — another candidate obs surgery.

## 5. What is already validated, and the one thing that is not

`scripts/test_clutter_env.py` passes: obs shape/finiteness, row settles upright, gap measured
at 7.0–18.8 mm, `TOPPLE_DOT` reachable (16/16 for a block laid on its side), a 30 mm shove
registers 37.1 mm of disturbance **without** toppling, `target_at_goal` fires 16/16 and
rejects all three negative controls (held high; outside radius; perfect placement but a
distractor down).

Against `CHALLENGE_SUITE.md` §4's validation ladder, the clutter env has cleared V1/V3/V5/V6.
It has **not** cleared:

> **V4 Achievability** — *"a scripted / IK expert completes it at a measured success rate.
> This is the load-bearing rung — it is what 'validate the env is achievable' means"*

and the env doc states it outright: *"**Still not verified:** a scripted extraction
demonstrating the task is achievable at the tight end of that range."*

**This is the single most important fact for planning.** We are not being asked to train a
policy on a task known to be solvable; we are being asked to solve a task whose solvability
has never been demonstrated. Every hour spent on BC or RL before V4 passes is spent on an
unbounded risk. Stage 0 is therefore *not* "set up the pipeline" — it is **"prove the task
can be done at all, and measure at what rate."**

That ordering is also exactly eva_bc's own doctrine: its Stage 1 gate is "expert success rate
and failure anatomy on a fixed suite", and its README notes the perturbed expert number
"matters because it caps what DAgger can later teach."

## 6. Open questions going into Stage 0

Ordered by how much they change the plan.

| # | question | why it matters | how to settle it |
|---|---|---|---|
| Q1 | **Finger outer width in y** (collider extents, not body origins) | decides threading vs singulation; if `t > 10.5 mm` the nominal task is unthreadable and the whole approach changes | probe: read finger collider AABBs from the physics view at several `q` |
| Q2 | **Does the fingertip reach below the 44 mm TCP floor?** | determines whether any low push/contact is available | same probe, plus a commanded-descent sweep |
| Q3 | **Tip-vs-slide: does a push at reachable heights topple?** | kills or confirms the entire "singulation" solution family | scripted push at 3 heights × 3 speeds, measure `up_z` |
| Q4 | **Effective block/table friction** | sets `h_crit` | read the table's physics material off the stage |
| Q5 | **Is the −0.8 topple penalty as computed?** | confirms the exploration analysis | log per-term reward on a deliberately-toppling rollout |
| Q6 | **Achievable straight-line Cartesian tracking accuracy at x ≈ 0.25, z ≈ 0.05** | an expert needs ≈ ±1 mm of y-precision to thread a 7 mm gap; C9 already warns of 14–48 mm tracking error at x = 0.30 | commanded-vs-achieved TCP sweep through the differential-IK controller |
| Q7 | **Throughput and VRAM at N envs on a 10 GiB 3080** | every budget in both repos assumes different hardware | timed rollout at N = 16 / 128 / 512 / 1024 / 2048 |
</content>
