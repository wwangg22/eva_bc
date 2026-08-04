# REFERENCE — facts, conventions and commands that survive the reset

**2026-08-03.** The success criterion changed (2 mm, `15_STRICT_METRIC.md`) and the expert is
being started over. Twelve stage-result documents were deleted with that reset. **This file is
what was worth keeping out of them**: measurements that are properties of the environment or
the toolchain rather than of any particular manoeuvre, and rules that were each paid for by a
wrong answer.

Nothing here depends on the success predicate. Everything that did is gone.

---

## 1. The task, in one box

```
Extract the target from a row of five 36 x 30 x 70 mm blocks and set it down in the goal
zone, WITHOUT moving any of the four neighbours more than 2 mm from where it spawned.
The row spawns at a random heading, and the target is any one of the five blocks.
```

| | |
|---|---|
| row | five slots at 42 mm pitch; **the target occupies a uniformly drawn one of the five** |
| row pose | centre at r ≈ 250 mm ±10 mm, **heading `U(−0.30, +0.30)` rad**, applied rigidly |
| free gap between neighbours | **12 mm** (42 mm pitch, 30 mm blocks) |
| goal | (185, −185) mm, fixed — its bearing from the row now varies with the heading |
| spawn jitter, in the ROW's frame | target x ±12 mm and **yaw ±0.20 rad** (no y); distractors x ±10, y ±5 mm |
| measured free gap, pooled | **2.6–20.9 mm**, median ~8 |
| max block radius the env can draw | **0.3087 m** — verified graspable (P41), envelope is 0.32 |
| blocks settle at | z = **32 mm**, not 35 — this shifts every height gate |
| masses | target 0.040 kg, distractors **0.025 kg** (deliberately lighter, so they tip) |
| friction | block 0.9 / table 1.0 / effective **0.95** (PhysX averages, not 0.9) |
| `h_crit` (tip threshold) | 15.8 mm across the row (b = 30), 19.0 mm along it (b = 36) |
| physics dt / decimation | 2.5 ms / 8 → **one legal action lasts 20 ms** |
| episode | 14 s = 700 env steps |

**Current baseline to beat: 3.0 %** (frozen `pose_p33` expert, 768 held-out episodes, seeds
88000–88005, on the randomised row). The same expert scores **17.1 %** on `-Fixed-v0`, the
frozen-row control. **Mission target ≈70 %.**

⚠ **Report success PER SLOT.** The heading is an isometry and cannot change the clutter; the
slot is not — an end slot has one adjacent neighbour instead of two. A pooled rate mixes
configurations of materially different difficulty. See `17_ROW_RANDOMISATION.md`.

**Variants.** `-v0` / `-Play-v0` are the task. `-Tight-v0` is the 6 mm-gap rung.
**`-Fixed-v0`** (strict rule, frozen row) and **`-Lenient-v0`** (topple-only, frozen row) are
diagnostic controls — nothing should be developed against them.

---

## 2. The gripper — the single most consequential fact

```
BinaryJointPositionAction:  where(a[6] > 0, OPEN, CLOSE).  Two states. Nothing between.
_GRIPPER_OPEN = 0.045 per finger  ->  90 mm of separation
target depth 36 mm, 41.2 mm at the full 11.4 deg of spawn yaw
                                  ->  48.8 mm of excess, 24.4 mm PER FINGER
```

**Every grasp sweeps each finger 24 mm through a row whose gaps are 12 mm, and no policy can
choose otherwise.** Measured cost: **17 points** (P39 — narrowing to 46 mm takes the expert
from 16.4 % to 33.3 % with enclosure *improving*). Big Will's decision on 2026-08-03 was to
**keep the 90 mm gripper** — the harder task is the more interesting one — so this is a
constraint to design around, not a knob.

There is no rate limit either. A demo containing a gripper ramp teaches an action the policy
cannot submit.

⚠ **The old "finger blade ~47 mm along the opening axis, ±19.2 mm perpendicular" figure is
retracted.** It was quoted from Stage 1 onward and used to justify grasp geometry. P38's row-
pitch sweep puts the actual fouling reach at **33–39 mm from the target centre** — about 1.8×
the perpendicular figure. The *conclusion* it supported (straddle fore-and-aft, never enter a
row gap) survives on other evidence; the numbers do not.

---

## 3. Action and observation, exactly

**Actions — 7-D.** `JointPositionActionCfg(scale=0.5, use_default_offset=True)`, so

```
q_target = q_default + 0.5 * a        =>   a = 2 * (q_desired - q_default)
```

The expert's `joint1` command lands near `|a| = 1.57` and the peak over the manoeuvre is
**4.63** on `joint4`. **All six joints leave [−1, 1].**

⚠ **`clip_actions = 1.0` in the shipped rl_games config makes the manoeuvre unreachable.**
BC is unaffected (`cfg.clip is None`, and `JointAction` skips its clamp entirely) — but check
it, do not assume it: `collect_demos.py` asserts on it at startup.

**Observations — 42-D**, `policy` group:

| slice | term | dim |
|---|---|---|
| `0:8` | `joint_pos` (rel) | 8 |
| `8:16` | `joint_vel` (rel) | 8 |
| `16:23` | `target_pose` in root frame | 7 |
| `23:35` | `clutter` — per distractor (dx, dy vs target, up-axis) | 12 |
| `35:42` | `actions` | 7 |

`dataset.py` splits this as `STATE = 0:16`, `ENV_STATE = 16:42`.

---

## 4. Kinematics — `probes/_kin.py`

No IK solver and no motion planner. **Candidates are scored by forward kinematics read back
from the sim**, so a search cannot converge on an unexecutable pose. (cuRobo is not installed
and is the wrong tool here anyway — see `00_ENVIRONMENT.md`.)

| member | purpose |
|---|---|
| `fk(q_arm)` | writes joints, `sim.forward()`, returns achieved TCP / `a_hat` / `o_hat` / `low_z` / **all body origins** |
| `cem(pos, seed, …)` | FK-scored cross-entropy search over the 6 arm joints |
| `box_penetration(bodies, boxes, margin)` | deepest intrusion of any **body origin** into any keep-out box |
| `refine(q0, pos, o_des=…)` | per-env damped-least-squares correction with an orientation channel |
| `hold_phys` / `run_phys` / `teleport_arm` | physics-only execution that bypasses the MDP |
| `gap()` / `tcp_now()` / `finger_pos()` | read-backs; ground truth for "did it close on it" |

**The CEM cost is a constrained form, never a weighted sum.** A weighted sum failed in both
directions: `w_pos = 1` let orientation outrank position and the search walked 340–520 mm away;
`w_pos = 20` made orientation noise and returned flipped wrists. Position enters as a **hinge**
(free within 1 mm, then 0.2/mm); floor and keep-out have the same shape; axis terms are bounded
tiebreakers.

Constants that must never drift:

```python
TCP_OFFSET = (-0.0419, 0.0, 0.0)   # MEASURED (CHALLENGE_SUITE C10). Never -0.075 or -0.048.
Q_OPEN, Q_CLOSE = 0.045, 0.0       # binary: 89.07 mm clear gap, or shut
GAP_A, GAP_B = 1.0035, 0.00125     # gap = 1.0035*(q_L+q_R) - 1.25 mm, resid 0.035 mm
```

The inherited `-0.075` TCP offset is **33.1 mm too far forward**. A constant offset error is
invisible in a reach reward — the policy just learns a shifted target — but it is fatal to any
scripted grasp, which then closes 33 mm past the object and shuts on air every time. It caused
several flatly wrong "this object cannot be grasped" measurements.

### 4.1 Two IK branches at the same TCP

**The single most expensive kinematics fact in this project.** Two solutions can be identical
in every statistic the codebase computes — same tool point, same `a_hat`, same `o_hat`, same
keep-out penetration — and differ by **1.90 rad** on `joint6`. Commanding that as one step with
the gripper over the row sweeps every neighbour down.

* **Audit the seam, not just the segment.** Where two independently solved paths *meet*, verify
  agreement in **joint space**.
* **When a path must end at a known pose, solve backward from it.** Forward solving finds *a*
  solution; only backward solving guarantees *the* one.

Worth 73 points when it was found. It will not announce itself.

---

## 5. Traps in the toolchain

- **`env.step` auto-resets a terminated env from *inside* the step call.** Any last-write-wins
  read describes a *fresh spawn*, not the episode that just ended — a toppled env reports
  `topple = False` because the scene is already re-spawned. **Latched quantities are safe;
  final reads are not.** This bit twice in one session. Read `TerminationManager._term_dones`
  immediately after the step for the taxonomy.
- **Physics-only execution (`run_physics`) is required for anything measuring contact**, and it
  is *not* how a policy runs. Both paths exist and their equivalence is measured, not assumed.
- **The paired-comparison noise floor is ~8 % of episodes.** 43 of 512 flip between two runs of
  a bit-identical frozen manoeuvre, while aggregate means agree to ~0.2 points. Quote
  aggregates; never build an argument on small episode-level churn.
- **A CEM is not reproducible under a fixed seed.** 1 320 GPU reductions are not bit-stable and
  60 iterations amplify 1e-7 into 0.1 rad. **The only reliable freeze is to persist the
  artifact**, not to re-derive it reproducibly.
- **Throughput** (Q7): 4 096 envs fit in 3.2 GiB; 2 048 envs run at 34.5 k env-steps/s. The
  10 GiB card is not the limit — one GPU job at a time is.
- `python -u`, always. Output buffers and you see nothing.

---

## 6. BC facts, for when an expert exists again

The flow-matching pipeline in `act/` works and is not the problem. Kept because re-deriving any
of this costs GPU-days.

- **Never shorten the chunk execution horizon.** eva_bc measured 59.4 / 32.8 / 3.1 / 0 / 0 % at
  `n_action_steps` = 15 / 8 / 4 / 2 / 1. **15 is not a tunable.**
- **Training-seed variance dominated everything measured here: sd 9.8 points**, against 1.6 for
  binomial sampling and 0.4 for the flow's own noise. **≥3 seeds per arm or the comparison is
  void.**
- **Training loss does not predict success.** One seed had the *lower* final loss and scored
  10.4 points worse. **Select checkpoints in the simulator.**
- **The flow objective's irreducible loss is `σ·π/2` per dimension** — derived and verified
  numerically — integrating `Var(v|obs,x_τ)` over `τ ~ U[0,1]`. It is *not* the chunk MSE, and
  the per-cell bound is loose exactly when the ambiguity is low-dimensional. Measure it offline
  with a nearest-neighbour calculation before spending a GPU-minute.
- **Make the generator and the gate the same code path.** `collect_demos.py` is both the demo
  writer and the port gate. A separate verifier can agree with the expert while the recorder
  disagrees with both.

---

## 7. Rules, each paid for by a wrong answer

### On measurement

- **Every negative result needs a positive control.** Six wrong answers in Stage 0, all
  plausible, none of which crashed.
- **Run the ablation before the sweep.** A five-point parameter sweep found +2.9 points on an
  inferred mechanism; a two-minute ablation (same trajectory, gripper forced open → **0.0 %**)
  was worth more than all of it. **A sweep assumes you know which parameter; an ablation tells
  you.**
- **Compute the power before running the arm.** Selection-level sd here is 10–13 points; at 3
  selections per arm the 95 % CI is ±12. Five probes and 4 608 episodes once produced verdicts
  that *all* sat inside their own noise. **A comparison whose replications span more than the
  effect is not a measurement.**
- **Pair everything.** One spawn snapshotted and restored between arms; one pose draw shared
  across arms. Fingerprint the scene and assert the pairing held.
- **Report hazard rates, not raw counts.** First-contact counts systematically understate every
  phase after a bad one — they made a **100 %-hazard** segment look like a minor third-place
  contributor.
- **A hazard rate per phase is not an attribution.** Where a threshold is *first crossed* and
  where the *magnitude* accumulates are different quantities with different fixes.
- **Measure the direction, not just the rate.** Eight stages reported a close-phase hazard rate
  and all were correct; none recorded which way the block went, and that one statistic
  reordered every candidate fix and killed two.
- **A control with two variables in it is not a control.**
- **Remove the object to test whether the object matters.** One probe, two mechanisms retired.
- **A predictor mined from a dataset must be tested on data that had no part in producing it,
  and the search has to be counted.** Two mined predictors died in one day.
- **When nothing predicts the quantity, select on the quantity** — then verify the winner *and
  the loser* on episodes that had no vote.
- **A registered prediction can be refuted by a smoke test.** Four minutes.
- **A probe that *searches* needs a cell whose answer is already known.** P41's first run
  reported 9 of 27 spawns ungraspable — including the nominal row the expert grasps at 16.4 %.
  Without that control in the table the env would have been "fixed" for a problem it did not have.
- **Derive tolerances from the noise they will see, don't pick them.** A 20 mrad check on a
  quantity whose estimator has a 43 mrad sd fails on its own noise, twice.
- **If a worst case has more than two interacting signs, enumerate it.** Four sign choices were
  hand-derived and the resulting "worst corner" landed *inside* the ordinary grid.
- **Write the decision rule before iterating on the measurement.** "A reachability wall is
  contiguous" made five runs agree on the verdict while disagreeing on which cells failed.

### On believing things

- **Ask whether the predicate encodes the task.** Scoring with the environment's own predicates
  is right and is not sufficient. It cost this project eight stages and ~5 000 episodes.
- **A taxonomy only constrains what it enumerates.** "One failure mode" was true of the buckets
  being counted; the bucket that mattered was not among them.
- **Look for the constraint that is absent.** `DISTURB_TOL` covers the four distractors and
  says nothing about the target — which is the whole basis of the current plan.
- **When a measurement contradicts the story, the story is wrong.**
- **A geometrically impossible result is a bug report, not a finding.**
- **A documented gate is not an enforced gate.** Grep the code for the gate before quoting it.
- **A correct prediction from a shaky argument is not a validated argument.**
- **Docstring geometry is not measured geometry** (§2).
- **Report the constraint metric separately from the task metric.** One arm delivered 96 % of
  targets to the goal while toppling 100 % of neighbours.
- **"Frozen" means frozen — check which parts actually are.** Freezing a grasp pose left 22 of
  23 waypoints being re-drawn every run.
- **Replay something old as a control, periodically.** Every experiment paired *within* a run,
  so a defect that varied *between* runs was structurally invisible for two stages.
- **Watch the videos early.** Sixteen episodes found in minutes what ~5 000 scored episodes did
  not.

---

## 8. Commands

```bash
source /home/eva/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_bc
```

```bash
# THE BASELINE -- the frozen expert on the randomised task. 3.0 %.
python -u clutter/act/collect_demos.py --num_envs 128 --arms appr --close 40 \
    --holds-scale 0.25 --seeds 88000,88001,88002,88003,88004,88005 --headless \
    --json clutter/runs/p42_random_row.json

# ...and on the frozen-row control. 17.1 %. Add --task for any variant.
python -u clutter/act/collect_demos.py --task Rebot-ClutterExtract-Fixed-v0 --num_envs 128 \
    --arms appr --close 40 --holds-scale 0.25 --seeds 88000,88001,88002,88003,88004,88005 \
    --headless --json clutter/runs/p42_fixed_regression.json

# P41 -- reachability gate for the randomised row, plus the per-slot close
python -u clutter/probes/p41_row_reach.py --num_envs 128 --headless \
    --json clutter/runs/p41_row_reach.json

# threshold calibration -- null action, full episode. 1 um of drift, 0/768 disturbed.
python -u clutter/probes/p35_disturb_calibration.py --num_envs 128 --steps 700 --headless \
    --json clutter/runs/p35_disturb_calib.json

# the anatomy: when / which block / which way / how much
python -u clutter/probes/p36_disturb_anatomy.py --num_envs 128 --headless \
    --seeds 88000,88001,88002,88003,88004,88005 --json clutter/runs/p36_disturb_anatomy.json

# the ablations: is it the arm or the fingers, and how far does the culprit reach
python -u clutter/probes/p38_disturb_ablation.py --num_envs 128 --seeds 88000,88001,88002 \
    --headless --json clutter/runs/p38_ablation.json

# levers, all through the collector so they share the baseline's code path
#   --yaw-gain 0      jaw stays on the nominal axis            +2.9 pts
#   --grip-open 0.023 46 mm separation  (NOT ADOPTED -- env change, Big Will said keep 90 mm)
#   --close / --holds-scale                                    hold durations, in physics steps

# the env's own smoke test -- 4 negative controls, both constraints
cd /home/eva/Desktop/isaacLab/eva_rl && python -u scripts/test_clutter_env.py --headless
```

**BC pipeline** (`act/`), unchanged and waiting for an expert worth cloning:

```bash
python -u clutter/act/collect_demos.py ... --record appr --out clutter/runs/demos.hdf5
python -u clutter/act/analyse_demos.py --data clutter/runs/demos.hdf5 --out .../audit.json
python -u clutter/act/train_flow.py --data clutter/runs/demos.hdf5 --out clutter/runs/bc_s1 \
    --steps 100000 --seed 1
python -u clutter/act/eval_flow.py --num_envs 128 --seeds 88000,88001 \
    --ckpt clutter/runs/bc_s1/ckpt_final.pt --out clutter/runs/bc_eval_s1.json
python -u clutter/act/record_video.py --ckpt clutter/runs/bc_s1/ckpt_final.pt \
    --seeds 88000,88001,88002,88003 --stills --out-dir clutter/runs/videos
```

---

## 9. Standing constraints

- eva_bc work lives in **`eva_bc/clutter/`**.
- **eva_rl's `challenge/` may be edited for the CLUTTER env only** (Big Will, 2026-08-03):
  `mdp/clutter.py`, `clutter_env_cfg.py`, `scripts/test_clutter_env.py`, the clutter docs.
  `mdp/common.py`, `mdp/rewards.py`, `mdp/terminations.py` and the other env cfgs are shared
  and out of scope.
- **Pull, commit after validating and updating the docs, then push.** Both repos, both with
  `user.name`/`user.email` set locally.
- **The 90 mm gripper stays.** It costs 17 points and it is the challenge.
- One GPU job at a time.
