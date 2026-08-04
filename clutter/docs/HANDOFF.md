# HANDOFF — ClutterExtract

**Rewritten 2026-08-03.** The success criterion changed and the expert is being started over.
Everything written against the old criterion was deleted; this file is short on purpose.

---

## 1. The task

```
Extract the target from a row of five blocks and set it down in the goal zone, WITHOUT moving
any of the four neighbours more than 2 mm from where it spawned.
```

`Rebot-ClutterExtract-v0` / `-Play-v0` / `-Tight-v0`. **Solved = ≈70 % on random spawns.**

**Where we are: 16.4 %.** That is the frozen `pose_p33` expert over 768 held-out episodes
(seeds 88000–88005), and it is a baseline to beat rather than a solution to extend.

The criterion was tightened on 2026-08-03 after Big Will spotted from sixteen videos what
~5 000 scored episodes had not: the predicate only checked whether a neighbour **toppled**
(41° of tilt), so one dragged the length of the table and set down upright scored a full
success. The same expert scored **73.3 %** under that rule. Full account, including how it
survived eight stages: **`15_STRICT_METRIC.md`**.

---

## 2. Read this in this order

| | |
|---|---|
| **`REFERENCE.md`** | ⭐ everything durable: env facts, action/obs layout, kinematics, toolchain traps, BC facts, the rules, the commands. **Start here.** |
| **`15_STRICT_METRIC.md`** | what the success criterion is now, why, and that it is calibrated |
| **`16_DISTURBANCE_ANATOMY.md`** | ⭐ **why the current expert fails, measured four ways — and the plan** |
| `06_EXPERT_DESIGN.md` | kept through the reset for §3: the gripper is binary. Carries its own retraction banner |
| `00_ENVIRONMENT.md` | machine, software, repos |
| `03_ENV_FACTS.md` | env internals: registration, rl_games traps, launching |

---

## 3. Why the current expert fails — the four measurements that matter

All from `16_DISTURBANCE_ANATOMY.md`. Under the 2 mm rule the taxonomy has **exactly one
non-zero bucket**: `distractor_disturbed` 83.6 %, with `time_out`, `target_dropped` and
`distractor_toppled` all at **0.0 %**. Toppling was never a separate failure mode — a block
must slide before it can tip.

1. **The arm is innocent.** The identical commanded trajectory with the gripper forced open
   disturbs the row **0.0 %** of the time (384 episodes, p90 0.311 mm) against 67.2 % when it
   closes normally. **The arm can go anywhere in the row safely. The only unsafe act is closing
   the jaw inside it.**
2. **It happens at the first step of the close**, and it is a jump, not a creep: p01 = 160 and
   step 160 *is* that first step; the block is already at 4.05 mm when it first exceeds 2 mm.
   This kills "close more slowly" and "shorten the close hold".
3. **The neighbour is hooked and carried, not shoved aside.** Motion is fore-aft: **|dx| is
   9.2× |dy|**. `close` starts it (81 % of first crossings), `carry` accumulates it (79 % of
   total motion). 100 % of first crossings are the inner pair, and **d1 : d2 = 2.50 : 1
   (z = 10.8)** — an unexplained chiral bias worth exploiting.
4. **The jaw is far too wide for the row.** Widening the row makes the problem vanish between
   48 and 54 mm of pitch, so the fouling reaches **33–39 mm** from the target centre against
   neighbour faces at 27 mm. The gripper opens to **90 mm to grasp a 36 mm block** — 24 mm of
   pure excess travel per finger, and `BinaryJointPositionAction` has two states so **no policy
   can choose otherwise.**

Levers already measured: jaw yaw-matching → **+2.9 pts**; narrowing the gripper to 46 mm →
**+16.9 pts**. **The gripper narrowing was NOT adopted** — Big Will, 2026-08-03: keep the 90 mm
jaw, the harder task is the better one. So 17 points are deliberately left on the table and the
90 mm sweep is a constraint to design around.

---

## 4. The plan — start over from the expert

### The idea, and why it is the one to try

Two facts, both measured, and their conjunction is the whole plan:

* **An open jaw disturbs nothing, anywhere in the row** (0.0 %, §3.1).
* **`DISTURB_TOL` binds the four distractors and says nothing about the target.** Sliding,
  dragging or tipping the *target* is entirely legal.

So the manoeuvre the constraints actually permit is:

```
1  reach in with the jaw OPEN                     measured safe: 0.0 % disturbance
2  pull the TARGET clear of the row (-x, toward the robot) with a fingertip, the closed
   jaw, or a partial close
3  close on it OUTSIDE the row, where the 24 mm sweep fouls nothing
4  carry to the goal
```

This is the extrinsic-dexterity solution the environment's own docstring describes — *"the
correct first action is often not a grasp"* — and **no expert in this project has ever tried
it.** Every one, from the first probe to `pose_p33`, closes the jaw between the neighbours.

### Order

```
1  P40  MEASURE STEP 2 ALONE.  Reach in open, drag the target -x by 30-40 mm, watch the
        neighbours.  Near 0 % -> viable and the rest is engineering.  If dragging rakes
        the row -> dead, and re-plan from there.  Cheap; do this first.
2       DESIGN THE MANOEUVRE around whatever P40 says works: where to contact the target,
        how far to drag, where the grasp pose sits once it is clear.
3       POSE + CHAIN, then freeze by PERSISTING the artifact (a CEM is not reproducible
        under a seed -- REFERENCE.md section 5).
4       MEASURE on seeds 88000-88005, 768 episodes, against the 16.4 % baseline.
5       Only once the expert is worth cloning: demos -> flow BC -> 3 seeds.  The act/
        pipeline is unchanged and working; REFERENCE.md section 6 has its constraints.
```

**Do not train anything until step 4 reports.** Cloning costs ~50 min/seed and the ceiling is
the expert's rate.

### Live questions, not yet scheduled

* **The 2.50 : 1 d1/d2 asymmetry** (§3.3) is unexplained. Something about the manoeuvre is
  chirally biased, and whatever makes `distractor_2` safer 2.5× more often is a property of the
  pose, not of the task.
* **Row orientation never varies.** `ROW_X = 0.25`, fixed pitch, per-block jitter only — the
  row never rotates or translates as a unit, so the approach azimuth is fixed and that is what
  lets a single frozen joint vector work at all. Big Will asked for this on 2026-08-03 and it
  is now a legitimate env edit. **Deferred until an expert exists** — rotating the row before
  there is a manoeuvre to rotate would be measuring nothing.
* **A diverse, multi-mode expert.** Same reasoning: it is the same piece of work as the row
  rotation, since randomising the heading is what supplies the observable that makes multiple
  modes learnable rather than merely ambiguous.
* **`-Tight-v0` has never been measured**, under either predicate.

---

## 5. What is on disk

```
clutter/
  expert/     clutter_expert.py  -- adaptation, schedule and pose machinery; reusable
              pose_p33.json      -- the frozen 16.4 % pose+chain+approach. THE BASELINE.
  probes/     _kin.py            -- the kinematics core (REFERENCE.md section 4)
              p35 calibration | p36 anatomy | p38 ablations
  act/        collect_demos.py   -- demo writer AND port gate, one code path
              dataset / train_flow / eval_flow / policy_runner / record_video
              analyse_demos.py   -- offline ambiguity audit, no simulator
              schedule_utils.py  -- expand / approach_prefix, shared so probes cannot drift
  runs/       JSON + logs are committed; checkpoints, HDF5s and videos are gitignored
  docs/       this file, REFERENCE.md, 15_, 16_, 06_, 00_, 03_
```

Probes P01–P34 and twelve stage-result documents were deleted on 2026-08-03. They were written
against the topple-only criterion and every success number in them describes a task that no
longer exists. **All of it is recoverable from git history** (`eva_bc@3e83f44` and earlier);
`REFERENCE.md` is what was worth carrying forward.

---

## 6. Standing constraints

- eva_bc work lives in **`eva_bc/clutter/`**.
- **eva_rl's `challenge/` may be edited — CLUTTER env only.** `mdp/clutter.py`,
  `clutter_env_cfg.py`, `scripts/test_clutter_env.py`, the clutter docs. `mdp/common.py`,
  `mdp/rewards.py`, `mdp/terminations.py` and the other env cfgs are shared and out of scope.
- **Pull, commit after validating and after updating the docs, then push.** Both repos.
- **The 90 mm gripper stays.**
- One GPU job at a time. `python -u`.
