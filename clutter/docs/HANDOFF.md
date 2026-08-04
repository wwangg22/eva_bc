# HANDOFF — ClutterExtract effort

**Updated 2026-08-03, mid-Stage-2.** Supersedes all earlier versions.

Read order for a fresh session:

0. **`15_STRICT_METRIC.md`** — ⚠ **the task's success criterion changed on 2026-08-03.** The
   expert scores **16.4 %**, not 73.3 %. Read this before any number anywhere else
1. **this file** — the manoeuvre, the numbers, the retractions, the plan
2. **`11_STAGE2_RESULTS.md`** — Stage 2's probes. **§2e–§2h are the newest and the most
   consequential in the whole effort** — they retract most of the P26 family and replace it
   with a measured, frozen pose ← **read early**
3. **`12_UPSTREAM_SYNC.md`** — what arrived in the 2026-08-03 pull. **eva_bc's EXP07
   x0-steering closed successfully at 91.4 %**, which retracts `10_STAGE4`'s §0
4. **`09_STAGE2_BC_PLAN.md`** — demo generation + flow BC, planned in depth from a
   line-by-line review of `eva_bc/act/`. **Supersedes §10.1 of this file.**
5. **`10_STAGE4_STEERING_PLAN.md`** — x0-steering. **Supersedes §10.4 of this file.**
6. `08_STAGE1_RESULTS.md` — probes P12–P26
7. `07_STAGE0_RESULTS.md` — probes P01–P11 (carries its own retraction banner)
8. `02_PLAN.md` — the pre-registered ladder and decision rules

⚠ `01_TASK_ANALYSIS.md` and `06_EXPERT_DESIGN.md` are **pre-measurement reasoning and are
substantially retracted** (§9). Do not build on them. `03_ENV_FACTS.md`,
`04_EVA_BC_LEARNINGS.md` and `05_PORTING_MAP.md` remain accurate except where §9 says
otherwise.

---

## 1. STATUS

> # ⚠ READ `15_STRICT_METRIC.md` FIRST. THE TASK CHANGED; EVERY NUMBER BELOW IS A DIFFERENT TASK'S.
>
> **The environment now requires neighbours to stay within 2 mm.** Big Will set the threshold
> and lifted the "do not edit `challenge/`" constraint for the clutter env on 2026-08-03, so
> the strict predicate is no longer a private re-scoring — it is `target_at_goal` itself, plus
> a `distractor_disturbed` termination. Committed to `eva_rl` as `ceeb24c`; calibrated by P35
> (null-action drift **1 µm**, so the cliff sits **2 097×** above solver noise).
>
> ```
>                                        lenient      2 mm strict
> frozen expert, 768 held-out episodes     73.3 %          16.4 %
> ```
>
> **56.9 points.** Two independent code paths agree to 0.1 (offline re-scoring 16.3 %,
> env-native 16.4 %). The seed spread is exactly binomial (ratio 1.00), so 16.4 % ± 2.6 is a
> clean estimate.
>
> **`DR2` no longer holds — the expert does not clear 70 %, so by the project's own
> pre-registered rule the ladder is re-entered a rung lower and the expert gets fixed before
> anything else is trained.** All 56.9 points are one mechanism: the finger blades sweeping the
> neighbours during the `close` phase. The taxonomy is now a single bucket —
> `distractor_disturbed 83.6 %`, every other termination **0.0 %**, including
> `distractor_toppled`, because a block must slide before it can tip.
>
> ---
>
> **The prior correction, still the right account of *why* this was missed:**
>
> Big Will watched the policy videos on 2026-08-03 and spotted what 4 600+ episodes of
> measurement did not: **the benchmark's success predicate never checked whether a neighbouring
> block was MOVED, only whether it TOPPLED.** `mdp.target_at_goal` ended in
> `& ~any_distractor_toppled(env)`, and `any_distractor_toppled` is `up_z < 0.75`, i.e. ~41 deg
> of tilt. A block dragged across the table and set down upright was a full success.
>
> Re-scored on the same 768 held-out episodes, with a strict predicate that also requires every
> neighbour to stay within 10 mm of its spawn:
>
> ```
>                     lenient    <2mm    <5mm   <10mm |  median      p90    >50mm
> expert (v3 holds)     73.3 %  16.3 %  22.4 %  30.9 % |  13.7 mm  205.5 mm  25.1 %
> BC v3 seed 1          72.0 %  19.3 %  26.3 %  34.0 % |  11.9 mm  206.2 mm  22.0 %
> BC v3 seed 2          69.7 %  17.6 %  23.4 %  30.5 % |  13.2 mm  206.8 mm  22.1 %
> BC v3 seed 3          73.7 %  18.9 %  25.4 %  33.6 % |  11.7 mm  206.3 mm  22.4 %
>                                                 ----
> BC mean               71.8 %                  32.7 %        expert 30.9 %
> ```
>
> **The ordering reverses.** Lenient: BC 1.5 points BELOW its expert. Strict: BC **1.8 points
> ABOVE** it, on every seed. BC is reproducing the expert faithfully — and the expert is the
> thing that shoves neighbours. **Fix the expert, not the policy.**
>
> **The expert falls 73.3 % -> 30.9 %, a loss of 42.4 points.** The median "success" displaces
> a neighbour by **13.7 mm — more than the 12 mm free gap the task is built around** — and in
> **22-25 % of all episodes a neighbour is CARRIED TO THE GOAL ZONE** with the target (the
> p90 of ~206 mm matches the 120-277 mm row-to-goal distances almost exactly).
>
> The evidence was in the record the whole time and was never connected to the metric: every
> hazard table since P17 reports a **`close`-phase disturbance rate of 71-76 %**.
>
> **The real success rate of everything this effort has built is about 31 %, not 73 %.**

## 1.0 The numbers as previously reported — all under the lenient predicate

**The expert is a frozen artifact scoring 72.1 % on never-used spawns** —
`expert/pose_p33.json`, loaded with `ClutterExpert(pose_q=…, chain=…)`. Central estimate
across nine 512-episode measurements: **≈73 %**. 100 % enclosure, ~90 % at-goal, ~20 % topple.

That replaces "74.6 % over 768 episodes", which was one 3-draw sample of a distribution with a
±12-point interval. The number went down; the **confidence went up a great deal**, and for the
first time the number is attached to a specific reproducible artifact rather than to a
configuration that re-draws its own manoeuvre on every run.

| gate | threshold | result | verdict |
|---|---|---|---|
| Gate 1 (pre-registered) | ≥85 % nominal | 72.1 % | **not met** |
| Gate 1 revised floor (`DR2`) | ≥70 % | 72.1 % | **met** |
| **Gate 2a/2b (the `env.step` port)** | each pair within ±5 pts | **−0.2 / +0.6 / +0.6** | **PASSED** |
| **Gate 2 (BC policy)** | ≥60 % pooled / ≥58 % band | **71.8 % mean** lenient / **~32 %** strict | **passed leniently; UNKNOWN strictly** |
| **Mission (for the *policy*)** | ≈70 % | 71.8 % lenient / **~32 %** strict | **met leniently; NOT met strictly** |

⚠ Both gate thresholds were defined against the lenient predicate. **They need restating —
Big Will's call.** My recommendation: adopt the strict predicate at a threshold he chooses,
treat every lenient number as historical, and re-baseline. See `14_FEEDBACK_AND_NEXT.md` §4.

**The port to `env.step` is done and it is free.** Under the full MDP, from the reset pose,
with an action-driven approach, the frozen expert scores **73.6 %** over 512 episodes — against
72.7 % for the physics-only reference on the identical spawns. Full account in
`13_STAGE2_BC_RESULTS.md`; it cost one real defect (§9, R19–R21) to get there.

`DR2` applies: **proceed to Stage 2, recording the BC ceiling in advance** (§10.1).

The residual failure mode is identified and unfixed: **the finger blades sweep the
neighbouring blocks during the close, independently of the target block entirely** (§5).
Every other phase of the trajectory is at zero hazard.

### 1.1 Where Stage 2 actually is

| item | state |
|---|---|
| `09_STAGE2_BC_PLAN.md` — flow-BC plan, 8 clutter-specific findings | **written** |
| `10_STAGE4_STEERING_PLAN.md` — x0-steering plan | **written**; §0 retracted after the upstream pull |
| `12_UPSTREAM_SYNC.md` — the 2026-08-03 pull review | **written** |
| **P28** IK branches / **P30** the π roll | **run.** 6 clusters not 1; roll worth **+7.0** paired |
| **P26-v2 / v3 / v4** | **run.** 66.5 / 74.6 / **62.8 %** — all inside their own ±12-pt intervals |
| **P32** forced wrist side | **run.** Hypothesis **REFUTED** (+2.8 ± 8.3). Found **ICC = 0.82** |
| **P33** pose tournament | **run.** Winner **verified 75.4 %** on 512 fresh eps, +18.8 over the mean |
| **P34** frozen-pose reload control | **run ×3.** Found the chain was never frozen; now it is |
| `expert/pose_p33.json` + `pose_q=` / `chain=` / `_score_q` / `_rng_seeded` | **written, run, verified** |
| **Q7** throughput | **run.** 4 096 envs in 3.2 GiB; 2 048 at 34.5 k env-steps/s |
| **P29** approach segment | **run ×2.** Prediction **REFUTED**; found the branch seam; fix ships at 73.0 % |
| **Gate 2a / 2b** the `env.step` port | **run. PASSED** — −0.2 / +0.6 / +0.6 pts, flips at the ~8 % noise floor |
| `clutter/act/` — collector, dataset, trainer, eval, audit | **written and run** (§10.1a; details `13_STAGE2_BC_RESULTS.md`) |
| `runs/demos_v1.hdf5` | **753 successful demos** from 1 024 episodes (73.5 %), 472 steps each |
| demo audit — **N2 measured offline** | **run. N2 CONFIRMED**: the `close` hold carries **54 %** of the chunk ambiguity, at ratio 1.005 |
| flow-BC on `demos_v1`, 3 seeds | **58.1 / 68.5 / 77.7 %** — mean 68.1, **sd 9.8**; one seed fails to start 9.5 % of episodes |
| **P27** hold duration | **RUN.** Close flat 560 → 40 phys steps, enclosure 100.0 % throughout; shortening the other holds gains the **expert** +3.9 |
| `runs/demos_v3.hdf5` | **774 demos**, 325 steps, static 10.2 %, ambiguity floor 0.0607 (v1: 0.0989) |
| flow-BC on `demos_v3`, 3 seeds | **69.7 / 72.0 / 73.7 %** — mean **71.8**, **sd 2.0**, expert 73.3 % on the same spawns |
| training-loss → success | **measured: no useful relationship.** A lower-loss seed scored 10.4 pts worse |
| eval determinism / policy sampling variance | **measured. 0.4 pts** over three noise sequences; `env.reset(seed=)` pins torch, so checkpoint comparisons are exactly paired |
| **P27** hold duration | **edited for the frozen pose, NOT RUN** — the lever on the one remaining failure mode |
| **P24** yaw gain | **written, NOT RUN** — still needs the frozen-pose edit |

**Next action: finish the seed replication, then P27.** The taxonomy has collapsed the problem
to a single question — the policy topples 30.7 % against the expert's 18.8 %, and everything
else is at zero — so every remaining lever should be aimed at the close phase.

### 1.2 The one-paragraph summary of this session's science

The session set out to close a gate leak and instead **invalidated the method that produced
every expert number so far, then replaced it.** P26 compared *different poses* across arms and
paired nothing; the pose draw carries a **12-point sd**, so with 3 selections per arm every
P26 verdict had a ±12-point interval and sat inside its own noise — including "screening is
worth +10", "screening both rolls is worse", and "v4 is worse than v3". Hunting for a forward
statistic that predicts pose quality failed for the **fifth** time, and one candidate
(`wrist side`, +13.0 pts observational, Fisher *p* = 0.0056) was **refuted** by the
properly-powered test it motivated, as was its successor (`|wrist_y|`, r = −0.804 → −0.223).
But P32's batch-paired design also showed the variance is *real*: **ICC = 0.82**, a pose's
score reproducing on independent spawns to within the binomial floor. So P33 stopped
predicting and **selected**: 8 candidates ranked on shared batches, winner re-measured on 512
episodes that had no vote — **+18.8 points**, ranking confirmed at both ends. P34 then found
that freezing the pose does not freeze the *manoeuvre* — `_dense` draws its 22 other waypoints
from a CEM every run, and a seed cannot fix it because 1 320 GPU reductions are not bit-stable
— so the chain is now persisted, not re-derived. **Net: the expert's headline fell 74.6 → 72.1,
and it became a reproducible artifact with a measured out-of-sample score.**

Nothing has been committed or pushed. `git status --porcelain` in `eva_bc` shows only
`?? clutter/`.

⚠ **Before any push:** the repo-root `.gitignore`'s bare `runs/` rule swallows
`clutter/runs/` — 72 files, 6.1 MB, every result JSON and log this effort has produced. See
`12_UPSTREAM_SYNC.md` §6 for the measured options and the exact commands. **Big Will's call.**

---

## 2. MISSION AND STANDING CONSTRAINTS

Train a policy solving **`Rebot-ClutterExtract-v0`** (with `-Tight-v0`, `-Play-v0`) at **≈70 %
success on random starts**, using the **eva_bc** staged pipeline: scripted expert →
flow-matching chunk BC → batched sim eval → DAgger → RL via x0-steering on a frozen base.

- All work lives in **`eva_bc/clutter/`**. Nothing outside it has been modified.
- **Do not modify eva_rl's `challenge/` package.** It is the benchmark under test; env
  properties are findings to *report*, not defects to fix.
- **Push only when Big Will asks.** `runs/` and `__pycache__/` are gitignored, so the tracked
  deliverable is 11 docs + 1 expert module + 27 probe files.
- **One GPU job at a time** (10 GiB card). Check with `nvidia-smi` and `pgrep -f "python.*p2"`.

---

## 3. THE MANOEUVRE THAT WORKS

Implemented in **`clutter/expert/clutter_expert.py`**, which carries the probe reference for
every constant. Each line was paid for by a failure.

```
opening axis      o_hat = x̂            ORTHOGONAL grasp: fingers straddle the target FORE
                  (phi = 90°)           AND AFT at x ≈ 205 / 295 mm, |y| < 5 mm. They never
                                        enter a 12 mm row gap. This is the whole reason the
                                        task is solvable at all.                      (P11)

grip height       z = 0.055 m           A value that WORKS, not a tuned optimum. The clean
                                        9-height sweep is nearly flat (19.5–31.5 % topple
                                        over 40–80 mm) and the within-height spread across
                                        pose draws is just as large.                  (P23)

pose selection    1. hard gate          TCP error and o_align pin the TOOL FRAME and nothing
                     pos_err < 1.5 mm   else. A pose scoring 0.55 mm / o_align 1.000 put the
                  2. penetration = 0    wrist inside distractor_2 and collapsed enclosure to
                  3. o_align ≥ 0.99     32 %. Lexicographic, NEVER a weighted sum — a sum
                     ON EVERY CANDIDATE lets 0.003 of o_align buy a 10 mm intrusion.
                  4. SCREEN 4 IN SIM    The FK gates are necessary and nowhere near
                     on enclosed∧¬topple sufficient; the sim screen is worth +10 points and
                                        cuts pose variance from sd 17.5 % to 4.8 %.
                                        ⚠ The gate in step 3 LEAKED in three places until
                                        Stage 2 and let o_align 0.86–0.98 poses ship.
                                                                    (P13/P15/P25/P26/P28)

wrist roll        j6 SIGN MATCHES       A pi roll of joint6 leaves TCP (0.00 mm), both axes
                  THE WRIST'S y SIGN    (1.00000) and keep-out penetration IDENTICAL — the
                                        finger origins merely swap — and is worth 7.0 POINTS
                                        end to end, because the collision MESHES rotate with
                                        the bodies. 12 of 12 screen pairs pick the roll whose
                                        sign(j6) matches sign(wrist_y). Free; do NOT screen it
                                        (that doubles the pool and overfits: 66.5 %).   (P30)

path              DENSE CARTESIAN       ≤30 mm waypoint spacing, each solved LOCALLY
                                        (restarts=1, std0=0.08) so the IK branch cannot
                                        change between waypoints. THE SINGLE LARGEST FIX
                                        IN STAGE 1.                                   (P17)

yaw               MATCHED PER ENV       through the grasp AND the lift. refine() gained an
                                        o_des channel. Block turn during the close
                                        3.72° → 0.28°. (Possibly a net negative — see §10.2)
                                                                                      (P16)

lift              +150 mm               The block hangs 68 mm below the TCP, so the old
                                        75 mm lift left its bottom face 5 mm above the
                                        distractor tops, on a path running diagonally over
                                        distractor_0 and distractor_1 to the goal.    (P12)

withdraw          VERTICAL AT THE GOAL  The old retreat drove back across the row with an
                                        empty gripper: 98.4 % contact hazard, for a motion
                                        the task never asked for.                     (P17)

close             PLAIN BINARY          Ramping the finger target scores 97.7 % and is NOT a
                                        legal action (BinaryJointPositionActionCfg has no
                                        intermediate aperture). Duty-cycling IS legal and is
                                        measurably worse.                          (P19/P20)

execution         PHYSICS-ONLY          hold_phys / run_phys for all measurement. env.step
                                        runs the MDP, so a topple resets the scene and
                                        re-spawns the blocks upright — the evidence erases
                                        itself.                                (Stage 0 §7.5)
```

### 3.1 Why the orthogonal grasp works

With `o_hat = x̂` the fingers sit in front of and behind the row, on the target's own
centreline. The row's 12 mm y-pitch — the stated difficulty of the task — stops being the
binding constraint. The block is 36 mm deep in x, comfortably inside the 89 mm jaw opening.

This was Big Will's suggestion. I had dismissed it by argument in Stage 0 and was wrong; see
§9.

### 3.2 Why the wrist has to thread a gap

A parallel jaw's approach axis is perpendicular to its opening axis, so `o_hat = x̂` confines
`a_hat` to the y–z plane. The wrist stub trails the TCP by a fixed 41.9 mm along it:

```
gripper_end = TCP − 0.0419 · a_hat = (250, −41.9·sin t, grip_z − 41.9·cos t)   [mm]
```

Parking the wrist above the row needs `cos t < 0` — a **downward** approach axis. P14 swept
the full circle at the actual grasp pose and found it genuinely unattainable: at
`a_des = (0,0,−1)` the CEM achieves `a_align = −0.11`, and nothing past `t = 105°` is reached.
`CHALLENGE_SUITE` C1 is correct where it matters.

So the wrist sits in one of the 12 mm gaps, at |y| ≈ 21 mm, z ≈ 18–20 mm. P25 confirmed the
alternative is worse: raising the grip until the wrist clears the row forces a near-horizontal
approach axis, which lays the finger blades flat — enclosure 19–27 %, success **2.0 %**.

⚠ **Corrected 2026-08-03.** This section used to say the wrist "is put there **deliberately**
and verified by the keep-out term, not left to chance." Half of that is true. Over eight
independent solves the wrist went to the **+y gap five times and the −y gap three times**
(P28), and nothing in the search prefers either — the keep-out term merely **verifies whichever
gap it lands in**. The side is *drawn*, not chosen. Harmless for the expert; a defect in the
data-generating process for BC, since the two families are mirror images producing opposite
joint signs for observations that differ only by the row's own ±5 mm of jitter.
`ClutterExpert(wrist_side=±1)` now locks it.

---

## 4. THE CODE

```
clutter/
├── docs/            11 markdown files, ~3 700 lines
├── expert/
│   ├── __init__.py
│   └── clutter_expert.py    THE DELIVERABLE
├── probes/
│   ├── _kin.py              shared FK / CEM / execution core
│   └── p01…p26              one question each, self-contained, runnable alone
└── runs/                    logs + JSON (gitignored)
```

### 4.1 `probes/_kin.py` — the kinematics core

No IK solver, no motion planner. **Candidates are scored by forward kinematics read back from
the sim**, so a search cannot silently converge on an unexecutable pose.

| member | purpose |
|---|---|
| `fk(q_arm)` | writes joints, `sim.forward()`, returns achieved TCP / `a_hat` / `o_hat` / `low_z` / **all body origins** |
| `cem(pos, seed, …)` | FK-scored cross-entropy search over the 6 arm joints |
| `box_penetration(bodies, boxes, margin)` | deepest intrusion of any body origin into any keep-out box — **added in Stage 1** |
| `cem(…, avoid=boxes)` | prices that penetration into the cost — **added in Stage 1** |
| `refine(q0, pos, o_des=…)` | per-env damped-least-squares correction; the `o_des` orientation channel is **new in Stage 1** |
| `hold_phys` / `run_phys` / `teleport_arm` | physics-only execution that bypasses the MDP |
| `gap()` / `tcp_now()` / `finger_pos()` | read-backs; ground truth for "did it close on it" |

**The CEM cost is a constrained form, not a weighted sum**, because a weighted sum failed in
both directions: `w_pos = 1` let orientation outrank position and the search walked 340–520 mm
away; `w_pos = 20` made orientation noise and returned flipped wrists. Position now enters as a
**hinge** (free within 1 mm, then 0.2/mm), the floor and keep-out terms have the same shape, and
axis terms are bounded tiebreakers.

Constants that must never drift:

```python
TCP_OFFSET = (-0.0419, 0.0, 0.0)   # measured, CHALLENGE_SUITE C10. NEVER -0.075 or -0.048.
Q_OPEN, Q_CLOSE = 0.045, 0.0       # binary gripper: 89.07 mm clear gap, or shut
GAP_A, GAP_B = 1.0035, 0.00125     # gap = 1.0035·(q_L+q_R) − 1.25 mm, resid 0.035 mm
```

### 4.2 `expert/clutter_expert.py` — the Stage-1 deliverable

```
ClutterExpert(env, grip_z=0.055, lift_dz=0.150, phi=90, screen=4,
              roll_mode="rule", wrist_side=0, holds=None, …)

  .plan()           solve the nominal chain ONCE:
                      _gated_solve  -> pos_err / penetration / o_align / wrist-side gates
                      _canon_roll   -> pick joint6's roll by the 12/12 rule       [Stage 2]
                      _screen       -> 4 candidates tried in the sim
                      _dense        -> Cartesian polyline outward BOTH ways from the grasp
  .adapt()          bind the chain to THIS reset: per-env DLS refine to the target's actual
                    position and yaw, through the grasp AND the lift
  .schedule(chain)  the motion as (phase, kind, args) — the SINGLE SOURCE OF TRUTH, so the
                    physics-only evaluator and a future env.step demo recorder are
                    checkably equivalent
  .env_steps(chain) the schedule's length in ENV steps, per phase                 [Stage 2]
  .run_physics()    execute physics-only, return the env's own success predicates
```

`plan_full=False` solves the grasp pose only and skips the chain — used by probes that study
the close in isolation, which makes 2-D sweeps affordable.

**New in Stage 2**, all additive with Stage-1-identical defaults except `roll_mode`:

| member | purpose |
|---|---|
| `_wrist(q)` | `gripper_end` origin in the env frame — the wrist-side test |
| `_rolls(c)` | `c` plus its `joint6 ± π` twin, when the twin is inside the joint limits |
| `_canon_roll(c)` | picks the roll by `sign(j6) == sign(wrist_y)` — **12/12, free** (P30) |
| `_gated_solve(grip, attempts)` | the gates the recipe *claims*, enforced on **every** candidate; falls back to **best alignment**, never highest wrist |
| `roll_mode` | `"rule"` (default) / `"screen"` (measured worse) / `"off"` (Stage 1) |
| `wrist_side` | 0 = either, ±1 = lock the wrist to one 12 mm gap |
| `holds` | hold durations in **physics** steps; `HOLDS` is the default dict |
| `env_steps(chain)` | schedule length in env steps, per phase + `TOTAL` + `STATIC` |

### 4.3 The eva_bc pipeline, as reviewed

Nine files, ~1 600 lines, read end to end. Full notes in `09_STAGE2_BC_PLAN.md` §1; the
load-bearing points:

| file | what matters |
|---|---|
| `dataset.py` | every `(demo, t)` is one sample; `action_is_pad` carries **two** meanings (past-end **and** `train_mask == 0`); group names are **parsed**, not sorted |
| `normalize.py` | mean/std lives **outside** the policy; stats ride in the checkpoint as buffers |
| `modeling_flow.py` | rectified flow, 703 k params; `key_padding_mask = pad & ~pad.all(dim=1,keepdim=True)` is the all-masked-row NaN guard |
| `eval_act.py` | `_buf (N, n_action_steps, 7)` / `_idx (N,)`; **`steer_x0` outranks `fixed_x0`**; no clamping at inference |
| `steer_core.py` | the controller stays **free-running**; z enters only via the x0 of refills that happen while it is held |
| `steer_wrapper.py` | one rl_games step = one 15-step window; reward is an **undiscounted sum**, so `gamma` discounts per *window* |
| `residual_core.py` | the 15-D shared feature tail; **`:159` garbles a quaternion — drop it** |

**The x0 contract** (`modeling_flow.py:117`) that every later stage depends on:

```
x0.dim() == 2  ->  (chunk_size, action_dim)      broadcast over the batch   [fixed_x0]
x0.dim() == 3  ->  (B, chunk_size, action_dim)   per-env                    [steer_x0]
```

---

## 5. THE OPEN PROBLEM — the finger blades

**P22 is the probe to read.** Teleport the target 2 m below the table, leave the distractors
where they spawned, put the arm in the identical grasp pose, close on empty air:

| arm | neighbour moved >1.5 mm | toppled |
|---|---|---|
| target present, normal close | 74.2 % | 19.5 % |
| **target REMOVED, same close** | **75.8 %** | **21.9 %** |
| target present, jaw kept OPEN | **0.0 %** | **0.0 %** |

**The target is a bystander.** Removing it changes nothing; holding the jaw open changes
everything.

The collision meshes, read at last through `Usd.TraverseInstanceProxies` (17 298 points per
finger — the measurement missing since P01's retracted AABB):

```
gripper_left    x −19.2 .. +19.2 | y −41.9 .. +46.7 | z −58.7 .. +34.7   [mm, body frame]
gripper_right   x −19.2 .. +19.2 | y −46.7 .. +41.9 | z −58.7 .. +34.7
gripper_end     x −157.2 .. −73.2 | y −92.0 .. +92.0 | z −41.0 .. +40.9
```

The blades reach **~47 mm along the opening axis** and only **±19.2 mm perpendicular to it**.
With the jaw open the finger origins sit at x ≈ 205 / 295 mm — **outside** the row's
232–268 mm x-band, which is exactly why the descent hazard has been 0 % since P15. Closing
drags them 26.5 mm each **into** that band, where the neighbours live at y = ±42 mm with faces
at ±27 mm. Margin: **7.8 mm.**

This is the volume the gripper sweeps **while actuating**. No waypoint, path, or pose change
touches it — which is why it survived every fix in Stage 1.

---

## 6. THE HEADLINE NUMBERS

### 6.1 Expert on random spawns — **the frozen P33 pose**, 512 never-used episodes

| metric | Stage 1 (P26-v1) | **current — `pose_p33.json`** |
|---|---|---|
| enclosed at close | 100.0 % | **100.0 %** |
| target at goal | ~90 % | ~90 % |
| topple | ~22 % | ~20 % |
| **SUCCESS** | 73.7 % ± ~12 (3 draws) | **72.1 %** on fresh spawns; ≈73 % central |

⚠ **The old row is not a like-for-like comparison and the arrow does not mean the expert got
worse.** "73.7 %" was the pooled mean of *three pose draws* from a distribution whose
selection-level sd is **12 points**, so its own 95 % interval was ≈ [68.6, 78.8] and its
replications ran 62.8–74.6 %. The new number is **one specific frozen pose and chain**,
measured on 512 episodes that had no part in choosing it, reproducible from a file.

The right comparison is against the thing it replaces: a *freshly drawn* pose averages
**56.6 %** (P33's 8-candidate mean). The frozen pose is **+16 points** on that.

Full derivation in `11_STAGE2_RESULTS.md` §2e–§2h.

Per-phase hazard (contacts, over the envs that reached the phase still clean): every phase is
at **zero** except **`close`**, which is the sole remaining mode.

Stratified by minimum free gap — **success is HIGHEST where the row is TIGHTEST**, the reverse
of the Stage-0 reading:

| min free gap | n | success |
|---|---|---|
| 0–4 mm | 30 | **90.0 %** |
| 4–6 mm | 82 | **91.5 %** |
| 6–8 mm | 233 | 64.4 % |
| 8–10 mm | 303 | 71.0 % |
| 10–14 mm | 120 | 82.5 % |

**Working hypothesis, fitted after the fact and flagged as such: a tightly packed row supports
itself.** Blocks are 70 mm tall with `h_crit = 15.8 mm`; a nudged block with 2 mm of room leans
on its neighbour and stops, one with 10 mm accelerates past the 41.4° termination. Toppling
needs somewhere to fall. **This predicts `Tight-v0` (6 mm pitch) may be *easier* than nominal.**

### 6.2 The five trajectory fixes: 25 % → 57.7 %

| fix | effect | probe |
|---|---|---|
| **dense Cartesian pathing** | 18 % → 62.5 %, paired net **+57 / 128** | P17 |
| whole-arm clearance in the pose search | at-goal 28 % → 100 %; descent contacts 34 → **0** | P15 |
| lift 75 → 150 mm | removed the `carry` phase, which held 48.4 % of topple onsets | P12 |
| yaw matching | block turn during close 3.72° → 0.28° | P16 |
| withdraw at the goal | retreat hazard 98.4 % → 0 % | P17 |

### 6.3 Pose selection — the P26 family, and why it is superseded

All P26 arms: 3 selections × 2 held-out batches × 128 envs = 768 episodes.

| arm | pooled | **95 % CI of the run mean** | selection sd |
|---|---|---|---|
| `screen = 0` | 63.7 % | [59.0, 68.4] | 4.1 % |
| `screen = 4`, random roll (v1) | 73.7 % | [68.6, 78.8] | 4.8 % |
| `screen = 8`, random roll (v1) | 67.2 % | [54.8, 79.5] | 10.9 % |
| `screen = 4` × both rolls (v2) | 66.5 % | [51.9, 81.2] | 12.9 % |
| `screen = 4`, roll by rule (v3) | 74.6 % | [65.3, 83.9] | 8.2 % |
| `screen = 4`, + gate fix (v4) | 62.8 % | [47.5, 78.0] | 13.5 % |

**Five of the six intervals overlap.** The pooled selection-level sd is **10.1 points**;
resolving a 5-point effect at that sd needs **63** selections per arm and every arm used
**3**. The whole family — 4 608 episodes — is consistent with one underlying rate ≈ 68 % plus
pose-draw noise. Earlier versions of this section drew four conclusions from these numbers;
all four are retracted in §9.

**What replaced it.** P32 showed the variance is real and *stable* — **ICC 0.82**, true pose
sd **12.0 points**, a pose reproducing on independent spawns to within the binomial floor. So
P33 selected by measurement instead of by statistic:

| | |
|---|---|
| freshly drawn pose (8-candidate mean) | **56.6 %** |
| P33 winner, selection score (256 eps) | 73.4 % |
| P33 winner, **verified on 512 eps that had no vote** | **75.4 %** |
| same pose, frozen chain, never-used spawns (P34) | **72.1 %** |
| selection optimism, after removing a common batch offset | ≈ +1.8 pts |

**+16 points over a fresh draw, and reproducible from a file.**

### 6.4 Measured environment facts

| fact | value |
|---|---|
| finger blade reach | **~47 mm** along the opening axis, **±19.2 mm** perpendicular |
| block / table / effective friction | μ_s 0.9 / **1.0** / **0.95** (PhysX averages, not 0.9) |
| `h_crit` across row (b=30) / along row (b=36) | 15.8 mm / 19.0 mm |
| arm's approach axis at home | `a_hat·x̂ = −0.98` — **fingers point BACK toward the robot** |
| downward approach axis at the grasp pose | **unattainable**, `a_align = −0.11` |
| measured min free gap, pooled | 2.6–13.0 mm, median ~8 (doc claimed 7.0–18.8) |
| blocks settle at | z = 32 mm, **not** 35 — shifts every height gate |
| row layout | distractors y = ±42, ±84 mm; target y = 0; all x = 250 mm |
| goal | (185, −185) mm — the carry runs **over** distractor_0 and distractor_1 |
| spawn jitter | target x ±12 mm, **yaw ±0.20 rad**, no y; distractors x ±10, y ±5 mm |
| physics dt / decimation | 2.5 ms / 8 → one legal action lasts 20 ms |
| gripper | **binary only**: 89.07 mm or 0. No intermediate aperture, no rate limit. |

---

## 7. EVERY EXPERIMENT, AND ITS VERDICT

### Stage 0 — P01–P11 (full detail in `07_STAGE0_RESULTS.md`)

| probe | question | verdict |
|---|---|---|
| `_kin` | shared FK / CEM core | working; the seed of the expert |
| `p01` | finger extents, blade width, friction | friction answered; **geometry retracted**, replaced in P22 |
| `p02` | attainable grasp orientations | **VOID** — ran on the broken CEM; superseded by P11 |
| `p03` | control grasp: solo / clutter / pre-singulated | solo 100 % held, clutter 0 %, gap+25 mm 0 % |
| `p04` | global sample of the reach set | home pose read; "0 samples" was a density artifact |
| `p05` | commanded-vs-achieved trace of one grasp | found the standoff and lift-planning bugs |
| `p06` | lateral clearance needed | 31 mm first success, 48 mm reliable |
| `p07` | can the fingers wedge the row apart | **no** — 4.2 mm against ~31 mm needed |
| `p08` | depth clearance needed | 65 mm, sharp transition 62→65 |
| `p09` | can a distractor be pushed aside | **no** — every reachable contact topples it |
| `p10` | end-to-end on random spawns | 25 % success, 99 % at goal, 75 % topple |
| `p11` | **the orthogonal grasp** | **100 % held, 0 % topple on the nominal row** |

### Stage 1 — P12–P26 (full detail in `08_STAGE1_RESULTS.md`)

| probe | question | verdict |
|---|---|---|
| `p12` | which body hits which block, in which phase | **48.4 % of topple onsets in `carry`**, victims d0/d1; carried block clears the row by 5 mm |
| `p13` | paired lift-height sweep | **INVALID** — ran on a pose with 32 % enclosure. Built the paired snapshot/restore harness, which everything after uses |
| `p14` | which approach axes exist, and where does the arm go | **a downward axis is unattainable** (`a_align = −0.11`). Its "row-clear" poses were inside the *target*, which its keep-out set omitted |
| `p15` | does whole-arm clearance fix the descent | **yes** — descent contacts 34 → **0**, at-goal 28 % → **100 %** |
| `p16` | yaw matching and retreat removal | mechanisms confirmed (turn 3.72° → 0.28°) but **no headline effect** — `place` had a 100 % hazard masking everything |
| `p17` | **is the path between waypoints valid** | **NO. `carry→place` deviated 108 mm — an IK branch flip. 18 % → 62.5 %** |
| `p18` | is the wrist stub the culprit in the close | "yes" — **RETRACTED**, the mirror control was confounded |
| `p19` | closing-slam dynamics at 2.5 ms resolution | wrist travels **0.13 mm**; ramped close = 97.7 % but **not policy-legal** |
| `p20` | policy-legal replacements for the ramp | duty-cycling is **worse**; grip 55 mm looked best (single cell, later shown to be noise) |
| `p21` | **Gate 1 measurement** | **57.7 % pooled over 768 episodes, sd 13.6 %** |
| `p22` | **who actually hits the neighbours** | **the finger blades, directly. The target is a bystander.** |
| `p23` | grip height on the isolated mechanism | **nearly flat** (19.5–31.5 %); first run had a probe bug caught by its own impossible output |
| `p24` | yaw gain × phi | **WRITTEN, NOT RUN.** Prediction pre-registered (§10.2) |
| `p25` | put the wrist above the row | **falsified, −56.2 points.** Clearing the row forces a flat jaw: enclosure 19–27 %, success 2.0 % |
| `p26` | **screen candidate poses in the sim** | 73.7 % vs 63.7 %, **but the "+10.0 points" is RETRACTED** (§9 item 5) — four replications of the same arm span 12 points |

### Stage 2 — P26-v2…v4, P27–P34, Q7 (full detail in `11_STAGE2_RESULTS.md`)

| probe | question | verdict |
|---|---|---|
| `p27` | how short can the holds be? | **RUN, and the best result of the stage.** The close is **flat 560 → 40 physics steps** with **100.0 % enclosure at every duration** — 70 env steps spent on something needing at most 5. And shortening the *other* holds makes the expert **+3.9 points better** (72.1 → 76.0) while cutting the demo 394 → 247 steps and its static fraction 45.7 % → 13.4 %. Expected to be a trade; it is not one |
| `p28` | **do independently screened poses agree?** | **NO — 6 clusters from 8 draws.** Prediction falsified. Found the wrist-side draw, the `o_align` leak, and the `joint6` alias |
| `p29` | the `home → chain[0]` approach segment | **RUN ×2. Prediction REFUTED.** A joint lerp buries a finger 4.85 mm in `distractor_1` (5.5 %); a *geometrically perfect* forward Cartesian solve scores **0.0 %** because it lands **1.90 rad** from the frozen chain on `joint6`. Solved **backward** from `qs[0]`: **73.0 %** vs a 74.2 % teleport baseline. Also: `\|a\|max = 4.63` on **`joint4`**, 6/6 joints outside `[−1,1]` |
| **Gate 2a/2b** | does the manoeuvre survive `env.step`? | **PASSED.** `phys` 72.7 → `tele` 72.5 → `tele0` 73.0 → `appr` **73.6 %**, 512 eps/arm, all deltas ≤0.6 pts, flips 8.4–9.6 % (P34's noise floor). `target_dropped` and `time_out` **never fire** — topple is the only failure mode |
| **demo audit** | is the correct chunk determined by the observation? | **N2 CONFIRMED.** `close` ratio **1.005** — the correct chunk is as unpredictable from a near-identical observation as from none — nearest same-demo neighbour a median **20 steps** away in a 70-step hold. **54 %** of the ambiguity is there |
| **flow BC, seed 1** | does it transfer? | **68.5 %** over 768 held-out eps vs **71.4 %** for the expert on the identical spawns — **−2.9 ± 3.4 pts**, where eva_bc lost 21.6. Taxonomy: topple 30.7 %, everything else ≤0.5 % |
| `p30` | **is a π roll of `joint6` a symmetry?** | **NO — worth +7.0 points**, paired, 384 eps/arm, 3/3 draws. Prediction held. **The one P26-era result pairing protects** |
| `p31` | grasp-bit threshold rule | **NOT WRITTEN** — needs a trained base first |
| `q7` | throughput / VRAM, N = 16…4096 | **RUN.** 4 096 envs in **3.2 GiB**; 2 048 at **34.5 k env-steps/s** ≈ 3× the machine eva_bc used. Neither memory nor speed constrains anything |
| `p26-v2` | screen **both** rolls (pool 4 → 8) | 66.5 % — read as "WORSE" at the time; **inside the noise** (§9 item 6) |
| `p26-v3` | roll chosen by **rule**, pool back to 4 | 74.6 % — read as "buys nothing"; **inside the noise** (§9 item 7) |
| `p26-v4` | + the third `o_align` gate fix | **62.8 %**, CI [47.5, 78.0]. A non-result from an underpowered run, and the trigger for the meta-analysis |
| `p32` | **force the wrist side** | **HYPOTHESIS REFUTED: +2.8 ± 8.3**, 4 608 batch-paired episodes. But measured **ICC = 0.82** — pose quality is stable and therefore selectable |
| `p33` | **pose tournament, 8 candidates** | **Winner verified 75.4 % on 512 episodes that had no vote, +18.8 over the candidate mean.** Ranking held at both ends. `\|wrist_y\|` lead refuted prospectively |
| `p34` | frozen-pose reload control | **Ran 3×.** Found the chain was **never** frozen (`_dense` redraws 22 waypoints; a seed can't fix it). Chain now persisted. Pose scores **72.1 %** on fresh spawns |

---

## 8. WHAT DIDN'T WORK, AND WHY

Each of these is a **measured negative**. Do not revisit without new evidence.

| attempt | why it failed |
|---|---|
| **Lateral singulation** (push a neighbour aside) | Requires touching an **inner** face; every inner face is inside a 12 mm gap and the closed gripper is ~38.6 mm across. A lateral push can *compress* this row, never *spread* it. |
| **Wedging the row apart** | 4.2 mm symmetric spread against ~31 mm needed. Pressing harder **tilts** the neighbours (`up_z` 0.934 → 0.765, threshold 0.75) instead of translating them. |
| **Pushing a distractor along +x** | Contact is possible only at z 58–80 mm against `h_crit = 19.0 mm`, so every reachable contact topples it. 5× the push speed changed nothing. |
| **Duty-cycling the binary gripper** | Legal, and measurably *worse* than a plain close: 29.7 % and 24.2 % against 39.8 %. |
| **Ramping the finger joint target** | Scores **97.7 %** and is **not a legal action** — `BinaryJointPositionActionCfg` has no intermediate aperture. A demo containing it teaches a policy something it cannot emit. |
| **Mirroring the wrist to the +y gap** | The pose exists but cannot hold the block: `o_align` 0.885, at-goal 21 %. |
| **Raising the wrist above the row** | Requires `cos t < 0.31`, i.e. a near-horizontal approach axis, which lays the blades flat. Enclosure 19–27 %, success **2.0 %**. **−56.2 points** pooled by measured wrist height. |
| **Predicting a good pose from forward kinematics** | **FIVE tries, all failed**: `o_align` (necessary, nowhere near sufficient; r² 0.059 and *negatively* signed on paired data), wrist height (actively **anti**-correlated, −56.2 pts), screen score (r² 0.016, though range-restricted), **wrist side** (+13.0 pts observational → **+2.8 ± 8.3 under assignment**, P32), **\|wrist_y\|** (r = −0.804 observational → **−0.223** prospectively, P33). The working conclusion is that no cheap forward statistic will, and that **measurement is the selection mechanism** — see P33. |
| **Mining a predictor out of the pose dataset** | Twice in one day. Both looked strong observationally (Fisher *p* = 0.0056; r² = 0.646) and both died in the properly-powered prospective test they motivated. ~8 000 episodes to kill two false mechanisms — cheap, but they should have been registered as *leads*, not written up as findings. |
| **Tuning grip height** | The clean 9-height × 3-draw sweep is nearly flat; the within-height spread across pose draws is as large as the between-height variation. P20's 3-point result was mostly noise. |

---

## 9. RETRACTIONS — read before trusting any earlier document

Stage 0 produced **six** confidently-worded wrong answers, Stage 1 **six** more, Stage 2
**sixteen** so far (items 1–11 below, plus 19–23). Every one produced a plausible result rather than a crash, which is why the
controls mattered.

### Stage 2, third batch — the `env.step` port (2026-08-03, `13_STAGE2_BC_RESULTS.md`)

19. ~~**"The joint-space `home → chain[0]` lerp is clean; the whole path lies above
    z = 125 mm."**~~ P29's registered prediction. **Refuted:** 4.85 mm of keep-out penetration
    (`gripper_left` into `distractor_1`), 100 % approach hazard, **5.5 %** success. The premise
    was a claim about the *TCP*, and the TCP is not the arm — and it was false even of the
    TCP, which dips to **z = 86 mm**, 54 mm off the straight line. P17's lesson, at a new
    location.
20. ~~**"Endpoint agreement in tool space is agreement."**~~ The forward dense-Cartesian
    approach has 0.00 mm penetration, every body above z = 105 mm, 0.8 mm of line deviation,
    and it scores **0.0 %** — with **96 % of targets still delivered to the goal and 100 %
    topple**. It ends **1.90 rad** from the frozen chain on `joint6`: a different IK branch at
    the same TCP, invisible to every statistic this codebase computes. **Where two
    independently solved paths meet, the joint-space agreement is the thing to check.**
    Solving backward from the known endpoint makes the seam the identity by construction.
21. ~~**N7's "`joint1` at ≈1.57 is the binding action magnitude."**~~ It is **`joint4` at
    4.63**, and **all six** joints exceed 1.0. `clip_actions = 1.0` does not merely make the
    goal hard to reach for a from-scratch PPO agent; it makes this manoeuvre **unreachable**.
22. ~~**"47 % of the demo is a held pose."**~~ With the approach segment the demo is 472 env
    steps, of which 180 (**38.1 %**) are a hold. The N2 *mechanism* is unaffected and is now
    measured directly rather than argued: ratio **1.005** in the `close`, **54 %** of the
    chunk ambiguity.
23. **`ex.score()` is not valid under `env.step`.** It reads the scene at the end of the run,
    and a terminated env has already been auto-reset with its blocks re-spawned upright — the
    topple erases itself. Caught in the smoke run (`tele` reported 0.0 % topple against
    `phys`'s 15.6 % on identical spawns). Read `TerminationManager._term_dones` right after the
    step instead. **Any probe that moves from `run_physics` to `env.step` inherits this.**

### Stage 2, second batch — the P26 family (2026-08-03)

These supersede items 3 and 4 below rather than merely adding to them.

5. **"Simulator screening is worth +10 points."** Withdrawn **as a quantity**. Four
   replications of the identical `screen = 4` configuration produced 73.7, 66.5, 74.6 and
   **62.8 %** — a 12-point spread. A configuration whose own replications span 12 points
   cannot support a 10-point effect estimate. *Not* claimed: that screening does nothing —
   its validity is **unmeasured** (see item 9).
6. **"Screening both rolls is worse (66.5 vs 73.7)."** Withdrawn as a *measurement*; the
   7.2-point gap was inside the noise. The **reasoning** — enlarging a noisy selector's pool
   amplifies its noise — still looks right and is still why it stays off.
7. **"The roll rule is worth ≈ +0.4 points."** Withdrawn. That arithmetic was fitted to a
   0.9-point difference between two quantities each carrying a ±12-point interval. The rule's
   justification reverts to determinism, plus P30's **paired** +7.0.
8. **My registered prediction that P26-v4 would reach 78–80 %.** It reached **62.8 %** — but
   the run's own CI is [47.5, 78.0], so this is a *non-result from an underpowered run*, not
   a clean falsification. Saying "falsified" would overclaim in the other direction.
9. **My own r² = 0.016 verdict on the screen score.** All 15 rows were **winners** — already
   selected by maximising that very score. Range restriction attenuates the correlation toward
   zero by construction. The number does not show the screen is invalid; measuring that needs
   held-out scores for *losing* candidates, which no run has produced.
10. **"The wrist side is worth +13.0 points" (registered this session).** **REFUTED** by P32:
    **+2.8 ± 8.3, t = 0.34**, 4 608 batch-paired episodes. A predictor mined from an
    underpowered observational set, with the search not counted in the p-value.
11. **"`|wrist_y|` predicts pose quality (r = −0.804)."** **REFUTED** by P33 at r = −0.223,
    on candidates that had no part in generating it — and refuted against a threshold
    registered in the probe *before* it ran.

**The methodological retraction underneath all of these:** P26 compared *different poses*
across arms and **paired nothing**, while the pose draw is the dominant variance term
(sd 12.0). Stage 1's own rule was "pair everything." P26 violated it for five runs, including
two designed this session. P30 is untouched by this precisely because it *was* paired.

### Stage 2, first batch

1. **"The wrist is put in its gap deliberately" (§3.2).** It is **drawn**: +y five times, −y
   three, over eight independent solves (P28). The keep-out term verifies whichever gap it
   lands in. **Corrected in place.**
2. **"Pose selection gates on `o_align ≥ 0.99`" (§3, and `06_EXPERT_DESIGN.md`).** The gate
   **leaked in three independent places** and poses at `o_align` 0.8627, 0.8661 and 0.9788
   shipped. `plan()`'s fallback kept the highest **wrist** rather than the best alignment;
   `_screen` generated its other candidates with an ungated `_solve`; and `_gated_solve`'s own
   fallback could still hand a sub-gate candidate to the screen, where it could win. The first
   two are fixed and measured, **the third is fixed and UNMEASURED.**
3. **My own prediction that the roll rule would reach ~78 %.** It reached **74.6 %**, i.e.
   nothing. **Falsified**, and the arithmetic that explains why is in
   `11_STAGE2_RESULTS.md` §2d — with four random-roll candidates the screen already found a
   good roll 94 % of the time.
4. **"Screen the roll, since it matters."** Implemented, measured, **withdrawn**: it doubles
   the candidate pool and reproduced `screen = 8`'s overfitting exactly (66.5 %, optimism
   +18.7). `HANDOFF.md` §10.2 item 1 had already warned that the candidate count could not
   rise before the screening statistic was de-noised. **I read that warning and walked into it
   anyway.**

Also worth recording, though not a retraction: **P30's prediction held for a reason that may be
wrong.** I argued the roll must matter because the blades' body-frame z extent (−58.7…+34.7 mm)
is not mirror-symmetric — an argument that only holds if the roll axis is *not* the body's z,
which was never checked. The prediction was right; the argument is unverified. A right answer
from a shaky argument is not evidence the argument was sound.

### Stage 1

1. **"The wrist stub is the culprit" (P18).** The mirror control changed *two* variables at
   once — wrist side **and** jaw alignment (`o_align` 0.885 vs 0.995) — and alignment is what
   moved the victim. The wrist travels **0.13 mm** through the entire slam. **Withdrawn.**
2. **"Fingers strike the target, the target strikes its neighbour" (P19/P20).** Removing the
   target changes nothing (P22). **Withdrawn.** P19's *ordering* measurement — neighbour moves
   at step 4, target at step 8 — was correct and directly contradicted the story attached to
   it. It was written down and not acted on.
3. **"Grip at 55 mm" (P20) — both the reason and the effect.** The stated mechanism concerned
   the target, which is a bystander; and the clean sweep shows the effect is far weaker than
   reported. 55 mm and 80 mm are indistinguishable end to end. **Withdrawn as a tuned
   optimum**; retained only as a value that works.
4. **P13's lift-height comparison.** Run on a pose with 32 % enclosure; the arms are not
   comparable. The 150 mm lift is retained on geometry and on P15/P17.
5. **Stage 0's "success rises monotonically with clearance".** **Inverted** on 768 episodes
   through a trajectory without the defects: 90 % at 0–4 mm against 64 % at 6–8 mm.
6. **P14's "the wrist must thread a gap"** — retracted, then **restored**. The algebra was
   over-generalised from one grip height (at 80 mm the wrist *can* clear the row), but every
   pose that clears it is unable to grasp. **The conclusion stands for any usable pose.**
   Recorded as a round trip because the intermediate claim was published before P25 ran.

### Stage 0 (abbreviated — full text in `07_STAGE0_RESULTS.md` §7)

- **USD instance proxies.** A plain `Usd.PrimRange` stops at the instance boundary and reports
  a confident **zero** colliders and zero materials. Use `Usd.TraverseInstanceProxies()`.
- **Finger thickness from an AABB** — withdrawn; finally replaced by the mesh read in §5.
- **The orthogonal grasp was dismissed by argument.** I reasoned that `o_hat = x̂` forces
  `a_hat` out of vertical and that C1 forbids top-down approaches. **C1 constrains the approach
  *axis*, not the direction of *travel*** — and P05's own data already showed a working grasp
  descending vertically with `a_hat` far from vertical. That error cost the most, and it took
  Big Will asking the direct question to undo.
- **CEM cost defects.** `w_pos = 1` walked 340–520 mm away; `w_pos = 20` returned flipped
  wrists; the floor penalty was **inert** because `base_link` sits at z = 0, so a whole-arm
  minimum is identically zero — the search returned grasp poses with `gripper_end` at z = 17 mm,
  inside the table.
- **The lift was planned *after* closing** — eva_rl's documented most expensive mistake, which
  I had in my own notes. Diagnostic signature: a clean stall at the object width, then
  `gap → −1.2 mm`.
- **`env.step` resets on topple** and re-spawns the blocks upright with fresh jitter, which
  inverted P09's headline and voided the 0 % topple rates quoted for P06/P07/P08.
- **"Gate 0 does not pass"** and the recommendation to relax `ROW_PITCH` — **both withdrawn.**

---

## 10. FUTURE PLAN

**Subject to change** — this is the plan as it stands on 2026-08-03, and three of its items
already changed once this session because a probe said so. `09_STAGE2_BC_PLAN.md` and
`10_STAGE4_STEERING_PLAN.md` are the detailed versions; this is the ordering and the reasons.

### 10.0 THE IMMEDIATE QUEUE — in this order, one GPU job at a time

```
  DONE, session of 2026-08-03 (b):
    P29 x2      approach segment.  Prediction refuted; the BRANCH SEAM found and fixed.
    collector   clutter/act/collect_demos.py written; Gate 2a AND Gate 2b both PASS.
    demos       runs/demos_v1.hdf5 -- 753 successful of 1 024 (73.5 %), 472 steps each.
    audit       clutter/act/analyse_demos.py -- N2 CONFIRMED and localised (54 % in `close`).
    train       seed 1, 100 k steps, running / done.

  1  eval        DONE.  Three seeds: 58.1 / 68.5 / 77.7 %, mean 68.1, sd 9.8.  Expert 71.4.
  2  P27         DONE.  Flat 560 -> 40 physics steps, enclosure 100.0 % throughout; and
                 shortening the OTHER holds gains +3.9 points.  No trade.
  3  demos_v3    DONE, retrained and evaluated.  69.7 / 72.0 / 73.7 % against v1's
                 58.1 / 68.5 / 77.7.  The MEAN gain (+3.7) is not established at n=3
                 (Welch t = 0.64); the VARIANCE collapse is (sd 9.8 -> 2.0, F = 23.6,
                 p ~ 0.041), and the worst seed improved 11.6 points.
  4  controls    Two BC-level questions are now open and BOTH cost 3 training runs per arm
                 (~2.5 h) because of the seed variance.  In priority order:
                   (a) UNIMODALITY.  `collect_demos.py --multimodal` is written and unrun:
                       re-solves the pose AND its own backward approach per batch.  Tests
                       10.1's registered explanation for why BC transferred so well.
                   (b) FILTERED BC.  Seed 2 beat its own expert by +6.3 pts.  The pool is
                       only the 73.5 % of episodes that SUCCEEDED, so the policy may be
                       distilling a better-than-average conditional.  Control: train on all
                       1 024 episodes (they are already in the HDF5 with success attrs;
                       train_flow.py would need a --pool flag).
  5  Stage 3/4   DAgger (conditional) or x0-steering, per 10.3 / 10.4.  Note DAgger's premise
                 has weakened: the policy has no compounding-error failure mode to correct --
                 its taxonomy is topple and nothing else.
```

**Rule earned earlier and still binding: every remaining probe that varies one knob must hold
the pose fixed** (`pose_q=` + `chain=`) **and pair its batches by reset seed.** P29 has been
edited and run this way; P27 has been edited and not yet run; **P24 still needs the edit.**

**Rule earned by P29 and binding on anything that plans a path:** where two independently
solved paths meet, check the **joint-space** agreement at the join, not the tool-space one.
Two IK branches at the same TCP are identical to every statistic this codebase computes and
differ by 1.90 rad — enough to score 0.0 % while delivering 96 % of targets to the goal.

Deliberately **not** queued: **P34-style re-litigation of the screen** (§9 item 9 — its
validity is unmeasured, but the tournament makes the question moot for Stage 2, and a proper
answer costs 6+ selections per arm); **more pose search** (P33 already banked +16, and a
second tournament would buy ~+3 at best against a 12-point sd); **P24 yaw gain** (same
pose-variance problem, and it has waited since Stage 1 without hurting anything).

### 10.1 Stage 2 — demo generation + flow BC  ← **where we are**

`DR2` is satisfied, so this is unblocked. Full detail in `09_STAGE2_BC_PLAN.md`; the five
things that will actually decide it:

1. **The schedule is 382 env steps and 47 % of it is a held pose** — 70 env steps of `close`
   alone. During a hold the 42-D observation is **constant to numerical noise** while the
   correct action chunk differs at every step, because the lift begins at a fixed offset from
   the hold's *start*. A memoryless chunk policy cannot recover phase from that; a flow head
   learns the *distribution* over "how many steps until the lift" and samples one. **That is
   survivable only if lifting early is safe**, i.e. only if the close is not much longer than
   the fingers need. **P27 measures it.** Registered prediction: flat from 560 physics steps
   down to ~120, then a cliff. Fallback if not: censor the ambiguous tail with `train_mask`.
2. **`run_physics` teleports to `chain[0]`; `env.step` cannot.** The motion from `_START_POSE`
   (TCP ≈ 350, 0, 170 mm) to `chain[0]` (250, 0, 125 mm) **has never been planned or measured**
   — every Stage-0/1 number was collected downstream of a teleport. It cannot be skipped (a
   demo must start where `env.reset()` leaves the arm) and it cannot stay a teleport (the
   policy would begin in a state it has never seen — covariate shift at t = 0). **P29 plans and
   audits it.**
3. **"Bit-exact reproduction" is not achievable and asking for it would be self-deception.**
   The gate splits: **2a** teleports to `chain[0]` and runs the rest under `env.step`, paired
   against physics-only on the same spawns (±5 points); **2b** replaces the teleport with the
   P29 approach, paired against 2a. One informed change per run.
4. **Demo diversity is capped by P28.** `plan()` draws from **four** structurally different pose
   families (wrist ±y × folded/extended arm), doubled by the wrist roll. Mixing mirror-image
   joint trajectories for near-identical observations is P17's branch flip moved inside the
   network, where no audit can see it. **Lock `wrist_side`, keep `roll_mode="rule"`, and take
   demos from one cluster.**
5. **The flush trigger is dead code for clutter.** `EventCfg` is reset-only — there are **no
   mid-episode perturbations at all**, unlike pick-place's nudges and friction randomisation.
   Default `--no-flush`, and gate on flush-on ≡ flush-off.

**The BC ceiling, recorded in advance** (`DR2` requires it). The expert is at **72.1 %** (the
frozen pose on fresh spawns); eva_bc's flow BC lost **21.6 points** to an 85.7 % expert.

```
>= 58 %      better transfer than eva_bc ever achieved — Gate 2 passes outright
50 – 58 %    ON TREND.  Judge Gate 2 on the taxonomy, not the number alone.
43 – 50 %    below trend; suspect the static-observation ambiguity (item 1) first
<  43 %      a PORTING DEFECT, not a learning limit.  Stop and find it.
```

Band shifted down 2 points from the earlier version, tracking the expert's own move from 74.6
to 72.1. It remains deliberately **wider** than the porting map's flat "≥ 60 %": a 72.1 → 58
result is a 14.1-point loss, i.e. *better* transfer than eva_bc achieved, and writing "≥ 60"
as the pass mark would have set us up to call an on-trend result a failure.

**One thing in our favour that eva_bc did not have:** every demo will come from a **single
frozen chain**. eva_bc's 21.6-point loss was against demos from a scripted expert that
re-planned per episode. P28 found six pose clusters in eight draws here, and upstream's
POSTMORTEM §9 attributes its base's failures to exactly that kind of mode confusion. A
unimodal dataset should transfer better than the trend line predicts — **registered as a
prediction, not assumed**; if BC lands above 58 % this is the first explanation to test.

### 10.1a THE BC EXECUTION PLAN — file by file, gate by gate

**Subject to change**, like everything in §10 — but this is the level of detail the next
session needs to start typing rather than re-deriving. Written 2026-08-03 after confirming the
interfaces against `eva_bc/act/` directly.

#### Step 0 — the interface facts, verified not assumed

| fact | eva_bc | **clutter** | consequence |
|---|---|---|---|
| observation | 41-D | **42-D** (8 `joint_pos` + 8 `joint_vel` + 7 `target_pose` + 12 `clutter` + 7 `last_action`) | `dataset.py` `OBS_DIM` **must change 41 → 42** |
| `STATE_SLICE` | `slice(0,16)` | `slice(0,16)` — unchanged | joint pos+vel, same layout |
| `ENV_STATE_SLICE` | `slice(16,41)` | **`slice(16,42)`**, `ENV_STATE_DIM` 25 → **26** | |
| action | 7-D | 7-D — **unchanged** | 6 arm + 1 binary gripper |
| episode | 1 500 steps | **690** (`episode_length_s = 13.8`) | = 46 windows of 15 |
| demo length | — | **382 env steps** | from `ClutterExpert.env_steps()` |

`dataset.py`'s HDF5 contract, which the collector must produce exactly:

```
/data/demo_<i>/obs/policy     (T, 42) float32
/data/demo_<i>/actions        (T,  7) float32
/data/demo_<i>/train_mask     (T,)    uint8      0 = censored from the loss
/data/demo_<i>.attrs["success"]  bool             default_demo_filter keeps only True
```

`train_mask` and end-of-episode padding **ride the same `action_is_pad` channel**, and the
vendored loss multiplies by `~action_is_pad` — so a censored step contributes no gradient.
That is the mechanism for §10.1 item 1's fallback.

#### Step 1 — `clutter/act/collect_demos.py`  ← **start here**

Not a port of `expert/run_expert_v1.py`; a new file that drives the **frozen** expert through
`env.step`. The whole point of P33/P34 is that this generates one consistent manoeuvre.

```python
spec = json.load(open(".../expert/pose_p33.json"))
ex   = ClutterExpert(env, pose_q=spec["q"], chain=spec["chain"])   # no CEM runs
for (phase, kind, *a) in ex.schedule(ex.adapt()):
    ...                       # emit 7-D actions, step env.step, record obs BEFORE the step
```

Four things that will bite, all already diagnosed:

1. **The action encoding.** `JointPositionActionCfg(scale=0.5, use_default_offset=True)` means
   `q_target = q_default + 0.5·a`, so `a = 2·(q_desired − q_default)`. The expert's `joint1`
   command comes out at `|a| ≈ 1.57`. **Record `max |a|` in the collector and fail loudly if it
   exceeds 1.0** — see §9/N7 and P29.
2. **The gripper is binary.** `BinaryJointPositionActionCfg` — 89.07 mm or 0, no intermediate
   aperture, no rate limit. A demo containing a *ramp* teaches a policy something it cannot
   emit (§8). Emit only the two states.
3. **`run_physics` teleports to `chain[0]`; `env.step` cannot.** Gate 2a takes the teleport and
   uses `env.step` for everything after; Gate 2b plans the `_START_POSE → chain[0]` approach.
   **P29 measures that segment and has not run.**
4. **Record obs *before* the step**, and record the action actually submitted — not the
   commanded joint target.

#### Step 2 — Gates 2a / 2b, in this order

```
2a   teleport + env.step for the rest, PAIRED against run_physics on the same seeds
     PASS: within +/-5 points.   NOT bit-exact -- P34 measured ~8 % episode churn between
     two runs of a bit-identical manoeuvre, so bit-exactness is not available at all.
2b   action-driven approach from _START_POSE, paired against 2a
     PASS: within +/-5 points of 2a.  If it fails, the approach segment is the defect,
     not the manoeuvre -- P29 is the diagnostic.
```

#### Step 3 — the dataset

```
8 spawn seeds x 128 envs = 1024 episodes, ~72 % keep -> ~730 successful demos
cost: ~3.4 min of simulator time at N=128 (Q7).  Nothing here needs rationing.
```

Record per demo: `success`, the spawn seed, `min_free_gap`, and the phase segment boundaries
(the last is what makes a `train_mask` censor possible later without regenerating).

#### Step 4 — training

```
python -u clutter/act/train_flow.py --data <h5> --out runs/bc_s1 \
    --steps 100000 --batch-size 64 --chunk-size 50 --n-action-steps 15 --seed 1
```

3 seeds. **`n_action_steps = 15` is not tunable** — eva_bc measured 59.4 → 32.8 → 3.1 → 0 → 0 %
at 15/8/4/2/1, so chunk commitment is load-bearing (EXP02).

#### Step 5 — evaluation, and the two things that must be reported

Pooled ≥128 episodes on held-out spawn seeds, deterministic. **Report against the §10.1 band**
(≥58 passes, 50–58 on trend, <43 = porting defect), and produce the **mandatory failure
taxonomy** — buckets, not just a number. Upstream's EXP06 looked flat-but-fine on the number
and the taxonomy is what revealed the symmetric 26/26 churn that explained it.

#### The single biggest risk, restated

**47 % of the 382-step demo is a held pose** — 70 env steps of `close` alone, during which the
42-D observation is constant to numerical noise while the correct chunk differs at every step.
A memoryless chunk policy cannot recover phase; a flow head learns the *distribution* over
"steps until the lift" and samples one. **P27 measures whether the close can be shortened**;
the fallback is the `train_mask` censor described in Step 0. **Run P27 before generating the
final dataset** — but note it must be edited to use the frozen pose first, or the 12-point pose
sd swamps the hold effect.

### 10.2 Optional — push 72.1 % toward Gate 1's 85 %

Not a blocker, and **substantially less attractive than it was**, because the single best idea
on the old list has already been executed as P33 and banked +16 points. Ordered by expected
value per hour, **rewritten after this session's results**:

0. ~~**Close the third `o_align` leak and re-measure (P26-v4).**~~ **Done — 62.8 %, and §9
   items 5–9 explain why that number decides nothing.** Superseded.
1. ~~**Screen over 2–3 spawn batches per candidate.**~~ **Done, in the form that matters.**
   This was the right instinct — averaging a candidate over more spawns is exactly what
   de-noises the selector — and P33 is its fully-realised version: candidates scored over 256
   *paired* episodes and the winner re-verified on 512 more. **The remaining headroom is
   small:** at a 12-point pose sd, a second tournament with K = 12 instead of 8 buys
   `(1.64 − 1.43) × 12 × 0.9 ≈ +2.3` points expected. Real, but a whole GPU hour for ~2 points
   while Stage 2 waits.
1b. **A cheaper version, if it is ever wanted:** keep P33's runner-up (`cand 2`, verified
   73.2 %) as a standby. Two verified poses within 2 points of each other is also useful
   evidence that the tournament found a *plateau* rather than a spike.
2. **Run P24 — yaw gain × `phi`** (written, not run; **prediction pre-registered** in
   `08_STAGE1_RESULTS.md` §7b). The blade geometry predicts **full yaw matching makes the sweep
   worse**:

   ```
   blade reach along the opening axis      ~47 mm
   full spawn yaw                          11.4°
   corner swing when the jaw is rotated    47 · sin(11.4°) = 9.3 mm
   margin to a neighbour's face            27 − 19.2       = 7.8 mm      →  intrudes
   ```

   Two existing measurements already fit and **neither was recognised at the time**: P16 (yaw
   matching raised close-phase contacts from 65/128 to 93/128 while improving every grip
   statistic) and P19 (the **square** jaw outscored the matched one end to end, 69.5 % vs
   64.1 %, dismissed as noise). `yaw_gain` is already implemented as a dial.
3. **Exploit the packing effect (§6.1).** Success is *highest* at 0–6 mm of free gap
   (90–91.5 %). If a tight row supports itself, the objective may not be "never touch" but
   "touch in a way the neighbour can lean out of". **Topple is what matters, not contact** —
   the close-phase *contact* rate is ~65 % while the *topple* rate is ~22 %.
4. **If 85 % stays out of reach, report the ceiling as a benchmark property.** The blades reach
   ~47 mm along the opening axis, the row pitch is 42 mm, and the gripper is binary with no rate
   limit a policy can request. That is a defensible finding — but it must be **measured, not
   asserted**. The Stage-0 error (declaring an env change necessary without testing the family
   that worked) is the exact mistake to avoid repeating.

### 10.3 Stage 3 — DAgger (conditional)

Only if the Stage-2 failure taxonomy shows covariate-shift failures. **Structural caveat: a
topple is terminal**, so DAgger can only address *pre-topple* drift — a strictly narrower
target than eva_bc had on its home task.

### 10.4 Stage 4 — RL on the frozen base

Full detail in `10_STAGE4_STEERING_PLAN.md`. Primary: x0-steering, gates
**S0a → S0b → H1 → 1 → 2 → 3 → 4c** in order. **Gate 4: ≥ 70 % pooled — the mission.**

**Carry the inherited risk explicitly:** eva_bc's EXP07 passed its plumbing gates and **never
reached Gate 3**. The method its own README recommends has never produced a success number on
any task. Do not stake the mission on it; §8 of the steering plan is the hedge and is not
optional.

Four things the review added that were not in the earlier sketch:

1. **Gate S0a — does x0 have *authority*, and is the broadcast form usable?** `set_steer`
   builds a **rank-1** x0 (one value per action dim, `expand`ed across all 50 chunk positions),
   but training drew `x0 = randn_like(x1)` — **iid** across positions. The flow model has never
   seen a rank-1 x0. That may be exactly why it has authority (eva_bc measured fixed draws
   spanning 14.1–56.2 % on identical weights) and may be why a steered chunk is incoherent.
   **Untested in eva_bc.** Registered thresholds: authority ≥ 10 points of spread across 8
   draws, and the best rank-1 draw within 10 points of `x0 = zeros`. If coherence fails, a
   distribution-preserving blend `x0 = α·tanh(z)⊗1 + √(1−α²)·randn` becomes primary — a 3-line
   change and a strict generalisation.
2. **Gate 4c — is z state-dependent?** The EXP06 failure was a **state-independent** residual:
   PPO learned *effort*, not discrimination. Log z per window, regress it on
   `[min free gap, gap−y, gap+y, target yaw, min up_z]`. **R² < 0.05 on all seven dims refutes
   the mechanism whatever the success number does.** eva_bc listed this gate and never ran it.
3. **H1 — search over ~32 constant `fixed_x0` draws BEFORE any PPO.** eva_bc's own data says
   the worst-to-best spread is **42 points**. It is a 32-eval search with no training, and it
   sets the number steering has to beat to have earned its GPU time. **eva_bc never did it.**
   Training first and searching afterwards is how a +42-point constant gets credited to a policy.
4. **Keep `distractor_toppled` ON.** eva_bc turned its analogous termination off for perfect
   window alignment; here that termination **is the task**, and the −40 is the sharpest signal
   against the exact failure being attacked. The cost is bounded — ≤14 steps of a fresh episode
   under a stale z, once per episode — and must be **logged** as
   `windows_containing_a_reset / total_windows`. Arm B (both off) is registered as the fallback.

Also: the steering observation should be **62-D, not 57-D** — the eva_bc tail plus
`min(up_z)`, its per-episode latch, and the three free gaps. That is the EXP01 salience lesson
applied one level up: `min` over four interleaved dims is a nonlinear function an MLP has no
reason to form, and P26 §6.1 shows the failures stratify on exactly that. Registered as arm
`obs=62` against arm `obs=57`.

**Config non-negotiables:** `clip_actions: 1.0` for the steering agent (z is the RL action
there, so 1.0 is correct), `agent.params.config.name=rebot_clutter`, and the key path is
`agent.params.env.clip_actions` (`eva_rl/scripts/rl_games/train.py:166`).

### 10.5 Still unmeasured

- **Q7: throughput and VRAM at N = 16 / 128 / 512 / 1024 / 2048** on the 10 GiB card. Every
  env-count budget in both repos assumes different hardware. Needed before any training run is
  sized. **This has been outstanding since Stage 0 and should be done before Stage 2's demo
  generation.**
- **`Tight-v0` (6 mm gaps) end-to-end.** P21's attempt failed with `Simulation context already
  exists` — it needs a **separate process per task**, not a second `gym.make` in one.
  **Prediction registered: Tight is within 10 points of nominal, and §6.1 suggests it may be
  better.**
- **`|a| > 1`, and whether the shipped rl_games config can reach its own goal.** Actions are
  absolute (`q_target = q_default + 0.5·a`), and the goal at (185, −185) mm sits at azimuth
  −45° = −0.785 rad while `default joint1 = 0`, so the carry should need `a[0] ≈ −1.57`.
  `RlGamesVecEnvWrapper` declares `action_space = Box(−clip_actions, +clip_actions)`
  (`rl_games.py:232`) and rl_games rescales `[−1, 1]` to those bounds — so at
  **`clip_actions = 1.0` the arm cannot pass `joint1 = −0.5 rad` and the goal is out of reach**,
  while at the shipped **100.0** training is destroyed. Harmless for BC (`eval_act.py` applies
  no clamping) and **irrelevant to x0-steering** (the RL action there is z, not the arm
  command) — but if confirmed it is a **finding about the benchmark**, and it is the reason
  from-scratch PPO is a poor hedge. **P29 measures it.** *Measured, not asserted* — the number
  is not in hand yet.
- **P24 (yaw gain × φ)** — still written and still not run. Prediction pre-registered in
  `08_STAGE1_RESULTS.md` §7b.
- **P26-v4** — the third `o_align` leak is fixed in code and **unmeasured**.

---

## 11. NON-NEGOTIABLE CONVENTIONS

### Inherited from eva_bc, each paid for in GPU-weeks

- **Pre-register** design, beliefs and decision rules before coding; record verdicts in place
  with dated retractions, **never silent edits**.
- **≥3 training seeds per arm; champion on a held-out spawn seed; pooled ≥128-episode numbers
  only.** eva_bc measured **26.6 points** of seed variance — single-run comparisons are void.
- **Gate every wrapper on bit-exact reproduction** of the previous stage with the learned
  component zeroed.
- **Never shorten the chunk execution horizon** (59.4 → 32.8 → 3.1 → 0 → 0 % at 15/8/4/2/1).
- **Never trust train reward**; only held-out eval success counts.

### Earned in Stage 0

- **Every negative result needs a positive control.** Six wrong answers, all plausible, none
  crashed. The control grasp is what separated instrument defects from facts about the task.
- **Gate on `o_align`, not just TCP error.** They are not interchangeable.
- **Use `hold_phys`/`run_phys` for anything measuring contact.** `env.step` erases the evidence.
- **Plan the entire trajectory before the fingers close.**

### Earned in Stage 1, and equally binding

- **Pair everything.** One spawn, snapshotted and restored between arms; one pose draw shared
  across arms. At sd 13.6 % an unpaired comparison is unreadable.
- **Report hazard rates, not raw counts.** First-contact counts systematically understate every
  phase after a bad one — they made a **100 %-hazard** segment look like a minor third-place
  contributor.
- **Verify segments, not only waypoints.** A verified endpoint pair says nothing about the
  joint-space line between them. Audit the achieved TCP against the Cartesian line.
- **A control with two variables in it is not a control.** P18 changed wrist side and jaw
  alignment together and produced a confident wrong answer that survived two probes.
- **When a measurement contradicts the story, the story is wrong.** P19 measured 0.13 mm of
  wrist travel and neighbour-before-target ordering, then reported a mechanism requiring both
  to be false.
- **Remove the object to test whether the object matters.** One probe, two mechanisms retired.
- **A proxy metric that omits the success condition will select the candidates that fail it.**
  Scoring poses on disturbance alone found poses that disturb nothing because they grasp
  nothing — 2.0 % success.
- **A geometrically impossible result is a bug report, not a finding.** P23's first run showed
  the *far* block toppling while the near one never did; no correct grasp can do that, and it
  was visible before any interpretation was attempted.
- **Report the selection score next to the held-out score.** It is the only way selection
  overfitting is visible; it is what caught `screen = 8`.

### Earned in Stage 2

- **A search cannot choose what it cannot score.** The wrist roll changes nothing the CEM
  measures — TCP to 0.00 mm, axes to 1.00000, penetration identical — and is worth 7 points.
  Before trusting any selector, ask what it is *blind* to, not just what it optimises.
- **Adding candidates to a noisy selector makes it worse, not better.** Screening 4 candidates
  in 2 rolls is screening 8, and 8 was already known to overfit a single 128-env batch. The
  warning was written in this file and I walked into it anyway. **De-noise the selection
  statistic before enlarging the pool.**
- **A rule beats a search when the rule is free.** 12/12 on a variable worth 7 points, at zero
  additional trials.
- **A documented gate is not an enforced gate.** The `o_align ≥ 0.99` floor leaked in three
  independent places and shipped poses at 0.86, 0.87 and 0.98. **Grep the code for the gate
  before quoting it in a recipe table.**
- **State the confound even when the answer is unaffected.** `sign(j6) == sign(wrist_y)` and
  `|j6| < π/2` are 12/12 and perfectly correlated in the data that produced them. Picking the
  prettier one silently is how P20's grip-height optimum got published and withdrawn.
- **A correct prediction from a shaky argument is not a validated argument.** P30's prediction
  held; the mesh-asymmetry reasoning behind it was never checked and may be wrong.
- **Probes written to de-risk a port will find bugs in what they are porting *from*.** Three
  probes aimed at the eva_bc pipeline found four defects in our own expert instead.

### Earned in Stage 2, second batch — the expensive ones

- **Compute the power before running the arm, not after.** Selection-level sd here is
  **10–13 points**; at 3 selections per arm the 95 % CI is ±12. Five P26 runs, 4 608 episodes,
  and *every verdict sat inside its own noise.* Two lines of arithmetic beforehand would have
  said so. **A comparison whose replications span more than the effect is not a measurement.**
- **A predictor found by searching a dataset must be tested on data that had no part in
  producing it — and the search has to be counted.** Two mined predictors died in one day:
  wrist side (Fisher *p* = 0.0056 → +2.8 ± 8.3) and `|wrist_y|` (r = −0.804 → −0.223). Both
  felt corroborated by "independent" within-run reproductions that were the same numbers
  re-sliced. **Register leads as leads.**
- **Correlating a selection variable with an outcome inside the selected subset measures
  almost nothing.** My r² = 0.016 for the screen score came from 15 rows that were all
  *winners* — range restriction drives r toward zero by construction. It does not show the
  screen is invalid.
- **When nothing predicts the quantity, select on the quantity.** Five forward statistics
  failed. P32 then showed the variance is stable (**ICC 0.82**), so P33 ran a tournament and
  banked **+18.8 points** with no theory at all. **Verify the winner on episodes that had no
  vote, and verify the *worst* too** — that is what makes it a test rather than a story.
- **"Frozen" means frozen — check which parts actually are.** Freezing the grasp pose left 22
  of 23 waypoints being re-drawn from a CEM on every run. **Seeding did not fix it either**:
  1 320 GPU reductions are not bit-stable and 60 CEM iterations amplify 1e-7 into 0.1 rad. The
  only reliable freeze is to *persist the artifact*, not to re-derive it reproducibly.
- **Nothing had ever replayed a batch.** Every experiment paired across arms *within* a run,
  so a defect that varied *between* runs was structurally invisible for two stages. **Replay
  something old as a control, periodically.**
- **The paired-comparison noise floor here is ~8 % of episodes** (43 of 512 flip between two
  runs of a bit-identical manoeuvre) while aggregate means agree to ~0.2 points. Quote
  aggregates; do not build arguments on small episode-level churn.

### Earned in Stage 2, third batch — the `env.step` port

- **Audit the seam, not just the segment.** P17 gave us "verify segments, not waypoints".
  P29 adds: where two independently solved paths *meet*, verify the agreement in **joint
  space**. Two IK branches at the same TCP are identical to every statistic in this codebase —
  same tool point, same axes, same keep-out penetration — and differ by **1.90 rad**. That
  difference is worth **73 points**.
- **When a path must end at a known pose, solve backward from it.** Forward solving finds *a*
  solution; only backward solving guarantees *the* one. `plan()` had always done this for the
  descent; nobody noticed it was load-bearing until the approach was built forward.
- **Report the constraint metric separately from the task metric.** The forward-Cartesian arm
  delivered **96 % of targets to the goal while toppling 100 % of neighbours**. An evaluation
  that reported only the task metric would have called it a success.
- **Anything last-write-wins is invalid under `env.step`.** Isaac Lab auto-resets a terminated
  env from *inside* the step call, so a final-state read describes a fresh spawn. This bit
  twice in one session. **Latched quantities are safe; final reads are not.**
- **Measure the irreducible loss before training.** It is a nearest-neighbour calculation on
  the dataset and costs no GPU-minutes. But **do the derivation** — the flow objective's floor
  is `sigma·pi/2` per dimension, not the chunk MSE, and the per-cell bound is loose exactly
  when the ambiguity is low-dimensional. An undertaken derivation would have had us "explain"
  a plateau that was never below its floor.
- **A registered prediction can be refuted by a smoke test.** The 32-env smoke run of the
  collector killed P29's prediction before P29 ran. Four minutes.
- **Make the generator and the gate the same code path.** `collect_demos.py` is both. A
  separate verifier can agree with the expert while the recorder disagrees with both.

---

## 12. COMMAND REFERENCE

```bash
source /home/eva/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_rl

P=/home/eva/Desktop/isaacLab/eva_bc/clutter/probes
R=/home/eva/Desktop/isaacLab/eva_bc/clutter/runs

# THE HEADLINE NUMBER — reload the frozen pose and re-verify it.  72.1 % on fresh spawns.
python -u $P/p34_pose_reload.py --num_envs 128 --out $R/p34_reload_v4.json

# how to USE the frozen pose from any new probe or the demo collector:
#   import json; spec = json.load(open(".../expert/pose_p33.json"))
#   ex = ClutterExpert(env, pose_q=spec["q"], chain=spec["chain"])
# `screen`, `wrist_side` and `roll_mode` are all ignored in this mode — the pose is chosen.

# ---- STAGE 2: the env.step port, the demos, the policy  (13_STAGE2_BC_RESULTS.md) ----
A=/home/eva/Desktop/isaacLab/eva_bc/clutter/act

# Gate 2 -- four arms, paired on the same spawns, nothing written.  PASSED.
python -u $A/collect_demos.py --num_envs 128 --arms phys,tele,tele0,appr \
    --seeds 77000,77001,77002,77003 --json $R/gate2.json

# the dataset -- one arm, recorded.  753/1024 kept.
python -u $A/collect_demos.py --num_envs 128 --arms appr --record appr \
    --seeds 30000,30001,30002,30003,30004,30005,30006,30007 \
    --out $R/demos_v1.hdf5 --json $R/demos_v1.json

# offline audit -- no simulator.  Measures the N2 loss floor before spending a GPU-minute.
python -u $A/analyse_demos.py --data $R/demos_v1.hdf5 --out $R/demos_v1_audit.json

# train (~25 min for 100 k steps at ~65 steps/s, 1.1 GiB) and evaluate
python -u $A/train_flow.py --data $R/demos_v1.hdf5 --out $R/bc_s1 --steps 100000 --seed 1
python -u $A/eval_flow.py --num_envs 128 --seeds 88000,88001 \
    --ckpt $R/bc_s1/ckpt_final.pt --out $R/bc_eval_s1.json

# P29 -- RUN.  Prediction refuted; the backward-Cartesian fix is frozen into pose_p33.json.
python -u $P/p29_approach_segment.py --num_envs 128 --reps 2 --out $R/p29_approach_v2.json

# P27 -- edited for the frozen pose, NOT YET RUN.  The lever on the one failure mode left.
python -u $P/p27_hold_duration.py --num_envs 128 --reps 2 --out $R/p27_holds.json

# the paired expert comparator, on the SAME held-out spawns the policy is evaluated on
python -u $A/collect_demos.py --num_envs 128 --arms appr \
    --seeds 88000,88001,88002,88003,88004,88005 --json $R/expert_on_eval_seeds.json

# film the policy: one env, one episode per file, camera 0.67 m out, 60 deg lens
python -u $A/record_video.py --ckpt $R/bc_v3_s3/ckpt_final.pt \
    --seeds 88000,88001,88002,88003 --stills --out-dir $R/videos

# THE UNIMODALITY CONTROL -- re-solves the pose and its own backward approach per batch.
# Tests whether a single frozen chain is what bought BC its 2.9-point gap.  NOT YET RUN.
python -u $A/collect_demos.py --num_envs 128 --arms appr --record appr --multimodal \
    --seeds 30000,30001,30002,30003,30004,30005,30006,30007 \
    --out $R/demos_mm.hdf5 --json $R/demos_mm.json
python -u $A/train_flow.py --data $R/demos_mm.hdf5 --out $R/bc_mm --steps 100000 --seed 1
python -u $A/eval_flow.py --num_envs 128 --seeds 88000,88001,88002,88003,88004,88005 \
    --ckpt $R/bc_mm/ckpt_final.pt --out $R/bc_eval_mm.json

# Stage 2 probes that HAVE run
python -u $P/p33_pose_tournament.py --num_envs 128 --cands 8    # +18.8 pts, verified
python -u $P/p32_wrist_side.py --num_envs 128 --pose_reps 6     # the refutation; ICC 0.82
python -u $P/p30_wrist_roll.py --num_envs 128 --reps 3          # the +7.0-point roll (paired)
python -u $P/p28_pose_branches.py --num_envs 128 --draws 8      # 6 clusters from 8 draws
for N in 16 64 128 256 512 1024 2048 4096; do                   # 4096 fits in 3.2 GiB
    python -u $P/q7_throughput.py --num_envs $N --out $R/q7_N$N.json || break
done

# Stage 0/1 controls worth re-running when something looks wrong
python -u $P/p22_who_hits.py    --num_envs 128             # the target-removal control
python -u $P/p17_path_verify.py --num_envs 128             # the segment audit

python scripts/test_clutter_env.py --num_envs 64
python scripts/rl_games/train.py --task Rebot-ClutterExtract-v0 --num_envs 1024 \
    agent.params.config.name=rebot_clutter agent.params.env.clip_actions=1.0
```

⚠ The `clip_actions` key path is **`agent.params.env.clip_actions`**
(`eva_rl/scripts/rl_games/train.py:166`), **not** `agent.params.config.env.…` as earlier
versions of this file printed. And see `09_STAGE2_BC_PLAN.md` §2/N7: at `clip_actions = 1.0`
the arm can only reach `q_default ± 0.5 rad`, which may put the goal out of reach for a
from-scratch PPO agent — the expert's own `joint1` command is ≈ −1.57 action units.

Notes:
- **`python -u`** — probe output is buffered otherwise and results only appear at exit.
- Isaac Lab 3.0 is headless by default; `--viz kit` for the viewer.
- Hydra passthrough works for anything: `env.episode_length_s=13.8`.
- A second `gym.make` in one process fails with `Simulation context already exists` — use a
  separate process per task.
