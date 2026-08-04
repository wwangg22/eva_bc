# 17 — ROW RANDOMISATION: the row moves, and the target is not always the middle block

**2026-08-04.** `eva_rl@challenge/mdp/clutter.py::reset_clutter_row`.

Big Will:

> "I think its smarter to fix the env completely (make row randomize, and also can you alternate
> the block of interest (right now it is always the middle one)"

Both were on the deferred list (`HANDOFF.md` §6.5, tasks #20/#21), deferred on the argument that
*rotating the row before there is a manoeuvre to rotate would measure nothing*. That argument was
wrong in one important way, and the redirection is right: **the expert is being rebuilt from
scratch anyway.** Building it against a row that never moves would bake in the assumptions
(the grasp axis is world-aligned, the target is at y = 0) that the randomisation exists to
forbid — and then it would have to be rebuilt a second time. The cost of doing this first is one
day; the cost of doing it after an expert exists is the expert.

---

## 1. WHAT CHANGED

One reset event now owns the whole row.

```python
reset_clutter_row(env, env_ids, row_x=0.250, pitch=0.042,
                  row_yaw=0.30, row_xy=0.010, random_slot=True)
```

It replaces the five independent `reset_root_state_uniform` terms. That consolidation is forced,
not stylistic: **a rigid row pose cannot be composed out of per-asset resets** — every block must
share one heading and one centre — and the target's slot has to be drawn once and then read by
the other four.

| | before | after |
|---|---|---|
| row heading | always 0 | **`U(−0.30, +0.30)` rad**, rigid about the row centre |
| row centre | always (250, 0) mm | **±10 mm on both axes**, rigid |
| target slot | **always the middle** | **uniform over all 5** |
| per-block jitter | target x ±12 mm, yaw ±0.20; distractors x ±10, y ±5 | unchanged, but now **in the row's own frame** |

### 1.1 The two changes are different in kind, and conflating them would be an error

**The row's pose is an isometry of the row.** Rotating and translating all five blocks together
leaves the free gaps, the blocks' faces, and every neighbour's bearing from the target exactly as
they were. It cannot make the *clutter* harder. What it constrains is the **arm**: the grasp axis
is no longer world-aligned, the row is no longer at a known place, and the row's ends swing
outward toward the edge of the reachable envelope. It forbids a class of solution, it does not
add a physical obstacle.

**The target's slot is not an isometry.** At an end slot the target has one adjacent neighbour
instead of two, and the outer half of the finger sweep passes through free air. This genuinely
changes the geometry, and asymmetrically — see §4.

The practical consequence: **success must be reported per slot.** A pooled rate now mixes
configurations of materially different difficulty, and a policy that improves only on the easy
ones is indistinguishable from one that improves everywhere.

### 1.2 Distractor ordering is preserved

The four distractors take the four remaining slots **in row order**, so `distractor_0` is always
the most negative along the row's own axis whatever the target does. Implemented without a
branch: the *j*-th distractor takes slot `j + (j >= slot)`, which is `j` below the target's slot
and `j + 1` above it.

This keeps `clutter_obs` and every per-block statistic interpretable — in particular the
unexplained **d1 : d2 = 2.50 : 1** chiral bias from `16_DISTURBANCE_ANATOMY.md` still refers to
the same two physical positions in the row.

### 1.3 Observability

Nothing was added to the observation, and nothing needed to be. `clutter_obs` is each
distractor's `(dx, dy)` **relative to the target** plus its up-axis, so it already encodes which
slot the target occupies (two neighbours on one side and none on the other *is* "slot 0") and
which way the row runs. The target's own quaternion is in `target_pose`.

One caveat measured while writing the smoke test, and it is a property of the task rather than a
defect: **the row's heading is only observable from the block centres to about 43 mrad.** The
±10 mm fore-aft jitter tilts a 168 mm baseline by `(0.010/√3) / 0.133`. Measured max deviation
over 16 envs: **82 mrad**, consistent with that. A policy reading only positions sees the heading
through that noise; the block orientations carry it exactly.

### 1.4 New variant, and one existing variant tightened

| gym id | rule | row |
|---|---|---|
| `Rebot-ClutterExtract-v0` / `-Play-v0` | strict 2 mm | **randomised** |
| `-Tight-v0` | strict 2 mm | randomised, 36 mm pitch (6 mm gap) |
| **`-Fixed-v0`** ⭐ new | strict 2 mm | **frozen: square, centred, target in slot 2** |
| `-Lenient-v0` | topple-only | **frozen** (was: randomised by inheritance) |

`-Fixed-v0` is a **diagnostic control, not a rung of the task**, and it is what makes the
randomisation an ablation rather than a confound. Without it, "the expert got worse" cannot be
separated into "because the row moved" and "because the target changed slots".

`-Lenient-v0` was also pinned to the frozen row. It exists to re-run pre-2026-08-03 baselines,
and *the old task* is the predicate **and** the spawn distribution — a variant reproducing only
half of it would not re-run anything. It is also the right env for the disturbance diagnostics,
being the only one that both lets a trajectory run past the 2 mm crossing and matches the layout
every measured number was taken on.

---

## 2. VALIDATION

### 2.1 The smoke test — a new V8 block, and V1 rewritten

`eva_rl/scripts/test_clutter_env.py`, **passes**.

V1 measured the row pitch by projecting onto **world y**, which under a rotated row reports
`pitch · cos(yaw)` and fails a correct row. It now projects onto the row's **own** axis.

| check | result |
|---|---|
| settled free gap, over 16 envs, measured along the row's own axis | **2.6 – 20.9 mm** |
| every block inside the r ≤ 0.32 m design envelope (docs C4) | 0.237 – 0.293 m ✓ |
| blocks' own yaw vs the event's recorded `_clutter_row_yaw` | **0.00 mrad** |
| row axis from centres vs from orientations | 82 mrad (≈1.9σ of the 43 mrad fit noise) ✓ |
| heading actually varies | −0.156 to +0.267 rad, sd 0.147 ✓ |
| spacing stays inside the band the per-block jitter alone allows | ✓ |
| target's rank along the row equals `_clutter_target_slot` | ✓, 128 spawns |
| all five slots drawn | **[0, 1, 2, 3, 4]** ✓ |
| `FIXED_ROW` pins heading to 0 and the target to slot 2 | 0.00 mrad, slot 2 in every env ✓ |

Two ways to measure the heading are used, deliberately. From the blocks' **orientations** it is
exact (the distractors get no yaw jitter, so each one's yaw *is* the row's) and it catches "the
row was translated but not rotated". From the blocks' **centres** it is only good to 43 mrad, and
it catches the converse: blocks rotated but laid out along a line that is not their own axis.

**The first version of V8 failed, and it was the test that was wrong.** It fitted the axis to
centres only and demanded 20 mrad — unachievable given the fore-aft jitter. Both "failures" were
my measurement. *A tolerance has to be derived from the noise it will see, not chosen.*

### 2.2 P41 — is any spawn the env can draw ungraspable?

The question that had to be answered before shipping. Rotating the row swings its ends outward;
the analytic worst corner puts a block at **r = 0.3087 m** against a **r ≤ 0.32 m** design
envelope whose own entry records roll freedom degrading past r = 0.30 (11/12 roll bins at
r = 0.10–0.15, **5/12 beyond 0.30**). If a corner of the spawn distribution has no legal grasp,
the success ceiling is silently below 100 % and every later measurement moves against it.

Gates are the expert's own (`pos_err ≤ 1.5 mm`, `o_align ≥ 0.99`, `pen ≤ 0.1 mm`,
`low_z ≥ 12 mm`) **plus one it does not have**: the arm is teleported to each accepted pose with
the row physically present and the distractors must move ≤ 2 mm. Keep-out is tested against body
**origins**, and P38 measured the finger blades reaching 33–39 mm from theirs.

```
25/27 cells solved within every gate      median 1 CEM call per cell
every slot solves at some heading         every heading solves at some slot
BOTH worst corners solve at r = 0.3087    pos_err 0.00 mm, o_align 1.000,
                                          distractors move 0.71 / 0.81 mm
```

**Verdict: no reachability wall. The env ships as configured.**

The verdict rule is contiguity, not a pass rate, and that matters. An unsolved cell means either
"no pose exists" or "this search did not find one", and those demand opposite actions. **A
genuine reachability wall is contiguous** — it takes out a slot, or a heading, or everything past
a radius. The unsolved cells are neither: they sit at r = 0.265 while eight cells solve at
0.275–0.309, and **which cells are unsolved changes between runs of identical code** (run 7:
(1,+0.30) and (3,−0.30); run 8: (2,−0.30) and (4,0.00)) — the CEM's documented
non-reproducibility, `REFERENCE.md` §5. The *verdict* was stable across every run; the cell list
was not. That is exactly the property the rule was designed to have.

### 2.3 P41 part B — does the slot change the difficulty?

Closing the jaw at each solved pre-grasp. P38 attributes 100 % of the disturbance to this one
motion, so this isolates the slot's effect with no expert involved.

```
slot  cells  disturbed   median     p90    neighbours
   0      5        0 %    1.01     1.10    1 adjacent
   1      5        0 %    0.84     1.50    2 adjacent
   2      4       25 %    1.34     2.75    2 adjacent
   3      5       20 %    1.63     2.17    2 adjacent
   4      4        0 %    0.83     0.93    1 adjacent
                                            end slots 0/9, interior 3/14
```

Registered prediction was "the end slots disturb roughly **half** as often". **Refuted** —
0/9 against 3/14. But **n is tiny**, every median sits in 0.8–1.6 mm against a 2 mm threshold so
the rate is a coin-flip near the boundary, and slot 1 (two neighbours) has the *lowest* median of
all. Read it as *suggestive that end slots are easier*, and no more. The real number needs an
expert and jittered spawns.

### 2.4 The regression that matters — `-Fixed-v0` reproduces the baseline

The event rewrite could have silently changed the spawn distribution. It did not.

```
                       pre-refactor    -Fixed-v0 (post)
strict success            16.4 %           17.1 %
seed range             14.1 – 22.7      13.3 – 21.1
distractor_disturbed      83.6 %           82.9 %
topple / time_out / dropped  0.0 %          0.0 %
2 / 5 / 10 mm             identical       identical
```

0.7 points apart, inside the ±2.6 binomial CI at n = 768.

**This is a distributional check, not an episode-level one, and it cannot be anything else:**
one event drawing five blocks consumes the RNG in a different order than five events drawing one
each, so the individual spawns at a given seed genuinely differ. The claim being tested is that
the *distribution* is unchanged, and the four statistics above are consistent with that.

---

## 3. THE NEW BASELINE

```
frozen pose_p33 expert, 768 held-out episodes, seeds 88000-88005

    -Fixed-v0   (frozen row)        17.1 %
    -v0         (randomised row)     3.0 %      <- THE TASK
```

**−14.1 points.** Expected, and it is not a reachability failure: the taxonomy is
`distractor_disturbed` **97.0 %** with `time_out` **0.0 %** — the arm still reaches the row every
time, it just is not aiming at the right block from the right direction. The frozen expert is a
fixed 23-waypoint joint chain planned for the nominal centre pose, adapted per env by **3
damped-least-squares iterations**; an end slot at 0.30 rad is up to ~90 mm and 0.3 rad away from
what it was planned for, which `refine` cannot cover.

Seed spread widened (1.6 – 7.8 % against 13.3 – 21.1 %), consistent with the slot draw adding a
large per-episode difficulty term.

**The 3.0 % is the number to beat.** Not 16.4 %, which describes a task that no longer exists.

---

## 4. WHAT THIS OPENED UP

### 4.1 The row centre's pose family does not continue to the far corner

P41 solved every cell by continuation — seed each cell from a solved neighbour and search
locally. It works for most of the grid in **one** CEM call. It does **not** reach slot 0 at
positive yaw: those cells miss on **position**, by 1.5–5.9 mm, not on alignment, while a *global*
search finds a pose there with `pos_err 0.82 mm` and `o_align 1.000`.

So the pose family that works at the row centre does not continue to that corner — a different
branch is needed. This is the codebase's documented **two-IK-branches-at-one-TCP** trap
(`REFERENCE.md` §4.1) appearing as a search boundary, and it is a direct warning for the new
expert: **a single pose family will not cover the slot × heading space.** Any expert built by
continuation from one seed will have a hole in exactly that corner, and it will not announce
itself — the failures there look like ordinary misses.

### 4.2 Every accepted pose lives in a narrow wrist band

Across every run, poses that cleared all gates had `wrist_z` **17.2 – 23.8 mm**. Every pose that
cleared the *geometric* gates and then shoved the row on contact had `wrist_z` **≥ 28 mm**
(28.9, 29.0, 33.5, 35.6, 42.2, 46.3, 48.7 …), moving neighbours 3–80 mm.

Blocks settle with their centres at 32 mm and their tops at ~67 mm, so 28–48 mm is squarely
inside the block band while ~19 mm is below it. P23 separately found a *high* wrist (72–73 mm,
above the tops) scoring well. **There appear to be two safe regimes — under the blocks and over
them — with an unsafe band between**, and the existing `wrist_min_z` gate is a floor where a
*band* is what the data shows. Small sample, but a cheap gate to add and it separates families
that `o_align` cannot.

### 4.3 A well-solved pose may not need the extract-then-grasp manoeuvre at all

Part B's numbers deserve attention beyond the slot comparison. Closing the jaw **in situ**, at a
per-cell solved pose, moved the neighbours **0.8–1.6 mm median, p90 2.75 mm**. The frozen
expert's in-situ close, measured in P38, gives **4.8 mm median and 44 mm p90**.

If that survives contact with reality, the 53-point deficit is largely **the pose**, not the
manoeuvre — and the extract-then-grasp plan (P40) may be solving a problem that per-env pose
solving already solves.

**It probably will not survive intact, and the reason is specific:** P41 spawns have *no
per-block jitter*, so every free gap is exactly 12 mm, while real spawns run 2.6–20.9 mm with a
median near 8 — and P36 measured that disturbed episodes have a 9.37 mm median gap against
12.93 mm for clean ones. A no-jitter test flatters the pose precisely where the expert actually
fails. So this is a **lead, not a result** (task #26), and the way to settle it is to re-run
part B on jittered spawns before anything is built on it.

---

## 5. WHAT DIDN'T WORK

* **Fitting the row axis to block centres and demanding 20 mrad.** The ±10 mm fore-aft jitter
  gives that fit a 43 mrad sd; the test failed twice on its own noise before I derived the number
  instead of picking it. *Derive tolerances from the noise they will see.*
* **Hand-deriving the "worst corner" signs.** Four sign choices interact (heading, both centre
  offsets, the block's own jitter) and I got them wrong: the cell labelled "worst corner"
  came out at **r = 0.252 m**, *inside* the plain grid's own 0.286 m, so the cell that existed to
  test the extreme was testing nothing. Fixed by **enumerating** the 16 combinations, which is
  free. *If a worst case has more than two interacting signs, enumerate it.*
* **Under-budgeting the search and reading the result as physics.** Run 1 reported 9/27 cells
  unreachable — including the nominal row the expert grasps at 16.4 %. The built-in positive
  control is what caught it; without that cell in the table I would have shrunk `ROW_YAW_RANGE`
  to fix a problem that did not exist. *A probe that searches for something needs a cell where
  the answer is already known.*
* **A global CEM fallback.** Added to distinguish "no pose exists" from "the search lost the
  family". It was the **only** source of bad-family poses in the table — 4 of run 5's 7 failures
  — clearing `pen = 0.00` and still shoving the row 3–17 mm. It is kept only because it is also
  the only thing that solves §4.1's corner, and it is now safe only because the contact gate
  rejects what it produces.
* **Five runs of iterating on search quality before writing down the decision rule.** The rule
  ("a wall is contiguous") makes runs 4–8 agree on the verdict despite disagreeing on which cells
  failed. Had it been written first, run 4 would have been sufficient.

---

## 6. FILES

**eva_rl** — `challenge/mdp/clutter.py` (`ROW_X`, `ROW_PITCH`, `N_SLOTS`, `ROW_YAW_RANGE`,
`ROW_XY_RANGE`, the four `*_JITTER_*`, `_uniform`, `reset_clutter_row`);
`challenge/clutter_env_cfg.py` (`FIXED_ROW`, one `reset_row` event, `RebotClutterExtractFixedEnvCfg`,
`-Tight`/`-Lenient` pinned); `challenge/__init__.py` (registers `-Fixed-v0`);
`scripts/test_clutter_env.py` (V8, V1 rewritten); `docs/envs/clutter-extract.md`.

**eva_bc** — `clutter/probes/p41_row_reach.py`; `clutter/runs/p41_row_reach.json`,
`p42_fixed_regression.json`, `p42_random_row.json`; this document.

```bash
# the env's own smoke test -- row randomisation, both constraints, four negative controls
cd /home/eva/Desktop/isaacLab/eva_rl && python -u scripts/test_clutter_env.py --headless

# P41 -- reachability gate + the per-slot close
python -u clutter/probes/p41_row_reach.py --num_envs 128 --headless \
    --json clutter/runs/p41_row_reach.json

# the two baselines
python -u clutter/act/collect_demos.py --task Rebot-ClutterExtract-Fixed-v0 --num_envs 128 \
    --arms appr --close 40 --holds-scale 0.25 --seeds 88000,88001,88002,88003,88004,88005 \
    --headless --json clutter/runs/p42_fixed_regression.json      # 17.1 %
python -u clutter/act/collect_demos.py --num_envs 128 --arms appr --close 40 \
    --holds-scale 0.25 --seeds 88000,88001,88002,88003,88004,88005 --headless \
    --json clutter/runs/p42_random_row.json                        # 3.0 %  <- THE TASK
```
