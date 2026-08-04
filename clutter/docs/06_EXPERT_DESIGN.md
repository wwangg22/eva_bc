# 06 — The scripted expert: why not cuRobo, why not differential IK, and what instead

> ## ⚠ KEPT DELIBERATELY THROUGH THE 2026-08-03 RESET — read §3 before designing anything
>
> The expert this document describes is being **started over**: it scores **16.4 %** under the
> 2 mm rule (`15_STRICT_METRIC.md`) and its central decision — close the jaw *in situ*, between
> the neighbours — is the thing that fails. Twelve stage-result documents were deleted with it.
>
> **This one was kept because §3 is still the governing constraint and is metric-independent:
> the gripper action is BINARY**, so the 90 mm opening is a property of the environment that no
> policy can choose around. P39 later measured what that costs — 17 points — and Big Will's
> decision was to keep it. Any new expert has to live with it.
>
> What in here is now known wrong:
>
> * §4's candidate strategies were ranked against a topple-only criterion. The winner
>   (orthogonal grasp, close in situ) is the 16.4 % expert.
> * The blade geometry quoted throughout ("~47 mm along the opening axis, ±19.2 mm
>   perpendicular") is **retracted** — P38 measured the actual fouling reach at 33–39 mm.
>   The *conclusion* it supported (straddle fore-and-aft, never enter a 12 mm row gap) still
>   holds on other evidence.
> * §8's achievability gate was written against the old predicate.
>
> What survives and matters most: **§3, and the observation in §3's "And pushing is also
> constrained" that the constraint set is asymmetric.** `DISTURB_TOL` binds the four
> distractors and says nothing about the target — which is the basis of the current plan
> (`16_DISTURBANCE_ANATOMY.md` §5).

*This document contains the single most important geometric finding of the scan, and it
**corrects** an assumption I recorded earlier in `00_ENVIRONMENT.md`.*

---

## 1. ⚠ CORRECTION to `00_ENVIRONMENT.md`

I earlier wrote that the expert should be rewritten against Isaac Lab's
`DifferentialIKController`. **That is wrong, and both repos say so explicitly:**

- eva_bc `PLAN.md:41` — *"**DLS differential IK diverges from table-level configs** — never
  trust raw DLS near the table."*
- eva_rl `scripts/scripted_expert/generate_pick_place.py:7-9` — *"Pure kinematic scripting
  failed on the grasp (differential IK stalls at a z-floor ~0.045 m; joint-table replays jam
  on contact)."*

A z-floor of ~0.045 m is *exactly* the height band this task lives in (C9's TCP floor is
44 mm, and the grasp must sit at z ≈ 0.050–0.060). Differential IK fails precisely where we
need it.

**The correct instrument is FK-scored CEM over the 6 arm joints, driving Cartesian waypoint
chains** — the pattern already implemented twice in eva_rl:
`scripts/analysis/grasp_geometry.py` and `scripts/challenge/slot_insertion_probe.py`.

Why it is the right instrument here (`slot_insertion_probe.py:12-17`):

> *"No IK solver is used… scored by forward kinematics evaluated in the sim itself… cannot
> silently converge to an unreachable pose: the achieved TCP error is reported."*

That property — **the achieved TCP error is reported** — is the direct answer to eva_bc's most
expensive lesson ("planner-valid ≠ executable", executed height spread 16 mm, corrective
nudges of 12 mm moving the arm 1–4 mm). A search that scores candidates by FK *read back from
the sim* cannot produce an unexecutable plan, because unexecutable poses score badly.

Three supporting rules from eva_rl's own HANDOFF, each hard-won:

1. **Position-only IK is not a sufficient specification for this arm.** The cost function must
   include a finger-separation-axis term — `grasp_geometry.py:127`:
   `cost = pos + 0.25 * (1.0 - (sep @ Y).abs())`.
2. **Interpolating between two IK solutions in joint space does not keep the TCP straight.**
   Solve a Cartesian waypoint chain with a tight search radius instead.
3. **Finish every CEM search before closing the gripper.** CEM uses
   `write_joint_state_to_sim`, which teleports the arm and re-opens the fingers hundreds of
   times; searching for a lift path *after* closing silently drops the object. eva_rl's
   handoff says this one cost the most time of anything in that effort.

---

## 2. Why cuRobo is the wrong tool, independent of it not being installed

Five reasons, in order of decisiveness.

**1. There is no collision-free grasp pose to plan to at t = 0.** cuRobo's contract is *find a
collision-free path to a collision-free goal*. Section 3 shows no collision-free single-block
grasp of the target exists in the nominal spawn. `plan_grasp` would return `"no-candidates"`
forever, and the expert's honest-empty-return policy would correctly abandon every episode.

**2. The core skill is deliberate contact.** The first correct action is a push — a controlled
collision. Every guard in `run_expert_v1.py` exists to *prevent* exactly that.

**3. The tolerance is below the arm's demonstrated executed-state precision.** Executed height
of planner-valid poses spreads ~16 mm on this arm, and specific grasp-table row families
execute 30–50 mm off. A planner with 16–50 mm of executed error cannot thread a 7–18.8 mm gap.
eva_bc already tried to patch this open-loop (the v15 z-ladder) and it *"can't span the 16 mm
executed-height spread."*

**4. The dependency is absent and unrecoverable here.** No `curobo`, no URDF
(`assets/00-arm-rs_asm-v3/urdf/…` is missing, so `onboard_robot.py` cannot be re-run), no
MorphIt fit. And the fit itself is disqualifying: the distal finger spheres in
`rs_rebot.yml:100-124` have radii **8.7–10.8 mm**, i.e. a 17–22 mm effective prong in the
planner's collision model — **wider than the 12 mm gap**. cuRobo would declare the task
infeasible even if the real fingers fit. Note also that `rs_rebot.yml:298` locks the fingers
**open at 0.045** — the exact configuration that cannot fit this row.

**5. The API is not upstream cuRobo.** eva_bc targets a flat-layout "cuRoboV2"
(`curobo.motion_planner`, `curobo.scene`, …) pinned at commit `8e734f3`, v0.8.0.post1. Isaac
Lab's own `isaaclab_mimic/motion_planners/curobo/` uses the *old* API
(`curobo.wrap.reacher.motion_gen.MotionGen`) and is not a drop-in.

---

## 3. THE DECISIVE GEOMETRIC FACT — the gripper action is **binary**

This is what I got wrong in the pre-measurement task analysis (deleted 2026-08-03), where I reasoned about "opening to
q ≈ 0.017 to admit a 30 mm block". **There is no such command.**

`BinaryJointPositionActionCfg` offers exactly two apertures:

```
a[6] >= 0  ->  q = 0.045 per finger  ->  clear gap 89.07 mm
a[6] <  0  ->  q = 0.000 per finger  ->  clear gap ~0
```

There is **no intermediate aperture**. So an open gripper's finger inner faces sit at
**±44.53 mm** from the tool centre.

Now lay that against the row (block y-spans, nominal):

| body | y span (mm) |
|---|---|
| d0 | −99 … −69 |
| d1 | −57 … −27 |
| **target** | **−15 … +15** |
| d2 | +27 … +57 |
| d3 | +69 … +99 |

An open gripper **centred on the target** puts each finger face at ±44.53 mm — i.e. **17.5 mm
deep inside d1 and d2**. It cannot descend onto the target (fingers land on the neighbours'
tops), cannot advance radially onto it (fingers hit their front faces), and cannot be
pre-closed to an intermediate width.

Solving for alignments where **both** fingers sit in free gaps gives exactly two windows, each
**≈6.94 mm** wide:

- centre ∈ (+17.53, +24.47) mm → fingers at ≈ −23.5 and +65.5 → **encloses target + d2**
- centre ∈ (−24.47, −17.53) mm → **encloses target + d1**

**Every legal open-gripper placement encloses two blocks, never one.** A direct single-block
grasp of the middle block is geometrically unavailable in the nominal spawn.

That is by design — `clutter_env_cfg.py:12-15`: *"The gripper's fingers have to come down that
gap, or the policy has to push a neighbour aside first. That makes the correct first action
frequently not a grasp."*

> **Supersedes** the pre-measurement task analysis (deleted 2026-08-03). The feasibility question is not "is the finger
> thin enough to thread a 12 mm gap" — the fingers are 44.53 mm out and cannot be brought in.
> `p01_gripper_geometry.py` is still worth running (finger thickness bounds the two ≈6.94 mm
> windows and tells us how much of each gap a finger actually occupies), but its verdict
> section needs rewriting against the binary-aperture constraint.

### And pushing is also constrained

Tip-vs-slide threshold `h > b/(2μ)` with μ = 0.9 static (block-authored):

| push axis | block width b | tips above |
|---|---|---|
| y (across the row) | 30 mm | **16.7 mm** |
| x (toward/away from robot) | 36 mm | **20.0 mm** |

The TCP floor is **44 mm**. So *every* gripper-height push on a distractor is quasi-statically
above its tipping threshold, and a tip past 23.2° is unrecoverable (`TOPPLE_DOT = 0.75` fires
at 41.4°, but the centre of mass passes the edge at 23.2°). Whether a *fast* push slides
instead of tipping is a dynamics question **nobody has measured** — and it is now a top-tier
open question.

---

## 4. Candidate solution strategies

None is proven. Each is licensed or killed by a specific measurement (§5).

**(A) Pair-capture — no push at all.** Align to one of the two ≈6.94 mm windows so the open
gripper encloses target + one neighbour, close (fingers stall at ≈60 mm on the pair), lift
above `EXTRACT_Z = 0.090`, carry, set down, open.
*Legality:* `target_at_goal` requires only the **target** within 45 mm at z < 0.055 with
nothing toppled — the passenger is legal.
*Risk:* closing pushes the neighbour ~12 mm inward at a contact height of ~50 mm, which is
above its 16.7 mm tipping threshold. Rough estimate: pivoting about its far bottom edge, 12 mm
of travel at 50 mm height is ≈13.9° of tilt — under the 23.2° irreversibility point, and the
block then jams against the target. **Marginal but plausible. Must be measured.**
*Cost:* `disturbance` (−3.0 × displacement) makes these demos carry a shaped negative — fine
for BC, relevant if we later do RL on this base.

**(B) Singulate then grasp.** Push d1 and d2 outward ~20 mm each, then straddle the target
normally. Highest fidelity to the task's stated design intent (this is the VPG story the
write-up invokes). Needs the push not to topple — i.e. needs the dynamic-push question to come
back favourable. Highest risk.

**(C) Plow-back.** Advance the open gripper radially so the fingers push d1 and d2 backward in
+x, then close on the exposed target. Simplest to script; b = 36 mm so it tips above 20 mm;
likeliest to topple.

**(D) Topple the *target* deliberately.** Nothing penalises the target falling over — only
distractors are constrained, and `target_at_goal` accepts z < 0.055, which a 70 mm block lying
on its side satisfies (centre at 15–18 mm). A closed gripper is narrow enough to contact the
target's front face without touching either neighbour, so pushing it over **backward (+x)**
lands it clear of the row, where it can be picked up lying and carried to the goal.
*Status:* my own addition, not in the agent's list. It exploits the letter of the success
predicate. Worth measuring precisely because it may be the cheapest reliable path — and if it
works, it is also a finding about the env worth reporting upstream.

---

## 5. Stage-0 measurements that decide the design

Nothing should be built until these land. One probe script, `test_clutter_env.py` /
`grasp_geometry.py` style: teleport + scripted action, 64–512 envs, failures collected.

| Q | question | decides |
|---|---|---|
| **Q1** | finger thickness along the closing axis; swept width of a **closed** gripper | how much of each 12 mm gap a finger occupies; whether a closed "blade" fits a gap at all; whether strategy D's pusher clears the neighbours |
| **Q2** | does a lateral push at z ∈ {0.045, 0.055, 0.070} × {slow, fast} × {5, 10, 20 mm} topple a 0.025 kg distractor? | licenses or kills **B** and **C** outright |
| **Q3** | can the open gripper be threaded into a ≈6.94 mm alignment window, closed on target + neighbour, and lifted without toppling? | licenses **A** — the only strategy needing no push |
| **Q4** | does a closed-gripper push on the target's front face topple it backward cleanly, without disturbing neighbours? | licenses **D** |
| **Q5** | effective block/table friction (read the table's material off the stage, don't assume μ = 0.9) | sets every `h_crit` above |

**Ground-truth grasp verdict** — use the enclosure check from `grasp_geometry.py:207-211`,
which is the direct analogue of eva_bc's executed-state lesson and, per eva_rl's handoff,
*"cannot be faked"*:

```python
gap  = 1.0035 * robot.data.joint_pos.torch[:, fing_dof].sum(dim=1) - 0.00125
encl = (gap - BLOCK_W).abs() < 0.012
rose = bpos[:, 2] > z0 + 0.045
held = rose & near & encl
```

---

## 6. What to port from eva_bc's expert, and what to leave

### Port verbatim (scaffolding)

| component | site |
|---|---|
| `step_action` action encoding — `a[:6] = (q_target − q_default)/0.5`, `a[6] = ±1` | `run_expert_v1.py:428-438` |
| `run_traj` / `hold` / `mark` | `:457-476` |
| `build_train_mask` — zero over `missed`/`lost` segments, boundary at **detection** | `:267-277` |
| the HDF5 writer and attr schema | `:785-802` |
| the segments/outcomes labelling contract | `:457-467` |
| `GateMonitor` structure (thresholds all need re-deriving) | `collect_dagger.py:59-110` |
| DAgger takeover skeleton + `mask[:takeover_t] = 0` | `collect_dagger.py:202-283` |
| the multi-seed chain pattern in `gen_demos_nominal.sh` | |

### Rework

`Expert` class entirely (no cuRobo); `rs_rebot.yml` (unregenerable, wrong TCP convention,
fingers locked open); `build_scene`; `placed()` → `target_at_goal` semantics; `can_axis`
(Y-up cylinder) → `_up_z`; obs plumbing 41-D → 42-D.

### Do not port at all

- **`grasp_table.pt`.** Its pocket z band is **0.0121–0.0450**; the clutter grip height is
  z ≈ 0.050–0.065 on a 70 mm block — **entirely above the table's coverage**. It is
  cylinder-derived (24 mm dia), not cuboid. And its generator does not exist in either repo,
  so it cannot be regenerated for a new object or height band. Unusable *and* unregenerable.
- `carry_waypoints.pt` (basket-specific: r_split 0.27, basket at (0.22, −0.12); our goal is
  (0.185, −0.185)).
- `plan_grasp`'s approach/lift semantics — a 10 cm **vertical descent** onto the grasp is
  exactly the motion that lands open fingers on the neighbours' tops.
- `attach_from_scene` transport (the carry leg here is trivial free space).

---

## 7. Expert design rules, each traceable to a measured failure

1. **No motion planner.** CEM over the 6 arm joints, scored by FK read from the sim, with an
   explicit finger-separation-axis term. Report achieved TCP error every time.
2. **Cartesian waypoint chains, not joint interpolation between IK solutions.**
3. **Finish every CEM search before closing the gripper.**
4. **Use `TCP_OFFSET = (-0.0419, 0, 0)`.** Never 0.075 (eva_bc's place-hover convention),
   never 0.048 (its grasp-pocket convention). eva_bc had *three* live TCP constants in one
   file and an 80 mm calibration error from reading an instrumentation line that ran at the
   wrong trajectory phase. We will have **one**, named, with its provenance in a comment.
5. **Executed-state checks at every phase boundary**, mirroring `_grasp_z_correct`: after
   align, check measured finger-body y against the gap centres and **abort before inserting**
   if off by more than half the window; after close, check `gap`; after every push, check all
   four `_up_z`.
6. **Retry by identity, not list position** (eva_bc re-served the exact failed candidate on
   4/10 retry sequences by excluding positions).
7. **Vectorize from day one — 64–512 envs.** eva_bc's expert ran `num_envs=1` *because
   cuRobo forced it*, costing 27 s/episode and pinning one randomisation per run. Nothing here
   forces that; both eva_rl probe scripts are already batched.
8. **Separate RNG streams per subsystem.** eva_bc's single shared stream meant the first
   behavioural divergence reshuffled every later episode's perturbation (12/31 episodes drew
   different events), silently breaking paired A/B.
9. **A topple is a hard episode end — unlike a dropped can, there is no recovery.** So
   `train_mask = 0` over the whole failing attempt, and a toppled episode is not written. The
   DAgger gate can fire on `any_distractor_toppled` only to **stop**, never to teach.
10. **Record 42-D obs** and update `act/dataset.py`'s slice map.

---

## 8. The gate that closes the achievability question

Reproduce eva_rl's own V4/V5 rung (`CHALLENGE_SUITE.md:328-339`): a measured expert success
rate at **both** `ROW_PITCH = 0.042` and `Tight` (0.036 → 6 mm gap), **stratified by the
measured per-episode minimum free gap** (the 7.0–18.8 mm distribution).

That single table answers the question eva_rl's handoff has been carrying — *"a scripted
extraction at the narrow end would close the achievability question"* — and it is the honest
ceiling on anything BC or DAgger can later learn, exactly as eva_bc's 77.4 % perturbed expert
number capped that pipeline.
</content>
