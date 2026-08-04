# Experiment S2-N — how much action noise does this task tolerate?

**Pre-registered 2026-08-02, before the first run.** Written first on purpose: two of the
three pre-registered beliefs in `strategy_probe.py` turned out wrong, which is exactly why
writing them down before looking is worth the two minutes.

## Why

Two questions, one sweep.

1. **Demo collection (Stage B).** The expert plans everything before touching the block and
   then runs open-loop, so its nominal demos lie on a single deterministic manifold indexed by
   the block spawn. A cloned policy that drifts off that manifold has no data telling it how to
   get back. The fix is DART-style noise injection: perturb the **executed** action, record the
   **nominal** one. That is only valid because the action is a joint *position target*, so the
   expert's command is the correct thing to do from any nearby state. The sweep sets the
   largest noise the trajectory survives.

2. **Stage D (`sigma_init`).** `HANDOFF.md` §9 requires measuring this task's noise tolerance
   rather than inheriting pick-place's. There, `sigma ≈ 0.37` action units destroyed a base
   living on mm-precision grasps (−46 pts) and `sigma ≈ 0.08` was healthy — but pick-place had
   a 50 mm placement tolerance and this task has 1.5 mm. The same number cannot be assumed.

Units: 1 action unit = 0.5 rad of commanded joint offset (`scale=0.5`,
`use_default_offset=True`). Noise is an OU walk, `rho = 0.95` (~20-step / 0.4 s correlation
time), decayed by 0.7 per step during every settle/hold so phase boundaries re-converge to
nominal and deviations cannot compound over a 599-step episode.

## Design

`collect_demos.py --task Rebot-PrecisionSlot-v0 --num_envs 128 --rollouts 1 --dry_run` at
`noise_std ∈ {0, 0.01, 0.02, 0.04, 0.08}`. Fixed spawn seed, so every arm sees the same 128
block poses and the only difference between arms is the noise. n=128 per cell.

## Beliefs, stated before running

1. Seated success decreases **monotonically** in `noise_std`.
2. The binding failure mode is **precision, not grip**: `push` runs inside a 1.5 mm per-side
   channel, so failures should show up as depth/lateral misses with the grip still held
   (`push:128` in the held row), not as dropped blocks.
3. `0.02` is tolerable (≥ 95 % seated); `0.08` is not (< 70 %).
4. The **achieved** arm deviation is much smaller than the commanded one — the OU walk is
   low-pass filtered by a stiffness-2000 position controller — so at `noise_std = 0.02`
   (10 mrad commanded) the measured peak per-phase deviation lands around 3–6 mrad.

## Decision rule

Use the largest `noise_std` with **seated ≥ 90 %** for the DART demo pool. If even `0.01`
falls below 90 %, abandon whole-trajectory noise and inject only during `reach` (where the
8-step settle plus the 30-step close guarantee re-convergence before the grasp).

Report the whole curve regardless — it is the Stage D input.

## Results

*(filled in below by the run; nothing above this line is edited afterwards)*

| `noise_std` | seated (n=128) | reset | grip held at `push` | depth p10 | lateral p90 | worst-carry arm dev |
|---|---|---|---|---|---|---|
| 0.00 | **128/128 = 100.0 %** | 0 | 128 | 45.7 mm | 1.21 mm | 147 mrad (`turn`) |
| 0.01 | 113/128 = 88.3 % | 13 | 118 | 28.3 mm | 31.06 mm | 160 mrad (`push`) |
| 0.02 | 73/128 = 57.0 % | 44 | 96 | 8.5 mm | 147.87 mm | 350 mrad (`push`) |
| 0.04 | 37/128 = 28.9 % | 68 | 89 | −7.6 mm | 151.41 mm | 441 mrad (`push`) |
| 0.08 | 10/128 = 7.8 % | 78 | 79 | −23.8 mm | 147.35 mm | 530 mrad (`push`) |

Per-phase peak arm deviation [mrad], `lift / back / spin / turn / push`:

| `noise_std` | lift | back | spin | turn | **push** |
|---|---|---|---|---|---|
| 0.00 | 33 | 98 | 35 | 147 | **74** |
| 0.01 | 33 | 98 | 37 | 147 | **160** |
| 0.02 | 34 | 98 | 41 | 148 | **350** |
| 0.04 | 40 | 101 | 52 | 153 | **441** |
| 0.08 | 60 | 116 | 83 | 170 | **530** |

Grip retention per phase (envs of 128 still holding the block at the phase end):

| `noise_std` | lift | back | spin | turn | **push** |
|---|---|---|---|---|---|
| 0.00 – 0.04 | 128 | 128 | 128 | 128 | **128 / 118 / 96 / 89** |
| 0.08 | 127 | 127 | 127 | 126 | **79** |

**The grip is never lost in free space below `noise_std = 0.08`, at any level tested.** Every
single grip failure up to 0.04 happens in `push`. That one row is the whole result.

### Verdict on the pre-registered beliefs

* **Belief 1 — CONFIRMED.** Success is monotone in `noise_std`: 100.0 → 88.3 → 57.0 → 28.9 →
  7.8 %. The only belief that survived.
* **Belief 2 — REFUTED, and this is the useful result.** The failure is *not* a precision
  miss with the grip intact. At `noise_std = 0.01` the grip is lost in 10 envs during `push`
  and **13 envs terminate mid-episode** (`block_dropped` / `block_toppled`), against 0 in the
  baseline. Lateral p90 blows up from 1.21 mm to 31.06 mm, and by 0.02 it is 147.87 mm — that
  is not a policy missing the centreline by a millimetre, that is blocks ending up nowhere near
  the slot. Depth p10 goes *negative* at 0.04, i.e. the median-bad block never entered at all.
* **Belief 3 — REFUTED.** 0.02 was predicted "tolerable, ≥ 95 %" and delivers 57.0 %. Even
  0.01, half the smallest level I thought worth testing, already costs 11.7 points.
* **Belief 4 — mis-framed.** The zero-noise baseline already carries 33–147 mrad of *systematic*
  position-controller tracking lag, so "achieved deviation vs commanded" was the wrong
  comparison — the peak is dominated by lag and cannot resolve the noise at all (lift moves
  33 → 34 between zero noise and 0.02). A running *mean* was added for this reason and shows
  the noise plainly: at 0.08, mean deviation is 34.9 / 59.5 / 49.2 / 71.2 mrad in free space
  against 106.7 in `push`. What actually matters is that the lag is identical every episode
  (contributing no state diversity) while the noise is random (contributing all of it).

### Why, and what it implies

DART's correctness argument is that a joint **position target** is the right command from any
nearby state, because commanding `q_t` pulls the arm to `q_t` wherever it started. That
argument holds **only in free space**. During `push` the block is inside a channel with
1.5 mm of per-side clearance; from a laterally perturbed state, commanding the nominal target
does not pull the block back to the centreline, it drives it *harder into a wall*. The
position controller has stiffness 2000 and does not yield, so the block levers out of the pads
and drops. The recorded label is then not merely noisy — it is **wrong**.

The diagnostic that separates these: worst-carry deviation moves from `turn` (147 mrad) in the
baseline to `push` (160 mrad) at 0.01. `push` is normally the *best*-tracked phase because its
waypoints are 2 mm apart; it becomes the worst only because the arm is fighting the walls.

So the constraint is not "how much noise does the arm tolerate" but "**where is the
position-target label still correct**". That is a property of contact, not of magnitude, and
no amount of tuning `noise_std` fixes it.

---

# Experiment S2-N2 — phase-restricted noise

**Pre-registered before running, and before seeing the 0.02/0.04/0.08 cells of S2-N.**

## Design

Identical to S2-N but noise is injected only during the free-space phases
`reach, lift, back, spin, turn`, and `push` runs clean. The 25-step settle at the end of
`turn` already decays the noise by `0.7^25 ≈ 1e-4`, so the block enters the slot from a
near-nominal state by construction.

Sweep `noise_std ∈ {0.02, 0.05, 0.10, 0.20}` at n=128 — a coarser and *higher* grid than S2-N,
because free space has no 1.5 mm constraint and the interesting range should be well above
where the constrained phase broke.

## Beliefs, stated before running

1. Success stays **≥ 95 %** at 0.05 and the mid-episode resets return to ~0, because every
   removed failure was caused by contact during `push`.
2. The binding failure at high noise becomes **grasp** failure, not insertion: noise during
   `reach` misaligns the pads, and unlike `push` this *is* recoverable in principle but this
   expert cannot recover, so it shows up as `reach:grasp:g0 = missed`.
3. Achieved arm deviation in the carry phases scales roughly linearly with `noise_std` and is
   the number worth reporting to Stage D.

## Decision rule

Take the largest `noise_std` with **seated ≥ 90 %** and **resets ≤ 2 %** for the DART pool.
Collect the main pool at that value. If the whole grid stays above 95 %, extend upward rather
than settling for coverage that is too timid to matter — the point of the pool is state
diversity, and a noise level that changes nothing buys nothing.

## Results

| `noise_std` (free space only) | seated (n=128) | reset | grip held `lift/back/spin/turn/push` | depth p10 | lateral p90 |
|---|---|---|---|---|---|
| 0.00 (baseline, from S2-N) | 128/128 = 100.0 % | 0 | 128/128/128/128/128 | 45.7 mm | 1.21 mm |
| 0.02 | **124/128 = 96.9 %** | 0 | 128/128/128/128/**128** | 45.4 mm | 1.07 mm |
| 0.05 | **118/128 = 92.2 %** | 2 | 128/128/128/127/**127** | 41.0 mm | 1.30 mm |
| 0.10 | 96/128 = 75.0 % | 30 | 127/127/127/126/126 † | 7.5 mm | 131.06 mm |
| 0.20 | 20/128 = 15.6 % | 101 | 98/97/97/91/86 † | −4.1 mm | 153.08 mm |

† **These two grip rows are overstated and must not be read as "the grip was fine".** They were
produced before a reporting bug was fixed: an env that resets mid-episode is handed a *fresh
scene* by Isaac Lab, so its finger gap is measured on a new episode and reads as a healthy
grip. The 0.10 row claiming `126/128` sits next to its own `lateral p90 = 131 mm`, and the two
cannot both be true. `collect_demos.py` now masks the phase statistics with `& ~resets`, and
prints the seated rate over *attempted* episodes alongside the (misleadingly high) rate over
*kept* ones — at 0.10 those are 75.0 % and 98.0 % respectively. Rows 0.00–0.05 are unaffected
(0, 0 and 2 resets).

Mean per-step arm deviation [mrad] — the metric that can actually see the noise:

| `noise_std` | lift | back | spin | turn | push |
|---|---|---|---|---|---|
| 0.02 | 19.5 | 42.8 | 26.2 | 50.0 | 42.5 |
| 0.05 | 25.8 | 49.6 | 35.5 | 59.0 | 43.2 |

### Side-by-side with S2-N at the same magnitude

Same seed, same 128 spawns, same noise magnitude — the only difference is whether the arm is
perturbed while the block is between the slot walls:

| `noise_std = 0.02` | seated | reset | grip at `push` | lateral p90 | `push` peak dev |
|---|---|---|---|---|---|
| noise in **all** phases (S2-N) | 57.0 % | 44 | 96/128 | 147.87 mm | 350 mrad |
| noise in **free space** only (S2-N2) | **96.9 %** | **0** | **128/128** | **1.07 mm** | **74 mrad** |

`push` deviation returns exactly to its 74 mrad no-noise baseline. That is the diagnosis
confirmed: the magnitude was never the problem, the *contact* was.


### Verdict on the S2-N2 pre-registered beliefs

* **Belief 1 — mostly confirmed.** Predicted "≥ 95 % at 0.05 and resets back to ~0". Got
  **92.2 %** and **2** resets, against 57.0 % and 44 at the same magnitude with `push` noise.
  The reset collapse (44 → 0/2) is exactly as predicted and is the load-bearing half; the
  success number landed just under the stated bar.
* **Belief 2 — REFUTED.** The binding failure at high noise is *not* grasp misses. It is the
  **carried block colliding with the fixture**: 30 mid-episode terminations at 0.10 and 101 at
  0.20, i.e. `block_dropped` / `block_toppled`, which fire while transporting, not while
  reaching. This is a *third* distinct mechanism, separate from both the contact-jamming of
  S2-N and any grasp error. It has a clean geometric explanation — `stage_x` was set to 0.165
  precisely because the carried block rides ~5 mm ahead of the TCP and the wall front faces
  are at x = 0.210, so free space is only free within ~40 mm, and a 50 mrad perturbation
  across six joints moves the TCP far enough to close that margin. **"Free space" is bounded,
  and its bound is what caps the usable noise.** (To be confirmed against the
  `reach:grasp:g0 = missed` count in the collected pool's outcome histogram; the sweep cells
  ran with `--dry_run` and wrote no per-demo outcomes.)
* **Belief 3 — confirmed.** Mean per-step arm deviation scales smoothly with `noise_std`:
  `turn` goes 50.0 → 59.0 → 81.0 → 137.9 mrad across 0.02 / 0.05 / 0.10 / 0.20.

### Decision

**`noise_std = 0.05`, free-space phases only** — the largest level satisfying the
pre-registered rule (seated ≥ 90 %, resets ≤ 2 %).

The DART pool is collected at a **mix of 0.02 and 0.05** rather than a single level, so it
spans a range of achieved deviations (mean `turn` deviation 50 and 59 mrad) instead of one
value. Same GPU cost, strictly broader coverage of the tube around the nominal manifold —
which is the entire purpose of the pool.

### Deliverable for Stage D

`HANDOFF.md` §9 requires this task's action-space noise tolerance to be *measured* rather than
inherited from pick-place, where `sigma ≈ 0.37` destroyed the base and `sigma ≈ 0.08` was
healthy. Measured here, on the frozen scripted expert:

| `sigma` (action units) | expert seated, free-space noise | with contact noise |
|---|---|---|
| 0.02 | 96.9 % | 57.0 % |
| 0.05 | 92.2 % | — |
| 0.10 | 75.0 % | — |
| 0.20 | 15.6 % | — |

This task is **far** more noise-sensitive than pick-place: 0.08, the *healthy* setting there,
sits between our 75 % and 15.6 % cells. `sigma_init` for the additive residual should therefore
start no higher than **0.05** (`sigma_init ≈ -3.0`), not the inherited −2.5. This is a prior,
not a substitute for measuring the trained base's own tolerance — the base is a different
controller from the scripted expert — but it is the right order of magnitude and it is
measured on this task.
