# 09 — Stage 2: demo generation + flow-matching BC

**Written 2026-08-03, after a line-by-line review of `eva_bc/act/`.** Supersedes
`05_PORTING_MAP.md` §9 Phase 0–3 where they disagree; the porting map's *mechanical* diff
(§3 obs layout, §4 actions, §5 x0 contract) remains correct and is not repeated here.

Read `HANDOFF.md` first. This document assumes the expert (73.7 % over 768 episodes) and the
five Stage-1 fixes as given.

---

## 0. WHAT THIS DOCUMENT IS FOR

`05_PORTING_MAP.md` answered *"what has to change to make eva_bc's code compile against a
42-D observation."* It is a **mechanical** diff and it is still right.

This document answers a different and harder question: **"what about clutter breaks the
assumptions the eva_bc pipeline was designed around, and what do we do about it?"**

Reviewing `act/` end to end produced **eight** clutter-specific findings that the porting map
does not contain, three of which are load-bearing enough to change the order of work. Section
2 is the substance of this document; everything else is scaffolding around it.

---

## 1. THE PIPELINE, AS REVIEWED

Nine files, ~1 600 lines. What each one actually does, and what is load-bearing.

| file | role | load-bearing detail |
|---|---|---|
| `dataset.py` | HDF5 → `(obs_t, action chunk, action_is_pad)` samples | **Every `(demo, t)` is one sample.** `action_is_pad` carries *two* meanings — past-episode-end **and** `train_mask == 0`. Group names are **parsed** (`int(k.split("_")[-1])`). |
| `normalize.py` | mean/std, **outside** the policy | Stats ride in the checkpoint as buffers. Deployment must normalize obs and unnormalize actions itself. |
| `modeling_act.py` | vendored LeRobot ACT transformer | untouched; only `ACTEncoder` + sinusoid helper are imported |
| `modeling_flow.py` | rectified flow head on that backbone | `x_τ = (1−τ)x0 + τx1`, target `v = x1 − x0`, masked MSE. `key_padding_mask = pad & ~pad.all(dim=1,keepdim=True)` — the all-masked-row NaN guard. **703 k params.** |
| `train_flow.py` | 100 k steps, AdamW, lr 1e-4, batch 64 | model widths flow from `dataset.py` constants via `make_config`; **no edits needed** |
| `eval_act.py` | `BatchedACTController` + batched sim eval | `_buf (N, n_action_steps, 7)` / `_idx (N,)`; `_idx == n_action_steps` ⇒ empty. `steer_x0` **outranks** `fixed_x0`. |
| `residual_core.py` | frozen-base obs builder + additive blend | the 15-D shared feature tail; the `:159` quaternion bug |
| `steer_core.py` / `steer_wrapper.py` | x0-steering | see `10_STAGE4_STEERING_PLAN.md` |
| `train_steer.py` | PPO over windows | ditto |

**The x0 contract** (`modeling_flow.py:117`), which every later stage depends on:

```
x0.dim() == 2  →  (chunk_size, action_dim)      broadcast over the batch   [fixed_x0]
x0.dim() == 3  →  (B, chunk_size, action_dim)   per-env                    [steer_x0]
```

**The three eva_bc numbers that set our expectations:**

- flow BC on a **85.7 %** expert reached **64.1 %** pooled (champion of 3 seeds) — a
  **21.6-point** teacher-to-student drop.
- **training-seed variance was 26.6 points** (32.8 / 50.0 / 59.4 % on identical data).
- shortening the execution horizon collapsed success monotonically: **59.4 → 32.8 → 3.1 → 0
  → 0 %** at `n_action_steps` = 15 / 8 / 4 / 2 / 1.

---

## 2. EIGHT FINDINGS FROM THE REVIEW

Ordered by how much they change the plan.

### N1 — `run_physics` teleports. `env.step` cannot. There is a missing motion segment.

`ClutterExpert.run_physics` opens with

```python
K.teleport_arm(chain[0], Q_OPEN)
```

`chain[0]` is the top of the approach: TCP at **(250, 0, 125) mm**. The env's reset puts the
arm at `_START_POSE`, TCP near **(350, 0, 170) mm**. **The motion between those two poses has
never existed, never been planned, and never been measured.**

It cannot be skipped and it cannot be teleported:

- Skipping is not an option — a demo has to start where `env.reset()` leaves the arm.
- Teleporting in the demo collector *would* work mechanically, but the resulting HDF5 would
  contain no action for the approach, and the policy at deployment would begin in a state it
  has never seen. That is covariate shift at t = 0, the worst possible place.

**So Stage 2 begins by planning and verifying a new trajectory segment.** It gets the same
treatment every other segment got in Stage 1 — dense Cartesian audit against the straight
line, whole-arm keep-out check, per-phase hazard rate. The P17 lesson (*verify segments, not
only waypoints*) applies with full force to a segment that has never been looked at.

**Consequence for the "bit-exact reproduction" gate.** Bit-exact reproduction of the
physics-only number is **not achievable** and asking for it would be self-deception. The gate
splits in two:

- **Gate 2a — executor equivalence.** Teleport to `chain[0]` at reset (allowed: the collector
  owns the reset), then run the *entire remaining schedule through `env.step`*. Success must
  match the physics-only result on the **same spawn batch**, paired. This isolates
  "does `env.step` execute the schedule the same way physics-only does" from everything else.
- **Gate 2b — the approach segment.** Replace the teleport with an action-driven
  `home → chain[0]` move. Paired against 2a on the same spawns. Any drop is attributable to
  the new segment and nothing else.

One informed change per run. That is the whole reason for splitting.

### N2 — The close phase is 70 env steps of a **static observation**. This is the central BC risk.

Compute the schedule in env-step units (one env step = `decimation` 8 physics substeps;
`_hold(steps)` is `steps` **physics** steps, `_move(steps)` is `steps` **env** steps):

| phase | schedule arg | env steps | kind |
|---|---|---|---|
| `settle` | hold 80 phys | 10 | static |
| `descend` | 3 moves × `st_in` 25 | 75 | moving |
| `predwell` | hold 160 phys | 20 | **static** |
| `close` | hold 560 phys | **70** | **static** |
| `carry` | 17 moves × `st_out` 6 | 102 | moving |
| `dwell` | hold 160 phys | 20 | **static** |
| `release` | hold 240 phys | 30 | **static** |
| `withdraw` | move 25 | 25 | moving |
| `final` | hold 240 phys | 30 | **static** |
| | **total** | **382** (7.64 s) | **180 static = 47.1 %** |

(`i_grip = 3`, `m = 20`: the approach is 3 legs of 23.3 mm — one dense point each at
`seg = 30 mm` — and the outward chain is 5 + 7 + 5 = 17.)

During `close` the arm holds one joint target, the fingers stall within the first few
physics steps (P19 measured the slam at 2.5 ms resolution), the block is gripped and still,
and the distractors are still. **The 42-D observation is then constant, to numerical noise,
for tens of consecutive env steps — while the correct action chunk is different at every
one of them**, because the lift begins at a fixed offset from the *start* of the close.

A memoryless chunk policy **cannot** recover phase from a constant observation. What it can
do is learn the *distribution* over "how many more steps until the lift", which is exactly
what a flow-matching head is for. At inference it draws one mode — so the policy will begin
its lift at *some* point in the close, sampled, not at step 70.

That is survivable **only if lifting early is safe**, i.e. only if the close is not much
longer than the fingers need. 560 physics steps was chosen for a *physics-only measurement*
where duration was free. Nobody has ever asked how short it can be.

**→ P27 measures it.** See §5.

The same argument applies, more weakly, to `predwell`, `dwell`, `release` and `final`: every
static frame is an ambiguous label. 180 of 382 frames is a lot of ambiguity to hand a
memoryless policy.

**The `train_mask` alternative, and why it is second choice.** eva_bc's censor machinery
could zero the tail of each hold, which would delete the conflicting labels outright. But the
policy would then have *no* label for the states it will actually occupy at inference (the
observation at close-step 30 is the same as at close-step 5, so it would still be *served*,
just by generalisation from step 5) — which lands in the same place as simply shortening the
hold, with more moving parts. **Shorten the holds; keep masking as the fallback if P27 says
the close cannot be shortened.**

### N3 — Multiple screened poses may sit in different IK branches. Check before mixing them.

`plan()` seeds the CEM from the home pose with `std0 = 0.6, restarts = 12` — a deliberately
wide global search — and `_screen` picks among draws by simulated outcome. Nothing constrains
two independently-screened poses to lie in the same IK branch.

Demo diversity wants several poses (the pose draw is worth up to 40 points of variance, sd
4.8 % even after screening). But if two poses are in different branches, the dataset contains
two very different joint trajectories for near-identical observations, and a chunk policy that
interpolates between them reproduces **exactly the failure P17 found in the CEM** — a
joint-space path that leaves the Cartesian line — only now inside the network where no audit
can see it.

**→ Measure it (P28).** Solve the pose 8× under 8 spawn seeds, then report the pairwise
per-joint L∞ distance and cluster. Decision rule fixed in advance:

- all 8 within **0.35 rad** per joint of a common centre → one branch → **use all 8**;
- otherwise → keep only the **largest** cluster, and report how many demos that discards.

This is cheap (no rollouts, just `plan()`) and decisive.

### N4 — The section-4.2 flush is **dead code** in clutter.

`eval_act.py:301-312` and `residual_core.flush_check` clear an env's action queue on a
detected object discontinuity — a teleport-size position jump, or a fast z drop. That
detector exists because **pick-place had mid-episode perturbation events**: nudges and
friction randomisation (`03_ENV_FACTS.md` §6).

**Clutter has no mid-episode events at all. `EventCfg` is reset-only** — `reset_all`, five
`reset_root_state_uniform` terms, `record_spawn_xy`. Nothing perturbs the scene after t = 0.

So the flush can only fire on physics the policy itself caused, and when it does — a
distractor beginning to topple — re-predicting from fresh observations does not help, because
`distractor_toppled` is about to end the episode anyway.

**Decision: `--no-flush` is the default for clutter.** Keep the machinery (it is shared with
the steering stack and removing it would fork the code), and **gate it**: flush-on and
flush-off must produce identical episodes on a fixed seed. If they differ, the detector is
firing on normal manipulation and the thresholds are wrong for this task — which is itself
worth knowing.

The porting map's suggestion to redesign the trigger around `up_z` is hereby **withdrawn**:
it would build a detector for an event that no longer has a useful response.

### N5 — Clutter's grasp bit is probably trivial, exactly where eva_bc's was hard.

eva_bc needed a 5-input MLP (finger pos ×2, finger vel ×2, last grip command) to separate
*grasped* from *closed-on-air*, because a hand-written aperture threshold scored **94.6 %
accuracy at 40 % FPR** — the two aperture distributions overlapped almost completely
(medians −0.053 vs −0.056).

Clutter's geometry is different in a way that should matter:

```
jaw fully open            89.07 mm
block across the fingers  36 mm            (orthogonal grasp closes on the x-faces)
closed on air             gap → −1.25 mm   (GAP_A/GAP_B calibration, resid 0.035 mm)
separation                ~37 mm
```

The expert's own success predicate is already a threshold — `|gap − width| < 12 mm` — and it
reads **100 % enclosure** over 768 episodes. A 37 mm separation against a 12 mm tolerance is
not a marginal classification problem.

**Prediction, pre-registered: a two-input rule — `|gap − 0.036| < 0.012` AND commanded-closed
(`obs[41] < 0`) — clears eva_bc's 0 % FPR bar on clutter data.** If it does, the MLP, its
training corpus, and the on-policy negative-mining run are all unnecessary, and Phase 6 of
the porting map collapses to nothing.

**The eva_bc lesson that survives regardless: the commanded-grip channel is mandatory.**
Physical finger joints alone scored higher AUC (0.976) but **27.1 % FPR**. Aperture alone
cannot distinguish "holding" from "mid-close". Keep `obs[41]` in the rule.

Measured in **P31**, on expert post-close frames plus on-policy closed-on-air frames from the
first BC checkpoint. It therefore lands **after** the base exists — same ordering as eva_bc.

### N6 — Nothing in clutter penalises grip chatter.

pick-place carried `gripper_toggle = −50`. Clutter's `RewardsCfg` has **nine terms and no
toggle penalty** (`03_ENV_FACTS.md` §6). The gripper is binary with **no rate limit**, and
`a[6] < 0` closes on the sign alone.

A flow policy sampling near the decision boundary can open and close the jaw every step. In
the row, each open→close cycle is another blade sweep through the neighbours' band — the
mechanism P22 identified as the *entire* remaining failure mode.

Two things make this less likely than it sounds, and one makes it worse:

- **for:** chunk commitment — 15 steps are executed before any re-prediction, and the chunk is
  generated jointly, so within-chunk consistency is learned from the data.
- **for:** `obs[35:42]` is `last_action`, so `obs[41]` is the previous grip command. The
  policy has the exact input needed to learn hysteresis.
- **against:** the action normalizer folds the grip channel in with the six arm dims. The grip
  column is roughly `+1` for the first ~105 steps and `−1` for the remaining ~277, so
  `mean ≈ −0.45, std ≈ 0.89`. The **unnormalized output crosses zero at the mean, not at
  normalized zero** — a fact `05_PORTING_MAP.md` §4 already records and which is easy to get
  wrong when reasoning about the head's bias.

**Decision: change nothing, measure it.** `toggles_per_episode` is a first-class number in the
Gate-2c report. Only if it is a leading failure mode do we consider a fix, and the fix would
be a data-side one (the expert toggles exactly once per episode), never a controller-side
clamp — a clamp the policy cannot emit is the ramped-close mistake all over again.

### N7 — The expert emits `|a| > 1`, and that has consequences for the *hedge*, not for BC.

Actions are absolute: `q_target = q_default + 0.5·a`, so `a = 2·(q_target − q_default)`.

The goal is at env-local **(185, −185) mm**, azimuth **−45° = −0.785 rad**. `joint1` is the
base yaw and `default joint1 = 0`, so the carry needs roughly

```
a[0] ≈ 2 × (−0.785 − 0) ≈ −1.57
```

**For BC this is harmless.** `eval_act.py` applies no clamping and no sign quantisation —
`env.step(actions)` takes the raw tensor. `05_PORTING_MAP.md` §4 already says so.

**For rl_games it is a hard wall.** `RlGamesVecEnvWrapper` declares
`action_space = Box(−clip_actions, +clip_actions)` (`rl_games.py:232`) and rl_games'
`preprocess_actions` rescales `[−1, 1]` to those bounds; `step` then clamps
(`rl_games.py:303`). So with **`clip_actions = 1.0`** the reachable joint range is
`q_default ± 0.5 rad` and **`joint1` cannot pass −0.5 rad — the goal is out of reach.** With
the shipped **`clip_actions = 100.0`** every action is multiplied by 100 and training is
destroyed (eva_bc lost two full runs to exactly this).

If confirmed, this is a **finding about the benchmark**, not a defect in our code: *the
shipped rl_games config cannot reach its own goal at the only clip value that trains.* The
fix on our side is a per-run override to `clip_actions ≈ 2.0` — an **agent**-config value, so
no `challenge/` file is touched.

**And note which way this cuts: x0-steering is immune.** The RL action there is `z`, clipped
to ±1 and squashed by `tanh`; the *arm* action still comes from the flow policy unclamped.
That is an argument for steering over from-scratch PPO that is specific to this task and did
not exist for eva_bc.

Also corrected while checking: the key path is **`agent.params.env.clip_actions`**
(`eva_rl/scripts/rl_games/train.py:166`), **not** `agent.params.config.env.clip_actions` as
`HANDOFF.md` §12 currently prints. Fixed there.

### N8 — Horizon and dataset budget follow from N2's table.

- demo length **382 steps ≈ 7.64 s**, tightly clustered (the schedule is fixed; only the
  refine differs per env).
- **`episode_length_s = 13.8` → 690 steps = 46 × 15.** Divisible by `n_action_steps`, and
  1.81× the demo length — margin for a policy that hesitates. eva_bc's most embarrassing
  false zero came from a 500-step horizon against 677-step demos.
- 500 demos × 382 = **191 k samples**, against eva_bc's 314 k. 100 k steps at batch 64 =
  6.4 M samples ≈ **33 epochs**.
- chunk 50 = 1.0 s covers 46 mm of the descent (0.93 mm/step) or 245 mm of the carry
  (4.9 mm/step).

---

## 3. THE DEMO FORMAT AND VOCABULARY

`dataset.py:2-23` is normative. What we emit:

```
data/demo_{i}/
    obs/policy   (T, 42) float32     gzip
    actions      (T,  7) float32     gzip
    train_mask   (T,)    uint8       uncompressed
  attrs: success (bool), num_samples (int), episode_kind (str),
         segments (JSON), outcomes (JSON)
```

**Segments** are the schedule phases, verbatim, so a segment id is checkable against
`ClutterExpert.schedule()`:

```
home · approach · settle · descend · predwell · close · carry · dwell · release · withdraw · final
```

(`home` and `approach` are the two new phases from N1.)

**Outcomes**, per episode — the vocabulary is deliberately *smaller* than pick-place's,
because clutter's expert has no retries, no regrasps and no recovery:

```json
{"enclosed": true, "extracted": true, "at_goal": true, "toppled": false,
 "topple_phase": null, "min_free_gap_mm": 7.4, "target_yaw_rad": 0.081,
 "pose_id": 3, "screen_score": 0.84, "spawn_seed": 11}
```

`episode_kind` is `"nominal"` for everything Stage 2 produces.

**Pool filters.** `default_demo_filter` (success only) is the whole story. eva_bc's
`nominal_pool_filter` / `recovery_pool_filter` / `_nominal_clean` key off `":g{n}"` regrasp
ids, `clean ≡ close_disp < 0.005` and `via.startswith("carry-direct")` — **none of which
exist here**. We add one clutter filter and keep the default:

```python
def clutter_nominal_filter(attrs):   # success AND the neighbours were never touched
    oc = json.loads(attrs.get("outcomes", "{}"))
    return bool(attrs.get("success")) and not oc.get("toppled", True) \
           and oc.get("max_disturb_mm", 0.0) < 1.5
```

`max_disturb_mm` is worth carrying because P22 measured a **~65 % contact rate against a
~22 % topple rate** — two thirds of successful episodes *touched* a neighbour and got away
with it. Whether training only on the untouched third helps is a pool ablation, not an
assumption. **Registered as arm `pool=strict` against `pool=default`.**

**`train_mask`.** All ones for kept episodes. Clutter's expert has no partial failures to
censor: an episode either delivers the block with everything standing, or it terminates.
The mask machinery stays wired up because N2's fallback needs it.

---

## 4. THE PORTING DIFF

`05_PORTING_MAP.md` §3 lists every site. Deltas after this review:

| site | porting map says | **now** |
|---|---|---|
| `dataset.py:37-42` | 42 / 16 / 26 / 7 | unchanged — correct |
| `dataset.py:45-98` | rewrite filters | **replace with the two filters in §3** |
| `residual_core.py:159` | drop the permutation | unchanged — **still a real bug**, `subtract_frame_transforms` already returns XYZW |
| `residual_core.py:35` | `CAN_REST_Z_IN_BASKET → 0.035` | unchanged |
| `eval_act.py:45-50` flush | redesign around `up_z` | **withdrawn (N4)** — default off, gate on/off equality |
| `report_coverage.py` | rewrite | axes are **measured free gap, target yaw, pose_id, outcome counts** |
| Phase 6 grasp bit | train an MLP | **test the threshold rule first (N5, P31)**; the MLP is the fallback |
| `--episode-length-s` | 13.8 | unchanged (N8) |

**Files needing zero edits, confirmed by reading them:** `normalize.py`, `modeling_act.py`,
`configuration_act.py`, `modeling_flow.py`, `train_flow.py`, `train_act.py`,
`residual_wrapper.py`. Every model width flows from the three `dataset.py` constants through
`train_flow.make_config`.

Our code lives in `clutter/act/` and imports the eva_bc modules; **`eva_bc/act/` itself is not
edited.** Where a constant must differ, we shadow it in `clutter/act/clutter_dims.py` and pass
it in. The one exception we cannot shadow is `dataset.py`'s module-level `OBS_DIM` —
so `clutter/act/dataset.py` is a **20-line subclass module** that re-exports
`RebotDemoDataset` with the clutter slices, rather than a fork.

---

## 5. PREREQUISITE PROBES

Each has a **pre-registered prediction**. Recording the prediction before the run is what
made P24's yaw analysis worth anything and what P19/P20 skipped.

### P27 — How short can the holds be?

*Question.* `close` is 560 physics steps. What is the shortest value that preserves
enclosure and topple rate? Same for `predwell` / `dwell` / `release`.

*Design.* Paired: one reset, snapshot, restore between arms. Close ∈ {560, 400, 280, 200,
140, 80, 40} physics steps, everything else fixed. 3 pose draws × 2 spawn batches × 128 envs.
Score enclosure, topple, at-goal, success — the **conjunction**, per the P25 lesson.

*Prediction.* The fingers travel 26.5 mm each at `stiffness = 2000, damping = 40`. P19 saw
the slam resolve in the first few milliseconds. **Predicted: flat from 560 down to ~120
physics steps (15 env steps), then enclosure falls off a cliff.** If it is flat to 80, the
close drops from 70 env steps to 10 and N2's ambiguity is 6× smaller.

*Falsifier.* If success degrades **gradually** from 560, the close is doing something other
than closing — most likely letting the row settle after being disturbed — and shortening it
trades BC learnability against expert success. That is a real trade and would be reported as
one, not resolved silently.

### P28 — Are independently screened poses in the same IK branch?

*Design.* `plan()` under 8 spawn seeds, `plan_full=False` (grasp pose only, ~1/10 the cost).
Report per-joint L∞ pairwise distances, `o_align`, `wrist_z`, `screen_score`, and the
resulting cluster structure.

*Prediction.* **Registered: they are one branch.** The CEM is seeded from the home pose and
the hinge cost is dominated by position; the wrist is confined to a 12 mm gap at
y ≈ −20 mm, which is a narrow basin. **Predicted max pairwise L∞ < 0.35 rad.** If wrong,
§2/N3's decision rule fires and demo diversity is capped at one cluster.

### P29 — The `home → chain[0]` approach segment

*Design.* Two candidates:
1. **joint-space lerp** `q0 → qs[0]` over K steps;
2. **dense Cartesian** through the same machinery as every other segment.

For each: audit achieved TCP against the straight line (P17's method), whole-arm
`box_penetration` over the whole path, per-phase hazard.

*Prediction.* **Registered: the joint-space lerp is clean.** The path is entirely above
z = 125 mm while the row tops out at 67 mm, so even a large Cartesian excursion cannot reach
the blocks. **Predicted: penetration 0, hazard 0 %, and the lerp is chosen** — the dense
solve exists as the fallback, not the default. Deviation from the Cartesian line is *not* a
failure criterion here; only penetration and hazard are. (Getting that distinction wrong is
how P14 ended up scoring poses that were inside the target.)

### P30 — Is a π roll of `joint6` a symmetry? *(added after P28's first four draws)*

*Why.* P28's draws 0/1/3/4 agree on joints 1–5 to a few hundredths of a radian and disagree on
`joint6` **by about π** (−1.946 + π = +1.196). Everything `plan()` scores is blind to that
difference — the CEM's orientation term is `|o_hat·o_des|`, deliberately **sign-free** because a
parallel jaw is symmetric, and `box_penetration` takes a max over body origins, so swapping
which finger is where changes nothing.

But the measured finger meshes are **not** symmetric about the roll axis. In y they are exact
mirrors (free to swap); in **z they are identical, not mirrored** — the blade runs −58.7 mm one
way and +34.7 mm the other. A roll that maps left onto right's position also flips that profile,
and the two configurations then sweep volumes differing by up to **24 mm**, against the **7.8 mm**
of margin that P22 showed is the entire remaining failure mode.

*Prediction, registered.* **They differ.** TCP and axes agree to <1 mm / <0.005; end-to-end
success differs by **more than 5 points**. If so, the wrist roll is a **free, unexploited degree
of freedom that `plan()` currently picks at random** — and screening it is worth what `screen = 4`
was. *Falsifier:* identical scores ⇒ it is an alias, canonicalise `joint6` in `plan()`, and P28's
apparent diversity was an artifact of the search.

### P31 — The grasp-bit threshold *(after the first BC checkpoint)*

*Design.* Positives: expert post-close frames. Negatives: on-policy closed-on-air frames from
BC rollouts. Score the two-input rule from N5, and eva_bc's 5-input MLP, on **FPR at the
operating point**, holding the **0 % FPR** bar.

*Prediction.* **Registered: the rule clears 0 % FPR and the MLP is unnecessary.**

---

## 6. GATES

Ordered. Do not reorder — each one exonerates a layer so the next failure is attributable.

| gate | test | threshold | if it fails |
|---|---|---|---|
| **2a** | teleport to `chain[0]`, run the schedule under `env.step`; paired against physics-only on the same spawns | within **±5 points** | the executor differs — MDP terminations, action quantisation, or the 20 ms action hold. Diagnose before writing any demo. |
| **2b** | replace the teleport with the P29 approach | within **±5 points** of 2a | the new segment is the cause, and nothing else can be |
| **2c** | HDF5 shape assertion passes; `train_flow.py --steps 200` runs and loss decreases | — | a format bug |
| **2d** | `eval_act.py` runs, `obs.shape == (n, 42)`, success finite | — | plumbing |
| **2e** | **≥ 3 training seeds**, champion on a **held-out** spawn seed, pooled **≥ 128** episodes | **≥ 60 %** | see the decision rules |
| **2f** | flush-on ≡ flush-off, fixed seed | identical episodes | the detector fires on normal manipulation (N4) |

**The BC ceiling, recorded in advance, as `DR2` requires.** The expert is at **73.7 %**.
eva_bc's flow BC lost **21.6 points** to an 85.7 % expert. A proportional loss puts us at
**~52 %**; the absolute-gap reading puts us at **~52 %** as well. Both agree, which is
reassuring but they are not independent.

```
≥ 60 %        better than eva_bc's transfer efficiency — Gate 2 passes outright
52 – 60 %     ON TREND. Gate 2 is judged on the taxonomy, not the number alone.
45 – 52 %     below trend; suspect the N2 static-observation ambiguity first
< 45 %        a PORTING DEFECT, not a learning limit. Stop and find it.
```

This is a **wider** band than the porting map's flat "≥ 60 %", and it is wider on purpose:
73.7 → 60 would be a 13.7-point loss, i.e. **better transfer than eva_bc ever achieved**, and
writing that down as the pass mark would have set us up to call an on-trend result a failure.

---

## 7. TRAINING PROTOCOL

Fixed before any run, per the standing conventions:

```
demos          8 spawn seeds × 128 envs = 1024 episodes, ~74 % succeed → ~750
               keep 500, balanced across seeds
               POSE SELECTION USES A DIFFERENT SPAWN BATCH FROM DEMO GENERATION
               (P26 protocol — screening on the batch you then record is leakage)
chunk / exec   50 / 15          n_action_steps 15 is NOT negotiable (59.4→0 % at 15→1)
inference      10 Euler steps
optimiser      AdamW, lr 1e-4, wd 1e-4, batch 64, 100 k steps
seeds          --seed 1,2,3 (train_flow.py is unseeded by default — eva_bc's 26.6-point
               variance was measured on unseeded runs)
eval           Rebot-ClutterExtract-Play-v0, episode_length_s 13.8, --no-flush,
               ≥128 episodes, champion chosen on a HELD-OUT spawn seed
arms           pool = default | strict (§3)
```

**Costs, extrapolated from eva_bc's 12 GB card and therefore provisional** — Q7 (throughput
and VRAM at N = 16…2048 on this 10 GiB card) is *still* unmeasured and now blocks demo sizing:

| item | eva_bc | expected here |
|---|---|---|
| flow BC training | 35–40 min, 1.0 GB | similar — 703 k params, state-only |
| policy eval, 64 eps @ 16 envs | ~5 min | faster: 382-step demos vs 677 |
| demo generation | 3 h 47 m / 504 eps | **much** faster — batched 128-env rollouts, but `plan()` + `screen=4` dominates |

**Q7 runs first.** It has been outstanding since Stage 0.

---

## 8. THE FAILURE TAXONOMY (MANDATORY)

Gate 2e is not a number, it is a number **plus** this table. eva_bc's champion taxonomy
(34 grasp-phase / 12 carry / 0 drop, "74 % of the residue is grasp-alignment shaped") is what
made every later decision informed rather than guessed.

Per failed episode, from the latched per-episode record:

| bucket | test |
|---|---|
| `topple_before_close` | `toppled` and the first `up_z < 0.75` precedes the first commanded-close |
| `topple_at_close` | `toppled` within 20 steps of the first commanded-close |
| `topple_in_carry` | `toppled` after the block passed `EXTRACT_Z` |
| `never_enclosed` | commanded closed but `|gap − 0.036| > 0.012` at every step |
| `enclosed_then_lost` | enclosed at some point, not at the end |
| `never_lifted` | max target z < `EXTRACT_Z` = 90 mm |
| `lifted_never_placed` | cleared `EXTRACT_Z`, never `at_goal` |
| `timeout_in_motion` | horizon reached with the arm still moving |
| `grip_chatter` | `toggles_per_episode > 2` (the expert's value is 1) |

**The diff that matters** is expert-vs-BC on the same buckets. The expert's own residue is
~100 % `topple_at_close`. Any bucket the BC policy populates that the expert does not is a
*learning* failure and points at N2, N3 or N6 specifically.

---

## 9. DECISION RULES, PRE-REGISTERED

- **DR2-a.** Gate 2a or 2b fails → stop, diagnose, do not generate demos. A demo set built on
  an unverified executor is worse than none: it looks like data.
- **DR2-b.** Gate 2e in `[52 %, 60 %)` with a taxonomy dominated by `topple_at_close` →
  **proceed to Stage 4.** The policy inherited the expert's own failure mode and BC is not the
  bottleneck.
- **DR2-c.** Gate 2e in `[52 %, 60 %)` with a taxonomy dominated by `never_enclosed` or
  `grip_chatter` → **do not proceed.** Those are learning failures; fix them (N2 hold
  shortening, N6 pool ablation) and retrain. Stage 4 on a base that cannot grasp is wasted.
- **DR2-d.** Gate 2e `< 45 %` → porting defect. The first three suspects, in order: the
  obs/action alignment in the collector (`ep_obs` is the observation **before** the step),
  the normalizer stats, and `n_action_steps` at eval ≠ at training.
- **DR2-e.** If P27 shows the close cannot be shortened, N2's `train_mask` fallback is
  **required**, not optional, before Gate 2e is believed.

---

## 10. WHAT WOULD FALSIFY THIS PLAN

Named in advance so they are recognisable when they happen:

1. **Gate 2a fails by a lot** (> 15 points). The most likely cause is that the 20 ms action
   hold interacts with the close differently from the physics-only drive — `_move` writes a
   fresh target every 8 substeps, and `env.step` does exactly the same, so they *should*
   agree. If they do not, something about the MDP (action-rate reward? no) or the reset is
   different, and every Stage-1 number is measured through an executor the policy will never
   use. That would be the most expensive finding available and it is worth the run to exclude.
2. **P28 finds multiple IK branches.** Demo diversity collapses to one cluster and the
   expected BC ceiling drops with it.
3. **The static-observation ambiguity is worse than modelled** — e.g. the policy lifts during
   `predwell`, before the gripper has closed at all. Signature: `never_enclosed` dominant with
   a normal topple rate.
4. **`|a| > 1` turns out to be false** because the arm reaches the goal with `joint1 > −0.5`.
   Then N7's benchmark finding evaporates. **Measured in Gate 2a, not asserted.**

---

## 11. ORDER OF WORK

```
1.  Q7          throughput / VRAM at N = 16…2048           [outstanding since Stage 0]
2.  P28         IK-branch clustering of screened poses      cheap, gates demo diversity
3.  P30         is a pi roll of joint6 a symmetry?          raised BY P28; gates P28's verdict
4.  P27         hold-duration sweep                         gates the schedule
5.  P29         home → chain[0] approach segment            gates Gate 2b
6.  collector   clutter/act/collect_demos.py + Gate 2a, 2b
7.  demos       8 seeds × 128 envs → ~500 kept
8.  dataset     clutter/act/dataset.py + Gate 2c
9.  train       3 seeds × 100 k steps × 2 pools
10. eval        Gate 2d, 2e, 2f + the §8 taxonomy
11. P31         grasp-bit threshold (needs the base)
```

Steps 2–4 are all probe work in the existing harness and can be written before any of the new
plumbing exists. **Step 1 is first because every later step's env count depends on it.**
