# VISION_PLAN — a visual **flow** policy for `Rebot-PrecisionSlot-*`, 80 %+ with no privileged state

*v2, rewritten 2026-08-03 19:10 after pulling **EXP08** (`c10bf1a`). Big Will: "there is a new
commit in eva_bc that implements the visual backbone, lets use that! Yes we want our visual policy
to still use flow!" — both settled, and the plan below is built on that stack.*

---

## 0. The bar and the contract

**Target: ≥ 0.80 later-cohort success on `Rebot-PrecisionSlot-v0`** (128 ep / 32 envs, seed 777,
`success_rate_later`) — the same protocol as every other number on this project.

**Student input — nothing else, ever:**

| dims | term | student |
|---|---|---|
| `[0:8]` | `joint_pos_rel` | ✅ encoders |
| `[8:16]` | `joint_vel_rel` | ✅ encoders |
| `[16:23]` | `block_pose_in_root` | ❌ **privileged** |
| `[23:27]` | `slot_frame` (depth, lateral, yaw, **is_inserted**) | ❌ **privileged — this is the success predicate itself** |
| `[27:34]` | `last_action` | ✅ our own command |

**Student = wrist D405 RGB 160×90 + workspace D455 RGB 160×90 + 23-D proprio
(`obs34[0:16] ⊕ obs34[27:34]`).** Identical in shape to EXP08's contract, which is a gift: the
ported code already asserts 23.

The **teacher stays privileged**, and that is not cheating — only the deployed policy's inputs are
constrained. Enforcement, copied from EXP08 §4 rather than reinvented: a single
`STUDENT_PROPRIO_SLICES` constant, `assert proprio.shape[-1] == 23` in both collection and
training, the privileged array kept in shards but flagged teacher-only and **never** referenced by
the student loader, and a programmatic audit in Gate A.

---

## 1. What EXP08 hands us (this is most of the build)

`c10bf1a` added a complete vision flow stack for the *pick-place* task:

| file | what it is | our use |
|---|---|---|
| `act/modeling_flow_vision.py` | **`FlowMatchingVisionPolicy`** — the flow head with the env_state token replaced by ResNet camera tokens (ACT image path: backbone → layer4 → 1×1 conv → one token per spatial cell + 2-D sinusoidal pos-emb). At 160×90, resnet18 gives 5×3 = 15 tokens/camera; encoder sequence = `[robot_state, wrist ×15, workspace ×15]` = **31 tokens**. ImageNet normalisation inside `encode()`, so shards stay uint8. | **port to `slot_act/`** |
| `act/dataset_vision.py` | shard dataset; handles both executed-action and DAgger chunk-label formats | port |
| `act/train_flow_vision.py` | trainer | port |
| `act/eval_flow_vision.py` | sim eval | port |
| `experiments/exp08_collect.py` | champion rollout collection into `.pt` shards | port + **re-point at the slot env** |
| `experiments/exp08_dagger_collect.py` | student-drives / teacher-labels DAgger collector | port |

This answers Big Will's directive exactly: **flow head, vision backbone, no ACT-vs-flow
substitution.** The `slot_act/` port has a precedent and a checker (`scripts/check_port.py`,
16/16), so the same discipline applies.

**One thing does not port: the env.** `exp08_collect.py` targets a registered
`Rebot-PickPlace-Vision-Play-v1`. That task **does not exist in this `eva_rl` checkout** — I
checked; `eva_rl` is clean and has no `pick_place_vision` package. Which suits us: we attach the
cameras **post-`parse_env_cfg`**, on our side, exactly as `--slot-dx` and `--arm-jitter` already
do. `eva_rl` stays untouched, as it has all project.

---

## 2. The cameras, from Big Will's `e56e7df`

* **`WORKSPACE_CAM_CFG`** — D455, 90° HFOV, world pose `(0.9, 0.0, 0.6)` aimed at ≈`(0.25, 0, 0.05)`.
  Our slot sits at `(0.245, 0.0)`: framed essentially as-is.
* **`WRIST_CAM_CFG`** — D405, 84° HFOV, on `gripper_end`, tilted −30° about camera-local X;
  optical axis **1.8° off the TCP at 0.171 m**.

Both attach to the slot env unchanged: `WRIST_CAM_CFG.prim_path` is built from `_GRIPPER_END` =
`<base>/link1/…/link6/gripper_end`, and `precision_slot_env_cfg.py:293` builds its `ee_frame` from
the **identical** chain. Verified, not assumed.

**EXP08 already validated this mount through real grasps** (step 0, v4): cam→TCP pinned at 0.170 m
across all 300 samples, active object in FOV **79 %** of its active phase, out only during the
first ~0.7 s swing — and near-centred (|h| ≲ 8°) through descent, grasp and carry. Depth at frame
centre falls 0.32 → 0.10 m through the descent, which EXP08 calls an "ideal terminal-servoing
signal". **That is exactly the signal our insertion needs**, so I expect the mount to transfer.

---

## 3. Four lessons inherited from EXP08 — each already cost that session hours

These go into the code as **assertions**, not as things to remember:

1. **Never read `Camera.data.pos_w/quat_w_*` on a link-mounted camera.** Isaac Lab 3.0 never
   recomposes the fabric `worldMatrix`, so the pose is **frozen at spawn** while the *render*
   correctly tracks the link. EXP08 retracted a whole batch of FOV numbers to this. Derive the
   pose as (gripper body physics pose) ∘ (camera-in-body at reset), or use depth.
2. **Every camera needs a frame-freshness check over ≥ 100 steps.** With multiple `Camera`
   sensors, EXP08 saw a second camera's buffer **freeze after ~35 steps** while the arm provably
   crossed its view. `lift_vision`'s smoke test only checked shapes and would not have caught it.
   Per-camera temporal mean-abs-diff, on a moving robot, or we collect a dataset through a dead
   lens.
3. **DLSS is a live suspect for GPU-load-dependent frames.** EXP08's student scores 67.2 % in its
   own eval and ~40 % inside the DAgger driver; the bisection verdict is that frames depend on GPU
   load, with Isaac Lab's default `antialiasing_mode = DLSS` the prime suspect (160×90 is far
   below DLSS's ~300 px minimum input). **We set AA off from the first frame** — free insurance,
   and it keeps our data out of the regime that is currently blocking their Gate D.
4. **Discard the first episode per env.** EXP08 does it for frame sync; we already do it for the
   PhysX warm-start bias. Same rule, two reasons.

---

## 4. Teacher: the champion, not the scripted expert

EXP08 distils from the champion policy. Same choice here, for a reason specific to DAgger: the
teacher must label **arbitrary student-visited states**, and our scripted expert plans open-loop
from the initial state — it cannot label a state the student wandered into. The champion
`runs/bc_armB_seed0/ckpt_final.pt` is a closed-loop policy at **0.979** and can.

---

## 5. What the number will probably be — calibrated on EXP08, not on hope

EXP08's pick-place ladder, measured:

| stage | result |
|---|---|
| teacher in the vision env (Gate B) | **93.75 %** |
| vision flow-BC (Gate C) | **67.2 %** — a **−26.5 pt** drop |

If that delta transferred verbatim, our 0.979 teacher would give **≈ 0.71 — under the 0.80 bar.**
So: **plan for BC + DAgger from the start**, and treat vision-BC alone as unlikely to clear it.
Saying that now beats discovering it after a night of training.

Two reasons our delta may differ, in both directions:

* **Easier than pick-place:** the slot **never moves** (`SLOT_CENTER = (0.245, 0.0)`, welded). The
  only randomisation is the block spawn — x ± 20 mm, y ± 30 mm, yaw ± 0.35 rad. Pick-place had two
  cans *and* a randomised basket. Our student has to perceive far less.
* **Harder than pick-place:** a 1.5 mm per-side clearance versus dropping into a basket. Session 6
  showed lateral tolerance is a **step function at the clearance** — vision noise that a basket
  absorbs, this slot does not.

Net guess: the loss concentrates in the **grasp** (needs real block localisation) while the
insertion — nearly fixed, and the part the state policy does at 0.979 without seeing the goal —
should survive.

---

## 6. Gates, cheapest first. Each can stop the plan.

**G0 — camera render + freshness (~20 min, no training).** Attach both cameras, drive the
**champion** for ≥ 150 steps in 1 env, and check: per-camera temporal mean-abs-diff stays above
threshold every step (lesson 2); no near-uniform frames (std < 2 tripwire); wrist depth-min stays
2–5 cm (proves the render tracks the link, lesson 1); and save stills at spawn / grasp / staging
(`x = 0.165`) / mid-insertion for **Big Will to review**. The question that matters: *once the
block is held, does the wrist view still show the slot?*

**G1 — state-parity + throughput (~20 min).** Same seed, cameras on vs off: the privileged 34-D
trajectory must match near-bitwise under the champion (states, not pixels — renders are
non-deterministic). Catches the camera load perturbing physics. Plus FPS at 1/16/32 envs, which
sets the collection budget from measurement.

**G2 — collection audit (~1 h).** Champion driven in the vision env, standard protocol, first
episode per env discarded. **Recorded success must land ≈ 0.979 (accept 0.93–1.00).** If it does
not, suspect camera-load physics jitter or a frame-sync bug *before* blaming the champion — that
is EXP08's exact experience.

**G3 — the blind control (~1 h) — the load-bearing gate.** Train the **same
`FlowMatchingVisionPolicy`** on 23-D proprio with the **images zeroed**, and evaluate. The slot is
fixed, so a blind policy can execute the whole insertion; it just cannot know where the block is.
An 89 mm gripper opening on a 30 mm block absorbs a lot of a ± 30 mm spawn error.
* blind ≥ 0.60 → the eval barely tests vision; widen the spawn box before claiming anything.
* blind ≈ 0.10–0.35 → the comparison is meaningful.
**Without this number a visual result of 0.85 is unfalsifiable.**

**G4 — vision flow-BC.** Chunk 50 / execute 15 (load-bearing; shortening it collapsed pick-place
59.4 → 0 %). Decision rule, mirroring EXP08's: ≥ 0.50 → go to DAgger. 0.35–0.50 → one architecture
iteration, then go anyway. < 0.35 → stop and diagnose per-phase.

**G5 — champion-DAgger to the bar.** Student drives, champion labels student-visited states.
Target **≥ 0.80**. EXP08 guesses 2–3 rounds for its equivalent.

**G6 — two seeds before any headline.** Standing rule; this project has retracted two single-seed
claims already.

---

## 7. Beliefs, pre-registered

1. **Blind (proprio-only) lands 0.10–0.35.** It executes a perfect insertion into thin air when it
   misgrasps.
2. **Vision flow-BC lands 0.55–0.75** — EXP08's Gate C shape, nudged up because our scene is
   simpler. **Under the 0.80 bar**, hence G5.
3. **DAgger clears 0.80** within 3 rounds.
4. **Most residual failures are `never_entered` at the staging waypoint, not `never_lifted`.** If
   `never_lifted` dominates instead, the failure is *perception* (finding the block), not
   precision, and the fix is the workspace camera or resolution — not more DAgger.
5. **The wrist camera dominates at the insertion, the workspace camera at the grasp.** A
   one-camera ablation should move the failure *buckets*, not just the rate.
6. **A visual policy that matches 0.979 means a leak.** Audit the observation builder before
   celebrating.

---

## 8. Build order

1. Port `modeling_flow_vision.py`, `dataset_vision.py`, `train_flow_vision.py`,
   `eval_flow_vision.py` → `slot_act/`; extend `scripts/check_port.py` to cover them.
2. `slot_act/cameras.py` — post-parse attach of both cams at 160×90,
   `update_period = decimation·dt = 0.02`, AA off, `env_spacing` raised (lift_vision uses 6.0 so
   neighbour envs stay small in a 90° view).
3. `scripts/collect_vision.py` — champion → `.pt` shards (images uint8, 23-D proprio, actions,
   privileged 34-D flagged teacher-only).
4. G0 → G1 → G2 → G3 → G4 → G5.

**Storage:** 599 steps × 2 cams × 160×90×3 = **51.8 MB/episode** raw. A 256-episode gate pool ≈
13 GB; 1024 ≈ 53 GB. Disk has 1.7 TB free, so this is an I/O question, not a capacity one.

---

## 9. What could waste it, and the guard

| risk | guard |
|---|---|
| wrist view blind once the block is held | **G0**, stills for Big Will, before any collection |
| a camera's buffer silently freezes | **G0** freshness check — EXP08 lost hours to exactly this |
| DLSS/AA corrupts frames under GPU load | AA off from frame one; EXP08's Gate D is blocked on it *right now* |
| cameras perturb the physics | **G1** state-parity against the camera-free env |
| the task does not need vision | **G3** blind control |
| a privileged dim leaks | one obs builder, asserted 23-D, shared with the blind arm; Gate-A-style audit |
| one seed quoted | **G6** |
| `eva_rl` edited | post-parse patching only, as all project |

---

## 10. RESULTS — port complete, **G0 PASSED** (2026-08-03 19:15)

### 10a. The port

`act/` → `slot_act/`, following the existing port discipline:

| file | how |
|---|---|
| `modeling_flow_vision.py` | **verbatim** — one string differs (a doc reference). It depends only on `configuration_act` / `modeling_act` / `modeling_flow`, and `slot_act/modeling_flow.py` is **byte-identical** to `act/modeling_flow.py` (checked), so a verbatim copy is the honest port: future divergence shows up as a diff, not as drift. |
| `dataset_vision.py` | teacher array renamed `obs41` → `obs34`; student contract unchanged at 23-D |
| `train_flow_vision.py`, `eval_flow_vision.py` | `act.*` → `slot_act.*` imports |
| **`cameras.py`** (new) | post-`parse_env_cfg` attach + the single `student_proprio()` that every consumer must use |

CPU-validated: `student_proprio` on a 0…33 ramp keeps `[0:16] ⊕ [27:34]`, contains **no** value
in `[16, 27)`, and `audit_no_privileged` raises on a planted `obs34` key.

### 10b. G0 — cameras render, keep rendering, and track the link

Champion driving, 1 env, 300 steps, `-v0`, seed 777:

```
wrist_cam      frozen 0/299   diff min/med 30.0/40.7   std min 39.0   near-uniform 0   ALIVE
workspace_cam  frozen 0/299   diff min/med 39.3/40.2   std min 44.0   near-uniform 0   ALIVE
static-render diff (no physics step): wrist 0.0  workspace 0.0
wrist depth-min 0.0244–0.0504 m, out of band 0/300   RENDER TRACKS THE LINK
block x 0.1588 → 0.2141   (stage_x 0.165, slot 0.245)
```

* **Neither camera freezes.** EXP08's second-camera artifact does not reproduce here.
* **`wrist depth-min` stays 2.4–5.0 cm for all 300 frames** — the gripper housing sits
  permanently a few cm in front of the D405, which is only possible if the *render* rides the
  link. This is the check that works despite `Camera.data` poses being frozen at spawn.
* **The render is deterministic**: two renders at an unchanged physics state differ by **0.0**,
  so the ~40/255 per-step diffs are genuine scene motion, not sampling noise. Frames carry no
  per-frame jitter for the student to fight.
  *Caveat, stated rather than buried:* the probe renders twice at a **settled** state, and a
  temporal accumulator would converge there too. It is evidence that AA-off gives a clean feed,
  not a disproof of temporal-AA effects during motion — which is moot for us since we run AA off,
  but it is not the discriminator EXP08 needs for its own blocker.

**A bug in the instrument, caught by the instrument.** The first G0 run reported *both* cameras
`frozen 299/299`. The cameras were fine: `rgb()` returned `data[..., :3].to(uint8)`, and
`.to(uint8)` on already-uint8 data is a **no-op that returns the same tensor** — so "last frame"
and "this frame" were the same buffer and the diff was structurally zero. `rgb()` now clones, and
the docstring says why. Had the check been written only as "does the shape look right", the bug
would have shipped in the opposite direction: a real freeze would have gone unnoticed.

**Stills for Big Will:** `slot/runs/vision_g0/wrist/step_*.png` and `.../workspace/step_*.png`
(12 each at 25-step spacing). The question that decides the mount: **once the block is in the
gripper, does the wrist view still show the slot?**

---

## 11. EXP08's renderer verdict, retested here — we take a DIFFERENT route, with evidence
*2026-08-03 19:45, after pulling `2846bb8`.*

### 11a. The four errors in that log, and what each one costs

EXP08 closed its renderer investigation. Read as a list of traps:

1. **Train/test render mismatch is catastrophic, not gradual.** They trained a student on DLSS
   frames and evaluated it with AA off: **0.0 % / 1.6 %**. Not degraded — floored. *Guard: the
   render configuration is part of the dataset. Collection, training and eval must share it, and
   changing it invalidates every shard and every checkpoint.*
2. **Optional GPU work inside a policy-driving loop perturbs the frames.** Their in-loop DAgger
   collector dropped the student from ~80 % to ~40 % while driving; the fix was to record raw
   state and compute teacher labels **post-hoc** (validated exact, max abs diff 7.8e-8). *Guard:
   collection loops do buffer reads and device clones only; every label is computed after the
   rollout stops.*
3. **Renderer cold start contaminates the first round.** Gate C's official 67.2 % includes a
   DLSS-cold-start round 1 at 6/16; the warm level is ~78–83 %. *Guard: our existing
   discard-first-episode-per-env rule, plus a warmup discard, and never quote a number whose
   first round was cold.*
4. **Data collected under a suspect configuration is quarantined, not mixed.** Their v1 in-loop
   DAgger data was excluded from training outright.

### 11b. My own G0 claim was wrong, and this is how it was wrong

G0 reported "static-render diff = **0.0** — deterministic render", and I told Big Will the frames
were clean. **That probe could not have detected the artifact.** It called `sim.render()` twice
*without advancing the frame index*, and EXP08's jitter is frame-index-deterministic: rendering
the same index twice returns the same image by construction. An instrument that cannot fail.

Same class of mistake as the uint8-aliasing bug earlier in this session, and worth naming: **both
were checks whose passing condition was guaranteed by their own implementation.**

### 11c. Retested properly — step the sim (advancing the frame index) on a held pose

`scripts/vision_shimmer_probe.py`. Per-pixel **temporal** std of the 160×90 frames the policy
consumes, 40 frames, identical trajectory in every row (same seed, same `max|joint_vel| = 0.587`
during the hold, so the comparison is like-for-like):

| render | wrist temporal std | workspace | pixels > 2 | first→last drift |
|---|---|---|---|---|
| **1× (what G0 shipped)** | **35.24** | 36.84 | 99.8 % | 37.9 |
| 4× supersample | 9.06 | 9.26 | 99.9 % | 10.7 |
| **8× supersample** | **4.68** | 4.69 | 99.4 % | 5.8 |
| DLSS @ 1× | 3.89 | 3.73 | 70.0 % | 13.6 |

**EXP08's finding reproduces exactly**: 35.2 / 36.8 against their 31–37 on a static scene, from a
completely independent implementation. Their diagnosis is right and it applies to this rig.

**But their conclusion does not have to be ours.** They compared non-temporal modes only at 1×,
where the choice really is "DLSS or 14 % shimmer". Supersampling is a third option, and it works
for the reason the physics predicts: the jitter is per-pixel and near-independent, so box-
averaging k² samples cuts its temporal std by ~k — measured 35.2 → 9.1 → 4.7 for k = 1, 4, 8.

**8× supersample buys DLSS-class temporal stability (4.68 vs 3.89) with no temporal filter at
all** — therefore no frame history, therefore nothing for GPU load to perturb. That is precisely
the failure mode that floored their DAgger driver and forced the post-hoc redesign. Note also
DLSS's *drift* is 13.6 against 5.8 for 8×: its frames stay smooth but move more over the hold,
consistent with a temporal accumulator still converging.

Cost is not the obstacle: 150 steps took 30 s at 4× and 32 s at 16×, one env. G1 measures it at 32.

### 11d. Decisions, recorded

* **Render at 8× and box-average to 160×90.** `SUPERSAMPLE = 8` in `slot_act/cameras.py`.
* **AA off, DLSS off.** We do not need a temporal filter, so we do not take its coupling.
* **The render config is part of the dataset contract.** It gets written into every shard and
  asserted at train and eval time; a mismatch must raise, not silently score 1.6 %.
* **Post-hoc labels, and no optional GPU work in any policy-driving loop** — adopted from EXP08
  wholesale rather than rediscovered. Even with no temporal filter, this costs us nothing.
* **Warmup discard on top of the first-episode-per-env rule.**

One thing we still inherit and cannot dodge: **8× does not reach zero** (4.68). If the vision
student underperforms and the failures look like perception rather than precision, residual
shimmer is a live suspect and 16× is one constant away.
