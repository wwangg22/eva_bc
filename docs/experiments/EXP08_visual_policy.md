# EXP08 — pick-place VISUAL policy (distill the 91.4% champion to cameras)

*2026-08-03. Status: STEP 0 IN PROGRESS — this doc pre-registered before any code
(project convention). The full pipeline design (§3) is NOT final: it is the
recommendation brought to Big Will for the design conversation. Only step 0 (a
mechanical render check) runs before that conversation.*

## Question

Can a camera-based policy (wrist D405 + workspace D455, no privileged object
state) reach >90% on the pick-place suites — matching the state-based champion —
via teacher-student distillation from the 91.4% stack?

## Context / anchors

- Champion (teacher): frozen flow-BC base `runs/exp03_N3/ckpt_final.pt` +
  steering head `runs/exp07_steer/s1_seed1/nn/exp07_steer.pth`, deterministic:
  **91.4% pooled** (89.1 @seed42 / 93.8 @seed123, 64 eps each, 30 s, drop-term off).
- Wrist D405 mount fixed 2026-08-03 (reBot_RL `e56e7df`): −30° about camera-local
  X, optical axis 1.8° off TCP, camera→TCP 0.171 m, →fingertips 0.208 m. Picked by
  Big Will from static start-pose renders ONLY — behavior through a grasp unverified.
- Workspace D455: `WORKSPACE_CAM_CFG`, 0.9 m back, 40° down, unchanged.
- Control at 50 Hz (sim 400 Hz, decimation 8); episodes 1500 steps.

## Step 0 (pre-registered): grasp-sequence strip through the NEW wrist cam

**Why:** the −30° pick was made from start-pose stills. If the can leaves the wrist
view during approach/grasp/carry, every frame of collected data inherits a blind
camera — cheap to re-pick the tilt now, expensive after collection.

**Protocol:** drive the champion stack (deterministic, exact ladder eval protocol:
drop-term off, 30 s, window-aligned) in **1 env**, wrist cam at 640×360 grafted
onto the Play task scene (spawn-time OffsetCfg — the training-faithful path;
runtime pose writes render blank, EXP07-era lesson). Save a wrist still every
**25 steps (0.5 s)** for 3 episodes at spawn seed 123 (3/3 success in the close-up
videos, so the strip should cover approach→grasp→carry→place for both cans).
Per frame, log numerically (no image viewing by Claude — Big Will reviews):
frame std (near-uniform tripwire <2.0), mean abs pixel diff vs previous frame,
camera→each-can distance, angle of each can off the optical axis (in-FOV proxy:
HFOV 84° → half-angle 42°; V half-angle ≈27° at 16:9), placed count.

**Beliefs (before running):**

1. The can being grasped stays inside the wrist FOV through approach and grasp
   (off-axis angle < ~40°) and the strip shows it clearly until the fingers close.
   Risk flagged: during CARRY the can sits at/below the fingertips (~0.21 m,
   inside D405 range) but may be partially finger-occluded — acceptable if visible
   at approach + grasp moments.
2. No near-uniform frames (std < 2.0) at any capture point — the −30° view never
   points into the wrist housing or empty sky during a normal episode.
3. The NON-target can and the basket enter the wrist view only incidentally; the
   workspace D455 is what carries scene-level context. (This shapes the obs design:
   wrist = servoing detail, workspace = task layout.)

**Decision rule (pre-registered):** Big Will reviews the strips. If the active can
is out of frame at the grasp moment → re-run the fine tilt sweep around −22.5°
and re-pick BEFORE any data collection. If frames go near-uniform → mount bug,
debug before proceeding. Otherwise → proceed to the design conversation (§3).

## Step 0 artifacts

- Script: `experiments/exp08_wrist_strip.py` (reBot_ACT)
- Output: `runs/exp08_vision/wrist_strip_seed123/ep{k}/step{NNNN}_placed{p}.png`
  + `summary.txt` (per-frame numerics + per-episode success)

## §3 Design — AGREED with Big Will 2026-08-03

**DECIDED: teacher-student distillation from the 91.4% champion** (Big Will
asked whether to instead rerun the expert pipeline — planner demos → flow-BC →
DAgger → steering — with vision obs; agreed NO after this reasoning):

1. Expert demos are multi-modal (planner mode choices) — exactly what capped
   state BC at 55.5%; the EXP07 steering RL is what fixed mode selection
   (+36 pts). Re-running that pipeline on vision re-imports the plateau, and the
   rescue (steering RL) would then run WITH rendering in the loop at 4–16× fewer
   envs. Champion rollouts have mode selection already resolved — the student
   inherits it via plain BC.
2. DAgger labels: champion labels any student-drift state at a network forward's
   cost and handles its own distribution at ~91%; the planner's takeover ceiling
   from policy-visited states is only 68% — weakest exactly where DAgger needs
   labels.
3. The EXP07 steering head consumes privileged obs56 — deployable on a state
   teacher, not on a vision student; champion distillation keeps ALL privileged
   inputs on the teacher side at collection time.
4. Collection audit gate for free: recorded episodes must reproduce ~91%.
5. Ceiling caveat (honest): pure distillation tops out at the champion (~91.4%).
   If BC+DAgger stalls short of 90%, the finisher is x0-steering on the VISION
   base (recipe is base-agnostic) — fallback, not plan. Do NOT mix expert demos
   into the student data (re-introduces conflicting modes); expert-labeled
   recovery segments only as a targeted later supplement for beyond-champion
   robustness.

(original recommendation notes below, kept for the record)

1. **Distillation, not RL-from-pixels.** lift_vision RL-from-pixels needed
   curriculum surgery and still lagged; rendering cuts env counts 4–16× (hurts RL
   far more than supervised collection); we own a 91.4% teacher + proven flow-BC
   recipe + proven steering finisher.
2. **Student obs:** wrist D405 + workspace D455 RGB at ~160×90 + non-privileged
   proprio (the 41-D base obs MINUS the two can poses + placed flags — exact split
   to settle). No privileged state in the student.
3. **Architecture fork to settle:** (a) flow-BC head on own-CNN features, chunk 50 /
   execute 15 — keeps the ENTIRE EXP07 steering recipe applicable on the vision
   base later (recommended; EXP02 showed chunk commitment is load-bearing); vs
   (b) the simpler Nature-CNN per-step student from scripts/distillation.
   Gotcha carried over: Isaac Lab `image_features` resnet18 outputs the FULL
   1000-d classifier vector, not 512-d pooled — prefer raw pixels + own CNN.
4. **Protocol:** same spawn suites (seeds 42/123, 64 eps, 30 s, drop-term off) so
   every number is comparable to the 55.5 / 64.1 / 91.4 anchors. Gates to be
   pre-registered here AFTER the design conversation (env-build → render sanity →
   collection audit → BC gate → DAgger gate → steering gate), each with a result
   guess.
5. **Order:** nominal vision policy first; domain randomization for sim2real
   (lift_vision `visual_randomization.py` template) only after the nominal gate.
6. **Fallbacks, in order:** truncated-resnet features if own-CNN underfits;
   workspace-cam-only if the wrist view proves unstable through grasps (step 0
   tells us); asymmetric RL fine-tune (vision actor / privileged critic) only if
   BC+DAgger+steering all stall.

## Running log

- 2026-08-03 ~03:15: doc created; step 0 script next. GPU verified free; both
  repos synced (reBot_ACT @818391b + uncommitted HANDOFF, reBot_RL @e56e7df
  confirmed on eva_rl).
- 2026-08-03 (post-compaction session): repos synced (reBot_ACT still 2 ahead of
  origin — push pending Big Will; reBot_RL up to date), GPU free. Big Will's
  directive this session: **ensure the policy gets NO privileged information** —
  codified as the §4 contract (23-D proprio = obs41[0:16]⊕[34:41]; objects_canonical
  AND basket_center_xy excluded; programmatic obs-group audit in Gate A). §4
  Gates A–E pre-registered with result guesses BEFORE env code, per convention.
  Next: Step 1, build `pick_place_vision` + Gate A smoke.
- 2026-08-03: **GATE A PASSED — all 5 checks** (`reBot_RL/runs/exp08_gateA/`,
  smoke script `scripts/test_pick_place_vision_env.py`, env pkg
  `tasks/manager_based/pick_place_vision/`, task `Rebot-PickPlace-Vision-Play-v1`).
  - a. shapes/dtypes: policy (N,41) f32; wrist/workspace (N,90,160,3) uint8 with
    real content; student proprio exactly 23; cfg audit found no privileged funcs.
  - b. freshness: NO stale frames at 4/16/64 envs — the step-0 twin-freeze
    artifact did NOT recur in the real two-camera env (pre-registered guess
    wrong, in the good direction). Threshold lesson: a frozen buffer repeats the
    same tensor (diff ~0); the distant D455 view legitimately changes by only
    ~0.44–2.5 uint8-units/frame at 160×90 — the discriminator must be ~0, not
    "small" (first run false-failed at thresh 0.5; fixed to 0.05).
  - c. wrist image-min depth [0.024, 0.051] m every frame = gripper housing →
    render tracks the link.
  - d. FPS (120 steps, sinusoid drive): 67 @ 4 envs / 243 @ 16 / 813 @ 64
    env-steps/s — near-linear, rendering is batch-efficient at this resolution.
  - e. state match vs camera-free env (same seed/actions, spacing-matched 6.0):
    **max abs diff 0.000e+00 over 120 steps — bitwise identical.** Cameras do
    not perturb physics; Gate B can lean on the state-env 91.4% anchor directly.
- 2026-08-03: **Gate B seed 42: 61/64 = 95.3%** (`experiments/exp08_collect.py`,
  shards + meta in `data/exp08_vision/seed42/`, 3.8 GB, flush_count 4). One tick
  ABOVE the pre-registered 88–94% window — benign direction, and explained:
  (a) binomial sd at n=64 is ±3.5 pts (champion's own s42 eval: 89.1%,
  s123: 93.75%); (b) the recorded episodes are each env's episodes 2–5 (first
  discarded for frame sync) — a DIFFERENT spawn draw than the eval's episodes
  1–4, so per-seed rates are not episode-comparable, only distribution-
  comparable. Judgement deferred to the pooled 2-seed number. Per-step contract
  self-audit (proprio == obs41 slices) held every step. Seed 123 collecting.
- 2026-08-03: **GATE B PASSED — pooled 120/128 = 93.75%** (seed 123: 59/64 =
  92.2%, flush_count 10; seed 42 above). Inside the pre-registered 88–94%
  window; champion anchor 91.4% pooled. Dataset: 7.6 GB, 128 episodes (120
  successes → the Gate C training pool). Gate C training launched:
  `act/train_flow_vision.py`, both seeds, success-only, chunk 50 / execute 15,
  100k steps, batch 64, lr 1e-4 → `runs/exp08_bc/v1/`.
- 2026-08-03: **Gate C training DONE** — 100k steps in 2.63 h, loss 0.54 →
  ~0.04 (`runs/exp08_bc/v1/ckpt_final.pt`, 34.7 M params incl. 2× shared-arch
  resnet18 towers). **Gate C eval BLOCKED**: the eval job was externally
  stopped twice ~2 min after launch (harness-level stop, not a crash; no OOM,
  GPU clean; training + collection jobs earlier ran hours undisturbed). Not
  relaunching per the one-retry protocol. Pending command:
  `python act/eval_flow_vision.py --ckpt runs/exp08_bc/v1/ckpt_final.pt
  --episodes 64 --seed 42 --out runs/exp08_bc/v1/eval_seed42.json` (then the
  same with seed 123).
- 2026-08-03: kills resolved-ish: Big Will confirms he did NOT stop the jobs
  (cause still unknown); relaunch on his instruction ran clean end-to-end.
- 2026-08-03: **GATE C: 86/128 = 67.2% pooled** (s42 68.8%, s123 65.6%) —
  inside the pre-registered 55–75% guess, clears the ≥50% rule → **Gate D
  (champion-DAgger)**. First fully non-privileged number of the project, and it
  already beats the state BC base (55.5%) and matches planner-DAgger (64.1%).
  Failure anatomy (from per-episode records): ever_placed≥1 = 115/128 (89.8%),
  final_placed==1 = 29 (stalls after can 1 — the dominant mode, 23%),
  placed 0 = 13 (10%); no episode placed 2 then lost one. Classic compounding
  BC drift → exactly DAgger's target. Results:
  `runs/exp08_bc/v1/eval_seed{42,123}.json`.
- 2026-08-03: **ANOMALY — student success collapses inside the DAgger driver.**
  DAgger r1 collection ran clean (128 eps, 1.1 GB) but the student scored
  42.2%/39.1% (s42/s123) while driving — vs 68.8%/65.6% in its own Gate C eval,
  same checkpoint/seeds/protocol (gap z≈6, systematic). Forensics so far:
  - Eval rerun (same script, fresh process): **67.2% again, per-round
    (6,14,11,12) vs original (6,15,12,11) — the run is near-deterministic.**
    Gate C stands; the DAgger driver is what's off.
  - Per-round breakdown kills two hypotheses: eval round 1 is warmup-depressed
    in both runs (6/16 — renderer warmup, real + repeatable) with rounds 2–4
    at ~77%; the DAgger runs are ~40% in EVERY round including the same
    sim-age windows. Not episode order, not sim age, not sampling luck (two
    seeds agree within each script).
  - Only functional difference found by code diff: the DAgger script loads the
    champion machinery (rl_games import chain incl.) and calls it between
    student steps. Bisection running: `--no-label` (champion never loaded) and
    `--load-only` (loaded, never called) rollouts, seed 42. Outcomes map:
    both ≈77% → the call perturbs; no-label ≈77% + load-only ≈40% → loading
    pollutes global torch state (cudnn/TF32 flags suspected); both ≈40% →
    my loop's student path differs structurally from the eval loop.
  - Gate D training is HELD until resolved — training on data from a perturbed
    student distribution would bake the anomaly in.
- Gate D design (per §4): student DRIVES, champion labels student-visited
  states. Labels are full 50-step chunks computed champion-side (steering z
  from privileged obs56 → x0 = tanh(z) → frozen base flow), collected at the
  student's 15-step window boundaries ONLY — those are the exact states where
  a deployed student commits chunks (its only decision points), and labeling
  is 15× cheaper than every-step. All episodes kept (labels are
  champion-quality regardless of student outcome). Round-1 training: from
  scratch on BC pool + DAgger round (aggregated), same recipe.
- 2026-08-03 ~03:40, **step 0 run 1 (v1) complete**: 3/3 episodes SUCCESS (each
  1500 steps, both cans placed), 180 stills in
  `runs/exp08_vision/wrist_strip_seed123/ep{0,1,2}/`. **Belief 2 CONFIRMED**:
  zero near-uniform frames (frame std 20.6–62.8 across all 180) — the −30° view
  never points into the housing or empty space. **Belief 1 numerically DOUBTFUL**:
  at 0.5 s sampling the active can's off-axis angle only falls to 34–53° at the
  closest pre-grasp sample (half-FOV: 42° H / ~27° V / ~46° corner) — the can
  seems to enter frame only in the final ~1 s of approach, and sits at 45–70°
  (likely out of frame) during transit/carry. Episode anatomy visible in the
  numerics: ~2–3 s initial dwell (camera pose frozen, big frame std swings =
  gripper/scene motion only), then approach (cam→can distance falls 0.28→0.18 m),
  grasp+carry between samples, placed-count flips 0→1→2 mid-episode.
- **Protocol amendment (pre-registered before v2 run)**: the total off-axis angle
  can't separate horizontal from vertical error, which is exactly what decides
  whether a DIFFERENT tilt would help (vertical error → re-tilt; horizontal →
  tilt won't fix it). v2 logs signed camera-frame angles h (about up-axis) and
  v (pitch) + an explicit in-FOV flag every **5 steps** (0.1 s, catches the grasp
  instant); stills unchanged at every 25 steps. Same seed + deterministic stack →
  stills should reproduce v1 bit-identically (free determinism check, verified
  by md5 after the run).
- 2026-08-03 ~04:10, **v2 complete** (same 3/3 success, same placed-flip steps and
  distances as v1 → policy rollout deterministic). Findings:
  - **Renders are NOT bit-deterministic**: v1↔v2 stills differ ~5 mean-abs-pixel
    on 179/180 pairs (RTX denoiser noise; states identical). LESSON: determinism
    audits in vision collection must compare states/actions, not pixels.
  - `placed_mask` verified a real containment test (±5 cm XY of basket center AND
    below the 4.8 cm rim) — placed flips are genuine in-basket events.
  - **h/v decomposition (ep1 grasp of the 2nd can)**: at closest approach the can
    crosses h≈0 (horizontally centered) but sits at **v ≈ −34°…−50° (below the
    ±27° vertical half-FOV)** — by these numbers the can would be just below the
    frame's bottom edge even at the grasp instant, and 25–30° below frame during
    transit. Would argue for MORE downward tilt — BUT see anomaly.
  - **ANOMALY blocking any conclusion**: during the second approach the camera
    closes 0.305→0.180 m on can B while the reading to placed can A stays frozen
    at exactly 0.296 m / h+53.5 / v−59.5 for 100+ steps — physically impossible
    for a static placed can + moving camera (would require A rigidly attached to
    the wrist). Either a pose buffer is stale in a way not yet understood or the
    scene story differs from assumption. **v3 launched** with ground truth: raw
    world coords of camera + cans, cam→TCP h/v every sample (wrist-rigid ⇒ must
    be constant ~1.8°; validates the rotation math), TCP→can distance
    (in-gripper test). No mount verdict until v3 explains the anomaly.
- 2026-08-03 ~04:40, **v3 complete — ANOMALY SOLVED, and it's serious: the wrist
  camera never moves.** Ground truth: `cam pos_w = (+0.210, +0.000, +0.260)`
  CONSTANT through the whole episode while the TCP sweeps 0.138–0.296 m away at
  ±60° — the "wrist" camera sits at a fixed world pose (= gripper_end spawn pose
  ∘ offset) and does not follow the link. Corollaries:
  - The cam→TCP "1.8° off-axis" from the tilt sweep was true only AT the start
    pose (arm parked at spawn) — a fixed camera is indistinguishable from a
    wrist-mounted one in start-pose stills, which is all anyone ever reviewed.
  - Re-reading v2/v3 with this key: the champion's actual behavior is clean
    grasp→lift→transport→release (TCP→can pinned at 0.047 m through the carry,
    lift to z≈0.10, release drops it into the basket). All earlier "the can never
    reaches the TCP" confusion was the static camera's parallax, not behavior.
  - **Code forensics (IsaacLab-3.0 worktree)**: `Camera` reads poses via
    `FabricFrameView.get_world_poses`, which decomposes the cached fabric
    `omni:fabric:worldMatrix` WITHOUT calling
    `fabric_hierarchy.update_world_xforms()` (only `set_world_poses`/
    `set_scales`/init call it). PhysX writes link transforms to fabric, but
    nothing recomposes CHILD prims' world matrices — so a camera prim under an
    articulation link keeps its spawn-time world matrix forever. The renderer
    reads the same fabric matrix → the RENDER is almost certainly static too.
  - **Twin test launched (decisive, numeric)**: second camera spawned under the
    ENV ROOT at exactly the reported frozen pose (0.210, 0, 0.260, same rot);
    per-5-step mean-abs frame diff wrist-vs-twin. Diff at denoiser level (~5)
    for 600 steps ⟹ wrist render static CONFIRMED; large motion-correlated
    diffs ⟹ render tracks the link and only the telemetry was stale.
  - If confirmed: **every wrist-cam artifact to date was a fixed-viewpoint
    camera** (tilt sweeps, Big Will's −30° pick, lift_vision's wrist feed), and
    the vision-env design needs a real mounting mechanism (candidate: env-root
    camera + per-step `set_world_poses` from the link's physics pose — that API
    path DOES call `update_world_xforms`; the old "runtime pose writes render
    blank" failure was on a LINK-PARENTED camera where the write composes against
    the stale parent, which fits the theory).
- 2026-08-03 ~05:10, **twin tests inconclusive by themselves, but two facts
  established**: (1) the wrist cam's reported world quat EXACTLY equals the cfg
  offset rot and pos = gripper_end spawn + offset (parent prim IS world-aligned
  at spawn — old cfg comment vindicated); (2) an env-root twin camera at that
  exact pose (verified by runtime pose copy) **freezes its frame buffer after
  ~35 steps** (std pinned at 52.8 for 560 steps while the arm provably crosses
  its view) — a SECOND camera-freshness artifact, so twin-vs-wrist diffs (~60)
  can't separate "wrist static+live" from "wrist tracking". Twin approach
  abandoned. NEW LESSON: with multiple Camera sensors only the first appears to
  re-render every step in this setup — sweep-style scripts that grab each camera
  ONCE at a static pose never noticed. **Depth discriminator launched** on the
  wrist cam alone: center-pixel depth ~constant 0.10–0.17 m all episode ⟹
  wrist-mounted; center depth tracking the frozen ray's background (~0.3–0.5 m,
  dips on arm pass) ⟹ static render.
- 2026-08-03 ~05:30, **depth test DECIDES IT — the RENDER IS WRIST-MOUNTED; only
  the pose telemetry is stale.** Image-min depth stayed 0.024–0.051 m in all 120
  samples over 600 steps: the gripper/mount sits permanently a few cm in front of
  the lens — impossible for a camera fixed in mid-air after the arm departs
  (center depth 0.063–0.325 m: fingers/objects/table alternating at frame
  center). Reconciliation: the RTX renderer composes child-prim transforms
  itself, while `Camera.data.pos_w/quat_w_*` read the never-recomposed fabric
  `worldMatrix` → frozen at the spawn pose. Consequences:
  1. **The v1 stills ARE genuine wrist-view strips** — Big Will's review plan
     stands; the −30° mount needs no fix; lift_vision's historical wrist feed
     was real. The twin-vs-wrist diff (~60) is fully explained (static twin view
     vs live moving wrist view).
  2. **RETRACTION**: all v1/v2 h/v / off-axis / in-FOV numbers used the stale
     camera pose — meaningless. The "can sits below the vertical FOV even at
     grasp" concern is WITHDRAWN, not confirmed. (The v3 champion-behavior story
     — grasp→carry with TCP→can 0.047 m→release — used robot/object physics
     buffers and REMAINS valid.)
  3. LESSON: NEVER consume `Camera.data` poses on link-mounted cameras in Isaac
     Lab 3.0 — derive camera pose as (gripper body physics pose) ∘ (camera-in-
     body measured at reset), or use render-side signals (depth). Start-pose
     scripts (tilt sweeps) were accidentally correct: stale == true at spawn.
  4. Open artifact for the vision-env smoke test: the env-root twin camera's
     frames froze after ~35 steps — every camera in the vision env needs a
     frames-actually-update check over ~100 steps (std/diff over time), since
     lift_vision's smoke test only checked shapes.
- **v4 strip launched** (3 episodes, stills + CORRECT telemetry): camera pose
  composed from live gripper body pose; built-in checks (step-1 pose must
  reproduce spawn telemetry; cam→TCP rigid ≈0.171 m all episode). First attempt
  crashed (`_GRIPPER_END` is a prim path, body_names wants the leaf) — fixed,
  relaunched. (A 3-episode relaunch was killed externally ~2 min in — cause
  unknown, not Claude; re-run as 1 episode which suffices: same seed/trajectory.)
- 2026-08-03 ~06:05, **v4 complete — STEP 0 VERDICT: the −30° mount is GOOD
  through real grasps (pending Big Will's visual confirmation).** Composition
  validated: cam→TCP pinned at 0.170 m / h+0.0 v−1.4 across ALL 300 samples.
  True per-phase in-FOV rates (ep0, success, both cans placed):
  - can-1 active phase (56 samples): active can IN FOV **79%**; out ONLY during
    the first ~0.7 s swing from the spawn pose (h+50°→in frame by h≈+27°,
    step ~40), then **near-centered (|h|≲8°, |v|≲2°) through descent, grasp
    (TCP→can 0.037 m), the ~2.7 s carry, and release (−10° at placement)**.
  - can-2 active phase (57 samples): active can IN **79%** — same shape.
  - idle after both placed (187 samples): cans still visible 58–68% (basket
    region in view).
  - Depth-at-center falls 0.32→0.10 m through the descent (can/gripper at frame
    center) — ideal terminal-servoing signal.
  - Also corrects the v1-era behavior misread: grasp happens EARLY (can-1 held
    by step ~146), then a slow deliberate carry — placed-flip timing ≠ grasp
    timing.

## Step 0 belief scorecard

1. Active can inside FOV through approach+grasp — **CONFIRMED** (79% of active
   phase; the misses are the initial swing, when the workspace D455 is the
   relevant sensor anyway). Carry occlusion caveat moot: can rides at frame
   center while held.
2. No near-uniform frames — **CONFIRMED** (std 20.6–62.8 across 180 v1 stills).
3. Wrist = servoing detail / D455 = scene context split — **SUPPORTED** (active
   can out of wrist view exactly when far; near-centered when close).

**Decision per pre-registered rule: NO re-pick needed.** Stills for Big Will's
confirmation: `runs/exp08_vision/wrist_strip_seed123/ep{0,1,2}/` (v1, valid
renders) + telemetry `runs/exp08_vision/wrist_strip_seed123_v4/summary.txt`.
Next: the §3 design conversation.

## §4 Gates — PRE-REGISTERED 2026-08-03 (before any env code, per convention)

### The no-privileged-information contract (Big Will's explicit directive)

The champion's obs is 41-D (`act/dataset.py` layout, source of truth
`pick_place_v1_env_cfg.py`):

| slice     | term                | dims | student? |
|-----------|---------------------|------|----------|
| `[0:8]`   | joint_pos_rel       | 8    | YES — encoders exist on hardware |
| `[8:16]`  | joint_vel_rel       | 8    | YES — encoders |
| `[16:32]` | objects_canonical   | 16   | **NO** — ground-truth can poses + placed flags; needs an object tracker on hardware |
| `[32:34]` | basket_center_xy    | 2    | **NO** — randomized per episode; the D455 sees the basket, the student must read the goal from pixels |
| `[34:41]` | last_action         | 7    | YES — the policy's own memory |

**Student input = wrist D405 RGB 160×90 + workspace D455 RGB 160×90 + 23-D
proprio (`obs41[0:16]` ⊕ `obs41[34:41]`). Nothing else. Ever.**

`basket_center_xy` note: we considered treating it as a legitimate goal command
(a static, measurable quantity on the real rig). Rejected — it would demand a
per-placement calibration step at deployment, and the D455 frames contain the
basket anyway. Strict exclusion.

Enforcement (not vibes):
1. Vision env cfg keeps TWO obs groups: `policy` = the full privileged 41-D
   (consumed ONLY by the champion teacher at collection/DAgger time) and
   `student` = images + 23-D proprio. The student network never receives the
   `policy` group.
2. A `STUDENT_PROPRIO_SLICES` constant beside `OBS_DIM` in the ACT-side code;
   collection and training both `assert proprio.shape[-1] == 23`.
3. Gate A includes a programmatic obs-group audit: iterate the student group's
   cfg terms and assert none of {`objects_canonical`, `basket_center_xy`,
   `object_pose_in_robot_root_frame`, `objects_placed_flags`} appears.
4. If depth channels are added later, they must come from the cameras'
   `distance_to_image_plane`, never from sim state.

### Gate A — vision env build + render sanity

Build `pick_place_vision` pkg in reBot_RL (based on `RebotPickPlaceV1EnvCfg_PLAY`
+ `WRIST_CAM_CFG`/`WORKSPACE_CAM_CFG` at 160×90, `update_period = 8/400 = 0.02`),
register `Rebot-PickPlace-Vision-Play-v1` (play/collection variant only — no
train variant until something needs it). Smoke test must check, on a moving
robot (champion or scripted motion):
- a. obs shapes/dtypes for both groups; student group passes the audit in (3).
- b. **Per-camera frame freshness over ≥100 steps** (temporal mean-abs-diff per
  camera stays above threshold — the step-0 twin froze after ~35 steps).
- c. Wrist depth-min ≈2–5 cm every frame (gripper housing → proof the RENDER
  tracks the link even though `Camera.data` poses don't).
- d. FPS at 1/16/64 envs (collection budget planning).
- e. Same seed ⇒ the privileged 41-D trajectory matches the CAMERA-FREE env
  bitwise-or-near under the champion (states, NOT pixels — renders are
  non-deterministic).
- **Result guess:** the multi-camera freshness artifact recurs and costs a
  debugging round; FPS at 64 envs lands ~200–600 env-steps/s total; state
  trajectories match.

### Gate B — collection audit

Champion (frozen N3 base + exp07 steering, deterministic) driven in the vision
env; standard protocol (seeds 42/123, 64 eps each, 30 s, drop-term off);
first episode per env discarded (frame sync). Recorded success must land
≈91.4% — accept 88–94%. Shards store images + 23-D proprio + champion actions;
the 41-D privileged obs is kept in shards but flagged `teacher_only` (audit/
DAgger use, excluded from student loaders by construction).
- **Result guess:** 89–93%; if outside, suspect camera-load physics jitter or a
  frame-sync bug before blaming the champion.

### Gate C — vision flow-BC

Flow-BC head (chunk 50 / execute 15) on own-CNN features from both cameras +
23-D proprio. Eval on the standard protocol.
- **Result guess:** 55–75% pooled (grasp-precision loss vs the state base's
  64.1%; vision fills some gaps, loses others).
- **Decision rule:** ≥50% → proceed to DAgger (Gate D). 35–50% → one
  architecture iteration (first fallback: truncated-resnet18 512-d features),
  then proceed regardless. <35% → stop, diagnose (per-phase failure breakdown),
  consult Big Will.

### Gate D — champion-DAgger

DAgger rounds with champion labels on student-visited states (student sees
images+proprio; teacher labels from the privileged group — that's the whole
point of the two-group env). Target: **≥90% pooled** (champion parity −1.4).
- **Result guess:** 82–90% after 2–3 rounds.
- **Decision rule:** ≥90% → EXP08 nominal goal MET. 85–90% and rising → more
  rounds. Stalled <85% after 3 rounds → Gate E.

### Gate E — fallback finisher: x0-steering on the vision base

EXP07 recipe on the frozen vision base. **Deployment subtlety pre-registered:**
the EXP07 steering head consumed privileged obs56 — a vision-base steering head
must instead consume student-visible features (CNN features + proprio + the
controller-free steer features), or it re-smuggles privileged state into the
deployed stack. Guess: +4–10 pts if ever needed.

### After the gates

Step 6 (domain randomization, sim2real) starts only after Gate D/E passes —
`lift_vision/visual_randomization.py` is the template. Not part of EXP08's
nominal goal.
