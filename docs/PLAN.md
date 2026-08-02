# reBot_ACT — Robust Pick-and-Place Pipeline Design

**Architecture:** SOTA grasp synthesis + motion planning → expert demonstrations & recoveries → **flow-matching chunk policy** (π0-style action head; ACT retired 2026-08-01, see §2.3) → planner-DAgger refinement → frozen policy + residual RL → final robust policy.

**Task (upgraded from reBot_RL):** two cans → basket; objects spawn **further away** (toward the reach-envelope edge), **basket position randomized**, object geometry/dynamics randomized. Final gate: >90% success on a fixed robustness benchmark (nominal AND perturbed composite), failed grasps < 0.5/ep.

**Key design principle (Big Will, 2026-07-31):** the expert is itself a high-quality manipulation *system*, not a trajectory generator. It searches over **which grasp × how to execute × whether that grasp supports the rest of the task**, and can replan/regrasp from difficult intermediate states. No simple fixed-grasp + IK/RRT expert.

**Clean grasps are a first-class gated objective (Big Will, 2026-08-01):** the policy must be *taught* clean grasps — the RL era's emergent shove/rake grasps and the old `grasp_table.pt` static lookup are exactly what we're replacing, and neither survives into this pipeline. "Clean" is defined by measurable criteria (§1.3 step 4) and gated at Gate 1 (expert), Gate 2 (demos/ACT), and Gate 6 (final). Only clean-grasp episodes enter the nominal BC pool.

Research grounding: five deep-dive reports (2026-07-31): codebase survey, motion-planner landscape, ACT/DAgger/residual-RL ecosystem, GraspGen/GraspGen-X/Grasp-MPC, cuRoboV2/GPU-TAMP (v2 source read directly at a pinned clone). Citations inline.

---

## Stack decision (summary)

| Component | Choice | Fallback |
|---|---|---|
| Motion planning | **cuRoboV2** (Apache-2.0, pin recent main commit post-#697), `MotionPlanner.plan_grasp` + goalset + `AttachmentManager` + arbitrary-state replan | cuRobo v1 @ Isaac-Lab-pinned commit; then mplib (CPU); wrap behind a thin planner interface so swaps are cheap |
| Grasp synthesis | **Pluggable candidate-generator interface.** Source 1 (day one): analytic cylinder sampler (closed-form side-grasp ring). Source 2: **GraspGen-X zero-shot** (config wizard on our gripper URDF, own uv venv, offline server) — switched on when object-geometry diversity lands | Sim-execution verification is the final arbiter for any source |
| Grasp→task ranking | IK batch pre-filter (grasp ∩ induced place feasibility) → `plan_grasp` goalset (full approach/grasp/lift plan decides winner) | — |
| Reactive execution | 50 Hz joint-target tracking of B-spline interpolated plans; replan-on-event | cuRoboV2 `ModelPredictiveControl` (MPPI) if open-loop tracking + replan proves insufficient; **Grasp-MPC: no public code — watch only** |
| TAMP | **Not adopted** — task combinatorics = object order (2 skeletons) + grasp choice (goalsets). cuTAMP is pinned to cuRobo v0.7.8, unclear license, 12 GB-hostile | Retry other object order on place failure — that's the whole task planner |
| BC | **Flow-matching chunk policy** (rectified-flow velocity head on our transformer backbone, state-only, chunk 50/execute 15 — PIVOT 2026-08-01, Big Will: future-proof for harder multimodal tasks; π0-lineage method) | vendored LeRobot ACT (already in `act/`, kept as emergency baseline only — NOT trained by default) |
| DAgger | Custom ~150-LOC loop (expert is in-process; libraries only add impedance) | — |
| Residual RL | Frozen-ACT-in-env wrapper + **rl_games PPO** (ResiP precedent) | skrl SAC + RLPD tricks (Stage 5 replay seed) |
| Dataset | Isaac Lab / robomimic-compatible HDF5 via `RecorderManager` + custom attrs (`perturb_steps`, `episode_kind`, `success`, grasp/plan metadata) | — |

**Environments:** cuRoboV2 installs into `env_isaaclab6` (needs only `torch>=2.5`; we have 2.11.0+cu128, py3.12; pip-provided CUDA headers, no system-toolkit dance). GraspGen-X pins `torch<2.7` → **its own uv venv**, run as an offline grasp server (files/ZMQ). Clones live in `reBot_ACT/third_party/`. If cuRobo install disturbs env_isaaclab6 in any way, fall back to a dedicated planner env + separate process exchanging poses/trajectories over a pipe (clean, since only privileged state crosses the boundary).

---

## 0. Ground rules & constraints

- **Hardware:** RTX 4080 Laptop, 12 GB VRAM, shared sim+planner+training. One GPU job at a time; demo gen headless. Budget: sim 2.5–4 GB + cuRoboV2 (est. 1.5–3 GB, unpublished — measure in spike) + GraspGen-X ~1–2 GB (run serially or in the separate venv process).
- **Code layout:** pipeline code HERE (`reBot_ACT/`); the Isaac Lab task package stays in `reBot_RL/source/reBot_RL/` (shared sim asset); Stage 0 edits happen there. External repos → `third_party/` after vetting.
- **Robot facts that bound everything** (codebase survey):
  - 6-DOF arm + parallel-jaw gripper (stroke 0–0.05 m/finger, fingers along gripper −X, TCP offset 0.075 m); action space 7-D `[joint1..6 pos targets, binary gripper]` @ 50 Hz (decimation 8 over 400 Hz physics). Native action space is kept end-to-end.
  - **Wrist cannot point fingers down near the table** → side-on/radial grasps only. Grasp candidate generation must bake this in; learned generators' top-down proposals get IK-filtered anyway.
  - **Graspable envelope ceiling ≈ r 0.32–0.35 m** (empirical); basket r 0.29 already marginal for carries. Ranges come from the Stage 0 reachability audit, not aspiration.
  - **DLS differential IK diverges from table-level configs** (documented in the old expert). cuRoboV2's collision-aware IK+TrajOpt and `plan_cspace` retreats are precisely the cure; never trust raw DLS near the table.
  - Obs today: 39-D privileged; Stage 0 makes it 41-D (basket pose added). `objects_canonical` target-first ordering is preserved for BC.
  - URDF: the USD was converted from an upstream URDF (converter 0.3.0) — locate or regenerate it; cuRoboV2 `RobotBuilder` and the GraspGen-X wizard both consume URDF + meshes.
- **Carried-over lessons** (memory `rl-teacher-lessons`): perturbation-in-training → robustness; commitment penalties kill chatter; honest fixed eval; reward-weight × dt gotcha; one informed change per long run.
- **Known cuRoboV2 sharp edges:** issue #692 (open): `plan_grasp` interpolated trajectories zero-padded — trim via `*_interpolated_last_tstep`; batched `multi_env` mode loses PRM graph seeding and per-problem retries (run failed problems as singles); pin post-#697 (batch-planning fixes merged 2026-07-10).

---

## Stage 0 — Env upgrade (in reBot_RL): farther objects, randomized basket, diversity axes

Current env hard-codes the basket: 5 static cuboids baked at config time from `BASKET_CENTER = (0.22, −0.12)` in `mdp/common.py`, read by rewards/events/eval/expert.

**0.1 Randomizable basket** — replace with a single **kinematic `RigidObjectCfg`** (5-box compound collision authored once as a tiny USD). Reset event samples basket center: r ∈ [0.20, 0.27], azimuth ∈ [−50°, +50°], ≥0.14 m from every object spawn; writes root pose + per-env buffer `basket_center_w`. Replace every `BASKET_CENTER` read (grep to zero call sites: `common.py`, `rewards.py`, `events.py`, eval scripts, expert). Add relative basket xy to policy obs → 41-D.

**0.2 Wider object spawns** — uniform sampler event: r ∈ [0.20, 0.32] (hard clamp 0.33), azimuth ∈ [−45°, +45°], pairwise separation ≥ 0.06 m; lying prob → 0.25. Keep friction randomization + `nudge_objects` (off for nominal collection, on for recovery). Remove prestage curriculum events and RL shaping terms (sparse success + drop only; rewards return in Stage 6).

**0.3 Diversity axes (new, per spec)** —
- *Object geometry:* per-episode can scale ∈ [0.8, 1.25]× current (respect max gripper opening ~0.09 m and mass rescale); later, 1–2 alternate small YCB bodies behind the same interface. This is also what earns GraspGen-X its slot.
- *Dynamics:* mass ×[0.5, 2], friction sweep (event exists), light drive-gain jitter.
- *Robot start config:* small randomization around `_START_POSE` (planner-expert is start-agnostic; RL's fixed start was a crutch).
- *Clutter (phase 2 of Stage 0, after Gate 1 first passes):* 1–2 distractor primitives on the table — planner treats them as obstacles; benchmark suite S9 covers them. Don't block the pipeline on clutter.
- PhysX buffer re-check at high env counts (kinematic basket + distractors add bodies; overflow = silently dropped contacts).

**0.4 Reachability audit** — rebuild empirical graspability map for the widened region using the *planner itself* (batch IK over grasp goalsets on a grid over r × azimuth × {upright, lying} × scale). Clamp all Stage 0 ranges to map-minus-margin. **Verify:** ≥95% of the sampled spawn region admits ≥1 feasible grasp+place; map plot to Big Will.

**Registration:** `Rebot-PickPlace-v1` / `-Play-v1`; v0 stays for comparison.
**Gate 0:** extended smoke test (per-env basket poses differ; placed-predicate follows basket; spawns settle clean; obs 41) + video to Big Will.

---

## Stage 1 — Grasp-synthesis + motion-planning expert (cuRoboV2)

### 1.1 Robot onboarding (first work item, everything depends on it)
- Locate/regenerate RS-rebot **URDF + meshes**; `RobotBuilder(urdf, assets).fit_collision_spheres()` (auto MorphIt fitting) → `compute_collision_matrix()` → `save(yml)`; `RobotDebugger` retract-collision check; sphere-fit visualization (Rerun/Viser) rendered **for Big Will's review**. Tool frame defined between fingertips (TCP −X convention). Effort ~1–2 days.
- Install: clone `NVlabs/curobo` pinned post-#697; `uv pip install .[cu12]` into env_isaaclab6; run their examples on Franka config first; **record VRAM** (driver must be ≥580.65.06 — check).

### 1.2 Grasp candidate synthesis (pluggable interface)
`GraspSource.propose(object_mesh, object_pose, gripper_spec) → [(grasp_pose, score), …]`
- **Source 1 — analytic (day one):** cans are cylinders → closed-form antipodal family: ring of radial side-grasps (tool z toward the axis) × height offsets × 180° symmetry flips × approach azimuths, K = 16–32. Respects the side-on-only wrist by construction. Place candidates likewise: goalset of M poses over the basket interior × yaw.
- **Source 2 — GraspGen-X zero-shot** (arXiv 2606.00998, Apache-2.0 code, NVIDIA Open Model License checkpoints): run its config wizard on our gripper URDF (swept-volume conditioning; no retraining), serve proposals from its own uv venv. Known risks: ~0.50 zero-shot AUC on unseen parallel-jaw grippers, degradation on small objects → discriminator score is advisory; **sim-execution clean-grasp verification is the arbiter**. Onboarded **before demo generation** (ledger item 8): head-to-head vs the analytic source on clean-grasp rate + full-task success over 100 episodes; the measurably cleaner source (or a blend) generates the demos. Also copy the grasp→cuRobo wiring from GraspGen-X's `end2end` extra (the closest official template). Becomes mandatory once 0.3's geometry diversity lands (analytic sampler is cylinder-only).
- Not adopted: base GraspGen (no checkpoint for our gripper; retraining ≈3K GPU-hours; torch-2.1 pin), Grasp-MPC (no code release).

### 1.3 Full-problem ranking & episode planning (the core machinery)
Per episode / per replan, with privileged poses:
1. Build `SceneCfg` from primitives: table cuboid, 5 basket cuboids, other-can cylinder, distractors.
2. **Batch IK pre-filter** (`InverseKinematics.solve_pose` on goalsets): drop unreachable grasp candidates AND grasp candidates whose induced object-in-hand pose admits no IK-feasible place — grasp choice is conditioned on downstream placement from the start.
3. **`plan_grasp(goalset)`**: goalset plan → approach pose → constrained linear approach (`ToolPoseCriteria.linear_motion`) → constrained lift; winner by full-plan feasibility (`goalset_index`), i.e. reachability + collisions + joint limits + smoothness/cost + (available if wanted) torque feasibility via B-spline trajopt. Per-stage success flags and trajectories come back in `GraspPlanResult`.
4. Close gripper + **clean-grasp verification** in sim. A grasp counts as **clean** only if ALL hold: (a) object displacement during finger closing < 5 mm and yaw disturbance < 10° (no pushing/spinning the can into the fingers); (b) both fingers make contact within ~2 control steps of each other (symmetric antipodal contact); (c) in-hand pose drift over the lift < 5 mm / 5° (no slip or settle); (d) no gripper–table/basket contact during approach/close. Anything less → release + regrasp with next-ranked candidate. These four criteria define `clean_grasp` everywhere downstream (gates, BC data filter, benchmark metric); thresholds tuned once during the spike, then frozen.
5. `AttachmentManager` attaches the can (per-env link-local spheres) → **`plan_pose(place_goalset)`** transport → constrained descend → release → retreat → next object (order: try can-A-first; on place failure, other order — the entire task planner).
6. **Robustness margin:** among feasible candidates prefer larger clearance margins / mid-range joint configs (cuRobo cost weights + our tie-break scoring).
- Throughput: ~5–6 plans/episode × 50–200 ms ≈ ≲1.5 s planning/episode → thousands of episodes overnight at batch=1. Scale-up option: `BatchMotionPlanner(multi_env=True)`, B=8–16 per-env worlds, singles as retry fallback.

### 1.4 Execution bridge (50 Hz)
Trim interpolated trajectories (`*_interpolated_last_tstep`, bug #692), resample to 20 ms, feed as joint-position targets (B-spline output is jerk-bounded → tracks cleanly at our stiff gains). Gripper stays binary/scripted with verification. Keep per-stage failure counters (lesson: diagnose by numbers).

### 1.5 Recovery / replan from arbitrary states (native)
On any perturbation or verification failure: rebuild `SceneCfg` from current privileged poses, replan from current `JointState` — `plan_grasp` for regrasp (dropped/slipped can = just another on-table object, possibly newly lying), `plan_pose` for transport correction, `plan_cspace` for pure joint-space retreats out of weird configs (the DLS-divergence zone). The expert is **memoryless at the task level**: stage inferred from state (holding? over basket? objects unplaced?), so it is queryable from ANY state — the property Stages 3–5 depend on.

### 1.6 Fallback ladder
Thin `PlannerInterface` wrapper around cuRoboV2 so we can swap: (a) cuRobo v1 @ Isaac-Lab-pinned commit (restrictive license, clunkier batch, but battle-tested with Isaac Lab in-process), (b) mplib CPU transit legs + analytic grasps, (c) last resort: the old scripted expert (48.8%) exists in `reBot_RL/scripts/scripted_expert/` — floor, not target.

**Gate 1 (decides everything downstream):**
- **Spike gate (run FIRST, before any other Stage 1 build-out):** K=16 side-grasp goalset `plan_grasp` from retract on the real scene — goalset feasibility rate near the table + VRAM + wall time. If feasibility is poor on our constrained wrist, revisit candidate generation before building the generator.
- **Full gate:** expert on `-v1`: ≥95% nominal success, ≥90% with nudges, failed_grasps < 0.3, **clean-grasp rate ≥ 95% (per §1.3 step 4 criteria)**, teleport-perturb-at-random-phase recovery ≥90%. Expert video to Big Will before mass generation.

---

## Stage 2 — Datasets & nominal BC policy (flow matching)

**2.1 Format:** Isaac Lab HDF5 (`RecorderManager`, `ActionStateRecorderManagerCfg` + custom terms), robomimic-compatible: `obs/*` (full 41-D + named components), `actions` (7-D), per-episode attrs: `success`, `perturb_steps`, `episode_kind` (`nominal|recovery_scripted|recovery_dagger`), orientation class, **grasp/plan metadata** (chosen candidate id, goalset rank, plan cost — enables later analysis of strategy diversity). 50 Hz native (chunking absorbs it; downsampling non-standard). Failures → separate `failures.hdf5` (Stage 5).

**2.2 Nominal generation with preserved diversity (per spec §2):** ~**500 nominal episodes**, stratified quotas over r-band × azimuth × orientation × basket sector × object scale. **Do NOT collapse to canonical trajectories:**
- sample the executed grasp from the **top-p feasible candidates** (not argmax) with probability ∝ margin score;
- keep cuRobo's natural plan diversity (different seeds/goal winners produce different homotopies);
- randomized robot start configs;
- both object orders where both are feasible.
Multiple valid strategies in-data is what lets a generative action head (flow matching) model mode structure instead of averaging across it.
**Clean-grasp filter:** a nominal demo enters the BC pool only if every grasp in it was clean on the FIRST attempt (§1.3 step 4). Episodes where the expert needed a regrasp are not discarded — they move to the recovery pool (Stage 3), where regrasping is exactly the behavior we want supervised (with the failed attempt's own steps loss-masked per the block below: the recovery pool supervises the fix, never the miss). The nominal pool teaches "grasp cleanly"; the recovery pool teaches "if it wasn't clean, fix it."

**Per-step loss-mask labels (2026-08-01, Big Will directive):**
- The expert runner (`expert/run_expert_v1.py`) now emits per-episode `segments` (`[{t, phase, seg}]` step-indexed phase transitions; phases: settle/approach/descend/close/lift/reopen/transport/release/retreat) and `outcomes` (segment id → `{outcome, ...}`).
- Demo gen converts these to a per-timestep `train_mask`: mask=0 over segments whose outcome is `missed` (failed grasp attempt — approach/descend/close/lift up to failure DETECTION) or `lost` (transport that dropped the can). Rationale: the policy must learn recovery behavior, never learn to predict/imitate failures (no stochastic "miss" modes in the BC target).
- The mask boundary is failure DETECTION: post-detection segments (reopen after a miss, re-approach, refetch after a drop) are recovery skill and stay trainable. The reopen segment has outcome `recovery`.
- Mechanism: the mask rides the same per-timestep channel as LeRobot ACT's `action_is_pad` censoring — chunks straddling a masked→trainable boundary get only their masked timesteps zeroed.
- Placement provenance: transport outcomes record `via` ∈ {pose, pose-high, carry, carry-direct, +hop}. carry-direct is a scripted joint interp (not a planned collision-free trajectory) — exclude from the nominal pool by default; ablate its inclusion in the recovery pool.
- Per-attempt grasp records include `close_disp` and a clean flag, so the "first-attempt-clean" nominal filter is exact at attempt granularity (not episode-coarse).

**2.3 Flow-matching chunk policy (PIVOT 2026-08-01, Big Will: full pivot, no ACT training — future-proof for harder multimodal tasks; π0-lineage method).** Research basis (survey 2026-08-01, subagent report in JOURNAL): ActionFlow (arXiv:2409.04576) evaluated and REJECTED — SE(3)-pose-space equivariance machinery inapplicable to our 7-D joint-delta actions, and no public code. Instead: **rectified-flow head grafted onto the vendored ACT transformer** (the standard conversion used by π0/LeRobot-pi0/X-IL/Much-Ado):
- **Drop the CVAE** (latent z, CVAE encoder, KL term) — the flow noise sample x0 replaces z as the stochasticity source.
- Obs encoder unchanged (state 16 + env_state 25 → tokens). Decoder queries = **projected noisy chunk x_τ** (B,50,7) + positional embeddings, bidirectional self-attention over action tokens, cross-attend to obs tokens. Time conditioning: sinusoidal τ embed → MLP → AdaLN (fallback: add to tokens).
- **Loss:** x_τ = (1−τ)·x0 + τ·x1, target v = x1 − x0, elementwise MSE with `reduction="none"` — the per-timestep loss mask (expert-failure censor + `action_is_pad`) multiplies in exactly as it did for L1 (LeRobot pi0 does literally this). τ ~ uniform first; π0's high-noise-emphasis Beta as a later tweak. Actions normalized per-dim mean/std (FM is noise-scale-sensitive).
- **Inference: 10 Euler steps** (π0 default; quality flat down to ~5 per Much-Ado — drop to 5 if Stage 6 rollout cost bites). Chunk 50 / execute 15 receding horizon, temporal ensembling OFF, batched eval wrapper — all unchanged from the ACT design.
- Reference code: TorchCFM (atong01/conditional-flow-matching) for the ~100 lines of FM math; much-ado-about-noising (MIT) as cross-check + MIP baseline option. Vendor files only, never pip install.
- **Expectation setting (honest, from controlled studies — Much Ado About Noising, ICLR 2026):** on state-only single-task few-hundred-demo regimes FM ≈ L1 parity; the payoff is future multimodality + principled noise source for DAgger/residual stages. Gate 2 bar unchanged.
- Vendored ACT files stay in `act/` as emergency baseline only (NOT trained by default). Baseline sanity check: stock robomimic BC-RNN once.

**Gate 2:** nominal suite ≥ **85%** AND clean-grasp rate ≥ **90%** before any recovery data; failure anatomy recorded.

---

## Stage 3 — Recovery data: scripted perturbations + planner-DAgger

**3.1 Expert-driven perturbation rollouts:** injection schedule at random phase ∈ {pre-grasp, lift, transport, above-basket}: object nudge (widened `nudge_objects`), **forced slip** (force gripper open 2–4 steps or impulse on held can), small pose teleports. Expert (memoryless + native replan) continues correctly → post-event trajectory IS the recovery demo, including **regrasp with a different grasp** and replanning around changed geometry. Record `perturb_steps`. Keep ultimately-successful episodes. ~**400 episodes** over perturbation-type × phase grid.

**3.2 Planner-DAgger from ACT's own failures:** roll current ACT (perturbations at benchmark rates); expert queried per-step for labels (in-process, cheap). Preferred mode: **gated (HG-DAgger-style)** — deviation/failure gate trips (TCP deviation δ, failed-grasp detector, drop, stall) → expert takes over → record its recovery-to-success. Full per-step relabel as ablation. Iterate collect → retrain (nominal + recovery + DAgger pools) → benchmark; expect 2–3 rounds. Port the loop shape from `reBot_RL/scripts/distillation/collect_episodes.py` (already implements roll-student/label-teacher).

**Training-mix rule:** BC trains ONLY on expert-generated actions. Student actions are never labels; failed rollouts never enter the BC pool.

**Gate 3:** perturbed-suite success within 5 pts of expert's perturbed number; recovery-success (≥1 perturbation fired, still completed) ≥ 85%.

---

## Stage 4 — Stochastic-event handling (loss censoring + chunk hygiene)

**4.1 Loss censoring:** for a training chunk `[t, t+chunk)` straddling `perturb_step` e (t < e): set `action_is_pad[e−t:] = True` — no penalty for failing to predict the unforeseeable. Chunks starting ≥ e supervise recovery normally. ~5 lines in the Dataset; the vendored ACT already multiplies L1 by `~action_is_pad`; KL untouched. **No direct prior art found** (verified search) — novel-ish engineering; DAgger relabeling is the load-bearing fix, censoring only removes a wrong gradient. **Ablation in round 1** (masked vs not, identical data); keep only if ≥ neutral. Composition with the expert-failure loss mask (§2.2): the two censors OR together on the same `action_is_pad` channel; the expert-failure mask is UNCONDITIONAL (never ablated away — we never supervise a miss), only the perturbation censor is subject to this ablation.

**4.2 Deployment chunk hygiene:** receding horizon bounds staleness to 0.3 s; add explicit **flush trigger** on detected discontinuity (held-object z drop, gripper force loss, object teleport > threshold — cheap in sim): clear queue, re-predict immediately. No off-the-shelf implementation resets mid-episode (verified) — small custom addition. Stage 5's failure detector later replaces privileged detection (visual-policy forward-compat).

---

## Stage 5 — Failure data (kept, never behavior-cloned)

All failed rollouts (expert and student) accumulate in `failures.hdf5` (full obs/action/outcome). Uses:
1. **Residual-RL replay seed** — RLPD-style 50/50 offline sampling (critics learn what bad looks like).
2. **Failure detector** — classifier on obs history → P(failure imminent): deployment flush/regrasp trigger + intervention metric.
3. **Curriculum** — cluster failure states (anatomy harness exists); over-sample matching ICs/phases next collection round. This replaces reward-shaping whack-a-mole as the failure-mode closer.
4. **Benchmark seeds** — hardest failure ICs frozen into suite S8.

---

## Stage 6 — Residual RL on frozen flow policy

References (verified): **ResFiT** arXiv 2509.19301 (frozen chunked base + single-step off-policy residual, code: amazon-far/residual-offpolicy-rl); **ResiP** arXiv 2407.16677 (PPO per-step residual on frozen chunked base, GPU-parallel sim, 5–50% → >95%); **RLPD** arXiv 2302.02948; **RFS** arXiv 2602.01789 (residual RL on a frozen FLOW-MATCHING base — direct precedent for the 2026-08-01 FM pivot; RL outputs initial noise x0 + residual, PPO in sim).

- **Determinism of the frozen FM base:** a flow policy is a deterministic map from (obs, x0) — sample x0 once per chunk from a seeded generator (or fix x0) and the Euler rollout is exactly reproducible (ReinFlow arXiv:2505.22094 observation). Default: ResiP-style deterministic base extraction (fixed/seeded x0 per chunk) so base-policy variance doesn't pollute PPO advantages; RFS-style learned-x0 steering is the upgrade path if the plain residual stalls.
- **Architecture:** frozen flow policy (queue + flush logic) embedded in an env wrapper/ActionTerm. `a = a_base + α·tanh(a_RL)` on the **6 arm joints only** (α ≈ 5–10% of range, zero-init output layer → training starts exactly at base). **Gripper stays with base** (residual on binary action ill-posed). RL obs = env obs ⊕ a_base ⊕ chunk phase (both papers feed base action — required, else nonstationary). Residual stays per-timestep closed-loop even though the base is chunked (ResiP's main lesson). Budget note: 10 Euler steps = 10 decoder passes per chunk refill inside RL rollouts; drop to 5 if throughput bites. Wrapper exposes a plain env → any trainer unmodified.
- **Algorithm:** **rl_games PPO first** (ResiP precedent; tuned infra; sim steps cheap). Fallback: skrl SAC + RLPD tricks (LayerNorm, high UTD, 50/50 replay from Stage 5). SB3 off-policy: known poor fit for massive GPU envs.
- **Reward:** sparse success + drop penalty + gripper-toggle penalty + mild time penalty (**remember ×dt weight scaling**). No dense shaping — base already solves the task.

**Gate 6 (final):** benchmark ≥ **90% nominal AND perturbed composite**, failed_grasps < 0.5, **clean-grasp rate ≥ 90%** (the residual must not re-learn shove-grasps — watch this metric per checkpoint), drops < 2%, no completion-time regression vs base policy.

---

## Stage 7 — Fixed robustness benchmark (built EARLY — right after Gate 1 spike)

One runner (`reBot_ACT/benchmark/run_benchmark.py`, evolved from `evaluate_pick_place.py` + Isaac Lab `robomimic/robust_eval.py` structure), fixed seeds, 256+ episodes/suite, JSON per checkpoint. Every artifact (expert, each ACT round, residual) gets the same table.

| Suite | Contents |
|---|---|
| S0 nominal | full randomization, no perturbations |
| S1 pose/orientation | extreme-band spawns, 50% lying, adversarial yaw |
| S2 imperfect grasp | init from offset/marginal grasp states |
| S3 slips/drops | forced gripper-open / held-object impulse mid-transport |
| S4 external pushes | elevated nudge magnitude/frequency |
| S5 dynamics | mass ×[0.5,2], friction sweep, gain jitter |
| S6 target movement | basket teleports a few cm mid-episode |
| S7 combined | S1+S3+S4+S5 stacked |
| S8 known-hard | frozen failure-derived ICs (Stage 5) |
| S9 clutter/generalization | distractor objects + unseen object scales/geometry |

**Metrics/suite:** success, recovery-success (success | ≥1 perturbation), placed_a/b, drops, collision events (wall/table impulse > threshold), mean completion time, grasp_attempts / failed_grasps / clean-grasp rate (regrasp frequency), intervention frequency (gated-DAgger).

---

## Execution order & verification ledger

| # | Work item | Verify |
|---|---|---|
| 1 | cuRoboV2 install + Franka example + VRAM measure | example runs; VRAM number recorded |
| 2 | RS-rebot URDF → RobotBuilder spheres → debugger | sphere viz to Big Will; retract collision-free |
| 3 | **SPIKE: K=16 side-grasp `plan_grasp` on real scene** | feasibility rate + wall time + VRAM → go/no-go on stack |
| 4 | Stage 0 env (basket rand, wide spawns, diversity axes, obs 41) | Gate 0 smoke + video |
| 5 | Reachability audit (planner-driven map) | ≥95% coverage; map plot to Big Will |
| 6 | Stage 1 full expert (ranking, execution bridge, recovery) | Gate 1 numbers; expert video |
| 7 | Stage 7 benchmark harness (before ACT) | runs on expert; JSON sane |
| 8 | **GraspGen-X onboarding (wizard + venv + server) + head-to-head vs analytic source** | clean-grasp rate + full-task success per source on 100 episodes; winner (or blend) generates the demos |
| 9 | HDF5 recording + 500 stratified nominal demos (clean-grasp filtered) | replay check; robomimic BC-RNN baseline trains |
| 10 | Flow-matching policy (FM head on ACT backbone) + dataset + nominal training | Gate 2 ≥85% + clean-grasp ≥90% |
| 11 | Stage 3 recovery + DAgger rounds (2–3×) | Gate 3; per-round benchmark deltas |
| 12 | Stage 4 censoring ablation | masked ≥ unmasked, else drop |
| 13 | Stage 5 failure detector + curriculum round | detector AUC; S8 delta |
| 14 | Stage 6 residual RL | Gate 6 final |
| 15 | (later) visual policy distillation | separate phase; per vision-env-plan memory |

## Top risks
1. **cuRoboV2 youth** (3.5 months old; #692 open; API moving; no Isaac Lab v2 reference integration). Mitigate: pin post-#697, thin `PlannerInterface`, v1/mplib fallbacks, spike before build-out.
2. **Goalset feasibility on the constrained wrist** near the table (same geometry that breaks DLS; batch mode loses graph seeding). Mitigate: generous goalsets, good retract seed, spike is item #3 for exactly this reason.
3. **VRAM/coexistence unknowns** (v2 runtime memory unpublished; CUDA-graph + allocator sharing 12 GB with Isaac Sim, in-process precedented only for v1). Mitigate: measure in spike; separate-process fallback is cheap.
4. **GraspGen-X zero-shot quality** on 2.4 cm objects with an unseen gripper (~0.50 AUC). Mitigate: advisory scores only; IK/collision filter + sim-execution verification arbitrate; analytic source carries the pipeline regardless.
5. **Loss censoring unproven** (no prior art). Mitigate: 5 lines behind an ablation flag; DAgger is the real fix.
6. **Expert ceiling** — everything downstream inherits expert quality. Mitigate: per-stage failure counters, regrasp-by-design, ranking includes robustness margin, reachability audit before freezing ranges.
