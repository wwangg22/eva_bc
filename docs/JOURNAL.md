# reBot_ACT engineering journal

## 2026-08-01 overnight — cuRoboV2 onboarding + spike (ledger items 1–3)

### Done & verified
- **cuRoboV2 0.8.0.post1 installed** into env_isaaclab6, pinned `8e734f3` (post-#697,
  includes #701). Deps light; `packaging` downgraded back to 26.0 for isaacsim-core
  (vcs-versioning's complaint is build-time only). Franka examples pass (pose + grasp).
  **Peak VRAM 513 MiB** — coexistence with Isaac Sim on 12 GB is comfortable.
- **RS-rebot onboarded** (`expert/onboard_robot.py` → `expert/rs_rebot.yml`):
  URDF found at `Desktop/ReBOT/reBot-Isaacsim/urdf/00-arm-rs_asm-v3/` (copied to
  `assets/`), limits match the USD validation doc. MorphIt fit 126→134 spheres.
  Patched: lock fingers @0.045, attached_object link on gripper_end (8 spheres),
  grasp_contact_link_names, retract = _START_POSE. Sphere viz PNG:
  `expert/rs_rebot_spheres.png` (unviewed — for Big Will).
- **Frames validated**: cuRobo FK == Isaac Lab FK to 0.1 mm / 1e-3 quat at an
  identical joint vector. cuRobo quats are **WXYZ**; Isaac Lab 3.0 **XYZW**.
- **Stage 0 complete (subagent), Gate 0 PASS**: `Rebot-PickPlace-v1` +
  `-Play-v1` registered. Movable kinematic basket (`data/basket/basket.usda`),
  annulus spawns r∈[0.20,0.32] az±45°, 41-D obs (basket xy after objects_canonical),
  sparse rewards only, mass/gain/start-config/scale diversity, nudges off by default.
  Smoke: `scripts/test_pick_place_env_v1.py`. Stills for Big Will:
  `logs/camera_previews/pick_place_v1_check/`. Found v0 bug: v0 "lying" spawn quat
  actually stood cans upright (left as-is in v0).
  Follow-up: `probe_pick_place_policy.py:91,178` + `generate_pick_place.py:235`
  still read fixed `BASKET_CENTER`; switch to `mdp.basket_centers_local(env)` for v1.

### Hard-won debugging ledger (spike)
1. **0% instant** → my synthetic grasp ring pointed fingers OUTWARD (away from robot).
   FK over `grasp_table.pt` (12,953 proven grasps): finger dir = −x̂ exactly;
   **fwd azimuth = pocket azimuth −159°±10° (fingers point BACK toward robot — the
   arm reaches past the can); elevation +26°±5° (fingers tilt up); TCP−pocket = 27 mm
   along finger axis**. Approach must be vertical descent (old expert's DOWN phase).
2. **Still 0%** → `base_link` MorphIt spheres protrude to z=−0.027 (base sits ON the
   table) → ANY table obstacle = permanent collision → all IK dead. Fixed by clamping
   base spheres above z=−0.001 in onboard patch. After clamp: proven poses 24/24 IK
   (fingers disabled) with table at surface.
3. **Still 0% via plan_grasp** → status "No grasp in goal set was reachable" fires
   AFTER IK succeeds — it's trajopt failing. Root cause: **start state**. I had used a
   settled pose measured from a broken probe (gripper 3 mm off the table → gripper
   spheres inside table → start invalid → every trajectory invalid). The env's true
   start is `_START_POSE = (0, −1.35, −0.3, −0.85, 0, 0)` (lift/rebot_lift_env_cfg.py)
   — NOT rebot_arm.py's USD init (joint2=−π/2 puts gripper 13 cm INSIDE the table;
   PhysX shoves it out at reset — explains the FK "z=−0.135 mystery").
4. Goalset source of record: **table_candidates()** — K=16 nearest proven grasps by
   pocket xy (translated to target), through the pluggable GraspSource seam.
   Synthetic ring family kept for reference; its (daz, elev) feasibility is
   region-dependent (2–6/18) — parameters would need per-region fitting.
5. CUDA-graph note: changing goalset size / solve args after graph capture raises
   "CUDA graph reset is not available" — keep goalset shapes FIXED (pad to max_goalset)
   per planner instance.

### Env/ops notes
- Disk hit 0 bytes free mid-run (whole root fs). Freed ~19 GB via `pip cache purge` +
  `conda clean -tp`. Biggest consumer: `~/.cache/huggingface` = **67 GB** (not touched —
  ask Big Will). USC checkpoint cleanup (authorized): deleted 204 ckpt files, 997 MiB.
- One Isaac process at a time; smoke tests ≤64 envs; short bash timeouts;
  images/videos are saved for Big Will, never viewed here.

### SPIKE FINAL: **GO — 90% (9/10), 0.29 s/full grasp plan, 632 MiB GPU.**
Winning construction: `table_candidates()` — match proven grasp-table entries by
RADIUS, azimuth-rotate the whole pose to target (joint1-offset trick), small radial
residual only. Rigid lateral translation of 2–3 cm = 0/16 IK even in free space.

### Stage 1 expert bring-up (`expert/run_expert_v1.py`)
- Full loop runs in Isaac on Play-v1: plan_grasp → 50 Hz execution (resample 0.025→0.02,
  #692 trim) → close+clean-check → lift verify → attach_from_scene → place goalset →
  release → home (plan_cspace). First full-success episode achieved (ep3 run 2:
  both cans, 0 failed grasps, all clean). Video: `expert/expert_v1_env0.mp4`.
- Gotcha: `terminations.object_dropping` auto-reset mid-episode scrambles the scene —
  disabled for expert runs (with `rewards.dropping_penalty`, which reads it).
- Failure anatomy → fixes v2: (a) lying cans (25% of v1 spawns) need live-can-height
  targeting (table has low-z pocket entries from the old sweep pass); (b) settle 6 steps
  at descent end before closing (PD lag caused air-closes); (c) place goalset now
  includes azimuth-rotated carry-band orientations (carry_waypoints.pt q_over FK) —
  fixes near/center-azimuth basket place-plan failures; (d) descend to z 0.07 before
  release (bounce-out risk); attempts 2→3.
- v2 run: grasps solved (12/12 lifts, 0.17 failed/ep) but places 7 fails → v3 place
  ladder: aim at emptier basket half (+0.022 m), retry heights (0.13,0.16),
  carry-config plan_cspace rung. Lying grasps: align finger-separation axis (tool ±Y)
  or finger dir with the can's cylinder axis (end-cap pinch / axial); exclude
  previously-failed goalset winner on retry.
- v3 run (6 eps): **67% success, clean_grasp 100%, failed_grasps 0.17/ep**. Remaining
  failure = second can into occupied basket (all rungs refused, 2/6). → rung 4: direct
  joint interp to azimuth-adjusted q_over (old expert's standard carry move) so a
  place attempt always happens.
- **v4 eval (12 eps, randomized basket+spawns+lying): 100% success (12/12),
  failed_grasps 0.17/ep, clean_grasp 92%, plan_fail 0** (3 pose-place refusals all
  recovered by fallback rungs). Video: `expert/expert_v1_env0.mp4`; JSON:
  `expert/expert_v1_results_12ep.json`. 32-ep confirmation run left going overnight
  (`expert_eval_32ep.log`). Note: two background eval runs were externally killed
  mid-run (cause unknown — possibly disk-crisis reboot aftermath); relaunching
  detached with nohup + durable logs worked.
- 32-ep A/B sequence (late-night lesson: one change per run — violated, then corrected):
  v1-tail config **87.5%** (plan_fail 4, clean 89%); +4 fixes (edge-fallback, reset_seed,
  hop, lost-recovery) → 81.3% plan_fail 17; −reset_seed → 71.9% plan_fail 13;
  −edge-fallback (kept hop + lost-recovery) → **84.4%, plan_fail 6, failed 0.22,
  clean 82%**. Verdicts: unfiltered edge candidates and mid-session reset_seed() are
  both HARMFUL; hop + lost-grip-refetch are neutral-to-good. Final config = current
  files. Honest nominal estimate: **~85–88%** (32-ep SE ≈ 6 pts).
- Gate 1 gap (95%): (a) pose-place refusals ~11/60 places (root-cause the trajopt
  refusal from lift pose — attached-sphere fit? basket-wall activation?); (b) lying-can
  region edges (no in-family candidates — needs a true lying grasp family, not table
  reuse); (c) second-can basket congestion; (d) carry-direct precision when r_err>0.03.
  Then: nudge/perturbation + teleport-recovery suites, batched demo-gen, formal
  clean-grasp gate.

## 2026-08-01 morning — Big Will's video review + labeling infra
- **Video verdict (Big Will): expert looks clean/good.** Directive: keep miss-then-recover
  episodes, but the ACT policy must learn RECOVERY, never stochastic misses — mask the
  failed attempt's steps out of the BC loss.
- **Label design (mine, beyond the directive):** mask boundary = failure DETECTION.
  Masked (mask=0): failed attempt's approach/descend/close/lift ("missed") and any
  transport that dropped the can ("lost"). Trainable: everything post-detection —
  reopen (outcome "recovery"), re-approach, refetch. Added placement PROVENANCE
  (`via` = pose/pose-high/carry/carry-direct/+hop; carry-direct = unplanned interp →
  excluded from nominal pool, ablate in recovery pool) and per-attempt close_disp +
  clean flag (exact attempt-granular first-attempt-clean filter). Rides ACT's
  `action_is_pad` channel; ORs with the perturbation censor (which stays ablatable;
  the failure mask is unconditional). PLAN.md §2.2/§4.1 updated to match.
- run_expert_v1.py now emits per-episode `segments` [{t, phase, seg}] + `outcomes`
  {seg → {outcome,...}}, and dumps every rung-1+2 place refusal to
  place_fail_cases.pt {q6, tgt, basket_xy, held, cans} for offline replay.
  Both changes are planner-behavior-neutral (labels + dumps only — A/B-safe in one run).
- Validation: 12-ep run (expert_eval_12ep_labels.log). Offline root-cause tool
  replay_place_fail.py (subagent-authored): hypothesis ladder H0 baseline /
  H1 no-attach / H2 no-basket-walls / H3 higher-goals / H4 start-state-validity.
- 12-ep label validation: labels correct (segments step-indexed, per-attempt
  outcomes, mask frac 15.5% on a failure-heavy sample). 8/12 success (12-ep noise);
  ALL 4 failures were lying cans (no-candidates or 3x plan-fail at r 0.19–0.29) —
  zero episodes lost to placement. 4 place refusals dumped -> place_fail_cases.pt.

## 2026-08-01 — ROOT CAUSE FOUND: place refusals = attach frame bug (Gate 1 gap (a))
- Replay ladder verdict on all 4 dumped cases: H1 (no-attach) PASS 4/4, everything
  attached FAIL 4/4 including plan_cspace-to-home and no-walls — "Start state in
  collision" with the attached object.
- probe_attach.py showed the mechanism: `fit_spheres` returns sphere centers in
  **WORLD frame** (docstring claims obstacle frame — the cuboid pose is baked into
  the trimesh via `get_trimesh_mesh(transform_with_pose=True)`), and
  `attachment_manager.update()` with `world_objects_pose_offset=None` writes them
  **verbatim as link-local** on attached_object. Result: phantom can ~30 cm from the
  gripper (world (0.60,-0.12,0.02), 8 mm inside the table) while the real can is at
  (0.28,-0.06,0.10). Refusal rate was pose-dependent lottery: phantom lands in
  air -> plan succeeds (with a bogus obstacle!), lands in table/walls -> refusal.
- **FIX (one line): pass `world_objects_pose_offset=Pose.from_list([0,0,0,1,0,0,0])`**
  — update() then applies ee.inverse() @ identity to the world-frame centers,
  converting them correctly to link-local. Probe after fix: spheres land exactly on
  the can (0.279,-0.065,0.102), 8 cm clear of table, no self-overlaps. Replay after
  fix: 6/6 hypotheses PASS on all 4 cases. NOTE: this bug poisoned EVERY transport
  plan since bring-up (phantom obstacle somewhere arbitrary) — fix may improve paths
  generally, not just refusals. Upstream cuRobo doc/frame bug worth reporting.
- Also seen: "morphit_sphere_fit: attempt 0 returned too few spheres" — fit gives
  3-4 spheres (2 real ~r0.019 + degenerate r<0.004) for the can; acceptable, but a
  fixed 4-sphere analytic can model would be more deterministic if slip issues appear.
- 32-ep confirmation (single change = attach fix): expert_eval_32ep_v5_attachfix.log.

## 2026-08-01 — 32-ep attach-fix confirmation: PASS (place refusals eliminated)
- expert_eval_32ep_v5_attachfix.log, single change vs v4 = the identity
  world_objects_pose_offset on attach_from_scene.
- **place_plan_fail: 0** over 32 eps / ~59 placements (v4 rate was ~1 per 5-6 eps;
  ~18% of places refused). Gate 1 gap (a) CLOSED.
- success 27/32 = 84.4%; failed_grasps_mean 0.50; clean_grasp_frac 80.3%.
- Remaining failure anatomy (5 eps):
  - 4 eps involve lying cans: air-closes (close_disp=0.0 at tz=0.028 vs can top
    0.024) and one NO-CANDIDATES at r=0.203 (table has nothing below r 0.221).
    -> next A/B change (already staged): tz offset 0.002 (not 0.016) when lying.
  - 1 ep (23->24): upright can at (0.308,0.027) PLAN-FAIL x3 AFTER first can was
    already in basket -> second-can congestion specimen (gap (c)); "Start or End
    state in collision" x4 in stderr at matching timestamp. Investigate after
    lying fix (one change at a time).
  - one benign "place rung1 IK-none; retry high" (ladder recovered; different
    mechanism than the attach refusals).
- Next run: 32 ep, single change = lying tz fix; expect lying air-miss failures to
  drop substantially toward Gate-1 95%.

## 2026-08-01 — Stage 2 infra landed (recorder + vendored ACT + dataset)
- run_expert_v1.py --record-h5: per-step obs(41)/action(7) capture + build_train_mask
  (mask=0 over missed/lost segments; unit-tested). Byte-neutral when flag absent.
- act/ vendored from LeRobot @ 2aba372b (2026-07-31, Apache-2.0, PROVENANCE.md lists
  every modification): modeling_act.py, configuration_act.py, normalize.py,
  dataset.py (RebotDemoDataset), train_act.py skeleton. All CPU tests pass.
- Obs layout (from ObservationsV1Cfg, verified): [0:8] joint_pos_rel, [8:16]
  joint_vel_rel, [16:32] objects_canonical, [32:34] basket_center_xy, [34:41]
  last_action -> observation.state=[0:16], environment_state=[16:41] (16/25 split).
- dataset action_is_pad[j] = (t+j >= T) or (train_mask[t+j]==0) — expert-failure
  mask rides the pad channel; verified vendored loss divides by VALID-step count and
  feeds pad mask to the VAE encoder too (censored steps out of latent).
- GOTCHA for eval wrapper: this LeRobot version does normalization OUTSIDE the
  policy (processor pipeline). train_act.py normalizes explicitly and stores stats
  in checkpoints; deployment MUST normalize obs and UNNORMALIZE predicted actions.

## 2026-08-01 — 32-ep lying-tz eval (v6): REGRESSION, fix rejected
- Single change vs v5: lying close height off 0.016 -> 0.002 (tz 0.028 -> ~0.014).
- 25/32 = 78.1% (v5: 84.4%); plan_fail 4 -> 9; place_plan_fail still 0 (attach fix
  holding). ALL 7 failed eps are lying-can episodes.
- Anatomy: tz=0.014 ADDS lying PLAN-FAILs at r 0.24-0.29 (finger-table margin/
  candidate native-z mismatch) while air-closes persist anyway (disp=0.0000,
  lift z 0.008 even at the low close height). So lying failure is NOT a pure
  close-height problem — candidate geometry vs lying cylinder needs real diagnosis.
- Also 2 eps NO-CANDIDATES at r 0.167/0.179 (known: table empty below r 0.221).
- Decision: do NOT keep off=0.002; before another sim eval, run offline tz-sweep
  probe on the exact failing states (extracted from demos_smoke_32ep.h5 obs) to map
  plan feasibility vs tz. Recorder h5 smoke-test PASSED (32 demos, 3.3 MB).

## 2026-08-01 — Lying-can root cause candidate: align-filter family bug
- H5 trace (demo_19) killed the tz theory: at this run's per-env scale (~0.8x ->
  lying center z=0.008, top ~0.018), v6's clipped tz=0.012 is MID-BARREL (correct
  height), yet object_b was never touched across 3 "successful" grasps (can moved
  <=0.3 mm) and object_a was pushed/rolled 29 mm during descend+close.
- table_candidates() align filter (spike_plan_grasp.py ~line 180) scores
  max(|dot(tool_y, axis)|, |dot(-tool_x, axis)|): accepts BOTH end-cap pinch
  (fingers on the can's flat ends — span 29-44 mm at [0.8,1.25]x scale, at/beyond
  gripper opening, rims push the can away) AND axial barrel grip (reach along axis,
  fingers straddle the barrel — the family that actually works). cuRobo picks by
  plan cost, blind to family -> lottery; failures = end-cap winners.
- Scale facts (pick_place_v1_env_cfg): per-ENV scale in [0.8,1.25] sampled at
  startup (num_envs=1 -> constant within a run; v5 and v6 both show resting z
  0.008 -> same scale, A/B intact). NOTE for A/B methodology + Stage 2 stratified
  gen: single-env runs pin ONE scale for all episodes.
- Plan: v7 single change vs v6 = axial-only align filter (drop the tool_y term);
  tz probe (offline, running) informs whether off=0.002 stays.

## 2026-08-01 — TRUE lying-can root cause: table-scrape stall lottery (FK audit)
- Align-family theory REFUTED by data: zone (end-cap vs axial availability) vs
  outcome over all 22 v6 lying attempts = 56% success in BOTH zones. The axial-only
  filter edit was reverted untested. (Also: az_t-159deg+180 == az_t+21deg — the two
  reference azimuths are the SAME line; only one family line exists.)
- FK audit of v6 h5 (commanded q from recorded actions, achieved q from obs):
  commanded TCP z is UNIFORM (-0.006..-0.014) across all lying attempts — planning
  consistent; executed pocket lands ~tz-0.011 => at tz=0.012 the fingertips sweep
  AT/BELOW the table surface. Execution then stalls on contact 7-52 mm above
  command: stall <=17 mm -> barrel trapped anyway ("grasped"), stall >=28 mm ->
  air-close ("missed"). Success was a stall lottery, explaining ~56% flat rate,
  close_disp=0.0 misses, AND v5-vs-v6 both being mediocre (0.024 vs 0.012 bracket
  the sweet spot).
- v7 single change: lying off 0.002 -> 0.012 (tz ~0.020 at this run's scale) —
  executed fingertips ~centerline; probe bonus: tz>=0.020 serves r<0.221 targets
  (kills NO-CANDIDATES). Run: expert_eval_32ep_v7_lyingtz020.log + demos_smoke_v7.h5.
- Tools: probe_lying_tz.py (offline tz sweep), FK audit one-off (commanded-vs-
  achieved TCP from h5 actions/obs — technique worth keeping).

## 2026-08-01 — v7: GATE 1 NOMINAL PASSED (96.9%)
- expert_eval_32ep_v7_lyingtz020.log, single change vs v6 = lying off +0.012
  (executed-fingertip centerline calibration). 31/32 = 96.9% (>=95% GATE MET);
  plan_fail 1, place_plan_fail 0, clean_grasp_frac 87.3%, failed_grasps 0.31/ep.
- Ladder: v4 (attach bug) -> v5 84.4% (attach fix) -> v6 78.1% (tz too low,
  scrape lottery) -> v7 96.9% (calibrated tz; air-closes AND NO-CANDIDATES gone).
- Sole failure: lying can at r=0.205 NO-CANDIDATES — grasp-table tail below
  r~0.21 even with tz=0.020 mid-band serving. Residual ~1-2%/ep. Future fix:
  extend table with proven low-r lying entries (ties into Stage-0 reachability
  audit), not another tz change.
- demos_smoke_v7.h5 (32 eps, 31 nominal-successful) recorded — first Gate-1-grade
  demo material.
- Next: 32-ep --perturb shakedown (machinery live-validation + Gate 1 perturbed/
  recovery pillars), then multi-seed 500-demo Stage-2 generation.

## 2026-08-01 — v8 perturb shakedown: machinery GO, recovery 71.9% (<90 gate)
- 32/32 events fired (21 nudge / 11 slip), labels + episode_kind + perturb_steps
  all flowing to h5 (demos_smoke_pert.h5). Injection order verified: recorded
  action == executed action during slips.
- Perturbed/recovery success 23/32 = 71.9% vs Gate-1 >=90 target.
- Failure buckets: (a) slip-drop lands near/against basket -> refetch PLAN-FAIL
  (4 eps; slip recovery 7/11) — needs congested-refetch strategy (freer-side
  approach / higher tz retry / push-out); (b) nudged lying can -> repeated
  air-closes at new spot (2 eps); (c) post-nudge congested plan-fails (3 eps).
  Note: some slip "successes" may be drops INTO the basket (placed check passes).
- Decision: freeze runner; run diversify smoke then launch the 500-demo nominal
  chain overnight (Gate-1 nominal passed; Stage-2 gen doesn't depend on perturb
  fixes). Recovery engineering resumes from demos_smoke_pert.h5 offline traces
  after the chain (mid-chain edits would fork later seeds' behavior).
- Slip-failure drop audit (demos_smoke_pert.h5 final states): 3/4 failed refetches
  are cans that ROLLED outside the servable annulus (r 0.207 / 0.214 < 0.221 floor;
  r 0.343 > 0.326 ceiling), all lying. NOT basket congestion. Structural fix =
  grasp-table extension (low-r + high-r lying entries, offline cuRobo gen+verify)
  — also covers v7's sole nominal failure. Queued after the demo chain.

## 2026-08-01 — Stage-2 nominal demo chain LAUNCHED
- Diversify smoke (8 ep, seed 999): 7/8, gi winners spread 0-11, order shuffle
  4/4 split — multi-modality preserved. demos_smoke_div.h5.
- Chain: gen_demos_nominal.sh, seeds 101-108 x 63 eps = 504 episodes, --diversify,
  per-seed demos_nominal_s<seed>.h5 + results snapshot. ~11 h GPU. RUNNER FROZEN
  until chain completes (each seed reloads the module).
- Queue after chain: grasp-table extension (low/high-r lying) -> perturb suite to
  >=90 -> Stage-3 recovery data gen -> ACT nominal training (Gate 2).

## 2026-08-01 — Morning tooling complete (all CPU-validated, GPU-pending)
- act/dataset.py: multi-file support + nominal_pool_filter / recovery_pool_filter
  (attempt-granular; v7 split 18 nominal / 13 recovery / 1 fail).
- act/report_coverage.py: stratification report (r-band/orientation/basket/scale/
  mask%). Note: scale proxy uses t=0 (pre-settle) — switch to settle-end later.
- act/eval_act.py: BatchedACTController (per-env queues, external normalize/
  unnormalize, discontinuity flush per PLAN 4.2), success = mdp.placed_mask (same
  predicate as expert + env metrics; objects_canonical NOT used — target-first
  reordering makes slots unstable). GPU checklist in agent report / this entry's
  sibling files. Vendored select_action bypassed (shared deque across batch).

## 2026-08-01 ~13:05 — chain nearly done; failure anatomy; train launch staged
- Chain seeds 101–107 done (~28 min each), seed 108 running, ETA ~13:26. Success 379/441 = 86.0% (vs v7 nominal 96.9%).
- Failure anatomy (62 eps): 35 miss+plan-fail-mix (typ. one can delivered, other can 1–3 plan fails — goalset infeasible, consistent with the known grasp-table annulus gap r<0.221 / r>0.326), 21 repeated-miss (air-close lottery residue, likely worse at small scales), 3 lost-in-transit, 2 all-plan-fail, 1 other. Both dominant modes already on roadmap (table extension; scale-aware close height is a candidate if misses persist).
- Coverage over s101–107: 265 nominal-clean + 114 recovery demos, 177,704 / 82,763 chunk samples; r-bands, lying (~26%), basket sectors all populated; mask 3.5–9.2%/file.
- train_act.py: --data now nargs=+, new --pool {default,nominal,recovery} wiring dataset pool filters. Verified: nominal pool over 7 files = 265 demos (matches report_coverage).
- Next (on chain-done watcher): re-run coverage incl. s108 → check GPU free → launch ACT nominal train (100k steps, batch 64, chunk 50/15, lr 1e-5) → monitor loss.

## 2026-08-01 ~13:20 — PIVOT: ACT → flow-matching chunk policy (Big Will directive)
- Big Will: full pivot, no ACT baseline training — flow matching is the future-proof choice for harder tasks ahead. FM still emits action chunks (π0: chunk 50 via 10 Euler steps), so chunk/execute-15 receding horizon, dataset, masks, normalizer, eval controller all carry over.
- Research survey (subagent, full report with URLs archived in session transcript; key facts in PLAN §2.3):
  - **ActionFlow (arXiv:2409.04576) REJECTED** — SE(3)-pose-space equivariance (Invariant Point Attention + Lie-algebra flow) inapplicable to our 7-D joint-delta actions; no public code.
  - **Adopted recipe** (π0/LeRobot-pi0/X-IL standard): keep ACT transformer, drop CVAE+KL, decoder queries = projected noisy chunk + AdaLN time conditioning, rectified-flow loss v=x1−x0 elementwise MSE (loss mask multiplies in unchanged — LeRobot pi0 does exactly this), 10 Euler steps inference (flat down to ~5).
  - **Honest expectation** (Much Ado About Noising, ICLR 2026, 28-benchmark controlled study): FM ≈ L1 parity on state-only single-task few-hundred-demo regimes; payoff is future multimodality. Gate 2 bar unchanged.
  - **Stage 6 de-risked**: RFS (arXiv:2602.01789) = residual RL on frozen FM base precedent; determinism via seeded/fixed x0 per chunk (deterministic map from (obs,x0)).
  - Vendor references: TorchCFM (FM math), much-ado-about-noising (cross-check). Vendored ACT stays as emergency baseline, NOT trained.
- Docs updated: PLAN.md (architecture line, stack table, §2.3 rewritten, Stage 6, ledger), memory, tasks #21/#23.
- Next: chain s108 done → coverage → write act/modeling_flow.py + act/train_flow.py (reuse dataset/normalizer/pool filters/--pool CLI) → CPU shape/overfit test → GPU train nominal pool.

## 2026-08-01 ~13:45 — chain DONE; flow policy implemented, verified, TRAINING LAUNCHED
- Chain: 8/8 seeds, 504 eps in 3h47m, 85.7% success. Final pools: 292 nominal-clean / 140 recovery; 314k trainable samples; all coverage bands populated; mask 7.7%.
- act/modeling_flow.py (340 L) + act/train_flow.py (154 L) + eval_act.py flow dispatch (subagent, spec = PLAN §2.3). Key implementation choice: decoder self-attention gets key_padding_mask from action_is_pad so censored/garbage positions can't leak into valid positions' velocity predictions (load-bearing for mask invariance). Time cond: sinusoidal τ×1000 → MLP → added to tokens. CVAE deleted; queries = Linear(x_τ)+sinusoidal pos.
- CPU gates (subagent ran, I re-ran & reproduced): (a) shapes/finite OK, 703k params small cfg; (b) mask invariance BIT-IDENTICAL loss under garbage×1e3 in censored region; (c) seeded x0 determinism exact; (d) 1-chunk overfit 62.3× loss drop, Euler-sample MAE 1.196→0.057; (e) train smoke + ckpt round-trip through eval_act.load_checkpoint + BatchedACTController.
- TRAINING: runs/flow_nominal_v1 (pid 56981) — 292 demos/196,033 samples, 100k steps, batch 64, lr 1e-4, chunk 50/15, 10 Euler steps. ~45 steps/s, 1.0 GB VRAM, ETA ~40 min. step 100 loss 0.478.
- Next: on completion → GPU eval checklist (smoke 4 eps → flush sanity → determinism → 64-ep Gate 2 eval ≥85%); then grasp-table extension.

## 2026-08-01 ~14:40 — Gate 2 first pass 59.4%; horizon bug found+fixed; recovery-pool retrain launched
- flow_nominal_v1 trained clean (100k steps/35 min, loss 0.478→0.06 plateau). First smoke 0/4 → DIAGNOSED as eval-harness bug: task default horizon 500 steps but demo lengths median 677/max 1234 (only 7.9% ≤500) — policy could never finish. Offline teacher-forcing had already cleared the policy (MAE 0.047 vs scale 0.56; gripper sign 99%). Fix: eval_act.py --episode-length-s (default 30 s).
- Smoke 8ep @30s: 4/8. Gate 2 64ep seed 42: **59.4%** (twice, reproducible), flushes 6. Instrumented eval (placed_max/placed_final/max_can_z per ep — diagnostic only): 26 fails = 15× "one can placed, second never" + 9× "nothing placed" + 2 misc; placed_max==placed_final ALWAYS → NO drop-after-place mode. Signature = stuck-after-miss, exactly the skill absent from nominal-clean pool (zero regrasp examples by construction).
- Action (one change): retrain on --pool default = all 432 successes (292 nom + 140 recovery, miss steps masked/recovery supervised) → runs/flow_nomrec_v2, same hypers. Then re-eval 64ep seed 42.
- Note for later: 0.10 lift threshold in max_can_z diag is above the expert's lift apex (~0.095) — read max_can_z raw, don't trust the >0.1 cut.

## 2026-08-01 ~15:15 — v2 A/B verdict: recovery pool HURTS at equal weight; v1 stays champion
- flow_nomrec_v2 (all 432 successes, same hypers): **51.6%** @64ep seed 42 vs v1's 59.4%. Fail anatomy WORSENED where it matters: nothing-placed 9→20 eps. Equal-weight recovery data (140 demos, 26% of pool; longer noisier episodes) dilutes nominal precision instead of adding a stuck-state skill.
- Interpretation: offline recovery demos ≠ on-policy recovery. The stuck states the policy reaches are ITS OWN miss states, not the expert's; distribution mismatch. This is the textbook argument FOR DAgger (PLAN Stage 3) and mirrors the RL-era lesson: shaping/data tweaks plateau, on-policy correction closes gaps.
- CHAMPION: runs/flow_nominal_v1/ckpt_final.pt @ 59.4% (64ep seed42, 30s horizon). runs/flow_nomrec_v2 kept for reference; safe to delete later.
- Video: eval_act.py got --video/--video-length (RecordVideo, rgb_array, offscreen) — recording 4 eps (seed 7) of v1 champion to runs/flow_nominal_v1/videos/ for Big Will's review (I do not view rendered output).

## 2026-08-01 ~15:25 — SESSION END (computer restart). Video recorded; HANDOFF.md written
- Video: runs/flow_nominal_v1/videos/flow_v1_champion_4ep_seed7.mp4 (4 eps seed 7, 3/4 success in-recording, 60 s) — for Big Will's review.
- Full resume state: reBot_ACT/HANDOFF.md (+ memory act-pipeline-plan.md points there). No processes running; GPU idle (16 MiB).

## 2026-08-01 evening — DAgger collector shakedown ladder (seed 201 throughout)
- v1 crash: CUDA-graph capture order — takeover led with plan_cspace (retreat-home), expert runs lead with plan_grasp goalset-16; cuRobo can't re-capture. Fix: discard warmup goalset plan at startup.
- v2 (8 rollouts): machinery works end-to-end but gate=miss ALL 8 at t≈140 = close+early-lift window → 25-step/absolute-z miss gate fired on SUCCESSFUL grasps (contradicted 59% eval rate); forced reopen mid-lift also poisoned takeover start states (4 instant plan-fail takeovers).
- v3 gate fix: miss now rise-relative (45 closed steps AND no can rose 2 cm above streak-start z; real lifts clear it by ~31). Result: diverse trips (miss/stall/drop) BUT 0/7 policy successes → drop gate was firing on normal basket releases (>2 cm fall from >5 cm IS placement). Fix: exempt cans within 0.10 m of basket center. Also stall 300→450 (healthy first-can progress observed past t=500).
- v4 (10 rollouts): 4 policy_success (~expected), gates genuine. Takeovers only 2/6 successful → autopsy of failures h5 (obs decode at takeover_t): (a) 2× lying can at r=0.192/0.207 — BELOW grasp-table annulus floor 0.221, expert has no entry (same gap as Gate-1 perturb 71.9%); (b) instant-fail with zero grasp attempts, both cans servable — policy miss leaves gripper pressed at can level → cuRobo start-state collision kills all plans (expert misses never hit this: their lift precedes detection).
- Fix (b): scripted up-retreat (shoulder −0.35, elbow −0.15, 30-step interp, grip open) after reopen, before any planning. v5 running.
- Mask integrity verified on written takeovers: train_mask[:takeover_t] sum 0, post-takeover 100% trainable, RebotDemoDataset + nominal_dagger_pool_filter load clean.
- QUEUE: v5 verdict → grasp-table extension (GPU; now blocks BOTH Gate-1 recovery and DAgger takeover yield — confirmed twice by data) → full collection (~300 rollouts) → retrain nominal+dagger → 64ep eval vs 59.4%.

## 2026-08-01 late — v5/v6 shakedowns: takeover start-state root-caused OFFLINE; ROUND-1 COLLECTION LAUNCHED
- v5: policy_success 6/10 (matches 59.4% eval — gates fully calibrated). But up-retreat didn't fix instant takeover fails. OFFLINE REPRO (exact recorded q6+cans through cuRobo, ablated worlds): 3/3 failures = start-state collision — demo_0 table+can, demo_1 can only, demo_2 table only; SAME scenes plan fine from Q_HOME. Policy parks arm within cuRobo sphere margins (PhysX contact-free ≠ cuRobo margin-free). First repro attempt was INVALID (wrong plan_grasp signature made even Q_HOME fail) — always replicate the exact Expert call.
- v6 fix: takeover retreat = up (0.35/0.15) then 40-step interp to Q_HOME, hold, then plan. Result: instant-fails GONE (all failures now 450-970 steps of real expert effort), miss takeovers 3/4 recovered. Remaining 3 failures = 1× annulus (lying r=0.207) + 2× expert repeated-miss on in-range lying cans (post-policy nudged states) — expert weaknesses, not collector defects.
- VERDICT: collector VALIDATED. Round-1 collection launched: 300 rollouts / target 100 takeovers, seed 202, -> expert/dagger_r1.h5 (+_failures.h5, .json), ~3.5 h. Parallel: grasp-table extension BUILD on CPU (GPU verify queued behind collection).

## 2026-08-01 late — grasp-table extension tool BUILT (CPU-verified; GPU verify queued)
- extend_grasp_table.py (subagent): synthesize DONE — 6258 candidates (5040 analytic family sweep, 1218 edge-entry re-target), 168 scenes, covering lying r 0.14–0.37 + upright-pocket r 0.14–0.22 (all previously-empty bins now have 270–480 candidates). extension_candidates.pt written.
- Table format learned: grasp_table.pt = {q (12953,6), pocket (12953,3), fwd (12953,3)}; new entries need a SOLVED q → verify harvests q from winning plan_grasp solves + re-FKs pocket/fwd for serve consistency.
- Serve switch: REBOT_GRASP_TABLE env var (default byte-identical). Format compat PASSED: gap targets r=0.19/0.37 lying + r=0.16 upright serve full K=16 from merged table (original serves 0), pocket error ~5e-9 m; exclude/retry padding intact.
- GPU steps queued (after DAgger r1): smoke `extend_grasp_table.py verify --limit 6 --rounds 1 --out /tmp/ext_smoke.pt --report /tmp/ext_smoke.json`; full `verify` (~20-40 min) -> grasp_table_extended.pt + extend_report.json. Then A/B the extended table (perturb suite re-run for Gate 1; optional DAgger round-2 with REBOT_GRASP_TABLE set).

## 2026-08-01 ~18:15 — ROUND-1 VERDICT: DAgger works but needs iteration+diversity; table verify started
- Collection r1: 100 takeovers / 208 rollouts (~1h45m), takeover success 68% (retreat fix held), 46 failures banked. Pool sanity: 0 mask/success violations, 96,298 samples, filter loads 392 demos total (292 nom + 100 dagger).
- flow_dagger_v3 (nominal+dagger, one change): **59.4%** — aggregate unchanged vs v1. BUT per-episode analysis: v1-vs-v1 across two runs = ZERO churn (eval fully deterministic per spawn) while v3-vs-v1 = **17 fixed + 17 broken + 9 both-fail**. DAgger recovery genuinely transfers (17 fixes are real, above a 0-churn floor); the regression is the cost of single-seed/single-scale collection data (seed 202 pinned one object scale for all 100 takeovers).
- Round-2 plan: (1) verify grasp-table extension (smoke running); (2) collect r2 FROM v3 with REBOT_GRASP_TABLE=extended + 3 seeds (203/204/205, ~80 rollouts each) for scale diversity — v3's 17 newly-broken states get collected on-policy by construction; (3) retrain nominal+r1+r2 → 64ep A/B. The 9 both-fail episodes are annulus/hard-geometry — extension's job.

## 2026-08-01 ~19:45 — Gate-1 perturb re-run (extended table) verdict: 65.6% — AGGREGATE UNINFORMATIVE, anatomy shifted; instrumented re-run launched
- v9 suite (32 eps, --perturb, REBOT_GRASP_TABLE=extended): recovery 65.6% (21/32) vs v8 71.9%. BUT both runs used seed=None (OS entropy — rng unrecorded, episodes unpaired, not reproducible). Δ=2 eps inside unpaired n=32 noise → NO aggregate conclusion. Lesson: perturb suites must pass --seed; recorded from now on.
- Plan-fail anatomy DID move as designed: 9 plan-fails, 8 at lying r=0.175–0.199 (BELOW the measured r=0.20 physical floor — the acceptable bucket), 1 outlier at r=0.262. The 0.20–0.221 annulus gap no longer produces plan-fails: extension works at plan level.
- Dominant residual mode is now PHYSICAL misses: 7/11 failed eps show 2–6 failed grasps — ep-10 pattern: post-nudge attempt rakes can 2.9 cm (bad alignment), then retries close on pure air (close_disp=0.0, can never lifts) at refetched positions. Open question: are these served by NEW table rows (plan/FK-verified only — never physics-executed) or is it can-still-rolling-at-close dynamics?
- Instrumentation added (A/B-exempt): spike_plan_grasp.LAST_CAND_ROWS (table row behind each served candidate; rows >=12953 = extension) + run_expert_v1 attempt line now logs xy/r/lying/row/new + plan_to_close_drift (can motion between plan fetch and close — the rolling-can signal). Semantics untouched (row list mirrors poses through align-sort/exclude/padding).
- v10 instrumented suite RUNNING: --episodes 32 --perturb --seed 301 (SEEDED — reproducible), extended table, log expert_eval_32ep_v10_perturb_instrumented_s301.log. Buckets to read: (a) fails at r<0.20 → true floor, acceptable; (b) air-miss with high plan_to_close_drift → rolling-can timing, fix = settle-wait before plan; (c) air-miss on NEW rows with low drift → extension entries physically bad → need physics-verify pass on the 506.

## 2026-08-01 ~20:20 — v12 geometry verdict: expert air-misses are NOT geometry; aperture instrumentation launched
- v12 (paired seed 301) reproduced v10 EXACTLY (77.4% recovery, same aggregates) — suite deterministic, instrumentation non-invasive.
- Pocket-formula lesson (2 bugs of mine): physical pocket = tool_origin − 0.048·x̂ — calibrated against the grasp table's own {q,pocket} pairs (std=0 over 200 rows). TCP_OFF=0.075 is the PLACE-hover convention, not the grasp offset. Planner tool-frame z carries a ~+0.097 constant bias vs can center on successes → analyzer reads z relative to success median. Formula self-check: successes |dp_xy| median 3.7 mm → SANE.
- **Failed lying attempts have the pocket AT the can**: |d_along| median 1.5 mm, |d_perp| 2.8 mm, |d_z| 3.4 mm (rel), align 0.79–1.0 — indistinguishable from successful attempts. Geometry/alignment/z-height ALL ruled out. Fingers close in the right place; can shows close_disp≈0 and never lifts.
- Style split (fwd_dot from logged x̂): LYING-BAD = 17 AXIAL / 5 ENDCAP; LYING-OK = 3 AXIAL / 7 ENDCAP. AXIAL (approach along can long axis) over-represented in failures BUT 3 AXIAL succeed with identical align/fwd profiles → style correlates, doesn't determine.
- Can is a CUBOID 0.024×0.024×0.036 (not a cylinder). Lying = 3.6 cm axis horizontal.
- Discriminator now running (v13, seed 301 paired): finger joint positions (indices 6,7) at close. aperture≈closed-full ⇒ can not physically between fingers (position/penetration mystery); aperture≈2.4 cm ⇒ gripped but lift slips (force/friction). q6() confirmed MEASURED joints → FK honest.
- Retry-exclude defect confirmed in data (secondary): exclude drops list POSITIONS, re-serves the just-failed row (4550→4550, 3113→3113, 782→782, 11288→11288). Fix candidate: exclude by table row. Deferred until air-miss mechanism known.

## 2026-08-01 ~20:50 — v13 aperture verdict: failed lying grasps close to EMPTY at the right place; pad-sweep hypothesis; video+finger-pos run launched
- Gripper convention (pick_place_env_cfg): joint 0.045=OPEN, 0.0=CLOSED, aperture≈f1+f2.
- v13 (paired s301, aggregates identical again): LYING-BAD fingers close to ~0.000/0.000 (FULLY CLOSED, nothing between) at a position obs puts within ~5mm of the can (xy validated, can rest z identical OK-vs-BAD at 0.0083–0.0086 — sunken-can hypothesis DEAD). LYING-OK stall at f1+f2≈0.0395 (endcap grip on the 3.6cm axis) or ≈0.0248 (grip across the 2.4cm width). UPRIGHT-OK ≈0.0245. Aperture arithmetic decodes grasp style exactly.
- grip_override ruled out (self-clearing, forces OPEN; slip events only).
- Remaining hypothesis: finger PAD swept volume misses the can by mm — pads have finite extent, approach is ~57° tilted; the kinematic pocket point is at the can but the physical pad patch passes above/beside. Style stat supports (AXIAL 17 bad / 3 ok vs ENDCAP 5 bad / 7 ok) but doesn't determine.
- v14 running: 10 eps s301 (reproduces failing eps 8+9), finger BODY world positions logged at close + video of ep 9 for Big Will (new --video-ep flag). Caveat to check: camera may perturb determinism — confirm ep8/9 failure signatures reproduce before trusting the video.
- Also instrumented meanwhile: cuRobo failure status now surfaces in PLAN-FAIL lines (next runs) — seeded-run plan-fails were at IN-BAND r (0.202/0.223/0.281 ×3 retries), suspect neighbor-can collision margins, NOT the r<0.20 floor story from unseeded v9.

## 2026-08-01 ~21:10 — ROOT CAUSE of expert lying air-miss: pads close ~8mm ABOVE the can top; retry z-ladder A/B launched
- v14 (10 eps s301, finger BODY positions logged; determinism held with camera on): failed lying closes have finger bodies at z=0.096–0.108 — the UPRIGHT-grasp altitude — over a lying can whose top is 0.020. The one successful lying grip: z=0.092. xy is dead-on (≤9mm) in ALL cases. Bodies: gripper_left/gripper_right; pad catch-zone bottom ≈ body_z − ~0.072.
- Mechanism: post-perturb lying cans rest at z=0.008 (nominal calibration assumed 0.012) → tz = clip(0.008+0.012)=0.020; executed pad height SPREADS 16mm across table-row families (sag varies) → low-sag rows catch, the rest sweep air. v7's off=+0.012 calibration was valid for nominal rest height; the 4mm-lower perturbed rest eats the entire margin. v6 lesson still binds below (tz 0.012 = table-scrape stall lottery) so a blanket lowering risks the 96.9% nominal gate.
- FIX (one behavioral change, v15): retry z-ladder — lying retries target tz_extra=−4mm×attempt (cap −8mm, clamp floor 0.012 unchanged); attempt 0 BIT-IDENTICAL → nominal Gate-1 preserved by construction. Converts the currently-wasted identical retries (exclude-by-position defect) into a descending probe.
- Video of failing ep 9 for Big Will: expert/expert_v1_ep9.mp4 (gripper visibly closes above the lying can).
- A/B RUNNING: 32 eps --perturb --seed 301 paired vs 77.4% baseline (v10/v12/v13 identical). Watch: LY-BAD attempt-1/2 conversions; plan-fail statuses (new instrumentation) for the in-band r=0.223/0.281 failures.

## 2026-08-01 ~21:35 — v15 ladder verdict: WEAK (reverted); paired-A/B contamination lesson; v16 closed-loop z-fix launched
- v15 suite read 67.7% vs 77.4% BUT the suite comparison is INVALID: perturb schedule + candidate dropout share one rng stream, so any behavioral change that alters the number of draws (extra attempts, different retries) RESHUFFLES every later episode's perturb event — 12/31 episodes drew different events. Same-seed pairing survives only up to the first behavioral divergence. LESSON: for expert A/Bs, per-ATTEMPT metrics are the primary signal; suite-level n=32 is unpaired ±8pt noise once behavior changes. (Future option: dedicated rng streams per subsystem.)
- Ladder attempt-level: lying retries 2 lifted / 16 missed / 4 extra plan-fails (v13: 0 / 14 / 0). Open-loop −4/−8mm can't span the 16mm executed-height spread, and lower targets exit the feasibility manifold (goalset None). REVERTED.
- v16 (running, s301): closed-loop grasp-height correction — after descend on lying cans, measure finger-BODY z (PhysX) vs can_z + PAD_BODY_OFF (0.084, v14-calibrated), if >4mm high nudge shoulder/elbow via numerical-jacobian least squares (planner FK, no cuRobo replan — replan from descended pose risks start-state collision), hold 12 open steps, re-measure once. Cap 12mm/iter, joint-delta cap 0.15 rad. Non-lying and in-tolerance paths bit-identical.
- PRIMARY METRIC: lying close-attempt lift rate — v13 baseline 10/32 (31%). Secondary: suite rate (unpaired), zfix lines (dz distribution, post-nudge convergence).

## 2026-08-01 ~22:00 — v17: bad rows are UNEXECUTABLE (pads clamp 3-5cm high, nudges do nothing); v18 = detect-and-swap
- My v16 PAD_BODY_OFF=0.084 was calibrated from POST-LIFT logging (the attempt print runs after the lift traj — fingers rode up 8cm). At close the true offset is ~0.004; v16 never fired (wasted run, killed). LESSON: know WHEN in the trajectory an instrumentation line executes before calibrating from it.
- v17 (corrected 0.004): zfix measurements show failing rows execute the close with finger bodies at fz=0.048-0.071 vs target ~0.012-0.019 — THREE TO FIVE cm high, not 8mm. Jacobian nudges (12mm commanded) move them 1-4mm or even backwards: the arm is CLAMPED (joint limits / contact) — commanding lower does nothing. The successful lying row in the same episodes needed no correction (executes at target). So cuRobo-valid poses split into executable and unexecutable row families, and only detection helps.
- v18 (running, s301): detect-and-swap — measure at close; dz>10mm → abort BEFORE closing, reverse out open along the planned lift traj (start-state-collision safe), EXCLUDE THE TABLE ROW (new exclude_rows plumbing through plan_grasp/table_candidates — also fixes the exclude-by-position re-serve defect), retry with a different row; closed-and-missed lying rows also excluded. 4-10mm residuals still get one nudge. metrics["misexec"] counts aborts.
- PRIMARY METRIC unchanged: lying close-attempt lift rate (v13 baseline 10/32=31%); plus misexec-abort → conversion rate on subsequent attempts.

## 2026-08-01 ~22:30 — REDIRECT (Big Will): stop expert iteration; analyze POLICY failure modes + residual-RL fit; failure videos
- v18 detect-and-swap KILLED mid-run on redirect. Parked OPEN and honest: v18 aborted attempts at dz=12-14mm that v17 measured at <=4mm (e.g. row 13362 ep2), AND its ep2 state diverged from v17 despite an identical prefix (object_a can r=0.305 vs 0.320, different perturb t) — unresolved run-to-run divergence, so v18's 33 aborts are on non-comparable states. Expert perturb pillar stands at 77.4% measured (v10/v12/v13), mechanism understood (unexecutable low lying poses), fix unproven. Code state: exclude_rows plumbing + measure-abort logic in place (behavioral, UNVALIDATED).
- POLICY failure taxonomy (champion flow_nominal_v1 @59.4%, 64ep diag JSON): 26 fails = 16 placed1_stuck (of which ~10 SECOND CAN LIFTED to 0.08-0.10 but never placed + ~6 second can never lifted) + 8 never_lifted_anything + 2 lifted_never_placed. Flushes negligible (6). DAgger r1 on-policy gates: 91% miss / 9% stall / 0 drop (146 takeovers). Expert recovery failures skew lying (LL 54% of failed recoveries vs 18% of successes) — same expert lying weakness.
- => ~90% of policy failures are PRECISION failures at two moments: grasp alignment (miss->stuck) and delivery (carried->never placed). Residual-RL (ResiP/RFS precedent) targets exactly this class. Open question for video review: is "carried->never placed" a steering error (residual-fixable, arm-only) or a release-timing error (needs grip in residual — plan §6 deviation)?
- Close-up failure video RUNNING: eval_act 1 env x 16 eps seed 42, viewer eye (0.7,-0.5,0.4) lookat (0.22,0,0.05), one mp4, ep i = video [30i, 30(i+1)]s. New eval diagnostics: final_can_pos, final_placed_per_can, basket_xy (per episode).

## 2026-08-02 — POSTMORTEM.md written (Big Will: "failure is good, allows us to learn")
- Full write-up at POSTMORTEM.md: architecture spec, results ladder, why "BC is a data problem" is only 1/3 of the story (covariate shift YES, state aliasing NO, interference NO — both measured), FM known-limitations cross-referenced against what we hit (gripper-air-close stuck mode matches arXiv 2503.23835 verbatim; frame-only aliasing matches IntentVLA/DSSP; FM multimodal collapse matches VFP), and the limitations WE discovered (planner-valid≠executable, rng-stream A/B contamination, post-lift instrumentation trap, exclude-by-position re-serve).
- Key new measurements backing it: per-phase v1-vs-v3 divergence on identical nominal states (lift grip |Δ|=0.357, reopen 0.901, arm ~0.05≈1cm fingertip); on 1466 nominal hold states v3 opens-in-horizon 20.3% vs v1 12.1% (8.7% v1-holds-v3-opens) → the mid-carry-drop mechanism behind v3's 17 broken episodes.
- Memory updated (bc-flow-postmortem). Video eval of 16 close-up episodes re-launched post-restart (first attempt killed by reboot).

## 2026-08-01 ~22:4x — exact per-version failure taxonomy (Big Will asked for detail)
- Reclassified v1 + v3 64-ep evals per-episode (scratchpad taxonomy_v1v3.py / transition_v1v3.py). CORRECTION to POSTMORTEM §3: v1 placed1_stuck split is 5 lifted-to-carry / 11 never-lifted-2nd (earlier "≈10/≈6" was wrong, from memory not data). POSTMORTEM taxonomy rewritten as v1-vs-v3 table + transition matrix.
- v1→v3 transitions (same 64 spawns): 17 fixed (11 placed1_stuck, 5 never-lifted, 1 lifted-never-placed → success = the DAgger-supervised miss states), 17 broken (9 → placed1_stuck, 4 → lifted_never_placed, 4 → never_lifted = downstream grip damage), 9 fail-both, 21 succeed-both. lifted_never_placed 2→5 = mid-carry-open signature matching §5 interference measurement.
- v2 checkpoint deleted in cleanup; only JOURNAL aggregates survive (51.6%, nothing-placed 9→20).

## 2026-08-01 ~23:30 — Experiment ladder started (pre-residual-RL causal ablations)

Big Will's directive: before more DAgger or residual RL, answer (a) can obs identify
grasp success, (b) does execute-15 cause precision failures, (c) is DAgger interference
reproducible. Full docs in `experiments/` (EXP_INDEX.md + one doc per experiment,
pre-registered beliefs + decision rules). Literature notes: `experiments/LITERATURE.md`
(PACE 2606.00537, ResiP, CR-DAgger, IWR/Sirius, ResFiT, RFS — all IDs verified).

- **EXP01 DONE (probe, CPU):** grasp-success info IS present single-frame (AUC 0.954
  full-obs, **0.976 from 4 finger dims alone**). Finger-only probe rejects **100% of 665
  on-policy closed-on-air states** (the exact freeze states); full-obs probe mislabels
  53.5% of them — salience failure demonstrated end-to-end. History adds ≤0.01 AUC and
  transfers WORSE. Discovered + validated a confound: can-config alone predicts expert
  misses at AUC 0.74 (task difficulty), inflating no-finger variants. Verdict: grasp-bit
  obs surgery justified; history conditioning NOT the fix. `experiments/exp01_results.json`.
- **EXP02 running:** n=8 → **32.8%** (−26.6 pts vs n=15's 59.4%!). Anatomy: never-lifted
  8→21 (commitment dithering), lifted-never-placed 2→11 (grip lottery: ~12% spurious-open
  per replan × 2× replans). 8 fixed / 25 broken. Awaiting n=4/2/1 monotonicity check.
- **EXP03 queued:** `train_flow.py` had NO seeding (v1 vs v3 differ in every random draw
  — replica variance never measured). Added `--seed`; 3 nominal + 3 dagger replicas
  overnight (`experiments/exp03_run_replicas.sh`).

## 2026-08-02 ~06:45 — EXP02 collapse + POSTMORTEM §5 RETRACTION (forensics)

- **EXP02:** n_action_steps 15→8→4→2 = 59.4% → 32.8% → 3.1% → 0.0%. Monotone; never-
  lifted 8→21→35→50; first-ever post-place drops at n=4. Chunk commitment is load-
  bearing; shorter fixed horizons are catastrophic. Fix direction = PACE phase-aware
  cutting / per-step residual, NOT faster replanning. (n=1 pending.)
- **POSTMORTEM §5 "kill shot" RETRACTED after forensic re-analysis** (new tool
  `experiments/exp03_grip_divergence.py` failed to reproduce it; original code recovered
  from transcript; ablation matrix in `experiments/exp03_divergence_forensics.py`):
  original had (1) v3 inputs normalized with v1's stats, (2) missed-attempt empty-lift
  frames counted as "hold states", (3) s101[:30] subset. Exact replication reproduces
  0.121/0.203/0.087 → execution faithful, method flawed. Corrected: **v1 ≡ v3 on true
  holds (13.6% vs 13.7% open, 0.7% divergent); on empty lifts v1 opens 5.1% vs v3 67.1%
  — v3's grip discrimination is BETTER, its recovery training worked offline.** v3's 17
  broken episodes now unexplained → unseeded train variance top suspect → EXP03 replicas
  are decisive. Correction block added to POSTMORTEM §5; belief revision pre-registered
  in EXP03 doc BEFORE replica data; memory bc-flow-postmortem rewritten.

## 2026-08-02 ~04:00 — EXP03 replicas: INTERFERENCE REFUTED; seed variance dominates everything

Chain 00:23–03:44 (6× train+eval, seeded). Success: nominal arm 32.8/50.0/59.4%,
dagger arm 60.9/53.1/56.2%. Pairwise success/fail churn within-arm 31–39 episodes;
v1↔v3 churn = 34 = **the "17 fixed/17 broken" IS the replica noise floor.** Grip
open-rate on true holds flat across all 8 checkpoints (12.6–13.1%). Consequences:
- Belief 1 (interference reproduces) REFUTED; EXP04 descoped. Dagger arm mean 56.7% vs
  nominal 47.4%, tighter spread — weak evidence recovery data STABILIZES.
- Every single-run A/B in the ladder is void, incl. v2 "offline recovery hurts" (51.6%
  inside the 32.8–59.4% nominal spread). POSTMORTEM §3 correction block added.
- v1 "champion" = lucky seed. Held-out seed-123 evals of D1/N3/v1/v3 launched
  (`experiments/exp03_run_heldout.sh`) for variance-aware champion selection.
- New standing rule: ≥3 seeds per training config, select on held-out spawns.
- N1 (32.8%) ≡ N3 (59.4%) in offline grip metrics → seed damage is closed-loop
  compounding, invisible to expert-state probes.

## 2026-08-02 ~04:45 — Held-out champion selection: exp03_N3 @ 64.1% pooled

Seed-123 held-out evals: N3 68.8%, D1 57.8%, v1 57.8%, v3 50.0%. Pooled over both
64-ep suites: **N3 64.1%** > D1 59.4% > v1 58.6% > v3 54.7%. New champion / frozen
residual-RL base: `runs/exp03_N3/ckpt_final.pt` (nominal pool, --seed 3). Also learned:
eval-suite noise at n=64 is ±5–10 pts (N3 +9.4, v3 −9.4 across suites) — cite pooled
numbers. Experiment ladder EXP01–03 COMPLETE (task #25); next: EXP06 residual RL design
(grasp-bit + base-action + phase inputs; ResiP baseline, RFS x0-steering expectation),
pending Big Will's read of the ladder results.

## 2026-08-02 morning–afternoon — EXP06 additive residual: two burned runs, then CLOSED exactly flat

Design per EXP06_residual_rl.md (pre-registered): frozen N3 base + per-step
α·tanh arm-joint residual (α=0.1), 64-D obs, PPO. What happened:
- r1/r2 collapsed to 0% with episodic reward ≈ −30. Root cause (found by diffing
  the executed actions, not the configs): **rl_games `clip_actions: 100` default
  rescales sampled actions ×100 before our tanh** — saturated, slammed joints.
  Standing rule: **clip_actions MUST be 1.0** in every rl_games config here.
- r3 (fixed) trained healthy (+1610 episodic) but the held-out verdict was
  **exactly flat: 55.5% → 55.5% pooled**, 26 fixed / 26 broken on identical
  spawns (causal, not churn-vs-noise), learned residual state-independent
  (~0.0084 |res| on success AND failure episodes).
- Side measurement that set up EXP07: **frozen-x0 sweep** — success spans
  14.1–56.2% across x0 draws; x0=zeros is the best single mode (55.5% pooled
  deterministic vs 64.1% stochastic = an 8.6-pt determinism tax).
- Verdict: additive nudging can't re-select the chunk family; the base's
  failures are mode errors. Pivot to x0-steering approved by Big Will.

## 2026-08-02 afternoon–evening — EXP07 x0-steering: designed, gated, trained, CLOSED at 91.4%

Doc-first per convention (EXP07_x0_steering.md: 6 pre-registered beliefs incl. a
62–72% result guess, gates S0/1/2/3, escalation + seed-variance rules). Build:
z ∈ R⁷ per 15-step window, x0 = tanh(z) broadcast over the chunk, free-running
controller (z enters only via refill x0), 56-D steering obs, window-aligned
eval-protocol training env (drop termination+penalty off; 30 s = 100 windows),
bare placed-stream reward. Gates:
- **Gate 1:** z=0 bit-exact vs the x0-zeros base, both suites, zero flips.
- **Gate S0:** exploration response measured BEFORE training — bias U(±0.125)
  harmless (60.9%), σ0.3 ≈ −2 pts (locally flat), σ0.6 −20 pts (real slope);
  σ_init −1.2 chosen from data.
- **Gate 2:** smoke + full run healthy. Window-RL logging lesson: episodic
  metrics are silently 0.0 until ~epoch 5 (episode=100 windows > epoch=24
  windows); judge at first episode completion (+1450 ≈ base level).
- Training s1_seed1: 200 epochs @2048 envs (~3.7 h wall, paused/resumed cleanly
  at ep~105 via rl_games --checkpoint; reward +1450 → +2133 → +2320 → +2416).
- **Gate 3 (held-out, deterministic z=clamp(mu)): 89.1% @42, 93.8% @123 =
  91.4% pooled vs 55.5% base — Gate 6 (90%) cleared.** Taxonomy
  (exp07_analyze_s1.py): 51 fixed / 5 broken; never-lifted collapsed 18/19;
  z state-dependent (|x0| 0.220 success vs 0.282 failure; z_std 0.21 vs 0.26).
  Belief scorecard in the EXP07 doc; POSTMORTEM §9 has the distilled mechanism.
- **Champion stack: frozen `runs/exp03_N3/ckpt_final.pt` + steering head
  `runs/exp07_steer/s1_seed1/nn/exp07_steer.pth`.**

## 2026-08-02/03 — eva_bc shared repo + cutover

- Curated public repo `github.com/wwangg22/eva_bc` (act/, expert/, experiments/
  code; docs/ with all write-ups; README with staged pipeline + Lessons
  Learned). Initial push 1c04eca; EXP07 verdict synced as 4471be9.
- **Cutover done per Big Will's plan: reBot_ACT is now the eva_bc working
  clone** (adopted its git history in place; runs//weights/assets/third_party
  stay local via .gitignore + .git/info/exclude; sync_from_source.sh retired in
  fc21c45). Docs live under docs/ — the old root-level copies are pending
  deletion. Pushes now happen straight from reBot_ACT.
- Close-up success videos of the final policy: eval_steer.py gained
  --video/--viewer-eye (ported from eval_act.py); recording in
  runs/exp07_steer/videos_closeup/ for Big Will's review.
