# 10 — Stage 4: RL on the frozen base (x0-steering), planned in depth

**Written 2026-08-03, after a line-by-line review of `act/steer_core.py`,
`act/steer_wrapper.py`, `act/train_steer.py`, `act/residual_core.py` and
`act/modeling_flow.py`.**

Read `09_STAGE2_BC_PLAN.md` first — this stage consumes its checkpoint. Nothing here can be
executed until Gate 2e has a number.

---

## 0. THE INHERITED EVIDENCE — **REWRITTEN 2026-08-03 (retraction R13)**

> **The original §0 is retracted.** It read, in full:
>
> > *"`04_EVA_BC_LEARNINGS.md` §0 is the single most important structural fact in the whole
> > inheritance … | **Gate 3** — the result | **NEVER REACHED** | … **The method eva_bc's own
> > README recommends has never produced a success number on any task.** We are not porting a
> > proven recipe. We are porting one that passed its plumbing gates and was cut short, and
> > every plan built on it must carry a hedge that does not depend on it."*
>
> That was true when written on 2026-08-03 and became false the same day. Upstream resumed
> the run and closed EXP07. Kept visible rather than deleted, per the project rule that the
> record of what was believed and when is part of the evidence.

The upstream pull (`eva_bc` `1c04eca..818391b`, reviewed in `12_UPSTREAM_SYNC.md`) supersedes
it:

| EXP07 stage | state in eva_bc |
|---|---|
| implementation | landed, CPU unit-tested |
| **Gate 1** — z = 0 reproduces the fixed-x0 base | **PASSED bit-exact, zero flips, both suites** |
| **Gate S0** — exploration-noise tolerance | **PASSED**, σ_init = −1.2 chosen from data |
| **Gate 2** — training health | **PASSED**, +1450 (ep5) → +2416 (ep200) |
| **Gate 3** — the result | **PASSED — 55.5 % → 91.4 % pooled, Gate 6 (90 %) cleared** |

**x0-steering is now a proven method — on one task, against one failure class.** 128 held-out
episodes, two suites, deterministic both sides, paired on identical spawns, **+35.9 pts** over
the like-for-like base, on the **first** pre-registered configuration with no PPO tuning.

The discrimination signature is what makes it credible rather than lucky: **51 fixed / 5
broken** (against EXP06's symmetric 26/26), the `never_lifted` bucket collapsing **18 of 19**,
no new failure mode invented, and a **state-dependent** z (mean |z| 0.220 on successes vs
0.282 on failures). PPO learned *where* to intervene, not merely *how hard* — which is exactly
what the additive residual failed to do.

### 0.1 So is Stage 4 de-risked? No — and the reason changed, not vanished

The old hedge rested on "the method is unproven." That justification is gone. A different one
replaces it, and I think it is the stronger of the two. From `12_UPSTREAM_SYNC.md` §5:

1. **EXP07 repaired a recoverable failure; ours is terminal.** Steering worked upstream
   because re-choosing z every window *is* retry logic — 18 of 19 `never_lifted` episodes were
   "committed to a bad grasp, never retried." A clutter topple **ends the episode**
   (`distractor_toppled`). There is nothing to retry. Steering has to be right in the one
   window that straddles the close. §3 items 3 and 4 said this before the upstream result and
   are untouched by it.
2. **We have no base to steer.** Stage 2 has not produced a checkpoint. Upstream's lever was a
   base with large exploitable mode structure (frozen x0 draws spanning 14.1 %–56.2 % success
   on identical weights). Whether *our* base has any such spread is unmeasured and is the most
   informative number obtainable before any PPO — see H1.
3. **Different scale.** Upstream ran 2048 envs; this is a 10 GiB card and **Q7 has never been
   run.** At 512 envs, matching upstream's 9.8 M window-transitions costs ~15 h, not 3.7.

**§8 therefore stands in full.** What changes is the cost basis: **H1 (a 32-draw `fixed_x0`
search before any PPO) is now cheap insurance rather than the primary plan**, because the
primary plan has a success precedent behind it.

---

## 1. THE CONTROL MODEL, AS REVIEWED

```
obs57 = core.build_obs(obs42, env)      # features only, no controller state
z     = policy(obs57)                   # (N, 7), rl_games-clamped to [-1, 1]
core.set_steer(z)                       # x0 = alpha_x0 * tanh(z), broadcast over the chunk
for _ in range(15):                     # ONE rl_games step = one 15-env-step window
    core.flush_check(env)
    action = core.controller.act(obs42) # refills consume the CURRENT steer_x0
    obs42  = env.step(action)
```

Three properties that are easy to misread and are load-bearing:

1. **The controller stays free-running.** `steer_core.py:1-31` is explicit: z enters *only*
   through the x0 of refills that happen while it is held. Each env refills when **its own**
   queue empties. This is what made eva_bc's Gate 1 come back bit-exact, and it **supersedes**
   the synchronous-window sketch in eva_bc's HANDOFF.
2. **`steer_x0` outranks `fixed_x0`** (`eval_act.py:110-115`). Setting one does not disable
   the other; the precedence does.
3. **The window reward is an undiscounted sum of 15 raw env rewards**
   (`steer_wrapper.py:48,57`), so `gamma: 0.99` discounts **per window**, i.e. per 0.3 s, not
   per 0.02 s. An episode is 46 windows, so the effective horizon is ~100 windows of
   discount against a 46-window episode — discounting is essentially off within an episode.
   That is a deliberate consequence of the window design, not an accident, but it should be
   said out loud.

Also worth recording: `steer_wrapper.step()` **fully overrides** the base wrapper's `step`,
so it does **not** apply `torch.clamp(actions, ±clip_actions)`. Clamping happens only inside
rl_games' `preprocess_actions`, on `z`. The arm action from the flow policy is never clamped.
See §2/N7 of the BC plan for why that matters a great deal here.

---

## 2. WHY x0-STEERING IS MECHANISTICALLY WELL-MATCHED TO CLUTTER

This is not an appeal to eva_bc's recommendation. It is an argument from the measured
geometry, and it can be wrong in ways §3 names.

**The residual failure is a 7.8 mm margin problem.** From P22:

```
finger blade reach        ~47 mm along the opening axis, ±19.2 mm perpendicular
neighbour faces at        ±27 mm
margin                    27 − 19.2 = 7.8 mm
the blades sweep          26.5 mm each, INTO the row's x-band, while actuating
```

**The lever x0 provides is exactly a few-millimetre shift of the whole committed chunk.** It
does not add an off-manifold correction (eva_bc's EXP06 residual did, and it fixed 26 episodes
while breaking 26). It selects a different sample from the base policy's own conditional
action distribution — a different, still-coherent, still-executable trajectory.

**And the information needed to choose is already in the observation.** From P26's gap
stratification over 768 episodes:

| min free gap | n | success |
|---|---|---|
| 0–4 mm | 30 | 90.0 % |
| 4–6 mm | 82 | 91.5 % |
| **6–8 mm** | **233** | **64.4 %** |
| 8–10 mm | 303 | 71.0 % |
| 10–14 mm | 120 | 82.5 % |

Failures are **not** uniformly distributed over spawns — they concentrate in a measurable
band. And the free gaps are computable from `obs[23:35]`, which is the distractors' `(dx, dy)`
relative to the target. So a steering policy has both a lever of the right size and the input
needed to aim it.

**The pre-registered mechanism, therefore:** *a steering policy reduces topple by shifting the
grasp pose a few millimetres as a function of the measured free gaps.*

**The falsifier is the EXP06 signature.** eva_bc's residual was **state-independent** —
~0.0084 units everywhere, identical on successes and failures. PPO learned *effort*, not
*discrimination*, despite having a grasp bit and relative poses in its observation. If our z
comes back state-independent, the same thing happened and the mechanism above is refuted
regardless of what the success number does. **§6 Gate 4c tests exactly this, and eva_bc's
EXP07 listed it and never ran it.**

---

## 3. WHY IT MIGHT FAIL — the honest case against

1. **The blade geometry is fixed.** 47 mm of reach against a 42 mm row pitch. If *no*
   achievable pose avoids sweeping a neighbour, no amount of steering helps, and the 22 %
   topple rate is a property of the gripper, not of the policy. The counter-evidence is that
   78 % of spawns already do **not** topple under a *single* nominal pose — so the achievable
   set is not empty; it is spawn-dependent. That is what makes conditioning plausible.

2. ~~**The steered x0 is off-distribution, and nobody has checked.**~~ **DOWNGRADED
   2026-08-03 (retraction R14) — upstream checked, and it is fine.** Original text kept below
   because the *geometry* it describes is still correct; only the verdict changed.

   Upstream registered this identical concern as EXP07 belief 5 ("broadcast z may be too
   coarse"), pre-committed an escalation to a 14-D constant+ramp parameterisation, ran it, and
   recorded: **REFUTED — 7-D constant-per-chunk z sufficed; escalation never triggered.** A
   rank-1 x0 *is* off the joint training distribution and the decoder handles it anyway. In
   hindsight that is unsurprising: a rank-1 draw sits inside the support of the *marginal*
   per-position distribution even though it is wildly atypical jointly, and the model is
   trained to map a neighbourhood of noise to coherent chunks.

   **Gate S0a is kept anyway** — one task's evidence that a structural assumption survives is
   not proof it survives on another with different action-space geometry, and S0a costs ~20
   minutes. Its **registered prediction changes**: from *"may produce incoherent chunks"* to
   *"expect coherence; the open question is authority, not validity."* If S0a shows incoherent
   chunks on our base, that is now a surprise worth a document of its own.

   *Original text, retained:* This is the finding from
   this review I am least comfortable leaving unmeasured.

   `set_steer` (`steer_core.py:54-62`) does

   ```python
   x0 = alpha_x0 * torch.tanh(z)                       # (N, 7)
   controller.steer_x0 = x0.unsqueeze(1).expand(-1, chunk_size, -1)
   ```

   so the noise is **constant across all 50 chunk positions** — a rank-1 tensor. But training
   drew `x0 = torch.randn_like(x1)`, i.e. **iid across positions**. A rank-1 x0 has roughly
   the right norm (`0.76 · √350 ≈ 14.2` against a typical `√350 ≈ 18.7`) and *completely* the
   wrong structure. The flow model has never seen one.

   That cuts both ways: it may be exactly why x0 has so much authority (eva_bc measured fixed
   draws spanning **14.1 %–56.2 %** success on identical weights), and it may be why a steered
   chunk is incoherent. ~~**Untested in eva_bc.**~~ *(tested and refuted upstream — see the
   R14 block above.)* Gate S0a tests it here, and §5 registers a distribution-preserving
   alternative.

3. **Clutter's failure is concentrated in a phase that is 70 env steps long** (or ~15 after
   P27). The relevant windows are the 1–5 that straddle the close. z applied in earlier
   windows is nearly free; z applied in the wrong window is nearly useless. PPO must find a
   narrow temporal target through a window-summed reward. eva_bc's task had its failure spread
   over a long grasp phase.

4. **A topple is terminal.** There is no recovery to learn — only avoidance. That halves the
   space of things RL could contribute.

---

## 4. THE TERMINATION DECISION (porting-map L4), DECIDED

`clutter_env_cfg.py:173-175`:

```python
topple_penalty = RewTerm(func=mdp.is_terminated_term, weight=-40.0,
                         params={"term_keys": "distractor_toppled"})
```

The penalty is keyed to the termination; nulling one without the other leaves a term pointing
at a name that no longer exists. eva_bc's EXP07 resolved the analogous
`object_dropping`/`dropping_penalty` pair by turning **both off**, so 30 s episodes were
exactly 100 windows and every reset landed on a window boundary — "zero train/eval mismatch".

**For clutter we keep `distractor_toppled` ON.** Reasons, in order:

1. **It is the task.** Turning it off is a benchmark modification in spirit even if not in
   code, and this effort's standing constraint is that env properties are findings to report,
   not defects to route around.
2. **`target_at_goal` already conjoins `& ~any_distractor_toppled`** (`clutter.py:65-73`), so
   the success predicate survives either way — but the *credit assignment* does not. The −40
   is the sharpest signal against the exact failure we are attacking.
3. **The misalignment it causes is bounded and measurable.** At most one termination per
   episode; the wrapper already calls `core.reset(done_ids)` on the auto-reset
   (`steer_wrapper.py:60-65`), so the controller re-syncs. The cost is up to 14 steps of a
   *fresh* episode executed under the *previous* episode's z, and those steps' rewards pooled
   into the terminating window's sum.
4. **Gate 1 is unaffected.** z = 0 ⇒ x0 = tanh(0) = 0 regardless of alignment, so bit-exact
   reproduction of the `fixed_x0 = zeros` base remains a valid, decisive test.

**Registered as arm B, not discarded:** `distractor_toppled = None` **and**
`topple_penalty = None` together, giving perfect window alignment. If arm A's training is
unhealthy in a way traceable to mid-window resets, arm B is the fallback and the comparison is
already designed.

**What must be logged to keep this honest:** `windows_containing_a_reset / total_windows`.
If it is a few percent, point 3 holds. If it is large, arm A is not what this section claims.

---

## 5. THE 57-D STEERING OBSERVATION FOR CLUTTER

eva_bc's 56-D = 41 base + a 15-D tail (fingers 4, grasp bit 1, target-in-gripper pose 7,
target→basket delta 3). Clutter's is 42 + 15 = **57**.

The 15-D tail, rebuilt:

| dims | eva_bc | **clutter** | why |
|---|---|---|---|
| 4 | finger pos 6,7 + vel 14,15 | **unchanged** — the proprio block is identical | EXP01: these are the salient dims |
| 1 | grasp-bit MLP | **threshold rule** — `\|gap − 0.036\| < 0.012` ∧ `obs[41] < 0` | BC plan N5; falls back to the MLP if P30 misses 0 % FPR |
| 7 | target-can pose in the gripper frame | **unchanged in shape**, `target` instead of a selected can | the two-object canonical selection (`residual_core.py:145-156`) **deletes entirely** |
| 3 | can → basket delta | `goal_delta = [0.185 − tx, −0.185 − ty, 0.035 − tz]` | `GOAL_XY` is a constant, not a per-env basket |

**Three mandatory changes to `task_features`:**

1. **Drop the `:159` quaternion permutation.** `subtract_frame_transforms` already returns
   XYZW in Isaac Lab 3.x (`isaaclab/utils/math.py:891-899`); the permutation maps
   `(x,y,z,w) → (y,z,w,x)`. It never showed up in EXP06/EXP07 because a randomly-initialised
   MLP learns whatever consistent bijection it is handed. Fix it here; do not copy it.
2. **Delete the canonical-object selection.** One target, no permutation, no `placed` masking.
3. **Add the topple margin — this is the EXP01 salience lesson applied one level up.**

On (3): eva_bc's most transferable representational finding is that *information can be
present in the observation and unused*. A 5-dim probe hit **AUC 0.968 at 0 % FPR** where the
same probe given all 41 dims mislabelled **53.5 %**.

Clutter's binding constraint is `min_i up_z_i < 0.75`. The four `up_z` values are present at
`obs[25, 28, 31, 34]` — but `min` over four interleaved dims is a nonlinear function an MLP
has no reason to form. **Surface it.** Likewise the free gaps, which §2 shows are what the
failures stratify on.

So the clutter tail is **15 + 5 = 20**, and `STEER_OBS_DIM = 62`, not 57:

```
[57]  min_i(up_z_i)                        the topple margin, explicitly
[58]  min_i(up_z_i) latched over episode    how close it has ALREADY come
[59]  min free gap  [m]                     the stratifier from P26 §6.1
[60]  free gap on the −y side of the target
[61]  free gap on the +y side of the target
```

The two signed gaps are there because the lever is a *signed* pose shift; a scalar minimum
tells the policy there is a problem but not which way to move. Symmetric information for a
symmetric decision.

**This is a deviation from the porting map, and it is deliberate.** Registered as arm
`obs=62` against arm `obs=57` (the straight port). If the 62-D arm does not beat the 57-D one,
the salience argument does not transfer and that is worth knowing — eva_bc's own evidence for
it is one probe on one task.

**Note the ordering constraint:** dims 57–61 must be appended *after* the eva_bc tail, so the
57-D arm is a strict prefix of the 62-D one and one checkpoint's stats cannot be silently
applied to the other (eva_bc corrupted a whole analysis that way).

---

## 6. GATES

Ordered. **No gradient step is taken before Gate 1 passes.**

### Gate S0a — does x0 have authority, and is the rank-1 form usable? *(new; eva_bc had no equivalent)*

*Design.* Freeze the base. Roll ≥64 episodes per condition on a fixed spawn seed:

| condition | x0 |
|---|---|
| A | `zeros` — the deterministic base, eva_bc's best fixed draw |
| B | fresh iid `N(0, I)` per refill — the free-running base |
| C | **rank-1**, `x0 = c·1` broadcast over the chunk, `c ~ U(−1,1)^7`, 8 draws — *the steering family* |
| D | rank-1 at reduced amplitude, `alpha_x0 ∈ {0.3, 0.6}` |
| E | **blended**, `x0 = α·tanh(z)⊗1 + √(1−α²)·randn` — distribution-preserving |

*Thresholds, registered:*

- **Authority:** spread across the 8 C-draws must be **≥ 10 points**. Below 5 points, x0 is
  not a useful lever on this task and DR4-a fires.
- **Coherence:** the best C-draw must be within **10 points** of A. If every rank-1 draw is far
  below the zeros baseline, the broadcast parameterisation is off-manifold in a way that
  breaks chunks, and **arm E becomes the primary** — `set_steer` gains a `blend` mode, which
  is a 3-line change and is a strict generalisation (α = 1 recovers the current behaviour).

*Prediction, registered.* eva_bc's fixed-x0 draws spanned 14.1 %–56.2 % on identical weights,
so **authority ≥ 10 points is likely and coherence is the real risk.** I expect C to show a
wide spread with a mean well below A.

### Gate S0b — exploration-noise tolerance

*Design.* eva_bc's EXP07 gate S0, unchanged in spirit: roll the base under `z ~ N(0, σ)` for
σ ∈ {0.05, 0.15, 0.30, 0.50}, ≥64 episodes each. Pick the **largest σ costing < 5 points**;
`sigma_init = log σ`.

*Why it cannot be inherited.* eva_bc chose σ ≈ 0.30 (`sigma_init = −1.2`) **from data on its
own task**, and its own record shows σ ≈ 0.37 costing **46 points** on an mm-precision base
(57.2 → 11.5 %) while σ ≈ 0.08 was harmless. Clutter's margin is 7.8 mm. **Selecting
`sigma_init` from precedent instead of from data is the single most likely way to burn a
training run here.**

### Gate 1 — z = 0 reproduces the base bit-exactly

`experiments/exp07_check_match.py`, ported: `--z-sigma 0 --z-bias 0` against the
`fixed_x0 = zeros` base, **episode-for-episode, exit nonzero on mismatch**. eva_bc's came back
bit-exact with **zero flips**; anything less means the wrapper is not transparent and every
later number is uninterpretable.

Run it **with `distractor_toppled` ON**, since that is arm A's protocol (§4).

### Gate 2 — training health

Read at epoch ~10 and ~50, against tripwires, **never against train reward alone** (eva_bc's
EXP06 r3 beat the base on train reward with exactly flat success):

| signal | healthy |
|---|---|
| `Metrics/clutter_success_rate` | rises above the base's Gate-2e number |
| `Episode_Reward/topple_penalty` | non-zero from epoch 1 — **if it is 0.0 the term is dead**, the cheapest tripwire in the repo |
| `Episode_Reward/success` | the dominant positive term |
| per-episode loggers | legitimately **empty until epoch ~2** — an episode is 46 windows, an epoch is 24. Do not read that as pathology. |
| `windows_with_reset / total` | small (§4 point 3) |

**Run `diag_training_env.py` before any full run.** Its four fixed-action conditions (zero /
bias / small-noise / large-noise, no RL) found in ten minutes what two rounds of training-log
analysis had misdiagnosed.

### Gate 3 — the result

**≥ 3 seeds, champion on a held-out spawn seed, pooled ≥ 128 episodes**, plus the §8 taxonomy
of `09_STAGE2_BC_PLAN.md` re-run and **diffed against the BC base**. eva_bc never reached this
gate. **Gate 4 target: ≥ 70 % pooled — the mission.**

### Gate 4c — is z state-dependent? *(the EXP06 falsifier)*

Log z for every window of the eval. Then:

1. `std over envs` at fixed window index vs `std over windows` at fixed env — a
   state-independent policy has the former ≈ 0.
2. Regress z on `[min free gap, gap−y, gap+y, target yaw, min up_z]`. Report **R²**.
3. Compare mean |z| on successful vs failed episodes.

**Registered threshold: R² < 0.05 on all seven z dims ⇒ the §2 mechanism is refuted**, whatever
the success number says. A success gain with a state-independent z is PPO having found a
better *constant* x0 — which is real, but it is `fixed_x0` tuning, and it should be reported
as that and reproduced far more cheaply by search over constants.

**Calibration added 2026-08-03 (R15) — what a genuine success actually looks like.** Upstream's
EXP07 passed this test, so its numbers set the scale, and the scale is *modest*:

| statistic | success episodes | failure episodes | separation |
|---|---|---|---|
| mean \|z\| | 0.220 | 0.282 | **+28 %** |
| within-episode z_std | 0.21 | 0.26 | **+24 %** |

against EXP06's refuted residual at a flat `0.0084` on both. **Do not demand a dramatic
split.** A ~25 % separation in the direction "larger and more variable where the base
struggles" is what a +35.9-pt result produced. The falsifier is *flatness*, not smallness.

---

## 7. REWARD ANALYSIS — magnitudes, computed

Per env step, from `clutter_env_cfg.py` `RewardsCfg`, before `reward_shaper.scale_value 0.01`:

| term | weight | value/step | notes |
|---|---|---|---|
| `reaching` | 2.0 | `2·(1 − tanh(d/0.10))` → **2.0** at contact, 0.48 at 100 mm | pays for hovering |
| `lifting` | 8.0 | 8.0 when above 55 mm and the ee is within 80 mm | |
| `extracted` | 15.0 | 15.0 when z > 90 mm **and nothing toppled** | |
| `carrying` | 12.0 | up to 12.0 once z > 70 mm | |
| **`success`** | **60.0** | **60.0 per step while at goal** | |
| `disturbance` | −3.0 | −3 × Σ planar displacement [m] → **−0.015** at 5 mm | |
| `topple_penalty` | −40.0 | **once**, terminal | |
| `action_rate` | −2e-2 | −0.02·‖Δa‖² | |
| `joint_vel` | −5e-3 | −0.005·‖q̇‖² | |

**Two consequences that change how the training log should be read:**

1. **The −40 topple penalty is 0.3 % of what a success is worth.** A successful episode holds
   the block at goal for ~250 steps → **+15 000 raw** from `success` alone. The real
   deterrent against toppling is *losing that stream*, not the penalty. Anyone who expects the
   −40 to dominate will misdiagnose the reward curve.
2. **`reaching` pays ~1 400 raw per episode for doing nothing but hovering.** That is the
   deceptive shaping `01_TASK_ANALYSIS.md` flagged. It is 9 % of a success, so it does not
   dominate — but it is the reward floor a "do nothing near the target" policy collects, and
   it is the correct **zero-action reference level** to log before training, exactly as
   eva_bc logged +1644.

**Register the zero-action reference before the first run.** It is one `diag_training_env.py`
condition and it is the tripwire that would have caught both of eva_bc's dead runs.

---

## 8. HEDGES — because §0

Ranked by expected value, to be run **in this order** if x0-steering stalls.

**H1 — `fixed_x0` search (nearly free).** If Gate S0a shows ≥ 10 points of authority, then
*before any PPO*, search over ~32 constant x0 draws and keep the best on a held-out spawn
seed. eva_bc's own data says this is worth up to **+42 points** between the worst and best
draw. It is a 32-eval search with no training at all, and it establishes the floor that
steering must beat to have earned its GPU time. **This is the highest-value item in the whole
stage and eva_bc never did it.**

**H2 — additive residual (EXP06).** Implemented, gate-tested, and **measured flat** on
eva_bc (55.5 % vs 55.5 %) with a state-independent residual. Low prior. Kept only because the
plumbing is already written and `residual_wrapper.py` needs no edits.

**H3 — PPO from scratch on the env's own reward.** The benchmark's intended path. **Blocked
by BC-plan N7** unless `clip_actions ≈ 2.0`: at 1.0 the goal is unreachable, at the shipped
100.0 training is destroyed. If N7 is confirmed, running this arm at 1.0 and reporting 0 % is
a *finding about the shipped config*, and running it at 2.0 is the fair baseline. **Run both**
— the pair is the finding.

**H4 — eva_rl's `train_bc.py` → `bc_to_rlgames.py` → PPO.** Needs a structurally correct
rl_games `.pth` as a transplant template, which needs a few epochs of `train.py` on clutter
first (`03_ENV_FACTS.md` §9). Cheap once H3 has run, since H3 mints the template.

**H5 — DAgger (Stage 3), conditional.** Only if the Gate-2e taxonomy shows covariate-shift
failures. **Structural caveat: a topple is terminal**, so DAgger can only address *pre-topple*
drift — strictly narrower than eva_bc's use of it. And eva_bc's own DAgger result was inside
the seed-variance noise floor.

---

## 9. DECISION RULES, PRE-REGISTERED

- **DR4-a.** Gate S0a authority < 5 points → **x0-steering is dead for this task.** Do not
  train. Go to H3 with both `clip_actions` values.
- **DR4-b.** Gate S0a coherence fails (every rank-1 draw far below zeros) → arm E (blended x0)
  becomes primary; `set_steer` gains a `blend` parameter; re-run S0a before proceeding.
- **DR4-c.** Gate 1 not bit-exact → **stop.** The wrapper is not transparent. No number
  produced after this point would mean anything.
- **DR4-d.** Gate 3 ≥ 70 % → **mission met.** Write it up, run `Tight-v0`, stop adding stages.
- **DR4-e.** Gate 3 < 70 % but Gate 4c shows real state-dependence (R² > 0.15) → the mechanism
  works and the budget was the limit. More epochs / more envs, not a new method.
- **DR4-f.** Gate 3 < 70 % **and** Gate 4c shows R² < 0.05 → the EXP06 failure mode has
  reproduced on a second task and a second parameterisation. **That is a publishable negative
  about the method**, and the honest move is to report it and fall back to H1/H3 rather than
  tune it.

---

## 10. COST

From eva_bc's 12 GB card; **re-measure under Q7 before committing** (10 GiB here).

| item | eva_bc | note |
|---|---|---|
| steering PPO, 2048 envs | ~740 windows/s ≈ 11 k env-steps/s → 200 epochs ≈ **3.7 h** | clutter episodes are 46 windows vs 100 |
| 3-seed replica chain | 6 × (train + eval) ≈ **4.7 h** | |
| Gate S0a | 5 conditions × 8 draws × 64 eps | **no training** — the cheapest decisive test in the stage |
| Gate S0b | 4 σ × 64 eps | no training |
| H1 fixed-x0 search | 32 × 64 eps | no training |

**Order of work:** Q7 → S0a → S0b → **H1** → Gate 1 → `diag_training_env` → Gate 2 → 3 seeds →
Gate 3 → Gate 4c.

H1 before any training is deliberate: it is cheap, it is informative either way, and it sets
the number PPO has to beat. Training first and searching afterwards is how a +42-point
constant gets attributed to a policy.

---

## 11. NON-NEGOTIABLES CARRIED FORWARD

- **`clip_actions: 1.0`** for the steering agent — z is the RL action there, so 1.0 is
  correct and the N7 problem does not apply. The shipped 100.0 destroyed two eva_bc runs.
- **`agent.params.config.name = rebot_clutter…`** — the shipped name `rebot_precision_slot` is
  shared by every challenge task and runs collide silently.
- **The key path is `agent.params.env.clip_actions`** (`eva_rl/scripts/rl_games/train.py:166`),
  not `agent.params.config.env.clip_actions`.
- **`mu_init` const 0.0 touches the weight only**; the bias stays `U(±0.125)`. Measured
  harmless, but it is why `diag_training_env.py` probes a `+0.125` bias condition.
- **`horizon_length: 24`** means 24 **windows** = 360 env steps ≈ half an episode.
- **`minibatch_size` in the yaml is dead** — overwritten at runtime as
  `horizon_length × num_envs // 4`.
- **Never shorten `n_action_steps`.** 59.4 → 32.8 → 3.1 → 0 → 0 % at 15 / 8 / 4 / 2 / 1.
- **One GPU job at a time.** 10 GiB card.
