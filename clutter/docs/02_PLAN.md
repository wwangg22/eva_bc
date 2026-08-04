# 02 — PLAN: the staged ladder for `Rebot-ClutterExtract-v0`

**Pre-registered 2026-08-02, before any code was executed.**

eva_bc's house convention: *design, beliefs and decision rules are written down before coding;
verdicts are recorded in place, including retractions, with dated blocks rather than silent
edits.* This document is that pre-registration. **Beliefs recorded here are predictions made
in advance, and they will be scored honestly whether or not they hold.**

Mission: **≈70 % success on random starts.** Big Will framed 70 % as a goal rather than a hard
requirement, *"especially if the task requires a lot of precision"* — which it does.

---

## Stage 0 — MEASURE. Is this task solvable at all?

**This is not setup. It is the load-bearing rung**, and it is the one eva_rl's own validation
ladder (`CHALLENGE_SUITE.md:328-339`, rung V4) has never cleared for this env.

### Why it comes first

Everything downstream — demos, BC, DAgger, RL — is unbounded risk until we know a solution
exists and roughly how reliable it can be. eva_bc's Stage-1 gate exists for the same reason,
and its README says the number *"matters because it caps what DAgger can later teach."*

### The questions

| Q | question | decides |
|---|---|---|
| **Q1** | finger collider thickness along the closing axis; swept width of a **closed** gripper; does any finger geometry reach below the ~44 mm TCP floor? | how much of each 12 mm gap a finger occupies; whether a closed "blade" fits a gap; whether strategy D's pusher clears the neighbours |
| **Q2** | does a lateral push at z ∈ {0.045, 0.055, 0.070} × {slow, fast} × {5, 10, 20 mm} topple a 0.025 kg distractor? | licenses or kills **B** and **C** |
| **Q3** | can the open gripper be aligned into a ≈6.94 mm window, closed on target + neighbour, and lifted with nothing toppling? | licenses **A** — the only strategy needing no push |
| **Q4** | does a closed-gripper push on the target's front face topple it backward cleanly, without disturbing neighbours? | licenses **D** |
| **Q5** | effective block/table friction, read off the stage — **not assumed** | sets every `h_crit` |
| **Q6** | commanded-vs-achieved TCP accuracy at x ≈ 0.25, z ≈ 0.05 under FK-CEM | whether ±1 mm alignment is attainable at all |
| **Q7** | throughput and VRAM at N = 16 / 128 / 512 / 1024 / 2048 on a 10 GiB card | every env-count budget in both repos assumes different hardware |

### Probes

| probe | answers | status |
|---|---|---|
| `p01_gripper_geometry.py` | Q1, Q5 | written; **verdict section must be rewritten** against the binary-aperture constraint before running |
| `p02_push_topple.py` | Q2 | to write |
| `p03_pair_capture.py` | Q3 | to write |
| `p04_target_topple.py` | Q4 | to write |
| `p05_reward_and_throughput.py` | Q7 + confirms the −0.8 topple arithmetic | to write |

All probes follow the `test_clutter_env.py` / `grasp_geometry.py` pattern: teleport plus
scripted action, batched 64–512 envs, failures collected in a list rather than asserted, JSON
written to `clutter/runs/`.

**Ground-truth grasp verdict** — from `grasp_geometry.py:207-211`, which eva_rl's handoff says
*"cannot be faked"*:

```python
gap  = 1.0035 * robot.data.joint_pos.torch[:, fing_dof].sum(dim=1) - 0.00125
encl = (gap - BLOCK_W).abs() < 0.012
rose = bpos[:, 2] > z0 + 0.045
held = rose & near & encl
```

### Beliefs going in (pre-registered)

1. **Q2 will come back unfavourable: a push at any reachable height topples the distractor**
   more often than it slides it. Quasi-static theory says every reachable height is above the
   16.7 mm threshold, by a factor of ~2.6. *Confidence: moderate-high.* If wrong — if a fast
   push slides — strategy **B** becomes the front-runner and the task matches its stated design
   intent.
2. **Strategy A (pair-capture) will work at the wide end of the gap distribution and fail at
   the tight end.** The alignment window is ≈6.94 mm nominal and shrinks with the measured
   7.0–18.8 mm gap spread and the target's ±0.20 rad yaw jitter. *Prediction: 40–70 % on
   nominal, near-0 % on `Tight-v0`.*
3. **Strategy D (topple the target) will work and will be the most reliable**, because it
   needs no mm-precision alignment — only a centred contact on a 30 mm-wide face. *Confidence:
   moderate.* Its risk is not physics but judgement: it is legal by the predicate and may be
   read as gaming the benchmark.
4. **Q6 will show FK-CEM achieving ≤2 mm TCP error** in the reliable band (x 0.22–0.26,
   z 0.045–0.10), i.e. good enough for A's ±3.5 mm nominal tolerance but marginal at the tight
   end. eva_bc's 16 mm executed spread was a *planner* artifact, not an arm limit — CEM scored
   on achieved FK should not reproduce it.
5. **Q7 will not support 2048 envs** with the flow controller on 10 GiB. eva_bc reported ~7 GB
   at 2048 envs on a 12 GB card with a *contact-light* task; clutter has 6 rigid bodies per env
   and `gpu_max_rigid_patch_count = 2**20`. *Prediction: 512–1024 is the working range.*

### Gate 0 — the decision rule

> **PASS if at least one of strategies A–D completes the task end-to-end (target in the goal
> zone, nothing toppled) in ≥50 % of a fixed 64-episode suite under privileged scripting.**

- **Pass** → pick the strategy with the best (success × spirit-compliance) and go to Stage 1.
  If two strategies pass, prefer the one that does not rely on toppling the target.
- **Fail on all four** → **stop and report to Big Will with the evidence**, including the
  measured gap distribution, the topple sweep, and the alignment-window arithmetic. Do not
  silently redefine the task, weaken the env, or fall back to "train RL and hope". A negative
  result here is a genuine finding about the benchmark and is worth more than a failed
  training run.

---

## Stage 1 — SCRIPTED EXPERT (closes V4)

Build the chosen strategy as a batched FK-scored-CEM expert with Cartesian waypoint chains.

### Design

Phase machine, using eva_bc's labelling contract so `build_train_mask` ports verbatim:

```
settle → (survey) → [singulate_+y → singulate_-y]?  →  align → insert → close
       → verify(gap & all four up_z) → extract(z>0.090) → carry → lower → release → retreat
```

Rules, each traceable to a measured failure elsewhere:

1. **No motion planner.** CEM over the 6 arm joints, scored by FK read back from the sim, with
   an explicit finger-separation-axis term (`cost = pos + 0.25 * (1 - |sep·Ŷ|)`). Report
   achieved TCP error every time.
2. **Cartesian waypoint chains, not joint interpolation between IK solutions** — interpolating
   in joint space does not keep the TCP straight.
3. **Finish every CEM search before closing the gripper.** CEM uses
   `write_joint_state_to_sim`, which teleports the arm and re-opens the fingers hundreds of
   times; searching for a lift path after closing silently drops the object. eva_rl's handoff
   says this one cost the most time of anything in that effort.
4. **One TCP constant:** `TCP_OFFSET = (-0.0419, 0, 0)`, named, with provenance in a comment.
5. **Executed-state checks at every phase boundary** — after align, check measured finger-body
   y against the gap centres and **abort before inserting** if off by more than half the
   window; after close, check `gap`; after every push, check all four `up_z`.
6. **Retry by identity, not list position.**
7. **Batched, 64–512 envs.**
8. **Separate RNG stream per subsystem.**
9. **A topple is a hard episode end** — unlike a dropped object there is no recovery. So
   `train_mask = 0` over the whole failing attempt, and a toppled episode is not written.
10. **Record 42-D obs**, HDF5 schema exactly as `05_PORTING_MAP.md §2`.

### The outcome vocabulary — decide before writing a single demo

`dataset.py`'s pool filters key off pick-place's `{grasped, missed, lost, delivered, recovery,
misexec}`. Ours must be designed first because it is baked into every file we produce.
Proposed: `{aligned, threaded, wedged, toppled, extracted, placed, dropped, recovery}`.

### Beliefs

6. Expert nominal success will land **70–90 %**. eva_bc's best expert version hit 96.9 % on a
   task with no hard bystander constraint; the topple termination should cost real points.
7. **`Tight-v0` (6 mm gaps) will be near-unsolvable** for strategies A/B/C, and roughly as
   solvable as nominal for D. This is a prediction that separates the strategies cleanly.

### Gate 1

> **≥85 % on nominal `Rebot-ClutterExtract-v0` over ≥128 episodes**, reported **stratified by
> the measured per-episode minimum free gap** (the 7.0–18.8 mm distribution), plus a measured
> `Tight-v0` number with no target attached.

**Calibration for the 85 %.** eva_bc's production expert hit 85.7 % and its BC champion
reached 64.1 % — a teacher-to-student ratio of **0.75**. Their task was two sequential objects
(per-object ≈80 %); ours is single-object, so the ratio should be kinder. But the ceiling is
real and was measured there: a ≤77 %-reliable teacher supervising the policy's hardest states
put a hard cap well below their gate.

**Decision rule:** expert ≥85 % → proceed. **70–85 % → proceed but record the ceiling risk
explicitly in the Stage-2 pre-registration**, with the expected BC number stated in advance so
we cannot later mistake a ceiling for a training failure. **<70 % → return to Stage 0 and
reconsider the strategy** before spending a demo-generation chain.

---

## Stage 2 — DEMO GENERATION + FLOW BC

### Design

- Port `act/` per `05_PORTING_MAP.md`. Mechanical parts first, then the three that need
  thought: the pool filters, `task_features`, and the eval success/flush logic.
- **Drop the `residual_core.py:159` quaternion permutation** (`05_PORTING_MAP.md` L2).
- Generate **~500 successful episodes across ≥8 seeds** (eva_bc's 504 → 292 clean pool
  produced their 64.1 % champion). Freeze the runner during the chain — each seed reloads the
  module, so a mid-chain edit forks later seeds' behaviour.
- Train flow BC: **chunk 50, execute 15, 10 Euler steps, 100k steps, batch 64, lr 1e-4**,
  temporal ensembling off. eva_bc: ~35–40 min at ~45 steps/s, 1.0 GB VRAM, 703k params.
- **≥3 training seeds.** Champion selected on a **held-out spawn seed**, confirmed on pooled
  ≥128 episodes.
- Set `episode_length_s = 13.8` for eval (690 = 46 × 15; see HANDOFF F9/D4) and report both
  13.8 and 14.0 once to confirm the change is immaterial.

### Beliefs

8. **Training-seed variance will be large here too** — eva_bc measured **26.6 pts** on an
   unseeded recipe. Prediction: ≥10 pts of spread across 3 seeds. This is why single-run
   comparisons are void.
9. BC will land at **0.65–0.80 × the expert rate**, i.e. 55–70 % if the expert is at 85 %.
10. **The dominant BC failure mode will be alignment-shaped**, as it was for eva_bc (74 % of
    their residue was grasp-alignment), and here it will show up as *toppling a distractor
    during approach* rather than closing on air.

### Gate 2

> **≥60 % pooled** over ≥128 held-out episodes, champion selected on a held-out spawn seed,
> with ≥3 training seeds and the full spread reported.

Plus a **mandatory failure taxonomy**: every failure classified into
`{never-approached, toppled-during-approach, toppled-during-close, toppled-during-extract,
failed-to-grasp, dropped-in-transit, missed-goal, timeout}`. eva_bc's experience is that
aggregate success rate hides real change — two of their policies matched to the decimal while
34 of 64 episodes flipped.

---

## Stage 3 — DAGGER (conditional)

**Run only if** the Stage-2 taxonomy shows covariate-shift-shaped failures — i.e. the policy
reaching states the expert never visits and then failing recoverably.

**A structural caveat that does not apply to pick-place:** a topple is terminal. The DAgger
gate can fire on `any_distractor_toppled` only to **stop**, never to teach, because there is
no recovery from it. So DAgger here can only address *pre-topple* drift and grasp failures —
a strictly narrower target than eva_bc had.

**Gate 3:** improvement beyond the measured seed spread, **≥3 seeds per arm**. eva_bc's
honest verdict on their own DAgger round was *"no measurable interference, weak evidence of
stabilization"* — and their original confident claims in both directions were later retracted
as single-seed artifacts.

---

## Stage 4 — RL ON THE FROZEN BASE

### Primary route: x0-steering (the eva_bc mandate)

Chunk-level RL: one action per 15-step window, `x0 = α_x0 · tanh(z)` with `z ∈ R⁷` broadcast
across chunk positions, `α_x0 = 1.0`, controller **free-running** (this is what made eva_bc's
gate 1 bit-exact, and it supersedes their HANDOFF's synchronous-window sketch).

**Mandatory config:** `clip_actions: 1.0`, `mu_init` const 0.0, `sigma_init` **chosen from
gate S0, never from precedent**, `horizon_length: 24` windows.

**Gates, in order — do not reorder:**

- **S0** — z-response diagnostic before training: fixed-z conditions (z=0; bias U(±0.125);
  σ 0.15/0.3/0.6) on a 64-episode suite. Pick `sigma_init` such that exploration-level success
  stays within ~10 pts of the z=0 base. **If even σ=0.15 collapses the base, stop and rethink.**
- **Gate 1** — z = 0 through the wrapper reproduces the x0-zeros base **bit-exactly**, both
  suites. **Any flip is a wrapper bug. Stop.**
- **Gate 2** — training health: per-term episodic reward at base level from the first epoch
  *after episode completion* (with 690-step episodes and 24-window epochs, that is epoch ~5 —
  earlier zeros are the logger's cadence, not pathology).
- **Gate 3** — pooled 128-episode eval vs the deterministic base, **+5-pt adoption rule**,
  plus a mandatory taxonomy diff (did the targeted failure bucket collapse *without* symmetric
  new breakage?) and a state-dependence check on z.

### The hedge: eva_rl's single-step path

eva_rl ships `train_bc.py` → `bc_to_rlgames.py` → PPO fine-tune. Its BC net already mirrors the
challenge rl_games actor exactly (`[256,128,64]` ELU), and `bc_to_rlgames.py` is fully generic
given a template checkpoint — **which we need to mint anyway** by running a few epochs of
`train.py` on clutter.

Run it as a **cheap baseline and de-risking arm**, not a replacement. Rationale: eva_bc's
x0-steering is unvalidated even on its home task (HANDOFF F6), and PPO from a BC init directly
optimises the sparse success-plus-constraint objective that this task actually poses.

**Honest caveat:** a single-step MLP forfeits chunk commitment, which eva_bc measured to be
load-bearing (59.4 → 0 % as the horizon shrank). But that experiment shortened the horizon of a
*chunk-trained* policy; a policy *trained* single-step is a different object, and this is the
standard approach. The comparison is worth having precisely because the two arms disagree
about something real.

### Beliefs

11. x0-steering will produce **a smaller gain than eva_bc's literature-based hope** (RFS
    reports 43 % → 86 % for additive vs steering). Prediction: **+5 to +15 pts** over the
    deterministic BC base.
12. **The topple constraint will make RL exploration expensive.** Prediction: a large fraction
    of early transitions end in termination, and the mid-window reset desync (HANDOFF D3) will
    be measurable rather than negligible in the first ~20 epochs.

### Gate 4 — the mission

> **≥70 % pooled over ≥128 held-out episodes on `Rebot-ClutterExtract-v0` random starts.**

Report `Tight-v0` alongside, without a target attached.

---

## Decision rules, collected

| # | trigger | action |
|---|---|---|
| DR1 | no strategy clears Gate 0 | **stop; report to Big Will with evidence.** Do not weaken the env or proceed on hope |
| DR2 | expert 70–85 % | proceed, recording the predicted BC ceiling **in advance** |
| DR3 | expert < 70 % | return to Stage 0, reconsider the strategy before spending a demo chain |
| DR4 | BC seed spread > 15 pts | add seeds before drawing any conclusion; do not compare arms |
| DR5 | Stage-2 taxonomy shows no covariate-shift failures | **skip Stage 3**, go straight to RL |
| DR6 | gate S0 shows even σ=0.15 collapses the base | stop; the base is too noise-fragile for steering — go to the hedge |
| DR7 | x0-steering < +5 pts pooled after the full budget | try **one** richer parameterization (constant + linear ramp, 14-D) before any PPO tuning; if still flat, close steering and report both arms |
| DR8 | strategy D is the only one that works | surface it to Big Will as a **benchmark finding**, keep a spirit-compliant strategy in parallel development |

## Scorecard

*To be filled in as results land. Beliefs 1–12 get scored honestly, right or wrong.*

| belief | prediction | outcome |
|---|---|---|
| 1 | push topples rather than slides at all reachable heights | — |
| 2 | pair-capture 40–70 % nominal, ~0 % Tight | — |
| 3 | target-topple works and is most reliable | — |
| 4 | FK-CEM achieves ≤2 mm TCP error in the reliable band | — |
| 5 | 512–1024 envs is the working range on 10 GiB | — |
| 6 | expert 70–90 % nominal | — |
| 7 | Tight near-unsolvable for A/B/C | — |
| 8 | BC seed spread ≥10 pts | — |
| 9 | BC = 0.65–0.80 × expert | — |
| 10 | dominant BC failure is topple-during-approach | — |
| 11 | steering gives +5 to +15 pts | — |
| 12 | topple constraint makes early RL exploration expensive | — |
</content>
