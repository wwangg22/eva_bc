# Postmortem: Why the BC / DAgger Pipeline Plateaued at 59.4%

*2026-08-01/02. Written after the Stage 2–3 plateau, before the residual-RL pivot decision.
Companion docs: PLAN.md (design), JOURNAL.md (full experiment ladder), HANDOFF.md,
**experiments/EXP_INDEX.md** (the 2026-08-02 ladder that tested this document's claims).*

---

## 1. TL;DR

- The flow-matching BC policy plateaued at **59.4%** (Gate 2 target: 85%). DAgger round 1
  moved episode-level outcomes substantially (17 fixed, 17 broken) but netted **zero**.
- "BC is a data problem" turned out to be only one-third of the story. We measured three
  distinct failure classes, and only the first is fixable with more/better data:
  1. **Data-distribution gaps** (covariate shift) — the classic story. DAgger's 17 fixed
     episodes prove this part responds to treatment.
  2. **State aliasing** — at the single-timestep level, *"just grasped the can, hold and
     lift"* and *"closed on air, reopen and retry"* are nearly the same observation with
     **opposite correct actions**. More data of both kinds makes this **worse**, not better.
  3. **Interference under fixed capacity** — adding 100 recovery episodes measurably
     shifted the policy's behavior on *nominal* states it already handled correctly
     (empirical numbers in §5).
- Separately, the DAgger teacher itself has a measured ceiling (68% takeover success,
  failures concentrated exactly on the states the policy visits most), which caps what any
  number of DAgger rounds can teach.
- The failure profile that remains is overwhelmingly **precision** (mm-scale grasp
  alignment, delivery/release) — the class of error that residual RL on a frozen base is
  best documented at fixing, and that BC is structurally worst at.

> **UPDATE (2026-08-02, experiment ladder — status of the TL;DR after testing):**
> - **Failure class 1 (covariate shift): the "17 fixed" evidence is void.** Seeded
>   replicas (EXP03) showed same-data different-seed training runs flip 31–39 episodes;
>   v1↔v3 flip 34. The per-episode DAgger story is indistinguishable from retraining
>   noise. DAgger is not shown to hurt either — arm means 56.7% (+dagger, n=3) vs 47.4%
>   (nominal, n=3) with tighter spread — weak evidence it *stabilizes* training.
> - **Failure class 2 (aliasing): confirmed but sharpened** (EXP01). The information is
>   PRESENT in a single frame and lives in the finger channels: fingers+last-grip
>   (5 dims, variant D) rejects 100% of the policy's 665 real closed-on-air freeze
>   states at AUC 0.968; physical finger joints alone (4 dims, variant G) have the best
>   AUC (0.976) but 27.1% FPR on those states — the commanded-grip channel is needed
>   for perfect transfer. The policy's failure is *salience/usage*, not missing input.
>   A grasp-success obs bit is the justified fix; history is refuted (≤+0.01 AUC,
>   transfers worse).
> - **Failure class 3 (interference): RETRACTED** (EXP03 forensics + replicas; see §5
>   correction). The measured mechanism was an analysis artifact; nominal-state grip
>   behavior is flat (12.6–13.1% open-rate) across all 8 checkpoints.
> - **New, dominant finding the original TL;DR missed: training-seed variance is
>   enormous** — the nominal recipe alone spans **32.8–59.4%** on seed. Every single-run
>   A/B in this document is inside that noise. Standing rule: ≥3 seeds per config,
>   champion selected on held-out spawn seeds, pooled ≥128-ep numbers only.
> - Also tested: execute-15 does NOT cause the precision failures — shortening the
>   horizon collapses the policy monotonically (59.4→32.8→3.1→0→0% at n=15/8/4/2/1;
>   EXP02). Chunk commitment is load-bearing.
> - **Champion after variance-aware reselection: `runs/exp03_N3/ckpt_final.pt` at
>   64.1% pooled (128 eps)** — the frozen base for residual RL. v1's 59.4% was one
>   lucky-seed, one-suite reading.

---

## 2. The architecture we used

**Flow-matching chunk policy** (`act/modeling_flow.py`, pivot decision 2026-08-01):

- **Backbone:** the vendored LeRobot ACT transformer (encoder + decoder), CVAE/KL deleted.
- **Conditioning:** single-timestep state, 41-D privileged obs split into
  `observation.state` (16-D joint pos/vel) + `observation.environment_state` (25-D: two
  cans pos+quat XYZW, basket xy, last action). **No history, no images.**
- **Flow head:** decoder queries = projected noisy action chunk x_tau + sinusoidal
  positions; scalar flow-time tau embedded and added to all tokens; rectified-flow loss
  v = x1 − x0 (elementwise MSE × the `action_is_pad`/censor mask); 10 Euler steps at
  inference; seedable generator (bit-deterministic given x0).
- **Chunking:** chunk 50, execute 15 (`n_action_steps`), ensembling OFF, per-env queues.
- **Action space:** 7-D — 6 arm-joint position targets (scaled (dq − q_default)/0.5) + 1
  binary grip channel (±1 → BinaryJointPositionAction, open=0.045/closed=0.0 per finger).
- **Normalization:** external mean/std (controller normalizes obs, unnormalizes actions).
- Training: 100k steps, lr 1e-4, batch 64, ≈35 min on this GPU. Eval: deterministic per
  spawn seed (v1-vs-v1 churn = 0 across 64 episodes).

Training pools: `nominal` = 292 first-attempt-clean expert episodes (8 seeds);
`nominal+dagger` adds successful `recovery_dagger` episodes with
`train_mask[:takeover_t]=0` (policy steps never supervised; teacher's own failed
sub-attempts masked; failed recoveries excluded).

---

## 3. The results ladder (all 64-ep, seed 42, deterministic)

| Policy | Pool | Result | Verdict |
|---|---|---|---|
| flow_nominal_v1 | nominal only | **59.4%** | champion |
| flow_nomrec_v2 | + offline scripted-recovery | **51.6%** | offline recovery data actively HURTS |
| flow_dagger_v3 | + 100 on-policy DAgger takeovers | **59.4%** | aggregate flat; 17 fixed / 17 broken / 9 both-fail |

> **CORRECTION (2026-08-02, experiments/EXP03_dagger_interference.md):** seeded replicas
> (3 per arm, identical data/recipe, training seed varied) showed the nominal recipe
> alone spans **32.8–59.4%** and same-data different-seed pairs flip 31–39 episodes —
> while v1↔v3 flip exactly 34. Every single-run delta in this table is inside training
> noise: v1's "champion" status is a lucky seed, v2's "actively HURTS" verdict is not
> supported (51.6% is well inside the nominal spread), and v3's "17 fixed / 17 broken"
> is the replica noise floor, not a treatment effect. Replica means: nominal 47.4%
> (n = 3), +dagger 56.7% (n = 3) — weak evidence recovery data *stabilizes* training,
> none that it hurts. `train_flow.py` was unseeded when v1–v3 were trained.

Failure taxonomy, exact per-episode classification (threshold: can counted "lifted" if
its max height exceeded 0.06 m; carry height ≈0.08–0.10 m; table rest ≈0.02 m):

| Bucket | v1 (26 fails) | v3 (26 fails) | meaning |
|---|---|---|---|
| placed-one-stuck, 2nd can NEVER lifted | 11 | 9 | first can delivered, then closed-on-air on can 2 and froze (max z ≈ 0.020–0.047) |
| placed-one-stuck, 2nd can lifted to carry height | 5 | 7 | can 2 reached z = 0.079–0.099 but was never landed in the basket |
| never lifted anything | 8 | 5 | both cans stayed at table height (max z ≤ 0.025) — missed from the start |
| lifted but placed nothing | 2 | 5 | ≥1 can lifted, zero delivered — v3's growth here is the mid-carry-open signature |
| dropped after successful place | 0 | 0 | `placed_max == placed_final` in every episode, both versions |

(v2's checkpoint was deleted in cleanup; from the JOURNAL record its anatomy *worsened
upstream*: nothing-placed episodes 9 → 20 at 51.6% — offline recovery data degraded even
first-can grasping.)

v1→v3 same-seed episode transitions (same 64 spawns, deterministic eval): **17 fixed / 17
broken / 9 fail-in-both / 21 succeed-in-both**. What v3 fixed is exactly what DAgger
supervised: 11 of the 17 fixes were v1 "placed-one-stuck" episodes, 5 were "never lifted
anything", 1 was "lifted-never-placed" — i.e. miss-then-frozen states got the
reopen-retry skill. What v3 broke is downstream: of the 17 newly broken episodes, 4
became "lifted but placed nothing" and 9 became "placed-one-stuck" — episodes v1
completed cleanly now lose cans after grasping, consistent with the measured grip-channel
interference (§5). On-policy gate data (146 DAgger takeovers): **91% miss** (closed on
air, froze ≥45 steps), 9% stall, 0 drops.

---

## 4. "BC is a data problem" — where that's true and where it broke

The slogan is right about covariate shift: a BC policy drifts into states the expert never
visited, and expert labels on *policy-visited* states fix it. Our v2-vs-v3 A/B is a clean
demonstration: offline scripted recovery (states the *expert* visits when perturbed)
**hurt** (51.6%), while on-policy DAgger recovery (states the *policy* actually gets stuck
in) genuinely fixed 17 episodes. Distribution match matters exactly as the theory says.

But two additional problems are **not** data-volume problems, and we hit both:

### 4a. State aliasing (a partial-observability problem wearing a data costume)

At the moment the gripper has just closed, the observation barely distinguishes:

- **closed on the can, about to lift** → correct action: *hold closed, lift*;
- **closed on air after a miss** → correct action: *reopen, retreat, retry*.

The distinguishing physical fact — is the can between the fingers — appears in the obs
only as a ±12 mm difference in finger-joint position (closed-on-can stalls at ~0.020–0.040
aperture, closed-on-air reaches ~0.000), a subtle feature the function approximator must
find under normalization, versus large, salient-but-irrelevant features (arm pose, can
positions) that are *identical* in both cases. Nominal-only data contains almost no
"reopen after closing on air" evidence (LITERALLY the documented failure mode in
[Disambiguate Gripper State](https://arxiv.org/pdf/2503.23835): *"almost no training data
where the robot opens the gripper after closing it on empty air — the robot gets
stuck"*) — hence v1's miss→freeze mode. Adding DAgger reopen data doesn't sharpen the
boundary; it puts **opposite supervision on nearly-identical inputs**, and the smooth
policy bleeds reopen-propensity into legitimate hold states (measured below). More data of
both classes concentrates the conflict; only a *representation* change (history/memory,
a grasp-success feature, tactile proxy) or a *reward* signal resolves it.

> **UPDATE (2026-08-02, experiments/EXP01_grasp_aliasing_probe.md):** tested directly.
> The states are NOT aliased at the information level: a probe on the same 41-D obs
> separates grasped-vs-missed at AUC 0.954 single-frame, and the signal lives in the
> finger channels — physical finger joints alone (4 dims) reach AUC 0.976, and
> **fingers + last commanded grip (5 dims, variant D) reject 100% of the policy's 665
> on-policy closed-on-air freeze states** (AUC 0.968). The full-41-D probe, with those
> same dims available, mislabels 53.5% of them — the distracted-by-41-dims failure,
> reproduced in a 34k-param probe. So this is a **salience/feature-usage failure**, not
> partial observability. Caveat that matters for the fix: physical joints WITHOUT the
> commanded-grip channel still mislabel 27.1% of freeze states — the disambiguating
> pair is "commanded closed" + resulting aperture, so any grasp-success bit must
> include the last grip command, not static aperture alone.
> History is refuted as the fix (≤+0.01 AUC; finger-history transfers *worse*, 40.8%
> FPR). The cheap justified fix is an explicit grasp-success obs bit computed from
> finger channels (the pseudo-tactile idea, validated on our own data). Note also the
> "bleeds reopen-propensity into legitimate hold states (measured below)" clause is
> retracted with §5 — the salience problem is real, the interference problem was not.

### 4b. Interference under fixed capacity

v3 = v1's exact dataset + ~25% more samples, same architecture, same 100k steps. BC
training is not "add knowledge monotonically"; it re-fits one smooth function to a
changed target distribution. We measured what changed (see §5): the recovery pool shifted
grip behavior on nominal hold states by an amount that converts directly into the observed
new failure bucket (mid-carry releases). The single-seed round-1 collection made this
worse by correlating all recovery data with one object scale — a spurious feature the
network could key on. This is the classic interference/forgetting tradeoff documented in
the DAgger literature (aggregation mitigates forgetting but mixed pools trade nominal
performance for recovery skill — see [Dataset Aggregation](https://www.emergentmind.com/topics/dataset-aggregation-dagger)
and Compliant Residual DAgger, which exists precisely because residual heads avoid
touching the base).

> **RETRACTED (2026-08-02):** the §5 measurements this section rests on did not survive
> re-analysis (see the §5 correction), and EXP03's 3-vs-3 seeded replicas show no
> interference signal: grip open-rate on true hold states is flat (12.6–13.1%) across
> nominal and +dagger checkpoints, dagger replicas do not have elevated mid-carry
> losses, and v3's "new failure bucket" is inside per-seed churn. The interference
> *literature* is real; our claimed measurement of it here was not. The single-seed
> scale-correlation concern about round-1 collection remains a reasonable hygiene point
> for future DAgger rounds, but nothing in our data demonstrates harm from it.

### 4c. The teacher ceiling (a data-quality problem money can't fix)

DAgger can at best teach what the expert can do from the policy's stuck states. Measured:
68% takeover success in round 1; failed recoveries skew heavily to lying cans (LL config =
54% of failures vs 18% of successes). We then spent an evening root-causing the expert's
lying weakness (§7) — it is partly *physical* (grasp-table rows that plan validly but
execute pads 3–5 cm high against joint limits). The expert's perturbed-recovery suite
stands at 77.4% measured. A ≤77%-reliable teacher supervising the policy's hardest 40% of
states puts a hard ceiling well below Gate 2's 85%.

---

## 5. The measured mechanisms (evidence)

**Per-phase action divergence, v1 vs v3, on identical nominal states** (same x0 seed;
action units; grip is ±1):

| phase | arm mean |Δ| | grip mean |Δ| |
|---|---|---|
| settle | 0.041 | 0.005 |
| approach | 0.054 | 0.009 |
| descend | 0.058 | 0.029 |
| close | 0.068 | 0.066 |
| **lift** | 0.067 | **0.357** |
| transport | 0.066 | 0.052 |
| reopen | 0.140 | **0.901** |
| retreat | 0.037 | 0.006 |

Reading: arm divergence ≈0.05 units ≈ 1.4° joint target ≈ **~1 cm at the fingertip** —
material for a task where we measured 2–3 mm deciding catch-vs-air. The grip channel is
the smoking gun: huge divergence exactly at lift (hold states aliased with reopen states)
and at reopen (the behavior v3 learned — that part is *working as intended*).

**The kill shot** — on 1,466 nominal lift/transport states where the expert grip is
−1.00 (closed) throughout:

- v1 commands an open somewhere in its executed 15-step horizon on **12.1%** of states
  (partly legitimate: samples near basket release);
- v3 does on **20.3%**;
- on **8.7%** of states v1 holds closed while v3 opens.

A spurious open mid-carry = the can falls = "lifted but never placed" — precisely the
bucket that grew 2→5 in v3's eval, alongside 17 broken episodes against a **zero**-churn
noise floor. Mechanism measured end-to-end: DAgger reopen supervision → grip bleed on
aliased hold states → mid-carry releases → flat aggregate despite 17 real fixes.

> **CORRECTION (2026-08-02, experiments/EXP03_dagger_interference.md forensics):** the
> two measurements above do NOT survive re-analysis. The original analysis (recovered
> from the transcript and reproduced exactly) had two defects: v3's obs were normalized
> with v1's stats, and "hold states" included lift frames of *missed* attempts — empty
> closed-gripper states where opening is v3's **learned recovery working**, not damage.
> Corrected (all 8 demo files, grasped-only holds, per-policy stats): v1 ≡ v3 on true
> hold states (13.6% vs 13.7% open-in-horizon, 0.7% divergent), and on empty-lift states
> v1 opens 5.1% (the freeze mode) vs v3 67.1% (correct conditional reopen). Offline,
> v3's grip channel is *better* than v1's. Consequently §4b's interference mechanism and
> this section's "kill shot" are **retracted**; v3's 17 broken episodes were resolved by
> EXP03's seeded replicas as **training-seed noise** (same-data different-seed pairs flip
> 31–39 episodes; v1↔v3 flips 34 — `train_flow.py` had no RNG seeding when v1/v3 were
> trained). The per-phase
> divergence table above shares the same defects (lift/reopen rows are dominated by
> aliased missed-attempt frames) and should not be read as interference evidence.

---

## 6. Flow matching: known limitations vs what we hit

What the literature says, and whether we observed it:

| Known limitation | Source | Did it bite us? |
|---|---|---|
| Multimodal supervision → averaged/ambiguous actions or unstable mode selection | [VFP](https://arxiv.org/html/2508.01622), [Flow-Guided Policies](https://openaccess.thecvf.com/content/ICCV2025W/ACVR/papers/Jung_Flow-Guided_Policies_Overcoming_Diffusion_Limitations_for_Robust_Robot_Imitation_Learning_ICCVW_2025_paper.pdf) | **YES** — the aliased hold/reopen states are exactly opposite-mode supervision on one input; the measured 0.36 grip shift is mode mass moving, and a ±1 binary channel has no safe in-between. |
| No gain over simple regression on unimodal, state-based regimes | Much Ado About Noising (ICLR 2026; pre-pivot research) | **YES, as predicted** — we adopted FM for future-proofing, not for gains here; 59.4% is a backbone/data ceiling, not an FM ceiling. |
| Frame-only conditioning cannot resolve history-dependent intent (aliasing) | [IntentVLA](https://arxiv.org/html/2605.14712v2), [DSSP full-history conditioning](https://arxiv.org/html/2605.14598) | **YES** — our single-frame conditioning is the enabling condition for §4a. This is an *input* limitation, not an FM-head limitation. |
| Gripper-state obs unreliable for grasp success; no reopen-after-air-close data → stuck | [Pseudo-Tactile feedback](https://arxiv.org/pdf/2503.23835) | **YES, verbatim** — our dominant on-policy failure (91% of takeovers) is this paper's titular failure mode. |
| Open-loop within chunk → stale reactions | chunking literature generally | Minor — flush triggers existed and fired rarely (6/64 eps); not a dominant mode. |
| Inference stochasticity (x0 sampling) → run-to-run flakiness | general FM/diffusion | **NO** — seeded generator made eval bit-deterministic (v1-v1 churn 0). Worth keeping: this also enables the frozen-base determinism residual RL wants. |

> **UPDATE (2026-08-02):** three rows of this table need revision against the ladder.
> Row 1 (multimodal supervision): the "measured 0.36 grip shift" evidence is retracted
> with §5; the opposite-supervision *setup* is real but we no longer have a measured
> behavioral consequence. Row 3 (frame-only conditioning enables aliasing): weakened —
> EXP01 shows a single frame *contains* the disambiguating signal (finger-only AUC
> 0.976); the enabling condition is feature salience under conflicting supervision, not
> the frame-only input. Row 5 (open-loop within chunk): revised from "minor" to
> **backwards-protective** — EXP02 shows chunk commitment is load-bearing (success
> collapses monotonically to 0% as n_action_steps shrinks 15→1); the open-loop window
> is not a cost we were paying, it is why the policy works.
> A row this table should have had: **training-run variance** — unseeded training spans
> 32.8–59.4% on this recipe (EXP03), which dwarfs every effect the table discusses.

What we did NOT hit (worth recording): no training instability, no mode collapse on
nominal behavior, no inference-latency problem (10 Euler steps, negligible), and the
censoring/mask machinery (`action_is_pad` excluded from decoder self-attention) was
verified bit-exact. The FM head itself was **not** the bottleneck at 59.4%.

---

## 7. Limitations we discovered ourselves (not in any paper we found)

1. **Planner-valid ≠ executable.** cuRobo happily returns collision-free IK/plans whose
   end poses the PD-controlled arm cannot physically track: on low lying-can grasps,
   specific grasp-table row families execute finger pads 3–5 cm above target (clamped by
   joint limits/contact) and *do not respond to corrective joint nudges* (12 mm commanded
   → 1–4 mm actual). Executed-height across rows spreads ~16 mm for one planned target.
   Any pipeline that treats "plan succeeded" as "grasp will happen" inherits silent
   failures. Detection needs an *executed-state* check (measured finger height vs can).
2. **Same-seed A/B pairing silently breaks.** Perturb scheduling and candidate dropout
   share one RNG stream; the first behavioral divergence (one extra attempt) reshuffles
   every later episode's perturbations (12/31 episodes drew different events). Paired
   comparisons are valid only up to the first divergence. Fix: per-subsystem RNG streams;
   until then, per-attempt metrics are the primary signal and n=32 suite deltas under
   ±8 pts are noise.
3. **Know WHEN an instrumentation line executes.** Our grasp-geometry log line ran
   *post-lift*, not at close; calibrating a close-time constant from it produced an 80 mm
   error and a wasted GPU run. Trajectory-phase context is part of a measurement's units.
4. **Retry loops must exclude by identity, not list position.** Excluding goalset
   *positions* re-served the exact failed table row on 4/10 retry sequences (rows
   4550→4550, 3113→3113): three identical failures that looked like three attempts.
5. **Offline recovery data ≠ on-policy recovery data** (v2's −7.8 pts). Known in
   principle (that's DAgger's thesis), but the magnitude of the *harm* — not just absence
   of benefit — was striking, and worth remembering: mismatched recovery data is worse
   than none.
   *(**2026-08-02: void as evidence.** 51.6% sits comfortably inside the nominal arm's
   32.8–59.4% seed spread — v2 was a single unseeded run. The offline-vs-on-policy
   thesis may still be true; our data no longer demonstrates it. Re-test only with ≥3
   seeds per arm.)*
6. **Aggregate success rate hides real change.** v3 = v1 to the decimal while 34 episodes
   flipped. Per-episode diffing against a measured churn floor (ours is zero, thanks to
   determinism) should be the default reading of any A/B.
   *(**2026-08-02: half-corrected.** The zero churn floor is for *inference* (re-running
   one checkpoint). The relevant floor for comparing two **training runs** is 31–39
   flipped episodes (EXP03 same-data seed pairs) — and v1↔v3's 34 sits exactly on it.
   The lesson inverts: per-episode diffing between separately-trained policies is
   nearly meaningless at n=64 without seed replicas to calibrate the floor.)*
7. **Training-seed variance is the largest effect in the whole pipeline** (found
   2026-08-02, EXP03). Same data, same recipe, different seed: 32.8% vs 59.4%. It is
   bigger than every treatment effect we studied, and `train_flow.py` was entirely
   unseeded until the ladder. Nothing in the BC literature we read prepared us for
   26.6 pts of seed noise at this dataset size; assume it until measured otherwise, and
   never conclude from single training runs. (Open question: can EMA weights, longer
   training, or a larger pool shrink it?)
8. **Offline behavior probes on expert states cannot rank policies** (EXP03): N1
   (32.8%) and N3 (59.4%) are indistinguishable in offline grip open-rate on expert
   hold states. What separates seeds is closed-loop error compounding, visible only in
   rollouts. Corollary: any offline probe must use each policy's *own* normalizer stats
   and outcome-filtered states — violating either produced §5's retracted "kill shot".

---

## 8. Implications for the path forward

The residual-RL stage (PLAN §6, frozen FM base + α·tanh arm-joint residual, PPO) is
well-matched to what's actually left:

- **~90% of remaining failures are precision-shaped** (grasp alignment; delivery), the
  documented sweet spot of ResiP-style residuals, and reward supervision has no teacher
  ceiling and no interference with the frozen base (the base cannot be dragged).
- **Aliasing interacts correctly with a residual:** the residual sees obs + base action +
  chunk phase; reward will teach it to *prevent* the miss (alignment), which removes the
  stuck states rather than trying to disambiguate them.
- **Two design questions to settle with data:**
  (a) whether the grip channel must join the residual action space (pending the failure
  close-up video review: steering-vs-release-timing on the delivery failures) — a
  deliberate PLAN §6 deviation if yes;
  (b) whether to add cheap obs surgery first — a grasp-success bit (finger aperture +
  stall detection, i.e. the pseudo-tactile idea) and/or 2–4 frames of history — which
  attacks §4a at the representation level and would benefit both the base and the
  residual.

> **UPDATE (2026-08-02, ladder verdicts — what this section becomes; full plan in
> HANDOFF.md §4):**
> - **Question (b) is answered:** grasp-success bit YES (EXP01: fingers+last-grip
>   probe = 0% FPR on the policy's 665 real freeze states at AUC 0.968; physical
>   joints alone hit AUC 0.976 but 27.1% FPR — the bit must be computed from finger
>   pos/vel PLUS the last grip command), history NO (≤+0.01 AUC, transfers worse). The
>   bit goes into the residual's inputs regardless; whether to also retrain the BC
>   base with it (≥3 seeds + ≥3 control seeds, verify the pooled champion beats 64.1%
>   beyond arm ranges) is the open fork for Big Will.
> - **Question (a) stays open** but starts per ResiP precedent: gripper EXCLUDED from
>   the residual initially, with the with-gripper ablation planned. The close-up video
>   was never rendered; EXP01/EXP03 data answered what it was for.
> - **The frozen base is now `runs/exp03_N3/ckpt_final.pt` (64.1% pooled, 128 eps)**,
>   selected on a held-out spawn suite — not v1. Full pooled anatomy of its 46
>   failures (taxonomy.py, both suites, exhaustive): **34 grasp-phase miss/freeze**
>   (18 never-lifted-anything + 16 placed-can-1-then-closed-on-air-on-can-2) + **12
>   carry/release** (9 lifted-never-placed + 3 stuck-at-carry-height); **0**
>   drops-after-place. 74% of the residue is grasp-alignment shaped — squarely the
>   arm-only residual's target; the 26% carry/release tail is the with-gripper
>   ablation's motivation.
> - **EXP02 adds a design constraint this section didn't know:** commitment is
>   load-bearing, so the residual must ride ON committed chunks (per-step additive
>   correction, ResiP-style) — never shorten the horizon. RFS-style x0-steering is the
>   documented fallback (plain residuals on flow bases: 43% vs RFS 86% in the lit).
> - **The BC-revisit ranked fixes below are superseded:** recovery down-weighting lost
>   its premise (no interference), history is refuted. The revised ranking: (1)
>   grasp-bit obs surgery, (2) multi-seed training + held-out selection as standard
>   practice, (3) variance-shrink candidates (EMA, longer training, bigger pool),
>   (4) teacher lying-recovery ceiling — unchanged and still real (68%/77.4%).

- **If BC is ever revisited:** 3+ seed DAgger collection (kill the scale correlation),
  recovery-pool loss down-weighting, and history conditioning are the ranked fixes; and
  the teacher's lying-recovery ceiling must be raised first or those states filtered from
  supervision.

> **UPDATE (2026-08-02 evening, EXP06+EXP07 verdicts — this section is now RESOLVED):**
> - **Plain additive residual (EXP06): CLOSED, exactly flat** — 55.5%→55.5% pooled,
>   symmetric 26-fixed/26-broken causal churn, learned residual state-independent
>   (~0.0084 |res| everywhere). The "precision-shaped residue" premise above was
>   wrong for off-manifold additive corrections on this flow base.
> - **x0-steering (EXP07): SUCCESS — 55.5% → 91.4% pooled (89.1/93.8), Gate 6 (90%)
>   cleared** with the first pre-registered config: z ∈ R^7 broadcast per 15-step
>   window, x0 = tanh(z), free-running controller, window-aligned eval-protocol
>   training, bare placed-stream reward, 200 epochs @2048 envs. Taxonomy: 51
>   fixed / 5 broken; the never-lifted bucket collapsed (18/19 → success); learned
>   z is state-dependent (|z| 0.220 success vs 0.282 failure eps). The RFS 43→86
>   pattern (on-manifold steering ≫ additive residual) reproduced on our stack.
> - **Question (a) is answered by construction:** the grip channel joins via x0's
>   grip column — steering reaches release timing through the decoder without a raw
>   grip action, and carry/release failures (10/11 fixed) confirm it worked.
> - Full log + belief scorecard: `experiments/EXP07_x0_steering.md`. Champion:
>   frozen `runs/exp03_N3/ckpt_final.pt` + steering head
>   `runs/exp07_steer/s1_seed1/nn/exp07_steer.pth`.

## Sources

- [Disambiguate Gripper State in Grasp-Based Tasks: Pseudo-Tactile as Feedback](https://arxiv.org/pdf/2503.23835)
- [IntentVLA: Short-Horizon Intent Modeling for Aliased Robot Manipulation](https://arxiv.org/html/2605.14712v2)
- [DSSP: Diffusion State Space Policy with Full-History Encoding](https://arxiv.org/html/2605.14598)
- [VFP: Variational Flow-Matching Policy for Multi-Modal Robot Manipulation](https://arxiv.org/html/2508.01622)
- [Flow-Guided Policies: Overcoming Diffusion Limitations (ICCVW 2025)](https://openaccess.thecvf.com/content/ICCV2025W/ACVR/papers/Jung_Flow-Guided_Policies_Overcoming_Diffusion_Limitations_for_Robust_Robot_Imitation_Learning_ICCVW_2025_paper.pdf)
- [Dataset Aggregation (DAgger) overview incl. forgetting/interference tradeoffs](https://www.emergentmind.com/topics/dataset-aggregation-dagger)
- [Behavioral Cloning in Imitation Learning (covariate shift background)](https://www.emergentmind.com/topics/behavioral-cloning-bc)
- ResiP (arXiv 2407.16677), RFS (arXiv 2602.01789), Much Ado About Noising (ICLR 2026) — cited pre-pivot in PLAN §2.3/§6.
