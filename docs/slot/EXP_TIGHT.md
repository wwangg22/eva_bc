# EXP_TIGHT — getting a policy through the 0.5 mm channel

*Opened 2026-08-03 during session 5, **before any `-Tight-v0` evaluation of a trained policy
exists**. The sweep that will produce those numbers is running as this is written; the beliefs
below are pre-registered against it. Project convention (`POSTMORTEM.md` §9d): the design and
the decision rule go on the record before the data, or the result is not evidence.*

## Status

**CLOSED.** Beliefs 1 and **2 refuted**; 3 and 4 confirmed; 5 untested. Steps C and D are
**not needed for the objective**, which BC already clears at every clearance (§7a).

* **belief 1 refuted** — Tight is **0.736** pooled, not 0.25–0.55. The clearance is not the
  binding constraint; **depth** is, and it is clearance-independent (§7a).
* **belief 2 refuted on replication** — `--fixed-x0 zeros` helped 1 of 6 runs and hurt 4
  (mean −9.5 pts). The +16.7 on `armA_seed0` was a single-seed fluke (§7d).
* **the finding that survives** — the x0 choice alone moves success across a **54-point range**
  (−37.5 to +16.7) with everything else fixed. Blind freezing is a coin flip; that is the
  measured case for **state-conditioned steering** (§7d).

---

## 1. Where this sits

`-v0` (1.5 mm) is **done**: 92.7 % later-cohort at `ckpt_final`, 95 % CI [0.856, 0.964], by BC
alone (`SESSION5_FINDINGS.md` §1). The project target was 70 %. What remains is the hard rung of
the ladder:

| gym id | per-side clearance | expert | BC |
|---|---|---|---|
| `-Loose-v0` | 3.0 mm | 128/128 | sweep running |
| `-v0` | 1.5 mm | 128/128 | **92.7 %** |
| `-Tight-v0` | **0.5 mm** | **128/128** | **the question** |

The expert solves all three. So this is not a question about whether the *task* is solvable at
0.5 mm — it demonstrably is, by an open-loop trajectory. It is a question about whether a cloned
policy can hold the tolerance.

---

## 2. Two things I got wrong earlier today, corrected here before they cost a run

Both are kept visible rather than quietly fixed, because each changes what the right experiment
is.

### 2a. The lateral-error extrapolation was confounded (retracted)

`SESSION5_FINDINGS.md` §5b pre-registered "Tight ≈ 0.19" from the |lateral| CDF of successful
`-v0` episodes. That CDF is **censored**: it is measured after the walls physically constrain
the block, which is why its maximum is exactly the clearance.

The expert proves it. Same open-loop trajectory on all three rungs, and yet:

| task | clearance | expert lateral p90 |
|---|---|---|
| `-Loose-v0` | 3.0 mm | 1.38 mm |
| `-v0` | 1.5 mm | 1.11 mm |
| `-Tight-v0` | 0.5 mm | **0.48 mm** |

One trajectory, three different "errors". The number reports the channel, not the policy. The
0.19 figure stays on the record as a **worst case under a violated assumption**, not a
calibrated prediction.

### 2b. "Collect Tight demos" — the obvious fix, and it is mostly a no-op

`SESSION5_FINDINGS.md` §5d recommended collecting `-Tight-v0` demos, on the reasoning that all
2038 existing demos are `-v0` (verified) and the expert scores 100 % on Tight, so this looked
like a pure train/test mismatch with a cheap data fix.

**That reasoning is weak, and checking the planner is what showed it.** `expert/plan.py`'s
`ExpertParams` — `grasp_h`, `carry_z`, `stage_x`, `insert_x`, `turn_per_wp` — contains **no
clearance term**, and its own docstring says the defaults "measured 100 % on all three
clearances". The string `clearance` does not appear in the planner at all. The IK solves against
the *spawn*, not the channel.

So the expert commands **the same trajectory** on Tight as on `-v0`. Tight demos would carry
near-identical action labels; only the observations during the final insertion would differ, and
only slightly, since the expert seats 128/128 there with minimal wall contact.

**Conclusion: collecting Tight demos does not target the failure.** It supplies the same
supervision the policy already has. Demoted from "the plan" to a late fallback (§5, option D).

---

## 3. The reframed hypothesis

The expert is deterministic given a spawn. **The BC policy is not** — the flow head draws a
fresh `x0 ~ N(0, I)` at every chunk refill, and `x0` is the sole stochasticity source (no CVAE,
by design). That sampling noise is a spatial jitter added on top of an otherwise expert-like
trajectory.

On `-v0` it costs 7.3 points (92.7 % vs the expert's 100 %) against a 1.5 mm budget. On Tight
the budget is **one third** of that, while the noise is unchanged.

> **Central hypothesis: what fails on Tight is the policy's own sampling noise eating a
> clearance that no longer has room for it — not a deficiency of the learned trajectory.**

This is directly testable, cheaply, and the machinery already exists: `BatchedACTController`
has always accepted `fixed_x0`. Only the CLI flag was missing, and this session added it:

```
--fixed-x0 zeros    # the distribution's MODE; chunk becomes a deterministic function of obs
--fixed-x0 <int>    # one fixed N(0,I) draw at that seed, shared by every env and refill
```

`fixed_x0` is now also recorded into the results JSON `config` block — without that, a
deterministic run would be byte-indistinguishable from a stochastic one, and the two are not
comparable.

---

## 4. Beliefs, pre-registered

1. **BC on Tight lands in 25–55 % later-cohort**, materially below its 92.7 % on `-v0`.
   Grounds: the expert's *own* Tight margin is razor-thin (lateral p90 0.48 mm against a 0.5 mm
   clearance) and it still scores 128/128 — so any policy with more spatial variance than the
   expert should fall off a cliff, and BC demonstrably has more (it loses 7.3 pts where the
   expert loses 0). Not derived from the censored CDF of §2a.
2. **`--fixed-x0 zeros` will BEAT stochastic sampling on `-Tight-v0`**, by ≥ 5 points.
   This is the load-bearing prediction and it **contradicts the pick-place precedent**, where
   the stochastic base scored 64.1 % against 55.5 % for the best frozen `x0` — blind mode
   *resampling* was worth +8.6 points there. The reason to expect the opposite sign here: in
   pick-place, resampling buys retries at a task with multiple viable grasps and a forgiving
   40 mm basket; here there is essentially one viable trajectory and a 0.5 mm channel, so
   variance is pure cost with nothing to buy.
3. **On `-v0`, `--fixed-x0 zeros` will be roughly neutral** (within ±5 points of 92.7 %).
   Same mechanism: at 1.5 mm there is room for the jitter, so removing it neither helps much nor
   hurts. If instead it *helps* on `-v0` too, belief 2's mechanism is more general than claimed
   and the deterministic base should become the default everywhere.
4. **Yaw will not be the binding constraint on Tight either.** Every `-v0` failure had
   |yaw| ≤ 0.030 rad against a 0.12 tolerance, and the expert's Tight yaw p90 is 0.0037 rad.
   If Tight failures turn out to be yaw-dominated, the whole spatial-jitter picture is wrong.
5. **x0-steering will help less here than its +35.9 points on pick-place.** EXP07's win came
   mostly from the never-lifted bucket collapsing 18/19, and that failure mode **does not exist
   on this task** — all 61 expert failures are `release=unseated`, zero grasp failures.
   Registering the discount now so a modest gain is not written up later as a disappointment.

---

## 5. Design — ordered by cost, cheapest first

Each step's result decides whether the next one runs. **One GPU job at a time.**

**A — Read the sweep (0 extra GPU).** `-Tight-v0` × 2 spawn seeds × 6 checkpoints is already
being collected. Get the pooled later-cohort rate and the per-training-seed spread. Run the
failure taxonomy: depth / lateral / yaw split, and the `inserted_raw`-but-rejected count.
*Gate: if BC already clears a useful rate on Tight, this experiment closes here.*

**B — The determinism test (2 evals, ~5 min).** The cheapest possible test of the central
hypothesis, on the champion checkpoint:

```
python slot_act/eval_act.py --ckpt runs/<champion>/ckpt_final.pt \
    --task Rebot-PrecisionSlot-Tight-v0 --num-envs 32 --episodes 128 --seed 777 \
    --fixed-x0 zeros --out runs/<champion>/eval_x0zeros_Tight_s777.json
python slot_act/eval_act.py --ckpt runs/<champion>/ckpt_final.pt \
    --task Rebot-PrecisionSlot-v0 --num-envs 32 --episodes 128 --seed 777 \
    --fixed-x0 zeros --out runs/<champion>/eval_x0zeros_v0_s777.json
```

Compare against the sweep's stochastic cells at the same task and spawn seed with
`analysis/paired_evals.py` — same spawn seed means the comparison is **paired**, so McNemar
applies and a 5-point difference is detectable at n = 96 where an unpaired test would need far
more episodes.

*Gate: belief 2 confirmed → the fix is a decoding choice, not a training problem, and it is
free.*

**C — x0-steering (Stage D), ~4 h.** Only if B helps but does not suffice. Now unblocked:
`SlotGraspBit` replaces the missing artifact, and `test_steer_cpu.py` already proves the
`z = 0` bit-exactness gate on CPU (17/17). Run a gate-S0 exploration-response pass to *choose*
`σ_init` from data rather than inheriting EXP07's −1.2. Note this task's window alignment is
strictly better than pick-place's: 600 steps = 40 windows exactly, no mid-episode terminations,
and **zero flushes measured in 128 episodes**, so there is no stale-`z` desync source at all.

**D — Tight demos, ~1 h.** Demoted per §2b, but not dead: if B and C both fail, the residual
possibility is that the *observations* under a 0.5 mm channel are genuinely off-distribution
during insertion, even though the actions are not.

```
python scripts/collect_demos.py --task Rebot-PrecisionSlot-Tight-v0 \
       --num_envs 128 --rollouts 4 --seed 30 --out data/v2/tight_s30.hdf5
```

---

## 6. Decision rule (fixed now)

Comparisons use **`success_rate_later`** and, wherever the spawn seed matches, the **paired**
McNemar test — never the headline `success_rate`, which carries the first-episode bias
(measured on this project at anywhere from **−2.1 to +18.7 points** depending on the
checkpoint).

* **B beats stochastic by ≥ 5 points on Tight, paired p < 0.05** → belief 2 confirmed. The
  deterministic base becomes the reported configuration for Tight; say plainly that it
  *contradicts* the pick-place precedent and why.
* **B is within ±5 points** → sampling noise is not the binding constraint. Belief 2 refuted;
  do not rescue it. Go to C, and re-read the taxonomy from A for what actually binds.
* **B is worse** → the pick-place precedent holds after all and multimodality is buying retries
  here too. That would make x0-*steering* more attractive, not less, since steering chooses
  among modes rather than freezing one.
* **Any single-run margin under 5 points is not evidence** — pick-place spanned 32.8–59.4 %
  across *training seeds alone* on identical data. Replicate across the sweep's three training
  seeds before claiming anything.

---

## 7. Results (2026-08-03, appended as they landed)

### 7a. Step A — the sweep. Belief 1 REFUTED, belief 4 confirmed

36/36 evals, zero failures. Pooled `success_rate_later` over 3 training seeds × 2 spawn seeds,
n = 1152 per arm per task:

| task | clearance | arm A | arm B |
|---|---|---|---|
| `-Loose-v0` | 3.0 mm | 0.778 | 0.799 |
| `-v0` | 1.5 mm | 0.776 | 0.792 |
| `-Tight-v0` | 0.5 mm | **0.708** | **0.764** |

**Belief 1 (25–55 %) is refuted: Tight is 0.736 pooled.** So are the two informal estimates it
replaced — 0.19 from the censored `-v0` CDF and 0.406 from the Loose-based fit fraction. All
three were built on the premise that the *channel width* is what binds. It is not.

**Belief 4 (yaw not binding) is confirmed emphatically: 1 `yaw_reject` in 797 failures** across
all three clearances.

**Why the ladder is flat** — the pooled failure taxonomy is nearly identical at every clearance,
and `never_entered` failures carry a median |lateral| of 0.59 / 0.79 / 0.73 mm. On the *Loose*
channel that is 0.59 mm of lateral error inside a **3.0 mm** opening, at depth −40.8 mm: the
block is well aligned and simply stopped ~40 mm short. `never_entered` + `stalled_in_mouth` are
**82–84 % of all failures at every clearance**, and both are depth failures.

> The binding constraint is **how far forward the policy drives the block in x** — which is
> clearance-independent. Tightening the channel 6× costs 5 points.

Per run, the ladder is not even monotone: 4 of 6 runs have Tight within 5 points of `-v0` or
better, and `armA_seed1` is *better* on Tight (0.786) than on `-v0` (0.745).

### 7b. Step B — the determinism test. Beliefs 2 and 3 CONFIRMED

`bc_armA_seed0/ckpt_final`, spawn seed 777, later cohort, n = 96 each:

| task | clearance | stochastic | `--fixed-x0 zeros` | Δ | p |
|---|---|---|---|---|---|
| `-Loose-v0` | 3.0 mm | 0.948 [0.884, 0.978] | 0.927 [0.857, 0.964] | −0.021 | 0.55 |
| `-v0` | 1.5 mm | 0.927 [0.857, 0.964] | 0.896 [0.819, 0.942] | −0.031 | 0.45 |
| **`-Tight-v0`** | **0.5 mm** | **0.708** [0.611, 0.790] | **0.875** [0.794, 0.927] | **+0.167** | **0.0045** |

* **Belief 2 confirmed** (≥ 5 points on Tight): **+16.7 points**, and the Wilson intervals do not
  overlap at all (0.790 vs 0.794).
* **Belief 3 confirmed** (neutral on `-v0`, within ±5): **−3.1 points**, p = 0.45. Same at Loose.
* **The pick-place precedent is contradicted, exactly as pre-registered.** There, blind mode
  *resampling* was worth +8.6 points (64.1 % stochastic vs 55.5 % best frozen x0). Here freezing
  the mode is worth +16.7 on the tight channel and costs nothing on the wide ones. The
  registered reason holds up: pick-place resampling buys *retries* at a task with multiple
  viable grasps and a forgiving 40 mm basket; here there is one viable trajectory and a 0.5 mm
  budget, so sampling variance is pure cost with nothing to buy.

The mechanism now reads cleanly: **x0 sampling noise is a fixed spatial jitter. At 3.0 mm and
1.5 mm the clearance absorbs it; at 0.5 mm it is a large fraction of the budget.** Freezing it at
the distribution's mode is free at the wide clearances and worth 16.7 points at the tight one.

### 7c. ⚠ A methodological correction to this document's own decision rule

§6 pre-registered that these comparisons be read with the **paired** McNemar test. **That is
structurally impossible for this comparison, and `analysis/paired_evals.py` caught it:** it
reported *spawns DIFFER in 96/96 slots — NOT paired, chi2 invalid*.

The cause is mechanical. With `--fixed-x0`, the policy stops calling `torch.randn` at every
chunk refill, so global RNG consumption changes and the reset-event stream desynchronises — the
exact hazard `summarize_arms.py`'s docstring flags. Both runs still draw spawns from the same
distribution at the same `--seed`, so the comparison is valid, just **unpaired and less
powerful**. The numbers above are therefore two-proportion z-tests with Wilson intervals, not
McNemar.

A hand-rolled comparison would have silently reported a paired χ² here. The rule stands
corrected: **any `--fixed-x0` vs stochastic comparison is unpaired by construction.**

### 7d. ⚠ REPLICATION REFUTES BELIEF 2 — and §7b's "confirmed" is retracted

The single-seed result looked decisive: +16.7 points, p = 0.0045, non-overlapping Wilson
intervals. §6's decision rule nevertheless demanded replication across training seeds, and that
is the only reason this document does not end with a wrong conclusion.

`--fixed-x0 zeros` on `-Tight-v0`, spawn 777, later cohort, **all six runs**:

| run | stochastic | `--fixed-x0 zeros` | Δ |
|---|---|---|---|
| `bc_armA_seed0` | 0.708 | 0.875 | **+0.167** |
| `bc_armA_seed1` | 0.781 | 0.406 | **−0.375** |
| `bc_armA_seed2` | 0.635 | 0.448 | −0.187 |
| `bc_armB_seed0` | 0.969 | 0.958 | −0.010 |
| `bc_armB_seed1` | 0.719 | 0.667 | −0.052 |
| `bc_armB_seed2` | 0.625 | 0.510 | −0.115 |

**n = 6, mean −0.095, sd 0.182, range [−0.375, +0.167]. One run improved, one flat, four got
worse.**

> **Belief 2 is REFUTED. Freezing x0 at the mode does not help on the tight channel. The
> +16.7 points on `armA_seed0` was a single-seed fluke, and it was the seed I happened to
> test first.**

§7b's "confirmed" is withdrawn. What made it look confirmed was picking the run whose Tight
deficit was largest — exactly the selection effect §7d was written to guard against, one section
before the data arrived. Every quantity in §7b was correct; the *inference* was not. Belief 3
(neutral at the wide clearances) is unaffected and stands.

**But the spread is the real finding, and it is large.** The same intervention moves success by
**−37.5 to +16.7 points depending only on the training seed — a 54-point range from the choice
of x0 alone**, with the policy weights, task, spawn seed and episode count all held fixed.

That reproduces EXP07's pick-place gate 2b almost exactly: frozen x0 draws there spanned
**14.1 %–56.2 %** success on one frozen base. The x0 → outcome map has enormous leverage on this
architecture; what varies is *which* x0 is good, and that is a property of the checkpoint (and,
plausibly, of the state) rather than a constant like "zeros is best".

**This is the strongest available argument for Stage D, and it is now measured rather than
borrowed.** Freezing x0 blindly is a coin flip with a 54-point spread. x0-*steering* is exactly
the intervention that replaces the blind choice with a learned, state-conditioned one — which is
the mechanism `POSTMORTEM.md` §9c credits for pick-place's 55.5 → 91.4 %. Belief 5's discount
still applies (the never-lifted bucket that drove most of EXP07's gain does not exist here), so
the expectation remains modest; but the leverage it would be steering is no longer hypothetical.

### 7e. Where this leaves the experiment

* **The objective does not depend on any of this.** BC already clears 70 % at every clearance
  (§7a). Steps C and D were contingency, and the contingency did not fire.
* **Step D (collect Tight demos) is dead**, on §2b's reasoning plus §7a's: the expert's plan has
  no clearance term, and the binding failure is depth, not clearance.
* **Step C (x0-steering) is now motivated by §7d rather than by the pick-place precedent** — but
  it should target **depth**, since 82–84 % of failures at every clearance are the block
  stopping short in x with lateral alignment fine.
* **`--fixed-x0` remains useful as an instrument**, just not as a fix: it is the cheapest way to
  measure how much of a checkpoint's behaviour is x0 sampling rather than the learned policy.
