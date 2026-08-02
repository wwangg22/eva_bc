# EXP03 — Is DAgger interference real, or single-seed noise?

*2026-08-01. Status: QUEUED (GPU; runs after EXP02 eval chain).*

## Question

The v1-vs-v3 comparison (POSTMORTEM §3/§5) is **n = 1 vs n = 1**: one nominal training
run against one nominal+DAgger training run. The 17-broken episodes and the measured grip
divergence could in principle be *training-seed variance* rather than a causal effect of
the recovery data. Discovery that forced this experiment: `train_flow.py` had **no RNG
seeding at all** — v1 and v3 differ not only in data but in every random draw
(init, shuffling, flow-time sampling, x0 noise). Train ≥3 replicas per arm and see if the
arms separate beyond within-arm spread.

## Beliefs going in (pre-registered, before any result)

1. **Interference will reproduce** (the postmortem mechanism is measured, not inferred:
   0.357 grip divergence at lift, 20.3% vs 12.1% open-in-horizon on held states, and the
   grown lifted-never-placed bucket matches). Guess: every dagger replica shows higher
   mid-carry-release than every nominal replica.
2. **Within-arm churn will NOT be zero** — unseeded (now seeded) training re-draws
   everything, and BC on 59%-hard episodes plausibly flips several episodes on training
   seed alone. Guess: 5–15 episodes churn between two same-arm replicas. If it's much
   larger, the v1↔v3 "17 fixed / 17 broken" reading weakens a lot — that would be a
   *validation finding* worth its own section.
3. Aggregate success spread within arm: guess ±3–5 pts (2–3 episodes).

## Method

- **Seeding first**: added `--seed` to `act/train_flow.py` (torch/numpy/python +
  DataLoader generator) — surgical change, default `None` preserves old behavior.
- **Arms** (all: 100k steps, batch 64, lr 1e-4, chunk 50 — the exact v1/v3 recipe):
  - N1, N2, N3: `--pool nominal`, data = `expert/demos_nominal_s10*.h5`, seeds 1, 2, 3.
  - D1, D2, D3: `--pool nominal+dagger`, data = nominal + `expert/dagger_r1.h5`, seeds 1, 2, 3.
- **Exposure note:** as in the original v1/v3 pair, both arms get 100k optimizer steps, so
  the dagger arm sees ~23% fewer nominal samples. This replicates the original comparison
  (the question is "is THAT result reproducible"). If interference reproduces, one
  follow-up replica D4 runs with steps scaled to match nominal-sample exposure to separate
  "recovery data hurts" from "less nominal exposure hurts".
- **Eval:** each replica → 64 ep, seed 42, 16 envs, flush on (identical to baseline evals).
- **Metrics per replica** (scripts in this dir):
  1. success rate;
  2. taxonomy buckets (`taxonomy.py`): miss/freeze = never-lifted-anything +
     placed-1-stuck-never-lifted; mid-carry release = lifted-placed-nothing +
     placed-1-stuck-lifted-to-carry;
  3. grip divergence vs N1 on identical nominal states, per phase
     (`exp03_grip_divergence.py`, offline, GPU-light);
  4. per-episode churn matrices within-arm and across-arm.
- **GPU budget:** 6 × 35 min train + 6 × 12 min eval ≈ 4.7 h, strictly sequential,
  overnight chain with durable logs.

## Pre-experiment forensics (2026-08-02, BEFORE replicas ran) — the postmortem's "kill shot" was an artifact

While smoke-testing `exp03_grip_divergence.py` on v1-vs-v3, the tool did NOT reproduce
POSTMORTEM §5's numbers (20.3% vs 12.1% open-rate, 8.7% divergent). Per the
validate-odd-data directive we recovered the original analysis code from the session
transcript and ran an ablation matrix (`exp03_divergence_forensics.py`):

| condition | v1 open | v3 open | v1-holds-v3-opens |
|---|---|---|---|
| i. corrected: all 8 files, grasped-only holds, own stats | 0.136 | 0.137 | **0.007** |
| ii. + missed-attempt lift frames included | 0.116 | 0.149 | 0.039 |
| iii. grasped-only, v1's stats applied to v3 (orig. bug) | 0.123 | 0.128 | 0.008 |
| iv. exact replication of original (s101[:30], stride 5, no filter, v1 stats) | 0.121 | **0.203** | **0.087** |
| v. missed-attempt lift frames ONLY, own stats | **0.051** | **0.671** | 0.623 |

- Condition iv reproduces the postmortem numbers exactly → the original run is
  faithfully recovered; its *method* was flawed, not its execution.
- Two defects: (1) v3's obs were normalized with **v1's stats** (v3 trains/evals with
  its own pool stats); (2) lift frames of **missed** attempts were included — empty
  closed-gripper states where opening is v3's *learned recovery*, not damage. The
  s101-only subset concentrated defect 2.
- **Corrected picture:** on true hold states v1 ≡ v3 (13.6% vs 13.7%, 0.7% divergent).
  On empty-lift states v1 opens 5.1% (the freeze mode) vs v3 **67.1%** — v3 learned
  near-correct conditional reopen. Offline, v3's grip channel is *better*, not damaged.

**Belief revision (recorded before replica data):** belief 1's premise ("mechanism is
measured") is dead — the offline mechanism evidence for interference no longer exists.
v3's 17 broken episodes are now UNEXPLAINED, and unseeded training-run variance is the
top suspect. Revised expectation: arms likely do NOT separate; within-arm churn may be
large (possibly ~17 episodes, i.e., the v1↔v3 "fixed/broken" churn is the replica noise
floor, not a treatment effect). POSTMORTEM §5 carries a dated correction pointing here.

## Results (2026-08-02, chain 00:23–03:44; analysis: `exp03_analyze.py`, `exp03_grip_divergence.py`)

| replica | pool | seed | success | miss/freeze eps | mid-carry-release eps | grip open-rate on held states |
|---|---|---|---|---|---|---|
| N1 | nominal | 1 | **21/64 = 32.8%** | 27 | 16 | 0.129 |
| N2 | nominal | 2 | 32/64 = 50.0% | 16 | 16 | 0.126 |
| N3 | nominal | 3 | 38/64 = 59.4% | 18 | 8 | 0.129 |
| D1 | +dagger | 1 | **39/64 = 60.9%** | 18 | 7 | 0.131 |
| D2 | +dagger | 2 | 34/64 = 53.1% | 18 | 12 | 0.126 |
| D3 | +dagger | 3 | 36/64 = 56.2% | 17 | 11 | 0.130 |

Reference points: v1 (unseeded nominal) 59.4%, v3 (unseeded +dagger) 59.4%.

Pairwise success/fail **churn** (episodes that flip, same 64 spawns): within-nominal
{33, 31, 34}; within-dagger {39, 31, 34}; **v1↔v3 = 34** — the "17 fixed / 17 broken".

## Analysis

1. **Interference is NOT real** (pre-registered belief 1 REFUTED, consistent with the
   forensics above): dagger-arm mid-carry losses (7/12/11) are not elevated over nominal
   (16/16/8 — nominal replicas hold the two worst values). Grip open-rate on true hold
   states is flat across all six replicas and both unseeded originals (12.6–13.1%).
2. **The v1↔v3 "17 fixed / 17 broken" is exactly the replica noise floor.** 34 flipped
   episodes between v1 and v3; 31–39 flipped episodes between same-data different-seed
   replicas. DAgger round 1's per-episode effect is indistinguishable from retraining
   with a different seed. The POSTMORTEM's per-episode transition narrative is void.
3. **Training-seed variance is enormous: 26.6 pts (32.8–59.4%) within the nominal arm**
   — belief 2 confirmed but badly underestimated (guessed 5–15 episode churn; actual
   31–39). Meanwhile inference variance is zero (seeded x0, v1-v1 churn 0). All the
   randomness lives in training.
4. **This voids every single-run A/B in the results ladder**, including v2's "offline
   recovery data actively hurts" (51.6% is comfortably inside the nominal-arm spread).
   The teacher-quality and distribution-match theses from the postmortem may still be
   true, but our data no longer demonstrates them.
5. **DAgger shows no harm and weakly positive, more consistent results**: arm means
   56.7% (D) vs 47.4% (N); ranges 53.1–60.9 vs 32.8–59.4. n = 3 forbids a strong claim,
   but "recovery data stabilizes training" is now a live hypothesis (opposite sign to
   the one we started with).
6. Offline behavior probes on expert states have limited reach: N1 (32.8%) and N3
   (59.4%) are indistinguishable in offline grip open-rate — what separates seeds is
   closed-loop compounding, only visible in rollouts.

## Verdict (per pre-registered decision rule)

Arms do **not** separate beyond within-arm spread → v3's 17-broken was seed noise →
EXP04 (anti-interference recovery weighting) loses its premise and is **descoped**;
POSTMORTEM §3/§5 carry dated corrections. The actionable products of this experiment:

- **Champion selection must be variance-aware.** Best-of-6 on the seed-42 suite (D1,
  60.9%) overfits that suite; held-out eval (seed 123) of D1 / N3 / v1 / v3 ran via
  `exp03_run_heldout.sh`. Results (seed-42 / seed-123 / pooled 128 eps):
  **N3 59.4 / 68.8 / 64.1%** ← champion; D1 60.9 / 57.8 / 59.4%; v1 59.4 / 57.8 /
  58.6%; v3 59.4 / 50.0 / 54.7%. **`runs/exp03_N3/ckpt_final.pt` is the new frozen base
  for residual RL** (nominal pool, train seed 3). Note the suite-to-suite wobble (N3
  +9.4, v3 −9.4) — 64-ep single-suite readings carry ±5–10 pt eval noise on top of
  training noise; pooled 128-ep numbers are the citable ones. N3's held-out-suite anatomy:
  16 miss/freeze, 4 mid-carry (= all 20 held-out failures; buckets exhaustive).
  Pooled over both suites (46 failures, taxonomy.py): 34 grasp-phase miss/freeze
  (18 never-lifted + 16 placed-1-then-closed-on-air) + 12 carry/release (9
  lifted-never-placed + 3 stuck-at-carry); 0 drops-after-place — the same
  precision-shaped residue as ever.
- **Standard practice from now on: ≥3 seeds per training config**, report range, select
  on a held-out spawn set.
- The interesting open question shifts from "does recovery data hurt?" to "why is BC
  training variance 26 pts, and can we shrink it?" (candidates: longer training, EMA
  weights, larger nominal pool, ensembling).

## Decision rule (pre-registered)

Arms separate beyond within-arm spread on mid-carry-release and/or grip open-rate →
interference REAL → EXP04 (recovery down-weighting 0.1/0.25/0.5 and/or frozen-base
recovery adapter; CR-DAgger precedent). No separation → v3's 17-broken was seed noise →
rewrite POSTMORTEM §5 interpretation (the divergence numbers would then reflect generic
replica-to-replica drift, not DAgger-specific damage) and re-rank the ladder.
