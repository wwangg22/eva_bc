# Session 6 — diversifying the environment: does the policy still work?

*2026-08-03. Big Will's instruction for this session was "**diversify the env, see if your policy
still works**". This is the answer. It is also the session where the project stopped asking "can
it hit 70 %" — that was settled in session 5 — and started asking what the policy actually
learned.*

**Everything here is measured on checkpoints trained in session 5. No new training ran.** The
work is 40+ perturbed evaluations, four pre-registrations, one expert control, and one
replication round on a second training seed.

---

## 1. The short version

The policy was trained on a world with **one slot position, one arm start pose, one noiseless
observation, and a ±20 × ±30 mm × ±20° block spawn box**. Session 6 moved all of those.

| axis moved | result | is it the policy's fault? |
|---|---|---|
| slot **+10 mm** in x | 1.000 / 0.969 — no measurable loss | — |
| slot **+20 mm** in x | −8 pts, **replicated on 2 seeds** | **yes** — the expert does it 128/128 |
| slot **−10 mm** in x | −13.5 pts on one seed, −1.0 on another | no — a checkpoint idiosyncrasy |
| slot **±1 mm** in y | −3.1 pts (p = 0.375, null) | — |
| slot **≥2 mm** in y | **0/96, twice, on two seeds** | **no** — geometry (§3) |
| arm start ±0.10 rad | −2.1 pts (p = 0.41, null) | — |
| sensor noise 5 % | −4.2 pts (p = 0.15, null) | — |
| sensor noise 20 % | −93.8 pts | yes |
| **actuator** noise 2 % | **−83.3 pts** | yes |
| **actuator** noise 5 % | **0/96** — the same magnitude that is free on the observation | yes |
| spawn box ×1.5 | −15.6 pts; **in-distribution episodes 43/43 = 1.000** | only out-of-box |
| spawn box ×2.0 | −42.7 / −31.2 pts, **grasp starts failing** | only out-of-box |
| episode horizon 12 → 20 s | +11.5 pts on one checkpoint, −2.1 on another | it was **out of time** |

**The headline:** this policy is much less brittle than "BC memorises its training distribution"
predicts. It **tracks a moved goal** — its block's absolute final x follows the slot 1:1 over
±10 mm, and it does so on an axis it has never seen vary. It survives an arm start pose it has
**literally never seen**, at the randomisation level a sibling task in the same repo uses. The
one place it fails absolutely is a place where geometry, not learning, sets the limit.

---

## 2. The finding that changed how the result should be described

**The policy tracks the slot in x, and the evidence is not the success rate.** It is where the
block physically ends up:

| slot dx | median final block x | offset from nominal |
|---|---|---|
| 0 | 0.2570 m | — |
| +5 mm | 0.2620 m | **+5.0 mm** |
| +10 mm | 0.2670 m | **+10.0 mm** |

Exactly 1:1. And the median *insertion depth* of successes is 47.3 / 47.4 / 47.3 mm — unchanged,
because the block bottoms out against a back stop that moved with everything else.

This closes the clock-vs-closed-loop question (`HANDOFF.md` §3h) that `scripts/diag_feedback.py`
was written for and never ran. `diag_feedback` moves the *block* for one instant; this moves the
*goal* for the whole episode, 20 mm beyond anything in the training data, and the policy follows.

---

## 3. The finding that is not about the policy at all

**A 3 mm lateral slot shift scores 0.000 on `-v0` and 0.760 on `-Loose-v0`.** Same checkpoint,
same spawn seed, spawn-for-spawn identical episodes (McNemar, all 96 verified). The only
difference is `clearance`: 1.5 mm vs 3.0 mm.

Work the geometry and it has to be that way. The block is 30 mm wide (`BLOCK_HALF[1] = 0.0150`);
the mouth is `2 × (15 + clearance)` mm. For the block to enter, its centre must be within
**±clearance** of the slot centre. The policy delivers y = 0.0 ± 1.0 mm — it already uses most of
that budget on `-v0`. A 5 mm shift does not make entry hard; it makes entry **impossible**.

Twelve cells across three clearances collapse onto a **single curve in (shift / clearance)**,
with no exceptions:

| dy / clearance | cells | success |
|---|---|---|
| 0.00 | Tight, `-v0`, Loose at dy = 0 | 0.969 / 0.979 / 0.927 |
| **0.67** | `-v0` @ 1 mm, Loose @ 2 mm | **0.948 / 0.938** — both null vs their own baseline |
| **1.00** | Loose @ 3 mm (exactly zero margin) | **0.760** |
| **≥ 1.33** | six cells, 0.5–3.0 mm clearance, 1–10 mm shift | **0.000 ×5, 0.021 ×1** |

The three crossed pairs are what make this more than a fit — same checkpoint, identical spawns,
only a config number different:

* **2 mm shift: 0.000 on `-v0`, 0.938 on `-Loose-v0`.** Widening rescues it.
* **1 mm shift: 0.948 on `-v0`, 0.021 on `-Tight-v0`.** Narrowing destroys it. A lazy "wider is
  easier" story gets this direction wrong; the geometric model has to predict a failure at a
  shift the middle rung shrugs off, and it does.
* **3 mm on Loose = 0.760**, the exact-boundary case. "Degraded but working" is what precisely
  zero margin should look like.

So the honest statement is:

> The lateral tolerance of this task is the clearance. The policy saturates it. And the slot's
> lateral position enters the observation only through an **absolute value** (`lateral_error`),
> so no memoryless policy trained on this observation could do better — it cannot tell +2 mm from
> −2 mm.

That matters for what to build next: **a moved-in-y slot is an observation-design problem, not an
RL problem.** Steering the flow's `x0` cannot move a geometric bound.

---

## 4. What I got wrong, and the pattern in it

Four pre-registered mechanism predictions were falsified this session. Three of them were the
same mistake.

| # | prediction | reality |
|---|---|---|
| 4 | slot further away is *harder* (depth is the constraint); slot nearer is *easier* | backwards — +10 mm was free, −10 mm was the worse direction |
| 6 | arm jitter at 0.10 rad costs > 30 points; the **grasp** breaks first | 2.1 points, p = 0.41; `never_lifted` **stayed at zero** |
| 12 | the expert also dips at −10 mm (the arm is folded there) | 125/128 = 0.977, essentially flat |
| 7 (half) | 20 % sensor noise fails **laterally** | it fails by **not advancing** — median failure lateral 0.11 mm, the lowest of any cell |

Beliefs 4, 6 and 12 are all the same error: **I kept assuming parts of this policy were
open-loop.** "It memorised a path ending at x = 0.2545." "The fixed-start approach segment is the
memorised part." "The arm must be worse where the demos never went." Every one of those was
wrong, and each was wrong in the direction of underestimating how much the policy re-derives from
the observation.

Belief 7's miss is different and more useful: the *magnitude* was wrong (94 points, not 25) but
so was the **character**, and the character is the informative part — see §5.

`EXP_ROBUSTNESS.md` §7b records the second of these falsifications being *anticipated* before
round 2 ran, because round 1 had just falsified the first. That is the pre-registration doing its
job: it converted "I was wrong twice" into "I was wrong about the same thing twice", which is a
statement worth having.

---

## 5. Everything funnels into the push — and the push is not *degraded*, it is *not attempted*

Five unrelated stresses, one terminal signature:

| perturbation | dominant failure | median depth | median \|lateral\| |
|---|---|---|---|
| slot dx = −10 / +20 mm | `stalled_in_mouth` | +36 to +38 mm | 1.0–1.3 mm |
| spawn box ×2.0 | `stalled_in_mouth` 16/43 | +21.3 mm | — |
| **sensor noise 20 %** | `never_entered` **85/92** | **−43.8 mm** | **0.11 mm** |
| **actuator noise 2 %** | `never_entered` **77/82** | **−42.9 mm** | **0.35 mm** |
| **actuator noise 5 %** | `never_entered` **92/96** | **−44.0 mm** | **0.94 mm** |

Three of those are the *lowest* lateral errors measured anywhere in this project, including the
unperturbed gate — and the same depth, to within a millimetre, from two entirely different noise
types. Noise does not make this policy wander. So **where does the block stop?**

| cell | final block x | final y | final z | still held above the table |
|---|---|---|---|---|
| gate (2 failures) | 0.2480 | 0.0000 | 0.0550 | 2/2 |
| sensor noise 20 % | **0.1660** | 0.0000 | 0.0620 | 86/92 |
| actuator noise 2 % | **0.1670** | 0.0000 | 0.0620 | 81/82 |
| actuator noise 5 % | **0.1660** | 0.0000 | 0.0630 | 92/96 |

**x = 0.166 m is the staging waypoint** — the expert's `stage_x` is 0.165, and the carried block
rides ~5 mm ahead of the TCP. z ≈ 0.062 is carried height, not seated and not on the table.

So under noise the policy reaches, grasps, lifts, retracts to staging, aligns in y to **0.0000**
— three phases at nominal precision — and then **freezes there, holding the block, for the rest
of the episode.** The push is not degraded. It is never attempted.

That reframes the whole bottleneck. The block sits 90 mm from a hole it has aligned to a tenth of
a millimetre, and the policy cannot commit to the last phase. "Improve precision" was never the
problem; **leaving a state it reaches reliably** is.

`analysis/lateral_by_bucket.py` confirms this on the unperturbed sweep too: pooled over all 36
cells, on `-Tight-v0` the `stalled_in_mouth` failures and the successes have the **same** lateral
distribution to two decimal places (median 0.37 vs 0.34, p90 0.47 vs 0.49). Whatever stops those
116 episodes 40 mm short, it is not where they are pointing.

So the three failure buckets are three mechanisms:

* `stalled_in_mouth` — aligned as well as a success, stopped short → **push / depth**
* `never_entered` — p90 lateral 4–7 mm, far beyond any clearance → **aiming**
* `gross_miss` — 40–130 mm off → **transport**

and the fragile one is the push.

---

## 6. Two things nobody had counted

### 6a. The depth failures are **stuck, not slow** — and I claimed the opposite for an hour

Round 1 of the horizon probe (12 s → 20 s, two checkpoints chosen for opposite failure shapes)
gave **+11.5 points** for the stalled-in-mouth subject and **−2.1** for the never-entered one, and
I wrote up a "slow versus stuck" mechanism on top of the `never_entered` counts (13 → 7 vs
20 → 20). Round 2, at a second spawn seed, gave **+0.0 and +10.4**. The differential reverses.

| | seed 777 | seed 888 | pooled |
|---|---|---|---|
| `bc_armB_seed2` (stalled shape) | +11.5 | **+0.0** | +5.7 (p = 0.22) |
| `bc_armB_seed1` (never-entered) | −2.1 | **+10.4** | +4.2 (p = 0.38) |

**All four cells pooled (n = 384 per arm): 0.667 → 0.716, +4.9 points, p = 0.138.** No dependence
on failure shape. And it saturates: 30 s gives 0.729 against 20 s's 0.740.

So probe A's answer is **negative**, which is still worth having: the horizon is not the
explanation for the depth bottleneck, the depth failures are genuinely stuck, and they are
therefore a legitimate Stage D target rather than something a config change would have fixed.

**The mistake worth naming** (`EXP_DEPTH.md` §8c): §7 correctly flagged its own p = 0.088 as
underpowered and named the precedent — last session's +16.7-point effect that replicated as
1 improved / 4 worse. Then it said the *mechanism* evidence was firmer than the rate, and built
that mechanism out of `never_entered` counts **from the same two underpowered cells**. A second
statistic computed from the same cells is not independent evidence. At seed 888 the same
statistic gives 9 → 5 and 25 → 19 and "supports" the opposite conclusion equally well.

The protocol was right — round 2 was written and queued *before* §7 was drafted, precisely
because the p-values were weak. The prose got ahead of it.

### 6b. 3.0 % of episodes seat the block and then lose it

`placed_max > placed_final` — a fully seated block that is not seated at the end — happens in
**102 of 3456** later-cohort sweep episodes. It is not spread evenly: all six of
`bc_armB_seed2`'s cells are in the worst eight (5–10 %), while `bc_armA_seed0` and
`bc_armA_seed1` have cells at exactly **zero**.

This is a third failure mechanism, invisible in every success-rate table in this project because
success is sampled at the end of the episode. It is also a per-checkpoint trait, which puts it in
the same category as the horizon sensitivity in §6a: **two policies with the same score can be
failing in completely different ways.**

---

## 6c. Chunking is what makes this policy work, and what leaves it exposed

`--obs-noise 0.05` costs 4.2 points (p = 0.15, a clean null). `--action-noise 0.05` — the *same*
fraction of each channel's own training std — scores **0/96**. Even 0.02 costs 83 points.

The asymmetry has a mechanism, pre-registered before the cells ran and confirmed: the controller
re-reads the observation once per **15 steps**, so an observation error is low-passed by roughly
that factor before it can act on anything. An action error goes to the joints on every one of
those 15 steps with nothing in between. Within a window this policy is **open-loop by
construction**, and has no mechanism whatsoever for rejecting a disturbance.

Session 5 established that chunk-50/execute-15 is load-bearing — shortening the window collapsed
pick-place from 59.4 % to 0 %. This is the other side of that ledger, and it is worth knowing
before anyone deploys a chunked policy on hardware where 2 % actuator error is optimistic.

---

## 7. The one concrete improvement target

**dx = +20 mm costs ~8 points, replicates across two training seeds (−9.4 and −7.3, pooled
McNemar p = 0.0037), and the scripted expert does it 128/128.**

That is the only deficit found this session that is (a) reproducible, (b) attributable to the
policy rather than the robot or the geometry, and (c) demonstrated to have a ceiling above it.
Everything else is either free, already at a geometric limit, or checkpoint-specific noise.

By contrast, `dx = −10 mm` — which round 1 reported at −13.5 points, p = 0.001 — **does not
replicate**: 1.0 point on `bc_armA_seed0`, p = 1.0. That claim is retracted, and it is a clean
illustration of why `PLAN` 5.28's single-seed caution exists: 15–29 point training-seed variance
is larger than most of the effects this project measures.

---

## 8. Instruments built this session

| file | what it does |
|---|---|
| `analysis/robustness_report.py` | reads perturbed cells; **decides paired vs unpaired from the recorded `spawn_pos`** rather than assuming, and scores each cell against a baseline **on its own task** |
| `analysis/lateral_by_bucket.py` | \|lateral\| by outcome × clearance, pooled — the cut that separates aiming failures from push failures |
| `scripts/run_robustness{,2,3}.sh` | rounds 1–3, all idempotent |
| `scripts/run_dy_crossed.sh` | the clearance-crossed dy ladder |
| `scripts/run_horizon{,2}.sh` | EXP_DEPTH probe A and its replication |
| `scripts/run_expert_dx.sh` | the expert control |
| `scripts/run_action_noise.sh` | actuation noise (§10 of EXP_ROBUSTNESS) |
| `scripts/make_robust_videos.sh` | the two clips a table cannot convey |
| `eval_act.py` flags | `--arm-jitter`, `--obs-noise`, `--action-noise`, plus `spawn_yaw` in the records |

Two harness bugs were found by reading results rather than by anything failing:

* **`robustness_report.py` scored `-Loose-v0` cells against the `-v0` gate**, turning
  `loose_dy_p003`'s honest −0.167 into a meaningless −0.219. Fixed to pick a baseline per task.
* **`run_expert.py` serialised the pre-shift `insert_x`**, so all four expert cells recorded
  `0.2545` and looked byte-identical in the one field that distinguishes them. The runs were
  correct; the provenance was not. Same class as `make_videos.sh` writing three videos to one
  filename — *an artefact that looks complete and is quietly identical*.

---

## 9. What did not change

`eva_rl` has **zero** modifications — every perturbation is an in-process patch of `SLOT_CENTER`
or an `EventTermCfg` appended to a config instance, with the geometry read back out of the scene
afterwards to prove the walls moved and not just the score. `eva_bc` has only untracked `slot/`
and `docs/slot/`. Nothing is pushed.
