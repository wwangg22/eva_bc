# 15 — The strict metric, in the environment. Threshold 2 mm.

**2026-08-03.** Big Will's decision, verbatim:

> "we need a strict threshold. The expert should not move the neighbors beyond 2mm. And feel
> free to edit the environment in ReBOT_RL. Just ensure to update the corresponding docs, and
> make sure you only touch the environment YOU ARE WORKING on."

Two decisions. The threshold is **2 mm**, the tightest of the three that were measured. And the
standing "do not modify `challenge/`" constraint is lifted **for the clutter env only** — so
this is no longer a private re-scoring bolted onto the evaluator, it is the task's own
definition of success.

## 0. Where this came from — and why ~5 000 scored episodes missed it

Big Will watched sixteen videos of the trained policy and said:

> "in many of the success episodes, the robot actually grabs another box as well as the one of
> interest. It just so happened when the robot placed it down, the box didn't topple over. So
> clearly this should be considered a failure, if the boxes that aren't the one of interest get
> moved around."

He was right, and it was verifiable without running anything. `mdp.target_at_goal` ended in
`& ~any_distractor_toppled(env)`, and `any_distractor_toppled` is `up_z < TOPPLE_DOT` with
`TOPPLE_DOT = 0.75` — about **41 degrees of tilt**. A neighbour dragged the length of the table
and set down upright satisfied it completely.

**The mistake was not failing to look.** The project's own standing rule since Stage 0 was
*"score with the env's own predicates, read at the end rather than enforced"*, and it was
followed exactly. That rule is right — inventing a private success criterion is how a project
ends up optimising something the benchmark does not reward. But it was applied without ever
asking the prior question: **does the benchmark's predicate actually encode the task?** Here it
did not. The environment's own docstring says "extract it and set it down in the goal zone
*without toppling any neighbour*", and the predicate is a faithful encoding of that sentence.
It is the sentence that was too weak, and no amount of care reading the code would have caught
it.

**The evidence was in the record the whole time.** The environment computed
`distractors_disturbed` — the summed planar displacement of all four neighbours — on every
single step, and wired it to a shaping reward and nothing else. And every hazard table from P17
onward reported a `close`-phase disturbance rate of **71–76 %**, measured, published in the
docs, and never allowed to reach the metric, because the predicate had already decided that
only toppling counted.

Two rules came out of it, both in `REFERENCE.md` §7: **ask whether the predicate encodes the
task**, and **a taxonomy only constrains what it enumerates** — the failure analysis at the
time called the policy's failure mode singular, and it was singular *among the buckets being
counted*.

---

## 1. The headline

```
                                          lenient      2 mm strict
frozen expert (v3 holds), 768 held-out eps  73.3 %          16.4 %
```

**56.9 points.** The expert retains **22.4 %** of its measured success rate. Everything this
effort has built is a solution to a task that was 4.5× easier than the one now in the repo.

The mission target — "solve it, ≈70 % on random starts" — is now **53.6 points away**, not
1.5. That is the honest statement of where the effort stands.

### 1.1 Two independent code paths agree to 0.1 points

This matters more than the number, because a metric change that is also a measurement error
is worthless.

| path | what it is | result |
|---|---|---|
| offline re-scoring | the **lenient** env, `eval_flow.py` latching neighbour displacement itself and applying the 2 mm cut in Python afterwards | **16.3 %** |
| env-native (this doc) | the **strict** env, `target_at_goal` gated on `~any_distractor_disturbed`, plus a `distractor_disturbed` termination | **16.4 %** |

Different environments, different termination structure, different place the predicate is
evaluated. They agree to within a tenth of a point on 768 episodes. The 2 mm cut is measuring
what it claims to measure.

### 1.2 The seed spread is exactly binomial

```
seed     88000  88001  88002  88003  88004  88005
strict   17.2   14.8   14.8   22.7   14.1   14.8   %
```

`sd(observed) = 3.26 pts` against `sd(binomial, n=128, p=0.164) = 3.27 pts`. **Ratio 1.00.**
Spawn batch contributes nothing beyond counting noise, so the pooled figure is a clean
estimate: **16.4 % ± 2.6** (95 %, n = 768).

Contrast with the BC training seed, whose sd was **9.8 points** — three times binomial. That
asymmetry is unchanged and still governs how many runs any BC comparison needs.

---

## 2. What changed in the environment

Everything is inside the two clutter-only source files plus the clutter test and docs. No
other task's behaviour changes. Committed as `ceeb24c` in `eva_rl`.

### 2.1 `challenge/mdp/clutter.py`

```python
DISTURB_TOL = 0.002

def distractor_displacements(env)  -> (N, 4)   # per block, planar, from spawn
def max_distractor_displacement(env) -> (N,)   # the MAX, not the sum
def any_distractor_disturbed(env, tol=DISTURB_TOL) -> (N,) bool

def target_at_goal(env, name="target", tol=DISTURB_TOL):
    ... & ~any_distractor_toppled(env) & ~any_distractor_disturbed(env, tol)
```

**Why max and not sum.** `distractors_disturbed` — the existing shaping reward — is the
*summed* displacement over the four blocks, and it stays that way: a sum is what gives a dense
gradient. But the constraint has to be a max. "No neighbour was moved" must not be satisfiable
by nudging four blocks 0.5 mm each, and under a sum at a 2 mm budget it would be.

**Why the instantaneous read is safe.** Displacement from spawn is very nearly monotone — a
shoved block does not slide back — so evaluating it per step and terminating is equivalent to
latching it, without needing state.

### 2.2 `challenge/clutter_env_cfg.py`

```python
distractor_disturbed = DoneTerm(func=mdp.any_distractor_disturbed,
                                params={"tol": mdp.DISTURB_TOL})
disturb_penalty = RewTerm(func=mdp.is_terminated_term, weight=-40.0,
                          params={"term_keys": "distractor_disturbed"})
```

**The penalty is not decoration, and leaving it out would have been a real bug.** Adding a
termination to an MDP whose shaping terms are net-negative hands the agent a way to *profit*
by triggering it: ending the episode early stops the accumulation of `action_rate`,
`joint_vel` and `disturbance` penalties. Without `disturb_penalty`, shoving a neighbour would
have become the highest-value action available at the start of every episode. It mirrors
`topple_penalty` at the same −40.

`RebotClutterExtractLenientEnvCfg` / `Rebot-ClutterExtract-Lenient-v0` restores the old rule
(`tol = inf`, both new terms off). It exists for exactly one purpose: the pre-2026-08-03
baselines were all measured under the old predicate, and **a baseline that cannot be re-run is
a baseline that cannot be checked.** Nothing new should be measured there.

---

## 3. P35 — the threshold is calibrated, not asserted

A constraint threshold below the simulator's noise floor fails every episode regardless of
what the policy does, and the task would read 0 % forever. So, before trusting it: **show it
does not fire when nothing violates it.**

`probes/p35_disturb_calibration.py`. Reset, then submit a null action every step for a full
700-step episode on `-Lenient-v0` (lenient deliberately — the strict env would terminate on
the very quantity being measured and auto-reset the scene from inside `env.step`, R23). `a=0`
decodes to `q_target = q_default`, so the arm holds its reset pose clear of the row and the
gripper closes on air. Anything that moves, moves because of the solver.

**Registered prediction, written before the run:** max displacement < 0.2 mm, 0 of 768 envs
disturbed.

```
   step    1:  max  0.0000 mm       NULL-ACTION DISPLACEMENT, 768 episodes
   step    5:  max  0.0003 mm          p50  0.0005 mm
   step   30:  max  0.0005 mm          p90  0.0010 mm
   step  100:  max  0.0005 mm          p99  0.0010 mm
   step  325:  max  0.0010 mm          max  0.0010 mm
   step  700:  max  0.0010 mm       disturbed at 2 mm: 0/768 = 0.00 %
```

**Worst case 1 µm over a full episode.** The threshold sits **2 097×** above it. The
prediction held and was conservative by a factor of 200. Drift saturates by step ~325 rather
than growing, so a constant tolerance is the right shape.

Bracketed on the other side by the row's **12 mm** free gap: 2 mm is a sixth of the space a
block has to move in, which reads as "not touched" rather than "nudged but still in its slot"
(that is nearer 10 mm). The threshold is comfortably inside a three-order-of-magnitude window,
so the exact value is not load-bearing — anything from ~0.05 mm to ~10 mm would have been
defensible, and 2 mm is Big Will's call within it.

**`runs/p35_disturb_calib.json`.**

---

## 4. The failure taxonomy collapsed to one bucket

The most informative single line in the re-baseline:

```
termination taxonomy: time_out 0.0%, target_dropped 0.0%,
                      distractor_toppled 0.0%, distractor_disturbed 83.6%
```

**`distractor_toppled` now fires 0.0 % of the time**, on a manoeuvre that used to report ~20 %
topples. Not because the topples stopped — because **a block must slide before it can tip**,
so the disturbance term fires first in literally every episode that would have ended in a
topple. Toppling was never a separate failure mode. It was the tail of the disturbance
distribution, and the old metric was counting only the tail.

That retires the last of `13_` §6.2. There is now exactly **one** failure mode, and this time
the taxonomy is not merely complete over the buckets it enumerates — every other bucket is
empirically zero.

### 4.1 Successes are genuinely clean, not marginal

```
neighbour displacement among SUCCESSES: median 0.00 mm, max 1.88 mm
STRICT success < 2 mm: 16.4%   < 5 mm: 16.4%   < 10 mm: 16.4%
```

The three thresholds now give **identical** numbers, and the median successful episode moves
its neighbours by **0.00 mm**. The population is sharply bimodal: an episode either touches
nothing at all, or it shoves. There is no population of near-misses sitting just under the
cliff, which means **the exact threshold does not matter to the result** — 2 mm, 5 mm and
10 mm select the same 16.4 % of episodes. Big Will's choice of the strictest option costs
nothing in discrimination.

This is a genuinely useful finding. It means the 42-point gap between 2 mm and 10 mm seen in
the offline re-scoring was an artefact of the *lenient env's* long tail (episodes that shove
and then carry on to the goal anyway), not evidence that displacement is a continuum to be
traded off. With the termination in place, the continuum does not exist.

---

## 5. The gates, restated

Both old thresholds were defined against the lenient predicate. Restated against the task as
it now stands:

| gate | old (lenient) | status | restated (2 mm strict) | status |
|---|---|---|---|---|
| Gate 1 — scripted expert | ≥85 % | 73.3 %, not met | ≥85 % | **16.4 %, badly not met** |
| Gate 1 revised floor `DR2` | ≥70 % | 72.1 %, met | ≥70 % | **16.4 %, NOT met** |
| Gate 2 — BC ≥ expert − 5 pts | — | passed | **≥ expert − 5 pts** | to re-measure |
| Mission | ≈70 % | 71.8 %, met | ≈70 % | **NOT met** |

**`DR2` no longer applies and the ladder has to be re-entered a rung lower.** The pre-registered
decision rule was "if the expert clears 70 %, proceed to BC" — the expert does not clear 70 %,
so the correct move by the project's own rules is to go back and fix the expert before
training anything else.

Gate 2 is the one worth re-measuring cheaply: BC was **1.8 points above** its expert under the
strict re-scoring, so the port and the cloning are probably still sound and the deficit is
entirely upstream. Re-measuring it costs one eval run per seed and no training.

---

## 6. Where the 56.9 points went, and what to do about it

All of it is one mechanism, already identified in Stage 1 and never fixed: **the finger blades
sweep the neighbouring blocks during the `close` phase.** P22 localised it; every hazard table
since P17 has reported a close-phase disturbance rate of 71–76 %; the re-baseline now reports
83.6 % as the *only* thing that happens.

The blade geometry is the constraint. From `target_axis()`'s docstring: a blade reaches
**~47 mm** along the opening axis against **7.8 mm** of perpendicular clearance. The fingers
straddle the target fore-and-aft (the orthogonal grasp, P11), so they do not enter the 12 mm
row gaps — but their *length* sweeps through the neighbours' footprint as they close.

Untried levers, in the order I would try them:

1. **Grip height.** The blades sweep at grip height. The blocks are 70 mm tall and the current
   grip is at `grip_z = 0.055`. Gripping nearer the top may clear the neighbours entirely,
   at the cost of a worse moment arm during the carry.
2. **A straight-up lift before any lateral motion.** From the videos, the carry begins moving
   sideways within 6–12 env steps of the fingers shutting. If the blade is still inside the
   row when lateral motion starts, that is a second, separable contribution.
3. **Approach angle / blade clocking.** The roll was worth +7.0 paired in P30; it has never
   been optimised against *disturbance*, only against topple.
4. **A slower close.** P27 found the close hold flat from 560 → 40 physics steps at 100 %
   enclosure, but it measured *enclosure*, not disturbance. Those are different questions.

All four are paired physics probes on identical spawns, so they are immune to the 9.8-point BC
seed noise, and each is minutes rather than hours. **Score them on strict success**, which is
now simply the env's own `target_at_goal`.

---

## 7. What this retired

Twelve stage-result documents were deleted on 2026-08-03 when the expert was restarted. They
were all written against the topple-only predicate, so every success number in them — 25 %,
57.7 %, 72.1 %, 73.3 % for the expert and 68.1 % / 71.8 % for flow BC — describes a task that
no longer exists. `REFERENCE.md` is what was worth keeping out of them.

Specific claims that are known wrong, recorded here so they are not rediscovered:

* **The apparent 42-point gap between a 2 mm and a 10 mm threshold was an artefact.** It was
  measured on the *lenient* env, where a shoved episode plays on to the goal. With the
  termination in place, 2 / 5 / 10 mm select the identical 16.4 %.
* **"The policy has one failure mode" was true for the wrong reason, and is now true for the
  right one.** It enumerated an incomplete taxonomy; every bucket except `distractor_disturbed`
  is now empirically 0.0 %.
* **"Shortening the holds gains the expert +3.9 points" (P27) was measured leniently and has
  never been re-checked.** It is the configuration currently shipping (`--close 40
  --holds-scale 0.25`) and the 16.4 % baseline uses it, but the comparison that justified it is
  unverified under the new metric.
* **The Stage-1 claim that the orthogonal grasp "stops the row's y-pitch being the binding
  constraint" is conditional on the row's heading**, which has never varied. It is not a
  general property of the task.
* **`pose_p33.json` and the single-pose recipe are specific to one row geometry** and to a
  metric that counted only topples. The pose is retained as the 16.4 % baseline to beat, not as
  a design to build on.
* **The `-Tight-v0` variant has never been measured at all**, under either predicate.

## 8. Files

| file | what |
|---|---|
| `eva_rl` `challenge/mdp/clutter.py` | `DISTURB_TOL`, three new terms, gated `target_at_goal` |
| `eva_rl` `challenge/clutter_env_cfg.py` | termination, penalty, `-Lenient-v0` variant |
| `eva_rl` `scripts/test_clutter_env.py` | V7 block + negative control (d); control (c) fixed |
| `eva_rl` `docs/envs/clutter-extract.md` | rewritten constraint section, validation table |
| `probes/p35_disturb_calibration.py` | the null-action calibration |
| `runs/p35_disturb_calib.json` | 768 episodes, 1 µm |
| `runs/strict2mm_expert.json` / `.log` | the 16.4 % re-baseline |

Rule earned here, for `HANDOFF.md` §11:

> **A constraint threshold needs a null control before it needs a result.** Show it does not
> fire when nothing violates it, and quote the margin. Two minutes of compute; without it,
> a threshold inside the noise floor is indistinguishable from a policy that cannot do the
> task.
