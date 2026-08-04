# Session 4 — freezing the data, validating the instruments, training Stage C

Running record. Sections are appended as work completes; nothing here is written in advance of
the measurement it reports.

Entry point for the project as a whole is `HANDOFF.md`. This file is the *detail* behind
session 4's rows in it.

---

## 1. The background collection chain had deadlocked on itself

The previous session left four nominal pools (`nominal_s4..s7`) queued behind a shell waiter:

```
while ps aux | grep -q "[c]ollect_demos"; do sleep 20; done
```

The `[c]ollect_demos` trick stops `grep` from matching *its own* process. It does not stop `ps`
from matching the **waiter's own command line**, which contains the string `collect_demos` four
times over (once in the guard, three in the loop body it is guarding). So the waiter waited for
itself, forever. The machine was restarted in between, which hid it: the log's last entry was a
completed `dart005_s23`, exactly what a healthy chain in progress would look like.

Caught by checking `pgrep -af collect_demos` against wall-clock: the chain had been "running"
for hours with no new file. Killed it and ran the four pools directly.

**Gotcha for the list:** a process-waiter whose own command line contains the pattern it waits
on never terminates. Match on something the waiter cannot contain — a PID, a sentinel file —
or just run the work in sequence.

---

## 2. Two defects found by *reading* the training path, before running it

Both would have produced a plausible number rather than an error. Both were in code inherited
from pick-place and correct there.

### 2.1 The eval horizon was 2.5× too long

`eval_act.py --episode-length-s` defaulted to **30.0 s**, with a docstring explaining that
pick-place's expert demos ran ~1234 steps and the task default timed out 92 % of them.

On this task the arithmetic is different. From `precision_slot_env_cfg.py`:

| quantity | value |
|---|---|
| `sim.dt` | 1/400 s |
| `decimation` | 8 |
| control step | **20 ms** |
| `episode_length_s` | **12.0 s** |
| horizon | 600 control steps |

and every collected demo is **T = 599** — the demos *are* full-length episodes of this task,
by construction. A 30 s eval would roll the policy **900 steps past** the longest thing in its
training distribution, while `last_action` (obs dims 27:34) feeds its own output back into the
observation the entire way. Whatever number came out would not be a measurement of the trained
behaviour. It would also have cost 2.5× the GPU time to produce.

Default changed to 12.0 s.

### 2.2 The pool filter would have broken the arm comparison *or* trained on failures

`train_flow.py --pool` had four choices and the default was `default` = **no filter at all**.

| choice | keeps | effect on the S3 arms |
|---|---|---|
| `default` | everything | trains on the ~5 % of DART episodes that ended **unseated** |
| `nominal` | `success AND episode_kind == "nominal" AND clean` | drops **all 512 DART demos** from arm B → 512 vs 1024, the volume confound the experiment exists to rule out |
| `recovery` | `success AND NOT nominal` | arm B's DART half *only* |
| `nominal+dagger` | nominal + DAgger takeovers | no DAgger data exists yet |

None of them is "every successful demo". Added `success_pool_filter` and `--pool success`;
both arms now use it, so the only difference between them is composition.

Measured on `nominal_s0 + dart005_s20` (255 demos, 128 nominal + 127 dart, 11 failed):

```
success filter  244 kept  == 244 successful          PASS
nominal 128 + recovery 116 == 244                    PASS  (partition holds)
nominal filter kept 128 == the non-dart demo count   PASS  (it does exclude dart)
```

That third line is the confound, made visible: pointing arm B at `--pool nominal` would have
silently halved it.

### 2.3 A third, smaller one

The runbook in `EXP_BC_ARMS.md` said `--num_envs`; `eval_act.py` spells it `--num-envs`. This
one fails loudly, so it is only a time-waster, but it is fixed.

---

## 3. CPU validation of the training path (`scripts/test_pipeline_cpu.py`)

**Motivation.** Three separate times this project an unvalidated instrument produced a confident
wrong number: a determinism test that ran its own control condition last; a slip censor that
measured past the release; a headline success rate that was silently conditional on not
resetting. Training is the next instrument and it is the expensive one (2 arms × 3 seeds ×
100 k steps). Everything in that path except the Isaac Sim rollout is ordinary PyTorch and
checks in 40 s on CPU.

All 17 checks pass. The ones that could actually have failed:

**The loss censor reaches the loss.** `train_mask` rides the `action_is_pad` channel, and the
flow loss multiplies by `~action_is_pad`. Verified on a synthetic 40-step demo with a known
mask by *corrupting* the labels and watching the loss:

```
loss with clean labels                       6468.139160
loss with +1e3 garbage on CENSORED steps     6468.139160   <- bit-identical: censor works
loss with +1e3 garbage on TRAINABLE steps   540827.687500  <- responds: the loss is live
```

This is the check that would have caught the censoring machinery being inert — which is exactly
the state it was in before this session's predecessor found `train_mask == 0` on 0 demos.

**Chunk commitment is intact.** The load-bearing property of the method: shortening the
execution horizon collapsed pick-place success 59.4 → 32.8 → 3.1 → 0 → 0 % at `n_action_steps`
15/8/4/2/1. Counted forward passes directly with a proxy policy:

```
3 forwards over 45 control steps (n_action_steps=15)         PASS
every refill batched all 4 envs at once (queues in phase)    PASS
feeding an absurd observation mid-window changes nothing     PASS  (1 forward after 2 steps)
reset([0,2]) forces an immediate refill of exactly 2 envs    PASS
```

**The checkpoint round-trips bit-exactly.** `load_checkpoint` *rebuilds* the architecture from a
config dict rather than storing it, so silent train/eval architecture drift is possible.
Verified weights bit-identical, stats identical, and — the real test — that the saved and
reloaded policies emit the **same action from the same x0**, max abs difference `0.00e+00`.

**Normalization round-trips** to 1.7e-06, and no stat has a degenerate std (min 3.8e-03 on
`observation.environment_state`).

---

## 4. The frozen data set — 16 pools, 2038 demos

| pool | demos | successful | rate | noise σ | phases |
|---|---:|---:|---:|---:|---|
| `nominal_s0`–`s4`, `s6`, `s7` | 128 each | 128 each | **100.0 %** | 0.0 | — |
| `nominal_s5` | 128 | 127 | 99.2 % | 0.0 | — |
| `dart002_s10`–`s13` | 128 each | 126/122/123/121 | 94.5–98.4 % | 0.02 | reach,lift,back,spin,turn |
| `dart005_s20`–`s23` | 127/124/124/127 | 116/115/114/117 | 91.3–92.7 % | 0.05 | reach,lift,back,spin,turn |
| **total** | **2038** | **1977** | **97.0 %** | | |

`T = 599` for every demo → **1,220,762 frames**, of which **1,184,223** sit in successful demos.

Two things worth recording:

* **The expert is not quite 100 %.** Seven of the eight nominal pools are perfect; `nominal_s5`
  lost one. Over 1024 nominal attempts that is **1023/1024 = 99.90 %**. The previous session
  reported "512/512 across four seeds" — that was true of those four seeds and is not a claim
  that the expert cannot fail. It can, at about 1 in 1000, which is consistent with the measured
  non-reproducibility (identical actions from an identical state already flip 23–25 % of
  *marginal* outcomes; almost all spawns are not marginal).
* **DART pools lose ~5 % at σ=0.02 and ~8 % at σ=0.05**, monotone in σ, exactly as the noise
  sweep predicted. Those failures are what the censor was calibrated against.

`verify_demos.py` passes on both a nominal and a DART pool. On `nominal_s7` the alignment
identity `obs[t,27:34] == actions[t-1]` holds at **max abs difference 0.000e+00**; on
`dart005_s20` the noise-aware replacements pass (residual std 0.0371 vs declared 0.05, phase
restriction held to 0.05 % of σ past `push`, lag-1 autocorrelation 0.939 against the OU ρ=0.95).

> The measured 0.0371 against a declared 0.05 is not a defect and the tolerance is not papering
> over one. The OU process is decayed by `hold_decay` each step during settles, and settles are
> roughly 40 % of the trajectory, so the pooled standard deviation over *all* noise-active frames
> is necessarily below the stationary value. The stationary check is the phase-restriction and
> autocorrelation pair, both of which are tight.

---

## 5. The slip censor: built, calibrated — and deliberately NOT applied

This is the direct continuation of the failure-labelling question ("we should be masking out the
gradients for when the expert policy messes up"). The machinery works. The measurement says it
has no work left to do, and applying it anyway would cost good data.

### What the calibration found, over all 16 pools

```
2038 demos, T = 599, successful 1977 (97.0 %)
push phase starts at t = 358

                      window   AUC raw  AUC excess    excess p50 seated/failed
          whole gripped span     0.366       0.623          3.01 /     3.49
  CARRY only (grasp -> push)     0.465       0.693          2.35 /     3.21
 PUSH only (push -> release)     0.372       0.538          1.11 /     1.59

best window: CARRY only, excess AUC 0.693   (clears the 0.65 gate)
threshold at 2 % FPR on successes: excess > 6.49 mm
  catches 14.8 % of failed demos, censors 2.0 % of successful ones
  fires at t = p10 149 / median 188 / p90 291 of 599  -> keeps 33 % of those episodes' frames
```

The 12-pool run gave AUC 0.697; 16 pools give 0.693. The signal is real and stable. Note the
raw-AUC column is still **below 0.5 in every window** — the inverted statistic from session 3
(the expert drives the block into the back stop, so high raw slip means a *fully seated*
insert) survives at full sample size. Only the excess statistic, after subtracting the
per-timestep median over successful demos, points the right way.

### Why it is not applied

Both arms train with `--pool success`. On a success-only pool the censor's arithmetic inverts:

* the **14.8 % of failed demos it catches are already excluded** — the filter dropped all 61 of
  them before the mask is ever read. That entire benefit is unrealisable.
* the **2.0 % of successful demos it censors are not** — by construction (the threshold *is* the
  98th percentile of the success distribution). That is ≈ 40 demos, truncated at median t=188,
  keeping 33 % of their frames: roughly **16,000 frames of data from episodes that seated the
  block deleted**, in exchange for nothing.

So on this configuration the censor is a pure subtraction. It is switched off, and the pools stay
exactly as collected (`train_mask` all ones, `slip_censor_t = -1`), which also keeps them frozen
and reproducible.

### The alternative that was considered and rejected

The data-efficient use of DART failures would be to train on *everything* and let the censor
truncate each failed episode at the point the expert lost the plot — a failed episode's first
60 % is perfectly good supervision. That needs a censor with high **sensitivity**, and this one
has 14.8 % at a 2 % FPR. The other 85 % of failed demos would train end-to-end, bad ending
included. That is strictly worse than dropping them, so it was not done.

Lowering the FPR does not help either: it only shrinks the (unrealisable) catch rate further.

### What this answers

The masking question was the right question, and it has now been answered with numbers rather
than with machinery:

1. **Expert failures are labelled.** Every demo carries `success`, a per-phase `outcomes` map
   (`grasped`/`missed`, `held`/`lost`, `seated`/`unseated`), and the raw `slip_mm` signal.
   Nothing is thrown away silently.
2. **They are excluded from training** by `--pool success`, which is a cleaner instrument than
   masking for whole-episode failures.
3. **Sub-episode masking is available but unwarranted here**, because the failures are not
   preceded by a visible in-hand event 85 % of the time. Given that identical actions from an
   identical state already flip 23–25 % of outcomes on this task, most failures are simulator
   chaos rather than expert error — and chaos is not maskable. That was written into
   `calibrate_slip.py` as a possible verdict before the data was in; this is that verdict, at a
   weaker form (the signal exists, it is just not usable at this operating point).
4. **The machinery stays** for Stage D/DAgger, where masking is genuinely required: a gated
   takeover's policy-driven prefix is known-bad by construction, needs no detector, and is
   hard-zeroed by the collector.

---

## 6. Validating `eval_act.py` in sim — and the bug it was built to catch

`eval_act.py` had **never been run end-to-end on this task**. The plan called for validating it
before training rather than after, on the grounds that an unvalidated instrument had already
produced three confident wrong numbers here. It found a fourth within two minutes.

Method: train a deliberately throwaway checkpoint (2000 steps, 40 s, two nominal pools), run the
eval on it, and audit the resulting JSON for *internal arithmetic consistency* with
`scripts/check_eval_json.py` — every rate recomputed from the per-episode records, cohorts
checked to partition, horizons checked to be uniform.

### 6.1 The bug: three diagnostics were describing the wrong episode

First run reported, on the same 48 episodes:

```
success_rate 0.438      |lateral| median 126.79 mm      depth median 8.9 mm
```

A 15 mm half-width block cannot be 127 mm off-centre and simultaneously inserted in a 16.5 mm
channel. One of those numbers was lying.

It was the diagnostics. `depth_mm`, `lateral_mm` and `yaw_rad` were read **inside the
`if done:` branch — after `env.step`**. Isaac Lab's `ManagerBasedRLEnv` resets done envs
*inside* `step()`, so those three were measuring the freshly-respawned block of the *next*
episode. 127 mm is simply the spawn region's distance from the slot.

Everything that decides success was already sampled correctly, one control step before the end:
`placed_mask`, `is_inserted`, `final_obj_pos`, `max_obj_z`. So the **headline number was never
wrong** — only the three fields you would reach for to explain it. That is arguably the worse
failure mode: it would have produced a well-formed, entirely fictitious failure analysis.

Confirmed by cross-checking against `final_obj_pos`, which *was* pre-step:

| cohort | x (m) | y (m) | z (m) |
|---|---|---|---|
| success (n=21) | 0.2554 ± 0.0018 | +0.0001 ± 0.0008 | 0.0552 ± 0.0009 |
| fail (n=27) | 0.2143 ± 0.0374 | −0.0017 ± 0.0175 | 0.0488 ± 0.0152 |

`z = 0.0552` is `SEAT_Z` to within 0.2 mm and `x = 0.2554` is the expert's own `insert_x`
(0.2545). The successes are genuinely, precisely seated.

**Fixed** by hoisting the three reads next to `placed_mask` at the top of the loop. After the
fix, same checkpoint and seed:

```
|lateral| median  126.79 mm  ->  0.92 mm
depth    median     8.9 mm   ->  36.5 mm
```

### 6.2 The eval is process-level deterministic at a fixed seed

Re-running the identical `(ckpt, task, num-envs, episodes, seed)` returned **bit-identical**
rates: 0.438 / 0.562 / 0.375 both times.

This is worth stating carefully next to session 3's non-reproducibility finding, because they
look contradictory and are not:

* **Within a process, replaying identical actions from an identical state is NOT reproducible** —
  18 mm of block displacement, 23–25 % of outcomes flipped. That measurement was across
  *repeated episodes inside one process*, where PhysX contact manifolds and solver warm-start
  caches survive `env.reset()`.
* **Across processes, a fresh run with the same seed IS reproducible**, because the whole
  cache history is replayed from the same starting point in the same order.

The practical consequence for the arm comparison: **re-running an eval at the same seed yields
no new information.** Variance must be estimated by varying `--seed`, never by repeating a run.
An error bar built from repeated identical evals would be exactly zero and completely wrong.

### 6.3 The first-episode bias reproduces

0.562 first-episode vs 0.375 later — **+18.7 points**, on top of the +12.9 (4.3 σ) measured on
the expert in session 3. Same direction, same order of magnitude, now on a *learned* policy
rather than a scripted one, which rules out anything specific to the expert's plan. This is why
`success_rate_later` is the comparison statistic.

### 6.4 A free failure taxonomy, from a 2000-step model

Reconstructed from pre-step positions (the `x` bands are the slot geometry: mouth at
`MOUTH_X`, seated centre at 0.2554):

| failed block ended at | n | reading |
|---|---:|---|
| x < 0.190 | 8 | never advanced — z of 0.012–0.043 says dropped or never lifted |
| 0.190 ≤ x < 0.235 | 6 | short of the mouth |
| 0.235 ≤ x < 0.250 | **13** | **at the mouth, at seated height, short of depth** |
| x ≥ 0.250 | 0 | — |

Half of all failures are the same failure: the block reaches the mouth, at the correct height,
and does not go deep enough. Not grasping, not alignment — **depth**. Worth holding onto as a
prior for Stage D, where the natural intervention would be to steer the push.

### 6.5 One expectation of mine was wrong, and it was not a bug

`check_eval_json.py` initially asserted every episode ran 599 steps, matching the demo arrays.
Every episode ran **600**. The demos are one shorter because the collector stores one
(obs, action) pair per step and the final observation has no action after it. Same horizon,
different bookkeeping. The assertion was corrected to 600 with a note not to "fix" either side
to match the other.

### 6.6 The number nobody expected

**A 2000-step checkpoint trained for 40 seconds on 256 demos scores 43.8 % on `-v0`.**

That was not the plan's expectation for the *final* Stage C bar on the hardest clearance
(≥ 40 % on v0). It is a strong signal that this task is far more learnable from state
observations than the pick-place history suggested, and it makes the 70 % target look
reachable without RL. It also means the Stage C bar as pre-registered is nearly uninformative
and the interesting question moves to *how far above it* the arms land.

Recorded here before the real runs finish, so it cannot be retrofitted.

---

## 7. How well-conditioned is the data? (`analysis/label_consistency.py`)

Run while the sweep trained, so the answer would be in hand *before* the Stage C numbers rather
than invented afterwards to explain them.

### The question

BC fits `a = f(obs)`. If two frames from **different demos** have near-identical 34-D
observations but different recorded actions, no deterministic policy can fit both, and the
achievable error is bounded below by that disagreement. That bound is a property of the data.
Without it, a policy that has already hit the floor is indistinguishable from one that failed
to converge.

There was a specific reason to suspect a floor here: §3d of the handoff records that the arm's
null space makes the demos multimodal — the same task-space pose is reachable at joint
configurations differing by up to **133 mrad**, and the expert's IK picks a branch by warm start.
If that branch is not predictable from the observation, it is irreducible label noise.

### Method

Subsample frames, normalize each dimension by its std (so the neighbour metric is roughly the
one the network sees), and for every query frame find its nearest neighbour **from a different
demo** — same-demo neighbours are trivially close in both obs and action and would swamp the
statistic. Then bin the action disagreement by observation distance. The lowest bin is the floor.

Run twice: on the full 34-D observation, and on the env-state half only (dims 16:34), which is
what the policy would see if proprioception carried no branch information.

### Result — the floor is essentially zero

512 nominal demos, 306,688 frames, 30,000 sampled:

```
          obs-dist bin       n  |dArm| p50      p90      max   -> p50 mrad  grip flip
[   0.000,   0.009]   1,501      0.0002   0.0016   0.1161           0.1      0.00%
[   0.009,   0.066]   4,502      0.0022   0.0100   0.1151           1.1      0.00%
[   0.066,   0.232]   9,001      0.0180   0.0471   0.1516           9.0      0.61%
[   0.232,   0.454]   9,000      0.0560   0.0949   0.2571          28.0      0.34%
[   0.454,   3.963]   6,000      0.1013   0.1715   0.7310          50.6      0.80%

LABEL-NOISE FLOOR: 0.0002 action units = 0.1 mrad = 0.0 % of the action std (0.694)
```

**0.1 mrad.** The action is an essentially deterministic, smooth function of the observation, and
the disagreement rises monotonically and gently with observation distance — the signature of a
well-conditioned regression, not of a multimodal one. The env-state-only run gives the same
floor, so proprioception is not doing the work either; the labels are simply consistent.

The feared null-space multimodality does **not** show up as label noise. The reason is that the
expert warm-starts each IK solve from the previous waypoint, so within a demo the branch is
continuous, and across demos the same spawn produces the same warm-start chain — session 3
already measured the plan to be a bit-identical function of the block pose (0.0000 mrad). The
133 mrad of null-space freedom exists, but the expert never exercises it *inconsistently*.

This is the quantitative explanation for §6.6: a 2000-step model reaching 43.8 % is not a
surprise once you know the target function is this clean.

### The same measurement on DART data — the DART value proposition, quantified

462 successful DART (σ=0.05) demos:

| pool | median cross-demo NN distance | label-noise floor |
|---|---:|---:|
| nominal | 0.232 std-units | 0.0002 |
| DART σ=0.05 | **0.625 std-units** | 0.0002 |

**DART data covers 2.7× more observation space at identical label quality.** That is exactly
what noise injection is supposed to buy and it is not something the collection logs showed
directly. It also sharpens the arm comparison: if arm B does not win, it will **not** be because
its labels are noisier — they are measurably just as clean. It would have to be because the extra
coverage is in a region the policy never visits.

### The risk this measurement exposes

Every demo follows the same phase schedule with the same step counts, so cross-demo nearest
neighbours land a **median of 1–2 timesteps apart**. Combined with `last_action` occupying obs
dims 27:34, the policy has everything it needs to learn an *open-loop clock* — integrate its own
previous action and replay a trajectory indexed by time and by the initial block pose — rather
than closed-loop feedback. On this data that strategy would score very well.

It would also fail exactly where the task is hard, and it makes a **prediction for the arm
comparison**: arm A can learn the clock; arm B cannot lean on it as heavily, because DART
perturbs the executed action while labelling with the nominal one, so the clock and the
observation disagree by construction. If arm B wins, this is the most likely mechanism. If both
arms score similarly on `-v0` but arm A degrades more under Stage D perturbations, same story.

Worth testing directly later by evaluating a trained policy with the block spawn re-randomised
*mid-episode*, which a clock cannot survive and feedback can.

---

## 8. `scripts/diag_feedback.py` — the test for the clock hypothesis

Built and validated while the sweep trained, so it is ready the moment there is a champion to
point it at.

**Design.** At a chosen step during `reach` — while the block is still free on the table —
teleport it to a **new pose drawn from the env's own reset `pose_range`**, read from the config
rather than restated so it cannot drift from what the policy was trained on. Then let the
episode run.

* a **clock** policy continues to where the block *was* and closes on nothing;
* a **feedback** policy re-targets and grasps where the block *is*.

The new pose is always one the policy has seen at t=0, so nothing out-of-distribution is being
asked except the timing.

**Two controls, because a bare number here is uninterpretable.**
1. `--perturb-step -1` runs the identical harness with no teleport. Any drop from this baseline
   is caused by the perturbation, not the harness.
2. `--resample-only` writes the block's *current* pose back through the same code path and the
   same physics flush. If this differs from control 1, the write itself is disturbing the
   episode and the main result is confounded.

The controller queue is **flushed** at the teleport. That is the *best* case for the policy —
not flushing would confound "cannot react" with "was not asked to react yet."

### The timing constant that makes or breaks this test

The perturbation must land while the block is free. Measured directly from the demo actions
rather than assumed: the gripper channel `actions[:, 6]` has exactly **two** transitions in a
599-step demo, at step **40** (close) and step **464** (release). Segment boundaries are
`reach 0 → lift 71 → back 99 → spin 141 → turn 201 → push 358 → release 465`.

So the block is only free over steps **0–39**, and the initial default of `--perturb-step 40`
was exactly the last free step — the policy would have had zero steps to react, which tests
nothing. Default changed to **20** (mid-reach, ~20 steps ≈ 0.4 s of reaction time), with a hard
guard rejecting any value ≥ 40 and explaining why. Sweeping 5/15/25/35 gives a reaction-time
curve.

### Validation

Smoke-tested on the throwaway checkpoint. It found a device bug first — `ep_index` and
`moved_mm` live on the CPU while `done` comes off the sim device, and indexing a CPU tensor with
a CUDA index raises. Fixed, then:

```
[diag] perturbation drawn from the env's own reset pose_range:
       {'x': (-0.02, 0.02), 'y': (-0.03, 0.03), 'yaw': (-0.35, 0.35)}
[diag] perturb_step=20 resample_only=False  success=0.625  mean |move| 13.6 mm
```

The perturbation is real (13.6 mm mean displacement) and the harness runs clean. **n=8 is a
smoke test, not a result** — and `success_rate_later` was `None` because `--episodes` equalled
`--num-envs`, so every episode was a first episode. The real run needs `--num-envs 16
--episodes 64`.

Catching this at a moment when it cost three minutes, rather than at the end of a six-hour
chain, is the entire reason for smoke-testing a diagnostic before it is needed.

---

## 9. The learning curve so far (partial — the sweep is still running)

All on `-v0` (1.5 mm clearance), `--num-envs 32 --episodes 128 --seed 777`, reported on the
unbiased `success_rate_later` cohort:

| steps | data | later | first-ep | bias | n(later) |
|---:|---|---:|---:|---:|---:|
| 2 000 | 256 demos, throwaway | 0.375 | 0.562 | **+18.7 pts** | 32 |
| 10 000 | arm A, 1023 demos | **0.615** | 0.625 | **+1.0 pt** | 96 |
| 100 000 | arm A, 1023 demos | *pending* | | | |

**The pre-registered Stage C bar (≥ 40 % on v0) was cleared at 10 000 steps** — 3.5 minutes of
training. The bar was set from the pick-place history and is simply not calibrated to this task.

### The first-episode bias is not a constant

+18.7 points on the weak model, +1.0 on the stronger one. The natural reading is that the bias
acts on **marginal** episodes — ones that could go either way — and a weak policy has many while
a strong one has fewer. Whatever PhysX warm-start advantage the first episode enjoys can only
flip an outcome that was close to begin with.

This is worth stating carefully because it is tempting to conclude the bias has gone away and
drop the cohort split. It has not. The bias is a property of *policy × task*, not a constant, and
the arms will be compared at different strengths on three different clearances. `-Tight-v0` in
particular will produce many marginal episodes for any policy. **Keep reporting both cohorts and
keep comparing on `later`.**

### What it implies for the target

Extrapolating a two-point curve would be foolish, and 100 k results are hours away. But 61.5 %
at 10 k on the *hardest-but-one* clearance, from a policy that has never seen RL, makes it
plausible that **Stage C alone reaches the 70 % goal on `-v0`** and that Stage D becomes a
push-further rather than a rescue. `-Loose-v0` should be comfortably higher and `-Tight-v0`
(0.5 mm, one third of v0's clearance) materially lower.

Recorded now, with the sweep incomplete, so the prediction is on the record before the data is.

---
