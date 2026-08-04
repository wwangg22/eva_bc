# EXP_STEER — can x0-steering teach the frozen policy a behaviour it does not have?

*Opened 2026-08-03 12:20, **before any steering run**. Big Will's framing: "let's see if we can
use steering to actually teach it behavior it hasn't seen before (i.e. the noise stuff)."*

## Status

**RUN, AND THE PPO LAUNCH IS WITHDRAWN.** The cheap probe that §8 inserted ahead of training
answered the question and then invalidated the training design. Headlines:

* **Gate passes in its strong form** — `eval_steer.py` reproduces `eval_act.py` **episode-for-
  episode**, 96/96, zero mismatches (§9a).
* **A constant integration latent takes the champion from 0.146 → 0.823** under 2 % action
  noise, with **no training at all** — 81 % of the deficit, p = 6 × 10⁻²¹ (§11).
* **The steering action space cannot contain that latent.** `SteerCore` broadcasts one 7-vector
  across all 50 chunk positions; all four broadcast cells score **0.000/96**, two of them by
  never lifting the block (§12). The §6 launch command is **withdrawn** — do not run it.
* **Selection, not commitment, is where the points are**: +51 vs +16 (§13a). And the oracle over
  latents **does not move** when the candidate pool nearly doubles — 0.885 at both 5 and 9
  latents, so conditional latent selection is capped at **+6.2 points** (§13b).

**Reading order: §8 → §9 → §11 → §12 → §13.** §1–§7 are the original pre-registration, kept
verbatim; §4's beliefs 1–5 and §6's command are superseded but not deleted.

---

## 1. Why this target is much better than the one Stage D was originally aimed at

Stage D was going to steer for **`-Tight-v0` precision**. Session 6 killed that: the clearance is
not what binds (5 points across a 6× range), and the lateral error of a stalled failure equals
that of a success on Tight to two decimal places (`EXP_DEPTH.md` §6). There was nothing there to
steer *toward*.

The noise condition is a far better target on four counts:

1. **Enormous headroom.** `--action-noise 0.02` takes the champion from **0.979 → 0.146**. That
   is 83 points of recoverable ground, against the ~5-point margins Stage D was chasing.
2. **A known ceiling.** The scripted expert executes this task from a *fixed* plan; the arm is
   demonstrably capable (128/128 at every slot dx tested). The base policy also does it at 0.979
   without noise. So the behaviour exists — it is the *robustness* that is absent.
3. **A named failure state.** Not "it is imprecise" but: block held at **x = 0.166 m, y = 0.0000,
   carried height, for hundreds of steps** (`EXP_ROBUSTNESS.md` §13b). The policy completes three
   of four phases at nominal precision and never attempts the push. There is one transition to
   fix and we know exactly where it is.
4. **A dense reward that already points at it.** `RewardsCfg` has `kp_baseline` / `kp_coarse` /
   `kp_fine` (weights 6 / 12 / 20) plus `engaged` (15) and `inserted` (60). The keypoint terms are
   graded all the way from staging to seated, so a policy frozen at x = 0.166 sits on a **live
   gradient**, not a plateau. This is the single biggest practical difference from the Tight
   target, where the reward was nearly flat across the region of interest.

## 2. The mechanistic case that x0 is the *right* knob — and the honest limit

**The case for.** In a flow-matching policy, `x0` is the sole source of stochasticity: the chunk
is a deterministic function of `(observation, x0)`. Different `x0` therefore selects a different
sample from a **multimodal** conditional action distribution. And §13b's failure has exactly the
shape of a mode problem: the policy at staging holds a distribution over "push" and "hold", and
under noise the executed behaviour collapses to "hold". **Selecting a mode is what x0 does.**

This is also what the measurements already say. `EXP_TIGHT.md` §7 found the x0 choice alone spans
**54.2 points** across checkpoints (−37.5 to +16.7), reproducing EXP07's gate 2b (14.1–56.2 %).
That is not a subtle knob; it is the largest single lever measured on this project.

**The honest limit, stated before we spend anything.** x0-steering **cannot synthesise an action
the flow policy does not already represent.** It re-weights among modes; it does not add new ones.
So the experiment does not really test "can steering teach unseen behaviour" — it tests something
sharper and more interesting:

> **How much of what looks like "the policy can't do this" is actually "the policy can do this
> but samples the wrong mode"?**

If steering recovers most of the 83 points, the answer is "nearly all of it", and the practical
lesson is that BC robustness failures may be *sampling* failures rather than *capability*
failures — which would be a genuinely useful result, and one that transfers well beyond this task.
If steering recovers little, then the push behaviour under noise is genuinely outside the base
policy's support, and the fix has to be data or fine-tuning, not steering. **Both outcomes are
worth the GPU time, which is the main argument for running it.**

## 3. Design

Frozen base: **`bc_armB_seed0/ckpt_final.pt`** (the champion; 0.979 clean, 0.146 at 2 % action
noise). Steering: `SteerCore`, `x0 = α·tanh(z)`, α = 1.0, one z per 15-step execution window,
broadcast across all 50 chunk positions. PPO via rl_games on the 50-D steering observation.

Noise is applied **inside the wrapper's window loop**, to what the *controller* sees and commands
— not to what the steering policy observes. That is a deliberate choice and it is the easier
version: the steerer gets a clean view of a corrupted rollout. The harder version (noisy steering
obs too) is the follow-up **if this one works**, and the distinction must be stated in any result.

| arm | condition | what it tests |
|---|---|---|
| **A** | `--action-noise 0.02` | the headline: 0.146 → ? |
| **B** | `--action-noise 0.00` (control) | does steering help at all when the base is at 0.979? Guards against "steering just makes everything better" |
| **C** | zero-z control, no training | the base under noise, re-measured through the *steering code path* — the gate |

Arm C is the gate and it is not optional: `test_steer_cpu.py` already proves `steer_x0 = zeros` is
action-for-action identical to `fixed_x0 = zeros` on CPU (max abs diff **0.000e+00**, 17/17
checks), but that is not the same as proving the *wrapper's* noise injection matches
`eval_act.py`'s. If arm C does not reproduce ≈0.146, the training environment is not the
environment the deficit was measured in and nothing else is interpretable.

## 4. Beliefs, pre-registered

1. **Gate (arm C): zero-z steering under 2 % action noise reproduces `eval_act.py`'s 0.146
   within sampling error** (Wilson interval overlap). If not, stop.
2. **Arm A recovers more than half the deficit** — final success **> 0.55** (vs base 0.146,
   clean ceiling 0.979). Mechanism: the failure is a mode collapse at a single, identifiable
   state, the reward is dense and graded exactly there, and x0 is the mode-selection knob.
3. **The recovered policy still fails the *hard* version.** If arm A succeeds, adding noise to
   the steering observation as well will cost most of the gain back. Mechanism: a steerer that
   can see cleanly is solving a much easier problem than one that cannot, and I would rather
   predict that now than discover it as a caveat later.
4. **Arm B (clean) gains little** — under +5 points over the 0.979 base. Ceiling effects, and
   nothing to select between when the base already picks the right mode. **A large arm-B gain
   would be a red flag**, not a success: it would suggest the reward or the protocol is buying
   points the eval protocol does not measure.
5. **Steering will not close the gap fully.** Final arm A < 0.90. If the base's push-under-noise
   mode is only *thinly* represented, selecting it every window still gives a noisier trajectory
   than a policy that never needed correcting.

Belief 2 is the one I would bet on and it is the one worth being wrong about. Belief 5's
complement — arm A ≥ 0.90 — would mean the base policy contains a fully competent noise-robust
behaviour that ordinary sampling almost never finds, which would be the most surprising result of
the project.

## 5. Decision rule

* **Gate fails** → stop and fix the harness. Nothing below is readable.
* **Arm A > 0.55 and arm B < +5 pts** → steering recovers robustness the base cannot express by
  sampling. Run the hard version (belief 3) before claiming anything general.
* **Arm A > 0.55 and arm B also large** → suspect the protocol, not the method. Check whether the
  steering training env and the eval protocol have diverged (this is exactly how EXP07's
  train/eval mismatch was caught the first time — see `train_steer.py`'s module docstring).
* **Arm A < 0.30** → the behaviour is outside the base policy's support. Steering is the wrong
  tool and the honest next step is fine-tuning or noise-augmented BC data, not more RL.
* **Every comparison is unpaired** (noise consumes RNG; the reset stream diverges) — two-proportion
  z-tests and Wilson intervals, read with `analysis/robustness_report.py`.
* **Single-seed caution applies** (`PLAN` 5.28). Training-seed variance on this project is 15–29
  points and the x0 choice alone spans 54. **One steering run proves nothing**; the claim needs at
  least two seeds before it goes anywhere near a headline.

## 6. Launch

Plumbed and compiled; `test_steer_cpu.py` 17/17 and `check_port.py` 16/16 still pass after the
change. One GPU job at a time.

```
python slot_act/train_steer.py --ckpt runs/bc_armB_seed0/ckpt_final.pt \
    --run-name steer_actnoise02_seed1 --seed 1 --num_envs 2048 --max_iterations 200 \
    --action-noise 0.02
```

**Not yet plumbed:** `eval_steer.py` needs the same two flags before arm A can be *scored* under
noise, and the arm-C gate needs a zero-z evaluation path. That is the first task next session and
it is small — the sigma construction is eight lines already written twice (`eval_act.py`,
`train_steer.py`) and should probably move into a shared helper rather than be copied a third time.

## 7. What this costs and what could waste it

~200 iterations at 2048 envs. EXP07's equivalent took a few hours; the slot task's 12 s episodes
are 40 windows each, so a run is roughly that. The two ways this wastes the time, both guarded:

* **The gate is skipped** and arm A is compared against `eval_act.py`'s 0.146 measured through a
  different code path. Guarded by belief 1.
* **One seed is run and the result is quoted.** Guarded by the decision rule, and by the fact that
  this project has already retracted two single-seed claims *this session*
  (`EXP_ROBUSTNESS.md` §9d, `EXP_DEPTH.md` §8).

---

## 8. Addendum, written 2026-08-03 17:20 — still before any steering run

Implementing §6's "not yet plumbed" list surfaced a mistake in the gate as pre-registered, and
fixing it turned up a much cheaper experiment that should run first. Both are below, and both
are written before a single GPU second was spent on steering.

### 8a. Belief 1's reference number was wrong: 0.146 is not the right target

Belief 1 says *"zero-z steering under 2 % action noise reproduces `eval_act.py`'s **0.146**"*.
It cannot, and it should never have been written that way.

`robust_act002.json` — the run that produced 0.146 — records `"fixed_x0": null`. That is the
**stochastic** policy: a fresh `x0 ~ N(0, I)` on every chunk refill, i.e. a new draw every 15
env steps, 40 times an episode. Zero-z steering sets `x0 = alpha * tanh(0) = zeros` and **holds
it**: a deterministic policy. Those are two different policies, and there is no reason the
second should reproduce the first's success rate.

This is the same error the project has now made twice: comparing two cells that differ in a
second, unrecorded way. It is caught here only because `fixed_x0` is recorded in the config
block — the field exists precisely because *"a deterministic-mode run is otherwise
byte-indistinguishable from a stochastic one"* (`eval_act.py`, comment on the provenance dump).
The comment was right and I ignored it.

The upstream EXP07 gate had this correct: it compared z = 0 against `x0sweep_s-1_seed{42,123}`
— the **x0-zeros** base, not the stochastic one. The slot version must do the same. No
x0-zeros-under-noise cell exists on `-v0`, so the gate now needs its own reference:

| cell | harness | x0 | noise | role |
|---|---|---|---|---|
| **C0** | `eval_act.py` | `zeros` | none | is `zeros` a good draw at all, absent noise? |
| **C1** | `eval_act.py` | `zeros` | `--action-noise 0.02` | the gate **reference** |
| **C2** | `eval_steer.py` | z = 0 | `--action-noise 0.02` | the gate **subject** |

All three at the standard robustness protocol: champion `bc_armB_seed0/ckpt_final.pt`, `-v0`,
seed 777, 128 episodes / 32 envs, later-cohort scoring.

**And the corrected gate is *stronger* than the one it replaces.** With `--fixed-x0 zeros`
neither harness draws a CUDA `randn` for x0; both then draw exactly one `randn_like(action)`
per env step for the action noise, and both call `env.step` the same number of times in the same
order. So the noise streams should stay in lockstep, and `test_steer_cpu.py` already proves the
two controllers are action-for-action identical (max abs diff **0.000e+00**). Therefore:

> **Belief 1 (restated): C2 reproduces C1 EPISODE-FOR-EPISODE**, not merely within sampling
> error. Same successes, same per-episode lengths. If the streams turn out not to align, the
> fallback is Wilson-interval overlap on the later cohort — but a mismatch in the *per-episode*
> pattern with matching *rates* is itself worth a note, because it would mean the two paths
> consume randomness differently and every future paired comparison between them is invalid.

### 8b. The probe that should run before any PPO: does x0 move anything under noise?

Belief 2's entire mechanism is **mode collapse** — "the policy at staging holds a distribution
over push and hold, and under noise the executed behaviour collapses to hold". If that is true,
then *some* x0 must select the push. And a constant-x0 sweep tests that directly, for roughly
**1 % of the cost of a PPO run**, using code that already exists and has already been used
(`EXP_TIGHT.md` §7 measured a **54.2-point** span across x0 choices — this is the largest single
lever ever measured on this project).

Cells: `--fixed-x0 {zeros, 1, 2, 3, 4}` under `--action-noise 0.02`, same protocol as above.
Five constant draws — the same x0 held for the whole episode, every episode.

This is not a substitute for steering. A constant x0 is the *weakest possible* steerer: it
cannot condition on state, so it cannot pick "push" at the staging boundary and something else
during the carry. It is therefore best read as a **lower bound** on what a trained,
state-conditioned steerer could reach, and as a **direct test of the mechanism**.

**Pre-registered beliefs for the probe:**

6. **The spread across the five draws is large — best minus worst > 15 points.** Mechanism: the
   54.2-point span measured clean, and noise should if anything *widen* the gap between an x0
   that commits and one that does not. **If the spread is under 5 points, x0 is not a lever
   under this condition and belief 2 is dead.**
7. **The best of five beats the stochastic 0.146 by more than 15 points** (best > 0.30).
   Mechanism: a stochastic policy redraws x0 every 15 steps, so it re-selects a mode 40 times an
   episode. At the staging boundary that is 40 chances to pick "hold". Committing to one draw
   removes that, and §13b's failure is *specifically* a failure to commit.
8. **`zeros` is not the best draw.** It is the mode of the prior, and a policy that fails by
   being too conservative is unlikely to be rescued by the most average latent available. I
   would rather write that down than retrofit it.

**Decision rule for the probe:**

* **Spread > 15 pts and best > 0.30** → the mechanism holds and a state-conditioned steerer has
  strictly more to work with. Launch §6's PPO arms.
* **Spread > 15 pts but best < 0.30** → x0 matters but no constant draw is enough. Still launch:
  this is the case a *conditional* steerer is for, and it is the most interesting outcome.
* **Spread < 5 pts** → **do not launch PPO.** Under noise the chunk is essentially independent
  of x0, there is no mode to select, and the honest write-up is that the deficit is a capability
  failure, not a sampling failure. The fix would be noise-augmented BC data or fine-tuning.
  This branch would save several GPU-hours and is the reason the probe runs first.

### 8c. What was built for this

* **`slot_act/noise.py`** — the per-channel sigma construction, previously copied by hand in
  `eval_act.py` and `train_steer.py`. It now has one home. This is not tidiness: the gate
  compares a rate measured through `eval_act.py` against one measured through `eval_steer.py`,
  and that comparison is only meaningful if `--action-noise 0.02` denotes the *identical*
  perturbation in both. A third hand-written copy is exactly how the two numbers quietly stop
  being comparable.
* **`eval_steer.py`** — `--obs-noise` / `--action-noise` (same helper, same semantics, applied in
  the same order as `steer_wrapper.py`: perturb the controller's input and its command, never the
  steering observation), plus `episode_index_in_env` and the `success_rate_first_episode` /
  `success_rate_later` split that every other eval on this project reports. Without the split the
  gate would be comparing a 32-of-128 first-episode cohort against another one and calling the
  warm-start bias a steering effect.
* **`scripts/test_steer_cpu.py`** — five new checks pinning the noise semantics (22 total, all
  passing), including that the obs sigma zeroes `obs[27:34]` and that the sigmas are copies of
  the checkpoint stats rather than views of them.

### 8d. Pre-registered *before* the gate cells land: why does 2 % action noise cost 83 points?

Written 2026-08-03 17:35, with C0 in hand (x0 = zeros, clean, `-v0`: **0.938** later — so `zeros`
is a slightly *worse* draw than stochastic's 0.979, −4.2 pts) and C1 still running. This section
is a mechanism hypothesis, and it exists because §2's "mode collapse" story is **not** the only
one the existing evidence supports, and I would rather have both written down before the numbers
arrive than pick the one that fits afterwards.

**The magnitude is the puzzle.** The arm action is
`JointPositionActionCfg(scale=0.5, use_default_offset=True)` — an **absolute** target,
`q_target = q_default + 0.5 · a`. So per-step action noise does **not** integrate into a random
walk; that was the first thing I checked and it would have explained everything. It doesn't.
2 % of a channel's training std, uncorrelated every step, on an absolute target that a stiff
joint drive tracks — and it costs **83 points**, while 5 % noise injected directly into the
*observation* costs 4.2 (null). A perturbation that enters the physics is ~40× more damaging
per nominal unit than the same nominal perturbation entering the sensor. Something amplifies it.

**Hypothesis: the amplifier is `joint_vel`.** Position barely moves — the drive filters
high-frequency target jitter. But velocity is the *derivative* of that jitter, and
differentiation amplifies high frequencies. `obs[8:16]` is `joint_vel_rel`, and the policy sees
it every step. So a perturbation that is 0.02 σ in *action* units could be several σ in
*joint-velocity* units, which would make the policy's input out-of-distribution in exactly one
channel group — while `--obs-noise 0.05` puts a mere 0.05 σ on that same channel and is
correctly ignored.

This reframes the failure. §2 says "the policy holds a distribution over push and hold and
collapses to hold" — a *sampling* problem, which steering fixes. This says "the policy is being
fed an observation unlike anything in its training set and its nearest learned behaviour is to
keep holding" — a *distribution-shift* problem, which steering mostly cannot fix, because no
choice of x0 changes what the encoder is being shown.

**Prior evidence that bears on it, from work already done here.** `HANDOFF.md` §4b measured the
*scripted expert* under DART action noise: at `noise_std = 0.02` seated success falls to 57 %,
**all** of the loss is in the `push` phase, grip retention is 128/128 in every free-space phase,
and push-phase arm deviation goes 74 → 530 mrad. The explanation there was physical: inside a
1.5 mm channel, commanding the nominal target from a perturbed state drives the block into a
wall at stiffness 2000 and levers it out of the pads. **But that mechanism cannot explain the
BC policy's failure**, because the BC policy stops at **x = 0.166 m** — `stage_x`, in free space,
~44 mm short of the wall faces at 0.210 (§4c). Nothing is jammed. It stops on its own. Whatever
freezes it is in the *policy*, not the contact.

**Beliefs, pre-registered:**

9. **`joint_vel_rel` (`obs[8:16]`) is the most out-of-distribution channel group under 2 %
   action noise, by a wide margin** — its mean |z| against the training normaliser is more than
   2× that of any other group. This is the load-bearing prediction.
10. **`joint_pos_rel` (`obs[0:8]`) stays near-nominal** — mean |z| under 1.5, and materially
    lower than `joint_vel`'s. Positions are what the drive tracks; velocities are what it rings.
11. **The frozen block is not in contact.** At x = 0.166 nothing is touching the fixture, so the
    freeze is a policy decision, not a physical stall. (Already strongly implied by §4c's
    geometry; recorded so it is a prediction rather than an assumption.)

**How it gets tested — pure measurement, no training.** Add an obs-distribution diagnostic to
`eval_act.py` that accumulates, per channel, the runtime mean and std of the observation and its
|z| against the checkpoint's own normaliser, then run it clean and at 2 % action noise. Two
cheap cells. If belief 9 holds, the honest headline of this experiment changes from "steering
recovers robustness" to **"the deficit is a distribution shift in one input channel"** — and the
indicated fix is velocity-aware augmentation or dropping `joint_vel` from the observation, both
cheaper and more likely to work than PPO.

**This does not cancel the probe.** The two hypotheses make *different* predictions about the
constant-x0 spread — mode collapse predicts a large spread, distribution shift predicts a small
one — so §8b's probe discriminates between them, and it is already running.

---

## 9. RESULTS — the gate (2026-08-03 17:32)

### 9a. Gate: PASSES in its strong form

`analysis/steer_report.py`, champion `bc_armB_seed0`, `-v0`, seed 777, 128 ep / 32 envs,
later cohort (n = 96):

| cell | harness | x0 | later success | Wilson 95 % |
|---|---|---|---|---|
| **C1** | `eval_act.py` | `fixed_x0 = zeros` | **0.167** | [0.105, 0.254] |
| **C2** | `eval_steer.py` | `z = 0` | **0.167** | [0.105, 0.254] |

* shared episodes **96**
* outcome mismatches **0**
* episode-length mismatches **0**
* rate delta +0.000, z = +0.00, p = 1.0000

**The two harnesses are episode-for-episode identical.** Not "statistically indistinguishable"
— literally the same 96 outcomes and the same 96 lengths. §8a predicted this on the argument
that with `fixed_x0 = zeros` neither path draws a CUDA `randn` for x0, both draw exactly one
`randn_like(action)` per env step, and both call `env.step` the same number of times in the same
order, so the noise streams stay in lockstep. They do.

This is the strongest form the gate could take and it closes the question §3 raised: the
wrapper's noise injection **is** `eval_act.py`'s. Any number the steering path produces from here
is comparable to the robustness sweep, and — because the streams align — comparisons between the
two harnesses may be treated as **paired**.

It also retroactively validates `slot_act/noise.py`: the shared helper was introduced precisely
so that `--action-noise 0.02` would denote the same perturbation in both, and the exact match is
the evidence that it does.

### 9b. What the gate cells say before the probe finishes

Three numbers, all later-cohort n = 96, all on `-v0` at seed 777:

| policy | clean | 2 % action noise | cost |
|---|---|---|---|
| stochastic x0 (fresh draw per refill) | 0.979 | 0.146 | **−83.3 pts** |
| `x0 = zeros` (held) | 0.938 | 0.167 | **−77.1 pts** |

Two things follow, and the first was a pre-registered belief:

* **`zeros` is a slightly *worse* draw than stochastic, clean** — 0.938 vs 0.979, −4.2 pts. It is
  the mode of the prior, and the mode is not the best sample. Consistent with the `-Tight-v0`
  x0-zeros cell (0.958).
* **Under noise the two are within sampling error of each other** (0.167 vs 0.146, z = +0.40,
  p = 0.69). Freezing x0 at the most average latent available neither helps nor hurts.

That second line is the first evidence bearing on §2 versus §8d, and it points at §8d: if the
deficit were mode collapse, *holding* x0 would at least change which mode is collapsed onto.
It changes nothing measurable. One draw out of five is not the spread, and the remaining four
are running — but belief 8 ("`zeros` is not the best draw") now has to carry the hypothesis.

### 9c. Interlude — how big *is* 2 % action noise, physically?

Computed 17:36 from the checkpoint's normaliser and the env config only. No §8b cell beyond
`zeros` and no §8d cell had reported. It is placed here rather than edited into §8d so the
pre-registration stays frozen.

`--action-noise 0.02` means 0.02 × each action channel's training std. From
`bc_armB_seed0/ckpt_final.pt`'s normaliser, the six arm channels have std
`[0.930, 0.832, 0.423, 0.602, 0.520, 0.856]`, mean **0.694**. The action manager applies
`scale = 0.5`, so the per-step joint-target perturbation is

    0.02 × 0.694 × 0.5 ≈ 0.0069 rad ≈ **0.40°**   (per channel range 0.24°–0.53°)

uncorrelated every control step, and the control step is `decimation 8 / sim.dt 1/400` = **50 Hz**
(600 steps × 0.02 s = the 12 s episode).

Two things this makes concrete, neither of them a result yet:

* **In joint-velocity units it is not obviously small.** If the drive tracked a target step
  within one control period, the induced velocity would be 0.0069 / 0.02 ≈ **0.35 rad/s**. The
  arm joints' `joint_vel_rel` training stds are `[0.544, 0.422, 0.240, 0.813, 0.318, 0.446]`.
  So the *ceiling* on the induced velocity jitter is order-1σ of the training distribution —
  the right order of magnitude for belief 9 to be true, and far above the 0.05 σ that
  `--obs-noise 0.05` puts on the same channel for 4.2 points of cost. A real drive will not
  track a step in one period, so the measurement could land anywhere below that ceiling; the
  point is that the hypothesis is not numerically absurd, which is what needed checking.
* **In task units it is comparable to the clearance.** 0.0069 rad at a reach of order 0.3 m is
  ~2 mm at the TCP, against a 1.5 mm per-side clearance on `-v0`. Multiple joints contribute and
  partially cancel, so treat this as order-of-magnitude only — but it means a **third**
  hypothesis is live alongside §2's mode collapse and §8d's distribution shift: the perturbation
  may simply be large in the units that matter for the task, in which case *no* policy-side fix
  helps and the correct statement is that the task is not doable at this actuation precision.

The third hypothesis is separable from the other two by evidence already in hand, and it does
not survive: `EXP_ROBUSTNESS.md` §13b measured the failing policy **frozen at x = 0.166 m** with
the block at carried height and `y = 0.0000` — 44 mm short of the fixture walls at 0.210
(`HANDOFF.md` §4c), in free space, never attempting the insertion. A perturbation too large for
the clearance would produce failures *at* the slot, not a refusal to approach it. Whatever is
happening is upstream of contact.

---

## 10. §10 PRE-REGISTRATION (17:40) — written with 3 of 5 probe cells in, before any §10 cell runs

The probe has already produced the largest single effect measured on this project, and it
changes what the next experiment has to be. Cells in hand, all `-v0`, seed 777, later cohort
(n = 96), all under `--action-noise 0.02` unless marked:

| x0 | later success |
|---|---|
| stochastic (fresh draw every 15-step refill) | **0.146** |
| `zeros`, held | **0.167** |
| `seed 1` (50×7 matrix), held | **0.771** |
| `seed 2` (50×7 matrix), held | **0.458** |
| — *clean references* — | stochastic **0.979**, zeros **0.938** |

Two random draws, committed for the whole run, score **+62** and **+31** points over the
stochastic policy that redraws 40 times an episode. Belief 6 (spread > 15 pts) and belief 7
(best > 0.30) are already held with three cells to spare; belief 8 (`zeros` is not the best) is
held.

### 10a. What this does and does not show

It shows the push-under-noise behaviour **is inside the base policy's support** — no training,
no gradient, just a different constant latent. That is the "sampling failure, not capability
failure" answer §2 was written to look for, and it arrives without PPO.

It does **not** yet show that x0-*steering* can get there, for two separate reasons:

1. **Shape.** `--fixed-x0 <seed>` is a **(50, 7)** matrix — a different x0 vector at each chunk
   position. `SteerCore.set_steer` broadcasts **one** 7-vector across all 50
   (`steer_core.py:58`). If the effect needs per-position structure, the steering action space
   does not contain the answer and PPO would be searching the wrong set.
2. **Norm.** This is the one I think matters more, and it reframes `zeros` entirely.

### 10b. The norm argument — why `zeros` is probably not "a bad mode"

x0 lives in **d = 50 × 7 = 350** dimensions. A standard normal in d = 350 concentrates on a thin
shell of radius **√350 = 18.71**; essentially no mass sits near the origin. So `x0 = zeros` is
**not "the average sample"** — it is far outside the typical set the flow was trained to
integrate from. Its poor score needs no story about modes at all.

Every random draw, by contrast, lands on the shell. And so does the **broadcast** form: one
7-vector of norm ≈ √7 repeated 50 times has total norm √7 · √50 = **18.71**, identical. So
`b<seed>` vs `<seed>` isolates *structure at matched norm* — exactly the comparison needed.

**And this is a problem for the pre-registered PPO run.** Steering sets
`x0 = alpha_x0 · tanh(z)` with `alpha_x0 = 1.0` and `clip_actions = 1.0`, broadcast across the
50 positions. Therefore

* **ceiling:** all |z_i| = 1 → ‖x0‖ = 1.0 · tanh(1) · √350 = **14.25** — the typical shell at
  18.71 is *unreachable*;
* **initialisation:** mu head zero-init, `sigma_init = -1.2` → σ ≈ 0.30, and tanh(z) ≈ z there,
  so ‖x0‖ ≈ 0.30 · 18.71 = **5.61**.

If success tracks ‖x0‖, the steering policy **starts near the `zeros` regime and cannot reach the
good one**, and 200 iterations of PPO would be spent inside the bad half of the space. That is a
design defect worth finding for the price of three eval cells rather than a training run.

### 10c. Cells (`scripts/run_x0_family.sh`, ~35 min)

| group | cells | question |
|---|---|---|
| structure | `b1 b2 b3 b4` | does the effect survive broadcasting? (the family steering can express) |
| population | full seeds `5 6 7 8` | how common is a good latent? |
| specificity | `seed1` clean, `seed1` @ 5 % noise | is seed 1 noise-robust, or just better? |
| **norm ladder** | `seed1 × {0.30, 0.76, 1.50}` @ 2 % | is ‖x0‖ the variable? 0.30 = steering at init, 0.76 = its ceiling, 1.50 = past the shell |

### 10d. Beliefs, pre-registered

12. **Broadcast survives: `b1` lands within 15 points of full `seed1`** (so > 0.62). Mechanism:
    matched norm, and there is no obvious reason a coherent 50-step plan needs the integration
    noise to vary *across* the plan. **Falsified → the steering action space is the wrong shape**
    and the PPO arms should not launch as specified.
13. **Good latents are common: at least 6 of the 8 full seeds score > 0.40.** Two of two already
    do. This is the belief that makes the result about *commitment* rather than about seed 1.
14. **`zeros` is bad because of its norm, not its direction.** `seed1 × 0.30` falls below 0.35 —
    into the `zeros` regime — while `seed1 × 1.50` stays above 0.60. The asymmetry is the point:
    if *both* tails collapse it is a shell effect; if only the small tail does, it is something
    about small-norm x0 specifically.
15. **Seed 1 is not simply a better latent.** Clean, it scores within ±5 points of stochastic's
    0.979 (so > 0.93). If instead it is clearly best clean too, the finding is much weaker.
16. **The effect survives at 5 % action noise but is not a rescue** — `seed1` @ 5 % beats
    stochastic's 0.000 but stays under 0.40.

Belief 14 is the one with teeth for the plan: if it holds, `alpha_x0 = 1.0` is simply the wrong
constant, and the PPO arms should be re-specified at `alpha_x0 ≈ 1.4` (ceiling 19.9, comfortably
containing the shell) with `sigma_init` chosen so initialisation lands *on* the shell rather than
at a fifth of it. That is a change I would rather make from three eval cells than discover from a
flat learning curve.

### 10e. Addendum (17:42), still before any §10 cell — seed 3 scored 0.031

Fourth probe cell in: **`seed 3` = 0.031**, *below* both the stochastic 0.146 and `zeros` 0.167.
The four constant latents now span **0.031 → 0.771**, a **74-point** range, and that changes two
things written above.

**Belief 13 is in trouble already.** It predicted ≥ 6 of 8 full seeds above 0.40; one of the
first four is at 0.031. Good latents are evidently *not* uniformly common, so "commit to
anything" is not the lesson.

**The commitment effect is still real, and now it is quantified differently.** The mean of the
three non-degenerate draws {0.771, 0.458, 0.031} is **0.42**, against **0.146** for the policy
that redraws every refill. So committing to a *random* latent beats redrawing by roughly 27
points on average, and choosing a good one adds another 35 on top. Two separable effects, and
the population cells (seeds 5–8) are what pin the first one down.

**And it introduces a selection problem that must be handled before any number is quoted.**
Reporting 0.771 as "what a good latent achieves" would be selecting the maximum of 5–9
candidates on the *same* 96 episodes and then quoting it. With a 74-point spread, some of that
maximum is the spawn set suiting the winner. `scripts/run_x0_holdout.sh` re-runs the argmax at
**two fresh spawn seeds (888, 999)** with `zeros` and stochastic re-measured alongside, so a
drop cannot be blamed on the new seeds being harder.

17. **Pre-registered:** the winning latent keeps most of its margin out of sample — **> 0.55 at
    both held-out seeds**. If it regresses toward the ~0.42 population mean, the honest headline
    is "committing beats redrawing", not "0.771".

---

## 11. RESULTS — the constant-x0 probe (complete, 2026-08-03 17:44)

Champion `bc_armB_seed0`, `-v0`, spawn seed 777, 128 ep / 32 envs, later cohort **n = 96**,
all cells under `--action-noise 0.02`. Read back by `analysis/steer_report.py`.

| x0 | later success | Wilson 95 % | dominant failure bucket | failed block median (x, z) | at staging (x < 0.18) |
|---|---|---|---|---|---|
| **stochastic** (redraw every refill) | **0.146** | [0.089, 0.230] | — | — | — |
| `seed 3` | **0.031** | [0.011, 0.088] | `never_entered` 90 | (0.165, 0.062) | **91/93** |
| `zeros` | **0.167** | [0.105, 0.254] | `never_entered` 76 | (0.167, 0.062) | 73/80 |
| `seed 2` | **0.458** | [0.362, 0.558] | `never_entered` 43 | (0.168, 0.062) | 36/52 |
| `seed 1` | **0.771** | [0.677, 0.844] | `never_entered` 12 | (0.186, 0.062) | 10/22 |
| `seed 4` | **0.823** | [0.735, 0.886] | `never_entered` 8 | (0.222, 0.062) | 5/17 |

*Clean references: stochastic **0.979**, `zeros` **0.938**.*

* spread **79.2 points** (0.031 → 0.823), best vs worst z = +11.09
* best constant vs stochastic **+67.7 points**, z = +9.39, **p = 6.2 × 10⁻²¹**
* beliefs **6** (spread > 15 pts), **7** (best > 0.30) and **8** (`zeros` is not the best) all HELD

### 11a. The headline

**The push-under-noise behaviour is inside the frozen policy's support.** No gradient, no
training, no reward — a different constant integration latent takes the champion from 0.146 to
**0.823** under the perturbation that was costing it 83 points. Against the clean ceiling of
0.979, a single well-chosen x0 recovers **81 % of the deficit**.

This is the answer §2 was written to look for, and it arrives *before* the PPO run rather than
from it:

> What looked like "the policy cannot do this under noise" is very largely **"the policy can do
> this and almost never samples it."**

### 11b. One mechanism, one knob — the failure signature is identical everywhere

Every cell fails the same way. The dominant bucket is `never_entered` in all five, and the
failed block sits at **z = 0.062** — carried height — in all five. That is `EXP_ROBUSTNESS`
§13b's freeze at the staging waypoint, reproduced at every latent.

What the latent changes is **how often the policy commits to the push**, and the fraction frozen
at staging is monotone in success across the whole 79-point range:

| x0 | success | frozen at staging |
|---|---|---|
| `seed 3` | 0.031 | 97.8 % of failures |
| `zeros` | 0.167 | 91.3 % |
| `seed 2` | 0.458 | 69.2 % |
| `seed 1` | 0.771 | 45.5 % |
| `seed 4` | 0.823 | 29.4 % |

And the failures that *are* left move down the corridor as the latent improves: median failed-x
goes 0.165 → 0.167 → 0.168 → 0.186 → **0.222**. Seed 4's residual failures are at the slot
**lip**, not at staging — a different, harder problem, and the one the clearance actually
governs. The good latent does not merely fail less; it fails **80 mm further along**.

### 11c. Two separable effects, and belief 13 is already wrong

Belief 13 said good latents would be common (≥ 6 of 8 above 0.40). `seed 3` at **0.031** is
*worse than redrawing*, so they are not. But the four random draws still average

    (0.771 + 0.458 + 0.031 + 0.823) / 4 = **0.521**   vs   **0.146** for redrawing

so there are two distinct effects, and both are large:

1. **Commitment** — holding *any* draw for the whole run is worth about **+37 points** on
   average over redrawing one every 15 steps. Mechanism: the chunk is a deterministic function
   of `(obs, x0)`, so a fresh x0 each refill stitches together plans drawn from different modes.
   Under noise the observation moves more between refills, and the successive chunks disagree
   more.
2. **Selection** — choosing a good draw rather than an average one is worth another **+30**.

> ⚠ **CORRECTED at 18:05 — the +37 above is a small-sample artefact.** It was computed on the
> first *four* draws, whose mean (0.521) happened to be dominated by two good ones. Seeds 5, 6
> and 7 came in at 0.177, 0.000 and 0.146, and the population mean fell to ~0.34. The corrected
> split is in **§13a**: commitment is worth roughly **+20**, and selection roughly **+48**. The
> original text is left in place because retracting it quietly would hide exactly the kind of
> error this project keeps finding — quoting a mean from n = 4.

`zeros` separates them cleanly: it is fully committed and still scores 0.167, so commitment
alone is not sufficient. §10b's norm argument predicts why, and the norm ladder tests it.

### 11d. What this does to the PPO plan

Belief 2 pre-registered "arm A recovers more than half the deficit — final success > 0.55". A
**constant** latent already clears that (0.823) with no training at all. So the bar for the
steering run has moved: PPO is no longer interesting for reaching 0.55; it is interesting only
if a *state-conditioned* steerer beats the best constant, i.e. if picking x0 as a function of
the observation is worth more than picking it once. That is a genuinely different and harder
question, and it needs the best constant as its control arm.

**Two things must land before that run**, both already queued:

* **The held-out check (§10e, belief 17).** 0.823 is the maximum of five candidates scored on
  the *same* 96 episodes. `scripts/run_x0_holdout.sh` re-measures the argmax at spawn seeds 888
  and 999 with `zeros` and stochastic alongside.
* **The norm ladder (§10b, belief 14).** If success tracks ‖x0‖, then `alpha_x0 = 1.0` caps
  steering at ‖x0‖ = 14.25 against a typical-set radius of 18.71, and initialisation sits at
  5.61 — the steerer would start in the `zeros` regime and never reach the `seed 4` one. That
  would be a defect in the pre-registered launch command, found for the price of three eval
  cells.

---

## 12. RESULTS — broadcasting kills it, and that ends the PPO run as pre-registered

*Written 17:52 with b1 and b2 in (b3, b4 running); the §12 conclusion is stated on two cells and
will be re-read when four are in.*

### 12a. Belief 12 falsified, and not marginally

| x0 | later success | dominant failure | median failed block (x, y, z) |
|---|---|---|---|
| `seed 1` — full (50, 7) matrix | **0.771** | `never_entered` 12 | (0.185, 0.000, 0.062) |
| `b1` — row 0 of that matrix, repeated 50× | **0.000** | **`gross_miss` 95** | (0.167, **−0.182**, 0.060) |
| `b2` | **0.000** | — | — |

`b1` vs `seed 1` is p = 5.2 × 10⁻²⁸ on paired episodes. Belief 12 predicted `b1` within 15 points
of `seed 1`; it is 77 points below, at the floor.

Note what `b1` is **not**. Its norm is √7 · √50 = **18.71**, sitting exactly on the typical shell
— the same norm as the full matrix, and the norm §10b argued was the important variable. Its row
is an ordinary draw (‖row‖ = 2.69 against √7 = 2.65). The *only* difference from the 0.771 cell
is that all 50 chunk positions share one x0 vector instead of fifty independent ones.

**So the variable is structure, not magnitude. Belief 14's norm story is wrong as the primary
explanation** — a typical-norm x0 can score zero. (The norm ladder still runs, because norm may
matter *within* the structured family, but it is no longer the headline.)

And the failure changes *kind*, which is the strongest part of the evidence. Every structured
latent fails by freezing at staging with the block dead on the slot axis (`y = 0.000`).
Broadcasting produces `gross_miss` in **95 of 96** episodes, with the block ending **182 mm off
axis**. A broadcast x0 does not make the policy timid — it makes the decoded chunk *wrong*, and
the arm carries the block somewhere else entirely. That is what one would expect if the flow
decoder relies on x0 varying across chunk positions to produce a temporally varied plan.

### 12b. Why this ends the pre-registered launch

`SteerCore.set_steer` is:

```python
x0 = self.alpha_x0 * torch.tanh(z)                                  # (N, 7)
self.controller.steer_x0 = x0.unsqueeze(1).expand(-1, chunk_size, -1)   # (N, 50, 7)
```

That `.expand` **is** the broadcast. The steering action space is precisely the `b`-family
scaled by ‖tanh(z)‖. So the reachable set is:

* `z = 0` → `x0 = zeros` → **0.167** (and this is where PPO initialises: mu zero-init, σ ≈ 0.30)
* `z ≠ 0` → the broadcast family → **0.000** at full scale

The good latents — 0.771 and 0.823 — are **not in the action space at all**. A PPO run would
start at 0.167, find every direction it can move is worse, and converge back to `z ≈ 0`. The
learning curve would be flat and the natural (wrong) conclusion would be "x0-steering does not
help on this task", when the truth is "x0-steering as parameterised cannot express the thing that
helps by 68 points".

`scripts/run_x0_bcast_ladder.sh` measures the middle of that path (`b1 × {0.15, 0.30, 0.60}`) so
the claim rests on a curve rather than on two endpoints. `k = 0.30` is exactly where the steering
policy initialises.

**Decision: the §6 launch command is withdrawn.** Not because steering is a bad idea — because
this parameterisation provably cannot reach the behaviour, and running it would burn hours to
produce an uninterpretable flat curve. `HANDOFF.md` §1-NEXT is updated accordingly.

### 12c. What replaces it

Three options, in increasing cost. The oracle result constrains all of them: a *perfect*
per-episode chooser over the five probed latents scores **0.885** against **0.823** for simply
always using seed 4 — so **the headroom for any conditional policy is about six points**, and
11 of 96 episodes are solved by no latent at all.

1. **Do nothing clever: ship the constant.** Search a few dozen structured latents on a
   validation spawn seed, keep the best, hold it. Costs one eval per candidate and no training.
   Pending the held-out check, that is 0.146 → 0.823 under 2 % actuation noise. **This is the
   deliverable unless the holdout says otherwise.**
2. **Structured steering** — give the steerer a per-position action, `z ∈ R^350`, or a
   low-dimensional mixture over K fixed structured draws with the norm renormalised back onto the
   shell. Contains the good latents by construction. Justified only if the oracle headroom is
   larger than six points with more candidates.
3. **Episode-level selection** — choose the latent once, from the initial observation, instead of
   every window. This matches the finding's shape (redrawing per refill is the harmful thing) and
   is a contextual-bandit problem rather than an RL one. Its ceiling is the oracle: **+6.2 pts**.

Option 1 is what the evidence supports. Options 2 and 3 are worth stating so the record shows
they were considered and priced, not overlooked.

### 12d. Confirmed on all four broadcast cells (17:56) — and the failure gets worse than §12a said

| broadcast x0 | later | dominant buckets | median failed block (x, y, z) |
|---|---|---|---|
| `b1` | **0.000** | `gross_miss` 95, `never_lifted` 1 | (0.167, −0.182, 0.060) |
| `b2` | **0.000** | `gross_miss` 56, `never_entered` 29 | (0.162, −0.082, 0.048) |
| `b3` | **0.000** | **`never_lifted` 90**, `gross_miss` 5 | (0.221, −0.098, 0.012) |
| `b4` | **0.000** | **`never_lifted` 96** | (0.216, −0.123, 0.032) |

Four for four at the floor. Belief 12 is falsified on every cell, not on a lucky one.

**`never_lifted` is the important word.** The taxonomy documents it as the bucket that should be
*empty* on this task — the scripted expert grasps 2038/2038, and *"if it is ever non-empty for a
trained policy, that is a failure mode the demonstrator never had"*. Under `b4` it is **96 of
96**: the policy never picks the block up at all. Broadcasting does not degrade the insertion; it
destroys the behaviour from the first phase onward.

**And `zeros` shows the mechanism precisely.** `zeros` is *also* constant across all 50 chunk
positions, and it scores 0.167 with an ordinary staging freeze — no `never_lifted`, no
`gross_miss`. So "constant across positions" is not what breaks the policy. **"Constant and
non-zero" is.**

That points at a simple explanation. The flow decodes x0 into a 50-step action trajectory; a
value that is identical at every chunk position contributes no *temporal* variation, only a
uniform offset — and a uniform offset on a sequence of absolute joint-position targets is
approximately a **fixed joint bias**. The arm is commanded to a systematically displaced pose for
the whole chunk, so it reaches past the block (`never_lifted`) or carries it to the wrong place
(`gross_miss`, −0.18 m off axis). Scaling that offset toward zero should walk the behaviour back
to the `zeros` cell, which is exactly what `scripts/run_x0_bcast_ladder.sh` measures.

**Consequence for the withdrawn launch, sharpened.** Steering's `x0 = alpha·tanh(z)` broadcast is
this offset, with `‖tanh(z)‖` as its size. The steerer starts at `z = 0` (offset zero, the
`zeros` cell) and every direction it can explore adds a uniform joint bias. It is not merely that
the good latents are outside the action space — the action space is *a bias term*, and the task
has no use for one.

### 8d-bis. The causal cell, added 17:59 (before any §8d cell ran)

§8d as pre-registered can only establish a **correlation**: that some observation channel goes
out of distribution when action noise is applied. The constant-x0 probe now makes a *causal*
version available for the price of one extra cell.

`shift_act002_s4` runs the **same** 2 % action noise with `--fixed-x0 4` — identical physics,
identical perturbation, but the latent that scores **0.823** instead of **0.146**. Then:

* If `joint_vel` is just as far out of distribution under seed 4 as under the stochastic policy,
  the shift is a **consequence** of the noise and not the reason the policy freezes. Belief 9
  would survive as a measurement and die as an explanation.
* If the good latent keeps `joint_vel` near nominal, then the two session-7 findings are one
  finding: the good latent works *by* keeping the policy's own input in distribution.

I do not have a confident prior between those, which is exactly why the cell is worth running.
Pre-registering the fork rather than the answer.

---

## 13. RESULTS — the latent population (8 random draws, complete 18:07)

All `-v0`, seed 777, 128 ep / 32 envs, later cohort n = 96, `--action-noise 0.02`, and — verified,
not assumed — **spawn-identical across every cell**, so these are paired comparisons.

| x0 | 6 | 3 | 8 | stochastic | 7 | zeros | 5 | 2 | 1 | **4** |
|---|---|---|---|---|---|---|---|---|---|---|
| later | 0.000 | 0.031 | 0.062 | *0.146* | 0.146 | 0.167 | 0.177 | 0.458 | 0.771 | **0.823** |

Random draws only (n = 8): **mean 0.309, median 0.161, max 0.823**. Only **3 of 8** clear 0.40.

**Belief 13 is falsified decisively** (it predicted ≥ 6 of 8 above 0.40). Good latents are rare,
not common.

### 13a. The corrected decomposition — selection, not commitment, is where the points are

This replaces the +37/+30 split in §11c, which was computed on n = 4 and was wrong.

The right null for "does *committing* help?" is the **mean over random draws**, because redrawing
every refill is, in expectation, picking a random latent:

| effect | size | comparison |
|---|---|---|
| **commitment** | **+16.3 pts** | mean of random draws **0.309** − stochastic **0.146** |
| **selection** | **+51.4 pts** | best draw **0.823** − mean of random draws **0.309** |

So commitment is real but modest; **the overwhelming majority of the 68-point gain is choosing
the right latent.** Note also how close the stochastic 0.146 is to the *median* draw (0.161): had
I compared against the median I would have concluded commitment buys nothing at all. Both
comparisons belong in the record, and the mean is the correct one for this question.

### 13b. The oracle does not move when you add four more latents — and that settles options 2 and 3

| candidate set | oracle (≥ 1 succeeds) | best single | headroom |
|---|---|---|---|
| 5 latents (`zeros`, 1–4) | **0.885** | 0.823 | +6.2 pts |
| **9 latents** (`zeros`, 1–8) | **0.885** | 0.823 | **+6.2 pts** |

Adding seeds 5–8 — four more draws, at 0.177, 0.000, 0.146, 0.062 — solved **not one new
episode**. The same **11 of 96** episodes are failed by every latent tried, and the best *pair*
(seeds 1 + 4) reaches only 0.865.

This is the strongest constraint the session has produced on what to build next:

* **Option 3 (episode-level latent selection) is capped at +6.2 points**, and that cap did not
  loosen when the candidate pool nearly doubled. A contextual bandit over latents is not worth
  building.
* **Option 2 (structured steering with a richer action space)** inherits the same ceiling unless
  it can reach latents *qualitatively unlike* these nine — which it has no particular reason to,
  since all nine are ordinary draws from the same prior the flow was trained on.
* **Option 1 (search a few dozen, keep the best) is what the evidence supports**, and 8 draws
  were enough to find a latent within 6 points of the 9-latent oracle.

The 11 unsolvable episodes are the honest residual. They are not a latent problem, and nothing in
this experiment addresses them.

### 13c. A good latent cannot be recognised without running it

`analysis/latent_stats.py` correlates each latent's measured score against the cheap summary
statistics one would use to screen candidates offline: total norm, the DC component (the mean
across chunk positions — the part broadcasting keeps), the AC spread, and the gripper column's
DC. Seeds 1 (0.771) and 3 (0.031) are near-identical on all four.

With n = 8 the rank correlations are still too weak to act on, and — this is the point — even a
strong correlation at n = 8 would not be actionable. **Latent screening has to be empirical**,
which is precisely why the deliverable is "evaluate a few dozen on a validation spawn seed and
keep the winner" rather than anything cleverer. That is a cheap search: 8 draws found a latent
6 points from the oracle over all 9.
