# EXP_ROBUSTNESS — does the policy generalise, or has it memorised one geometry?

*Opened 2026-08-03, session 5, **before any perturbed evaluation has been run**. Prompted by
Big Will asking whether the environment is tested with "more nuanced scenarios, like sliding the
slot a little further away". It is not, and that turns out to be the sharpest available test of
what the policy actually learned.*

## Status

| round | what it perturbs | status |
|---|---|---|
| 1 (§1–§5, results §6) | slot dx/dy, spawn box | ✅ **COMPLETE, 9/9 cells** |
| 2 (§7, results §7f–§7h) | dy ladder × clearance, arm start pose, sensor noise, combo | ✅ **COMPLETE, 12/12 cells** |
| 3 (§9, results §9b–§9e) | replication of every claim on a 2nd training seed | ✅ **COMPLETE, 6/6 cells** |
| expert control (§8, results §8b–§8d) | is the dx dip the robot's or the policy's? | ✅ **COMPLETE, 4/4 cells** |
| actuation noise (§10, results §13) | `--action-noise`, the counterpart to sensor noise | ✅ **COMPLETE, 3/3 cells** |
| dy × clearance (§12) | the ladder crossed with all three clearances | ✅ **COMPLETE, 12 cells** |
| spawn yaw (§11) | resolving §6e's unattributed in-box drop | ✅ **COMPLETE, 2/2 cells** |

Every round's beliefs are written down before its cells exist. Where a belief leans on an earlier
round's result, the section says so and names the cell — see §7b, which records a round-1 belief
this experiment had already falsified before round 2 was designed.

**Headline over 21 cells.** The policy is far less brittle than a "BC memorises its training
distribution" prior predicts, with exactly one hard wall:

* **Slot x: tracks it.** ±10 mm free; the block's *absolute* final position follows the slot 1:1.
  +20 mm costs ~8 points, replicated on two training seeds (§9b). The −10 mm dip did **not**
  replicate and is retracted as a general claim (§9d).
* **Arm start pose: free.** ±0.1 rad costs 2.1 points (p = 0.41) despite the env pinning the arm
  to one pose in all 2038 demos. Zero training coverage, no measurable cost.
* **Sensor noise: free at 5 %, fatal at 20 %. Actuator noise: fatal at 5 %, and 83 points at
  *2 %*.** The same nominal magnitude is free on the observation and lethal on the action (§13) —
  chunked control low-passes an observation error over 15 steps and does nothing at all about an
  action error.
* **Under noise the policy does not wander — it FREEZES at the staging waypoint.** Every moderate
  noise cell, sensing or actuation, ends with the block held at x = 0.166 m (the expert's
  `stage_x` is 0.165), y = **0.0000**, at carried height, in 81–92 of 96 episodes (§13b). Three
  of four phases completed to nominal precision, and the push never attempted.
* **Spawn box: graceful, and in-distribution episodes are untouched.** ×1.5 costs 15.6 points and
  ×2.0 costs 42.7, but split on position *and* yaw, the fully in-distribution cohort scores
  **43/43 = 1.000** across both cells (§11b). Every point of loss is out-of-distribution, and
  **yaw is the more damaging axis** of the two.
* **Slot y: a step function at the clearance — geometry, not the policy.** Twelve cells across
  three clearances collapse onto a single curve in **(shift / clearance)** with no exceptions
  (§12): ≤ 0.67 is free, 1.00 gives 0.760, ≥ 1.33 is a floor. The same **2 mm** shift scores
  0.000 on `-v0` and **0.938** on `-Loose-v0`; the same **1 mm** shift scores 0.948 on `-v0` and
  **0.021** on `-Tight-v0`. Same checkpoint, same spawns, opposite outcomes.
* **Perturbations compose independently**, to within 0.9 points (§7h, belief 8).
* **The arm is not the limit.** A scripted planner told the new slot centre scores 0.977–1.000 at
  every dx the policy was tested at (§8b), so the +20 mm deficit is the policy's and has a
  demonstrated ceiling above it.

Two of my five round-1/2 mechanism predictions were falsified in the same direction: I kept
assuming parts of this policy were open-loop, and they are not. A third (belief 12, the expert
dipping at −10 mm) fell the same way.

**The one concrete improvement target this produced:** dx = +20 mm costs ~8 points, replicates
across training seeds, and the robot can do it perfectly. Everything else here is either free,
already at the geometric limit, or a checkpoint idiosyncrasy.

---

## 1. What the environment actually randomises (and what it does not)

Read out of `precision_slot_env_cfg.py`, not remembered:

| quantity | randomised? | range |
|---|---|---|
| block spawn pose | **yes, per episode** | x ±0.02 m, y ±0.03 m, yaw ±0.35 rad (±20°) |
| block friction | **startup only**, not per episode | static 0.9–1.1, dynamic 0.75–0.95, 16 buckets |
| slot position | **no** | welded at `SLOT_CENTER = (0.245, 0.0)` |
| arm start pose | **no** | fixed `_START_POSE` |
| block size / mass | **no** | — |
| observation noise | **no** | `enable_corruption = False` |

The three task variants (`-Loose-v0` / `-v0` / `-Tight-v0`) differ in **`clearance` alone**
(3.0 / 1.5 / 0.5 mm). Nothing else.

So Stage C's result — 0.708–0.799 pooled across all three clearances — is measured over exactly
one axis of variation: a ±20 mm × ±30 mm × ±20° box around one block spawn, into one slot, from
one arm pose. That is genuinely "random starts", and it is also thin.

## 2. Why the slot position is the right thing to perturb

The policy's observation carries the slot-relative error directly — `obs[23:27]` is
`slot_frame = (insertion_depth, lateral_error, yaw_error, is_inserted)`, all computed against
`SLOT_CENTER`. So a moved slot is **visible** to the policy. The question is whether it *uses*
those channels or has memorised a trajectory that happens to end at x = 0.2545.

This subsumes the open clock-vs-closed-loop question (`HANDOFF.md` §3h,
`scripts/diag_feedback.py`, still never run) and asks it more directly: `diag_feedback` moves the
*block* mid-episode; this moves the *goal*, which the policy must track for the entire episode.

**And there is an asymmetry that makes the two axes different experiments**, visible in
`mdp/common.py`:

```
def lateral_error(...):  return (object_pos_local(env, name)[:, 1] - SLOT_CENTER[1]).abs()
```

`lateral_error` and `yaw_error` are **absolute values** — the policy is told *how far* off it is,
never *which way*. `insertion_depth` is not: it is monotone in x. Hence:

* **x-shift**: a usable signed signal exists. Adaptation is *possible*.
* **y-shift**: the policy cannot distinguish +30 mm from −30 mm from that channel. It would have
  to infer direction from the raw block pose (`obs[16:23]`, in the robot root frame) against a
  slot position it has never seen move. Adaptation is close to *structurally impossible*.

## 3. Beliefs, pre-registered

1. **Nominal (dx = dy = 0) reproduces the sweep cell exactly.** The perturbation code path must
   be inert at zero. If it is not, everything below is void — this is the gate.
2. **y-shift collapses hard and fast.** At |dy| = 10 mm — one third of the spawn box's own y
   range, so not a large number in absolute terms — success falls below 0.30. Mechanism: the
   only lateral channel is unsigned, so the policy has no way to know which way to correct.
3. **x-shift degrades more gracefully than y.** At |dx| = 10 mm, success stays above the
   corresponding y-shift by at least 20 points. Mechanism: `insertion_depth` gives a signed,
   monotone error signal on that axis.
4. **x-shift is still not *solved*.** Even at dx = +10 mm success drops below 0.50, because
   §6d-ii of `SESSION5_FINDINGS.md` showed the binding failure is already **depth** — the block
   stopping ~40 mm short — and moving the slot further away makes exactly that worse. A negative
   dx (slot *closer*) may therefore be **easier than nominal**, which would be the cleanest
   possible confirmation that depth is the constraint.
5. **Widening the spawn box degrades gently.** At `--spawn-scale 2.0` (x ±40 mm, y ±60 mm, yaw
   ±40°) success stays above 0.50. Mechanism: the grasp phase is the part the policy does best
   (zero `never_lifted` failures at `ckpt_final`), and a wider spawn mostly stresses the grasp,
   not the insertion.

Belief 4's second half is the one I most expect to be wrong, and it is the most informative if
it is right.

## 4. Design

`slot_act/eval_act.py` gained `--slot-dx`, `--slot-dy`, `--spawn-scale`. No file in `eva_rl` is
edited: `SLOT_CENTER` is patched in-process before `parse_env_cfg`, across **all three** bindings
that hold it (`mdp.common`, `mdp` — a separate binding created by `from .common import *` — and
`slot_mdp`). All three are asserted after patching.

**The trap this guards against**, stated because it would have produced a beautiful wrong
number: patching only `mdp.common` moves the *success predicate* but not the *walls*, because
the env cfg reads `mdp.SLOT_CENTER`. Every insertion would then score as a failure and the run
would look like a clean, publishable collapse. So after `env.reset()` the harness reads the
actual **floor prim's world position** back out of the scene and asserts it matches the shifted
centre, and says so in the log.

`slot_dx` / `slot_dy` / `spawn_scale` are recorded into the results JSON `config` block, for the
same reason `fixed_x0` is: a perturbed run must not be byte-indistinguishable from a nominal one.

Champion checkpoint (`bc_armB_seed0/ckpt_final.pt`), `-v0`, spawn seed 777, 128 episodes /
32 envs, so every cell is directly comparable to the sweep's `-v0` s777 cell (0.969).

| condition | flag |
|---|---|
| nominal (gate) | — |
| slot 5 / 10 / 20 mm further | `--slot-dx 0.005 / 0.010 / 0.020` |
| slot 10 mm nearer | `--slot-dx -0.010` |
| slot 5 / 10 mm sideways | `--slot-dy 0.005 / 0.010` |
| spawn box ×1.5, ×2.0 | `--spawn-scale 1.5 / 2.0` |

~3 min each, ~30 min total, one GPU job at a time.

## 5. Decision rule

Comparisons are against the sweep's own `-v0` s777 cell for the same checkpoint. **Unpaired** —
any perturbation changes the reset draw, exactly as `--fixed-x0` did (`EXP_TIGHT.md` §7c) — so
two-proportion z-tests and Wilson intervals, never McNemar.

* **Belief 1 fails** → stop; the harness is wrong and nothing else is interpretable.
* **y collapses while x degrades gently** → the policy is using the slot-frame channels but is
  blind to lateral *sign*, which is a property of the **observation design**, not of the policy.
  That is a concrete, actionable finding: a signed lateral channel is a one-line env change
  (which I may not make — `eva_rl` is shared) or a feature the steering obs could supply.
* **Both collapse at 5 mm** → the policy has memorised the geometry and the Stage C number
  should be reported as "in-distribution only". This is the outcome that most changes how the
  result should be described.
* **Both degrade gently** → the policy genuinely tracks the goal, and the Stage C number is
  more robust than its single-axis protocol can demonstrate.
* **Single-seed caution applies** (`PLAN` 5.28): a margin under ~10 points on one checkpoint at
  one spawn seed is not evidence. Replicate on a second run before claiming a mechanism.

---

## 7. Round 2 — the axes the environment does not randomise at all

*Pre-registered 2026-08-03 ~09:15. Five round-1 cells had landed (`gate_nominal`, `dx_p005`,
`dx_p010`, `dx_p020`, `dx_m010`); four had not (`dy_p005`, `dy_p010`, `spawn15`, `spawn20`).
So the dx results below were visible when these beliefs were written and the dy and spawn results
were not — which is exactly why §7b is about the dx axis and §7d had to be added later.*

Round 1 perturbs the **goal** (slot dx/dy) and the **spawn box**. Both are things the policy can
in principle see: the slot-frame channels move, or the block starts somewhere it has started
before. Round 2 moves what the challenge env holds perfectly fixed across all 2038 demonstrations
— the parts of the world where the training distribution is a **point mass**, not a distribution:

| axis | what the env does | round-2 flag |
|---|---|---|
| arm start pose | `init_state.joint_pos = dict(_START_POSE)`, no reset event | `--arm-jitter 0.05 / 0.10` |
| observation | `enable_corruption = False`, no noise models | `--obs-noise 0.05 / 0.20` |
| slot −y | round 1 only tests +y | `--slot-dy -0.010` |
| all at once | — | `--slot-dx 0.010 --slot-dy 0.005 --spawn-scale 1.5 --arm-jitter 0.05` |

`--arm-jitter` is the sharpest of these. **pick_place randomises exactly this**, ±0.1 rad on
joints 1–6 (`pick_place_v1_env_cfg.py:137`), with the comment "planner-expert is start-agnostic;
RL's fixed start pose was a crutch." The challenge env did not inherit that term. So a policy
trained here has seen precisely **one** arm start pose, ever, and 0.10 rad is not an arbitrary
number — it is the value a sibling task in the same repo already considers reasonable.

`--obs-noise` is expressed as a fraction of **each channel's own training standard deviation**,
taken from the checkpoint's normaliser, so one scalar means the same thing for a joint angle in
radians and a quaternion component. It is applied to `obs[0:27]` only: `obs[27:34]` is the
policy's own last commanded action, which is internal state rather than a measurement, and
corrupting it would be testing a different thing (whether the policy can survive lying to itself
about what it just did).

### 7a. Beliefs, pre-registered

6. **Arm jitter at 0.05 rad costs less than 15 points; at 0.10 rad it costs more than 30.**
   Mechanism: the first ~150 steps of every demo are an approach from a fixed pose, so the early
   chunk is the most memorisable segment in the dataset, and it is also the segment with the
   least feedback available (the block is far away and the gripper is empty). I expect the
   *grasp* to break before the *insertion* does — i.e. `never_lifted`, a bucket that has been
   **empty at `ckpt_final` in every cell measured so far**, should become non-empty. If arm
   jitter degrades success but `never_lifted` stays at zero, this mechanism is wrong.
7. **Observation noise at 0.05 is nearly free; at 0.20 it costs more than 25 points.** Mechanism:
   chunked control already low-passes the observation — the policy re-reads it only once every
   15 steps — so per-step noise is attenuated ~4× relative to a per-step policy. The failure
   should appear as *lateral* degradation, not depth, because the noise perturbs where the policy
   thinks the slot is.
8. **The combo cell scores at or above the product of its parts.** If the single-axis retention
   rates are r_dx, r_dy, r_spawn, r_arm relative to the gate, an *independent-failures* model
   predicts the combo at their product. I expect it to be **higher** than that product, because
   the dominant failure mode is shared (depth) rather than independent. A combo cell *below* the
   product would mean the perturbations interact destructively, which is the outcome that would
   most change how the headline should be qualified.
9. **±10 mm in y scores the same within noise.** `lateral_error` is unsigned, so the two are
   indistinguishable in that channel. A significant asymmetry between `dy_p010` and `dy_m010`
   would prove the policy reads direction from the **raw block pose** (`obs[16:23]`, in the robot
   root frame) rather than from the slot-frame channels — a mechanism claim I would otherwise
   have no way to test.

### 7b. What round 1 has already falsified, recorded before round 2 runs

Belief 4 said "even at dx = +10 mm success drops below 0.50". Measured: `dx_p005` = 0.990,
`dx_p010` = **1.000**, against a gate of 0.979 — moving the slot *further away* was, if anything,
better than nominal. I was wrong about the direction as well as the magnitude, and the reasoning
that produced it ("depth is the constraint, so a further slot makes the constraint worse") is
therefore suspect on its own terms. §6b works through what the right explanation appears to be.

This matters for round 2 because belief 6's mechanism ("the early, fixed-start segment is the
memorised part") is the *same shape* of argument as the one that just failed: both assume the
policy is running something open-loop. Round 1 says it is not, on at least one axis. So belief 6
is stated with lower confidence than it would have been an hour ago, and its named falsifier
(`never_lifted` stays empty) is the part to watch.

### 7c. Which test, decided by the data

Round 1's read-back discovered something the pre-registration got wrong in the safe direction:
shifting `SLOT_CENTER` consumes no RNG, so the spawn stream is **untouched** and every dx/dy cell
is spawn-for-spawn **paired** with the gate. `analysis/robustness_report.py` verifies this from
the recorded `spawn_pos` rather than assuming it, and uses exact-binomial McNemar when the spawns
match and a two-proportion z-test when they do not. `--spawn-scale` provably breaks pairing (it
changes the sampling range); `--obs-noise` breaks it too (`torch.randn_like` consumes the same
generator the reset events draw from). `--arm-jitter` adds a reset event, which also draws. So
the arm/noise/spawn cells are unpaired and the dx/dy cells are paired, and the report says which
it used for each row.

### 7d. Cells added ~09:20, after `dy_p005` landed at **0/96**

*Added before any round-2 cell had been run. `dy_p005` is a round-1 cell; §6c works through it.*

The dy collapse is total — 0/96, not "degraded" — and the failure signature is not the one the
observation-design argument predicts. If the policy were simply aiming at the remembered centre,
the block would end at y ≈ 0 with |lateral| ≈ 5 mm. Measured instead: final y median **−10 mm**,
p10 **−54 mm**, final x median **145 mm** (nominal: 257 mm), and `max_obj_z` median **75 mm**
against nominal 67 mm. The block is lifted *higher* than normal and ends up back near the staging
area. That is a **collision**, not a near miss.

The geometry says it must be. The block is 30 mm wide (`BLOCK_HALF[1] = 0.0150`) and the mouth is
`2 × (15 + clearance)` mm. For the block to enter at all, its centre must be within ±clearance of
the slot centre — **1.5 mm on `-v0`**. The policy drives the block to y = 0.0 ± 1.0 mm (measured,
gate cell). So a 5 mm shift does not make entry hard, it makes entry *geometrically impossible*,
and what the 96 episodes record is the aftermath of driving a block into a wall face.

That yields a prediction no "the policy is fragile in y" story can produce:

10. **The dy tolerance is the clearance, and nothing else.** `dy_p001` (1 mm < 1.5 mm) stays above
    0.70. `dy_p002` degrades sharply. `dy_p003` (3 mm > 1.5 mm) is at or near zero. And the
    crossed cell is the decisive one: **`loose_dy_p003` — the same 3 mm shift on `-Loose-v0`,
    where the clearance is 3.0 mm — stays above 0.50**, while `dy_p003` on `-v0` is near zero.
    Same policy, same shift, opposite outcome, decided purely by a number in the env cfg.
    `loose_dy_p005` (5 mm > 3.0 mm) should then collapse in turn.

If belief 10 holds, the honest description of the dy result changes completely: it is not "the
policy cannot handle a moved slot", it is "**the task's lateral tolerance is the clearance, the
policy already saturates it, and the slot's lateral position is not in the observation with a
sign so nothing could do better**". If `loose_dy_p003` also collapses, the geometric model is
wrong and the fragility is the policy's after all.

**This is also the cheapest available test of the unsigned-channel argument in §2**, which I have
been asserting since the pre-registration without evidence. A policy that could see signed
lateral error would re-centre and enter regardless of clearance; one that cannot is capped at
exactly ±clearance. Belief 10 is what distinguishes them.

---

## 6. Round 1 results — all nine cells

`runs/bc_armB_seed0/robust_*.json`, read with `analysis/robustness_report.py`. Later cohort only
(96 of 128 episodes; the first episode each env runs carries the PhysX warm-start bias).

| cell | dx mm | dy mm | spawn | success | 95 % CI | Δ vs gate | test |
|---|---|---|---|---|---|---|---|
| `gate_nominal` | 0 | 0 | ×1.0 | **94/96 = 0.979** | [0.927, 0.994] | — | reference |
| `dx_m010` | −10 | 0 | ×1.0 | 81/96 = 0.844 | [0.758, 0.903] | −0.135 | McNemar b=14 c=1, **p = 0.00098** |
| `dx_p005` | +5 | 0 | ×1.0 | 95/96 = 0.990 | [0.943, 0.998] | +0.010 | McNemar b=0 c=1, p = 1.0 |
| `dx_p010` | +10 | 0 | ×1.0 | **96/96 = 1.000** | [0.962, 1.000] | +0.021 | McNemar b=0 c=2, p = 0.50 |
| `dx_p020` | +20 | 0 | ×1.0 | 85/96 = 0.885 | [0.806, 0.935] | −0.094 | McNemar b=10 c=1, **p = 0.0117** |
| `dy_p005` | 0 | +5 | ×1.0 | **0/96 = 0.000** | [0.000, 0.038] | −0.979 | McNemar b=94 c=0, **p = 1.0e-28** |
| `dy_p010` | 0 | +10 | ×1.0 | **0/96 = 0.000** | [0.000, 0.038] | −0.979 | McNemar b=94 c=0, **p = 1.0e-28** |
| `spawn15` | 0 | 0 | ×1.5 | 79/96 = 0.823 | [0.735, 0.886] | −0.156 | z = −3.63, **p = 0.00029** |
| `spawn20` | 0 | 0 | ×2.0 | 53/96 = 0.552 | [0.453, 0.648] | −0.427 | z = −6.99, **p = 2.8e-12** |

### 6a. Scoreboard against the pre-registration

| belief | outcome |
|---|---|
| 1. nominal reproduces the sweep cell | ✅ **94/96, bit-identical** to the sweep's own `-v0` s777 cell |
| 2. y collapses below 0.30 at 10 mm | ✅ **confirmed, and far harder than predicted** — 0.000 at *5* mm |
| 3. x beats y by ≥ 20 points at 10 mm | ✅ **confirmed by 84–100 points**, not 20 |
| 4. x drops below 0.50 at +10 mm; −dx easier | ❌ **falsified in both halves** — +10 mm scored 1.000; −10 mm was the *worse* direction |
| 5. spawn ×2.0 stays above 0.50 | ✅ confirmed, narrowly: 0.552 [0.453, 0.648] |

Belief 4 is the interesting failure, and it is worth being precise about *how* it was wrong. The
reasoning was: "depth is the binding constraint, so moving the slot further away makes the binding
constraint worse, and moving it nearer should help." Both halves came out backwards. The premise
— that failures are depth failures — is still true; what was wrong is treating "depth" as a
budget the arm spends, when it is really a **geometric window the arm has to land in**.

### 6b. The x axis: the policy tracks the goal, 1:1, over ±20 mm

The strongest evidence is not the success rate at all — it is where the block ends up. Median
final block x, over successful episodes:

| dx | median final block x | offset from nominal |
|---|---|---|
| 0 | 0.2570 | — |
| +5 mm | 0.2620 | **+5.0 mm** |
| +10 mm | 0.2670 | **+10.0 mm** |

Exactly 1:1, to the recording precision. And the median *insertion depth* of successes is
47.3 / 47.4 / 47.3 mm at dx = 0 / +5 / +10 — identical, because the block bottoms out against the
slot's back stop, which moved with everything else. **The policy is not replaying a trajectory
that happens to end at x = 0.2545.** It follows the goal.

This settles the clock-vs-closed-loop question (`HANDOFF.md` §3h) more directly than
`diag_feedback.py` was going to: `diag_feedback` moves the block for one instant, this moves the
goal for the whole episode, and the policy tracks it 20 mm past anything in its training data.

The *rate* response is flat-then-degrading rather than a peak:

* dx = +5 and +10 are **statistically indistinguishable from nominal** (p = 1.0 and p = 0.50).
  `dx_p010`'s 96/96 is two episodes better than the gate. It is not a "+10 mm is the sweet spot"
  finding and must not be written up as one.
* dx = −10 costs 13.5 points (p = 0.001) and dx = +20 costs 9.4 points (p = 0.012). Both real.

So: **insensitive from 0 to +10 mm, ~10 points down at −10 mm and +20 mm.** The asymmetry is
genuine — pulling the slot *toward* the robot hurts more than pushing it away — and the failure
mix says why it is not a tracking failure: at every dx the failures are the *same* failure,
`stalled_in_mouth` at a median depth of 36–38 mm against a 40 mm bar. Nothing misses the slot.
The policy gets the block into the hole and stops a few millimetres short.

### 6c. The y axis: 0/96 at 5 mm, and the reason is geometry, not control

Both y cells are floor-zero: not one success in 192 episodes, against 94/96 at the same spawns.
McNemar b = 94, c = 0.

The pre-registered mechanism was the unsigned `lateral_error` channel. That is probably true, but
it is **not what these 192 episodes show**, and the difference matters. If the policy were merely
aiming at the remembered centre, the block would end at y ≈ 0 with a 5 mm lateral error. Measured:

| | gate | `dy_p005` |
|---|---|---|
| final block y, median | −0.0 mm | **−10 mm** |
| final block y, p10 | −1.0 mm | **−54 mm** |
| final block x, median | 257 mm | **145 mm** |
| `max_obj_z`, median | 67 mm | **75 mm** |

The block ends up *behind* where it started its approach, off-axis, having ridden 8 mm higher
than a carried block ever does. That is the signature of driving a block into a wall face, not of
a near miss. The taxonomy agrees: 49 `gross_miss` + 47 `never_entered` at dy = +5, and 73/23 at
dy = +10.

And the geometry says it is forced. The block is 30 mm wide (`BLOCK_HALF[1] = 0.0150`); the mouth
is `2 × (15 + clearance)` mm. Entry requires the block centre within **±clearance = ±1.5 mm** of
the slot centre on `-v0`. The policy delivers y = 0.0 ± 1.0 mm — it is already using most of that
window. A 5 mm shift does not make entry hard, it makes entry **impossible**.

That reframes the finding. "The policy cannot handle a moved slot" would be wrong. The accurate
statement is: **the lateral tolerance of this task is the clearance; the policy already saturates
it; and the slot's lateral position is not in the observation with a sign, so no memoryless
policy trained on this observation could do better.** §7d turns that into a falsifiable
prediction (the clearance-crossed dy ladder) rather than leaving it as an argument.

### 6d. Why the x and y axes come out so differently — the one-sentence version

`insertion_depth` is **signed and monotone in x**; `lateral_error` and `yaw_error` are
**absolute values**. The policy tracks the goal on exactly the axis where the observation makes
the goal observable, and fails completely on the axis where it does not. Every number in §6b and
§6c is consistent with a policy that is using all of the goal information it is given, and none
of it with a policy that memorised a path.

This is a finding about the **observation design**, not about flow matching, not about the amount
of data, and not about the training seed. It is also the one place in this project where a
one-line change to a shared file (`lateral_error` returning a signed value) would plausibly move
a result by 90 points — and I have not made it, because `eva_rl` is shared and not mine to edit.

### 6e. The spawn box: degradation is concentrated where the box is new

Splitting each widened-spawn cell by whether the block's spawn **position** fell inside the
nominal box (x ±19.7 mm, y ±28.4 mm, read back from the gate cell's own recorded spawns):

| cell | inside nominal box | outside |
|---|---|---|
| `spawn15` | 38/41 = 0.927 [0.806, 0.975] — vs gate p = 0.13 | 41/55 = 0.745, **p = 7e-06** |
| `spawn20` | 18/23 = 0.783 [0.581, 0.903] — vs gate p = 0.0003 | 35/73 = 0.479, **p = 4e-14** |

At ×1.5 this is exactly what generalisation looks like: in-distribution spawns are unaffected
(0.927 vs 0.979, not significant), and the whole 15.6-point loss comes from the newly-added
region.

**At ×2.0 the in-box cohort also drops, and I cannot fully attribute that from this data.** The
honest reason is an instrument gap: `--spawn-scale` widens the **yaw** range too (±0.35 → ±0.70
rad, i.e. ±40°), and `spawn_pos` records only position — so the "inside the box" cohort is inside
in x and y while containing yaws that are up to twice anything in training. That is the leading
explanation and it is untested. `eval_act.py` now records `spawn_yaw` alongside `spawn_pos`
(added 09:27:43; every round-1 cell and round-2's `dy_p001` were launched before that and lack
the field, every later cell has it) which makes the three-way split a read-back rather than a
re-run, next time a spawn cell is collected. It is a **separate** field rather than a fourth
element of `spawn_pos` so that the pairing check keeps comparing exactly what it always compared.

The failure mix independently confirms belief 5's *mechanism*: `never_lifted` — a bucket that has
been **empty in every `ckpt_final` cell measured on this project** — goes 0 → 4 → 17 as the box
widens. A wider spawn stresses the grasp, exactly as predicted, and the grasp is the part that
had never failed before.

### 6f. What this does to the headline

Stage C's 0.708–0.799 was measured over one spawn box, one slot position, one arm pose. Round 1
says:

* **Slot x: robust.** ±10 mm free, ~10 points at −10/+20 mm, tracking verified 1:1 at the level
  of where the block physically ends up. The Stage C number is *not* "in-distribution only" on
  this axis.
* **Slot y: zero, and provably so.** Not a policy weakness to be trained away — a tolerance set
  by the clearance and an observation that omits the sign.
* **Spawn box: graceful.** ×1.5 costs 15.6 points with in-box episodes untouched; ×2.0 costs 42.7
  and starts breaking the grasp.

None of it is a memorised geometry. That was the outcome I thought least likely when writing §5's
decision rule, and it is the one the data picked.

---

## 8. Expert control — is the dx dip the policy's, or the robot's?

*Pre-registered 2026-08-03 09:35, queued but not run. `scripts/run_expert_dx.sh`.*

§6b leaves one thing unexplained: the policy is flat from dx = 0 to +10 mm but loses 13.5 points
at −10 mm and 9.4 points at +20 mm. Two explanations fit the policy data equally well:

* **(a) the robot is worse at those slot positions.** The reBot arm has a measured horizontal-only
  grasp envelope (`rebot-arm-no-topdown-grasp`); a slot nearer the base folds the arm and a slot
  further out approaches reach. Push quality is a function of wrist conditioning, and that is
  geometry, not learning.
* **(b) the policy is worse there.** Those are the two positions furthest from anything in its
  training data, and degradation with distance from the training distribution is the null
  hypothesis for any BC policy.

The scripted expert separates them, because it is **told the answer**: `--slot_dx` patches
`SLOT_CENTER` *and* moves `insert_x` with it, and `plan.py` re-solves its own IK seed for the new
target (the cache is keyed on `insert_x`). Its trajectory is open-loop and near-optimal by
construction, so anything it loses is the arm's, not a policy's.

Four cells, `-v0`, 128 envs, seed 777: dx = −10 / 0 / +10 / +20 mm.

### 8a. Beliefs

11. **The expert is flat at 0 and +10 mm** (both ≥ 0.95, matching its known ~0.97 on `-v0`).
12. **The expert dips at −10 mm.** Prediction: 0.80–0.95, i.e. a real but smaller dip than the
    policy's. Mechanism: this is where the arm is most folded, and `stage_x = 0.165` was itself
    tuned against the nominal slot — the retract-then-traverse geometry has less room.
13. **The expert is *flat* at +20 mm** (≥ 0.95), unlike the policy. Mechanism: `check_geometry`
    already refuses targets that would drive the block through the back stop, so if the run
    executes at all the reach is fine, and a further slot is a *longer straight push*, which is
    the easiest part of the expert's plan.

If 12 and 13 both hold, the two dips have **different causes**: −10 mm is the arm, +20 mm is the
policy. That is the outcome I expect and it is the one that most changes what Stage D should
target. If the expert dips at +20 mm too, both are the arm's and no amount of policy work
recovers them. If the expert is flat everywhere, both dips are the policy's.

**A caveat stated up front:** the expert and the policy are not scored on identical episodes.
`run_expert.py` runs 128 envs with its own reset draw, not `eval_act.py`'s 32-env / 128-episode
protocol, and the expert has no later-cohort split at all — every one of its episodes is an
`episode_index_in_env == 0` episode, which is the cohort carrying the PhysX warm-start bias
(measured on this project at **+12.9 points for the expert specifically**). So the expert's
absolute number is not comparable to the policy's. **Only its shape across dx is**, and that is
all this control is asked for.

---

## 9. Round 3 — replication on a second training seed

*Pre-registered 2026-08-03 09:35, queued but not run. `scripts/run_robustness3.sh`.*

Rounds 1–2 are **one checkpoint at one spawn seed**. Training-seed variance on this project is
15–29 points — larger than either dx effect — so "`bc_armB_seed0` dips at −10 mm" and "the policy
dips at −10 mm" are different claims, and only a second training seed separates them. This is
`PLAN` 5.28's caution applied to my own results rather than quoted at the end of a table.

Subject: **`bc_armA_seed0`** — a different training arm *and* a different seed, second-best in the
sweep at 0.927 on `-v0` s777, so there is headroom for a 10-point dip to show and ceiling enough
for a floor-zero cell to be unambiguous. Six cells: the gate plus the five that carry a claim.

### 9a. Beliefs

14. **The gate reproduces `bc_armA_seed0`'s own sweep cell (0.927) exactly**, as `bc_armB_seed0`'s
    did. Same gate logic, same consequence if it fails.
15. **`dy_p005` is floor-zero here too.** This is the one round-1 result I would bet a lot on,
    because §6c's mechanism is pure geometry — a 5 mm shift against a ±1.5 mm entry window — and
    geometry does not depend on the training seed.
16. **The dx dips replicate in sign but not necessarily in size.** −10 mm and +20 mm both below
    the gate; +10 mm within noise of it. I deliberately do **not** predict the magnitudes: if
    seed variance is the dominant term, the sizes should move around and only the ordering should
    survive.
17. **`spawn20` costs 30–50 points.** The mechanism (`never_lifted` appearing at all) should
    replicate, because it is about the grasp envelope rather than about the insertion.

The single most likely way this goes wrong is belief 16 in the other direction: one dip
replicating and the other not. That would mean one of the two is a single-seed artefact, and
§6f's "slot x: robust" would need re-qualifying on whichever side failed.

### 7e. The dy ladder landed, and §7d's crossed cell was the wrong one

*Written 09:47, with `loose_dy_p003` and `loose_dy_p005` still running.*

On `-v0` (1.5 mm clearance), champion, all cells spawn-paired against the same 94/96 gate:

| dy | success | test |
|---|---|---|
| 0 | 94/96 = 0.979 | reference |
| +1 mm | 91/96 = **0.948** | McNemar b=4 c=1, p = 0.375 — **not distinguishable from nominal** |
| +2 mm | **0/96 = 0.000** | b=94 c=0, p = 1e-28 |
| +3 mm | 0/96 = 0.000 | b=94 c=0, p = 1e-28 |
| +5 mm | 0/96 = 0.000 | b=94 c=0, p = 1e-28 |

**The cliff is between 1 mm and 2 mm, against a 1.5 mm clearance.** That is the geometric model
landing on its face value, not near it. Belief 10's `-v0` half is confirmed and is sharper than I
wrote it: I predicted `dy_p002` would "degrade sharply", and it is a total floor — 96 consecutive
failures with no successes at all.

**But I picked the wrong crossed cell.** §7d's decisive test was `loose_dy_p003`: a 3.0 mm shift
against a 3.0 mm clearance. Work the geometry: the mouth half-width is 15 + 3 = 18 mm and the
block half-width is 15 mm, so a 3 mm shift leaves **exactly zero** margin on one side. It is the
boundary case, which makes it a coin flip rather than a test — whichever way it falls, both the
geometric model and its negation can claim it.

The cell that *should* have been in §7d is **2 mm on `-Loose-v0`**: a total floor on `-v0` and
1 mm inside the mouth on Loose. Same policy, same shift, opposite predictions, no boundary to
argue about. It is queued as `scripts/run_dy_crossed.sh` along with:

* `tight_dy_p001` — 1 mm on `-Tight-v0` (0.5 mm clearance). The model says the cliff moves *in*
  with the clearance, so the shift that costs 3 points on `-v0` should be a floor here. This is
  the same prediction as `loose_dy_p002` run in the opposite direction, and a model that only
  ever predicts "wider is better" would get one of the two wrong.
* `loose_dy_p000` and `tight_dy_p000` — the dy = 0 baselines for those two tasks. Without them,
  "`loose_dy_p002` = 0.9" could just mean Loose is an easier task, which it is.

Recording this rather than quietly swapping the cell: the original crossed cell is still running
and will still be reported. It was a design error — the pre-registered test could not have
discriminated — and the fix is an additional cell, not a replaced one.

### 7f. Belief 10 confirmed — the lateral tolerance *is* the clearance

The crossed cells landed. Same checkpoint, same spawn seed, **spawn-for-spawn identical
episodes** (McNemar, all 96 verified), differing only in a number in the env cfg:

| shift | `-v0` (clearance 1.5 mm) | `-Loose-v0` (clearance 3.0 mm) |
|---|---|---|
| dy = 0 | 94/96 = 0.979 | 89/96 = 0.927 |
| dy = +1 mm | 91/96 = 0.948 (p = 0.375, null) | — |
| dy = +2 mm | **0/96 = 0.000** | queued |
| dy = +3 mm | **0/96 = 0.000** | **73/96 = 0.760** (Δ −0.167, p = 0.0015) |
| dy = +5 mm | 0/96 = 0.000 | **0/96 = 0.000** |

**The same 3 mm shift is a total floor on `-v0` and a 76 % success rate on `-Loose-v0`.** No
policy property can produce that — the checkpoint, the spawns, the observation and the action
space are identical across those two cells. Only the width of the hole differs.

And each rung's cliff sits on its own clearance:

* `-v0`, clearance **1.5 mm**: fine at 1 mm, floor at 2 mm.
* `-Loose-v0`, clearance **3.0 mm**: degraded-but-working at 3 mm (the exact boundary — the block
  half-width is 15 mm and the mouth half-width is 18 mm, so a 3 mm shift leaves precisely zero
  margin on one side, and 0.760 is what "precisely zero margin" looks like), floor at 5 mm.

So the correct statement about lateral robustness is not about the policy at all:

> **A block 2·h wide entering a channel of half-width h + c can tolerate a lateral goal
> displacement of at most c. The policy delivers the block to y = 0.0 ± 1.0 mm, which already
> saturates that budget on `-v0`. Nothing trained on this observation could do better, because
> the slot's lateral position appears in it only through an absolute value.**

Two things follow that matter beyond this experiment:

1. **§6c's write-up was right for the wrong reason, and now has the right one.** The
   pre-registered mechanism (belief 2) was "the unsigned channel means the policy cannot know
   which way to correct." That is true but unfalsifiable from a collapse alone — a policy that
   simply could not aim would look the same. The crossed cells discriminate: if the policy were
   the problem, widening the channel would not rescue it. Widening the channel rescues it.
2. **This bounds what Stage D could ever buy.** Steering the flow's `x0` cannot move the lateral
   tolerance, because the tolerance is geometric. If a future version needs to handle a slot that
   moves in y, the fix is an **observation change** (a signed lateral error, or the slot pose
   itself) — not more RL, not more data, not a better policy class.

`dy_m010` (the symmetry control, belief 9) and the `-Tight-v0` rung are still queued;
`tight_dy_p001` is the prediction running in the opposite direction — 1 mm costs 3 points on
`-v0` and should be a **floor** at 0.5 mm clearance. A model that only ever predicts "wider is
better" gets that one wrong.

### 7g. Round 2 results — all twelve cells

| cell | perturbation | success | Δ vs own-task baseline | test |
|---|---|---|---|---|
| `dy_p001` | slot +1 mm y | 91/96 = 0.948 | −0.031 | McNemar p = 0.375 — **null** |
| `dy_p002` | slot +2 mm y | **0/96 = 0.000** | −0.979 | p = 1e-28 |
| `dy_p003` | slot +3 mm y | **0/96 = 0.000** | −0.979 | p = 1e-28 |
| `dy_m010` | slot −10 mm y | **0/96 = 0.000** | −0.979 | p = 1e-28 |
| `loose_dy_p003` | +3 mm y, 3.0 mm clearance | **73/96 = 0.760** | −0.167 | p = 0.0015 |
| `loose_dy_p005` | +5 mm y, 3.0 mm clearance | 0/96 = 0.000 | −0.927 | p = 3e-27 |
| `arm005` | arm start ±0.05 rad | 91/96 = 0.948 | −0.031 | z = −1.16, p = 0.248 — **null** |
| `arm010` | arm start ±0.10 rad | 92/96 = 0.958 | −0.021 | z = −0.83, p = 0.407 — **null** |
| `noise05` | 5 % sensor noise | 90/96 = 0.938 | −0.042 | z = −1.44, p = 0.149 — **null** |
| `noise20` | 20 % sensor noise | **4/96 = 0.042** | −0.938 | z = −12.99, p = 1e-38 |
| `combo` | dx +10 mm, spawn ×1.5, arm ±0.05 | 79/96 = 0.823 | −0.156 | z = −3.63, p = 0.0003 |

### 7h. Scoreboard against round 2's pre-registration

| belief | outcome |
|---|---|
| 6. arm jitter: < 15 pts at 0.05 rad, **> 30 pts at 0.10** | ❌ **falsified at 0.10** — measured 2.1 pts, p = 0.41, a clean null |
| 7. noise: free at 0.05, **> 25 pts at 0.20** | ✅ direction right, magnitude badly wrong — 0.20 costs **94 points**, not 25 |
| 8. combo ≥ the product of its parts | ✅ **confirmed to within 0.9 points** |
| 9. ±10 mm in y score the same | ✅ trivially — both are exactly 0/96 |
| 10. dy tolerance is the clearance | ✅ **confirmed** — §7f |

#### Belief 6: the arm start pose does not matter at all

This is the round-2 result I would not have bet on. The challenge env pins the arm to a single
`_START_POSE` for every one of the 2038 demonstrations — literally zero coverage on that axis —
and jittering it by pick_place's own ±0.1 rad costs **2.1 points, p = 0.41**. The named
falsifier fired too: I predicted the grasp would break first and `never_lifted` would appear.
`arm005` produced one `never_lifted` and `arm010` produced **none**.

§7b flagged this in advance: belief 6's reasoning ("the early fixed-start segment is the
memorised part") was the same shape as belief 4's, which round 1 had just falsified. It was
wrong for the same reason. The policy is closed-loop enough that where the arm starts is simply
not information it needs — it re-derives the approach from the observation. Two independent
falsifications of the same underlying prior is worth more than either alone: **I have been
consistently over-estimating how much of this policy is open-loop.**

#### Belief 7: 20 % sensor noise fails through *depth*, not through aim

The magnitude miss is the smaller half. The mechanism claim — "the failure should appear as
lateral degradation, because the noise perturbs where the policy thinks the slot is" — is
backwards. At 20 % noise:

* `never_entered` = **85 of 92 failures**
* median failure |lateral| = **0.11 mm** — the *lowest* of any cell in this document, gate
  included
* median failure depth = **−43.8 mm**

The block is delivered dead on axis and then stops 44 mm short of the mouth. Noise does not make
the policy wander; it makes it **stop advancing**. Which is the third independent perturbation in
this experiment (with slot dx = −10/+20 and spawn ×2.0) whose damage shows up as *insufficient
depth* — the same bottleneck `SESSION5_FINDINGS.md` §6d-ii identified from the unperturbed sweep.
Whatever is fragile in this policy, it is the push, and every axis you stress finds it.

The 5 % cell being free (p = 0.15) is consistent with the pre-registered attenuation argument —
chunked control re-reads the observation once per 15 steps — but the 20 % collapse shows the
attenuation is a constant factor, not a robustness guarantee.

#### Belief 8: the perturbations are independent, to within a point

Retention against the gate: `dx_p010` 1.021, `spawn15` 0.840, `arm005` 0.968. An
independent-failures model predicts the combined cell at 0.979 × 0.831 = **0.8136**. Measured:
**0.8229**, a difference of **+0.9 points**.

Two things follow. First, the three perturbations do not interact destructively, so the
single-axis numbers in this document can be composed — a stress you have not measured can be
estimated from ones you have. Second, and more practically: **`combo` (0.823) and `spawn15`
(0.823) are the same number to three decimals.** Moving the slot 10 mm and jittering the arm are
free, so the entire cost of the combined cell is the widened spawn box. When something in this
system is going to break, it is the spawn distribution and the push — not the goal position and
not the initial pose.

---

## 10. Actuation noise — the one axis where I predict worse than the sensing equivalent

*Pre-registered 2026-08-03 11:00, queued but not run. `scripts/run_action_noise.sh`,
`--action-noise` on `eval_act.py`.*

Three unrelated stresses in this experiment do their damage to the **same phase**:

| perturbation | how it fails |
|---|---|
| slot dx = −10 / +20 mm | `stalled_in_mouth`, median depth 36–38 mm |
| spawn box ×2.0 | `stalled_in_mouth` = 16 of 43 failures, median depth +21.3 mm |
| 20 % sensor noise | `never_entered` = 85 of 92, median depth −43.8 mm at |lateral| **0.11 mm** |

That last row is the sharpest statement of it in the whole document: the lowest lateral error of
any cell here, including the gate, and the block still stops 44 mm short. **The push is what
breaks.** Aim is not the fragile part of this policy — advancing is.

An actuator error acts on the push more directly than anything else available, so it is the
obvious missing cell. `--action-noise` adds gaussian noise to the *command*, after the
controller, scaled by each action channel's own training std — so it is directly comparable in
magnitude to `--obs-noise`, and the policy's next observation still carries the clean
`last_action`, which is what a real actuator error looks like from the controller's side.

### 10a. Beliefs

18. **Action noise is worse than observation noise at equal nominal magnitude.** At 0.05,
    `--obs-noise` cost 4.2 points (p = 0.15, null); I predict `--action-noise 0.05` costs **more
    than 15 points**. Mechanism: chunked control re-reads the observation once per 15 steps, so
    an observation error is low-passed by a factor of ~15 before it can act; an action error goes
    to the joints on every one of those 15 steps with nothing in between. This is the one place
    where the chunking that makes this policy work is also what makes it exposed.
19. **0.02 is roughly free** (< 5 points). Below the level at which the arm's own tracking error
    matters.
20. **0.20 is a floor**, like `noise20` was — but failing through a *different* bucket. Sensor
    noise produced 85/92 `never_entered` at 0.11 mm lateral; I predict action noise produces
    `gross_miss` and `never_lifted` instead, because a corrupted command breaks the **grasp**,
    which no perturbation except a widened spawn box has managed to touch.

Belief 20's bucket prediction is the falsifiable half. If 0.20 action noise also fails as
on-axis-but-short, then the "everything funnels into the push" reading is stronger than I have
any right to expect — three sensing/geometry perturbations *and* an actuation one, all with the
same terminal signature.

---

## 9b. Round 3 results — one dip replicates, one does not

`runs/bc_armA_seed0/robust_*.json`, six cells, run 10:50–11:10.

| cell | `bc_armB_seed0` (round 1) | `bc_armA_seed0` (round 3) | pooled n=192 |
|---|---|---|---|
| gate | 94/96 = 0.979 | **89/96 = 0.927** — its own sweep cell exactly | 0.953 |
| `dx_m010` | 0.844, Δ **−0.135**, p = 0.001 | 0.917, Δ **−0.010**, p = 1.0 | Δ −0.073, **p = 0.0026** |
| `dx_p010` | 1.000, Δ +0.021, p = 0.50 | 0.969, Δ +0.042, p = 0.22 | Δ +0.031, p = 0.070 |
| `dx_p020` | 0.885, Δ **−0.094**, p = 0.012 | 0.854, Δ **−0.073**, p = 0.14 | Δ −0.083, **p = 0.0037** |
| `dy_p005` | **0/96** | **0/96** | 0/192 |
| `spawn20` | 0.552, Δ −0.427 | 0.615, Δ −0.312 | Δ −0.370 |

### 9c. Scoreboard

| belief | outcome |
|---|---|
| 14. gate reproduces `bc_armA_seed0`'s sweep cell | ✅ 89/96 = 0.927, exactly |
| 15. `dy_p005` is floor-zero on a second seed too | ✅ **0/96**, McNemar b = 89, c = 0 |
| 16. both dx dips replicate in sign | ⚠️ **half** — `dx_p020` yes, `dx_m010` **no** |
| 17. `spawn20` costs 30–50 points, `never_lifted` appears | ✅ −31.2 pts; `never_lifted` = **20 of 37** failures |

### 9d. `dx_m010` was a property of one checkpoint, not of the policy

This is the outcome §9a named as the most likely way to be wrong, and it happened. The −10 mm dip
is 13.5 points on `bc_armB_seed0` (p = 0.001) and **1.0 point on `bc_armA_seed0` (p = 1.0)**. The
pooled test is significant (p = 0.0026) but the two cells disagree by 12.5 points, which is
exactly the 15–29-point training-seed variance this project measures everywhere else — the effect
is not stable enough to attribute to "the policy".

So §6b's line "dx = −10 mm costs 13.5 points" is **retracted as a general statement**. The
supportable version:

> Moving the slot 20 mm further away costs ~8 points and **replicates across training seeds**
> (−9.4 and −7.3, pooled p = 0.0037). Moving it 10 mm nearer costs anywhere from 1 to 14 points
> depending on which checkpoint you ask, and is therefore a checkpoint property rather than a
> task property.

This also changes what the queued expert control (§8) can settle. Belief 12 asked whether the arm
is worse at −10 mm; if the dip is not reliably there in the first place, an expert that is flat at
−10 mm tells us little we did not just learn more directly. The **+20 mm** half of that control
(belief 13) is now the valuable one, because that dip *is* real.

`dx_p010` is worth one line of caution in the other direction: both seeds came out **above** their
gate (+2.1 and +4.2, pooled p = 0.070). It is not significant and I am not claiming it. But two
independent checkpoints both improving when the goal is moved 10 mm out of distribution is the
kind of thing worth a third seed before dismissing.

### 9e. What replicated cleanly

**`dy_p005` = 0/96 on both checkpoints, 192 consecutive failures.** Two different training arms,
two different seeds, McNemar b = 94 and b = 89 with c = 0 both times. §7f's geometric account
predicts precisely this — the mouth cannot admit the block, so the policy is irrelevant — and it
is the single most reproducible result in this document.

**`spawn20` and its mechanism.** −42.7 and −31.2 points, and in both cases `never_lifted` — a
bucket that is *empty* in every unperturbed `ckpt_final` cell on this project — becomes the
largest single failure category (17 of 43, then 20 of 37). Widening the spawn box breaks the
grasp, on both checkpoints, exactly as belief 5 predicted in round 1.

Note `bc_armA_seed0` degrades **less** than the champion under `spawn20` (0.615 vs 0.552) despite
starting lower (0.927 vs 0.979). Relative retention 0.663 vs 0.564. A higher nominal score does
not buy robustness, which is worth remembering before picking a champion on nominal score alone.

---

## 8b. Expert control results — the arm is fine at every slot position; both dips are the policy's

`logs/expert_dx/{m010,p000,p010,p020}/expert_Rebot-PrecisionSlot-v0.json`, run 11:05–11:20.

| dx | expert `insert_x` | expert seated | policy `bc_armB_seed0` | policy `bc_armA_seed0` |
|---|---|---|---|---|
| −10 mm | 0.2445 | **125/128 = 0.977** | 0.844 | 0.917 |
| 0 | 0.2545 | **128/128 = 1.000** | 0.979 | 0.927 |
| +10 mm | 0.2645 | **128/128 = 1.000** | 1.000 | 0.969 |
| +20 mm | 0.2745 | **128/128 = 1.000** | 0.885 | 0.854 |

### 8c. Scoreboard

| belief | outcome |
|---|---|
| 11. expert flat at 0 and +10 mm (≥ 0.95) | ✅ 1.000 and 1.000 |
| 12. expert **dips** at −10 mm (0.80–0.95) | ❌ **falsified** — 0.977, three failures out of 128 |
| 13. expert flat at +20 mm | ✅ 1.000 |

### 8d. What this settles

**The reBot arm executes this task at every slot position the policy was tested at.** The
scripted planner, told the new slot centre and re-solving its IK seed for the new target, loses
at most three episodes in 128 anywhere in the range. So explanation (a) from §8 — "the robot is
worse at those slot positions" — is out, and **both dips belong to the policy**.

Combined with §9d, the two dips are now clearly different things:

* **dx = +20 mm, ~8 points, replicated on both training seeds, and the arm can do it perfectly.**
  This is a genuine, reproducible policy deficit with a demonstrated ceiling above it. It is the
  best-characterised improvement target this project has found.
* **dx = −10 mm** is 13.5 points on one checkpoint and 1.0 on another, and the arm can do it
  perfectly. A checkpoint-level idiosyncrasy, not a task property.

The expert's per-phase trace makes the point more concretely than the rate does. At dx = +20 mm
the block is at x = 277.0 mm after PUSH and seats 128/128 on RELEASE; at dx = +10 mm, 267.3 mm
and 128/128. The push simply carries further, and nothing about the arm's geometry objects.

**Caveat, as pre-registered:** the expert's episodes are all `episode_index_in_env == 0`, the
cohort carrying the PhysX warm-start bias — measured at **+12.9 points for the expert
specifically** — so 1.000 is not the expert's honest steady-state rate and must not be compared
to the policy's absolute numbers. What is valid is the **shape across dx**, measured on the same
cohort at every point, and the shape is flat. There is also a ceiling effect: three of the four
cells are at 128/128, so this control can bound the arm's contribution to the +20 mm dip at
roughly zero but could not have resolved a 2-point one.

**A provenance bug found while reading these** (fixed at 11:22, after the four runs): the results
JSON serialises `args_cli`, and the slot shift was applied to a local variable, so all four files
record `insert_x = 0.2545` — the pre-shift value. The runs themselves were correct (the logs show
`insert_x 0.2545 -> 0.2745` and the block reaching 277.0 mm), and `slot_dx` was recorded
correctly, so the cells are distinguishable and the table above is right. But it is the same
class of error as `make_videos.sh` overwriting three videos into one file: an artefact that looks
complete and is quietly identical. `run_expert.py` now writes the effective value back into
`args_cli` before the report is built.

---

## 11. §6e's flagged unknown, resolved — and the pipeline proved bit-deterministic

`robust_spawn15yaw.json` / `robust_spawn20yaw.json`: the same two widened-spawn cells re-run on
the binary that records `spawn_yaw`. Same checkpoint, same seed, same flags — no new
perturbation. `scripts/run_spawn_yaw.sh`.

### 11a. Reproducibility, for free

| cell | original | re-collected | episode-for-episode |
|---|---|---|---|
| `spawn15` | 79/96 | **79/96** | 96/96 identical spawns, **96/96 identical outcomes** |
| `spawn20` | 53/96 | **53/96** | identical |

**Not one episode differs.** `yaw_of()` consumes no RNG, so this was the prediction — but it is
the first time anything on this project has re-run a cell and compared it outcome-by-outcome. It
means every paired comparison in this document rests on a pipeline that is deterministic given
`(checkpoint, task, seed, flags)`, and that none of the deltas measured here can be run-to-run
noise.

### 11b. The in-box drop at ×2.0 was yaw, and once you split on it the policy is *perfect*

§6e could only split on spawn **position**, and had to report that at `--spawn-scale 2.0` even
the position-in-box cohort dropped to 0.783 (p = 0.0003) with no attribution. `--spawn-scale`
widens the **yaw** range too — ±0.35 → ±0.70 rad — and that was the leading suspect. It was
right. Splitting on both:

| cohort | `spawn15` | `spawn20` |
|---|---|---|
| **position IN, yaw IN** | **30/30 = 1.000** | **13/13 = 1.000** |
| position IN, yaw OUT | 8/11 = 0.727 | 5/10 = 0.500 |
| position OUT, yaw IN | 32/40 = 0.800 | 30/42 = 0.714 |
| position OUT, yaw OUT | 9/15 = 0.600 | **5/31 = 0.161** |

**43 of 43 fully in-distribution episodes succeeded, across both cells.** Not 0.783, not 0.927 —
1.000. Every point of the 15.6- and 42.7-point losses comes from spawns outside the training box,
and the policy's behaviour on the distribution it was trained for is untouched by the fact that
its neighbours in the batch were drawn from a wider one.

That is the textbook shape of clean generalisation-limited degradation, and it is worth stating
plainly because §6e had to leave open the possibility of something stranger.

**Yaw is the more damaging axis.** At ×2.0, yaw-out-alone (0.500) is worse than
position-out-alone (0.714), and both-out is 0.161. The spawn box's yaw range is ±0.35 rad = ±20°,
and doubling it is evidently a much bigger ask than doubling ±20 mm of position — which is not
obvious a priori, and is actionable: **if this policy were to be retrained for wider coverage,
yaw is where the demonstrations are thinnest relative to what the task asks.**

---

## 12. The dy ladder collapses onto a single curve in (shift / clearance)

The crossed cells are complete. Twelve cells, three clearances (0.5 / 1.5 / 3.0 mm), one
checkpoint, one spawn seed, all spawn-paired. Plotted against the **ratio** of the lateral shift
to the rung's own clearance rather than against the shift in millimetres:

| dy / clearance | clearance | dy | success | cell |
|---|---|---|---|---|
| 0.00 | 0.5 mm | 0 | 0.969 | `tight_dy_p000` |
| 0.00 | 1.5 mm | 0 | 0.979 | `gate_nominal` |
| 0.00 | 3.0 mm | 0 | 0.927 | `loose_dy_p000` |
| **0.67** | 1.5 mm | 1 mm | **0.948** | `dy_p001` |
| **0.67** | 3.0 mm | 2 mm | **0.938** | `loose_dy_p002` |
| **1.00** | 3.0 mm | 3 mm | **0.760** | `loose_dy_p003` |
| **1.33** | 1.5 mm | 2 mm | **0.000** | `dy_p002` |
| 1.67 | 3.0 mm | 5 mm | 0.000 | `loose_dy_p005` |
| **2.00** | 0.5 mm | 1 mm | **0.021** | `tight_dy_p001` |
| 2.00 | 1.5 mm | 3 mm | 0.000 | `dy_p003` |
| 3.33 | 1.5 mm | 5 mm | 0.000 | `dy_p005` |
| 6.67 | 1.5 mm | 10 mm | 0.000 | `dy_p010` |

**There is not one exception.** Ratio ≤ 0.67 → indistinguishable from that rung's own baseline.
Ratio = 1.00 → 0.760, degraded but working. Ratio ≥ 1.33 → floor, in every case, across a 6×
range of clearances and a 10× range of shifts.

The three pairs that make this more than a curve fit, because they are the same policy scored on
identical spawns with only a config number different:

* **2 mm shift: 0.000 on `-v0`, 0.938 on `-Loose-v0`.** Widening the channel rescues it
  completely.
* **1 mm shift: 0.948 on `-v0`, 0.021 on `-Tight-v0`.** Narrowing the channel destroys it
  completely. This is the direction that a lazy "wider clearance is easier" story gets wrong —
  the model has to predict a *failure* at a shift the middle rung shrugs off, and it does.
* **3 mm shift on `-Loose-v0` = 0.760**, the exact-boundary case (mouth half-width 18 mm, block
  half-width 15 mm ⇒ precisely zero margin on one side). "Degraded but working" is what zero
  margin should look like, and is what it looks like.

So the result can be stated without reference to the policy at all:

> **A block of half-width h entering a channel of half-width h + c tolerates a lateral goal
> displacement of at most c.** The policy delivers y = 0.0 ± 1.0 mm — it is already inside the
> budget — and its success as a function of displacement is a step function at c, not a decay.

`tight_dy_p001`'s two successes in 96 are the only nonzero above the boundary anywhere in the
table, and 1 mm against a 0.5 mm clearance is twice the budget: those two episodes presumably
spawned with a yaw that let the block corner in. Everything else is exactly zero.

**What this rules out.** It is not "BC is fragile to distribution shift" — the *same* shift is
fine or fatal depending on a number the policy cannot see. It is not "the policy needs more
data" — no amount of data moves a wall. And it is not something Stage D steering can address,
because `x0` steering perturbs the action distribution and the constraint is on the geometry.
**If a moved-in-y slot ever needs to work, the fix is the observation** — a signed lateral error,
or the slot pose itself — and that is a change to `eva_rl`, which is shared and not mine to make.

---

## 13. Actuation noise results — and the failure mode all the noise cells share

`robust_act{002,005,020}.json`, run 11:57–12:10.

| cell | magnitude | success | Δ vs gate | test | comparison: `--obs-noise` at the same magnitude |
|---|---|---|---|---|---|
| `act002` | 0.02 | **14/96 = 0.146** | −0.833 | z = −11.6, p = 3e-31 | — |
| `act005` | 0.05 | **0/96 = 0.000** | −0.979 | z = −13.6, p = 6e-42 | `noise05` = **0.938**, p = 0.15 (null) |
| `act020` | 0.20 | 0/96 = 0.000 | −0.979 | z = −13.6, p = 6e-42 | `noise20` = 0.042 |

### 13a. Scoreboard

| belief | outcome |
|---|---|
| 18. action noise worse than obs noise at equal magnitude; > 15 pts at 0.05 | ✅ **confirmed, and then some** — at 0.05, obs noise costs 4.2 points and action noise costs **97.9** |
| 19. 0.02 is roughly free (< 5 pts) | ❌ **falsified** — 0.02 costs **83.3 points** |
| 20. 0.20 fails through `gross_miss` / `never_lifted`, not on-axis-short | ✅ at 0.20 — `never_lifted` 30, `gross_miss` 39, median \|lateral\| 72.9 mm. But at 0.02 and 0.05 it fails **exactly like sensor noise** |

The asymmetry is the point, and it is enormous. **σ = 0.05 of each channel's training std is free
on the observation and fatal on the action.** The pre-registered mechanism holds: chunked control
re-reads the observation once per 15 steps, so an observation error is low-passed by roughly that
factor before it can act, while an action error goes to the joints on every one of those 15 steps
with nothing in between. The chunking that makes this policy work is also exactly what leaves it
exposed on the action side.

Belief 19's failure sharpens it further: **2 % actuation noise already costs 83 points.** This
policy is running open-loop within each 15-step window, and inside that window it has no
mechanism at all for rejecting a disturbance.

### 13b. Every moderate-noise cell fails the *same specific way*: it parks at the staging waypoint

`act002`, `act005` and `noise20` all fail as `never_entered` at a median depth of −42.9, −44.0 and
−43.8 mm, with median |lateral| of 0.35, 0.94 and 0.11 mm. Two different noise *types*, three
magnitudes, and the same number three times. So where does the block actually stop?

| cell | final block x | final y | final z | still above the table |
|---|---|---|---|---|
| `gate_nominal` (2 failures) | 0.2480 | 0.0000 | 0.0550 | 2/2 |
| `noise20` | **0.1660** | 0.0000 | 0.0620 | 86/92 |
| `act002` | **0.1670** | 0.0000 | 0.0620 | 81/82 |
| `act005` | **0.1660** | 0.0000 | 0.0630 | 92/96 |

**x = 0.166 m is the staging position.** The expert's `stage_x` is **0.165**, chosen because the
carried block rides ~5 mm ahead of the TCP (`run_expert.py` module docstring). And z ≈ 0.062 is
carried height, not seated (0.055) and not on the table (0.032–0.035).

So under moderate noise the policy:

1. reaches and grasps — ✅
2. lifts and retracts to staging — ✅
3. aligns in y to **0.0000** — ✅, perfectly
4. **never executes the push**, and holds the block there for the remaining hundreds of steps.

It is not confused, not wandering, and not dropping anything. It completes three of four phases
to nominal precision and then **freezes at the last waypoint**, gripper closed, block dead on
axis, until the episode ends.

That is a mode-collapse signature rather than a tracking-error one. The plausible reading — and it
is a hypothesis, not a measurement — is that the push is the phase in which the conditioning
observation changes least per step, so under noise the flow's chunk averages over "push" and
"hold" and produces something close to "hold". Testing it would mean looking at the predicted
action chunks directly rather than at outcomes, which nothing in this project does yet.

### 13c. Why this matters more than the robustness number

§10's pre-registration expected `--action-noise` to confirm "everything funnels into the push".
It did, and it also **named the failure**: the push is not degraded, it is *not attempted*. Four
independent stresses — slot dx, a doubled spawn box, sensor noise, actuator noise — converge on a
policy that gets the block to within 90 mm of a hole it has aligned to a tenth of a millimetre,
and stops.

Any future work on this task should target that transition. The block being **at staging, held,
aligned, and stationary** is a state the policy reaches reliably and cannot reliably leave, and
that is a much more tractable-looking problem than "improve precision".
