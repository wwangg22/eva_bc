# Stage 2 — the `env.step` port, the demo set, and flow BC

> ## ⚠ SUPERSEDED IN ONE CRITICAL RESPECT — read `14_FEEDBACK_AND_NEXT.md` §1 first.
>
> **Every success number in this file uses `mdp.target_at_goal`, which never checks whether a
> neighbouring block was MOVED — only whether it TOPPLED.** Re-scored with a 10 mm displacement
> requirement, the expert falls **73.3 % → 30.9 %** and the BC policy **71.8 % → 32.7 %**. The
> median episode this file calls a success displaces a neighbour by **13.7 mm**, and 22–25 % of
> all episodes **carry a neighbour to the goal zone**.
>
> Everything else here — the port gates, P29's branch seam, P27's holds, the seed-variance
> decomposition, the N2 measurement — is unaffected and stands. **§6.2's claim that "the policy
> has one failure mode" is withdrawn**: it had one failure mode among the buckets that were
> being counted, and the bucket that mattered was not one of them.

**Started 2026-08-03.** Everything here is new work; nothing in this file existed before the
expert was frozen (`11_STAGE2_RESULTS.md` §2g/§2h, `expert/pose_p33.json`).

The one-paragraph version
-------------------------
The port from physics-only execution to `env.step` costs **+0.9 points, i.e. nothing** — but
only after a defect that was invisible to every earlier stage was found and fixed. The
`home → chain[0]` approach, a segment no Stage-0 or Stage-1 measurement ever executed because
`run_physics` teleports past it, destroys the run in **two independent ways**: a joint-space
interpolation drives a finger 4.85 mm into a neighbour (5.5 % success), and a *geometrically
perfect* Cartesian solve scores **exactly 0.0 %** because it arrives 1.90 rad away from the
frozen chain on `joint6` — a different IK branch at the same tool point. Solving the same
polyline **backward from the frozen grasp pose** makes that seam the identity and recovers
73.0 %. Gate 2 then passes on all three pairs, 753 successful demos were recorded at 73.5 %,
and an offline audit of that dataset **confirms the registered N2 risk and localises it**:
54 % of the chunk ambiguity lives in the 70-step `close` hold, where the correct chunk is as
unpredictable from a near-identical observation as it is from no observation at all.

**And the policy trained on it scores 68.5 % against its expert's 71.4 % on 768 identical
held-out spawns -- a 2.9-point loss, where eva_bc's flow BC lost 21.6.** It clears the
pre-registered pass mark (>= 58 %) by ten points and sits within touching distance of the
mission target (~70 %) with no DAgger and no RL. Its failure taxonomy has exactly **one**
non-trivial entry: `distractor_toppled`, at 30.7 % against the expert's 18.8 %. No grasp
failures, no drops, no mode confusion, nothing stalled. The policy learned the manoeuvre and
inherited -- and amplified -- the single defect the manoeuvre already had.

---

## 1. What was built

| file | what it is |
|---|---|
| `clutter/act/collect_demos.py` | the demo generator **and** the Gate 2 instrument — one code path, so the equivalence check runs through the code that writes the data |
| `clutter/act/dataset.py` | 42-D HDF5 dataset; a copy of `eva_bc/act/dataset.py`, not an import (see §1.1) |
| `clutter/act/train_flow.py` | rectified-flow BC; the vendored `act/modeling_flow.py` unmodified |
| `clutter/act/eval_flow.py` | batched sim eval + the mandatory failure taxonomy |
| `clutter/act/analyse_demos.py` | offline dataset audit; measures N2 without a simulator |

### 1.1 Why `dataset.py` is a copy and not an import

`eva_bc/act/dataset.py` hardcodes the pick-place layout as module-level constants
(`OBS_DIM = 41`, `ENV_STATE_SLICE = slice(16, 41)`). Clutter's observation is **42-D**.

Python resolves module globals at call time, so `act.dataset.OBS_DIM = 42` before
instantiating would have worked and touched nothing. It was rejected: it is invisible at the
call site, it would silently corrupt any pick-place work in the same process, and
`eva_bc/act/` is a **tracked** file outside `clutter/`. The copy drops the four pick-place pool
filters (`nominal`, `recovery`, `dagger`, `_nominal_clean`), which have no analogue here —
clutter demos have no `episode_kind` and no per-can `outcomes` — so it is *shorter* than the
original, not longer.

The observation layout, read from `clutter_env_cfg.py:ObservationsCfg.PolicyCfg`, not guessed:

```
[ 0: 8]  joint_pos    (joint_pos_rel, 8)                          \
[ 8:16]  joint_vel    (joint_vel_rel, 8)                          / observation.state (16)
[16:23]  target_pose  (block_pose_in_root, 7)                     \
[23:35]  clutter      (clutter_obs, 12: 4 x (dx, dy, up_z))        | environment_state (26)
[35:42]  actions      (last_action, 7)                            /
```

Only the environment-state width changes, 25 → 26. The action head, the state head and the
chunking are untouched, which is what makes eva_bc's measured hyper-parameters transferable.

---

## 2. P29 — the approach segment, and two ways to destroy a working manoeuvre

### 2.1 The registered prediction, and its refutation before the probe ran

P29's docstring, written before Stage 2 began:

> **The joint-space lerp is clean and is the one we keep.** The entire path lies above
> z = 125 mm while the row tops out at 67 mm, so no interpolation excursion can reach the
> blocks. **Predicted: penetration 0.0 mm, approach hazard 0 %, and A40 within ±5 points of T.**

The collector's first smoke run refuted it before P29 was even edited: a 60-step joint lerp
scored **9.4 %** against 65.6 % for the teleport, with **90.6 % of episodes terminating early**
and the built-in FK audit reporting **7.3 mm** of keep-out penetration.

The premise was the error. "The entire path lies above z = 125 mm" is a claim about the
**TCP**, and the TCP is not the arm — and it was not even true of the TCP, which dips to
**z = 86 mm**. This is P17's lesson at a new location: independently specified endpoints, a
joint-space line between them, and a Cartesian excursion nobody bounded.

### 2.2 The kinematic audit

`p29_approach_segment.py` was extended to report *where* the worst penetration is — which
body, which keep-out box, how far along the path, and the TCP there. "The path is dirty" is
not a mechanism and cannot be fixed.

```
candidate                 penetration   min body z   max line dev   waypoints
A  joint-space lerp           4.85 mm      45.0 mm         71.4 mm      —
B  forward dense Cartesian    0.00 mm     105.0 mm          0.8 mm      6
C  backward dense Cartesian   0.00 mm      89.0 mm         47.4 mm      6

A worst: gripper_left into distractor_1 at 75 % of the path, 4.85 mm
         TCP (305.1, −13.7, 86.3) mm, 53.8 mm off the straight line
```

`home` TCP is (387.7, +0.0, 164.5) mm; `chain[0]` is (250.0, +0.1, 124.4) mm; 143.4 mm apart.

### 2.3 The seam — the finding that matters

Arm **B** is geometrically spotless: 0.00 mm penetration, every body above z = 105 mm, 0.8 mm
off the straight line. It scores **0.0 %**, over 256 episodes, with **100 % topple** and
**96 % of targets still delivered to the goal**.

```
SEAM AT chain[0] — max |dq| between the approach's last waypoint and the frozen chain's qs[0]
   B forward    1.9012 rad    per joint  +0.13  +1.30  +1.06  −0.96  +0.21  −1.90
   C backward   0.0000 rad    <-- identity by construction
```

`_dense([tcp_home, tcp_0], q_arm0)` finds *an* IK solution at `chain[0]`'s tool point — not
**the** one the frozen chain uses. The schedule's first phase is a 10-step hold at `chain[0]`,
so the moment the approach ends the arm is commanded to move 1.90 rad on `joint6` **as a step
change, with the gripper directly over the row**. The wrist sweeps every neighbour down on its
way through, and then the manoeuvre proceeds to grasp and deliver the target perfectly.

`96 % at goal with 100 % topple` is a signature worth remembering: **the task metric and the
constraint metric can be completely decoupled.** An evaluation that reported only "did the
block reach the goal" would have called this arm a success.

This is P17's branch flip, moved to the join between two independently solved paths. `_dense`
enforces local solves **within** a path; nothing enforced the same rule **across a join**. The
fix is the one `plan()` has always used for the descent — solve outward from the grasp and
reverse — applied to the approach:

```python
back_pts, back_qs = ex._dense([tcp_0, tcp_home], ex.qs[0])   # backward, seeded from the chain
c_qs = list(reversed(back_qs))                               # c_qs[-1] IS ex.qs[0], exactly
```

### 2.4 The result

```
arm                        pooled (256 eps)   vs teleport   approach hazard
T   teleport (baseline)         74.2 %            —              0.0 %
A40 joint lerp, 40 steps         5.5 %         −68.8 pts       100.0 %
B40 fwd cartesian, 40 steps      0.0 %         −74.2 pts         0.0 %
C40 bwd cartesian, 40 steps     71.5 %          −2.7 pts         0.0 %
C80 bwd cartesian, 80 steps     73.0 %          −1.2 pts         0.0 %
```

A clean three-way dissociation. **A** is dirty along the path (approach hazard 100 %). **B** is
clean along the path and lethal at the seam (approach hazard 0 %, `settle` hazard 100 %). **C**
fixes only the seam and lands inside the noise floor.

**C80 ships.** The 6-waypoint path is frozen into `expert/pose_p33.json` under `approach`, for
P34's reason: `_dense` runs a CEM and the CEM is not bit-reproducible even under a fixed seed,
so a path that is re-solved is not the path that was measured.

### 2.5 Also settled here — the action magnitude (N7)

```
a = (q_target − q_default) / 0.5, over the frozen chain
   j1  −1.802 … −0.624   |a|max 1.802
   j2  −1.243 … −0.041   |a|max 1.243
   j3  −2.306 … −1.322   |a|max 2.306
   j4  +3.638 … +4.631   |a|max 4.631   <-- the binding one
   j5  −1.119 … −0.079   |a|max 1.119
   j6  +1.269 … +2.642   |a|max 2.642
```

**6 of 6 joints leave the `[−1, 1]` box**, and `joint4` needs **4.63**. N7 predicted ~1.57 on
`joint1` from the goal azimuth; the real constraint is 3× larger and on a different joint. The
consequence is unchanged in direction and much stronger in degree: **`clip_actions = 1.0`
makes this manoeuvre unreachable for an rl_games agent**, and even `clip_actions = 5.0` is only
just enough. Harmless for BC — the action manager's `cfg.clip` is `None`, checked at startup by
`collect_demos.py` rather than assumed, so `JointAction` skips its clamp entirely.

---

## 3. Gate 2 — the port

Four arms, each differing from its neighbour by **one** thing, all on the same four spawn
seeds (77000–77003), 128 envs each, 512 episodes per arm.

| arm | what it does | pooled |
|---|---|---|
| `phys` | settle + `run_physics` — the Stage-1 reference | **72.7 %** |
| `tele` | settle + teleport to `chain[0]`, then `env.step` | 72.5 % |
| `tele0` | teleport + `env.step`, **no settle** | 73.0 % |
| `appr` | reset pose → frozen approach → `env.step` | **73.6 %** |

```
2a  MDP vs physics-only          72.7 % → 72.5 %   −0.20 pts   PASS   flips  49/512 (9.6 %)
    the 30-step free settle      72.5 % → 73.0 %   +0.59 pts   PASS   flips  43/512 (8.4 %)
2b  action-driven approach       73.0 % → 73.6 %   +0.59 pts   PASS   flips  47/512 (9.2 %)

phys 72.7 % vs P34's quoted 72.1 %   (+0.56 pts)   — the reference check
```

**GATE 2 PASSES.** All three deltas are inside ±5 points and all three flip counts sit at
P34's measured ~8 % paired noise floor. The whole port — MDP terminations, the action encoding,
the loss of the free settle, and a trajectory segment that never existed before — costs
**+0.9 points**, which is to say nothing at all.

Three things this run established that were previously assumed:

1. **The 30 free physics steps every probe has run since P01 are worth −0.6 points**, i.e.
   nothing. They are not available to a policy at deployment and now they do not need to be.
2. **`target_dropped` and `time_out` never fire: 0.0 % each.** The *entire* failure mode is
   `distractor_toppled`, at 20.9 %. The scripted expert has exactly one way to fail.
3. **The gripper action encoding is correct.** `BinaryJointPositionAction` is
   `where(a[6] > 0, open, close)` and `ArmKin.act` emits `+1 / −1`; enclosure among
   non-terminated episodes is 100 %.

### 3.1 A methodological repair made mid-run

The taxonomy cannot come from the end-of-episode scene. `env.step` auto-resets a terminated
env, so a toppled scene has already re-spawned **upright** and `ex.score()` reports
`topple = False` — the failure erases itself. That is the same trap `run_physics` exists to
avoid (`07_STAGE0_RESULTS.md` §7.5), reappearing the moment the MDP was let back in. It was
caught in the smoke run, where the `tele` arm reported `topple 0.0 %` against `phys`'s 15.6 %
on identical spawns.

The fix reads `TerminationManager._term_dones` immediately after each `env.step` — `compute()`
writes it every step and `reset()` does not clear it — so the taxonomy records which term fired
**at the step the env actually died**, before the re-spawn.

### 3.2 The one pairing artefact, and why it is not one

The scene fingerprint mismatched on 1 of 4 seeds, by **1 unit out of 1.6 million**. The hash
rounds each coordinate to 0.1 mm and is read *after* the settle, so a coordinate on a rounding
boundary flips the sum by 1. That is the settle — which is precisely what the `tele`/`tele0`
arm measures. The check is now grouped by settle state and the drift reported separately.

---

## 4. The dataset

```
8 spawn seeds (30000–30007) x 128 envs = 1024 episodes
  753 successful (73.5 %)     <- the BC pool
  188 toppled     (18.3 %)
   83 survived but never placed (8.2 %)
    0 dropped, 0 timed out
472 env steps per demo, exactly — every kept demo is the same length
355 416 (demo, t) training samples          runs/demos_v1.hdf5, 62 MB
```

Verified by `analyse_demos.py`, not assumed:

* **the gripper channel takes exactly two values, `{−1, +1}`.** No ramp reached the tape. A
  demo containing an intermediate aperture would teach an action the policy cannot submit
  (P19/P20).
* **actions span `[−2.31, +4.63]`**, matching P29's audit of the nominal chain.
* **every episode is 472 steps** — successes never terminate early, because the only
  termination is `distractor_toppled` and that is a failure by definition.

### 4.1 Three near-degenerate observation channels — recorded, not changed

```
env_state[ 9] d0_up   std 7.49e−05   -> normalization multiplies its residual by 13 357
env_state[15] d2_up   std 6.14e−04   ->                                          1 630
env_state[18] d3_up   std 2.36e−05   ->                                         42 390
```

These are distractor up-axes. Successful demos never topple, so the channel sits at ~1.0
throughout and `(x − mean) / (std + 1e−8)` rescales what is left to unit variance. What is left
is not pure numerical noise — it is the real micro-nudge signal, and `d1_up` (std 0.0188, the
distractor P15 found is the victim in 66 of 128 envs) shows the same quantity at a healthy
scale. So the normalization is arguably doing the right thing, and float32 at 1.0 has a quantum
of 6e−8, i.e. **400 quanta inside d3's std** — this is not quantization-limited.

It is recorded rather than fixed because one informed change per run is the rule and this is a
hypothesis, not a diagnosis. **It is the first thing to test if BC underperforms.**

---

## 5. N2 — measured, confirmed, and localised

The largest registered risk to Stage 2, from `09_STAGE2_BC_PLAN.md` N2:

> 47 % of the demo is a held pose — 70 env steps of `close` alone — during which the 42-D
> observation is constant to numerical noise while the correct action chunk differs at every
> step. A memoryless chunk policy cannot recover phase from that.

That is a hypothesis with a number attached, and the number is measurable **offline, before a
training step is spent**. If two frames have near-identical observations and different correct
chunks, no memoryless predictor gets both right, and the size of that disagreement is a **floor
on the training loss**.

`analyse_demos.py` estimates it by nearest neighbours in **normalized** observation space -- the
space the network sees -- over 4 000 query frames per phase against 30 000 reference frames,
k = 16:

```
phase       nn_chunk_rmse  uncond_rmse   ratio   nn_dt   floor(hi)    floor
approach          0.0536       0.4901    0.109      4      0.0516    0.0402
settle            0.0563       0.2957    0.190      6      0.0529    0.0133
descend           0.1563       0.4711    0.332      8      0.1084    0.0581
predwell          0.2620       0.4032    0.650      8      0.1135    0.0711
close             0.4547       0.4526    1.005     20      0.3596    0.3594   <-- confirmed
carry             0.1150       0.9607    0.120      2      0.1076    0.0858
dwell             0.1633       0.2587    0.631      3      0.0489    0.0459
release           0.0706       0.0649    1.087      8      0.0427    0.0434
withdraw          0.0269       0.0393    0.685      4      0.0166    0.0191
final             0.0009       0.0000       --      8      0.0002    0.0000   (both ~0)
```

`ratio` is the fraction of the chunk's variation that **survives** conditioning on the
observation. Near 0, the observation determines the chunk. Near 1, it does not.

**The `close` hold is at 1.005** -- the correct chunk is as unpredictable from a near-identical
observation as it is from no observation at all -- **and its nearest same-demo neighbour is a
median of 20 env steps away inside a 70-step hold.** The registered prediction was right, and
the effect is at its theoretical maximum.

### 5.1 The loss floor, derived rather than asserted

The training log prints a **velocity** MSE, not a chunk MSE, so putting the two side by side
needs the work done first. Rectified flow regresses `v = x1 - x0` on `(obs, x_tau, tau)` with
`x_tau = (1-tau)*x0 + tau*x1`. Modelling one dimension of `p(x1 | obs)` as `N(mu, sigma^2)`,
everything is jointly Gaussian and the Bayes residual is

```
Var(v | obs, x_tau) = (sigma^2 + 1) - (tau*sigma^2 - (1-tau))^2 / ((1-tau)^2 + tau^2*sigma^2)
```

Integrating over `tau ~ U[0,1]` gives a clean identity, verified numerically to four decimals
for every sigma from 0.01 to 1.0:

```
floor(sigma) = sigma * pi / 2
```

Zero when the observation determines the chunk, linear in the ambiguity. Summed over the
`50 x 7 = 350` chunk cells, `floor = (pi/2) * mean_cells(sigma_cell)` -- the **mean**, not the
RMS, because ambiguity concentrated in a few cells costs far less than the same total spread
smeared over all of them.

Two corrections were needed to make the estimate honest, and both are in the script:

1. **A nearest neighbour is not at zero distance.** Whatever the observation *does* determine
   varies between query and neighbour, and that leaks into the estimate. Debiased by fitting
   `E[delta^2] = 2*sigma^2 + beta*d_obs^2` per cell over all (query, neighbour) pairs and
   reading off the intercept. This moved the whole-demo number 0.111 -> **0.0989** and moved
   the `close` not at all (0.3596 -> 0.3594) -- correctly, since inside a hold the observation
   distance carries no information to begin with.
2. **It is still an UPPER bound, and here a loose one.** The derivation prices all 350 cells
   independently, but the flow head sees `x_tau` for all of them at once, and the ambiguity
   here is essentially **one scalar** -- "how many steps until the lift". 350 noisy readings of
   a single latent identify it almost exactly, so the model resolves at training time an
   ambiguity the per-cell arithmetic charges it for 350 times over. **Measured: bound 0.0989,
   observed plateau ~0.045.** Consistent, not contradictory.

What the bound is still good for is **localisation** -- every phase is priced on the same
footing:

```
close       53.9 %   ( 70 steps, 14.8 % of the demo)
carry       20.9 %   (114 steps, 24.2 %)
descend      9.3 %   ( 75 steps, 15.9 %)
approach     6.7 %   ( 78 steps, 16.5 %)
everything else  9.2 %
```

**54 % of the ambiguity lives in 15 % of the demo.**

And the behavioural consequence is untouched by any of the above. At **inference** there is no
`x_tau` -- sampling starts from `x0 ~ N(0, I)` -- so the mode really is drawn from the prior
and the policy really does pick its own moment to lift. `ratio` is a property of the *data* and
is the column to read; the loss floor is a property of the *objective* and is the one to be
careful with.

Why it is concentrated there and not in every hold: a chunk is 50 steps and the hold is 70, so
frames 183–203 have chunks entirely inside the hold and are perfectly determined, while frames
203–253 have chunks containing part of the `carry` — and *where* the carry starts inside the
chunk is exactly the unobservable quantity. The ambiguity is not "what to do", it is "when the
lift begins".

**The consequence and what to do about it.** `HOLDS["close"] = 560` physics steps was chosen
for a physics-only measurement where duration is nearly free. **P27 is the probe that decides
whether it can be shortened**, and this measurement is the strongest argument yet for running
it: a shorter close attacks the dominant term directly. The registered fallback — a
`train_mask` censor over the ambiguous tail — is available without regenerating anything,
because `collect_demos.py` stores the phase boundaries in every episode's attrs.

**Order of operations, deliberately.** Train and evaluate on `demos_v1` *first*. A baseline
number is what makes any later intervention measurable, and cutting the close before knowing
what the ambiguity costs in *success* (as opposed to in loss) would be optimising a proxy —
which is the mistake P25 made and paid 2.0 % end-to-end for.

> **→ P27 has since run (§7) and the answer is better than the fallback.** The close is flat
> from 560 physics steps down to 40 with **100.0 % enclosure at every duration**, and
> shortening the other five holds makes the *expert* +3.9 points better as well. `demos_v3`
> drops the ambiguity floor from 0.0989 to 0.0607 and removes `close` from the top four
> contributors entirely. **The `train_mask` censor was never needed.** What remains is
> ambiguity in the *moving* phases — real per-spawn variation, not unobservable phase.

---

## 6. Training

```
python -u clutter/act/train_flow.py --data clutter/runs/demos_v1.hdf5 \
    --out clutter/runs/bc_s1 --steps 100000 --seed 1
```

```
dataset: 753 demos kept, 271 rejected, 355 416 samples; episode length 472–472
policy:  FlowMatchingPolicy, 23.3 M params, chunk 50, n_action_steps 15, 10 Euler steps
         rectified flow, no CVAE, temporal ensembling off, AdamW lr 1e−4, batch 64
throughput: ~66 steps/s on the 10 GiB card, 1.1 GiB used -> 100 k steps in ~25 min
```

`n_action_steps = 15` is **not a tunable**: eva_bc measured 59.4 → 32.8 → 3.1 → 0 → 0 % at
15 / 8 / 4 / 2 / 1 (EXP02). Chunk commitment is load-bearing.

### 6.1 Result — seed 1

**768 held-out episodes** (seeds 88000-88005, never used for demos or for anything else), with
the **frozen expert run on the identical spawns** as a paired comparator:

```
                                success   topple   dropped   other
expert, frozen, env.step         71.4 %   18.8 %    0.0 %    9.8 %
flow BC, seed 1                  68.5 %   30.7 %    0.0 %    0.8 %
                                 ------
paired gap                       -2.9 pts  (se 1.7 -> -2.9 +/- 3.4 at 95 %)
```

per seed, so the pairing is visible rather than asserted:

```
seed     88000   88001   88002   88003   88004   88005
expert   69.5 %  70.3 %  72.7 %  75.8 %  73.4 %  66.4 %
BC       64.8 %  75.0 %  68.0 %  69.5 %  72.7 %  60.9 %
delta    -4.7    +4.7    -4.7    -6.3    -0.7    -5.5
```

**Against the band registered in `HANDOFF.md` §10.1 before any policy existed:**

```
>= 58 %      PASS outright        <-- 68.5 %, ten points clear
50 - 58 %    on trend
43 - 50 %    below trend
<  43 %      porting defect
```

**And against the trend line the band was built from:** eva_bc's flow BC lost **21.6 points**
to an 85.7 % expert. This one loses **2.9**. That is the single most surprising number in the
stage, and `HANDOFF.md` §10.1 registered a candidate explanation *in advance*:

> "One thing in our favour that eva_bc did not have: every demo will come from a **single
> frozen chain**. [...] A unimodal dataset should transfer better than the trend line predicts
> -- **registered as a prediction, not assumed**; if BC lands above 58 % this is the first
> explanation to test."

The **outcome** prediction is confirmed. The **mechanism** is not tested, and should not be
written up as though it were: demonstrating it needs a control trained on a deliberately
multi-modal dataset (a pose re-solved per episode, which is what P28 showed `plan()` does by
default -- six clusters from eight draws). That control is cheap and is now the most
informative experiment available about *why* this worked.

### 6.1a Replication — three seeds, and the variance is not where it was expected

`HANDOFF.md` §10.0 queued three training seeds because "one draw is not a result". That was
the Stage-2a lesson about pose draws, and it applies here with force.

```
                 pooled     topple   stalled   no-grasp   final-10k mse   per-batch spread
expert            71.4 %    18.8 %    9.8 %*      --           --         66.4 - 75.8 %
flow BC seed 1    68.5 %    30.7 %    0.3 %      0.5 %       0.0370       60.9 - 75.0 %
flow BC seed 2    77.7 %    22.0 %    0.0 %      0.3 %       0.0339       75.8 - 79.7 %
flow BC seed 3    58.1 %    28.4 %    4.0 %      9.5 %       0.0360       51.6 - 60.9 %
                  ------
mean               68.1 %   sd 9.8 pts   range 19.6 pts   se 5.7
```

\* the expert's "stalled" is its residual bucket: gripped, no topple, block not in the goal
circle at the end. It is not directly comparable to the policy's `stalled`, which additionally
requires `target_extracted`.

**The mean is 68.1 % against the expert's 71.4 % — a 3.3-point gap that the 9.8-point seed
spread comfortably swamps.** Every seed clears the pre-registered ≥58 % pass mark, seed 3
exactly at it. The right statement is a range, and the earlier single-seed headline (68.5 %)
was one draw of three.

**Three things this makes clear, none of which one seed could have shown.**

**1. The training seed is worth an order of magnitude more than anything else.** Both
competing noise sources were measured, not assumed:

```
policy x0 sampling   ~0.4 pts   (seed 1 at three noise sequences: 68.5 / 68.1 / 67.7)
binomial at n = 768   1.6 pts
TRAINING SEED         9.8 pts   (sd over three seeds)
```

The sampling number needed a control of its own. A flow policy draws `x0 ~ N(0, I)` per chunk,
so the same checkpoint need not score the same twice — yet re-running seed 1's evaluation
returned **68.5 %, agreeing episode for episode on all 768**. The cause is that Isaac Lab's
`env.reset(seed=)` calls `torch.manual_seed`, which also pins the policy's noise. That is
convenient (checkpoint comparisons on shared spawn seeds are *exactly* paired) and it is a
trap, because one run then samples exactly **one** `x0` sequence and the apparent determinism
could be mistaken for low variance. `--policy-seed` decouples them; across three independent
noise sequences seed 1 gives 68.5 / 68.1 / 67.7.

**Consequence: every BC-level comparison from here costs three training runs per arm.** The
unimodality control, a `train_mask` censor, a shortened `close` — run at one seed, each would
be measuring this instead. This is Stage 2a's pose-draw lesson repeating in a new place, and
it is the second time in this effort that the dominant variance turned out not to be the one
the experiment was designed around.

**2. The training loss does not predict the score.** Seed 3 has a *lower* final loss than
seed 1 and scores **10.4 points worse**:

```
seed 2   mse 0.0339   77.7 %
seed 3   mse 0.0360   58.1 %      <-- lower loss than seed 1, 10.4 points worse
seed 1   mse 0.0370   68.5 %
```

The losses span 9 % relative; the success rates span 19.6 points. **A checkpoint or a seed
cannot be selected by loss here — it has to be evaluated in the simulator.** (This is eva_bc's
"BC loss is not success" lesson, now measured for clutter rather than inherited.)

**3. The bad seed fails *differently*, not just more.** Seeds 1 and 2 have essentially one
failure mode; seed 3 has two:

```
                  seed 1   seed 2   seed 3
never extracted     0.5 %    0.3 %    9.5 %     <-- 20x, a new failure mode
extracted, stalled  0.3 %    0.0 %    4.0 %
toppled            30.7 %   22.0 %   28.4 %
```

So the seed spread is not a smooth quality axis. Seed 3 acquired a mode in which the policy
**never picks the block up at all** in one episode in ten — while its topple rate is *better*
than seed 1's. Whatever distinguishes the seeds is not "how well it does the manoeuvre" but
"whether it reliably starts it", and that is a different question with different fixes. It is
also the first evidence in this effort of the mode-confusion failure `09_STAGE2_BC_PLAN.md`
worried about, appearing at one seed in three.

### 6.2 The taxonomy — RETRACTED IN PART, see `14_FEEDBACK_AND_NEXT.md` §1

**The policy has one failure mode, and it is the expert's failure mode.**

```
                     expert    BC seed 1
toppled              18.8 %      30.7 %      <-- the whole gap, and then some
target_dropped        0.0 %       0.0 %
never extracted        --         0.5 %
extracted, not placed  9.8 %      0.3 %
```

Read across: the policy is **better** than the expert at completing the delivery once it has
the block (0.3 % stalled against 9.8 %) and **worse** at not toppling a neighbour (30.7 %
against 18.8 %). Those cancel to −2.9 points.

That is a much better position than the number alone suggests. There is no grasp problem, no
approach problem, no reach problem, no mode confusion, no drift off the end of the demo
horizon. **Everything that remains is the close-phase blade sweep** -- the same residual that
§5 of `HANDOFF.md` has named as the sole open failure since Stage 1, now inherited by the
policy and amplified by about 12 points.

### 6.3 Two protocol checks, because the comparison would otherwise be unfair

Neither of these changed a conclusion; both had to be run before the conclusion could be
trusted.

1. **Latched vs final-step success.** `eval_flow.py` latches `target_at_goal` over a 700-step
   episode; `collect_demos.py` scored the expert once, at the end of its 472-step schedule.
   Those are different questions and the difference ran in the expert's disfavour. Both are
   now measured on both sides:

   ```
   expert   latched 71.4 %   final 71.4 %      identical, seed for seed
   BC       latched 68.5 %   final 68.5 %      identical, seed for seed
   ```

   So the latch is worth exactly zero here, and it is worth zero for a reason worth recording:
   **the policy places the block and then leaves it alone for 228 steps it has no training
   data for.** It does not wander back and knock it out of the circle.

2. **The auto-reset trap, a second time.** The first version of the final-step metric reported
   **0.0 % for every seed** -- which reads exactly like a finding ("the policy never ends at
   the goal") and is a measurement artefact. Every episode ends on a `time_out`, Isaac Lab
   auto-resets inside that same `env.step`, and the scene read afterwards is a fresh spawn.
   A *latched* quantity is immune (a re-spawned target sits 196 mm from the goal and 35 mm
   high, satisfying neither predicate); a **last-write-wins** quantity is not. R23 again, in a
   place the first fix did not reach.

---

## 7. P27 — the holds, and the best result of the stage

### 7.1 What it measured

The demo schedule's six hold durations were chosen for a **physics-only** measurement, where
waiting is nearly free. §5 showed what they cost behaviour cloning. P27 asks the question
nobody had asked: **how long do the fingers actually need?**

Run with the frozen pose *and* the frozen chain, so the only variable in the sweep is the hold
duration — without that edit the 12-point pose sd (P32) would have swamped it, exactly as it
swamped every P26 verdict. Three spawn batches, 128 envs, snapshot/restore between arms.

### 7.2 Arm 1 — the close, swept 560 → 40 physics steps

```
close (phys)  env steps  demo len  static %   enclosure   SUCCESS   vs 560
    560           70        394     45.7 %     100.0 %    72.9 %      —
    400           50        374     42.8 %     100.0 %    71.9 %    −1.0
    280           35        359     40.4 %     100.0 %    72.1 %    −0.8
    200           25        349     38.7 %     100.0 %    71.1 %    −1.8
    140           17        341     37.2 %     100.0 %    71.9 %    −1.0
     80           10        334     35.9 %     100.0 %    73.4 %    +0.5
     40            5        329     35.0 %     100.0 %    71.1 %    −1.8
```

384 episodes per cell. **Flat everywhere**, largest single-step drop 2.3 points, and
**enclosure is 100.0 % at every single duration** — the fingers are shut and loaded long
before the schedule stops waiting for them.

The registered prediction was "flat from 560 down to about 120 physics steps, then a cliff".
Its *shape* held — a plateau, not the gradual decline that would have meant the close was
settling the row rather than closing the gripper. Its *number* was too conservative, and the
honest statement is stronger than the verdict line the script printed: **no cliff was found at
all.** 40 physics steps is the shortest duration tested, not the shortest safe one. Anything
below it is unmeasured.

**70 env steps were being spent on something that needs at most 5.**

### 7.3 Arm 2 — the other five holds, and the surprise

With `close = 40`, scale `settle` / `predwell` / `dwell` / `release` / `final` together:

```
scale   demo len   static %   SUCCESS   per batch
1.00      329       35.0 %    72.1 %    71.9 / 71.1 / 73.4
0.50      274       21.9 %    76.0 %    73.4 / 76.6 / 78.1
0.25      247       13.4 %    75.8 %    75.8 / 73.4 / 78.1
```

**Shortening the holds makes the expert BETTER — about +3.9 points — while cutting the demo
from 394 env steps to 247 and its static fraction from 45.7 % to 13.4 %.** The gain is +1.5 /
+5.5 / +4.7 across the three batches: consistent in sign, se 1.2, so real but not large. What
matters more is the direction: this was expected to be a **trade** — expert success against BC
learnability — and it is not one. Both improve.

A plausible mechanism, offered as a hypothesis and not measured: the long holds let the
gripper keep loading a block it has already gripped, and a longer `release`/`final` gives the
arm more time to disturb a scene it has already finished with. But the sweep does not separate
that from anything else, and nothing here depends on it.

### 7.4 What shipped — `demos_v3`

Regenerated at `close = 40`, `holds-scale = 0.25`, and confirmed under `env.step`:

```
dataset   holds        keep rate   demo len   static %   flow floor   `close` share of floor
v1        original      73.5 %       472       38.1 %      0.0989          53.9 %
v2        scale 0.50    75.8 %       352       17.0 %      0.0688           —
v3        scale 0.25    75.6 %       325       10.2 %     0.0607           —
```

The expert improvement reproduces exactly under the MDP (73.5 → 75.6 %), so P27's physics-only
result was not an artefact of the execution path.

**v3 ships.** The two candidates are statistically tied on expert success (75.8 vs 75.6) so the
choice was made on the audit, which is free: v3 has the lowest ambiguity floor, the shortest
demos and the smallest static fraction. And the localisation has completely inverted —

```
v1:  close 53.9 %,  carry 20.9 %,  descend  9.3 %
v3:  descend 54.0 %,  carry 21.0 %,  approach 16.7 %      (`close` no longer in the top four)
```

— so what is left is ambiguity in the **moving** phases, which is genuine per-spawn variation
rather than unobservable phase. **N2 is not merely mitigated; the term it named is gone.**

### 7.5 The v3 retrain — and what actually improved

Three training seeds on `demos_v3`, evaluated on the same 768 held-out episodes, against the
**expert re-run with the same shortened holds on the same episodes**:

```
                        mean      sd     min      max     range
BC on demos_v1         68.1 %    9.8p   58.1 %   77.7 %   19.7p
BC on demos_v3         71.8 %    2.0p   69.7 %   73.7 %    4.0p
expert, v1 holds       71.4 %     --      --       --       --
expert, v3 holds       73.3 %     --      --       --       --
```

**BC on `demos_v3` sits 1.5 points below its own expert**, and every seed is above 69.7 %.

**What is and is not established, at n = 3 per arm.** The mean rose 3.7 points, and that alone
is *not* significant — Welch's t is **0.64**, because v1's own sd is 9.8. Reporting "+3.7
points" as the result would be exactly the error the P26 family made five times.

What *is* significant is the **variance**:

```
variance ratio   F = 23.6 on (2,2) df   ->   p ~ 0.041
worst seed       58.1 %  ->  69.7 %          +11.6 points
seed range       19.7 pts ->  4.0 pts        4.9x tighter
```

**Removing the ambiguity did not mainly make the policy better on average; it made it
reliable.** That is the more useful result of the two, and it is the one the evidence supports.
It also explains the v1 seed spread retrospectively: a dataset in which 54 % of the chunk
ambiguity is unobservable phase gives the optimiser 19.7 points of room to land in different
places, and one of the three landings was a policy that fails to start one episode in ten.

The taxonomy confirms the mechanism rather than merely tracking the number:

```
              no-grasp (3 seeds)          stalled                 toppled
demos_v1     0.5 / 0.3 / 9.5 %      0.3 / 0.0 / 4.0 %     30.7 / 22.0 / 28.4 %
demos_v3     0.1 / 1.2 / 0.0 %      0.0 / 0.5 / 0.0 %     28.0 / 29.2 / 26.6 %
```

**The "never picks the block up" mode is gone** — 9.5 % at v1's worst seed, at most 1.2 % now —
and so is `stalled`. Topple is unchanged at ~28 %, which is correct and expected: P27 shortened
holds, and nothing in this stage has yet touched the close-phase blade sweep. **The residual is
now, on both sides and at every seed, exactly one thing.**

### 7.6 Where Stage 2 ends

```
expert   73.3 %   frozen pose, frozen chain, frozen approach, P27 holds, under the full MDP
BC       71.8 %   mean of 3 seeds, 69.7 - 73.7, on 768 spawns none of them ever saw
gap      -1.5 points
mission  ~70 %    met, by behaviour cloning alone -- no DAgger, no RL
```

For comparison, eva_bc's flow BC lost **21.6 points** to an 85.7 % expert on the pick-place
task this pipeline was built for.

### 7.7 Video — what the failure actually looks like

`clutter/act/record_video.py` films the policy through the **same `ChunkController` the
evaluator uses** (both import `policy_runner.py`, so the film cannot drift from the
measurement). One environment, one episode per file, a camera attached to the scene cfg at
runtime — `challenge/` spawns none and is not modified — and re-aimed with
`set_world_poses_from_view` at 0.67 m with the lens narrowed from the rig's 90 deg HFOV to 60.

16 episodes of `bc_v3_s3` on single-env spawns: **11 SUCCESS, 5 `distractor_toppled`** (68.8 %,
against 73.7 % measured on 768 batched episodes — consistent at n = 16).

Two things the films add that the numbers did not.

**1. The failure is a wrist sweep, not a grasp failure.** The final frames show the gripper
having swung wide across the row with the target still correctly gripped. Nothing about the
grasp goes wrong; the arm knocks a neighbour over on its way out.

**2. Every topple fires in a six-step window.** Termination steps across five independent
spawns:

```
171   171   173   176   177        env steps
```

Against the v3 schedule — approach 0-77, settle 78-79, descend 80-154, predwell 155-159,
close 160-164, carry 165-278 — that is **6 to 12 env steps into the carry**, 0.12-0.24 s after
the fingers shut.

**Read that with care.** `distractor_toppled` fires when a block's up-axis passes
`TOPPLE_DOT = 0.75`, i.e. ~41 deg of tilt, which takes time to accumulate. The termination step
is a **lagging** indicator and the disturbance onset is earlier — consistent with Stage 1's
close-phase attribution rather than a correction to it. What *is* new is the tightness: five
independent spawns terminating within six steps says the trigger is a specific repeatable
moment in the manoeuvre, not contact noise accumulating at random. That is a good property for
whatever fixes it, and it suggests the lift-off transition is where to look first.

---

## 8. What is retracted or superseded by this file

* **R19 — "the joint-space `home → chain[0]` lerp is clean."** P29's registered prediction.
  Refuted: 4.85 mm of penetration, 100 % approach hazard, 5.5 % success. The premise ("the
  whole path lies above z = 125 mm") was about the TCP, and was false even of the TCP.
* **R20 — "`o_align`, penetration and endpoint agreement are enough to accept a path."**
  Arm B satisfies all three and scores 0.0 %. **A path must also agree with its neighbours in
  JOINT space at the join, not only in tool space.** Two IK branches at the same TCP are
  indistinguishable to every statistic this codebase computes.
* **R21 — N7's "`joint1` at ≈1.57 is the binding action magnitude."** It is `joint4` at
  **4.63**, and all six joints exceed 1.0.
* **R23 — "the failure taxonomy can be read off the scene at the end of the episode."** No:
  `env.step` auto-resets a terminated env from **inside** the step call, so any *last-write*
  quantity reads a fresh spawn. Bit twice — once in `collect_demos.py` (the `tele` arm
  reporting 0.0 % topple against `phys`'s 15.6 % on identical spawns) and once in
  `eval_flow.py` (final-step success reporting 0.0 % on all six seeds). **Latched quantities
  are safe; last-write-wins quantities are not.**
* **R22 — "47 % of the demo is a held pose."** With the approach segment included the demo is
  472 steps, of which 180 (**38.1 %**) are a held pose. The N2 *mechanism* is unaffected and
  is now measured directly (§5).
* **`ex.score()` is not valid under `env.step`.** It reads the scene at the end, and a
  terminated env has already re-spawned. Use the termination manager. Any future probe that
  moves from `run_physics` to `env.step` inherits this.

---

## 9. Conventions this stage adds

1. **Audit the seam, not just the segment.** P17 established "verify segments, not
   waypoints". This stage adds: where two independently solved paths meet, verify the
   **joint-space** agreement. Tool-space agreement is not agreement.
2. **When a path must end at a known pose, solve backward from it.** Forward solving finds *a*
   solution; only backward solving guarantees *the* solution.
3. **A registered prediction can be refuted by a smoke test.** The 32-env smoke run of the
   collector killed P29's prediction before P29 ran. Do not skip the smoke run to save the
   4 minutes.
4. **Measure the irreducible loss before training.** The Bayes floor of a chunk predictor is a
   nearest-neighbour calculation on the dataset, costs no GPU-minutes, and turns "the policy
   might not learn the hold" into "54 % of the ambiguity is in the close, here is the number".
   Do the derivation, though: the flow objective's floor is not the chunk MSE, and the
   per-cell bound is loose when the ambiguity is low-dimensional (§5.1).
5. **Report the constraint metric separately from the task metric.** 96 % at goal with 100 %
   topple was a real arm in a real run.
