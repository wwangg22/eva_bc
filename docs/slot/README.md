# START HERE — `Rebot-PrecisionSlot-*` with `eva_bc`

*Orientation doc. Everything else in `docs/slot/` is depth; this is the map. Written 2026-08-03.*

---

## 1. The one-paragraph version

The task: pick up a 30 mm block from a randomised spawn and insert it into a slot with 1.5 mm
per-side clearance, using the `eva_bc` framework. Target was **~70 % on random starts**. It is
**solved by behaviour cloning alone** — a flow-matching policy trained on scripted-expert demos
scores **0.979** on `-v0`, **0.927** on `-Loose-v0` and **0.969** on `-Tight-v0`. No RL was
needed for the headline. Everything since has been about *understanding and stressing* that
policy — how it behaves when the world is not the world it trained on.

**The champion is `slot/runs/bc_armB_seed0/ckpt_final.pt`.**

---

## 2. The five numbers worth remembering

| what | number | where |
|---|---|---|
| champion, `-v0`, later cohort | **0.979** | `EXP_BC_ARMS.md` |
| the goal | 0.70 | `PLAN.md` |
| lateral slot shift it survives | **≤ 0.67 × clearance**, then a cliff | `EXP_ROBUSTNESS.md` §8 |
| slot moved 20 mm further away | 1:1 tracking, −8 pts | `EXP_ROBUSTNESS.md` §7 |
| **2 % actuation noise** | **0.979 → 0.146** | `EXP_ROBUSTNESS.md` §10 |
| same noise, one good flow latent | **0.146 → 0.823**, *no training* | `EXP_STEER.md` §11 |

The last row is session 7's result and the most surprising thing the project has produced.

---

## 3. Which doc to read, by question

| if you want to know… | read |
|---|---|
| **what happened this session** | `EXP_STEER.md` — read its Status block, then §8 → §9 → §11 → §12 → §13 |
| the current state, what is running, what is next | `HANDOFF.md` §1, §1-NEXT, §1a |
| how robust the policy is to a changed world | `EXP_ROBUSTNESS.md` (long; §8 and §13b are the load-bearing bits) |
| whether the clearance is what makes it hard | `EXP_TIGHT.md`, then `EXP_DEPTH.md` §6 |
| why the demos are noised the way they are | `EXP_NOISE_SWEEP.md` — the DART work |
| how the scripted expert works and what it can/cannot reach | `EXPERT_PORT.md`, `EXPERT_RESULTS.md` |
| which `eva_bc` file became which `slot_act/` file | `PORT_MAP.md` |
| the plan and its protocol rules | `PLAN.md`, `HANDOFF.md` §10 |
| **mistakes that produced confident wrong answers** | `HANDOFF.md` §9 (gotchas) — the most useful section in the repo |
| a per-session narrative | `SESSION3/4/5/6_WRITEUP|FINDINGS.md` |

If you read exactly two things: **this file**, then **`EXP_STEER.md`'s Status block**.

---

## 4. The story, one line per stage

1. **Feasibility.** The arm cannot grasp top-down below z = 0.19 m, so the whole task is
   horizontal. Established a scripted expert that inserts 100 % of the time.
2. **Demos.** Collected with DART-style action noise — but only in **free space**, because
   inside the 1.5 mm channel a noised command drives the block into a wall and levers it out of
   the gripper. Chose `noise_std = 0.05` by a pre-registered rule.
3. **BC.** Ported `eva_bc`'s ACT/flow stack to a 34-D obs / 7-D action task (`slot_act/`).
   Six arms × seeds; the champion hit 0.979 and the goal was met without RL.
4. **Robustness (session 6).** Moved the slot, widened the spawn box, jittered the arm start,
   injected sensor and actuation noise. Headline: the policy **tracks a moved goal 1:1** out to
   20 mm past its data, **arm start pose is free**, lateral tolerance is a **step function at the
   clearance**, and **actuation noise is catastrophic** while sensor noise is nearly free.
5. **Steering (session 7).** Aimed to *learn* the flow's integration noise `x0` with PPO to
   recover the actuation-noise deficit. A cheap probe run first found that a **constant** `x0`
   already recovers it — and that the steering parameterisation **cannot express** such a latent.
   The PPO run was withdrawn before it was launched. See §5.

---

## 5. Where session 7 actually landed

* **The harness is verified to the strongest standard the project has used.** The steering
  evaluation path reproduces the ordinary one **episode-for-episode**, 96/96, zero mismatches.
* **The 83-point actuation-noise deficit is largely a *sampling* failure, not a capability
  failure.** The flow's chunk is a deterministic function of `(observation, x0)`. Holding `x0` at
  a good constant draw scores **0.823** where redrawing it every refill scores 0.146. Failure
  mode is identical across every latent — the block is carried to the staging waypoint and the
  push is simply never attempted; only its *frequency* changes.
* **But it is not a free lunch, and the data says so.** That same latent costs **14.6 points**
  when there is no noise (0.833 vs 0.979) and buys **nothing** at 5 % noise (0.000). A fixed
  latent is a condition-specific patch.
* **The pre-registered PPO design could not have worked.** `SteerCore` broadcasts one 7-vector
  across all 50 chunk positions; every broadcast latent scores **0.000/96**, two of them by never
  lifting the block. The good latents are not in the action space at all.
* **Selection over latents is nearly saturated.** A perfect per-episode chooser over 9 latents
  scores 0.885 against 0.823 for always using the best one — **+6.2 points** — and that ceiling
  did **not move** when the candidate pool nearly doubled.
* **The live thread is `‖x0‖`, not which latent.** Success rises monotonically with the latent's
  magnitude — 0.302 / 0.573 / 0.771 / **0.875** at 0.30× / 0.76× / 1.00× / 1.50× — and had not
  turned over at 1.5×, which is well outside the prior's typical shell. Scaling a *freshly drawn*
  `x0` needs no search and no chosen constant; that sweep is written
  (`scripts/run_x0_norm.sh`) and **not yet run**.

---

## 6. Where things live

```
slot/
  slot_act/          the ported policy stack (train_flow, eval_act, steer_*, noise.py)
  slot_mdp.py        the task-side surface eval/expert code reads (never edit eva_rl)
  scripts/           every experiment is a script here; all are idempotent
  analysis/          read-back tools -- each answers one pre-registered question
  runs/bc_armB_seed0/   the champion, its evals, and videos/
docs/slot/           this file and everything in the table above
```

### Clips worth watching (single workstation, one env, one episode)

`slot/runs/bc_armB_seed0/videos/`, produced by `scripts/make_single_env_videos.sh`. Each retried
across spawn seeds until the episode genuinely had the outcome its filename claims.

| clip | outcome | final block (x, y, z) | depth |
|---|---|---|---|
| `single_success_clean.mp4` | success | (0.257, 0.000, 0.055) | **+47.4 mm** |
| `single_failure_act002.mp4` | failure, 2 % actuation noise | (0.162, 0.000, 0.061) | **−47.8 mm** |
| `single_success_act002_s4.mp4` | success, **same noise**, `--fixed-x0 4` | (0.257, 0.001, 0.055) | **+47.3 mm** |

The middle clip is the one to watch. The block ends at x = 0.162 — the staging waypoint — at
carry height, on-axis to 0.42 mm, ~48 mm short of the mouth and touching nothing. The policy
grasps, carries, lines up, and then simply holds for the remaining ~400 steps. The third clip is
the identical condition with the flow's latent held fixed, and it seats the block.

Two rules that are not negotiable, both learned the hard way:

* **`eva_rl` is a shared asset repo — never edit it.** Everything task-side is worked around it.
* **Score on the `later` cohort.** The first episode a process runs is worth up to +18.7 points
  (PhysX warm start). `success_rate_later` exists for this; `success_rate` does not mean what it
  looks like. See `HANDOFF.md` §9.

---

## 7. Open threads

1. **`scripts/run_x0_norm.sh`** — is the magnitude of `x0`, applied to ordinary sampling, a
   general robustness knob? Written, stopped before completion, ~30 min.
2. **`scripts/run_x0_holdout.sh <seed>`** — the 0.823 is the max of nine candidates scored on the
   same 96 episodes. It has **not** been validated on fresh spawns yet. Nothing should be quoted
   as a headline until it is.
3. **`scripts/run_obs_shift.sh`** — why is actuation noise ~40× more damaging than the same
   nominal sensor noise? Pre-registered in `EXP_STEER.md` §8d with a causal cell.
4. **The 11 episodes of 96 that no latent solves.** Not a latent problem; nothing so far
   addresses them.
