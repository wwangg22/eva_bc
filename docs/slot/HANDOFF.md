# HANDOFF — porting eva_bc to `Rebot-PrecisionSlot-*`

**Updated 2026-08-03 end of session 5. ONE GPU job still in flight — see §1a.** Authoritative
entry point. Everything here is measured unless it says otherwise; retractions are kept in place
rather than deleted, because the retracted reasoning is usually more instructive than the
correction.

Read in this order:

1. **this file** — state, rules, plan. **§1a first if you are resuming: a job is running.**
2. **§6z of this file** — the complete session-5 record: the result, the four findings that
   matter, what worked, **the eight things I got wrong**, and what was built
3. **§7 step 5z** — the forward plan, ordered by information per GPU-hour
4. **`SESSION5_FINDINGS.md`** — the full session-5 evidence: upstream review, Stage C numbers,
   expert-failure result, grasp-bit replacement, failure taxonomy, retractions kept in place
5. **`EXP_ROBUSTNESS.md`** — pre-registered, **currently running**: does the policy generalise,
   or has it memorised one geometry?
6. **`EXP_TIGHT.md`** — pre-registered then **CLOSED**; beliefs 1 and 2 refuted by the data
7. `PORT_MAP.md` — `act/` → `slot_act/` interface facts and the port's landmines
8. `SESSION4_WRITEUP.md` — session-4 record: data freeze, instrument validation, Stage C
9. `SESSION3_WRITEUP.md` — full session-3 record: every wrong diagnosis and how it was caught
10. `EXP_NOISE_SWEEP.md` — the two pre-registered noise experiments in full
11. `EXP_BC_ARMS.md` — the pre-registered Stage-C training experiment + exact runbook
12. `PLAN.md` — verdict log (session-4 rows 4.1–4.20, session-5 rows 5.1–5.30)
13. `EXPERT_RESULTS.md` — session-2 expert evidence; `EXPERT_PORT.md` — session-1, superseded

---

## 0. Working rules

```
cd /home/rei/Desktop/isaaclab/eva_bc/slot
source /home/rei/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6
```

- **Run python from `/home/rei/Desktop/isaaclab/eva_bc/slot`.** `reBot_RL` resolves from the
  installed package; scripts add `eva_bc` to `sys.path` themselves.
- Address the user as **Big Will**. He reviews all rendered output — **never view
  images/videos yourself**; hand him the path.
- **ONE GPU job at a time.** 11 GB RTX 2080 Ti. Check `nvidia-smi` and
  `ps aux | grep "[p]ython"` (NOT `pgrep -f`). Camera rendering above ~256 envs throws
  `CUDA error: an illegal memory access was encountered` — use ≤ 32 envs for video.
  **I broke this rule in session 4 and it cost real time.** Memory was never the problem (910 MB
  training + 1.5 GB eval against 11 GB) — *compute* was: the GPU sits at 96–99 % during training,
  so a concurrent eval went from **2 m 11 s to over 10 minutes** and dragged training from 47.7
  to 42 steps/s at the same time. Total GPU work is conserved at best and slightly worse in
  practice. Run jobs sequentially; the only reason to overlap is to get information *earlier*,
  and when the user is AFK that is worth very little.
- **Do not pass `--headless`** — deprecated in Isaac Lab 3.0, headless is the default.
- Long jobs = background chains with durable logs + a `Monitor` on the log.
- Filter log noise: `grep -vE "Warning\]|DeprecationWarning|absl::|descriptor_database|Cloning joints"`.
- **Do not touch tracked files in either repo.** Verified at end of session 3: `eva_bc`
  `git status --porcelain` shows only `?? docs/slot/` and `?? slot/`; **`eva_rl` is completely
  clean**. **Big Will has NOT authorised a push.**
- `eva_rl` is a *shared* asset repo — `pick_place/` sits behind an existing 87.9 % result and
  the challenge env is someone's authored task. Never edit it; evaluate around it (§3c).

---

## 1. Where things stand

| stage | status |
|---|---|
| **A** feasibility | **DONE** (session 1–2) |
| **B** scripted expert | **DONE — 99.9 % nominal (1023/1024)** (§2) |
| **B2** demo collection | **DONE and FROZEN — 16 pools, 2038 demos, 1977 successful (97.0 %)**, all 16 verified (§5) |
| **C** flow-matching BC | **DONE. Objective cleared on ALL THREE clearances**, pooled over 3 training seeds × 2 spawn seeds (n=1152/arm/task): Loose **0.778 / 0.799**, v0 **0.776 / 0.792**, Tight **0.708 / 0.764** (arm A / arm B). DART: **no measurable difference**. `SESSION5_FINDINGS.md` §6d |
| **D** RL on frozen base | not started, **not needed for the objective** (BC clears 70 % at every clearance), but now **measurably motivated**: the x0 choice alone spans a **54-point range** across checkpoints (`EXP_TIGHT.md` §7d). Aim it at **depth** — 82–84 % of failures at every clearance are the block stopping short in x with lateral alignment fine (§6d-ii). Startup blocker closed (§4a) |
| **E** PPO-from-scratch control | not started; the env's Factory reward has **never** been run |

**Session 4 status.** Data frozen and verified; both instruments validated (`test_pipeline_cpu.py`
17/17, `check_eval_json.py` 14/14); the Stage C sweep is running (~35 min per run, ~3.5 h total)
with the 36-eval sweep chained behind it. **See §1a for exactly what is in flight and how to pick
it up.** Full detail in `SESSION4_WRITEUP.md`.

**What session 4 built** (all new, all untracked):

| file | what it is for |
|---|---|
| `scripts/test_pipeline_cpu.py` | 17-check CPU validation of the whole training path, 40 s, no sim |
| `scripts/check_eval_json.py` | audits an eval JSON for accounting errors — cohort split, horizon, rate/record agreement |
| `scripts/run_stage_c.sh` | one seed, both arms, `--pool success` |
| `scripts/run_eval_sweep.sh` | Loose/v0/Tight × 2 spawn seeds, skips existing outputs |
| `scripts/summarize_arms.py` | the pre-registered decision rule in code, + Wilson CIs + pairing check |
| `scripts/diag_feedback.py` | mid-`reach` block teleport: clock vs feedback, with two controls |
| `scripts/make_videos.sh` | one video per clearance for a checkpoint |
| `analysis/label_consistency.py` | label-noise floor and demo coverage width |

**What session 5 added:**

| file | what it is for |
|---|---|
| `scripts/test_steer_cpu.py` | 17-check CPU validation of the **x0-steering** path, ~20 s, no sim — including EXP07's `z = 0` bit-exactness gate |
| `analysis/paired_evals.py` | McNemar over eval JSONs sharing a spawn seed; distinguishes *matched* spawns from *unverifiable* ones |
| `analysis/failure_taxonomy.py` | sorts failures into mechanism buckets by decision list; `--pool` aggregates across the sweep |

and changed, in `slot_act/`: `--pool success` + `success_pool_filter`; `--episode-length-s`
default 30 → 12; the post-reset diagnostics bug; `spawn_pos` recorded per eval episode; the MLP
`GraspBit` replaced by the analytic `SlotGraspBit` (and `grasp_bit_path` dropped from five call
sites); `run_eval_sweep.sh` hardened against `pipefail` aborting the whole sweep.

**The headline early result — the learning curve so far, all on `-v0` (1.5 mm), `success_rate_later`:**

| training steps | data | later-cohort success | n |
|---:|---|---:|---:|
| 2 000 | 256 demos (throwaway, 40 s) | **37.5 %** | 32 |
| 10 000 | arm A, 1023 demos | **61.5 %** | 96 |
| 30 000 | arm A, 1023 demos | 89.6 %  (95 % CI [0.819, 0.942]) | 96 |
| 50 000 | arm A, 1023 demos | 85.4 %  (95 % CI [0.770, 0.911]) | 96 |
| 100 000 (`ckpt_final`) | arm A, 1023 demos | **92.7 %**  (95 % CI [0.856, 0.964]) | 96 |

Paired McNemar on the same 96 spawns says **30k / 50k / 100k are mutually indistinguishable**
(χ² ≤ 2.12; 30k and 100k agree on 81 of 96 episodes). The curve **plateaus at 30 k steps** —
30 k would be enough for the *next* sweep. `analysis/paired_evals.py`, PLAN 5.3.

This task is far more learnable from state observations than the pick-place history suggested.
The pre-registered Stage C bar (≥ 40 % v0) was already cleared at **10 k steps**, so it is
uninformative. **The 70 % project target is cleared too — 92.7 % at `ckpt_final`, by BC alone, on `-v0`,
with no RL and no DAgger; 89.6 % of it arrives in the first ~12 minutes of GPU.** What remains is confirmation across the other
two clearances and a second spawn seed (the sweep), and then `-Tight-v0`. My pre-registered
prediction that BC will *not* clear Tight is on the record — and so is the fact that it is
**confounded**: see `SESSION5_FINDINGS.md` §5c, and §7 step 5 for what to do instead of trusting
it. At 47 steps/s, 30 k steps is 12 minutes of training.

Note the **first-episode bias is not monotone and not even signed**: +18.7 pts at 2 k
(0.562 vs 0.375), +1.0 at 10 k, +10.4 at 30 k, +11.5 at 50 k, and **−2.1 at 100 k**
(0.906 vs 0.927). Session 4 read the early points as "the bias shrinks as the policy improves";
the fuller curve does not support that, and the honest statement is just that it moves
unpredictably with policy × task. It certainly does **not** license dropping the cohort split —
a pooled `success_rate` would have been wrong by between −2 and +19 points depending only on
which checkpoint you happened to look at.

**Three defects were found and fixed before any real training ran** (§7 step 2, and
`SESSION4_WRITEUP.md` §2 and §6) — all three would have produced a plausible *number* rather
than an error:
* `eval_act.py --episode-length-s` defaulted to **30 s**, 2.5× this task's 12 s horizon;
* `train_flow.py --pool` defaulted to **no filter**, and the obvious alternative (`nominal`)
  would have silently halved arm B;
* `depth_mm` / `lateral_mm` / `yaw_rad` were read **after `env.step`**, i.e. after Isaac Lab
  resets done envs — so they described the *next* episode's spawn. The success rate was always
  right; only the fields you would use to *explain* it were fiction.

---

## 1-NEXT. ▶ START HERE NEXT SESSION — EXP_STEER

Big Will's ask at the end of session 6, verbatim: *"lets see if we can use steering to actually
teach it behavior it hasn't seen before (i.e. the noise stuff you are mentioning)."*

**Pre-registered in `EXP_STEER.md` (read it before running anything).** Plumbing is done and
verified; the honest framing, the beliefs and the decision rule are all written down.

**The first task is NOT to launch training.** It is:

1. ✅ **DONE (session 7, 17:20).** `--obs-noise` / `--action-noise` plumbed into `eval_steer.py`,
   plus `episode_index_in_env` and the first/later cohort split it was missing. The sigma
   construction now lives once, in **`slot_act/noise.py`**, used by all three harnesses.
   `test_steer_cpu.py` grew five checks pinning its semantics (**22/22 pass**).
2. ⚠ **CORRECTED — see `EXP_STEER.md` §8a.** The gate cannot be read against **0.146**.
   `robust_act002.json` records `"fixed_x0": null` — a *fresh x0 draw every refill*. Zero-z
   steering is `x0 = tanh(0) = zeros`, a *deterministic* policy. Different policies; no reason
   the rates should match. The reference must be **x0-zeros under noise**, which did not exist
   and is now cell C1 of `scripts/run_steer_gate.sh`. And the corrected gate is **stronger**:
   with `fixed_x0 = zeros` neither harness draws a randn for x0, so C1 and C2 should agree
   **episode-for-episode**, not merely in rate.
3. ⚠ **NEW STEP INSERTED — `EXP_STEER.md` §8b.** Before any PPO, run the **constant-x0 probe**:
   `--fixed-x0 {zeros,1,2,3,4}` under `--action-noise 0.02`. Belief 2's whole mechanism is mode
   collapse at staging; if that is right, *some* x0 selects the push. A constant draw is the
   weakest possible steerer (it cannot condition on state), so this is a **lower bound** on
   trained steering and a **direct test of the mechanism**, at ~1 % of a PPO run's cost.
   **If the spread across the five draws is under 5 points, do not launch PPO** — the chunk is
   essentially independent of x0 under noise, there is no mode to select, and the deficit is a
   capability failure rather than a sampling one.
4. ❌ **THE ARM-A LAUNCH IS WITHDRAWN — do not run it.** See `EXP_STEER.md` §12. The probe and
   the broadcast cells settled it before any GPU-hour was spent on PPO:

   * A **constant structured x0** takes the champion from **0.146 → 0.823** under 2 % action
     noise, with no training at all (spread across 5 latents: **79 points**, p = 6 × 10⁻²¹).
   * But `SteerCore.set_steer` **broadcasts one 7-vector across all 50 chunk positions**
     (`.expand`, `steer_core.py:58`), and that family is *fatal*: `b1` and `b2` — row 0 of a
     good matrix, repeated, at the identical norm 18.71 — both score **0.000/96**, failing as
     `gross_miss` with the block **182 mm off axis**. Structure, not norm, is the variable.
   * So steering's reachable set is `z = 0` → 0.167 (where PPO initialises) and `z ≠ 0` → 0.000.
     **The good latents are not in the action space.** PPO would produce a flat curve and the
     wrong conclusion.

   Replacement, in priority order: **(a)** the held-out check on the winning constant
   (`scripts/run_x0_holdout.sh <best_seed>`), **(b)** `scripts/run_obs_shift.sh` for the
   mechanism, **(c)** only if the oracle headroom grows past its measured **+6.2 points**,
   re-parameterise steering per chunk position or as episode-level latent selection.

Gate + probe are one idempotent script (7 cells, ~25 min), read back by one report:

```
bash scripts/run_steer_gate.sh
python analysis/steer_report.py
```

**Why this target and not the old one.** Stage D was aimed at `-Tight-v0` precision. Session 6
killed that: clearance moves the headline ~5 points across a 6× range, and on Tight the
`stalled_in_mouth` failures have the *same* lateral distribution as the successes (0.37 vs 0.34
median). There was nothing to steer toward. The noise condition has **83 points of headroom**
(0.979 → 0.146 at 2 % action noise), a **named failure state** (frozen at x = 0.166 m, y = 0.0000,
carried height), and a **dense reward that is live exactly there** (`kp_baseline`/`coarse`/`fine`
at weights 6/12/20 grade the whole staging→seated corridor).

**The honest limit, and it must be in any write-up:** x0-steering re-weights among modes the flow
policy already represents. It cannot synthesise a behaviour outside its support. So this does not
test "teach unseen behaviour" — it tests **how much of "the policy can't do this" is really "the
policy samples the wrong mode"**. Both answers are worth having, which is the argument for
running it. The `x0` choice alone already spans **54.2 points** across checkpoints
(`EXP_TIGHT.md` §7), so the knob is known to have leverage.

**Guards that must not be skipped:** the gate (belief 1); a clean-condition control arm (belief 4
— a *large* clean-arm gain is a red flag, not a success); and **two seeds minimum** before any
claim. This session retracted two single-seed claims already (§9d, EXP_DEPTH §8).

---

## ▶ 1a-s8. START HERE — SESSION 7+8 (2026-08-03 20:35)

**The full record is at the BOTTOM of this file: "SESSION 7+8 FULL RECORD".** Read `S8-STATE`
first — vision collection is RUNNING and the next stage is one command.

Two efforts this session:
* **EXP_STEER** — finished, PPO launch **withdrawn**. Headline: a constant flow latent takes the
  champion 0.146 -> 0.823 under 2 % action noise with no training at all. Still **unvalidated out
  of sample**. See `S7-RESULT`.
* **The VISUAL policy** — Big Will's live ask, 80 % bar, no privileged info. Stack ported from
  EXP08, cameras attached, render configuration decided by measurement, 256 champion episodes
  collecting now. See `S8-a` ... `S8-f`.

Also read `docs/slot/README.md` (the orientation map) and `docs/slot/VISION_PLAN.md` (the vision
pre-registration + results).

---

## 1a-s7. SESSION 7 STATE (2026-08-03 18:40) — superseded by 1a-s8 above

**Pushed.** Commit `b1540a8` is on `origin/main` (`git@github.com:wwangg22/eva_bc.git`), 122
files / 1.3 MB: `slot/` source + scripts + analysis + `docs/slot/`. The push was **not** a
fast-forward — two `clutter/` commits had landed on the remote in the meantime; verified disjoint
from `slot/`, rebased, pushed. Git identity was unset on this machine and is now set
**repo-locally** to `william <whw2112@columbia.edu>`, matching the repo's prior commits.

**Not tracked, deliberately:** `.gitignore` excludes `runs/`, `*.pt`, `*.mp4`, `*.hdf5`, so **the
eval JSONs every result in these docs cites are not in the repo** (116 files, 7.9 MB). Offered to
Big Will; awaiting a decision. Until then the numbers are reproducible from the scripts but not
verifiable from a clone.

**Running / queued:**

```
bash scripts/make_single_env_videos.sh   # 1-env clips, running at 18:40
bash scripts/run_x0_norm.sh              # STOPPED mid-run by request; idempotent, ~30 min
bash scripts/run_x0_holdout.sh 4         # NOT RUN -- 0.823 is still unvalidated out of sample
bash scripts/run_x0_bcast_ladder.sh      # NOT RUN -- the path PPO would have walked
bash scripts/run_obs_shift.sh            # NOT RUN -- EXP_STEER §8d + the causal cell
```

**The one thing that must not be quoted yet:** the 0.823 is the maximum of nine latents scored on
the *same* 96 episodes. `run_x0_holdout.sh` has not run. Between-latent spread is 20× the
within-cell SE so the *ranking* is sound, but whether that latent suits *these spawns* is exactly
what has not been tested.

`eva_rl`: **0 changes**, as throughout.

---

## 1a. ✅ NOTHING IS RUNNING — session 6 (EXP_ROBUSTNESS + EXP_DEPTH) is COMPLETE

45 perturbed evaluations + 4 expert runs, all finished 09:00–12:10 on 2026-08-03. GPU is free.
Every script is **idempotent** (skips existing outputs), so re-running any of them is safe:

```
bash scripts/run_robustness.sh      # round 1: slot dx/dy, spawn box              9 cells
bash scripts/run_robustness2.sh     # round 2: dy ladder, arm jitter, obs noise  12 cells
bash scripts/run_robustness3.sh     # round 3: replication on bc_armA_seed0       6 cells
bash scripts/run_dy_crossed.sh      # dy x clearance, the decisive crossed test   4 cells
bash scripts/run_spawn_yaw.sh       # spawn cells re-collected with spawn_yaw     2 cells
bash scripts/run_action_noise.sh    # actuation noise                             3 cells
bash scripts/run_horizon.sh         # EXP_DEPTH probe A                           2 cells
bash scripts/run_horizon2.sh        # probe A replication (s888 + 30 s)           3 cells
bash scripts/run_expert_dx.sh       # expert control                              4 runs
bash scripts/make_robust_videos.sh  # 2 clips for Big Will
```

Read any perturbed cell with the purpose-built instrument, which picks its own test **and its own
per-task baseline**:

```
python analysis/robustness_report.py --run runs/bc_armB_seed0
python analysis/robustness_report.py --run runs/bc_armA_seed0
python analysis/lateral_by_bucket.py            # aiming failures vs push failures
python analysis/failure_taxonomy.py 'runs/bc_armB_seed0/robust_*.json'
```

**Full write-ups: `SESSION6_FINDINGS.md` (start here), `EXP_ROBUSTNESS.md` §6–§13,
`EXP_DEPTH.md`.** The five-line version:

* **The policy tracks a moved goal.** Block's *absolute* final x follows the slot 1:1 (0.2570 /
  0.2620 / 0.2670 at dx = 0 / +5 / +10). It has not memorised a path. dx = +20 mm costs ~8 pts,
  replicated on two seeds, and the **expert does it 128/128** — a real, attributable target.
* **Slot y is a step function at the clearance.** 12 cells across 3 clearances collapse onto one
  curve in (shift ÷ clearance): ≤ 0.67 free, 1.00 → 0.760, ≥ 1.33 → floor. The same 2 mm shift is
  0.000 on `-v0` and 0.938 on `-Loose-v0`. **Geometry, not the policy** — and unfixable by RL.
* **Arm start pose ±0.1 rad is free** (−2.1 pts, p = 0.41) despite zero training coverage.
* **Actuator noise is catastrophic where sensor noise is free**: 0.05 → 0/96 vs 0.938. Chunking
  low-passes an obs error over 15 steps and does nothing about an action error.
* **Under noise the policy freezes at the staging waypoint** — block held at x = 0.166 (expert
  `stage_x` = 0.165), y = 0.0000, carried height, 81–92 of 96 episodes. The push is never
  attempted. **This is the bottleneck, stated concretely for the first time.**

**A correction to the note that used to be here:** comparisons are *not* automatically unpaired.
Shifting `SLOT_CENTER` consumes no RNG, so every dx/dy cell is spawn-for-spawn **paired** with the
gate and gets an exact-binomial McNemar. `--spawn-scale`, `--obs-noise`, `--action-noise` and
`--arm-jitter` do break pairing and get a two-proportion z-test. `robustness_report.py` decides
per cell from the recorded `spawn_pos` and prints which test it used and why — never assume.

**Two claims were retracted after replication, both mine, both written up in place:**
`dx_m010` = −13.5 pts (round 1, p = 0.001) → −1.0 pts on a second seed (§9d). And EXP_DEPTH §7's
horizon *differential* → reverses at the second spawn seed; pooled it is +4.9 pts, p = 0.14 (§8).

---

## 1a-done. ✅ EVERYTHING ELSE FINISHED (06:41)

Every chain finished. GPU is free. Artifacts on disk:

* **36/36 sweep evals**, zero failures (`runs/bc_arm*/eval_ckpt_final_*.json`)
* **6 fixed-x0 Tight evals** + 2 fixed-x0 wide-clearance evals (`eval_x0zeros_*`)
* **3 champion videos** for Big Will, `runs/bc_armB_seed0/videos/`:
  `Rebot-PrecisionSlot-Loose-v0.mp4`, `Rebot-PrecisionSlot-v0.mp4`,
  `Rebot-PrecisionSlot-Tight-v0.mp4`

**Result: the objective is met.** Pooled over 3 training seeds × 2 spawn seeds (n=1152 per arm
per task): Loose **0.778 / 0.799**, v0 **0.776 / 0.792**, Tight **0.708 / 0.764** (arm A / arm B)
— all three clearances above the 70 % target, by BC alone. DART: **no measurable difference**.

**Next, in priority order** (all optional — the objective does not depend on them):

1. `scripts/diag_feedback.py` — clock vs closed-loop, still never run. §3h.
2. Stage D x0-steering aimed at **depth** — motivated by the measured 54-point x0 spread
   (`EXP_TIGHT.md` §7d), discounted by belief 5. Unblocked and CPU-validated 17/17.
3. Stage E — PPO from scratch on the env's Factory reward, never run.

The historical §1a follows, kept because its contention measurements are still the evidence for
the one-GPU-job rule.

## 1a-history. ⚠ WHAT WAS RUNNING (written 2026-08-03 ~01:00, updated 01:25)

Two long GPU chains were launched during session 4 and are **still in flight**. Read this before
launching anything, or you will double-book the GPU.

> **01:30 UPDATE — the headline moved before the sweep finished.** Run 1 of 6 (arm A, seed 0)
> completed at 01:11; arm B seed 0 is training. Its **`ckpt_final` scores 92.7 % later-rate
> (89/96, 95 % CI [0.856, 0.964]) on `-v0`** — and 89.6 % already at 30k steps.
> **The 70 % project target is cleared on `-v0` by BC alone**, with the whole confidence
> interval above it, pending confirmation across tasks/spawn seeds by the sweep. Longer
> training is *better* (92.7 > 89.6), so the pre-registered `ckpt_final` is also the best
> checkpoint — no tension between the rule and the result. Full analysis, the failure taxonomy, the
> expert-failure result and a pre-registered prediction for the clearance ladder:
> **`SESSION5_FINDINGS.md`** — read it before interpreting the sweep.
>
> Checkpoint selection was **pre-registered before arm B was seen** (SESSION5 §1b): the arm
> comparison uses `ckpt_final`, full stop. `ckpt_0030000` beating it would be a *training-length*
> finding, reported separately and confirmed on spawn seed 888 — not a new selection rule.
> No run needs shortening: `--save-every 10000` already keeps every intermediate checkpoint.

> **04:12 UPDATE — CHAIN 1 IS DONE.** All 6 training runs completed at **04:11:13**, each
> 100 000 steps with 11 checkpoints. Chain 2 (the 36-eval sweep) fired automatically at 04:12:19
> and is running; ETA ~06:00. Training health across all six: final-1k loss 0.030–0.038, no
> divergence. Throughput **48.0 steps/s** on the four runs that had the GPU to themselves versus
> **39.3 / 40.1** on the two I overlapped evals onto — an 18 % training-side tax on top of the
> eval going 2 m 11 s → 12 min. That is the "one GPU job at a time" rule, measured on both sides.

### Chain 1 — the Stage C training sweep
```
for s in 0 1 2; do bash scripts/run_stage_c.sh $s 100000; done   >> logs/stage_c.log
```
Runs **arm A then arm B** for each seed, in that order, so `runs/bc_armB_seed2/ckpt_final.pt` is
the last artefact produced. Started **00:27:44**; at ~47 steps/s uncontended a run is ~35 min, so
the whole sweep lands around **04:00–04:30**.

Check progress with:
```
tail -1 logs/stage_c.log                 # {"step": N, "loss": ..., "sec": ...}
grep -c "=== arm" logs/stage_c.log       # how many of the 6 runs have STARTED
ls -d runs/bc_arm*                       # which run dirs exist
```

### Chain 3 (new, session 5) — EXP_TIGHT step B, chained behind chain 2
```
until [ 36 eval_ckpt_final JSONs ]; do sleep 120; done
bash scripts/run_x0_determinism.sh          # 3 evals, ~11 min
```
Runs `bc_armA_seed0/ckpt_final.pt` on all three clearances at spawn 777 with **`--fixed-x0
zeros`** — the flow's integration noise frozen at the distribution's mode, so each chunk is a
deterministic function of the observation. Tests `EXP_TIGHT.md` belief 2: that what fails on the
0.5 mm channel is the policy's *own sampling noise*, not the learned trajectory. Same
`--num-envs/--episodes/--seed` as the sweep, so every cell is **paired** — read it with
`analysis/paired_evals.py --label-from-path`, not with independent intervals.

Uses armA_seed0 as a fixed reference rather than "the champion" on purpose: its stochastic
baseline already exists at all three clearances at s777, and the mechanism under test does not
depend on which arm wins.

### Chain 2 — the evaluation sweep, chained to fire when chain 1 finishes
```
until [ -f runs/bc_armB_seed2/ckpt_final.pt ]; do sleep 120; done
bash scripts/run_eval_sweep.sh ckpt_final.pt
```
36 evals: 6 checkpoints × 3 tasks (Loose / v0 / Tight) × 2 spawn seeds (777, 888), each
`--num-envs 32 --episodes 128`. ~4 min each uncontended → ~2.5 h → done around **07:00**.
The script **skips any output JSON that already exists**, so it is safe to re-run.

### Chain 3 — learning-curve evals (2 of 3 landed as of 01:25)
Three evals of `bc_armA_seed0` at `ckpt_0010000/0030000/0050000` on `-v0` at seed 777, plus a
separate one-off `ckpt_final` eval. These were launched *concurrently with training*, which was a
mistake (see §0). They are diagnostic only and are **not** part of the arm comparison.

| train steps | later rate | n |
|---|---|---|
| 2 000 | 0.375 | — |
| 10 000 | 0.615 | 96 |
| 30 000 | 0.896 | 96 |
| 50 000 | 0.854 | 96 |
| 100 000 (`final`) | **0.927** | 96 |

The contention cost was measured again and is worse than session 4 recorded: with three GPU jobs
resident the `ckpt_0030000` eval burned **> 12 min of CPU against a 2 m 11 s solo baseline**.
Memory was never the constraint (4 GB of 62 GB system, ~4 GB of 11 GB VRAM). **One GPU job at a
time.**

### If a chain died
Everything is idempotent. Re-run `run_stage_c.sh <seed>` for any missing run dir, then
`run_eval_sweep.sh`. Nothing is destroyed by re-running.

### ✅ RESOLVED — the cleanup item below is done (checked 01:29, `spawn_pos` present)

### ⚠ One cleanup to CHECK (not assume)
`runs/bc_armA_seed0/eval_ckpt_final_Rebot-PrecisionSlot-v0_s777.json` is written by a separate
one-off early eval, not by the sweep — and the sweep **skips outputs that already exist**. It
should carry `spawn_pos` (the field was added before that job's python process started), but
verify rather than trust the timing:

```
python scripts/check_eval_json.py runs/bc_armA_seed0/eval_ckpt_final_Rebot-PrecisionSlot-v0_s777.json --expect-envs 32
```

The last lines say either `spawn_pos present` or `spawn_pos absent`. If absent, **delete the file
and re-run that one eval**, otherwise one cell of the pairing check (§3f) is blind. Nothing else
depends on it.

The three learning-curve JSONs (`eval_ckpt_00X0000_*`) are diagnostic only; `summarize_arms.py`
would warn about mixing checkpoints if they were globbed in, so glob `eval_ckpt_final_*` for the
arm comparison.

### When it is all done, in order
```
python scripts/summarize_arms.py 'runs/bc_arm*/eval_ckpt_final_*.json'   # the pre-registered rule
python analysis/paired_evals.py 'runs/bc_arm*/eval_ckpt_final_*_s777.json' --label-from-path
python scripts/check_eval_json.py runs/bc_armB_seed2/eval_ckpt_final_*_s777.json --expect-envs 32
bash scripts/make_videos.sh runs/<champion>/ckpt_final.pt      # 3 videos, hand Big Will the paths
python scripts/diag_feedback.py --ckpt runs/<champion>/ckpt_final.pt --num-envs 16 --episodes 64 \
       --perturb-step 20                                       # then -1 and --resample-only
```
Read `summarize_arms.py`'s **per-training-seed spread before the arm gap** — a gap that does not
clear the spread is not a result. Then go to §7 step 5 for Tight; the `-Tight-v0` cells of this
same sweep are the input to that decision, so **no extra GPU run is needed to start it**.

Grep the sweep log for `!!! EVAL FAILED` before trusting completeness — the sweep continues past
a failed cell by design (PLAN 5.14), so a missing JSON is a *skip*, not a silent success.

---

## 2. The expert

**100 % seated on all three clearances**, measured in a fresh process at n=128:

| task | clearance | seated | lateral p90 | \|yaw\| p90 | depth p10 |
|---|---|---|---|---|---|
| Loose-v0 | 3.0 mm | **128/128** | 1.38 mm | 0.0198 rad | 46.2 mm |
| **v0 (target)** | 1.5 mm | **128/128** | 1.11 mm | 0.0109 rad | 46.2 mm |
| Tight-v0 | 0.5 mm | **128/128** | 0.48 mm | 0.0037 rad | 46.2 mm |

Confirmed at scale: the four nominal v2 pools are **512/512 = 100 %** across four independent
seeds. Grip held 128/128 through every phase at a constant 29.95 mm finger gap; block z exactly
55.0 mm; 558/600 steps.

```
python scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 128
python scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 16 --video
```

Video for Big Will: `slot/logs/expert/expert_Rebot-PrecisionSlot-v0.mp4` — **note this predates
the `retreat` phase and should be regenerated** if he wants to see the current behaviour.

### 2a. Trajectory design, and what forced each choice

`reach → grasp(close) → lift → back → spin → turn → push → release → retreat → idle`.
All IK is solved **before the block is touched**, then executed open-loop.

| param | value | why |
|---|---|---|
| `grasp_h` | 0.031 | bounded below by wall clearance (a *loaded* gripper clears the walls only at TCP z ≥ 0.090) and above by the block's 35 mm half-height |
| `carry_z` | 0.095 | at 0.090 the carried block's bottom sat 0.9 mm above the 20 mm floor step. **53.1 % → 79.7 %** |
| `stage_x` | **0.165** | the single biggest fix. The carried block rides **~5 mm ahead of the TCP in x**; at 0.180 its nose reached the wall faces at 0.210 |
| `insert_x` | 0.2545 | drives the block into the **back stop**, which squares it and removes depth variance (p10 46.2 of max 47.5) |
| spin ≠ turn | separate | rotating the wrist *while* accelerating the block sideways cost 17/64 grips |
| `retreat` | added session 3 | backs the open gripper out; block pose bit-identical before/after (256.8, −0.5, 55.0) mm |

Strategy chosen by measurement: **Cartesian +x drag at TCP z = 0.090 scores 100 %**; vertical
lower-in scores 46.9 % (a depth-margin failure — 39.9 mm against a 40 mm threshold, flat across
every tolerance bin). This overturned the env write-up's premise that the block must be dropped
in from above.

---

## 3. Findings that change how numbers must be read

### 3a. ⚠ The FIRST episode in a process scores +12.9 points (4.3 σ)

Same plan, **bit-identical** initial state (`|q_start − q_default| = 0.00e+00`), n=128, three
executions in one process, run 0 genuinely first:

```
run 0  (FIRST in process): seated 128/128 = 100.0%   depth mean 46.74 mm  min 43.71
run 1                    : seated 114/128 =  89.1%   depth mean 45.41 mm  min 30.39
run 2                    : seated 109/128 =  85.2%   depth mean 44.89 mm  min 28.93
```

The first episode's outcome distribution is also *tighter*, not merely better-centred. After
it, ~13 % of envs fail at random and near-independently each time (14 and 19 flips against run
0, but **31** between runs 1 and 2 → overlap ≈ 1).

Presumed mechanism: PhysX contact manifolds / solver warm-start caches that `env.reset()`
teleports bodies past without flushing. **Policy-independent**, so trained policies inherit it.

**Consequences:**
- **Collection:** one rollout per process. Four rollouts in one process yielded
  100 / 87.5 / 84.4 / 89.1 %; four separate processes yielded **100 % four times out of four**.
- **Evaluation — the dangerous one.** `eval_act.py` fills its record buffer from auto-resetting
  envs, so the first `num_envs` records are all first-episode records. **`--episodes 128` on
  128 envs is *entirely* first-episode and maximally inflated; 128 episodes on 16 envs is
  mostly not.** Eval config silently moves the number ~13 points. `eval_act.py` now tags every
  episode with its index-for-that-env and reports `success_rate_first_episode` and
  `success_rate_later` separately. **Compare arms on `success_rate_later` only**, with
  identical `--num_envs` and `--episodes`.

### 3b. Beyond the first episode, the sim is not reproducible

Identical actions from an identical state move the block up to **18 mm** and flip **23–25 %** of
outcomes. Expected of GPU PhysX (order-dependent parallel contact solving) but large here
because the task sits on a 40 mm depth threshold. Any A/B claim needs pooling and repeats —
this is the task-specific evidence for eva_bc's protocol rule, not an inherited convention.

**But the action labels are clean.** `plan_determinism.py` proved the plan is a **pure function
of the block pose** (bit-identical, 0.0000 mrad, back-to-back *and* after a full 599-step
episode) and that `fk()` is pure (0.000 µm). The stochasticity is entirely in the *outcome*,
never in the observation→action map the policy learns.

Corollary: **success-filtering the nominal pool selects on luck, not behaviour.** No spawn bin
predicts failure, so it drops demos essentially at random.

### 3c. `mdp.is_inserted` cannot be used bare

It bounds block z only from **below** (`z > SLOT_FLOOR_Z − 0.005`, to catch a drop). A block on
top of the 30 mm walls, or dangling in a closed gripper, passes. Measured: one probe cell scored
**93.8 % with 13.28 mm mean lateral error** — geometrically impossible in a 16.5 mm half-width
channel; another scored 100 % with a 33.86 mm finger gap (pads jammed, gripping nothing).

`slot_mdp.placed_mask` wraps the **seated** predicate (`is_inserted AND |z − 0.055| < 0.006`).
`eval_act.py` reduces that exact call into the headline number, so **this one line decides every
success figure in the project**. `inserted_raw` is recorded alongside so the gap stays visible.
The env was **not** modified.

### 3d. The arm's null space makes demos multimodal

The push-end joint configuration varies by up to **133 mrad between envs** for a TCP target
identical across all of them — a 5-DOF task on a 6-DOF arm, resolved by a warm-start chain
seeded from each env's own grasp. Demos are genuinely multimodal in joint space for the same
TCP target. The policy can disambiguate (`joint_pos` is in its observation) and flow matching is
the right family, but this is not single-mode regression.

**Session-4 update — this does NOT show up as label noise.** Measured, don't assume: see §3e.
The freedom exists; the expert never exercises it *inconsistently*, because it warm-starts every
IK solve, so the same spawn produces the same branch. The worry was reasonable and the
measurement retired it.

### 3e. The label-noise floor is 0.1 mrad — the target function is essentially clean

`analysis/label_consistency.py`. For each frame, find its nearest neighbour **from a different
demo** in std-normalized observation space, then bin the action disagreement by observation
distance. The lowest bin is the floor no deterministic policy can beat.

```
        obs-dist bin       n  |dArm| p50      p90      max   -> p50 mrad
[ 0.000,  0.009]      1,501      0.0002   0.0016   0.1161           0.1
[ 0.009,  0.066]      4,502      0.0022   0.0100   0.1151           1.1
[ 0.066,  0.232]      9,001      0.0180   0.0471   0.1516           9.0
[ 0.232,  0.454]      9,000      0.0560   0.0949   0.2571          28.0
[ 0.454,  3.963]      6,000      0.1013   0.1715   0.7310          50.6
```

**0.0002 action units = 0.0 % of the action std.** Disagreement rises monotonically and gently
with observation distance — the signature of a well-conditioned regression. The env-state-only
variant (dims 16:34) gives the same floor, so proprioception is not carrying the load either.

This is the quantitative reason a **2000-step** checkpoint already scores 43.8 %.

### 3f. Two eval-instrument properties that constrain the statistics

* **An eval at a fixed seed is bit-reproducible across processes** (0.438 / 0.562 / 0.375 twice
  over, same ckpt/seed/num-envs). **Repeating a run is not a replicate** — an error bar built
  from repeats is exactly zero and completely wrong. Vary `--seed`. This does not contradict §3b:
  that was repeated episodes *inside* one process, where PhysX caches survive `env.reset()`.
* **The first-episode bias reproduces on a learned policy:** +18.7 points (0.562 vs 0.375),
  alongside the +12.9 on the expert. Not an artefact of the scripted plan. `success_rate_later`
  remains the comparison statistic.

Because the env seed fixes the reset draws, two eval runs at the same seed *should* face
identical spawns — which would make the arm comparison **paired** and admit McNemar, materially
more powerful than comparing two independent proportions. But the flow policy calls
`torch.randn` for `x0` on the same global generator the reset events draw from, so this only
holds while every env refills in lockstep and none flushes. **`spawn_pos` is now recorded per
episode and `summarize_arms.py` verifies the pairing rather than assuming it**, falling back to
the unpaired analysis and saying so if the spawns differ.

### 3g. DART data covers 2.7× more observation space at identical label quality

| pool | median cross-demo NN distance | label-noise floor |
|---|---:|---:|
| nominal | 0.232 std-units | 0.0002 |
| DART σ=0.05 | **0.625 std-units** | 0.0002 |

Exactly what noise injection is supposed to buy, and not visible in the collection logs. It also
sharpens the arm comparison: **if arm B loses, it will not be because its labels are noisier.**

### 3h. The open-loop-clock risk

Every demo follows one phase schedule with identical step counts — cross-demo nearest neighbours
land a **median of 1–2 timesteps apart** — and `last_action` occupies obs dims 27:34. A policy
can therefore integrate its own previous output and replay a trajectory indexed by *time* and by
the *initial* block pose, never re-reading where the block is now. **On this data that strategy
scores very well**, and it is not a manipulation policy.

`scripts/diag_feedback.py` is the test: re-randomise the block **mid-`reach`** from the env's own
reset range. Timing constant measured from the demo actions rather than assumed — the gripper
channel has exactly two transitions, **close at step 40** and release at 464, so the block is
only free over steps 0–39. Two controls ship with it (no-teleport, and a same-pose write through
the identical code path). Not yet run on a trained model.

---

## 4. The noise / DART work (the main scientific result)

Full detail in `EXP_NOISE_SWEEP.md`. Both experiments pre-registered with beliefs and a
decision rule written before running; **six of eight beliefs were wrong**.

### 4a. Why DART at all

The expert plans everything before touching the block and runs **open-loop**, so nominal demos
lie on one deterministic manifold indexed by the spawn. A cloned policy that drifts a
millimetre off it has never seen anything nearby. DART is valid *here* because the action is a
joint **position target**: commanding `q_t` pulls the arm to `q_t` from any nearby state, so we
perturb the **executed** action and record the **nominal** one as the label.

### 4b. Noise everywhere destroys the task; noise in free space does not

| `noise_std` | seated | resets | grip at `push` | lateral p90 |
|---|---|---|---|---|
| 0.00 | 100.0 % | 0 | 128/128 | 1.21 mm |
| 0.01 | 88.3 % | 13 | 118/128 | 31.06 mm |
| 0.02 | 57.0 % | 44 | 96/128 | 147.87 mm |
| 0.04 | 28.9 % | 68 | 89/128 | 151.41 mm |
| 0.08 | 7.8 % | 78 | 79/128 | 147.35 mm |

**Grip retention per phase is the whole result**: 128/128 in lift/back/spin/turn at *every*
level up to 0.04; all loss is in `push`. Peak arm deviation agrees — free-space phases barely
move (lift 33 → 60 mrad) while `push` goes **74 → 530 mrad**, the arm fighting the walls.

**Why:** DART's correctness argument holds **only in free space**. Inside a 1.5 mm per-side
channel, commanding the nominal target from a perturbed state does not pull the block back to
the centreline — it drives it *harder into a wall*. Stiffness is 2000 and does not yield, so the
block levers out of the pads. The label is not noisy, it is **wrong**. No `noise_std` fixes it.

Restricting noise to `reach,lift,back,spin,turn`, same seed and magnitude:

| `noise_std = 0.02` | seated | resets | grip at `push` | lateral p90 | `push` peak dev |
|---|---|---|---|---|---|
| all phases | 57.0 % | 44 | 96/128 | 147.87 mm | 350 mrad |
| **free space only** | **96.9 %** | **0** | **128/128** | **1.07 mm** | **74 mrad** |

`push` deviation returns to its no-noise baseline *exactly*.

Free-space curve: 0.02 → 96.9 %, 0.05 → 92.2 %, 0.10 → 75.0 %, 0.20 → 15.6 %.
**Chosen: 0.05** by the pre-registered rule (≥ 90 % seated, ≤ 2 % resets).

### 4c. A third failure mechanism — free space is bounded

High-noise failures are **not** grasp misses. The 30 terminations at 0.10 and 101 at 0.20 are
the **carried block hitting the fixture**. `stage_x` is 0.165 precisely because the block rides
~5 mm ahead of the TCP against wall faces at 0.210, so free space is only free within ~40 mm.
**That margin caps the usable noise**, not the grasp.

### 4d. Stage D deliverable — `sigma_init` measured, not inherited

Pick-place's *healthy* σ ≈ 0.08 sits between our 75 % and 15.6 % cells. This task is far more
noise-sensitive. **`sigma_init` should start at ≤ 0.05 (≈ −3.0), not the inherited −2.5.** This
is a prior from the scripted expert, not a substitute for measuring the trained base.

---

## 5. Demo data

`slot/data/v2/` — one rollout per process, slip signal stored, all pass `verify_demos.py`.

| pool | demos | successful | rate |
|---|---|---|---|
| `nominal_s0..s3` | 512 | 512 | **100.0 %** |
| `dart002_s10..s13` | 512 | 492 | 96.1 % |
| `dart005_s20..s23` | 502 | 462 | 92.0 % |
| **total** | **1526** | **1466** | **96.1 %** |

**914,074 frames**, T = 599 each. Spawn coverage spans **100 % of the reset range** on all
three axes: x [0.200, 0.240], y [−0.160, −0.100], yaw [−0.349, +0.350].

`nominal_s4..s7` were collecting at the end of the session — **check they finished** (`ls
data/v2/*.hdf5` should show 16) — to give arm A 1024 nominal demos matching arm B's 1024 mixed.

⚠ **`slot/data/pool_*.hdf5` (v1) must NOT be trained on** — collected 4 rollouts per process
(so ~10 points of yield lost to the first-episode bias) and lacking the slip signal. Kept only
as evidence for the §3a investigation.

### 5a. The verification that matters

A demo file can be perfectly well-*shaped* and still be wrong in the one way that destroys BC:
an off-by-one between observation and label. The 34-D obs carries `last_action` in its tail,
which makes it checkable exactly:

> **`obs[t, 27:34]` must equal `actions[t-1]`.**

Recording the obs *after* stepping — the natural mistake, since `env.step` returns one — would
make it `== actions[t]`, training the policy to predict an action it can already see.
**Measured on the nominal pools: max abs difference 0.000e+00.**

On **DART** pools this identity *cannot* hold, and that is correct: the executed action is
`nominal + noise` while the label is `nominal`, so the residual **is** the noise. Checking it
against the OU process it should be verifies alignment *and* noise injection together:

```
[PASS] DART residual std matches --noise_std      measured 0.0150 vs declared 0.0200
[PASS] noise is ACTIVE before the push phase      max |eps| = 0.0941
[PASS] noise is OFF from the push phase onward    max |eps| = 9.64e-06  (0.05 % of noise_std)
[PASS] noise is temporally correlated, not white  lag-1 autocorr 0.940 (OU rho 0.95)
```

The third line is **independent confirmation that the phase restriction held** — the fix that
took 57.0 % → 96.9 %. Until then it was only inferable from the success rate.

### 5b. Loss censoring — asked for by Big Will, built, calibrated, and it barely bites

The ported grip-based detector was **inert**: every phase reported `held` on all 512 demos while
50 ended unseated, because this expert's grip *never* fails (gap 29.96 mm throughout, even in
failures). `train_mask == 0` on **0 demos**.

Replaced with the **in-hand offset** (`block_pos − TCP` vs its post-grasp value) — it catches
sliding *and* jamming with one number, and is **noise-agnostic** (injected noise moves arm and
block together), so it does not delete DART's corrective labels.

Calibrated on 1526 demos / 60 failures:

| window | AUC raw | AUC excess |
|---|---|---|
| whole gripped span | **0.355** | 0.634 |
| **CARRY only (grasp → push)** | 0.457 | **0.697** |
| PUSH only | 0.358 | 0.535 |

AUC = P(a random **failure** scores higher). **0.355 means the raw statistic is inverted** — a
*success* detector. Mechanism: the expert drives the block into the **back stop**, after which
the gripper keeps advancing while the block cannot move, so high slip is the signature of a
*fully seated* insert. Censoring on it would have deleted the best demos.

Carry-window excess slip gives 0.697, above the `min_auc = 0.65` gate registered before looking
(a 3-pool subset gave 0.637 and the gate correctly withheld — **not** nudged to 0.63).

**Honest magnitude:** `default_demo_filter` drops failed demos from training entirely, so
censoring them does nothing. The censor's only real action is on the ~2 % of *successful* demos
with a genuine mid-carry slip ≈ **1.3 % of frames**. Right on principle; not a win worth
claiming.

### 5b-final (session 4). The censor is calibrated and DELIBERATELY NOT APPLIED.

Full-set calibration over all 16 pools (2038 demos, 61 real failures) confirms the signal —
CARRY-only excess **AUC 0.693**, against 0.697 on 12 pools, and the raw statistic is still
inverted (< 0.5) in every window, so the back-stop mechanism survives at full sample size.

Then the arithmetic was worked through, and it inverts on a success-only pool:

* both arms train with `--pool success`, so the **14.8 % of failed demos the censor catches are
  already excluded** — that entire benefit is unrealisable;
* the **2.0 % of successful demos it censors are not** excluded. By construction: the threshold
  *is* the 98th percentile of the success distribution. That is ≈ 40 demos truncated at median
  t = 188, keeping 33 % of their frames — roughly **16 000 frames deleted from episodes that
  seated the block**, in exchange for nothing.

So it is switched off and **the pools stay exactly as collected** (`train_mask` all ones,
`slip_censor_t = -1`), which also keeps them frozen and reproducible. Do **not** run
`--apply`; the earlier instruction to do so is superseded.

**The alternative that was considered and rejected:** train on *everything* and let the censor
truncate each failed episode where the expert lost the plot — a failed episode's first 60 % is
good supervision. That needs **sensitivity**, and this censor has 14.8 % at a 2 % FPR, so 85 % of
failed demos would train end-to-end with their bad endings intact. Strictly worse than dropping
them. Lowering the FPR only shrinks the unrealisable catch rate further.

**What this answers, for the record**, since it began as Big Will's question about labelling
expert failures and masking their gradients:

1. **Expert failures are labelled** — every demo carries `success`, a per-phase `outcomes` map
   (`grasped`/`missed`, `held`/`lost`, `seated`/`unseated`) and the raw `slip_mm` signal.
2. **They are excluded from training** by `--pool success`, a cleaner instrument than masking
   for whole-episode failures.
3. **Sub-episode masking is available but unwarranted here.** 85 % of failures are not preceded
   by a visible in-hand event, and given that identical actions from an identical state already
   flip 23–25 % of outcomes (§3b), most failures are simulator chaos rather than expert error —
   and chaos is not maskable. `calibrate_slip.py` was written to be *able* to return this verdict
   before the data was in.
4. **The machinery stays** for Stage D / HG-DAgger, where masking is genuinely required: a gated
   takeover's policy-driven prefix is known-bad by construction, needs no detector, and is
   hard-zeroed by the collector.

The gradient-masking path was verified to actually work, independently of whether it is used:
`test_pipeline_cpu.py` corrupts labels on censored steps and the loss is **bit-identical**
(6468.139160 both ways), while the same corruption on trainable steps moves it to 540827.

---

## 6. The port (`act/` → `slot_act/`)

Verified by `python scripts/check_port.py` → **PORT OK** (run it before every training run).

| constant | pick-place | slot |
|---|---|---|
| `OBS_DIM` | 41 | **34** |
| `ENV_STATE_SLICE` | `slice(16,41)` | **`slice(16,34)`** |
| `ENV_STATE_DIM` | 25 | **18** |
| `BIT_DIMS` | `[6,7,14,15,40]` | **`[6,7,14,15,33]`** |
| `RES_OBS_DIM` | 64 | **58** |
| `STEER_OBS_DIM` | 56 | **50** |
| `STATE_SLICE`, `STATE_DIM`, `ACTION_DIM`, `FINGER_FEATURE_DIMS` | — | unchanged |

**Obs layout (34-D):** `[0:8]` joint_pos_rel, `[8:16]` joint_vel_rel, `[16:23]` block pose in
**robot-root** frame (pos 3 + quat 4 XYZW), `[23:27]` slot_error (depth, lateral, yaw,
is_inserted), `[27:34]` last_action.

**Actions 7-D**: 6 arm joint targets (`scale=0.5`, `use_default_offset=True`, so
`action = (q_desired − q_default)/0.5`) + binary gripper (`< 0` closes, `≥ 0` opens).
1 action unit = 0.5 rad.

**Episode**: 600 steps @ 50 Hz, `decimation=8`, `dt=1/400`. **600 % 15 == 0 → exactly 40
steering windows**, so no episode-length override is needed (pick-place had to force 30 s).

### 6a. Four things that were not renames

1. **The package is `slot_act`, not `act`.** Keeping the name is a trap — both dirs are
   importable as `act` and `sys.path` order decides. Measured:
   `cd /tmp && PYTHONPATH=.../eva_bc python -c "import act.dataset as D; print(D.OBS_DIM)"` →
   **41**, silently. A guard *inside* the copy cannot help (in the failing case the copy is
   never imported); my first mitigation was wrong for exactly that reason.
2. **`residual_core.py` had `obs41[:, 40]`** — a bare integer, invisible to any `obs41→obs34`
   substitution, out-of-bounds at 34. Now `obs34[:, BIT_DIMS[-1]]`. That channel is required:
   physical finger joints alone score a *better* AUC (0.976 vs 0.968) but **27.1 % FPR** vs 0 %.
3. **The goal-delta tail is 4-D, not 3-D** (hence 58/50, not the 57/49 `PORT_MAP` predicted).
   The env's `lateral_error`/`yaw_error` are **absolute values** — they say how wrong the policy
   is, not which way to correct. `slot_mdp.goal_delta` supplies signed values plus yaw.
4. **The flush rule's z-drop half is removed** (both `residual_core.py` and `eval_act.py`). It
   was sized against a 40 mm basket rim; here the block legitimately descends ~8 mm the instant
   the fingers open, so it would flush the committed chunk at the most precision-critical
   moment. Chunk commitment is load-bearing (59.4 → 32.8 → 3.1 → 0 % when shortened).

### 6b. Env-cfg names that differ

`terminations.object_dropping` does **not** exist. The slot task has `block_dropped`
(`minimum_height=-0.05`) and `block_toppled` (`max_tilt=0.6`); `rewards.dropping_penalty`
(−30) and `toppling_penalty` (−10) both exist. Nulling them at eval is a **decision**, not a
rename: a dropped/toppled block must FAIL the episode, not silently re-randomise and be scored
on a second attempt. `--episode-length-s` defaults changed 30.0 → 12.0.

### 6c. Still outstanding in the port

- **`experiments/exp06_grasp_bit.pt` does not exist** and must be retrained (5 scripts point at
  `slot/experiments/`). Its pick-place `mu`/`sd` were fit on a 24 mm can; our block is 30 mm.
  Needed for **Stage D only** — BC does not use it. Pipeline: `exp01_probe.py` →
  `exp06_grasp_bit.py`, holding the **0 % FPR** gate.
- **`report_coverage.py`** still has the pick-place obs layout hard-coded
  (`for base in (16,23)`, `o0[32:34]` basket slice, cylinder-axis maths). Rewrite or drop; not
  on the critical path.
- `eval_residual.py` / `eval_steer.py` retain cosmetic `can_*` naming (functionally correct).
- Device is hard-coded `cuda:0` in 6 scripts and both yamls.

---

## 6z. SESSION 5 IN FULL — what was done, what worked, what did not

*Written 2026-08-03 ~07:00 for the compaction boundary. Evidence for every line is in
`SESSION5_FINDINGS.md`, `EXP_TIGHT.md`, `EXP_ROBUSTNESS.md` and PLAN rows 5.1–5.30.*

### 6z-a. THE RESULT

**The objective is met.** Flow-matching BC (`policy_type: flow`, chunk 50 / execute 15, 10 Euler
steps, no CVAE — `x0` is the sole stochasticity source), 100 k steps, ~1023 demos per arm.
Pooled over **3 training seeds × 2 spawn seeds**, n = 1152 per arm per task:

| task | clearance | arm A | arm B |
|---|---|---|---|
| `-Loose-v0` | 3.0 mm | 0.778 | 0.799 |
| `-v0` | 1.5 mm | 0.776 | 0.792 |
| `-Tight-v0` | 0.5 mm | 0.708 | 0.764 |

All three clear the 70 % target. **No RL ran at all** — Stage D was contingency and never fired.
Champion is `bc_armB_seed0` (0.938 / 0.969 / 0.958). Videos for Big Will are in
`runs/bc_armB_seed0/videos/{Rebot-PrecisionSlot-Loose-v0,-v0,-Tight-v0}.mp4`.

### 6z-b. THE FOUR FINDINGS THAT MATTER

1. **The bottleneck is DEPTH, not precision, and it is clearance-independent.**
   `never_entered` + `stalled_in_mouth` = **82–84 % of all failures at every clearance**, and
   `never_entered` failures carry a median |lateral| of 0.59 / 0.79 / 0.73 mm — on *Loose* that
   is 0.59 mm of error inside a **3.0 mm** opening, at depth −40.8 mm. The block is well aligned
   and stops ~40 mm short. This is why tightening the channel 6× costs only 5 points, and why
   4 of 6 runs have Tight within 5 pts of `-v0` or better. **Any further effort should target
   the push.** Corroboration: the expert's `insert_x = 0.2545` "drives the block into the back
   stop, which squares it and removes depth variance" — it solves depth with a hard stop; the
   clone inherits the trajectory but not the guarantee.
2. **Training-seed variance is 15–29 points on identical data**, dwarfing every effect measured
   this session. That is the pick-place pattern (32.8–59.4 % across seeds) reproducing exactly.
   **Every single-run number in this project must be read against a ±25-point seed spread.**
3. **The x0 choice alone spans 54 points** (−37.5 to +16.7) across checkpoints with weights,
   task, spawn seed and episode count fixed. Reproduces EXP07 gate 2b (frozen x0 draws spanning
   14.1–56.2 % on one frozen base). Blind freezing is a coin flip; this is the **measured** case
   for state-conditioned x0-steering, no longer borrowed from pick-place.
4. **DART made no measurable difference.** Gap +0.021 / +0.016 against within-arm seed spreads
   of 0.229 / 0.281. Do not claim it helped.

### 6z-c. WHAT WORKED (process, worth repeating)

* **Pre-registration, repeatedly saving me from myself.** `EXP_TIGHT.md` was written before any
  Tight number existed; three of its five beliefs were then refuted by the data. The
  replicate-across-seeds rule is the single highest-value item — see 6z-d item 1.
* **Instruments that distinguish "verified" from "unknown".** `analysis/paired_evals.py` caught
  an error in my own ad-hoc pairing check (a missing `spawn_pos` field compared equal to
  everything, so it reported "spawns identical" for pairs that were unverifiable), and later
  caught that `--fixed-x0` **breaks pairing outright** by changing global RNG consumption.
* **Exercising analysis code on synthetic data before the real data arrives.**
  `summarize_arms.py` would have judged `-Tight-v0` against the **Loose** bar and fired "the
  bottleneck is not data composition", mis-framing the whole sweep. Found by building a fake
  36-cell sweep with Tight deliberately low.
* **Provenance fields.** `fixed_x0`, `slot_dx/dy`, `spawn_scale` are recorded into every results
  JSON precisely so a perturbed run is never byte-indistinguishable from a nominal one.
* **CPU validation before GPU spend.** `test_steer_cpu.py` (17/17) proves EXP07's `z = 0`
  bit-exactness gate in 20 s instead of 35 min of GPU.
* **Gates that run first.** `EXP_ROBUSTNESS`\'s nominal cell reproduced the sweep cell exactly
  (94/96 both) before any perturbed cell was believed.

### 6z-d. WHAT DID NOT WORK (my errors, all caught, all instructive)

1. **I reported 92.7 % as the headline. It was the best of three training seeds.** The honest
   pooled figure is 0.776 / 0.792. Corrected in place; the original is left visible.
2. **I called `--fixed-x0 zeros` a confirmed +16.7-point win at p = 0.0045.** Replication across
   all six runs: **1 improved, 1 flat, 4 worse, mean −9.5**. It was a single-seed fluke *on the
   seed I happened to test first*, which was also the run with the largest Tight deficit — i.e.
   the most room to help. Only the pre-registered replication rule stopped it being written up.
3. **I predicted Tight at 0.19, then 0.406, with 0.25–0.55 pre-registered. Actual 0.736.** All
   three rested on the premise that channel width binds. It does not (6z-b item 1).
4. **I claimed the lateral CDF was censored, then over-corrected.** Running the *same* policy on
   two clearances showed the bulk quantiles barely move (p25 1.02×, p50 0.87×, p75 1.00× against
   the 2.00× pure censoring predicts) — mostly intrinsic, censored only in the tail.
5. **I recommended collecting Tight demos.** Reading `expert/plan.py` killed it: `ExpertParams`
   has **no clearance term**, so the expert commands the same trajectory on Tight; the demos
   would carry near-identical labels.
6. **`make_videos.sh` delivered one video while reporting three.** gymnasium\'s `RecordVideo`
   always writes `rl-video-step-0.mp4` to the same folder; each clearance overwrote the last.
   Only visible as a buried `UserWarning`. Fixed by renaming inside the loop.
7. **`run_eval_sweep.sh` had a `set -euo pipefail` + `| grep` hazard** that would have killed the
   remaining 30+ evals on the first failure during a 2.5 h unattended run.
8. **`experiments/exp06_grasp_bit.pt` never existed in either repo** — six call sites pointed at
   it, so every steering entry point would have died *after* paying Isaac Sim\'s boot cost.

The pattern across all eight: **an instrument or inference that was self-consistent rather than
correct.** Nothing errored. Every one produced a plausible number.

### 6z-e. WHAT WAS BUILT

| file | purpose |
|---|---|
| `scripts/test_steer_cpu.py` | 17-check CPU validation of the x0-steering path, no sim, ~20 s |
| `analysis/paired_evals.py` | McNemar across eval JSONs; separates *matched* from *unverifiable* spawns |
| `analysis/failure_taxonomy.py` | failure buckets by decision list; `--pool` aggregates the sweep |
| `scripts/run_x0_determinism.sh` | EXP_TIGHT step B |
| `scripts/run_robustness.sh` | EXP_ROBUSTNESS, 9 perturbed conditions |
| `docs/slot/SESSION5_FINDINGS.md` | the session record |
| `docs/slot/EXP_TIGHT.md` | pre-registered, now **CLOSED** (beliefs 1 and 2 refuted) |
| `docs/slot/EXP_ROBUSTNESS.md` | pre-registered, **RUNNING** |

Changed in `slot_act/`: analytic `SlotGraspBit` replaces the missing MLP artifact (and
`grasp_bit_path` dropped from five call sites); `--fixed-x0`, `--slot-dx`, `--slot-dy`,
`--spawn-scale` added to `eval_act.py` with provenance recorded in the results JSON.

---

## 7. The plan from here (subject to change)

### Step 1 — finish and freeze the data ✅ **DONE (session 4)**
16 pools, 2038 demos, 1977 successful (97.0 %). **All 16 pass `verify_demos.py`.**
The slip censor was calibrated over the full set (CARRY-only excess AUC **0.693**, clearing the
0.65 gate) and then **deliberately not applied** — see `SESSION4_WRITEUP.md` §5. In one line:
both arms train with `--pool success`, so the 14.8 % of *failed* demos the censor catches are
already excluded, while the 2 % of *successful* demos it censors are not. On a success-only pool
it is a pure subtraction of ~16 000 good frames. The pools stay exactly as collected.

### Step 2 — validate the instruments BEFORE training ✅ **DONE (session 4)**
```
python scripts/test_pipeline_cpu.py --data data/v2/nominal_s0.hdf5 data/v2/dart005_s20.hdf5
python scripts/check_eval_json.py <any eval json> --expect-envs 32
```
17/17 and 14/14. This was worth doing: it found the pool-filter defect, and the eval run it
prescribed found the post-reset diagnostics bug. See `SESSION4_WRITEUP.md` §3 and §6.

Two instrument properties measured that change how results must be read:
* **An eval at a fixed seed is bit-reproducible across processes** (0.438/0.562/0.375 twice
  over). Repeating a run yields **no** information, and an error bar built from repeats is
  exactly zero. Vary `--seed` to see spawn variance. This does *not* contradict §3b: that
  finding was about repeated episodes *inside* one process.
* **The first-episode bias reproduces on a learned policy**: +18.7 points, alongside the +12.9
  measured on the expert. Not an artefact of the scripted plan.

### Step 3 — Stage C training, per `EXP_BC_ARMS.md` *(running)*
Arms matched on **demo count**, differing only in composition, both with `--pool success`:
- **A**: 1023 nominal (`nominal_s0..s7`, minus `s5`'s one failure) — 612 777 samples
- **B**: 512 nominal (`s0..s3`) + 512 DART (2× dart002, 2× dart005), successes only

```
for s in 0 1 2; do bash scripts/run_stage_c.sh $s 100000; done   # ~35 min per run, ~3.5 h
bash scripts/run_eval_sweep.sh ckpt_final.pt                      # 24 evals, ~1.5 h
python scripts/summarize_arms.py 'runs/bc_arm*/eval_ckpt_final_*.json'
```

Throughput is **47.4 steps/s** measured, so 100 k steps ≈ 35 min; the whole sweep including
evaluation fits in ~5 h. Checkpoints every 10 k give a learning curve for free.

The eval sweep uses **two** spawn seeds (777 and 888, both outside the collection seeds), for
the reason in step 2: one seed gives a point, repeating it gives the same point.

Targets: **≥ 55 % Loose, ≥ 40 % v0** pooled — likely to be cleared easily given the 2000-step
result, so treat them as a floor rather than a goal.

### Step 4 — read the result honestly
The pre-registered rule from `EXP_BC_ARMS.md` is now **implemented in
`scripts/summarize_arms.py`**, including the "within seed spread → **do not claim DART helped**"
branch and Wilson intervals on the pooled rates. Applying it in code rather than by eye is
deliberate: pick-place spanned 32.8–59.4 % across *training seeds alone* on identical data, so a
gap that does not clear the within-arm spread is not a result.

If either arm misses the bar, the bottleneck is not data composition — go to
`diag_training_env.py` fixed-action attribution and per-term reward channels from epoch 1.

**Prior from the instrument run, worth checking against the real models:** of 27 failures from
the 2000-step checkpoint, **13 ended at the slot mouth, at correct seated height, short of the
40 mm depth**; 6 fell short of the mouth; 8 never advanced. The dominant failure was *depth*,
not grasping and not alignment. If that survives into the trained models it is the obvious
target for Stage D.

### Step 4a — the four things to do the moment the sweep lands *(≈ 1 h, mostly GPU)*

Ordered by information per minute. **Everything here is subject to change once the numbers are
in** — in particular, if Stage C already clears 70 % on `-v0`, steps 5 and 6 change character
from "needed" to "worth having".

1. **`summarize_arms.py`** — the pre-registered decision, applied in code including the pairing
   check. Read the *per-training-seed spread* before the arm gap; if the gap does not clear the
   spread, there is no result (pick-place spanned 32.8–59.4 % across training seeds on identical
   data).
2. **Failure taxonomy on the champion.** The instrument run's prior: of 27 failures, **13 ended
   at the slot mouth at correct seated height and short of the 40 mm depth**, 6 short of the
   mouth, 8 never advanced, **0 deep-but-unseated**. If that survives into the trained models,
   *depth* — not grasping, not alignment — is the Stage D target, and the natural intervention is
   steering the `push`.
3. **`diag_feedback.py`** at perturb-step 5/15/25/35 plus both controls (§3h). This decides
   whether Stage D should even be aimed at the policy, or whether the policy is a clock and the
   whole approach needs re-thinking. **Run this before committing to Stage D.**
4. **`make_videos.sh`** on the champion — Loose / v0 / Tight. Hand Big Will the paths; never
   view them.

### Step 5 — Tight (0.5 mm). **RE-ORDERED in session 5: data before RL.**

`-v0` is done (92.7 %, §1a). The remaining rung is `-Tight-v0`, and the session-5 evidence says
try the cheap thing first. Full reasoning in `SESSION5_FINDINGS.md` §5c–5d.

**5.1 — Measure.** The sweep already evaluates every checkpoint on `-Tight-v0` × 2 spawn seeds.
Read that number before doing anything else. My own pre-registered prediction (≈ 0.19) is
**confounded and I said so in writing before the data arrived**: the |lateral| it extrapolates
from is measured *after the walls constrain the block*, which is why its p100 is exactly the
clearance. The expert proves the censoring — same open-loop trajectory, lateral p90 falls
1.38 → 1.11 → **0.48 mm** as the channel narrows. If Tight lands above 0.19, the conclusion is
"the censored-measurement model was wrong", **not** "the policy is closed-loop".

**5.2 — If short: collect Tight demos and retrain.** The expert is **128/128 on Tight** while
**all 2038 demos were collected on `-v0`** (verified from the `task` attr). So the first
hypothesis is train/test clearance mismatch — a *data* problem with an existing fix:

```
python scripts/collect_demos.py --task Rebot-PrecisionSlot-Tight-v0 \
       --num_envs 128 --rollouts 4 --seed 30 --out data/v2/tight_s30.hdf5
python scripts/verify_demos.py data/v2/tight_s30.hdf5
bash scripts/run_stage_c.sh <seed> 30000        # 30 k is enough — §1a, PLAN 5.3
```

No new algorithm, no reward design, no PPO. Reaching for RL with a 100 %-capable expert and an
idle collection harness sitting there would repeat exactly the error-class misdiagnosis that
cost eva_bc two EXP06 runs (`POSTMORTEM.md` §9b).

**5.3 — Only then, Stage D (x0-steering).** Now unblocked and pre-validated:

- **Startup blocker closed.** `experiments/exp06_grasp_bit.pt` never existed in either repo;
  six call sites pointed at it. Replaced by the analytic `SlotGraspBit` (`SESSION5_FINDINGS.md`
  §4a) — the env's own `block_lifted` AND commanded-closed. Do **not** try to retrain the MLP:
  the finger channels are degenerate here (−0.04889 ± 0.00003 held vs −0.04887 ± 0.00005
  not-held) and there is no negative class in 2038 demos.
- **The bit-exactness gate is already green on CPU.** `scripts/test_steer_cpu.py` 17/17:
  `steer_x0 = zeros` is action-for-action identical to `fixed_x0 = zeros` (0.000e+00) including
  across a desynced mid-window flush. Re-run it after any change to the steering path — 20 s.
- **Window alignment here is exact, better than EXP07's.** 600 steps = 40 windows, no
  mid-episode terminations, and **zero flushes measured in 128 episodes** — so unlike
  pick-place there is no stale-`z` desync source at all.
- `clip_actions: 1.0` — a **scale**, not a clip; `100.0` cost eva_bc two full training runs.
  `sigma_init` should be *measured* by a gate-S0 pass, not inherited (EXP07 chose −1.2 from
  data; §4d's slot prior is ≤ 0.05).
- **Temper the expectation, in advance.** EXP07's +35.9 pts came mostly from the never-lifted
  bucket collapsing 18/19 — and **that failure mode does not exist here**: all 61 expert
  failures are `release=unseated`, zero grasp failures. What should transfer is re-choosing the
  chunk family at the approach/push windows.

### Step 5z — WHERE TO GO NEXT (written 2026-08-03 ~07:00, **subject to change**)

**The objective is already met** (§6z-a), so everything below is *improving* a delivered result,
not rescuing one. Ordered by information per GPU-hour. Nothing here is committed; the sweep
results should be re-read before starting any of it.

**0. Finish EXP_ROBUSTNESS (running, ~25 min left).** The gate passed. Read the 9 cells against
`EXP_ROBUSTNESS.md` \'s pre-registered beliefs. **This is the highest-value open question**,
because it decides how the headline should be *described*: if a 5 mm slot shift collapses the
policy, the Stage C number is "in-distribution only" and must be reported that way. The
y-vs-x asymmetry (unsigned `lateral_error` vs monotone `insertion_depth`) is the sharp part.

**1. Attack DEPTH — the measured bottleneck (§6z-b item 1).** 82–84 % of failures at every
clearance are the block stopping ~40 mm short with lateral alignment fine. Three cheap probes,
in order:
   a. **Does it stall or stop?** Log block x over the last 150 steps of failed episodes. A block
      that decelerates smoothly to a stop is a *policy* shortfall; one that stops abruptly is
      *contact* (jam on the lip / floor step). Different fixes. Pure analysis on data that
      already exists if the per-episode records are extended — cheap.
   b. **Is the horizon binding?** Episodes are exactly 600 steps and the expert uses 558. If
      failed episodes are still advancing at step 599, the policy is merely *slow* and a longer
      horizon converts failures into successes for free. One eval at
      `--episode-length-s 16` answers it. **Do this first — it is one command.**
   c. If (b) is negative, this is the target for Stage D steering.

**2. Stage D — x0-steering, aimed at depth.** Now motivated by measurement rather than by the
pick-place precedent: the x0 choice alone spans **54 points** (§6z-b item 3). Fully unblocked —
`SlotGraspBit` replaces the missing artifact, `test_steer_cpu.py` is 17/17 including the `z = 0`
bit-exactness gate, and this task\'s window alignment is *better* than pick-place\'s (600 steps =
40 windows exactly, no mid-episode terminations, **zero flushes measured in 128 episodes**, so
no stale-`z` desync source at all). Required discipline, all pre-registered by EXP07:
   * gate 1: `z = 0` must reproduce the frozen base episode-for-episode **in sim** (CPU already
     proven);
   * gate S0: **measure** the exploration response and choose `σ_init` from data — do not
     inherit EXP07\'s −1.2;
   * `clip_actions: 1.0` is mandatory (rl_games *rescales*; 100.0 cost eva_bc two runs);
   * temper expectations — EXP07\'s +35.9 came mostly from the never-lifted bucket, which
     **does not exist here** (zero grasp failures in 2038 demos).

**3. `diag_feedback.py` — clock vs closed-loop (still never run).** Partly subsumed by
EXP_ROBUSTNESS, which moves the *goal* rather than the *block*, but it remains the direct test.
Run at perturb-step 5/15/25/35 plus both controls (`--perturb-step -1`, `--resample-only`),
`--num-envs 16 --episodes 64`.

**4. Cheap wins available at any time:**
   * **Train at 30 k, not 100 k.** Paired McNemar says 30k/50k/100k are indistinguishable
     (χ² ≤ 2.12; 30k and 100k agree on 81 of 96 episodes). A 6-run sweep drops from ~3.5 h to
     ~1.2 h. Applies to the *next* sweep; do not shorten one mid-flight.
   * **More training seeds beats more anything else.** With a 15–29 point seed spread, 3 seeds
     cannot resolve a sub-10-point effect. If any future comparison matters, budget 5–8 seeds.
   * Regenerate the **expert** video (the current one predates the `retreat` phase).

**5. Stage E — PPO from scratch** on the env\'s Factory reward, which has **never been run**.
Cheap, and it contextualises every BC number. Any measurable learning curve is the deliverable.

**Explicitly NOT worth doing** (each killed by evidence this session):
   * collecting `-Tight-v0` demos — the expert\'s plan has no clearance term (§6z-d item 5);
   * retraining the EXP06 grasp-bit MLP — the finger channels are degenerate on this robot and
     there is no negative class in 2038 demos;
   * `--fixed-x0 zeros` as a *fix* — refuted on replication (§6z-d item 2). It remains useful as
     an *instrument* for measuring how much of a checkpoint\'s behaviour is x0 sampling;
   * chasing lateral/yaw precision — yaw is binding in **1 of 797** failures, and lateral is
     inside budget in the failures that matter.

### Step 6 — Stage E, cheap control
PPO from scratch on the env's Factory reward, which **has never been run**. Any measurable
learning curve is the deliverable. Cheap and it contextualises the BC/RL numbers.

### Opportunistic
- Regenerate the expert video (current one predates `retreat`) — Big Will wants one per env.
  `scripts/make_videos.sh` covers the trained policy; the expert one needs `run_expert.py --video`.
- HG-DAgger if Stage C plateaus: it supplies corrective data the open-loop expert genuinely
  cannot, and `build_train_mask` + the stored `slip_mm` signal are already wired for it (the
  policy-driven prefix is what actually needs censoring).
- **Learning-curve evals** on `ckpt_00X0000` to find where success saturates. Training loss is a
  poor guide here — it plateaued around **step 20 k** (0.053 → 0.050 → 0.047 → 0.040 over
  20k–50k, with per-batch std 0.011–0.018 comparable to the change) because flow-matching loss is
  noisy by construction (random τ per sample). Only eval success answers "was 100 k needed?".
- If 100 k turns out to be far past saturation, **do not shorten a run mid-sweep** — that breaks
  the matched protocol. Change it for the *next* experiment.

### A standing caution for whoever picks this up

Session 3 produced four confident wrong numbers; session 4 produced three more, all caught. The
pattern is identical every time: **an instrument that was self-consistent rather than correct**.
The defences that actually worked, in order of value:

1. **Validate the instrument on a case whose answer you already know**, before trusting it on one
   you don't. `test_pipeline_cpu.py` and `check_eval_json.py` exist for exactly this and cost
   under a minute each.
2. **Cross-check any diagnostic against a second field that measures the same thing.** The
   post-reset bug was found because `lateral_mm = 126.79` was geometrically impossible next to
   `final_obj_pos`, not because anything errored.
3. **Compute the standard error before naming a trend.** One retracted claim this project came
   from comparing four increasing means without dividing by √n.
4. **A detector's own output distribution is evidence about the detector.** A censor firing on
   > 50 % of demos, successes included, is broken; a detector whose AUC is < 0.5 has the sign
   backwards.

---

## 8. Files

```
eva_bc/docs/slot/
  HANDOFF.md            <- this file
  SESSION4_WRITEUP.md   session-4 record: data freeze, instrument validation, Stage C
  SESSION3_WRITEUP.md   full session-3 record incl. every wrong diagnosis
  EXP_NOISE_SWEEP.md    the two pre-registered noise experiments
  EXP_BC_ARMS.md        pre-registered Stage-C design + exact runbook
  PORT_MAP.md           act/ interface facts + port landmines (addendum A-K)
  PLAN.md               stage plan + 29-row verdict log
  EXPERT_RESULTS.md     session-2 expert evidence
  EXPERT_PORT.md        session-1 notes (superseded)
eva_bc/slot/
  expert/ik.py                    batched DLS IK, 5-DOF task, decaying nullspace bias
  expert/plan.py                  planner + execution schedule (generator of per-step commands)
  slot_mdp.py                     the 4-call mdp surface act/ needs; seated placed_mask
  slot_act/                       the ported BC+RL stack (18 files)
  scripts/run_expert.py           expert measurement harness, --video, --trace <phase>
  scripts/collect_demos.py        batched demo collection + DART noise + slip signal
  scripts/verify_demos.py         alignment/schema/noise verification (no Isaac Sim)
  scripts/check_port.py           static port consistency check (no GPU)
  scripts/calibrate_slip.py       offline censor calibration + apply
  scripts/test_pipeline_cpu.py    17-check CPU validation of the whole training path
  scripts/check_eval_json.py      audits an eval JSON for accounting errors (no GPU)
  scripts/run_stage_c.sh          one seed, both arms, --pool success
  scripts/run_eval_sweep.sh       Loose/v0/Tight x 2 spawn seeds x 6 checkpoints
  scripts/summarize_arms.py       the pre-registered decision rule, in code
  scripts/diag_feedback.py        mid-reach block teleport: clock vs feedback (+2 controls)
  analysis/label_consistency.py   label-noise floor + demo coverage width
  analysis/plan_determinism.py    fk purity, plan purity, execution repeatability
  analysis/conventions.py         quaternion order / Jacobian layout / TCP identity
  analysis/strategy_probe.py      horizontal vs vertical insertion + tolerance surface
  analysis/{insertion_feasibility,gripper_envelope}.py   session-1 sweeps (superseded numbers)
  data/v2/*.hdf5                  THE demo pools (use these)
  data/pool_*.hdf5                v1 pools — DO NOT TRAIN ON
  logs/                           durable run logs
```

---

## 9. Gotchas — every one already produced a confident wrong answer here

1. **Finish every CEM/IK search BEFORE grasping.** `write_joint_state_to_sim` teleports the arm
   and re-opens the fingers. Fixing the ordering alone once took a trial 0 % → 100 %.
2. **Interpolate in CARTESIAN space, never joint space.** A joint-linear path bows the TCP:
   8–12 % vs 100 % on identical geometry.
3. **`mdp.is_inserted` passes for a block on the wall tops or held in the air.** §3c.
4. **The 600-step episode budget is HARD.** An expert running 659 steps timed out every env and
   read as 0 %. `run_expert.py` prints `trajectory used N/600`.
5. **The carried block rides ~5 mm ahead of the TCP in x.** Never compute fixture clearance from
   the TCP alone.
6. **CEM's population size was `num_envs`** — n=4 scored 50 % where n=128 scored 100 %. The seed
   is cached to `logs/expert/seed_q.json`; delete it after changing geometry.
7. **A closed gripper and a loaded gripper are different objects.** Comparing them produced a
   retracted "physics contradicts geometry" claim.
8. **A retention test must not be fakeable.** Use the finger gap
   (`1.0035*(q_l+q_r) − 1.25 mm`, residual 0.035 mm). "Object within 80 mm of the TCP" once
   scored an untouched 107 mm block in an 89 mm gripper at 100 %.
9. **Depth is measured on the block CENTRE.** Max centre x is 0.2575 before the nose hits the
   back stop.
10. **`_GRIPPER_OPEN = 0.045` is a per-finger joint value**, not an opening. Separation = 2×q.
11. **TCP is `(-0.0419, 0, 0)` from `gripper_end`**, not the −0.075 from the lift task.
12. **`ProxyArray`, not tensors** — Isaac Lab 3.0 accessors need `.torch`.
13. **`rl_games clip_actions` is a SCALE, not a clip.** Must be 1.0. Cost eva_bc two runs.
14. **`UsdGeom.BBoxCache` never sees physics poses**; the robot uses payloads+instancing so a
    plain `Usd.PrimRange` finds **zero** geometry — use `Usd.TraverseInstanceProxies`.
15. **A self-consistent harness proves nothing.** The withdrawn slot probe placed the block at
    its own computed TCP, so it looked right under *any* TCP offset.
16. **The FIRST executed episode in a process scores ~13 points higher.** §3a.
17. **Beyond the first episode the sim is not reproducible** — 18 mm, 23–25 % outcome flips.
18. **Compute the standard error before naming a drift.** Four rollout means increasing
    monotonically ("17 mrad") were **1.12 σ, p = 0.266**; four monotone points out of 24
    orderings is a 1-in-12 coincidence. The spread was printed in the same table.
19. **A test for a position-in-sequence hypothesis must itself be positioned.** The first
    `plan_determinism.py` ran its repeat test *after* another episode had executed, so none of
    its "repeats" was a first episode and it returned a null. Moving the block to the top of
    `main()` — nothing else — produced the 4.3 σ effect.
20. **Loss-censoring machinery can be inert and look healthy.** An all-ones `train_mask` reads
    as "clean data" and can equally mean "the detector is blind".
21. **A censor firing on >50 % of demos, successes included, is a broken detector, not dirty
    data.** This guard caught two broken slip detectors. A censor's output distribution is
    evidence about the censor.
22. **Check the SIGN of a detector's separation.** Raw in-hand slip scored AUC 0.355 — it was a
    *success* detector, because the expert deliberately bottoms the block against the back stop.
23. **DART pools cannot satisfy `obs[t,27:34] == actions[t-1]`** — the residual *is* the noise.
    "Fixing" it by labelling with the executed action would destroy DART's corrective property.
24. **A background waiter that greps `ps` for its own job name never exits.** `while ps aux |
    grep -q "[c]ollect_demos"` — the `[c]` trick stops *grep* matching itself, not `ps` matching
    the **waiter's own command line**, which contains the pattern. Four queued pools silently
    never ran, and the log's last line was a cleanly-completed pool, i.e. indistinguishable from
    healthy progress. Wait on a PID or a file, and check `ls -la` mtimes before believing a log.
25. **Anything read after `env.step` describes the NEXT episode.** Isaac Lab resets done envs
    *inside* `step()`. Three diagnostics were read there and reported a median |lateral| of
    126.79 mm for episodes whose block was seated at y = 0.0001 ± 0.0008 m. The success rate was
    fine — only the fields used to *explain* it were fiction, which is the harder failure to
    notice. Sample everything pre-step.
26. **Defaults inherited from pick-place are wrong until checked, and they fail silently.**
    `--episode-length-s 30.0` against a 12.0 s task; `--pool default` (no filter) for an
    experiment whose whole point is the pool composition. Neither errors. Both produce numbers.
27. **`--pool nominal` excludes DART.** It requires `episode_kind == "nominal"`. Pointing arm B
    at it halves the arm and re-creates the volume confound while printing nothing unusual.
    Use `--pool success` for any comparison that mixes demo kinds.
28. **An eval at a fixed seed is bit-reproducible; repeating it is not a replicate.** An error
    bar built from repeated identical evals is exactly zero and completely wrong. Vary `--seed`.
    This does not contradict gotcha-adjacent §3b, which is about repeated episodes *inside* one
    process.
29. **`Rebot-PrecisionSlot-Play-v0` is a duplicate of `-v0`**, not a fourth difficulty. It
    differs only in `num_envs`/`env_spacing` (overridden by `parse_env_cfg`) and
    `enable_corruption`, which the base cfg already sets `False`.
30. **A `lateral_error` measured after seating is CENSORED by the channel, not a property of the
    policy.** Its maximum is the clearance by construction. Proof: the expert runs the same
    open-loop trajectory on all three rungs and its lateral p90 falls 1.38 → 1.11 → 0.48 mm as
    the channel narrows. Never extrapolate one clearance's error CDF to another — I did, in
    writing, and the retraction is `SESSION5_FINDINGS.md` §5c.
31. **A missing JSON field reads as agreement in a hand-rolled comparison.** My ad-hoc pairing
    check reported "spawns identical" for every pair, because `eval_ckpt_0010000` predates
    `spawn_pos` and `()` compared equal to everything. Always distinguish **matched** from
    **unverifiable** — `analysis/paired_evals.py` does; the throwaway version did not.
32. **`set -euo pipefail` + `cmd | grep` kills a whole unattended sweep on the first failure.**
    The failed command makes grep match nothing, grep exits 1, `pipefail` propagates, `set -e`
    aborts — discarding every remaining cell. Terminate such pipelines with an explicit,
    greppable `|| echo "!!! FAILED (continuing)"` so a failure cannot be confused with a skip.
33. **A loop that writes to a fixed output filename delivers ONE artifact, not N.** gymnasium's
    `RecordVideo` always emits `rl-video-step-0.mp4` into the same folder, so
    `make_videos.sh`'s three-clearance loop overwrote itself twice and reported success each
    time. Rename inside the loop, immediately, and fail loudly if the file is absent.
34. **The pick-place grasp bit cannot be ported OR retrained for this robot.** The artifact
    `exp06_grasp_bit.pt` exists in neither repo (only its training script), and its five input
    channels are degenerate here: the fingers saturate at the same position with or without a
    30 mm block between them (−0.04889 ± 0.00003 held vs −0.04887 ± 0.00005 not-held). There is
    also no negative class — all 61 expert failures are `release=unseated`, zero grasp failures.
    Use the analytic `SlotGraspBit`.

---

## 10. Protocol rules inherited from eva_bc (non-negotiable)

- **≥ 3 training seeds per arm, champion on a held-out spawn seed, pooled ≥ 128-ep numbers
  only. Single-run comparisons are void.** Same data + recipe spanned **32.8 %–59.4 %** across
  training seeds there.
- **Chunk commitment is load-bearing.** Shortening the execution horizon collapsed success
  **59.4 → 32.8 → 3.1 → 0 → 0 %** at `n_action_steps` 15/8/4/2/1. Never shorten it.
- **Verify wrappers by bit-exact reproduction before any training.**
- **Never trust train reward.** The flat residual run *beat* the base on train reward
  (+1700 vs +1644) with zero success change.
- **Per-term reward channels from epoch 1 are the cheapest tripwire.** For window-RL, episodic
  loggers fire only when episodes complete — judge after the first completed episode.
- **Fixed-action attribution beats theorizing** (`slot_act/diag_training_env.py`).
- **"Zero-init residual" needs three things**: zero the mu *weight*, know the mu *bias* stays
  random, and use a small initial sigma.
- **Pre-register each experiment's design, beliefs and decision rule before coding it**, and
  record verdicts in place including retractions. This session: six of eight noise beliefs
  wrong, four wrong diagnoses caught — all of which are only visible because they were written
  down first.

---

## 6y. SESSION 6 FULL RECORD (2026-08-03) — environment diversification

*Full write-ups: `SESSION6_FINDINGS.md`, `EXP_ROBUSTNESS.md` §6–§13, `EXP_DEPTH.md`,
`EXP_STEER.md`. This section is the compressed record for whoever picks this up cold.*

**No training ran this session.** 45 perturbed evaluations + 4 expert runs on session-5
checkpoints, 09:00–12:10, one GPU job at a time.

### 6y-a. What was done

| round | cells | script | doc |
|---|---|---|---|
| 1: slot dx/dy, spawn box | 9 | `run_robustness.sh` | ROBUSTNESS §6 |
| 2: dy ladder, arm jitter, obs noise, combo | 12 | `run_robustness2.sh` | §7f–§7h |
| 3: replication on `bc_armA_seed0` | 6 | `run_robustness3.sh` | §9b–§9e |
| dy × clearance (the decisive crossed test) | 4 | `run_dy_crossed.sh` | §12 |
| spawn cells re-collected with `spawn_yaw` | 2 | `run_spawn_yaw.sh` | §11 |
| actuation noise | 3 | `run_action_noise.sh` | §13 |
| EXP_DEPTH probe A + replication | 5 | `run_horizon{,2}.sh` | DEPTH §7, §8 |
| expert control | 4 | `run_expert_dx.sh` | §8b–§8d |
| videos | 2 | `make_robust_videos.sh` | — |

### 6y-b. THE FIVE FINDINGS THAT MATTER

1. **The policy tracks a moved goal — it has not memorised a path.** Median final block *absolute*
   x = 0.2570 / 0.2620 / 0.2670 at dx = 0 / +5 / +10 mm. Exactly 1:1, 20 mm past anything in
   training. This closes the clock-vs-closed-loop question `diag_feedback.py` was written for and
   never ran; task #10 is marked completed as *subsumed*.
2. **Lateral tolerance is a step function at the clearance — geometry, not the policy.** Twelve
   cells across three clearances collapse onto one curve in **(shift ÷ clearance)** with no
   exceptions: ≤ 0.67 free, 1.00 → 0.760, ≥ 1.33 → floor. Same **2 mm** shift: 0.000 on `-v0`,
   **0.938** on `-Loose-v0`. Same **1 mm** shift: 0.948 on `-v0`, **0.021** on `-Tight-v0`. Same
   checkpoint, spawn-for-spawn identical episodes. **No RL can move this** — `lateral_error` is an
   absolute value, so a memoryless policy cannot tell +2 mm from −2 mm. The fix would be an
   observation change in `eva_rl`, which is shared and not mine to make.
3. **Chunking is the exposure.** `--obs-noise 0.05` costs 4.2 pts (p = 0.15, null);
   `--action-noise 0.05` scores **0/96**; even **0.02 costs 83 points**. The controller re-reads
   the observation once per 15 steps, so an obs error is low-passed ~15× and an action error is
   not filtered at all. Within a window the policy is open-loop by construction.
4. **Under noise the policy FREEZES at the staging waypoint.** Every moderate-noise cell — sensing
   *or* actuation — ends with the block held at **x = 0.166 m** (expert `stage_x` = 0.165),
   **y = 0.0000**, at carried height (z ≈ 0.062), in **81–92 of 96** episodes. Reach, grasp, lift,
   retract, align — three phases at nominal precision — then the push is **never attempted**. This
   is the most concrete statement of the bottleneck the project has, and it is what EXP_STEER
   targets.
5. **Arm start pose is free.** ±0.10 rad (pick_place's own value) costs 2.1 pts, p = 0.41, despite
   the env pinning the arm to one `_START_POSE` in all 2038 demos. Zero training coverage, no
   measurable cost. `never_lifted` stayed at **zero**.

Supporting: in-distribution spawn episodes score **43/43 = 1.000** even when the batch is drawn
from a 2× box (§11b) — yaw is the more damaging axis of the two. Perturbations compose
**independently**, to within 0.9 points (§7h). The expert scores 0.977–1.000 at every slot dx, so
the arm is never the limit (§8d).

### 6y-c. WHAT WORKED (method)

* **Pre-registration, every round, before its cells existed.** It caught belief 6 being the same
  error as belief 4 *before* round 2 ran (§7b), and it is the only reason the two retractions
  below were caught rather than published.
* **Instruments that decide rather than assume.** `robustness_report.py` reads `spawn_pos` and
  picks paired-McNemar vs unpaired-z per cell, printing which and why. The pre-registration said
  "always unpaired"; the data said dx/dy cells *are* paired, which is far more powerful at n=96.
* **Crossing a perturbation with a task parameter.** The dy × clearance design is the single best
  experiment of the session: it converts an unfalsifiable mechanism story ("the policy is blind to
  lateral sign") into a prediction that only geometry can make, including one in the *unintuitive*
  direction (tighter clearance fails at a shift the middle rung shrugs off).
* **Controls that bound the alternative.** The scripted expert told the new slot position bounds
  the robot's contribution to ~zero. Without it, "+20 mm costs 8 points" is ambiguous.
* **Re-running a cell unchanged.** `spawn15yaw`/`spawn20yaw` came back **96/96 episode-identical**
  — the first end-to-end determinism check on this pipeline, and it retro-validates every paired
  comparison in the docs.

### 6y-d. WHAT DID NOT WORK — five falsified beliefs and two retractions

| # | I predicted | reality |
|---|---|---|
| 4 | slot further = harder, nearer = easier | backwards; +10 mm free, −10 mm the worse direction |
| 6 | arm jitter 0.10 rad costs > 30 pts, **grasp breaks first** | 2.1 pts, p = 0.41; `never_lifted` **stayed 0** |
| 12 | the expert also dips at −10 mm | 125/128 = 0.977, flat |
| 19 | 2 % action noise is roughly free | **83 points** |
| 7 (half) | 20 % obs noise fails **laterally** | fails by **not advancing**; lowest lateral of any cell |

**Beliefs 4, 6 and 12 are the same error**: I kept assuming parts of this policy were open-loop.
Each was wrong in the direction of underrating how much it re-derives from the observation.

**Retraction 1 — `dx_m010`.** Round 1: −13.5 pts, p = 0.001. Round 3 on a second training seed:
−1.0 pts, p = 1.0. A checkpoint idiosyncrasy, not a task property. Retracted in place (§9d).

**Retraction 2 — the horizon differential.** Probe A round 1 gave +11.5 / −2.1 for the two failure
shapes and I wrote a "slow versus stuck" mechanism on it. Round 2 at seed 888: **+0.0 / +10.4**.
Pooled over 4 cells (n = 384/arm): **+4.9 pts, p = 0.138**, no dependence on failure shape, and it
saturates by 20 s. **The depth failures are stuck, not slow** — the opposite of what §7 said, and
better news for EXP_STEER (there is a real target rather than a config fix).

**The methodological lesson, named in `EXP_DEPTH.md` §8c:** §7 correctly flagged its own p = 0.088
as underpowered, then built a mechanism story out of `never_entered` counts **from the same two
underpowered cells**. *A second statistic computed from the same cells is not independent
evidence.* The protocol was right — the replication was already queued — the prose got ahead of it.

### 6y-e. HARNESS BUGS FOUND BY READING RESULTS (nothing errored)

* `robustness_report.py` scored `-Loose-v0` cells against the `-v0` gate → `loose_dy_p003`'s honest
  −0.167 printed as a meaningless −0.219. Fixed: per-task baselines.
* `run_expert.py` serialised the **pre-shift** `insert_x`, so all four expert cells recorded 0.2545
  and looked byte-identical in the one field distinguishing them. Runs were correct; provenance
  was not. Same class as `make_videos.sh` writing three videos to one filename.
* A comment in `eval_act.py` claimed the policy sees a **clean** `last_action` under
  `--action-noise`. It does not: `mdp.last_action` returns `env.action_manager.action`, the noisy
  action actually passed to `step()`. Corrected in place.
* `--spawn-scale` widens the **yaw** range too, and `spawn_pos` recorded only position — so an
  "in-box" split was silently mixing in-distribution positions with 2× yaws. `spawn_yaw` is now
  recorded (added 09:27:43; round-1 cells and `dy_p001` predate it and lack the field).

### 6y-f. NEW GOTCHAS (continuing the numbered list)

35. **Comparisons are not automatically unpaired.** Shifting `SLOT_CENTER` consumes no RNG, so
    dx/dy cells ARE spawn-paired with the gate. `--spawn-scale`, `--obs-noise`, `--action-noise`,
    `--arm-jitter` all break pairing. Let `robustness_report.py` decide from `spawn_pos`.
36. **Score a cell against a baseline on its own task.** A `-Loose-v0` cell judged by the `-v0`
    gate is charged for the task difference as well as the perturbation.
37. **A second statistic from the same underpowered cells is not independent evidence** (§6y-d).
38. **`bash` reads scripts lazily — never edit a shell script while it is running.** Round 2 was
    executing `run_robustness2.sh`; added cells went into a separate `run_dy_crossed.sh`.
39. **The harness kills `nohup … &` background chains** when the tool call returns. Chain GPU jobs
    with a foreground `while kill -0 <pid>; do sleep 30; done` under `run_in_background: true`.
40. **`--episode-length-s` changes the reset stream**, so horizon comparisons are unpaired even at
    the same seed. Verify from `spawn_pos`, never assume.

### 6y-g. WHAT WAS BUILT

`analysis/robustness_report.py`, `analysis/lateral_by_bucket.py`; scripts `run_robustness{2,3}.sh`,
`run_dy_crossed.sh`, `run_spawn_yaw.sh`, `run_action_noise.sh`, `run_horizon{,2}.sh`,
`run_expert_dx.sh`, `make_robust_videos.sh`; `eval_act.py` flags `--arm-jitter`, `--obs-noise`,
`--action-noise` + `spawn_yaw` in records; `run_expert.py --slot_dx`; `train_steer.py` /
`steer_wrapper.py` `--obs-noise` / `--action-noise`. `check_port.py` 16/16 and
`test_steer_cpu.py` 17/17 still pass.

**Videos for Big Will** (he reviews all rendered output; never inspected here):
`runs/bc_armB_seed0/videos/robust_dx_p010.mp4` (slot 10 mm further, 96/96) and
`robust_dy_p005.mp4` (slot 5 mm sideways, 0/96 — the numbers say the block hits the wall *face*).

---

## 7z. FORWARD PLAN — subject to change

Ordered by information per GPU-hour. **This is a plan, not a commitment**; the session-6 results
already invalidated most of the previous plan and that should be expected to happen again.

**0. EXP_STEER** — §1-NEXT above. Plumb `eval_steer.py`, run the arm-C gate, then arm A + a clean
control arm, two seeds. This is the highest-value open question and the one Big Will asked for.

**1. If EXP_STEER arm A works, run the hard version** (belief 3): noise on the *steering*
observation too. A steerer that sees cleanly is solving a much easier problem, and the caveat is
better measured than confessed.

**2. If EXP_STEER arm A fails (< 0.30)** — the push-under-noise behaviour is outside the base
policy's support. Then the honest move is **noise-augmented BC**: regenerate demos (or just
re-train) with action noise injected, and see whether the base policy learns the correction
directly. Cheaper than it sounds — `collect_demos.py` and the expert both exist, and the expert is
robust (128/128) so its demonstrations under noise would still be clean-labelled.

**3. Attack the staging freeze directly, without RL.** The block is held at x = 0.166, aligned,
stationary. Two cheap probes: (a) does a *longer* chunk (or a shorter one) change it? — chunk size
is the parameter the mechanism implicates; (b) does `--fixed-x0 zeros` at that state push? — one
command, and `--fixed-x0` is already implemented.

**4. More training seeds beats more of anything else.** Seed variance is 15–29 points, larger than
almost every effect measured. Session 5 showed the learning curve plateaus at 30k steps
(McNemar χ² ≤ 2.118), so a seed costs ~1/3 of what the sweep paid.

**5. Not worth doing** — each killed by evidence, with the evidence named:
   * *Collecting `-Tight-v0` demos* — `ExpertParams` has no clearance term (`expert/plan.py`).
   * *Chasing lateral precision* — stalled failures have the same lateral error as successes on
     Tight (`EXP_DEPTH.md` §6).
   * *`--fixed-x0 zeros` as a general fix* — replicated 1 improved / 4 worse (`EXP_TIGHT.md` §7).
   * *Longer episodes as a fix for depth* — +4.9 pts, p = 0.14, saturates by 20 s (`EXP_DEPTH` §8).
   * *Making the slot y-robust by training* — geometrically impossible on this observation (§12).
   * *Retraining the grasp-bit MLP* — degenerate finger channels, zero negative class (session 5).

**Standing constraints:** `eva_rl` is shared — **0 modifications, and it must stay that way**.
`eva_bc` has only untracked `slot/` and `docs/slot/`. **No push is authorised** — Big Will said
"when i tell you to". Use `env_isaaclab6`. One GPU job at a time.

---
---

# ▶▶ SESSION 7+8 FULL RECORD (2026-08-03) — EXP_STEER, then the VISUAL policy

*Written 20:35 while vision collection runs. This session covers two distinct efforts: the
x0-steering investigation (§S7) and the start of the visual policy (§S8). Read §S8-STATE first
if you are picking this up cold.*

## S8-RESULT. ✅ THE VISUAL POLICY MEETS THE BAR (2026-08-04 03:36)

| arm | s777 | s888 | **pooled** | Wilson 95 % |
|---|---|---|---|---|
| blind (23-D proprio, images zeroed) | 0.223 | 0.286 | **0.254** | [0.202, 0.315] |
| **vision** (wrist + workspace + proprio) | 0.786 | 0.821 | **0.804** | [0.747, 0.850] |

* vision - blind **+54.9 pts**, z = +11.64, **p = 2.5e-31**
* champion in the vision env (G2) **0.949**; privileged-state champion 0.979
* trained on seeds 101/202, evaluated on 777/888 — **no spawn overlap**

**Honest reading:** 0.804 clears the 0.80 bar on the pooled POINT ESTIMATE, by 0.4 of a point,
and the Wilson interval contains values below 0.80. Both seeds straddle. The *comparison* to the
control is not in doubt; the *bar* would want more episodes or DAgger to be clear.

**Vision is doing the perceiving.** Median |lateral| of failures: blind **31.63 mm**, vision
**0.52 mm** (clearance 1.5 mm). `gross_miss` 35 -> 1, `never_lifted` 18 -> 0. Perception failure
became precision failure — pre-registered as belief 4.

Full detail + belief scorecard: `docs/slot/VISION_PLAN.md` section 12.
Checkpoints: `slot/runs/vision_bc/{blind,v1}/ckpt_final.pt`.

### S8-RESULT-b. The DAgger round REGRESSED — v1 is still the deliverable

| arm | pooled | Wilson 95 % |
|---|---|---|
| blind | 0.254 | [0.202, 0.315] |
| **v1 — BC only** | **0.804** | [0.747, 0.850] |
| v2 — BC + DAgger | **0.491** | [0.426, 0.556] |

**−31.3 pts, z = −6.92, p = 4.5e-12**, both seeds. The collection was CLEAN — audit 0.801 vs a
0.804 baseline (p = 0.939), labels distributionally identical to BC actions. The damage is in
*which states were labelled*: `gross_miss` **1 → 41**, median |lateral| of failures
**0.52 mm → 32.07 mm** (the blind control's signature). **DAgger broke perception, not precision.**

**Cause:** DAgger assumes the teacher can recover from any state the learner visits. Our champion
is a **BC clone of a scripted open-loop expert** — it has never seen a dropped block. ~20 % of
DAgger episodes are student failures whose late boundaries are exactly those states; the champion
emits plausible-looking actions there and they were trained on as ground truth. EXP08's clause
*"ALL kept — labels are champion-quality regardless of the student's outcome"* was taken verbatim
without checking that its premise holds for a cloned teacher. It does not.

**Second design error:** BC targets are the stitched EXECUTED stream (re-planned every 15 steps);
DAgger targets are a single open-loop 50-step plan. Different conditional distributions past the
first window, mixed 93:7.

**Fix needs NO re-collection** (obs34 is stored per boundary) — see `VISION_PLAN.md` §13d:
truncate labels where the block leaves the champion's manifold, or success-only DAgger.
**AWAITING BIG WILL'S CALL** between that and spending the time on resolution instead.

**Next, cheapest first (VISION_PLAN 12d):** more eval episodes to tighten the interval (no
training); 100k steps instead of 60k; **higher policy resolution** (160x90 is EXP08's basket-drop
pick and a 1.5 mm clearance may want more pixels — the change I would bet on); then DAgger
(`S8c`), which is now for moving clear of the bar rather than reaching it.

---

## S8-STATE. ⏳ WHAT WAS RUNNING (all COMPLETE as of 03:36)

```
bash -c 'for S in 101 202; do python scripts/collect_vision.py --episodes 128 \
    --num-envs 16 --seed $S --out data/vision_bc/seed$S; done'
```

* **seed 101: 80/128 episodes done, champion success 0.963** (reference 0.979 — healthy).
  ~14 min left. Then seed 202 starts, ~40 min.
* Log: `/tmp/claude-1000/.../scratchpad/collect.log`. **Idempotent**: `meta.json` marks a
  finished seed, so re-running the loop resumes rather than redoing.
* Output: `slot/data/vision_bc/seed{101,202}/ep_XXXX.pt`, ~52 MB/episode, ~13 GB total.

**When it finishes, the whole next stage is one command:**

```
bash scripts/run_vision_bc.sh          # blind control, then visual policy, then both evals
```

`STEPS=60000` by default; it trains the **blind arm first on purpose** (see S8-c).

---

## S8-a. The visual policy — what was built

Big Will's directive: *"start training the VISUAL policy… cameras should already be defined in
the RL github, check commit e56e7df… We want visual to achieve 80%+ success rate. The visual
policy should use NO privileged information (visual + proprioception only)"*, then
*"there is a new commit in eva_bc that implements the visual backbone, lets use that! Yes we want
our visual policy to still use flow!"*

**The contract (VISION_PLAN §0).** Student = wrist D405 RGB 160×90 + workspace D455 RGB 160×90 +
**23-D proprio** = `obs34[0:16] ⊕ obs34[27:34]`. Deleted as privileged: `block_pose_in_root`
(`[16:23]`) and `slot_frame` (`[23:27]`) — the latter's 4th element **is the success predicate**.
The scripted expert / champion teacher stays privileged; only the deployed policy is constrained.

**Ported from EXP08 (`c10bf1a`), `act/` → `slot_act/`:**

| file | how it ported |
|---|---|
| `modeling_flow_vision.py` | **verbatim**, one string differs. `slot_act/modeling_flow.py` is byte-identical to `act/`'s, so a verbatim copy means future divergence shows as a diff, not drift. `FlowMatchingVisionPolicy` = flow head, env_state token replaced by ResNet camera tokens; 15 tokens/camera at 160×90; encoder seq = `[state, wrist×15, workspace×15]` = 31 tokens. |
| `dataset_vision.py` | `obs41` → `obs34`; student contract identical at 23-D |
| `train_flow_vision.py`, `eval_flow_vision.py` | `act.*` → `slot_act.*` |

**Written new:**

* **`slot_act/cameras.py`** — post-`parse_env_cfg` camera attach (so **`eva_rl` stays untouched**;
  EXP08's `Rebot-PickPlace-Vision-Play-v1` does not exist in this checkout and we do not need it),
  the single `student_proprio()`, `audit_no_privileged()`, `rgb()`/`rgb_native()`,
  `frame_freshness()`, `wrist_camera_pose()`.
* **`scripts/vision_g0.py`** — camera render/freshness/depth gate.
* **`scripts/vision_render_probe.py`** + **`run_render_sweep.sh`** — render-quality sweep.
* **`scripts/vision_shimmer_probe.py`** — temporal-shimmer measurement.
* **`scripts/vision_fps.py`** — throughput vs (supersample, num_envs).
* **`scripts/collect_vision.py`** — champion → shards, EXP08's no-GPU-work rule obeyed.
* **`scripts/eval_vision.py`** — our own runner (the ported eval wants a registered `student`
  obs group we deliberately do not have).
* **`scripts/run_vision_bc.sh`** — blind control then visual policy, idempotent.

**Trainer/dataset changes:** `--blind` (zero the images, identical architecture) and the
**render contract** — dataset carries it from shards and asserts all shards agree, trainer stamps
it into the checkpoint, `eval_vision.py` refuses a mismatch (including blind-vs-sighted).

## S8-b. The measurements that decided the configuration

**G0 — cameras (PASSED).** Champion driving, 1 env, 300 steps: both cameras `frozen 0/299`;
wrist depth-min **0.0244–0.0504 m** for all 300 frames (the gripper housing sits a few cm in
front of the D405 — proof the *render* rides the link even though `Camera.data` poses are frozen
at spawn); no near-uniform frames. Stills in `slot/runs/vision_g0/{wrist,workspace}/`.

**Render quality — Big Will rejected the first stills as "super noisy". He was right.**
Isaac defaults `samples_per_pixel = 1` and I had set AA off. Sweep at a pinned pose
(`runs/vision_render_probe/`), noise proxy on the 160×90 frame the policy consumes:

| variant | render res | noise | vs 4× ref |
|---|---|---|---|
| `off_spp1` (what G0 shipped) | 160×90 | 27.51 | 29.28 |
| `off_spp8` | 160×90 | **27.51** | 29.28 |
| `fxaa_spp8` | 160×90 | **27.51** | 29.28 |
| ss2 | 320×180 | 14.24 | 16.15 |
| ss4 | 640×360 | 7.88 | — |
| ss8 | 1280×720 | 4.64 | 8.16 |
| ss16 | 2560×1440 | 3.20 | 7.53 |
| `dlss_spp1` | 160×90 | 2.35 | **22.58** |

* **`samples_per_pixel` and `FXAA` are NO-OPS in this build** — bit-identical output. Do not
  waste time on them again.
* Noise falls as **1/k** with supersample k — classic independent-per-pixel-noise averaging.
* **DLSS is rejected**: `vs_ref` 22.58 means it reconstructs from a lower internal resolution
  rather than denoising, and it is EXP08's prime suspect for GPU-load-dependent frames.

**Shimmer (the decisive one).** After pulling EXP08's `2846bb8`, I retested my own G0 claim and
**it was wrong** — see S8-e. Per-pixel **temporal** std on a held pose, 40 frames:

| render | wrist temporal std | pixels > 2 | drift |
|---|---|---|---|
| 1× | **35.24** | 99.8 % | 37.9 |
| 4× | 9.06 | 99.9 % | 10.7 |
| 8× | 4.68 | 99.4 % | 5.8 |
| DLSS @ 1× | 3.89 | 70.0 % | 13.6 |

**EXP08's independently measured 31–37 reproduces exactly.** Their diagnosis is right. But their
conclusion (take DLSS) does not follow for us, because they only compared non-temporal modes at
1×; supersampling reaches DLSS-class stability **without a temporal filter**, hence without the
GPU-load coupling that floored their DAgger driver 80 % → 40 %.

**Throughput decided `supersample = 4`, not 8** (`scripts/vision_fps.py`):

| config | envs | ep-steps/s | 256 eps | temporal std |
|---|---|---|---|---|
| no cameras | 16 | 369.0 | 0.12 h | — |
| ss2 | 16 | 79.6 | 0.54 h | ~18 |
| **ss4** | **16** | **35.8** | **1.19 h** | **9.06** |
| ss8 | 8 | 10.5 | 4.06 h | 4.68 |
| ss8 | 16 | **CUDA OOM** (11 GB card) | — | — |

8× is memory-bound to 8 envs and 3.4× slower per episode. 4× keeps 16 envs and still cuts raw
shimmer 3.9×. **Residual shimmer at 9.06 is the live suspect if the student's failures look like
perception rather than precision** — bump to 8 and re-collect (which invalidates all shards).

## S8-c. Why the blind control runs FIRST

The slot **never moves** (`SLOT_CENTER = (0.245, 0.0)`, welded). The only randomisation is the
block spawn: x ± 20 mm, y ± 30 mm, yaw ± 0.35 rad. So a blind policy can execute the entire
insertion — it just cannot find the block, and an 89 mm gripper opening on a 30 mm block absorbs
a lot of ± 30 mm.

**If blind ≥ 0.60, the eval barely tests vision** and no visual number means anything until the
spawn box is widened. Running blind first puts that verdict on the table *before* there is a
visual number to be pleased about. `run_vision_bc.sh` enforces the order.

## S8-d. Four traps inherited from EXP08 — adopted as code, not memory

1. **Train/test render mismatch is catastrophic, not gradual.** DLSS-trained student evaluated
   AA-off: **0.0 % / 1.6 %**. → the render config is part of the dataset, stamped and asserted.
2. **Optional GPU work in a policy-driving loop perturbs frames** (their in-loop DAgger collector
   cost 40 points). → our collection loop does buffer reads + one host copy, nothing else;
   `student_proprio` is computed after the episode ends.
3. **Renderer cold start**: their Gate C 67.2 % includes a cold round-1 at 6/16; warm is 78–83 %.
   → `--warmup-episodes 1` discards the first episode per env (also our PhysX rule).
4. **Quarantine data from a suspect config** — they excluded v1 DAgger data outright.

## S8-e. TWO OF MY OWN INSTRUMENTS WERE BROKEN, AND BOTH FAILED THE SAME WAY

Worth naming as a pattern: **a check whose passing condition is guaranteed by its own
implementation.**

1. **G0 v1 reported both cameras `frozen 299/299`.** The cameras were fine. `rgb()` did
   `data[..., :3].to(uint8)` — a **no-op on already-uint8 data that returns the same tensor** — so
   "last frame" and "this frame" were one buffer and the diff was structurally 0. Fixed by
   `.clone()`.
2. **G0's "static-render diff = 0.0 → deterministic render"** — I reported this to Big Will as
   *clean*. It called `sim.render()` twice **without advancing the frame index**, and the jitter
   is frame-index-deterministic, so 0.0 was guaranteed. The real number is **35.24**. Corrected
   in VISION_PLAN §11b.

## S8-f. Gotchas added this session

41. `.to(uint8)` on uint8 is a no-op returning the SAME tensor — clone before storing as "previous".
42. `sim.render()` twice does not advance the render frame index; to see temporal artifacts you
    must `env.step()`.
43. `samples_per_pixel` and `antialiasing_mode="FXAA"` are no-ops in this Isaac Lab build.
44. Supersample 8 at 16 envs OOMs an 11 GB card with two cameras.
45. Nulling `terminations.block_dropped` without also nulling `rewards.dropping_penalty` raises at
    manager build — the penalty is an `is_terminated_term` that resolves its termination by name.
46. **Never collect training data on an eval seed.** My first instinct was seed 777 — the seed
    every eval on this project uses. Collection runs 101/202; 777/888 stay clean.

---

## S7-RESULT. EXP_STEER — the summary that matters

Full detail in `docs/slot/EXP_STEER.md` (867 lines). The short version:

* **Gate passed in its strong form**: `eval_steer.py` reproduces `eval_act.py`
  **episode-for-episode**, 96/96, zero mismatches.
* **A constant integration latent takes the champion 0.146 → 0.823** under 2 % action noise,
  with **no training at all** (p = 6 × 10⁻²¹). 81 % of the deficit, recovered by sampling choice.
* **Every latent fails the same way** — frozen at the staging waypoint, block at carried height,
  on-axis. Only the *frequency* changes, and the frozen fraction is monotone in success.
* **The steering action space cannot contain the good latent.** `SteerCore` broadcasts one
  7-vector across all 50 chunk positions; all four broadcast cells score **0.000/96**, two by
  never lifting the block. **The pre-registered PPO launch was withdrawn.**
* **Selection ≫ commitment**: +51.4 vs +16.3 points. (An earlier +37/+30 split from n = 4 was
  **wrong and is corrected in place** — EXP_STEER §11c.)
* **Latent selection is saturated**: the oracle over 9 latents is 0.885 vs 0.823 for the best
  single — **+6.2 pts** — and it did **not move** when the pool nearly doubled. 11/96 episodes
  are failed by every latent.
* **Norm is a live axis**: 0.302 / 0.573 / 0.771 / **0.875** at 0.30× / 0.76× / 1.00× / 1.50×,
  not turned over at ‖x0‖ = 29.3 (prior shell 18.7). Steering caps at 14.25 and initialises at
  5.61 — the wrong half of the axis.
* **Not free**: the good latent costs **14.6 pts clean** (0.833 vs 0.979) and buys **nothing** at
  5 % noise (0.000).

**UNVALIDATED**: 0.823 is the max of 9 candidates on the *same* 96 episodes.
`scripts/run_x0_holdout.sh 4` has **not run**. Do not quote it as a headline until it does.

---

## S7-S8-PLAN. FORWARD PLAN — subject to change

**Priority 1 — finish the visual policy (the live deliverable).**

1. Collection completes (~1 h left). Read **G2**: pooled champion success must be ≈ 0.979
   (accept 0.93–1.00). Below that, suspect camera-load physics or frame sync — *not* the teacher.
2. `bash scripts/run_vision_bc.sh` → blind control trained + evaluated, then the visual policy.
   Read the visual number **only** against blind on the same seed.
3. **Decision rule** (VISION_PLAN §6): blind ≥ 0.60 → widen the spawn box before claiming
   anything. Visual ≥ 0.50 → proceed to DAgger. 0.35–0.50 → one architecture iteration
   (resolution first, given the 1.5 mm clearance). < 0.35 → stop and diagnose per phase.
4. **DAgger to the bar.** EXP08's ladder was teacher 93.75 % → BC 67.2 % → DAgger targeting ≥ 90.
   Ours: teacher 0.979 → BC ? → DAgger ≥ 0.80. Port `exp08_dagger_collect_v2.py`, which computes
   labels **post-hoc** (validated exact to 7.8e-8) precisely to avoid the in-loop perturbation.
5. Two seeds before any headline.

**Priority 2 — close the EXP_STEER loose ends (cheap, ~1 h total).**

* `bash scripts/run_x0_holdout.sh 4` — validate 0.823 out of sample. **Blocking for any claim.**
* `bash scripts/run_x0_norm.sh` — is scaled *sampled* x0 a general robustness knob (no latent
  search)? Stopped mid-run at Big Will's redirect; idempotent.
* `bash scripts/run_x0_bcast_ladder.sh` — the path PPO would have walked, for the record.
* `bash scripts/run_obs_shift.sh` — EXP_STEER §8d + the causal cell (`shift_act002_s4`).

**Priority 3 — repo hygiene.**

* `check_port.py` does not yet cover the four new vision files.
* The 116 eval JSONs the docs cite are gitignored under `runs/` — Big Will was offered tracking
  them and has not decided.
* Vision work is **uncommitted**: `slot_act/cameras.py`, `slot_act/*vision*.py`,
  `scripts/vision_*.py`, `scripts/collect_vision.py`, `scripts/eval_vision.py`,
  `scripts/run_vision_bc.sh`, `scripts/run_render_sweep.sh`, `docs/slot/VISION_PLAN.md`.

**Explicitly NOT worth doing, with the evidence that killed it:**

* x0-steering PPO as pre-registered — the action space is a joint-bias term (§12).
* A contextual bandit over latents — oracle headroom is +6.2 pts and did not grow (§13b).
* Screening latents offline by summary statistics — seeds 1 (0.771) and 3 (0.031) are
  indistinguishable on norm, DC, AC spread and gripper DC (§13c).
