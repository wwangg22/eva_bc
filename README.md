# eva_bc — staged flow-BC + residual / x0-steering RL pipeline for pick-and-place

A staged pipeline for training pick-and-place manipulation policies in **Isaac Lab 3.0**
(state-based, no vision). It goes: scripted planner expert → flow-matching chunk BC →
batched sim evaluation → DAgger → a validated grasp-success observation bit → RL on the
frozen BC base (additive residual as a measured-flat baseline, **x0-steering** as the
recommended approach).

The code was developed on a specific 2-can pick-and-place task ("reBot") and is published
as a **working reference**, not a polished library: every number quoted below is from that
task, and the full lab-notebook history lives under `docs/`.

## Philosophy

Two rules shaped everything here, and they are the reason the repo looks the way it does:

1. **Staged, with evaluation gates.** Each stage is a separate script with its own
   held-out evaluation gate that must pass before the next stage runs. No stage is merged
   into another; no training script contains its own eval loop. Wrappers are verified by
   *bit-exact* reproduction of the previous stage's behavior before any training happens
   on top of them.
2. **Pre-registered experiments.** Before coding each experiment, its design, beliefs,
   and decision rules were written into a doc (`docs/experiments/EXP0*.md`). Verdicts —
   including retractions of earlier claims — are recorded in place. `docs/POSTMORTEM.md`
   carries dated CORRECTION/UPDATE blocks rather than silent edits.

## Pipeline stages

### Stage 1 — Scripted planner expert (`expert/`)

**Purpose:** a cuRobo-based kinematic expert drives the sim task with privileged state and
records demonstrations to HDF5, with per-segment phase labels and a `train_mask` that
loss-censors the expert's own failed sub-attempts (misses/losses are never supervised).

Scripts: `expert/onboard_robot.py` (build the cuRobo robot config from the URDF),
`expert/spike_plan_grasp.py` (GO/NO-GO feasibility spike before committing to the expert),
`expert/run_expert_v1.py` (the expert runner/recorder), `expert/gen_demos_nominal.sh`
(seeded batch demo generation), `act/report_coverage.py` (stratification/coverage report
over demo files).

```bash
python expert/run_expert_v1.py --episodes 8 --video          # bring-up + visual check
bash expert/gen_demos_nominal.sh                              # 8 seeds x 63 eps
python act/report_coverage.py demos_nominal_s101.h5 ...       # coverage audit
```

**Gate:** expert success rate and failure anatomy on a fixed suite (measured here: ~94%
nominal, 77.4% under perturbation — the perturbed number matters because it caps what
DAgger can later teach). Root-cause tooling for expert failures is included
(`expert/replay_place_fail.py`, `expert/probe_*.py`, `expert/debug_ik*.py`,
`expert/extend_grasp_table.py`).

### Stage 2 — Flow-matching chunk BC (`act/train_flow.py`)

**Purpose:** train a rectified-flow chunk policy on the demos. Architecture in
`act/modeling_flow.py`: a flow head grafted onto the vendored LeRobot ACT transformer
(`act/modeling_act.py`, see `act/PROVENANCE.md`), CVAE/KL deleted, `action_is_pad`
censoring kept exact via a decoder `key_padding_mask`. Chunk 50, execute 15, temporal
ensembling OFF, 10 Euler steps, seedable x0 (bit-deterministic inference given x0).
External mean/std normalization (`act/normalize.py`); dataset in `act/dataset.py`.

```bash
python act/train_flow.py --data demos.hdf5 --out runs/flow_nominal --steps 100000 --seed 1
```

`act/train_act.py` is the legacy plain-ACT trainer — kept because the flow model/eval
config construction imports from it, and as the L1/CVAE baseline.

**Gate:** held-out batched sim eval (Stage 3), **≥3 training seeds**, champion selected on
a *held-out spawn seed* and confirmed on pooled ≥128-episode numbers. Single-run
comparisons are void on this recipe (see Lessons Learned).

### Stage 3 — Batched sim evaluation (`act/eval_act.py`)

**Purpose:** roll a checkpoint across N parallel envs with per-env action queues
(receding horizon) and a privileged "flush" trigger: on a detected can discontinuity the
env's queue is cleared so the next step re-predicts from fresh observations. The
controller state is fully vectorized (tensor queue, not python deques) — verified
bit-exact against the original implementation, scales to 2048 envs.

```bash
python act/eval_act.py --ckpt runs/flow_nominal/ckpt_final.pt --episodes 64 --num-envs 16 --seed 42
python experiments/taxonomy.py eval1.json eval2.json          # failure-bucket anatomy
python experiments/taxonomy.py --diff base.json other.json    # per-episode transitions
```

**Gate:** this *is* the gate for every other stage. Deterministic per spawn seed
(same-checkpoint re-run churn = 0), so per-episode diffs against a known churn floor are
meaningful.

### Stage 4 — DAgger collection (`expert/collect_dagger.py`)

**Purpose:** HG-DAgger-style gated collection. The frozen BC policy drives; simple
failure gates (miss / stall / drop / timeout) trip an expert takeover from the current
sim state; only the *recovery* behavior is labeled trainable (`train_mask` hard-zeroed
before the takeover step).

```bash
python expert/collect_dagger.py --ckpt runs/flow_nominal/ckpt_final.pt \
    --rollouts 80 --target-takeovers 40 --seed 203 --out dagger_r2_s203.h5
```

**Gate:** retrain (Stage 2) on nominal+DAgger with ≥3 seeds per arm and compare pooled
held-out numbers against the nominal arm's seed spread — on this task the honest verdict
was "no measurable interference, weak evidence of stabilization" (`docs/experiments/EXP03_dagger_interference.md`).

### Stage 5 — Grasp-success-bit probe (`experiments/exp06_grasp_bit.py`)

**Purpose:** a small MLP on 5 raw obs dims (finger pos + finger vel + last commanded
grip) that outputs a validated grasp-success bit, used as an extra observation feature
for the RL stages. CPU-only; exports the probe weights (`exp06_grasp_bit.pt`,
regenerated per task — not shipped in this repo).

```bash
python experiments/exp06_grasp_bit.py     # writes exp06_grasp_bit.pt + results json
```

**Gate (pre-registered):** **0% false-positive rate** on on-policy closed-on-air freeze
states AND ≥95% accuracy on expert post-close frames. Runtime semantics: probe output
ANDed with commanded-closed (an open command forces bit = 0). The upstream probe study is
`experiments/exp01_probe.py` / `docs/experiments/EXP01_grasp_aliasing_probe.md`.

### Stage 6 — RL on the frozen BC base (two variants, both rl_games PPO)

#### 6a. Additive per-step action residual (baseline / ablation — measured FLAT here)

`act/residual_core.py` (64-D obs builder + `arm = base + alpha * tanh(res)` blending),
`act/residual_wrapper.py`, `act/train_residual.py`, `act/eval_residual.py`,
`act/residual_ppo_cfg.yaml`.

```bash
# gate 2a: zero residual must reproduce the base eval episode-for-episode
python act/eval_residual.py --ckpt runs/exp03_N3/ckpt_final.pt --seed 42 \
    --x0-mode global --out runs/exp06/gate2a_seed42.json
python act/train_residual.py --ckpt runs/exp03_N3/ckpt_final.pt \
    --run-name r1_seed1 --seed 1 --num_envs 128 --max_iterations 3000
```

On this task the healthy run came out *exactly* flat (55.5% → 55.5%, 26 episodes fixed /
26 broken, causal because both sides are deterministic). Kept as the honest baseline and
because all its infrastructure carries over to 6b. Full arc:
`docs/experiments/EXP06_residual_rl.md`.

#### 6b. x0-steering (recommended — RFS-style, arXiv 2602.01789)

Chunk-level RL steers the flow policy's integration noise x0: one RL action per 15-step
execution window, `x0 = alpha_x0 * tanh(z)` held for the window, window-summed reward.
Outputs are always on-manifold — the base's own decoder turns x0 into a coherent chunk,
and the grip channel is influenced *through* the base rather than being handed to RL.
`act/steer_core.py`, `act/steer_wrapper.py`, `act/train_steer.py`, `act/eval_steer.py`,
`act/steer_ppo_cfg.yaml`.

```bash
# gate 1: z=0 must reproduce the x0-zeros base BIT-EXACTLY
python act/eval_steer.py --ckpt runs/exp03_N3/ckpt_final.pt --seed 42 \
    --out runs/exp07_steer/gate1_seed42.json
python experiments/exp07_check_match.py gate1_seed42.json x0zeros_base_seed42.json
python act/train_steer.py --ckpt runs/exp03_N3/ckpt_final.pt \
    --run-name s1_seed1 --seed 1 --num_envs 2048 --max_iterations 200
```

Why steering over additive: fixed x0 draws alone span **14.1%–56.2%** success on the same
frozen base (real leverage to steer), and the RFS literature reports plain residuals at
43% vs x0-steering at 86% average. Pre-registered design:
`docs/experiments/EXP07_x0_steering.md`.

**Gates (both variants, ordered):** (i) wrapper bit-exactness vs the base;
(ii) early-health check — per-term reward channels sane from epoch 1;
(iii) pooled ≥128-ep held-out result vs the deterministic-base baseline with a
pre-registered +5-pt adoption rule, plus a mandatory taxonomy diff (did the targeted
failure bucket collapse *without* symmetric new breakage?).

### Stage 7 — Diagnostics (cross-cutting)

- `act/diag_training_env.py` — fixed-action attribution harness: reproduces the exact
  training configuration and rolls a small matrix of fixed action conditions (zero /
  bias / small noise / large noise) through it, no RL. The tool for any "training reward
  looks wrong" mystery.
- `experiments/taxonomy.py` — failure-bucket classifier + per-episode A/B diff over eval
  JSONs.
- `experiments/exp07_check_match.py` — bit-exactness comparator for wrapper gates
  (exit 0 = match, exit 1 = chain must abort).
- `experiments/exp03_grip_divergence.py`, `exp03_divergence_forensics.py`,
  `exp03_analyze.py`, `exp06_analyze_r3.py` — offline behavior-divergence and
  failure-transition analyses (see the caveats about offline probes in Lessons Learned).

## Repo layout

| Path | Contents |
|---|---|
| `act/` | Policy, training, eval, RL wrappers. `modeling_act.py` (vendored ACT, see `PROVENANCE.md` + `LICENSE`), `modeling_flow.py`, `train_flow.py` / `train_act.py`, `dataset.py`, `normalize.py`, `eval_act.py`, residual RL (`residual_*.py`, `train_residual.py`, `eval_residual.py`, `residual_ppo_cfg.yaml`), x0-steering (`steer_*.py`, `train_steer.py`, `eval_steer.py`, `steer_ppo_cfg.yaml`), `diag_training_env.py`, `report_coverage.py` |
| `expert/` | cuRobo expert: `onboard_robot.py`, `spike_plan_grasp.py`, `run_expert_v1.py`, `collect_dagger.py`, grasp-table extension + failure-forensics tools, robot configs (`rs_rebot.yml`, `rs_rebot_scpatch.yml`), batch scripts |
| `experiments/` | Runnable experiment code: probes, analyses, taxonomy, gate comparators, eval chains |
| `docs/` | `PLAN.md` (design), `POSTMORTEM.md` (why BC plateaued — with dated corrections), `HANDOFF.md`, `JOURNAL.md` (running log) |
| `docs/experiments/` | Pre-registered experiment docs `EXP01`–`EXP07` + `EXP_INDEX.md` + `LITERATURE.md` |
| `docs/expert/` | Expert-side probe write-ups |

Note: `experiments/exp03_analyze.py` and friends reference eval JSONs under `runs/`
(excluded from this repo) — they are reference implementations of the analysis method,
runnable once you have your own eval outputs.

## Dependencies

- **Isaac Lab 3.0** + **Isaac Sim** (the versions this was developed against). Note
  Isaac Lab 3.0 quaternions in obs here are **XYZW**.
- **PyTorch** (any recent CUDA build that your Isaac Sim install supports).
- **rl_games 1.6.1** (Stage 6; and read Lessons Learned item on `clip_actions` before
  touching the yaml).
- `h5py`, `numpy` (data + analysis).
- **cuRobo** (Stage 1 expert only; the expert code imports a local motion-planner
  wrapper API — `curobo.motion_planner.MotionPlanner` — adapt to your cuRobo install).
- Originally run in a single conda env (`env_isaaclab6`) containing all of the above.

### Bring your own task env (required)

This repo deliberately does **not** include the Isaac Lab task package it was developed
against (`reBot_RL`, task id `Rebot-PickPlace-Play-v1`). You supply your own Isaac Lab
**manager-based** task env plus a small `mdp` interface module. The touchpoints, all
visible in `act/eval_act.py` (`main()`), `act/residual_core.py`, and `act/steer_core.py`:

- `import <your_pkg>.tasks` — registers the gym task id (passed via `--task`).
- `mdp.placed_mask(env) -> (N,) bool` — the task success predicate.
- `mdp.object_pos_local(env, name) -> (N, 3)` — per-object env-local position.
- `mdp.basket_centers_local(env) -> (N, 2)` — goal-container centers, env-local.
- `mdp.OBJECT_NAMES` — canonical object name list.
- An `objects_canonical` observation term (target-first canonical ordering) — the
  residual/steer obs builders mirror its target-selection logic exactly.

Also note: expert scripts and `expert/rs_rebot*.yml` contain machine-specific absolute
paths (robot URDF/asset root, grasp table `.pt`) that must be re-pointed to your own
robot assets. Small model artifacts (the grasp-bit probe `exp06_grasp_bit.pt`, grasp
tables) are not shipped — they are task-specific and regenerated by the corresponding
scripts.

## Adapting to a new pick-and-place task — touchpoint checklist

1. **Observation layout** — `act/dataset.py` hardcodes the 41-D obs split
   (16 proprio + 25 environment-state) and documents each slice. Update the dims and the
   slice map to your env's policy obs (read them from your env config, don't guess), and
   update the slice constants reused in `act/report_coverage.py`,
   `act/residual_core.py`, `act/steer_core.py`.
2. **Action dim** — 7 here (6 arm-joint position targets + 1 binary grip). Update in the
   dataset/config plumbing and the residual/steer action handling (arm-dims vs
   grip-channel split).
3. **The `mdp` interface** — implement the five touchpoints listed above in your task
   package.
4. **Expert** — onboard your robot into cuRobo (`expert/onboard_robot.py` pattern), run a
   feasibility spike (`expert/spike_plan_grasp.py` pattern) *before* building the full
   runner, then adapt `expert/run_expert_v1.py` phases to your task.
5. **Retrain the grasp-bit probe** — `experiments/exp01_probe.py` then
   `experiments/exp06_grasp_bit.py` on your own demos + on-policy failure states; hold
   the 0%-FPR gate.
6. **Re-select the champion properly** — ≥3 training seeds, evaluate every candidate on
   a held-out spawn seed, pool ≥128 episodes; only then freeze a base for RL.
7. **Re-verify every gate in order** — wrapper bit-exactness gates are cheap and
   permanent; run them after any controller/wrapper change.

## Lessons Learned

These are the distilled, measured lessons from the full experiment ladder
(`docs/POSTMORTEM.md`, `docs/HANDOFF.md`, `docs/experiments/EXP0*.md`). They cost real
GPU-weeks; read them before running anything.

- **Training-seed variance dominates single-run A/Bs.** The same data + recipe spanned
  **32.8%–59.4%** success across training seeds, and same-data different-seed pairs
  flipped 31–39 of 64 episodes — which is exactly the flip count of the "treatment"
  comparison it invalidated. Several confident earlier verdicts ("DAgger nets zero",
  "offline recovery data actively hurts") turned out to be single unlucky/lucky seeds.
  Standing rule: **≥3 seeds per arm, champion selected on held-out spawn seeds, pooled
  ≥128-episode numbers only; single-run comparisons are void.** Per-episode diffing
  between two separately-trained policies is nearly meaningless until you've calibrated
  the seed-replica churn floor.

- **Chunk commitment is load-bearing.** Shortening the execution horizon at eval time
  (no retraining) collapsed success monotonically: **59.4% → 32.8% → 3.1% → 0% → 0%** at
  n_action_steps = 15/8/4/2/1. The open-loop window is not a latency cost being paid —
  it is *why the policy works* (within-chunk commitment carries it through states where
  single-step re-prediction dithers). Consequence for RL design: never shorten the
  horizon; put RL *on top of* committed chunks (per-step additive) or *at chunk
  granularity* (x0-steering).

- **rl_games `clip_actions` is an action SCALE, not a clip.** `preprocess_actions`
  clamps the policy sample to [-1, 1] and then **rescales it to the action-space
  bounds** — so `clip_actions: 100.0` multiplied every action by 100, saturating the
  blend and shoving the arm at full amplitude from step one. Two full training runs
  (and two wrong post-hoc theories) were spent before reading the rl_games source.
  It must be **1.0** here. General form of the lesson: every `env:` block value in an
  adapted RL config is semantically load-bearing; read the consumer's source.

- **"Zero-init residual" needs three things, not one.** (1) zero the mu *weight*
  (rl_games `mu_init` touches only the weight); (2) know that the mu *bias* stays
  randomly initialized (measure whether that's harmless — here it was); (3) a **small
  initial sigma** — initial exploration noise is part of the starting condition, not a
  free parameter. sigma ~= 0.37 (about 1 degree/joint/step) destroyed a base that lives
  on mm-precision grasps (~46 pts of success), while sigma ~= 0.08 was healthy.
  Measure the noise-tolerance of the frozen base *before* training (see next item).

- **Fixed-action attribution diagnostics beat theorizing.** When training reward looks
  wrong, don't interpret training logs — reproduce the *exact* training configuration
  and roll fixed action conditions through it (zero action / constant bias / small
  noise / large noise), no RL (`act/diag_training_env.py`). One 10-minute run pinpointed
  what two rounds of training-log post-hoc analysis had misdiagnosed: nothing an
  epoch-0 agent emits could produce the observed reward, so the wrapper/config was
  broken before learning began (it was `clip_actions`).

- **Verify wrappers by bit-exact reproduction before any training.** Every wrapper
  (residual path, steering path, vectorized controller rewrite) was gated on
  reproducing the frozen base's eval **episode-for-episode, bit-exact** with the
  learned component zeroed. The gate is cheap, decisive, and *permanently* exonerates
  the wrapper in every later debugging session ("the wrapper was never the problem at
  any point"). A comparator that exits nonzero and aborts the chain
  (`experiments/exp07_check_match.py`) makes it enforceable in scripted runs.

- **Additive action residuals on a flow base can wash out exactly symmetrically.** The
  healthy residual run fixed 26 episodes and broke 26 — causally measured, since both
  sides were deterministic on identical spawns. The learned residual was
  state-INDEPENDENT (~0.008 units everywhere, identical on successes and failures):
  PPO learned *effort*, not *discrimination*, despite having the grasp bit and relative
  poses in its obs. Off-manifold nudges help marginal misses and break marginal
  successes. The on-manifold alternative is **x0-steering**: fixed x0 draws alone span
  14–56% success on the same frozen base (that spread *is* the available leverage), and
  the RFS literature reports 43% (additive) vs 86% (steering) on comparable setups.

- **Train reward can rise while success stays flat.** The flat residual run *beat* the
  base on training reward (+1700 vs +1644) with zero success change — dense shaping
  streams (e.g. a placement-hold reward) happily pay for earlier/longer partial
  progress. Never trust train reward; only held-out eval success counts.

- **Information can be PRESENT in the obs but UNUSED (salience failure).** The
  grasped-vs-closed-on-air distinction — the policy's dominant failure — is decodable
  from a *single frame* of its own obs: a 5-dim probe (finger pos/vel + last grip
  command) hit AUC 0.968 with **0% FPR** on 665 real on-policy freeze states, while the
  same probe given all 41 dims mislabeled 53.5% of them (distracted by salient,
  irrelevant features). So the fix is not history (refuted: <=+0.01 AUC, transfers
  worse) but *re-surfacing* the signal as an explicit validated bit. Crucial detail:
  the probe **needs the commanded-action channel** — physical finger joints alone score
  higher AUC (0.976) but 27.1% FPR on freeze states, because the disambiguating fact is
  "commanded closed AND resulting aperture", not aperture alone.

- **Offline behavior analyses must use each policy's own normalizer stats and
  outcome-filtered states.** A central "kill shot" finding of the original postmortem
  (DAgger data bleeding grip-opens into hold states) was **retracted** after forensics:
  the analysis had normalized policy B's obs with policy A's stats and had counted
  lift-frames of *missed* grasps (where opening is correct learned recovery) as "hold
  states". Corrected, the effect vanished entirely. Also measured: offline probes on
  expert states cannot rank policies at all — what separates good seeds from bad is
  closed-loop error compounding, visible only in rollouts.

- **Per-term reward channels from epoch 1 are the cheapest health tripwire.** The
  broken runs were diagnosable at epoch 1: `Episode_Reward/placed = 0.0` from the very
  first log meant the run was dead before learning, not "slow to learn". Wire per-term
  episodic reward logging into every RL run and *look at it immediately*. Caveat for
  chunk-window RL: episodic loggers only report when episodes **complete**, so with
  long horizons the first epochs are silently empty — know your logger's cadence before
  reading zeros as pathology.

- **Vectorize per-env controller state, and gate the rewrite on bit-exactness.**
  Per-env python containers (deques of actions) were the actual scaling limit long
  before VRAM: the vectorized tensor-queue controller (`(N, 15, 7)` buffer + index)
  runs 2048 envs at ~10-15k steps/s in ~7 GB VRAM. The rewrite was accepted only after
  reproducing the deque implementation bit-exactly — the same gate pattern as the
  wrappers, applied to an internal refactor.

Further lessons live in the docs and are worth the read: planner-valid is not
executable (cuRobo returns plans whose end poses the PD-controlled arm cannot track —
detection needs an *executed-state* check); same-seed A/B pairing silently breaks at the
first behavioral divergence when subsystems share an RNG stream; retry loops must
exclude failed candidates by identity, not list position; and aggregate success rate
hides real change (two policies matched to the decimal while 34 of 64 episodes flipped).

## Syncing from the source project

This repo is a curated snapshot of a live development tree. To re-sync after the source
evolves:

```bash
bash sync_from_source.sh   # copies curated files in (source path set at the top)
git diff                   # review what changed
git add -A && git commit   # commit yourself — the script never commits
```

`sync_from_source.sh` encodes the same include/exclude rules used to build this repo
(code + configs + docs in; runs/, HDF5 demos, checkpoints, logs, videos, and anything
over 5 MB out) and preserves the docs/ re-layout (top-level `.md` → `docs/`,
experiments `.md` → `docs/experiments/`). It never writes into the source tree.

## Provenance

Developed on a specific 2-can pick-and-place task in Isaac Lab 3.0 with an 8-DoF
(6 arm + 2 finger) arm; all success numbers, dims, thresholds, and reward magnitudes
quoted above are from that task and setup. The experiment docs under `docs/` are the
full, unedited lab-notebook history — including pre-registered beliefs that turned out
wrong and claims that were later retracted (kept, with dated correction blocks).
`act/modeling_act.py` and `act/configuration_act.py` are vendored from LeRobot
(Apache-2.0; exact commit and per-file modification list in `act/PROVENANCE.md`,
license text in `act/LICENSE`).
