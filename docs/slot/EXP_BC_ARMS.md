# Experiment S3 — does DART data help behaviour cloning on this task?

**Pre-registered 2026-08-03, before any training run.**

## The question I actually face

I spent the collection budget on two things: nominal demos and DART (noise-injected) demos.
The decision this experiment informs is whether the DART half was worth collecting — and, if
Stage C underperforms, whether to invest in HG-DAgger next or in something else entirely.

## The confound I must not walk into

The tempting comparison is "512 nominal" vs "512 nominal + 1024 DART". It is worthless: the
second arm has **three times the data**, so a win attributes equally to DART and to volume, and
volume is the boring explanation. eva_bc's own history is full of single-run comparisons that
turned out to be void; this one would be void by construction.

So the arms are **matched on demo count** and differ only in composition:

| arm | demos | composition |
|---|---|---|
| **A — nominal** | 1024 | 8 nominal pools (seeds 0–7) |
| **B — mixed** | 1024 | 4 nominal pools (seeds 0–3) + 2 DART-0.02 + 2 DART-0.05 |

Arm B shares half its demos with arm A, which is fine — the contrast is the other half.

Collecting the four extra nominal pools costs ~20 minutes of GPU. That is cheap against the
cost of an uninterpretable result.

## Protocol (inherited from eva_bc, non-negotiable)

* **≥ 3 training seeds per arm.** The same data and recipe spanned **32.8 %–59.4 %** across
  training seeds there; same-data different-seed pairs flipped 31–39 of 64 episodes. A
  single-run comparison is void.
* **Champion selected on a held-out spawn seed**, never on training loss (there is no val
  split by design).
* **Pooled ≥ 128-episode evaluation.**
* **Compare on `success_rate_later`**, not `success_rate`. The first episode a process executes
  scores ~13 points higher than every later one (128/128 vs 114/128 and 109/128 from an
  identical plan and bit-identical initial state, 4.3 σ). `eval_act.py` now reports the two
  cohorts separately; the headline `success_rate` moves with `--num_envs` and is not comparable
  across configurations.
* **Identical `--num_envs` and `--episodes` for every arm.** Same reason.
* Chunk 50 / execute 15, `num_inference_steps` 10. **Never shorten the execution horizon** —
  shortening it collapsed success 59.4 → 32.8 → 3.1 → 0 → 0 % at `n_action_steps`
  15/8/4/2/1. Chunk commitment is load-bearing.

## Beliefs, stated before running

1. **Arm B beats arm A on `success_rate_later`.** The nominal pool lies on a single
   deterministic manifold indexed by the block spawn — the expert plans once and executes
   open-loop — so a policy that drifts a millimetre off it has never seen anything nearby. The
   DART half is the only data in the pool that pairs an off-manifold state with a correct
   corrective command.
2. **The margin is larger on `-v0` than on `-Loose-v0`.** Loose has 3.0 mm of per-side
   clearance and forgives drift; v0 has 1.5 mm and does not.
3. **Both arms clear the Stage C bar** (≥ 55 % Loose, ≥ 40 % v0), because the expert's
   demonstrations are near-perfect and the observation is only 34-D with the block pose given
   directly rather than from pixels.
4. **Seed spread within an arm is large — comparable to the between-arm gap.** Pick-place saw
   32.8–59.4 % across training seeds on identical data. If that reproduces, three seeds is the
   *minimum* and the honest report is a distribution, not a number.

## Decision rule

* If B > A on pooled `success_rate_later` for **both** tasks and the gap exceeds the
  within-arm seed spread → DART data helps; collect more of it and proceed to Stage D on the
  arm-B champion.
* If the arms are within the seed spread of each other → DART made no measurable difference at
  this volume. Do **not** claim it helped. Proceed on the pooled best checkpoint and put the
  next effort into HG-DAgger, which supplies corrective data the expert genuinely cannot.
* If either arm misses the Stage C bar → the bottleneck is not data composition, and the
  diagnosis moves to the policy/controller (fixed-action attribution via
  `diag_training_env.py`, per-term channels from epoch 1).

## What gets recorded regardless

Per arm and seed: pooled success on `-v0` and `-Loose-v0`, split into first-episode and later
cohorts; mean episode length; flush count; and the failure taxonomy (never-grasped /
grasped-never-lifted / lifted-never-engaged / engaged-but-shallow / toppled / dropped /
inserted-then-lost). The taxonomy is what tells us *which* stage the policy fails at, which is
the input to Stage D.

## How to run

Pools (one rollout per process — see `HANDOFF.md` §1 first-episode bias):

```
cd /home/rei/Desktop/isaaclab/eva_bc/slot
source /home/rei/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6
FREE=reach,lift,back,spin,turn
for s in 0 1 2 3 4 5 6 7; do
  python scripts/collect_demos.py --task Rebot-PrecisionSlot-v0 --num_envs 128 \
    --rollouts 1 --seed $s --noise_std 0.0 --out data/v2/nominal_s$s.hdf5
done
for s in 10 11 12 13; do
  python scripts/collect_demos.py --task Rebot-PrecisionSlot-v0 --num_envs 128 \
    --rollouts 1 --seed $s --noise_std 0.02 --noise_phases $FREE --out data/v2/dart002_s$s.hdf5
done
for s in 20 21 22 23; do
  python scripts/collect_demos.py --task Rebot-PrecisionSlot-v0 --num_envs 128 \
    --rollouts 1 --seed $s --noise_std 0.05 --noise_phases $FREE --out data/v2/dart005_s$s.hdf5
done
```

Calibrate the loss censor against pools that contain real failures, then verify:

```
python scripts/calibrate_slip.py data/v2/*.hdf5
python scripts/calibrate_slip.py data/v2/*.hdf5 --apply --fpr 0.02   # only if it separates
python scripts/verify_demos.py data/v2/nominal_s0.hdf5
python scripts/check_port.py
```

Validate the non-simulator half of the path first (CPU, ~40 s, no GPU contention):

```
python scripts/test_pipeline_cpu.py --data data/v2/nominal_s0.hdf5 data/v2/dart005_s20.hdf5
```

Train — three seeds per arm, matched demo counts. **`--pool success` is mandatory** (see below):

```
A=$(ls data/v2/nominal_s{0,1,2,3,4,5,6,7}.hdf5)
B=$(ls data/v2/nominal_s{0,1,2,3}.hdf5 data/v2/dart002_s1{0,1}.hdf5 data/v2/dart005_s2{0,1}.hdf5)
for seed in 0 1 2; do
  python slot_act/train_flow.py --data $A --pool success --out runs/bc_armA_seed$seed --seed $seed --steps 100000
  python slot_act/train_flow.py --data $B --pool success --out runs/bc_armB_seed$seed --seed $seed --steps 100000
done
```

Evaluate — **identical `--num-envs` and `--episodes` for every arm**, and compare on
`success_rate_later`, never the headline `success_rate` (which carries the first-episode bias
and moves with `--num-envs`):

```
for arm in A B; do for seed in 0 1 2; do for task in Rebot-PrecisionSlot-v0 Rebot-PrecisionSlot-Loose-v0; do
  python slot_act/eval_act.py --ckpt runs/bc_arm${arm}_seed${seed}/ckpt_final.pt \
    --task $task --num-envs 32 --episodes 128 --seed 777 \
    --out runs/bc_arm${arm}_seed${seed}/eval_${task}.json
done; done; done
```

`--num-envs 32 --episodes 128` is deliberate: it makes only 32 of the 128 episodes
first-episode ones, so `success_rate_later` has n=96 to work with. `--num-envs 128
--episodes 128` would make *every* episode a first episode and the split would be useless.
`--seed 777` is the held-out spawn seed — it is not among the collection seeds (0–7, 10–13,
20–23), so the champion is never selected on spawns it was trained on.

### Three flag corrections made 2026-08-03, before the first training run

Written down because each would have produced a *number*, not an error:

1. **`--pool success`, not the default.** `train_flow.py`'s default is `default` = **no
   filter**, which trains on the ~5 % of DART episodes that ended with the block unseated —
   the exact thing the failure-labelling work was for. The obvious alternative, `--pool
   nominal`, is worse: it requires `episode_kind == "nominal"` and therefore drops **arm B's
   entire DART half**, turning a 1024-demo arm into a 512-demo one and re-creating the volume
   confound this experiment exists to avoid. `success_pool_filter` was added for this.
   Measured on `nominal_s0 + dart005_s20`: 255 demos → success 244, nominal 128, recovery 116.
2. **`--num-envs`, not `--num_envs`.** `eval_act.py` spells it with a dash; the underscore form
   is an argparse error, so this one at least fails loudly.
3. **`--episode-length-s` now defaults to 12.0, was 30.0.** The 30 s default was inherited from
   pick-place, whose expert demos ran ~1234 steps. Here `decimation=8` and `sim.dt=1/400` give
   a 20 ms control step, and every collected demo is **T = 599 = exactly 12.0 s**, which is
   also the task's own `episode_length_s`. Evaluating at 30 s would roll the policy 900 steps
   past anything it has ever seen, with `last_action` feeding back into the observation the
   whole way — and would have cost 2.5× the GPU time to do it.
