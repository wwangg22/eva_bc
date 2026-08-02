# EXP02 — Open-loop execution-horizon ablation (no retraining)

*2026-08-01. Status: **COMPLETE** 2026-08-02 (`experiments/exp02_run_evals.sh`).*

**Verdict: question (b) answered NO — decisively inverted.** Execute-15 does not cause
the precision failures; execution commitment is what keeps the policy alive. Success is
monotone in horizon: 59.4 / 32.8 / 3.1 / 0.0 / 0.0 % at n = 15 / 8 / 4 / 2 / 1. Per the
pre-registered rule, no n < 15 is adopted. The precision fix must come from phase-aware
cutting (PACE) or a per-step residual on committed chunks (ResiP/ResFiT) — both keep
long horizons during coherent motion. Optional follow-ups, deferred: n = 25/50 (is 15
even optimal from above?), PACE-style speed-profile cutting (stage 2 below).

## Question

Does executing 15 actions open-loop per chunk cause precision failures? The policy
re-observes the world only every `n_action_steps` env steps; between replans it is blind.
POSTMORTEM §5 measured 2–3 mm deciding catch-vs-air at the gripper — if aim error
accumulates within the blind window, replanning more often should recover some of it.

## Beliefs going in (pre-registered 2026-08-01, before any result)

1. **Moderate gain at n = 8 or 4 is the most likely good outcome** (order +3 to +8 pts):
   fresher replans during descend/close should convert some near-miss grasps.
2. **n = 1 could go either way**: fully closed-loop, but the policy was *trained* on chunk
   rollouts from expert data; executing 1 step of a fresh chunk every step can cause
   dithering/hesitation (chunk-to-chunk mode switching), a failure mode chunking exists to
   prevent. If n = 1 collapses, that is evidence the chunk commitment itself is load-bearing.
3. **The miss→freeze bucket should NOT shrink much**: a frozen policy replanning from the
   same aliased closed-on-air state gets the same wrong answer at any horizon. Horizon
   affects *getting into* the miss, not *getting out*. If never-lifted / stuck buckets DO
   shrink strongly at short horizon, that weakens the pure-aliasing story of POSTMORTEM §4a.
4. Delivery failures (lifted-to-carry-height but never landed) may improve with shorter
   horizon (fresher basket alignment) — this is where mm-scale placement precision lives.

## Method

- Frozen champion `runs/flow_nominal_v1/ckpt_final.pt`; eval-time `--n-action-steps`
  override only (verified supported, `act/eval_act.py:179,210`; chunk stays 50).
- n ∈ {15 (baseline, already on disk), 8, 4, 2, 1} × 64 episodes, seed 42, 16 envs,
  flush ON, 30 s episodes — bit-identical settings to the gate-2 baseline eval, so
  per-episode diffs are valid and the churn noise floor is 0.
- Outputs: `runs/flow_nominal_v1/eval_h{8,4,2,1}_64ep.json` (+ .log), each with full
  `per_episode` records (placed_final / placed_max / max_can_z / flushes).
- Analysis: success rate per n; failure taxonomy per n (same classifier as
  `experiments/taxonomy.py`); per-episode transition diff vs n = 15.

## Results

| n_action_steps | success | placed-1-stuck (never lifted 2nd / lifted 2nd) | never-lifted-anything | lifted-placed-nothing | drops |
|---|---|---|---|---|---|
| 15 (baseline) | 59.4% (38/64) | 11 / 5 | 8 | 2 | 0 |
| 8 | **32.8% (21/64)** | 5 / 6 | **21** | **11** | 0 |
| 4 | **3.1% (2/64)** | 9 / 1 | **35** | **15** | **2** |
| 2 | **0.0% (0/64)** | 3 / 2 | **50** | 9 | 0 |
| 1 | **0.0% (0/64)** | 0 / 0 | **61** | 3 | 0 |

*(complete. per-episode diffs via `taxonomy.py --diff`)*

## Analysis (interim, after n = 8)

**Belief 1 is WRONG in direction and magnitude**: halving the horizon cost −26.6 pts,
not a gain. Per-episode: 8 fixed, **25 broken** vs n = 15. The two exploded buckets:

- **never-lifted-anything 8 → 21**: more frequent replanning breaks grasp *commitment* —
  14 previously-clean episodes now fail before ever lifting. Chunk-boundary dithering
  during descend/close (the failure mode chunking exists to prevent — belief 2's
  mechanism, arriving already at n = 8).
- **lifted-never-placed 2 → 11**: coheres quantitatively with POSTMORTEM §5's measured
  grip lottery — v1 draws a spurious open somewhere in its predicted horizon on ~12% of
  nominal hold states; each replan is a fresh draw, and n = 8 nearly doubles the draws
  per carry. More replans → more mid-carry opens → more drops. Same mechanism, dose
  doubled.

Note 8 episodes DID get fixed (2 never-lifted → success, 3 placed-1-stuck-low →
success…), so fresh replans do convert *some* misses — the aggregate is just dominated
by the two damage modes. This is exactly the PACE argument (arXiv:2606.00537): the
horizon should be long during committed motion and short only at phase boundaries —
a fixed short horizon buys the benefit and pays 3× more in damage.

Validation to watch: if n = 4/2/1 continue monotonically down with the same two buckets
growing, the effect is real; an erratic pattern would suggest a controller/queue
harness artifact instead (queue mechanics are shared across n, so risk is low).

**Monotonicity CONFIRMED** (n = 4: 3.1%, n = 2: 0.0%). Never-lifted marches 8 → 21 →
35 → 50; n = 4 shows the first post-placement drops ever observed (2 episodes). Clean
dose-response — the effect is real. Two candidate mechanisms for the never-lifted
explosion, not separable from episode-aggregate data alone:
(a) **dithering** — each fresh chunk restarts approach, mode-switching between replans
prevents committing through descend→close;
(b) **quasi-static crawl** — a chunk's first few actions are near the current pose
(smooth start), so executing 1–4 steps per replan advances the arm so slowly that 30 s
expires before the first grasp (all episodes hit the 1500-step timeout in every
condition).
Either way the conclusion for the pipeline is identical: **chunk commitment is
load-bearing**; the answer to precision failures is NOT naive faster replanning — it is
phase-aware cutting (PACE-style, cut at low-speed boundaries only) and/or a per-step
residual on top of committed chunks (ResiP-style). This also *explains* the original
choice n = 15 working at all: the policy needs to execute deep enough into each chunk to
make real progress before re-observing.

## Stage 2 (conditional): phase-dependent execution

If the fixed sweep shows a precision-vs-commitment tradeoff (short horizon helps
grasp/placement but hurts transport or causes dithering), implement phase-dependent
horizons in `BatchedACTController` (long in free space, short near contact), motivated by
PACE (2026) / ResiP — literature notes to be added from the research agent's report.

## Decision rule (pre-registered)

Any n < 15 gaining ≥ +5 pts (≥ 3 episodes net, against a 0-churn floor) → adopt for all
subsequent work and re-run the champion's failure taxonomy at that n before EXP03 replica
evals (so replicas are measured at the adopted operating point). Mixed result → stage 2.
