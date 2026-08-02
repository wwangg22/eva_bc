# EXP01 — Can the current observation identify grasp success? (aliasing probe)

*2026-08-01. Status: IN PROGRESS (labeling + probe implementation).*

## Question

POSTMORTEM §4a claims the 59.4% plateau is partly **state aliasing**: at the moment the
gripper has just closed, "closed on can → hold and lift" and "closed on air → reopen and
retry" are nearly identical single-frame observations with opposite correct actions. Test
this directly: train a small classifier to predict *true grasp success* from the same
observations the policy sees. If a probe can separate the states, the information is
present and the policy's failure is one of *usage/architecture*; if it can't, the obs is
genuinely aliased and needs history or a tactile-proxy feature.

## Beliefs going in (pre-registered, before any result)

1. **Single-frame separability should be decent but imperfect** — the postmortem measured
   a real physical signature (closed-on-can stalls at ~0.020–0.040 finger aperture,
   closed-on-air reaches ~0.000, a ±12 mm joint difference), so the signal exists; but it
   is small, one-dimensional-ish, and normalization-era subtle. Guess: AUC ~0.85–0.95
   single-frame at close-time, rising with frames-after-close.
2. **History should help mainly *early*** (during the close itself, before apertures
   settle: velocity/stall profile is temporal), less at late hold states where aperture
   has stabilized.
3. **Aperture-only (pseudo-tactile) features should carry most of the signal** — if not,
   belief 1's mechanism story is wrong and we must find where the signal actually lives.
4. If single-frame separability is HIGH (≥0.95) at the states where the policy freezes,
   then aliasing is not literally "information absent" but "information unused by a
   smooth policy under conflicting supervision" — which still motivates an explicit
   grasp-success input bit (make the decisive feature salient) and/or reward-driven
   correction, but kills the "need history for disambiguation" version of the story.

## Data & labels (free from existing HDF5 — no GPU needed)

Sources: `expert/demos_nominal_s10{1..8}.h5` (504 expert eps incl. missed-grasp retries),
`expert/dagger_r1.h5` (100 takeovers: policy closed-on-air stretches + expert recovery),
`expert/dagger_r1_failures.h5` (46 failed takeovers).

Per-episode attrs give labels for free:
- `segments`: phase timeline (`approach/descend/close/lift/reopen/...`) with step bounds
  and per-grasp segment ids.
- `outcomes`: per grasp-segment `"grasped"` (label 1) vs `"missed"` (label 0), with
  `close_disp`.
- DAgger episodes: `takeover_t` + `gate_reason: "miss"` → the policy-phase closed-on-air
  frozen stretch (last ~45 steps before takeover) is an *on-policy* label-0 sample.

Sampling: for each expert grasp attempt, frames from close-end through the following
lift/reopen segment start + k, k ∈ {0, 2, 5, 10} frames after the close settles. Split
**by episode** (train/test 80/20), never by frame.

**Known asymmetry to control:** on-policy (DAgger) data contains only label-0 states
(collector saved only takeover rollouts). So: train the probe on **expert-only** frames
(both labels exist there), and report on-policy label-0 states as a held-out transfer
set. If the expert-trained probe fails on policy miss states, that is itself a finding
(policy miss states differ from expert miss states — distribution shift inside the probe).

## Probe variants

| variant | input | tests |
|---|---|---|
| A | single frame, full 41-D | is the info present at all? |
| B2 / B4 | 2 / 4 stacked frames (t, t−1[, t−2, t−3]) | does history add? |
| C | single frame, finger dims zeroed (pos 6–7, vel 14–15, last-grip 40) | is the signal ONLY in fingers? |
| D | fingers only (dims 6, 7, 14, 15, 40) | pseudo-tactile bit feasibility |
| E | D + 4-frame history | temporal stall signature |

Model: MLP 2×128 ReLU, Adam 1e-3, early stop on val; CPU (minutes). Metric: ROC-AUC +
accuracy at 0.5, reported per k (frames after close) and per variant; plus transfer AUC
on on-policy miss states (expected miss-rate given all label-0: report FPR = fraction
called "grasped").

## Results (2026-08-01, `exp01_probe.py` → `exp01_results.json`)

Samples: 4,368 expert post-close frames (3,756 grasped / 612 missed; 1,092 per k),
665 on-policy closed-on-air frames (all label 0, DAgger miss-gated stretches).
Episode-level 80/20 split; probe = MLP 2×128, class-weighted BCE, CPU.

| variant | input | test AUC (expert) | on-policy FPR@0.5 (665 known-miss states) |
|---|---|---|---|
| A | single frame, full 41-D | 0.954 | **53.5%** |
| B2 | 2-frame history | 0.951 | 57.7% |
| B4 | 4-frame history (stride 1) | 0.963 | 64.2% |
| B4s5 | 4-frame history (stride 5) | 0.967 | — |
| C | single frame, finger dims removed | 0.927 | 79.8% |
| D | finger dims only (pos 6,7 + vel 14,15 + last-grip 40) | 0.968 | **0.0%** (mean p ≈ 4e-12) |
| E | finger dims, 4-frame history | 0.915 | 40.8% |
| F | can/basket config only (dims 16–34) — confound check | 0.740 | 82.7% |
| G | physical finger joints only (6,7,14,15) | **0.976** | 27.1% |

Per-k detail in the JSON; history helps mainly at k = 0 (close-end: A 0.935 → B4s5
0.973), consistent with a temporal stall signature that a settled aperture later makes
redundant.

## Analysis

1. **The information is present, single-frame** (A = 0.954; G = 0.976). The aliasing of
   POSTMORTEM §4a is NOT "information absent" — belief 4's scenario is what's true.
2. **The decisive signal lives in the finger channels** (belief 3 confirmed): 4 physical
   finger dims alone beat the full 41-D obs (0.976 vs 0.954).
3. **Validated confound (why C looked good):** a probe seeing ONLY can/basket config —
   nothing about the robot — still reaches 0.74 on expert data, because expert misses
   correlate with hard can configurations (lying cans etc.). C's 0.927 is inflated by
   this task-difficulty shortcut. The on-policy transfer column exposes it: C and F call
   ~80% of real policy miss states "grasped".
4. **The salience mechanism, demonstrated end-to-end:** variant A *has* the finger dims
   available yet mislabels 53.5% of the policy's actual frozen closed-on-air states —
   exactly the distracted-by-41-dims failure we attribute to the policy itself. Variant D
   (5 dims) rejects **all 665** with near-certainty. Same data, same probe capacity,
   same training — only feature salience differs. This is the strongest evidence yet
   that the policy's miss→freeze is a *feature-usage* failure, not missing information.
5. **History is NOT the fix** (belief 2 mostly wrong): +0.01 AUC at best, and
   finger-history (E) *hurts* transfer (40.8% FPR) — expert close dynamics don't match
   frozen-policy dynamics, so temporal patterns transfer worse than static aperture.
6. Honest caveats: on-policy set contains only negatives (FPR is the only computable
   transfer metric, at an uncalibrated 0.5 threshold); D's e-12 confidence is logistic
   extrapolation on extreme standardized apertures, not a calibrated probability; expert
   misses ≠ policy misses in the non-finger dims (that gap is precisely why A transfers
   poorly).

## Verdict (per pre-registered decision rule)

- Single-frame AUC ≥ 0.95 → **grasp-success input bit is justified and cheap**, and D/G
  show it can be computed from finger channels alone — the pseudo-tactile idea
  (arXiv 2503.23835) validated on our own data.
  **Do not conflate D and G** (a downstream doc briefly did): the 0%-FPR transfer
  result belongs to D (fingers + last-grip, AUC 0.968); G (physical joints only) has
  the higher AUC 0.976 but 27.1% FPR on the on-policy freeze states. Any deployed
  grasp bit must include the last commanded grip alongside finger pos/vel.
- History ≫ single-frame is FALSE → do **not** add history conditioning as the primary
  fix.
- Concrete recommendation for the pipeline: add a 1-D grasp-success feature (finger
  aperture + stall logic, or the D-probe itself distilled to a threshold rule) to the obs
  for any future BC training, and include it in the residual policy's inputs (EXP06) —
  it is exactly the disambiguating bit the residual needs at the post-close decision.

## Decision rule (pre-registered)

See EXP_INDEX.md: A ≥ 0.95 → grasp-success bit (obs surgery) is justified and cheap;
B ≫ A → history conditioning; D ≈ A → the bit can be computed from finger channels alone
(pseudo-tactile per arXiv 2503.23835).
