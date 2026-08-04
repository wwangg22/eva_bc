# 12 — Upstream sync: what arrived, what it changes, what it does not

**2026-08-03.** Both repos pulled at Big Will's instruction and reviewed commit by commit.
This document is the review. It is separate from the plans it amends so that the amendments
have a dated, auditable source.

Headline: **eva_bc's x0-steering experiment (EXP07) closed successfully — 55.5 % → 91.4 %
pooled.** That retracts the central caveat of `10_STAGE4_STEERING_PLAN.md` §0 and refutes one
of my own registered worries. It does not change any clutter number, and it does not make
Stage 4 safe. §5 below is the part that matters most: what the result does *not* transfer.

---

## 1. WHAT WAS PULLED

| repo | before | after | commits |
|---|---|---|---|
| `eva_bc` | `1c04eca` | `818391b` | 4 |
| `eva_rl` | `05f0fb3` | `e56e7df` | 1 |

Both were fast-forwards. Neither touched anything under `eva_bc/clutter/`, and
`git status --porcelain` in `eva_bc` still reports exactly `?? clutter/` — the working tree is
unchanged apart from the incoming files. No conflict, nothing of mine overwritten.

```
eva_bc
  4471be9  EXP07 verdict: x0-steering 55.5% -> 91.4% pooled (Gate 6 cleared); add s1 taxonomy analyzer
  fc21c45  Retire sync_from_source.sh: reBot_ACT is now the eva_bc working clone (cutover complete)
  6ee2030  Post-EXP07 write-ups: POSTMORTEM section 9, JOURNAL arc entries, HANDOFF final state; eval_steer video support
  818391b  Log second close-up video batch (3/3 success, seed 123)

eva_rl
  e56e7df  Wrist camera: tilt D405 mount -30 deg (local X) to center the gripper; add tilt-sweep review script
```

---

## 2. eva_rl `e56e7df` — NO EFFECT ON CLUTTER (checked, not assumed)

Two files:

- `source/reBot_RL/reBot_RL/tasks/manager_based/lift/camera_cfg.py` — `WRIST_CAM_CFG.offset.rot`
  changes from `(0.5, −0.5, −0.5, 0.5)` to `(0.353553, −0.353553, −0.612372, 0.612372)`, i.e.
  the old rotation composed with `R_x(−30°)`. The commit message reports the optical axis now
  lands 1.8° off the TCP at 0.171 m versus 31.8° before.
- `scripts/wrist_cam_tilt_sweep.py` — new, a render harness for repeating the sweep.

**Why it cannot touch us:**

```
grep -rn "camera_cfg\|CameraCfg\|TiledCamera" --include=*.py .../manager_based/challenge/
   -> no matches
```

The `challenge/` package spawns no cameras at all. `Rebot-ClutterExtract-*` is a pure
state-based task — the 41-D observation is joint state, TCP pose and object poses, with no
image channel anywhere in it. The edit is confined to the `lift` (pick-and-place) task tree.

**Recorded consequence:** none. It is logged here so that a future session does not have to
re-derive that a camera commit was irrelevant. It does confirm one thing worth knowing: **the
`challenge/` package remains untouched by upstream** — its last change is still `05f0fb3`, so
every clutter measurement taken so far is still against the current benchmark.

---

## 3. eva_bc — EXP07 x0-STEERING CLOSED, SUCCESSFUL

### 3.1 The result

| quantity | value |
|---|---|
| like-for-like base (fixed x0 = zeros, deterministic) | **55.5 %** pooled (56.2 @42 / 54.7 @123) |
| stochastic base (fresh x0 each refill) | 64.1 % pooled |
| **steered** (z = clamp(mu), x0 = tanh z) | **91.4 %** pooled (89.1 @42 / 93.8 @123) |
| margin over the like-for-like base | **+35.9 pts** |
| Gate 6 bar | 90 % — **cleared** |

128 held-out episodes, two ladder spawn suites, deterministic on both sides, paired on
identical spawns. One RL seed. First pre-registered configuration — no PPO tuning, no
escalation to a richer parameterisation.

### 3.2 The discrimination signature — the thing I actually care about

This is the part that carries information for clutter, more than the headline does.

| | EXP06 additive residual | EXP07 x0-steering |
|---|---|---|
| pooled result | 55.5 → **55.5** (exactly flat) | 55.5 → **91.4** |
| paired episode churn | **26 fixed / 26 broken** | **51 fixed / 5 broken** |
| learned magnitude | `~0.0084` everywhere | mean \|z\| **0.220 success / 0.282 failure** |
| within-episode variability | — | z_std **0.21 success / 0.26 failure** |
| grasp-phase bucket | untouched | `never_lifted` **18 of 19 → success** |
| gripper failures | structurally unreachable | `lifted_never_placed` 10/11 fixed |
| new failure modes invented | — | **none** |

The additive residual was *state-independent*: PPO learned an effort level, not a policy. The
steering z is *state-dependent* in the predicted direction — larger and more variable exactly
on the episodes the base fails. That is the signature `10_STAGE4_STEERING_PLAN.md` §6 Gate 4c
was written to look for, and upstream has now demonstrated it is a signature a real success
actually produces, rather than a criterion I invented that nothing would satisfy.

### 3.3 The mechanism, as upstream states it (POSTMORTEM §9)

The distilled claim: **the base's failures were mode errors, not aim errors.**

- The frozen-x0 sweep spans **14.1 %–56.2 %** success across draws on identical weights. Which
  chunk family the decoder commits to dominates the outcome.
- Blind *resampling* of the mode is worth +8.6 pts on its own (55.5 deterministic → 64.1
  stochastic). A stuck mode is expensive even when you re-roll it at random.
- A per-step additive offset translates a committed trajectory. It cannot re-select the plan,
  so it repairs a marginal miss exactly as often as it breaks a marginal success — hence the
  symmetric 26/26 — and PPO, seeing symmetric reward, correctly converges to doing nothing.
- x0 re-selects the plan, and every selection is on-manifold because the base's own decoder
  produces it. `z ≈ 0` — the best blind mode — stays available per state, so PPO departs from
  it only where the expected gain is positive. That asymmetry is what 51/5 looks like.
- 91.4 % > 64.1 % is the sharpest statement of all: **chosen modes beat random modes.**

### 3.4 Design constants I can now inherit instead of guess

Everything below was measured upstream, not assumed, and each is directly reusable as a
*prior* for clutter (not as a substitute for our own gate — see §5).

| constant | upstream value | how it was chosen |
|---|---|---|
| `alpha_x0` | 1.0 | ≈1σ of the noise the base trained under |
| z dimensionality | 7 (6 arm + 1 grip), **broadcast rank-1** | belief 5 predicted this was too coarse; **refuted** |
| `sigma_init` | −1.2 (σ ≈ 0.30) | from gate S0, not precedent |
| σ cost curve | σ0.3 ≈ −2 pts, σ0.6 ≈ **−20 pts** | measured on 64 eps each |
| mu-bias `U(±0.125)` | +4.7 pts, i.e. harmless | measured, not waved through |
| `clip_actions` | **1.0, mandatory** | 100 destroyed two EXP06 runs |
| reward | bare placement stream, **no** magnitude penalty | z is bounded by construction |
| horizon | 24 windows | — |
| budget | 2048 envs × 200 epochs ≈ 9.8 M windows ≈ 147 M env steps ≈ 3.7 h | ~740 windows/s |
| training protocol | **= eval protocol** | drop-termination and its penalty are inseparable (`is_terminated_term`) |

### 3.5 The window-RL logging rule — a tripwire I would otherwise have tripped

An episode is 100 windows; an epoch is 24. **No episode has completed before epoch ~5**, so
`Episode_Reward/placed` reads exactly `0.0` for epochs 1–4 — which is also what a genuinely
collapsed run looks like. Upstream burned real confusion on this and now judges health at the
first epoch *after* episode completion.

**For clutter the arithmetic is different and I should compute it rather than copy it.** An
episode is `690 / 15 = 46 windows`. At horizon 24 the first completion lands at epoch
`ceil(46/24) = 2`. So our silent window is 1 epoch, not 4 — but it is still non-zero, and
the rule ("judge health at the first epoch after `ceil(episode_windows / horizon)`") is what
transfers, not the number 5.

### 3.6 Infrastructure changes

- `act/eval_steer.py` gained `--video / --video-length / --video-folder / --viewer-eye /
  --viewer-lookat`, ported from `eval_act.py`. Sets `enable_cameras` when `--video`, wraps in
  `gym.wrappers.RecordVideo` with `step_trigger=lambda s: s == 0`. Directly reusable for
  clutter close-ups later; nothing to change.
- `experiments/exp07_analyze_s1.py` — the paired taxonomy analyser. This is the shape our own
  Gate 3 analysis should take.
- `sync_from_source.sh` **retired** (`fc21c45`). The consequence for us is in §6.

---

## 4. WHAT THIS RETRACTS IN MY OWN DOCUMENTS

### R13 — `10_STAGE4_STEERING_PLAN.md` §0 "THE INHERITED RISK"

**Retracted.** The table said Gate 3 was `NEVER REACHED` and concluded:

> "The method eva_bc's own README recommends has never produced a success number on any task."

That was true when written and is now **false**. Gate 3 passed at 91.4 %. §0 has been
rewritten; the old text is preserved inside it as a struck block so the record of what I
believed and when survives.

The hedges in §8 are **not** withdrawn. They were justified by "the method is unproven," and
that justification is gone — but §5 below supplies a different and, I think, better one, so
they stay. What changes is their *cost basis*: H1 (a 32-draw `fixed_x0` search before any PPO)
is now cheap insurance rather than the primary plan.

### R14 — `10_STAGE4_STEERING_PLAN.md` §3 item 2, "the rank-1 x0 is off-distribution"

**Downgraded from a live risk to a measured non-problem — upstream.** I wrote:

> "A rank-1 x0 has roughly the right norm and *completely* the wrong structure. The flow model
> has never seen one. … **Untested in eva_bc.**"

The first two sentences remain literally true; training draws `randn_like(x1)`, iid across all
50 chunk positions, and `set_steer` broadcasts one 7-vector across all of them. What is no
longer true is "untested." Upstream registered the identical worry as belief 5, ran it, and
recorded **REFUTED — 7-D constant-per-chunk z sufficed; escalation never triggered.**

The honest reading: a rank-1 x0 is off the training distribution and the decoder handles it
anyway. That is unsurprising in hindsight — the model is trained to map a *neighbourhood* of
noise to coherent chunks and rank-1 draws sit inside the support of the marginal per-position
distribution even though they are wildly atypical jointly.

**I am keeping Gate S0a.** One task's evidence that a structural assumption survives is not
proof it survives on another with a different action-space geometry, and S0a is ~20 minutes.
But its registered prediction changes from "may produce incoherent chunks" to "expect
coherence; the open question is authority, not validity."

### R15 — the framing of Gate 4c

Unchanged as a test, sharpened as a criterion. §6 said `R² < 0.05` of z on state refutes the
mechanism. Upstream now gives a *positive* reference for what a working steering policy looks
like: mean |z| separating 0.220 / 0.282 between success and failure, z_std 0.21 / 0.26. Those
are modest separations — roughly 25 % and 20 % — which is a useful calibration: **do not
demand a dramatic split.** A clutter run showing 0.0084-everywhere is refuted; one showing a
20 % separation is behaving exactly like the upstream success.

---

## 5. WHAT DOES **NOT** TRANSFER — read this before quoting 91.4 % anywhere

The temptation after a +35.9-pt upstream result is to treat Stage 4 as de-risked. It is not,
and the differences are structural rather than incidental.

1. **Different failure class.** EXP07 repaired *grasp-family selection* — 18 of 19
   `never_lifted` episodes, i.e. "committed to a bad grasp, never retried." Steering worked
   because re-choosing z each window **is** retry logic. Clutter's residual failure is a
   **topple during the close**, and a topple is **terminal**: `distractor_toppled` ends the
   episode. There is nothing to retry. Steering must get it right the first time, in the one
   window that straddles the close. That is a strictly harder credit-assignment problem than
   the one upstream solved, and no part of the 91.4 % speaks to it. (`10_STAGE4` §3 items 3
   and 4 already said this; upstream's success does not touch either.)

2. **Different margin.** Upstream's lever was "which of several competent grasp approaches."
   Ours is a **7.8 mm** clearance between a 19.2 mm half-blade and a neighbour face at 27 mm.
   Whether x0 can move the committed chunk by millimetres *in a controllable direction* is
   exactly Gate S0a's question and remains unmeasured on our base — which does not yet exist.

3. **Different base quality.** The upstream base sat at 55.5 % deterministic with a large
   multimodal spread to exploit (frozen draws spanning 14.1–56.2 %). **We have no base at
   all** — Stage 2 has not produced a checkpoint. If our BC policy comes back tight and
   unimodal, there is less mode structure for steering to select among, and the entire
   mechanism has less to work with. The x0-draw spread on *our* base is the single most
   informative number available before any PPO, which is precisely why H1 survives R13.

4. ~~**Different scale.**~~ **RESOLVED same day — Q7 ran, and this concern is withdrawn.**
   The original text worried that a 10 GiB card might only hold 512 envs and cost ~15 h to
   match upstream's 9.8 M window-transitions. Measured: **4 096 envs fit in 3.2 GiB**, and
   2 048 envs run at **34 458 env-steps/s — about 3× upstream's ~11 k on a 12 GB card**,
   because the clutter scene is five blocks and an arm. The same transition budget is ~1.2 h
   of simulator time. **Memory is not the constraint and neither is throughput.** Full table
   in `11_STAGE2_RESULTS.md` §5.

5. **One seed, nominal spawns.** Upstream's own §9e says so: the perturbed/robustness
   composite was never run on the steered stack, the steering head is specific to that one
   base checkpoint by construction, and the 11 remaining failures are unstudied beyond
   bucketing.

**Net:** the upstream result converts x0-steering from *unproven* to *proven on a different
failure class*. That is a genuine and large update — it moves the prior on the mechanism a
long way — and it is not a result about clutter.

---

## 6. REPO CONSEQUENCE: `clutter/runs/` IS GITIGNORED — 72 FILES OF EVIDENCE

Found while checking that the pull had not disturbed anything. It needs deciding before the
push Big Will has reserved for himself.

`eva_bc/.gitignore` contains, among others:

```
runs/
*.log
*.h5
*.pt
```

A bare `runs/` with no leading slash matches **at every depth**, so it captures
`clutter/runs/` as well as the repo-root `runs/`. Measured:

```
git add -An clutter/            ->  48 files would be added
git ls-files -oi --exclude-standard clutter/
        clutter/runs             72 files
        clutter/probes/__pycache__  6
        clutter/expert/__pycache__  1
```

So a push today would carry the 48 files of docs, probes and expert code and **silently drop
every result JSON and every run log** — 6.1 MB, and the entire evidentiary basis for every
number in `07_`, `08_` and `11_STAGE*_RESULTS.md`. Documents that cite
`runs/p26_screen_v3.json` would point at nothing.

That is upstream's rule and it is the right rule *for upstream*, where `runs/` holds
multi-hundred-MB checkpoints. Ours holds 6.1 MB of JSON and text:

```
33 files  *.json    908 K     the machine-readable evidence
40 files  *.log     5.3 M     raw stdout (per-candidate screen prints, CEM traces)
```

**Not fixed unilaterally** — `.gitignore` is a tracked file outside `clutter/`, and the
standing constraint is that I do not touch those without saying so.

I did not guess at the options either. I built a throwaway repo reproducing the exact rules
and measured what each does (`scratchpad/gitest`):

| | change | `runs/*.json` | `runs/*.log` | edits outside `clutter/` |
|---|---|---|---|---|
| **A** | `clutter/.gitignore` with `!*.json`, `!*.log` | ✗ still ignored | ✗ still ignored | none |
| **B** | rename `clutter/runs/` → `clutter/results/` | ✓ | ✗ (root `*.log` still bites) | none |
| **C (recommended)** | root `.gitignore` gains `!clutter/runs/` | ✓ | ✗ | 1 line |
| **C+** | C, plus `clutter/.gitignore` with `!*.log` | ✓ | ✓ | 1 line |
| **D** | do nothing | ✗ | ✗ | none |

**A fails outright**, and it fails for the reason git documents: *"it is not possible to
re-include a file if a parent directory of that file is excluded."* `runs/` excludes the
directory, so a nested negation never gets the chance to run. I had written A down as my
recommendation on the assumption it would work; the test refuted it before this document was
finished. Worth stating, because the same assumption would have produced a push that silently
dropped the evidence and looked like it had worked.

Note also that `*.log` is a *file* pattern, so it survives the directory re-include — C alone
gets the JSONs and still loses the logs. Only C+ gets everything.

I lean **C+**: two small changes, reversible, keeps every path in every document honest, and
the logs are where the per-candidate screening prints live — the raw material behind the P26
and P30 tables. If 5.3 MB of stdout in a shared repo is unwelcome, **C** alone still preserves
every number any document cites.

Exact commands, for Big Will:

```
printf '\n# clutter/: 6.1 MB of JSON+log evidence, no checkpoints -- keep it tracked\n!clutter/runs/\n' >> /home/eva/Desktop/isaacLab/eva_bc/.gitignore
printf '!*.log\n' > /home/eva/Desktop/isaacLab/eva_bc/clutter/.gitignore
```

---

## 7. ACTIONS TAKEN

1. Both repos pulled, fast-forward, working tree verified unchanged.
2. eva_rl's camera commit checked against `challenge/` and cleared as a no-op, with the grep
   recorded above rather than asserted.
3. `10_STAGE4_STEERING_PLAN.md` §0 rewritten (R13); §3 item 2 amended (R14); Gate S0a's
   registered prediction restated; Gate 4c given upstream's positive reference (R15).
4. This document written.
5. The `.gitignore` finding raised for Big Will's decision, unfixed.

**Not done, deliberately:** nothing in `clutter/` has been re-planned around the 91.4 %. §5 is
the reason. Stage 2 still has to produce a base before Stage 4 has anything to steer, and the
queue in `HANDOFF.md` §10.0 is unchanged apart from Q7's promotion.
