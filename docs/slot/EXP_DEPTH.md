# EXP_DEPTH — the binding failure is depth. Probe A: is the episode horizon simply too short?

*Opened 2026-08-03 09:20, **before any cell has been run**. Follows `HANDOFF.md` §7 step 5z-1b,
which named this as the one-command probe worth running first, on the grounds that it is the
cheapest hypothesis that could explain the single largest failure bucket.*

## Status

**Probe A round 1: ✅ COMPLETE** (§7). Pre-registered 09:20, run 10:36–10:51.
**Probe A round 2 (replication): ✅ COMPLETE** (§8). Run 11:44–12:00.

> ### ⚠ §7's headline did not replicate. Read §8 before quoting anything from §7.
>
> Round 1 reported a **differential** — +11.5 points for the stalled-in-mouth checkpoint, −2.1 for
> the never-entered one — and built a "slow versus stuck" mechanism on top of it. At the second
> spawn seed the two gains are **+0.0 and +10.4**: the differential reverses. Pooled over both
> checkpoints and both seeds (n = 384 per arm), the horizon is worth **+4.9 points, p = 0.138**,
> with no dependence on failure shape, and it **saturates by 20 s** (30 s gives 0.729 vs 20 s's
> 0.740).

**The supportable answer, and it is a negative one:** the 600-step horizon is *not* the
explanation for the depth bottleneck. The depth failures are mostly **stuck, not slow**. That
makes them a legitimate target for Stage D steering rather than something a config change would
have fixed — the opposite of §7's conclusion, and the more useful one.

**One durable finding survives from round 1** and does not depend on the horizon at all: 3.0 % of
all later-cohort sweep episodes reach a fully seated block and do not end seated — a third
failure mechanism nothing in this project had counted, concentrated almost entirely in one
checkpoint (§7d).

---

## 1. The observation this is trying to explain

Across the 36-cell Stage C sweep, `never_entered` + `stalled_in_mouth` account for **82–84 % of
all failures at every clearance**. Tightening the channel 6× (3.0 mm → 0.5 mm per side) moves the
headline by ~5 points. Whatever is stopping this policy, it is not the width of the hole.

## 2. The split nobody had looked at yet

Median failure depth, per sweep cell, is **bimodal — and it clusters by checkpoint, not by
clearance**:

| checkpoint | median failure depth (mm), Tight s777 | shape |
|---|---|---|
| `bc_armB_seed0` | +34.2 | entered, stopped ~6 mm short of the 40 mm bar |
| `bc_armB_seed2` | +29.4 | entered, stopped ~11 mm short |
| `bc_armA_seed0` | +26.0 | entered, stopped short |
| `bc_armA_seed2` | −16.5 | never crossed the mouth |
| `bc_armA_seed1` | −33.9 | never crossed the mouth |
| `bc_armB_seed1` | −41.5 | never crossed the mouth |

Three training seeds land firmly on each side, from **identical data**. That is the same
15–29-point seed variance seen everywhere else on this project, but here it shows up as a
*qualitatively different failure mode*, not just a different rate — which is a stronger statement
than "seeds vary" and one I had not made before.

"The policy ran out of time" can only explain the **positive** shape. A block that never crossed
the mouth at 600 steps was not 100 steps away from crossing it. So the horizon hypothesis makes a
**differential** prediction, and that is what makes it worth one command instead of a shrug.

**→ §6 rules out the obvious rival explanation** (that the stalled blocks are wedged by lateral
error) on existing data, before spending any GPU time here. Read it next; it is why this probe is
worth running at all.

## 3. Design

Two subjects, one per shape, chosen for headroom (both 0.62–0.72, so an effect has room to show):

| run | shape | 12 s baseline (already on disk) |
|---|---|---|
| `bc_armB_seed2` | stalled-in-mouth, +29.4 mm | 60/96 = **0.625** |
| `bc_armB_seed1` | never-entered, −41.5 mm | 69/96 = **0.719** |

Both on `-Tight-v0`, spawn seed 777, 128 episodes / 32 envs — identical to the sweep cells that
supply the baselines, which is what makes the comparison legitimate. Only the 20 s cells are new
(`--episode-length-s 20`, i.e. 1000 steps instead of 600). `scripts/run_horizon.sh`.

Note the baselines come from `Tight`, the *hardest* rung, deliberately: it is where the failure
count is highest and therefore where a horizon effect would be most visible.

## 4. Beliefs, pre-registered

1. **`bc_armB_seed2` (stalled shape) gains more than 10 points at 20 s.** Mechanism: its failures
   sit at +29 mm against a 40 mm bar, i.e. ~73 % of the way in, and the insertion phase is the
   slowest part of the trajectory. If the push is merely slow rather than stuck, 400 extra steps
   is a lot of extra push.
2. **`bc_armB_seed1` (never-entered shape) gains less than 5 points.** Its failures are 41 mm
   *outside* the mouth; more time spent doing the wrong thing does not help.
3. **The gap between the two gains is the result**, not either gain alone. A shared gain would
   mean I have misread both shapes; a shared null would mean the horizon is simply not binding
   and the depth failures are genuinely stuck.
4. **`placed_max` > `placed_final` appears at 20 s.** If extra time lets a policy seat the block
   and then *unseat* it, that is a distinct and somewhat embarrassing failure mode — and the
   records already carry both fields, so it costs nothing to check. I put this at maybe 20 %.

**If belief 1 holds, the headline number is horizon-limited, not capability-limited**, and the
honest way to report Stage C changes: the same policy is worth more points than the protocol
credits it with. If belief 1 fails, the depth failures are genuinely stuck and Stage D steering
has a real target rather than a speculative one.

## 5. Decision rule

Both comparisons are against sweep cells run at a different `episode_length_s`, so the reset
stream is identical up to the point the horizons diverge but the *number* of episodes per env is
not — 20 s episodes mean fewer resets in the same wall-clock, and pairing must be **verified from
`spawn_pos`, not assumed**. `analysis/robustness_report.py` already does exactly that check and
falls back to a two-proportion z-test when the spawns do not match; the horizon cells are read
with the same instrument.

Single-seed caution does **not** apply in the usual way here, because the design is explicitly
two seeds chosen for opposite shapes — but it does mean neither individual gain is worth quoting
without the other.

---

## 6. Addendum, written before the probe ran: does lateral misalignment explain the stall?

*Analysis of existing sweep data (all 36 `eval_ckpt_final_*.json`), added 09:47. **Not probe
results** — `scripts/run_horizon.sh` had not started. Belongs logically after §2; it is down here
because §3–§5 were already written and I would rather append than silently renumber a
pre-registration. `analysis/lateral_by_bucket.py`.*

There is an obvious rival explanation for `stalled_in_mouth` that costs nothing to test: the
block wedges because its lateral error has eaten the clearance. If that were true the horizon
probe would be pointless, because more time does not un-wedge a jammed block.

Pooled over all six runs × both spawn seeds, |lateral| in mm by outcome:

| clearance | cohort | n | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|---|
| 3.0 | success | 908 | 0.31 | 0.66 | 1.17 | 1.83 |
| 3.0 | **stalled_in_mouth** | 102 | 0.37 | **1.09** | 1.97 | 2.45 |
| 1.5 | success | 903 | 0.29 | 0.65 | 1.11 | 1.40 |
| 1.5 | **stalled_in_mouth** | 109 | 0.37 | **0.93** | 1.33 | 1.46 |
| 0.5 | success | 848 | 0.18 | 0.34 | 0.46 | 0.49 |
| 0.5 | **stalled_in_mouth** | 116 | 0.17 | **0.37** | 0.44 | 0.47 |
| 0.5 | never_entered | 136 | 0.28 | 0.74 | 2.42 | **7.32** |

**On `-Tight-v0` the stalled failures and the successes have the same lateral distribution to two
decimal places.** 0.37 vs 0.34 at the median, 0.47 vs 0.49 at p90. Whatever stops those 116
episodes 40 mm short, it is not where they are pointing. On the looser rungs there *is* an
elevation (1.65× and 1.43× the success median) but both rows are censored by the channel and the
absolute gap is under half a millimetre.

So the three failure buckets are three mechanisms, and only one of them is about precision:

| bucket | lateral vs success | reading |
|---|---|---|
| `stalled_in_mouth` | same (Tight) to 1.6× (Loose) | **push / depth** — aligned, stopped short |
| `never_entered` | p90 4.4–7.3 mm, far beyond any clearance | **aiming** — missed the mouth |
| `gross_miss` | 40–130 mm | **transport** — never got there |

This is worth recording for a second reason. The champion alone tells the opposite story:
`bc_armB_seed0` on `-v0` has stalled failures at |lateral| 1.15 mm against successes at 0.45 mm,
which reads as a textbook jam. That is **n = 2**. Pooling was the difference between "the depth
failures are jams, fix the aim" and "the depth failures are pushes, and one of them may just be
out of time" — and the second is what the probe is for.

---

## 7. Probe A round 1 — ⚠ SUPERSEDED BY §8; kept as the record of a claim that did not replicate

*Both cells run 10:36–10:51. `runs/bc_armB_seed{1,2}/horizon_20s_Tight_s777.json`.*

| subject | shape | 12 s (sweep cell) | 20 s | gain | unpaired z-test |
|---|---|---|---|---|---|
| `bc_armB_seed2` | stalled-in-mouth (+29.4 mm) | 60/96 = 0.625 | **71/96 = 0.740** | **+11.5 pts** | z = +1.71, **p = 0.088** |
| `bc_armB_seed1` | never-entered (−41.5 mm) | 69/96 = 0.719 | 67/96 = 0.698 | −2.1 pts | z = −0.32, p = 0.75 |

### 7a. Scoreboard

| belief | outcome |
|---|---|
| 1. stalled shape gains > 10 pts | ✅ **+11.5** — but see the p-value below |
| 2. never-entered shape gains < 5 pts | ✅ −2.1, a clean null |
| 3. the *gap* is the result | ✅ **13.6 points**, and it is the part with mechanism evidence behind it |
| 4. `placed_max` > `placed_final` appears at 20 s | ❌ it was **already there at 12 s** — and it is a per-checkpoint trait, not a horizon artefact (§7c) |

### 7b. The honest reading of +11.5 points: direction yes, magnitude not yet

**p = 0.088 at n = 96 does not clear a 0.05 bar.** The pre-registration set the threshold at
"more than 10 points" and the cell delivers 11.5, so the belief is met as written — but "the
horizon is worth 11.5 points" is exactly the shape of claim that `EXP_TIGHT.md` §7 caught last
session, where a +16.7-point single-cell effect at p = 0.0045 replicated as 1 improved / 4 worse.
A pre-registration is not a licence to quote an underpowered number.

`scripts/run_horizon2.sh` is queued: both subjects again at spawn seed 888 (baselines already on
disk), plus `bc_armB_seed2` at 30 s to see whether the gain saturates.

**The mechanism evidence is much stronger than the rate evidence**, and it is what makes this
worth believing at all. Failure buckets and median failure depth, 12 s → 20 s:

| subject | `never_entered` | its median depth | `stalled_in_mouth` |
|---|---|---|---|
| `bc_armB_seed2` | 13 → **7** | −47.0 → **−38.4 mm** | 21 → 17 |
| `bc_armB_seed1` | 20 → **20** | −43.3 → −45.2 mm | 7 → 7 |

`bc_armB_seed2`'s blocks **advance** when given more time — the bucket halves and the ones still
in it are 9 mm further along. `bc_armB_seed1`'s do not move: same count, same depth, to within
noise. That is a direct read of **slow versus stuck**, and no significance test is doing the work.

### 7c. Belief 2's mechanism was wrong even though its prediction was right

I predicted `bc_armB_seed1` would not gain because its failures were "41 mm outside the mouth;
more time spent doing the wrong thing does not help" — i.e. that `never_entered` failures are
intrinsically time-insensitive. But `bc_armB_seed2`'s gain came **mostly from `never_entered`**
(6 of its 11 recovered episodes), from failures sitting at −47.0 mm, which is *further* out than
`bc_armB_seed1`'s −43.3 mm.

So the taxonomy bucket does not predict whether time helps. **The checkpoint does.** Two policies
whose failures are described identically by depth and lateral error differ completely in whether
those failures are recoverable, and the only way to tell them apart was to give both more time.
That is worth carrying forward: `failure_taxonomy.py` describes *where the block ended up*, which
is not the same as *why it stopped*.

### 7d. A bucket nobody had counted: seated, then lost

Belief 4 predicted `placed_max > placed_final` would *appear* at 20 s. It was there at 12 s:
10/96 for `bc_armB_seed2`, and 20 s made it slightly **better** (8/96), not worse.

Counting it across the whole Stage C sweep — 3456 later-cohort episodes — **102 (3.0 %) reached
a fully seated block and did not end seated.** And it is not spread evenly: all six of
`bc_armB_seed2`'s cells are in the worst eight, at 5–10 %, while `bc_armA_seed0` and
`bc_armA_seed1` have cells at exactly **zero**. One checkpoint seats the block and then knocks it
back out; another never does.

That is a **third** distinct failure mechanism, alongside the aiming and push failures in §6, and
it is invisible in every table in this project because success is sampled at the end of the
episode. It also means `bc_armB_seed2` was a slightly unlucky choice of subject here: ~8 points
of its 12 s baseline is unseating rather than depth. The gain does not come from fixing that
(unseating only moved 10 → 8), but the baseline it is measured against is depressed by it.

---

## 8. Probe A round 2 — ⚠ THE DIFFERENTIAL DOES NOT REPLICATE. §7 IS SUPERSEDED.

*Run 11:44–12:00. `scripts/run_horizon2.sh`. Read this before quoting anything in §7.*

Both subjects re-run at spawn seed 888, plus `bc_armB_seed2` at 30 s:

| subject | shape | seed | 12 s | 20 s | gain | p |
|---|---|---|---|---|---|---|
| `bc_armB_seed2` | stalled-in-mouth | 777 | 0.625 | 0.740 | **+11.5** | 0.088 |
| `bc_armB_seed2` | stalled-in-mouth | **888** | 0.708 | 0.708 | **+0.0** | 1.000 |
| | | pooled | 0.667 | 0.724 | +5.7 | 0.223 |
| `bc_armB_seed1` | never-entered | 777 | 0.719 | 0.698 | −2.1 | 0.751 |
| `bc_armB_seed1` | never-entered | **888** | 0.615 | 0.719 | **+10.4** | 0.126 |
| | | pooled | 0.667 | 0.708 | +4.2 | 0.378 |

The four per-cell gains are **+11.5, +0.0, −2.1, +10.4**. They do not sort by failure shape. The
subject that was supposed to gain and the subject that was supposed not to end at **+5.7 and
+4.2** — the same number.

**All four cells pooled (n = 384 per arm): 0.667 → 0.716, +4.9 points, z = +1.48, p = 0.138.**

### 8a. Corrected scoreboard

| belief | §7 verdict | round-2 verdict |
|---|---|---|
| 1. stalled shape gains > 10 pts | ✅ +11.5 | ❌ **one seed only** — +0.0 at s888 |
| 2. never-entered shape gains < 5 pts | ✅ −2.1 | ❌ **one seed only** — +10.4 at s888 |
| 3. **the gap between the gains is the result** | ✅ 13.6 pts | ❌ **the gap is −10.4 at s888. It is noise.** |
| 4. unseating appears at 20 s | ❌ already present at 12 s | ❌ unchanged (10 → 8 → 8) |

### 8b. What is actually supportable

> Doubling the episode horizon is worth roughly **+5 points**, pooled over two checkpoints and
> two spawn seeds, and that is **not distinguishable from zero at n = 384 (p = 0.14)**. There is
> **no evidence** that the gain depends on which failure shape a checkpoint exhibits. Going
> further to 30 s buys nothing (0.740 → 0.729), so whatever the effect is, it has saturated by
> 20 s.

The one thing that does look consistent is the `never_entered` count falling with more time —
13→7, 9→5, 20→20, 25→19, i.e. down in three of four cells. But the median depth of the survivors
moves incoherently (−47.0 → −38.4 mm in one cell, −37.5 → −51.6 in another), so the "**the blocks
advance**" reading from §7b was also a single-seed artefact. I should not have written it as a
mechanism.

### 8c. What went wrong in §7, and why it is the same mistake as last session

§7b explicitly warned that +11.5 at p = 0.088 was underpowered and named `EXP_TIGHT.md` §7 — the
+16.7-point single-cell effect that replicated as 1 improved / 4 worse — as the precedent. Then
it went on to say the mechanism evidence "is much firmer than the rate" and built a story on
`never_entered` counts from **the same two cells** whose rates were underpowered.

That is the error: **a second statistic computed from the same underpowered cells is not
independent evidence.** 13→7 and 20→20 looked like a mechanism because they came from the two
runs whose rates happened to differ. At s888 the same statistic gives 9→5 and 25→19, which
"supports" the opposite conclusion equally well.

The pre-registration did its job — round 2 was written and queued *before* §7 was drafted,
precisely because the p-values were weak. The failure was in the prose, not the protocol: I wrote
a confident mechanism paragraph while the replication that would refute it was already running.

### 8d. Where this leaves the depth question

The horizon is **not** the explanation for the depth bottleneck. `SESSION5_FINDINGS.md` §6d-ii's
finding stands unaltered: 82–84 % of failures at every clearance are depth failures, and giving
the policy 67 % more time recovers at most a few points of them, not significantly.

So probe A's real answer is a **negative** one, and it is worth having: the depth failures are
mostly **stuck, not slow**. That makes them a legitimate target for Stage D steering rather than
something a config change would have fixed — which is the opposite of what §7 concluded, and is
the more useful conclusion for deciding what to do next.
