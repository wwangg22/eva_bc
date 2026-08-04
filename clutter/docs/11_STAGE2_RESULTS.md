# 11 — Stage 2 results: prerequisite probes

*Evidence log for `09_STAGE2_BC_PLAN.md`. Every entry records the prediction that was
registered **before** the run, then the measurement, then the verdict — including when the
verdict is "the prediction was wrong".*

Status: **closed for the prerequisite probes.** Complete: **P28** (§1), **P30** (§2), expert
fixes (§2.6), **P26-v2** (§2b), **the roll rule** (§2c), **P26-v3** (§2d), **P26-v4/P32/P33/P34
and Q7** (§2e–§2h). **Still unrun: P27, P24** — both edited to take the frozen pose.

> **→ The port, the demo set and the policy are in `13_STAGE2_BC_RESULTS.md`.** That file
> supersedes this one for anything downstream of the frozen expert: P29's branch seam,
> Gate 2a/2b, `runs/demos_v1.hdf5`, the offline N2 measurement, and flow BC at **68.5 %**
> against a paired expert at **71.4 %**.

**Headline.** Three probes meant only to *de-risk* the port instead found **four defects in the
expert's pose-selection code** and one free 7-point variable it had been assigning at random
since Stage 0. The expert's end-to-end number is **unchanged** (74.6 % against 73.7 %) — the
screening step was already capturing most of that variable by accident — but the
data-generating process for Stage 2 is now deterministic where it used to be lucky, which is
what BC actually needs.

**One registered prediction held (P30, +7.0 paired) and one was falsified (P26-v3, expected
~78 %, got 74.6 %).** Both are written up in full, including the arithmetic that reconciles
them.

---

## 0. WHY THERE ARE PROBES BEFORE A PORT AT ALL

`05_PORTING_MAP.md` treated Stage 2 as mechanical: change six constants, rewrite two filter
functions, run. The review behind `09_STAGE2_BC_PLAN.md` found eight places where clutter
breaks an assumption the eva_bc pipeline was built around, and three of them are questions
about the **expert's own trajectory** that no Stage-0 or Stage-1 probe ever had a reason to
ask, because they only matter once a *learner* has to reproduce the motion:

| question | why it never came up before |
|---|---|
| do independently screened poses agree? | one pose per evaluation run; agreement was never needed |
| how short can the holds be? | duration is free in a physics-only measurement |
| what happens between `_START_POSE` and `chain[0]`? | every prior number was collected downstream of a teleport |

---

## 1. P28 — IK-branch structure of screened grasp poses

**Question.** Stage 2 wants demos from several spawn seeds, each screening its own grasp pose.
If two screened poses lie in different IK branches, the dataset contains very different joint
trajectories for near-identical observations, and a chunk policy that interpolates between
them reproduces P17's 108 mm branch flip *inside the network*, where no segment audit can see
it.

**Prediction, registered before the run.** *One branch.* The CEM is seeded from the folded home
pose, the hinge cost is dominated by position, and with `o_hat = x̂` the approach axis is
confined to the y–z plane so the wrist stub must sit in one of the 12 mm row gaps at
y ≈ −20 mm. That is a narrow basin. **Predicted max pairwise per-joint L∞ < 0.35 rad.**

**Measurement.** 8 independent spawn batches, `screen = 4`, 128 envs, `plan_full = False`.
Wrist position is `TCP − 0.0419·â`, computed from the recorded pose.

```
draw       j1      j5      j6  | wrist x  wrist y  wrist z | o_align  screen   pen   err
   0   -0.310  -0.492  -1.946  |   249.0    +20.9     18.2 |  0.9996   68.8%  0.00  0.71
   1   -0.353  -0.566  +1.209  |   255.2    +21.8     19.9 |  0.9936   76.6%  0.00  0.64
   2   +0.304  +0.489  -1.284  |   249.1    -21.5     18.1 |  0.9974   82.0%  0.00  0.86
   3   -0.271  -0.438  +1.367  |   247.4    +20.3     17.8 |  0.9906   88.3%  0.00  1.28
   4   -0.310  -0.497  +1.205  |   249.8    +21.1     18.9 |  0.9997   78.1%  0.00  0.90
   5   -0.448  -0.738  +1.452  |   271.1    +20.9     25.4 |  0.8627   98.4%  0.00  0.31
   6   +0.341  +0.543  +1.909  |   250.6    -21.7     20.3 |  0.9985   65.6%  0.00  1.27
   7   +0.380  +0.619  -1.028  |   266.2    -20.0     21.0 |  0.8661   85.9%  0.00  0.82
                                                            [mm]              [mm]  [mm]
```

```
max pairwise per-joint L-inf : 3.855 rad          median : 2.233 rad
per-joint spread             : j1 0.828  j2 0.486  j3 0.611  j4 0.651  j5 1.357  j6 3.855
clusters at tol = 0.35 rad   : SIX  —  [0] [1,3,4] [2] [5] [6] [7]
```

**Prediction falsified, decisively.** Six clusters where one was predicted, and the spread on
`joint6` alone (3.855 rad) is eleven times the branch tolerance.

But the raw clustering **overstates** the structure, because two of the six splits are a
symmetry rather than a difference. Restricting the distance to joints 1–5 and looking at what
is left in `joint6`:

```
pairs whose joints 1-5 agree within 0.35 rad — i.e. differ ONLY in wrist roll
   0 vs 1 :  max|dq[1..5]| = 0.137     |dj6| = 3.154     |dj6| - pi = +0.013
   0 vs 4 :  max|dq[1..5]| = 0.050     |dj6| = 3.151     |dj6| - pi = +0.009
   0 vs 3 :  max|dq[1..5]| = 0.054     |dj6| = 3.313     |dj6| - pi = +0.171
   2 vs 6 :  max|dq[1..5]| = 0.053     |dj6| = 3.193     |dj6| - pi = +0.051
   1 vs 4 :  max|dq[1..5]| = 0.091     |dj6| = 0.003
   3 vs 4 :  max|dq[1..5]| = 0.065     |dj6| = 0.162
```

**0 vs 4 differ by π to within 9 milliradians.** 0 vs 1 to within 13. 2 vs 6 to within 51.
That is not a coincidence and it is not a branch — it is the parallel jaw's own symmetry.

**So the real structure is a 2 × 2, doubled by the wrist roll:**

| family | wrist side | arm | draws | `o_align` | screen |
|---|---|---|---|---|---|
| **A** | **+y** (y ≈ +21 mm) | folded (j2 ≈ −1.72) | **0, 1, 3, 4** | 0.9906–0.9997 | 68.8 / 76.6 / 88.3 / 78.1 % |
| **B** | −y (y ≈ −21 mm) | folded | 2, 6 | 0.9974, 0.9985 | 82.0 / 65.6 % |
| **C** | +y | **extended** (j2 = −1.28) | 5 | **0.8627** | **98.4 %** |
| **C′** | −y | **extended** (j2 = −1.35) | 7 | **0.8661** | 85.9 % |

Three separate findings fall out, in ascending order of how much they matter.

### 1.1 The `joint6` split is the parallel jaw's own symmetry — measured, not assumed

Everything `plan()` scores is **blind** to this difference by construction:

- the CEM's orientation term is `|o_hat · o_des|` — **deliberately sign-free**, because a
  parallel jaw is symmetric and demanding a sign would reject half the valid solutions
  (`_kin.py`, `cem` docstring);
- `box_penetration` takes a **max over body origins**, so swapping which finger sits where
  cannot change it.

Both representatives therefore score identically on every statistic used to choose between
them, and the search returns whichever it happens to reach.

**Whether that is harmless is a real question, not a formality.** The finger meshes measured in
P22 are not symmetric about the roll axis:

```
gripper_left    x −19.2 .. +19.2 | y −41.9 .. +46.7 | z −58.7 .. +34.7   [mm, body frame]
gripper_right   x −19.2 .. +19.2 | y −46.7 .. +41.9 | z −58.7 .. +34.7
```

In **y** the two are exact mirrors, so swapping them is free. In **z they are identical, not
mirrored** — the blade runs 58.7 mm one way and 34.7 mm the other. A roll that maps left onto
right's position also flips that profile, so the two configurations sweep volumes differing by
up to **24 mm** — against the **7.8 mm of margin** P22 showed is the whole remaining failure
mode.

→ **P30 measures it.** It is not decidable from the mesh table alone, because the mapping from
body-frame z to world depends on the full orientation, and the standing rule since P01 is that
geometry gets measured rather than argued.

### 1.2 The wrist sits in **either** gap, 5 draws to 3 — and the handoff said otherwise

`HANDOFF.md` §3.2 states the wrist "sits in one of the 12 mm gaps, at **y ≈ −20 mm**, z ≈ 19 mm.
It is put there **deliberately** and verified by the keep-out term, not left to chance."

The first half is right and the second half is not. Over eight independent solves the wrist
went to **+21 mm five times and −21 mm three times**, and nothing in the search prefers either:

```
+y gap :  draws 0, 1, 3, 4, 5
-y gap :  draws 2, 6, 7
```

The mechanism is visible in the joint vectors — draw 2 sign-flips **joint1 and joint5** while
leaving joints 2, 3, 4 nearly unchanged. Both solutions put the TCP at x ≈ 250 mm, y ≈ 0, i.e.
azimuth 0, so a base yaw of ∓0.3 rad is being cancelled by an equal and opposite wrist bend.
This is the **mirrored-wrist family** P18 was looking at.

Note what it is *not*: it is not worse. P18 measured a mirrored pose at `o_align` 0.885 with
21 % at-goal, and **that finding is already retracted** — its control changed wrist side *and*
jaw alignment together. Draw 2 scores `o_align` 0.9974 and screens at 82.0 %, the best of the
four fully-aligned folded draws. **The mirror family is fine; P18's particular mirror was not.**

**Correction to `HANDOFF.md` §3.2: which gap the wrist threads is not chosen, it is drawn.**
The keep-out term verifies that it clears *whichever* gap it lands in. That is a weaker claim
than the one on record, and it is the accurate one.

### 1.3 The `o_align ≥ 0.99` gate is documented but **not enforced** — and it let two poses through

Draws **5 and 7 ship with `o_align` 0.8627 and 0.8661**, far below the `O_ALIGN_MIN = 0.99`
gate that `HANDOFF.md` §3 lists as step 3 of pose selection. Reading `clutter_expert.plan()`
shows why:

```python
ok = c0["o_align"] >= self.o_min and c0["pen"] <= 1e-4
if best is None or c0["wrist_z"] > best["wrist_z"]:
    best = c0                      # <-- 'best' tracks the HIGHEST WRIST, not alignment
if ok and c0["wrist_z"] >= self.wrist_min_z:
    break
c0 = c0 if (c0["o_align"] >= self.o_min and ...) else best      # <-- fallback ignores o_align
if self.screen:
    c0 = self._screen(grip, c0)    # <-- _screen's OWN candidates are never gated at all
```

Two independent leaks: the fallback keeps the highest-wrist attempt regardless of alignment,
and `_screen` generates its other `screen − 1` candidates with a bare `_solve` call that gates
only on position error. **The advertised gate applies to nothing that ships.**

Whether that is a bug depends on whether the simulator screen is a strictly better selector
than `o_align` — which is exactly what P26 argued. But the screen's enclosure test is

```python
held = (K.gap() - self.width).abs() < 0.012          # width = 36 mm
```

so any stall between **24 and 48 mm** counts as enclosed. A jaw 30° off the block's faces
closing on a corner falls comfortably inside that window. **The `held` term loses its
discriminative power in exactly the regime where the gate would have fired**, so the screen
score cannot be trusted to have caught what `o_align` was there to catch.

And the pattern in draws 5 and 7 is one we have seen before. Both have a **more extended arm**
(j2 ≈ −1.28/−1.35 against −1.72), a **higher wrist** (25.4 / 21.0 mm against 17.8–20.3), poor
alignment, and the **two highest screen scores in the run (98.4 %, 85.9 %)**. That is the P25
family — clearance bought with jaw alignment — which was measured end to end at **2.0 % success,
−56.2 points**.

**Pre-registered prediction:** draws 5 and 7 underperform end to end despite screening best,
and the mechanism is that `held` cannot distinguish a face grasp from a corner pinch.

**Decision: gate the candidates entering the screen on `o_align ≥ o_min`.** `o_align` is
*necessary* and not sufficient — that is the documented position and it survives this run
intact. The screen stays the decider; it just stops being offered candidates the recipe already
excludes.

### 1.4 What this changes for Stage 2

**`plan()` draws from four structurally different pose families and cannot tell them apart.**
For the expert that is invisible: each pose is screened on its own merits and a good one is
kept, which is why Stage 1 worked. For BC it is a defect in the data-generating process,
because it is the **joint** trajectory the policy has to reproduce, and families A and B are
mirror images that produce opposite joint signs for observations that differ only by the row's
own ±5 mm of distractor jitter.

Three actions, all cheap, in order:

1. **Enforce `o_align` on screen candidates** (§1.3). Removes C and C′.
2. **Resolve the wrist roll** (P30). If it is an alias, canonicalise `joint6` and A collapses
   from four apparent branches to one. If it is not, it is a **free variable the pose search has
   been assigning at random since Stage 0**, and screening it is worth what `screen = 4` was.
3. **Lock one wrist side for the demo set.** After 1 and 2, A has 4 draws and B has 2; N3's rule
   keeps A. The screen means differ by 4 points on n = 4 vs n = 2 — not a distinguishable
   preference, so the choice is made on **count**, and recorded as arbitrary.

**What is *not* discarded:** family B is a legitimate solution, and the fact that both sides
work is worth carrying into Stage 4. A policy able to choose the side *per spawn* would beat one
locked to A — but the expert cannot demonstrate that (`adapt()` refines a single nominal chain
per env), so it is out of scope for BC and noted here rather than lost.

---

## 2. P30 — is a π roll of `joint6` a symmetry?  **NO. It is worth +7.0 points.**

**Prediction, registered before the run** (`09_STAGE2_BC_PLAN.md` §5): *they differ* — TCP and
axes agree to <1 mm / <0.005, end-to-end success differs by **more than 5 points**.

**Verdict: the prediction holds, and the effect is larger and cleaner than expected.**

### 2.1 Kinematically the two are indistinguishable — to five decimal places

Three independent pose draws, 128 envs, `screen = 4`:

```
draw 0   joint6 -1.859 -> +1.283
   TCP            (249.8, +0.6, 55.0) vs (249.8, +0.6, 55.0) mm    |d| = 0.00 mm
   a_hat . a_hat'  +1.00000        |o_hat . o_hat'|  1.00000
   penetration     0.00 mm  vs  0.00 mm
   gripper_end    moves (  +0.0,  -0.0,  +0.0) mm
   gripper_left   moves ( -90.0,  +3.9,  -2.3) mm
   gripper_right  moves ( +90.0,  -3.9,  +2.3) mm
   left' vs right: 0.00 mm apart          <- the fingers swap EXACTLY
```

Identical on all three draws: **TCP to 0.00 mm, both axes to 1.00000, penetration identical,
and each finger origin lands exactly where the other one was.** Every statistic `plan()` uses
to choose a pose — position error, `o_align`, `box_penetration`, `low_z`, wrist height — is
**provably** blind to this variable, not merely insensitive to it.

### 2.2 Physically they are 7 points apart, and the sign is consistent

Paired (snapshot / restore, one spawn per draw, identical everything except `joint6`):

| draw | arm | enclosed | at goal | topple | **success** |
|---|---|---|---|---|---|
| 0 | as solved (`j6` −1.859) | 100.0 % | 92.2 % | **29.7 %** | 66.4 % |
| 0 | **rolled** (`j6` +1.283) | 100.0 % | 89.1 % | **14.8 %** | **76.6 %**  (+10.2) |
| 1 | as solved (−1.818) | 100.0 % | 93.8 % | 25.8 % | 69.5 % |
| 1 | **rolled** (+1.324) | 100.0 % | 88.3 % | 18.8 % | **75.8 %**  (+6.2) |
| 2 | as solved (−1.853) | 100.0 % | 94.5 % | 26.6 % | 71.9 % |
| 2 | **rolled** (+1.288) | 100.0 % | 90.6 % | 14.8 % | **76.6 %**  (+4.7) |
| | **pooled, 384 eps/arm** | | | | **69.3 % → 76.3 %, +7.0** |

**The mechanism is visible: topple falls by 7–15 points in every draw.** Enclosure is 100 % in
both arms, and at-goal is slightly *worse* rolled (−3 to −5 points) — so this is not a better
grasp, it is a **less destructive one**. That is precisely the residual failure mode P22
isolated, and this is the first thing since screening that has moved it.

### 2.3 Why — and why my own mesh reasoning got it backwards

The finger origins swap exactly, so the difference can only be in **collision-mesh
orientation**: each blade is rotated π about the approach axis inside a body that has moved to
its partner's place.

I argued in the probe header that this *should* matter, from the measured extents

```
gripper_left    x -19.2 .. +19.2 | y -41.9 .. +46.7 | z -58.7 .. +34.7   [mm, body frame]
gripper_right   x -19.2 .. +19.2 | y -46.7 .. +41.9 | z -58.7 .. +34.7
```

— on the grounds that y is mirror-symmetric between the two bodies (free to swap) while z is
*identical* rather than mirrored, so a roll that swaps them must flip a 93.4 mm asymmetric
profile.

**That argument is only correct if the roll axis is not the body's z**, and I convinced myself
mid-run that it probably was, i.e. that the roll would be an exact symmetry after all. **The
measurement says otherwise**, and the measurement is what counts. The 24 mm figure quoted in the
probe header is an upper bound derived from an assumed frame alignment and should not be quoted
as a measured quantity; what is measured is **7.0 points of end-to-end success**.

This is the P01 lesson recurring: geometry read off an extent table, without checking which
frame it lives in, produces a confident wrong answer. Here it produced a confident *right*
answer for a reason that may be wrong — which is worth flagging, because a right prediction from
a shaky argument is not evidence the argument was sound.

### 2.4 The confound, stated

Each arm's chain is **re-solved** from its own grasp pose outward (`_solve(..., restarts=1,
std0=0.08)`), so the two arms differ by the roll *and* by independent local-solve noise. The
effect survives that because it is consistent in sign across three independent draws and 384
paired episodes, and because the channel it moves (topple, −7 to −15 points) is not one that
local-solve noise has ever moved. It is still a second variable and it is recorded as one.

### 2.5 What this changes — the largest actionable finding of Stage 2 so far

**`plan()` has been assigning the wrist roll at random since Stage 0.** It is a free binary
variable, invisible to every forward-kinematic statistic, worth **7 points** — comparable to
the **+10** that simulator screening itself bought.

And there is a hint it is not even a coin flip. Every draw here solved to `j6 ≈ −1.85` and every
roll to `≈ +1.28`, and **rolled won all three times**. P28's independent draws agree: its one
`j6 ≈ −1.95` member of family A scored the **lowest screen score in the family** (68.8 % against
76.6 / 88.3 / 78.1 %). That suggests a systematic preference within a wrist-side family, which
would flip for the mirrored family.

**Decision: screen both rolls, do not hardcode a sign.** The preference is expected to be
systematic *within* a family and to invert between families, so a hardcoded sign would be right
half the time and silently wrong the other half — exactly the failure this probe just found.
The cost is 2× the screen trials, and a trial is 720 physics steps.

---

### 2.6 The three expert changes P28 and P30 forced

All in `expert/clutter_expert.py`. Each is additive or a defect fix; none changes the
manoeuvre.

| change | what | why |
|---|---|---|
| **`_rolls(c)`** | every screen candidate is tried in **both** wrist rolls | P30: worth +7.0 points and invisible to every FK statistic |
| **`_gated_solve()`** | replaces `plan()`'s inline retry loop; the `o_align` / penetration / wrist-side gates are enforced on **every** candidate, including the ones `_screen` generates; the fallback keeps the **best alignment**, never the highest wrist | P28 §1.3: the advertised gate applied to nothing that shipped, and its fallback selected the P25 family |
| **`wrist_side`** | 0 = either, ±1 = lock the wrist to one 12 mm gap | P28 §1.2: the side is drawn, not chosen; mirror-image joint trajectories are a defect in the data-generating process for BC |
| `holds` / `env_steps()` | hold durations are parameters; schedule length is computable | P27 needs to sweep them; the demo horizon must be sized, not guessed |

**A screen of K now costs up to 2K trials.** `_gated_solve` inside `_screen` is capped at 4
attempts rather than `plan()`'s 8, because 6 of P28's 8 draws cleared `o_align` on the first
attempt, so the smaller budget costs almost nothing and bounds the worst case.

---

## 2b. P26-v2 — screening both rolls made it WORSE. 73.7 % → 66.5 %.

The obvious response to P30 was to screen both rolls. It was implemented, measured on the
canonical protocol (3 selections × 2 held-out batches = 768 episodes), and **it is wrong**:

| arm | pooled | spread | sd | screen scores | optimism |
|---|---|---|---|---|---|
| Stage 1, `screen = 4`, random roll | **73.7 %** | 67.2–80.5 % | **4.8 %** | 79 / 84 / 88 % | **+9.9** |
| Stage 1, `screen = 8`, random roll | 67.2 % | 52.3–78.9 % | 10.2 % | 84 / 85 / 91 % | +19.3 |
| **v2, `screen = 4` × both rolls** | **66.5 %** | **51.6–80.5 %** | **11.9 %** | 78 / 88 / 89 % | **+18.7** |

**v2 reproduced `screen = 8`'s signature to the point** — and it should have been obvious in
advance, because screening K candidates in **both** rolls *is* a pool of 2K. `HANDOFF.md`
§10.2 item 1 says so in as many words: *"`screen = 8` currently overfits a single 128-env
selection batch… the fix is not fewer candidates but more screening spawns, after which the
candidate count can rise again."* **The candidate count rose without the screening spawns.**

The per-selection detail shows how selection overfitting actually fails — it is not a uniform
haircut, it is one catastrophic pick:

| selection | screen | held-out b0 | held-out b1 | gap |
|---|---|---|---|---|
| 0 | **89.1 %** | 54.7 % | 51.6 % | **−36.0** |
| 1 | 88.3 % | 80.5 % | 77.3 % | −9.4 |
| 2 | 78.1 % | 71.1 % | 64.1 % | −10.5 |

Selections 1 and 2 carry the ordinary ~10-point optimism. Selection 0 picked the highest of
eight scores on one 128-env batch and lost **36 points** on held-out spawns. With 8 candidates
whose true means sit within a few points of each other and a per-candidate standard error of
about 3.5 points, the maximum is mostly a draw from the noise.

**Recorded as a mistake I made, not as a property of the roll.** P30's paired measurement is
untouched by this — it compared one pose against itself with only `joint6` changed.

## 2c. The roll does not need screening. It needs a rule — and the rule is 12/12.

The v2 run produced **twelve** roll pairs, each scored on 128 envs. Writing `w_y` for the
wrist's y-coordinate:

| `w_y` [mm] | `j6` | score | `j6` | score | winner | margin |
|---|---|---|---|---|---|---|
| **+20.8** | **+1.202** | **70.3 %** | −1.939 | 60.9 % | `+` | +9.4 |
| **−20.1** | +1.912 | 80.5 % | **−1.230** | **89.1 %** | `−` | +8.6 |
| **+21.5** | **+1.217** | **79.7 %** | −1.925 | 71.1 % | `+` | +8.6 |
| **+20.1** | **+1.155** | **74.2 %** | −1.987 | 64.1 % | `+` | +10.1 |
| **+20.3** | −1.844 | 81.2 % | **+1.297** | **88.3 %** | `+` | +7.1 |
| **−21.1** | **−1.185** | **74.2 %** | +1.957 | 64.8 % | `−` | +9.4 |
| **−20.5** | +1.947 | 77.3 % | **−1.194** | **87.5 %** | `−` | +10.2 |
| **−21.2** | **−1.215** | **83.6 %** | +1.926 | 80.5 % | `−` | +3.1 |
| **−20.9** | **−1.244** | **78.1 %** | +1.898 | 70.3 % | `−` | +7.8 |
| **+20.5** | −1.942 | 64.1 % | **+1.199** | **70.3 %** | `+` | +6.2 |
| **−20.7** | **−1.217** | **65.6 %** | +1.925 | 64.8 % | `−` | +0.8 |
| **−21.6** | **−1.061** | **65.6 %** | +2.081 | 64.1 % | `−` | +1.5 |

**Every winner satisfies `sign(j6) == sign(w_y)`. Twelve out of twelve** — p = 2⁻¹² = 0.00024
on the sign alone — with a **mean margin of +6.9 points**, which matches P30's independently
measured **+7.0** to within a tenth of a point.

So the roll is not a variable to search. It is a **function of which 12 mm gap the wrist is
in**, and applying it costs nothing: the candidate pool stays at the validated 4.

**Mechanism.** The blades are asymmetric about the roll axis, so which way their tails point
relative to the neighbours depends on which side the wrist is threading. Matching the sign
turns the tails away from the row.

**The confound, stated.** Every winner **also** has `|j6| < π/2` (1.06–1.30) and every loser
`|j6| > π/2` (1.84–2.08). The two descriptions are perfectly correlated in these twelve pairs
and **cannot be separated by them**. The sign rule ships because it has a mechanism and `|j6|`
does not; `roll_mode="screen"` stays in the code as the experiment that would settle it. This
is recorded rather than glossed because "two rules, one dataset, pick the pretty one" is how
P20's grip-height optimum got published and later withdrawn.

**Registered prediction for the v3 re-measurement** (`screen = 4`, `roll_mode="rule"`, same
768-episode protocol): **≥ 73.7 %**, expected **~78 %**, with sd back near 4.8 %.

---

## 2d. P26-v3 — the rule works and buys nothing. **74.6 %. Prediction falsified.**

| arm | pooled (768 eps) | spread | sd | screen scores | optimism |
|---|---|---|---|---|---|
| Stage 1, `screen = 4`, random roll | 73.7 % | 67.2–80.5 % | 4.8 % | 79 / 84 / 88 % | +9.9 |
| v2, `screen = 4` × both rolls (= 8) | 66.5 % | 51.6–80.5 % | 11.9 % | 78 / 88 / 89 % | +18.7 |
| **v3, `screen = 4`, roll by rule** | **74.6 %** | 64.8–85.2 % | 7.6 % | 94 / 88 / 90 % | **+16.0** |

**74.6 % against 73.7 % is +0.9 points on a per-cell sd of 7.6 %. That is a null result**, and
the "~78 %" I registered is wrong.

### Why — and the arithmetic checks out

Stage 1's four screened candidates had **random** rolls. A good-roll candidate scores about
**7 points higher on the screen**, so the `argmax` over four candidates already picked one
almost always:

```
P(all four candidates have the bad roll) = (1/2)^4 = 6.25 %
```

**So Stage 1 was already capturing the roll benefit ~94 % of the time, by accident.** The rule
takes that from 94 % to 100 %, worth `0.0625 × 7 ≈ +0.4 points` — which is what was measured,
to within the noise.

The same arithmetic reaches backwards. `screen = 0` takes whatever roll the CEM returns, so it
has a **50 %** chance of the bad one; `screen = 4` has 6.25 %. Roll selection alone therefore
accounts for

```
(0.50 − 0.0625) × 7 ≈ +3.1 points
```

**of the +10.0 that `HANDOFF.md` §6.3 credits to "simulator-screened pose selection".** That
line is not wrong, but roughly a third of it was buying a variable nobody knew was there.

### What the rule is still worth

Not success. **Determinism.** `plan()` no longer leaves a 7-point variable to a coin flip, which
matters more for Stage 2 than for the expert: the demo set is the data-generating process, and
a teacher that is silently 7 points worse on ~6 % of its pose draws puts unexplainable variance
into the training set. It also costs nothing — the candidate pool stays at 4.

**Retained on those grounds, with the null end-to-end result recorded as the headline.**

### A third gate leak, found by v3 and NOT YET RE-MEASURED

Selection 1's winning candidate has `o_align = 0.9788`, below the 0.99 floor, and the run
printed `<-- GATES NOT MET`. `_gated_solve` returns its best-alignment attempt when *none*
clears the gate, so a sub-gate candidate still enters `_screen` — and can then win on screen
score, which is exactly what happened (cand 3, `o_align` 0.9788, screen 87.5 %, beating cand 1
at `o_align` 0.9961 / 85.2 %).

**And it was the worst cell of the run, by seven points:**

| selection | `o_align` | screen | held-out b0 | b1 | mean |
|---|---|---|---|---|---|
| 0 | 0.9993 | 94.5 % | 85.2 % | 81.2 % | **83.2 %** |
| **1** | **0.9788 ✗** | 87.5 % | 68.8 % | 64.8 % | **66.8 %** |
| 2 | 0.9982 | 89.8 % | 74.2 % | 73.4 % | **73.8 %** |

That is the third independent piece of evidence that **`o_align` is necessary** — P18's 0.848
and 0.885 both scored under 30 %, P28's 0.863/0.866 pair was the P25 family, and now 0.9788
costs 7 points against otherwise-identical machinery. n = 1, so it is corroboration, not proof.

**Fix applied** (`_screen` skips sub-gate candidates unless nothing else is available).
**It is unmeasured.** A v4 run on the same 768-episode protocol is the first thing to do.

### Registered prediction for v4

With the leak closed, selection 1's cell should move from 66.8 % toward the 73–83 % the other
two produced, putting the pooled number in the **78–80 %** range. *If v4 lands at 74–75 % again,
`o_align` is not the discriminator and the 66.8 % cell was an ordinary bad pose* — in which
case the gate should be reported as advisory and the effort spent on multi-batch screening
instead.

**Until v4 lands, 74.6 % is the number of record** (which is, within noise, the same 73.7 % it
has been since Stage 1).

---

## 2e. P26-v4 — **62.8 %.** And then the meta-analysis that invalidates the whole P26 family.

*Run 2026-08-03, `runs/p26_screen_v4.json` / `p26_v4.log`. 768 episodes, same protocol.*

### The result

| selection | `o_align` | wrist y | screen | b0 | b1 | mean |
|---|---|---|---|---|---|---|
| 0 | 0.9943 | −22 mm | 83.6 % | 61.7 % | 43.8 % | **52.8 %** |
| 1 | 0.9998 | **+21 mm** | 80.5 % | 78.9 % | 77.3 % | **78.1 %** |
| 2 | 0.9905 | −20 mm | 89.8 % | 56.2 % | 58.6 % | **57.4 %** |
| | | | | | **POOLED** | **62.8 %** (sd 13.4) |

The leak fix **did** fire — selection 0's screen log shows candidates 0, 1 and 3 only;
candidate 2 was skipped for failing the gate, exactly as designed. The mechanism works. The
outcome went the wrong way.

**The registered 78–80 % prediction is not supported.** I want to be precise about how badly,
because the honest answer is *not* "falsified":

```
v4 selection means   52.8, 78.1, 57.4      sd 13.5   se 7.8   95 % CI [47.5, 78.0]
```

The run's own confidence interval **reaches 78.0**. A prediction of 78–80 % is not refuted by
an experiment whose interval includes it. What v4 actually establishes is that the experiment
cannot decide the question — which turned out to be the finding worth having.

### Two design defects in my own fix, independent of the noise

1. **It removes candidates instead of replacing them.** `continue` shrinks the screening pool
   from 4 to 3 for that selection. Given §2b's demonstration that pool size interacts with
   selector quality, a fix that silently shrinks the pool is the wrong shape; it should
   re-draw.
2. It cannot fire on selection 0's *first* candidate (`best is None`), so the incumbent still
   enters unconditionally. A gate with an exemption for the incumbent is not a gate.

Both are moot given what follows, but they are recorded because the next person to re-enable
screening will otherwise re-inherit them.

### The meta-analysis — 18 selections, and what actually predicts success

P26 has now run five times. Rather than run it a sixth, I pooled **every pose selection ever
measured** in this effort (`scratchpad/p26_meta.py`, pure arithmetic on the logs) and asked
which recorded property predicts the held-out score.

| predictor | n | r | **r²** |
|---|---|---|---|
| **simulator screen score** | 15 | +0.126 | **0.016** |
| `o_align` | 15 | +0.065 | **0.004** |

> ### ⚠ CORRECTION, same day — this table is **range-restricted** and I over-read it
>
> Every one of those 15 rows is a **winner** — a candidate that had already been chosen by
> maximising screen score over its pool of four. Correlating a selection variable with an
> outcome *inside the selected subset* attenuates `r` toward zero by construction; the losing
> candidates, which carry the low end of the screen-score range, were never evaluated.
>
> So **r² = 0.016 does not show the screen is invalid.** It shows that among candidates the
> screen already liked, its remaining score differences carry little information — which is
> a much weaker statement and roughly what range restriction predicts.
>
> Measuring the screen's real validity needs held-out scores for *losing* candidates too, and
> no run has ever produced those. It is **unmeasured**, not refuted. The claim that survives
> is the one below, which rests on replication spread rather than on this correlation.

`o_align`'s 0.004 is subject to the same caveat but is corroborated independently by P32 (§2f:
r² = 0.059 with the *opposite* sign, on unselected poses) — so my §2d inference that the
0.9788 cell's 7-point deficit was *caused* by its alignment is withdrawn either way.

Now the variable nobody was controlling:

| wrist side | n | mean | sd | se | range |
|---|---|---|---|---|---|
| **+y** | 6 | **76.8 %** | 4.8 | 1.9 | 71.1 – 83.2 |
| **−y** | 9 | **63.8 %** | 9.3 | 3.1 | 52.8 – 78.9 |

```
difference  +13.0 pts   se 3.7   t = 3.54
split at min(+y) = 71.1 %:  +y 6/6 above, -y 2/9 above   Fisher one-sided p = 0.0056
```

Every single `+y` selection beat all but two of the nine `−y` selections. And the pattern does
**not** depend on pooling across configurations — it reproduces *within* four of the five runs
that drew both sides:

| run | +y | −y | Δ |
|---|---|---|---|
| v1 s8 | 78.1 | 61.7 | **+16.4** |
| v2 | 78.9 | 60.4 | **+18.5** |
| v3 | 83.2 | 70.3 | **+12.9** |
| v4 | 78.1 | 55.1 | **+23.0** |
| v1 s4 | 71.1 | 78.9 | −7.8 ← the exception, n = 1 on the −y side |

The `−y` side also carries **twice the spread** (sd 9.3 vs 4.8), which is its own signal: the
`+y` branch is not merely better on average, it is more *reliable*.

### Why P26 could never have answered any of the questions asked of it

| | |
|---|---|
| selection-level sd, screened runs | **10.1 points** |
| selections per arm, every P26 run | **3** |
| ⇒ se of a run mean | ~5.8 points |
| ⇒ 95 % CI of a run mean | **± ~12 points** |
| selections needed to resolve a 10-pt effect | **16** |
| selections needed to resolve a 5-pt effect | **63** |

Every P26 arm ever run, with its own interval:

```
v1 s0   63.7 %   [59.0, 68.4]        v2   66.5 %   [51.9, 81.2]
v1 s4   73.7 %   [68.6, 78.8]        v3   74.6 %   [65.3, 83.9]
v1 s8   67.2 %   [54.8, 79.5]        v4   62.8 %   [47.5, 78.0]
```

**Five of the six intervals overlap each other.** The whole family — 4 608 episodes — is
consistent with a single underlying success rate around 68 % plus pose-draw noise.

### What this retracts

**R16 — "simulator screening is worth +10 points."** Withdrawn **as a quantity**, not as a
direction. The original v1 comparison (63.7 → 73.7) had unusually tight cells and was
borderline significant at 3 selections per arm, but four subsequent runs of the *same*
`screen = 4` configuration produced 73.7, 66.5, 74.6 and **62.8** — a **12-point spread across
nominally identical setups**. A configuration whose own replications span 12 points cannot
support a 10-point effect estimate.

What is *not* claimed, per the correction above: that screening does nothing. Its validity is
unmeasured, and there is a countervailing observation — the `screen = 0` arms in P32 and P33
land at 53.9–57.0 % while five `screen = 4` runs average ~69 %. That comparison is confounded
by expert version and by pose-draw luck in both directions, so it settles nothing either, but
it is the reason the honest verdict is **"unresolved and unmeasured"** rather than **"worth
nothing."** Deciding it properly needs 6+ selections per arm, batch-paired — the P34 that has
deliberately *not* been queued, because §2g explains why the tournament makes the question
moot for Stage 2.

**R17 — "screening both rolls is worse (66.5 vs 73.7)."** Withdrawn as a measurement. §2b's
*reasoning* — that enlarging a noisy selector's pool amplifies its noise — still looks right
and is still the reason not to re-enable it, but the 7.2-point number was inside the noise.

**R18 — "the roll rule is worth ≈ +0.4 points."** Withdrawn. That arithmetic was fitted to the
v3 − v1s4 difference (74.6 − 73.7 = 0.9), which is a difference between two quantities each
carrying a ±12-point interval. The rule's justification reverts to determinism, which was
always the better argument, plus P30's **paired** +7.0.

**Not retracted: P30.** P30 compared the *same pose* with `joint6` rolled, on restored spawn
state, 384 episodes per arm, three independent draws, all three agreeing in sign. Pairing
removed the pose-draw term that is destroying P26. That is the entire difference between the
two experiments, and it is the lesson:

> **P26 compared different poses across arms and paired nothing.** The dominant variance in
> this system is *which pose the CEM drew*, and an unpaired design puts that variance straight
> into the error term. Stage 1's rule was "pair everything." P26 has been violating it in
> plain sight for five runs, including the two I designed this session.

### The confound in the wrist-side finding, stated before anyone else finds it

Under the `_canon_roll` rule the representative is chosen so that `sign(j6) == sign(wrist_y)`.
**Wrist side and `joint6` sign are therefore perfectly confounded by construction.** The +13.0
points may be a property of which side of the row the forearm leans to, or of which way the
gripper is rolled, and this dataset cannot separate them. It does not affect what to *do* —
either way, force the good branch — but it does mean the mechanism is unexplained.

Two further caveats: the analysis is **observational** (sides were drawn, not assigned), and
the `screen = 0` arm predates the wrist-side logging so it contributes to the power analysis
but not to the side comparison.

### What happens next: P32

The expert has accepted a `wrist_side` argument since the P28 fixes and it has never been set.
`p32_wrist_side.py` forces it: arms `+1` / `−1` / `0`, **6** selections each (not 3),
`screen = 0`, and — the part P26 never had — **evaluation batches paired across arms** by
explicit reset seed, with a spawn fingerprint recorded per batch so the pairing is *verified*
rather than assumed.

Registered predictions: (1) `+1` beats `−1` by ≥ 8 points, falsified if |Δ| < 5;
(2) `0` lands between them near `0.6·A + 0.4·B` given P28's 5:3 draw ratio; (3) the gap is
carried by **topple**, falsified if enclosure differs by more than 3 points.

---

## 2f. P32 — the wrist-side hypothesis is **REFUTED**, and the pose variance is finally located

*Run 2026-08-03, `runs/p32_wrist_side.json` / `p32.log`. 3 arms × 6 selections × 2 batches ×
128 envs = **4 608 episodes**. Pairing verified.*

### The result: my own registered prediction, falsified

| arm | pooled | selection mean | se | encl | at-goal | topple |
|---|---|---|---|---|---|---|
| `+1` wrist +y | **57.0 %** | 57.0 % | 5.3 | 100.0 % | 90.6 % | 38.3 % |
| `−1` wrist −y | **54.2 %** | 54.2 % | 6.4 | 99.7 % | 79.5 % | 36.5 % |
| `0` free | **53.9 %** | 53.9 % | 4.4 | 99.8 % | 84.7 % | 38.9 % |

```
+y minus -y = +2.8 pts,  se 8.3,  t = 0.34
PREDICTION 1 (>= 8 pts): NOT HELD      falsifier (|d| < 5): TRIGGERED
```

The observational +13.0 points was **+2.8 ± 8.3** under assignment. Prediction 3 held
trivially (the gap is topple-carried, `d_encl` +0.3) but there is no gap to carry.

**The pairing worked**, which is what makes the falsification trustworthy: *all 12 (selection,
batch) slots were bit-identical across the three arms* — verified by a spawn fingerprint, not
assumed. `env.reset(seed=…)` does reproduce clutter spawns.

### What went wrong in my reasoning — a clean case of a mined predictor

I searched an underpowered observational dataset for a predictor, found one at `p = 0.0056`,
and reported it without discounting the search. By then I had tested `o_align`, wrist height,
screen score and wrist side against the same 18 selections; finding *one* at p ≈ 0.006 across
that many looks is close to what chance produces, and the p-value I quoted was not corrected
for having gone looking. The within-run reproductions (+16.4, +18.5, +12.9, +23.0) felt like
independent corroboration and were not — the same 15 numbers, re-sliced.

> **Earned here:** *a predictor discovered by searching a dataset must be tested on data that
> had no part in producing it, and the search itself has to be counted.* P32 cost 4 608
> episodes to kill a hypothesis that had already been written into a probe docstring as fact.
> Cheap at the price, but it should have been registered as a lead, not a finding.

### The finding that actually matters: **ICC = 0.82**

P32 is the first experiment in this effort with spawn batches paired across arms, so for the
first time the variance can be decomposed. Over all 18 poses:

| | |
|---|---|
| batch-0 score vs batch-1 score, across poses | **r = +0.812** (r² = 0.659) |
| mean \|b0 − b1\| within a pose | 6.6 pts |
| within-pose sd (batch + sampling) | **5.7 pts** |
| — of which the binomial floor at 128 envs | 4.4 pts |
| **true pose sd**, within-pose noise removed | **12.0 pts** |
| **intraclass correlation** | **0.82** |

**Eighty-two per cent of the success variance is a stable property of the pose.** A pose's
score reproduces on an independent spawn batch to within ~6 points, and most of that is the
binomial floor — there is almost no pose × batch interaction. The spread is not noise; it is
signal that no forward statistic has been able to read.

This resolves a question that has been open since P25 and settles the strategy:

- **The variance is not a defect to be explained away.** It is a real, stable, 12-point-sd
  quantity attached to each candidate pose.
- **It is therefore selectable.** A tournament needs no theory of *why* a pose is good.
- **And Stage 2 needs exactly one pose**, not a good average — demos come from a frozen chain,
  and P28's six clusters are the multimodality a chunk policy cannot disambiguate.

### Two controls run alongside

**Are the seeded batches harder than P26's unseeded ones?** No — the spawn distributions match:

| | n | mean gap | median | p10 | p90 | frac < 6 mm |
|---|---|---|---|---|---|---|
| P32 (seeded) | 4 608 | 7.97 mm | 8.04 | 5.23 | 10.36 | 15.4 % |
| P26-v4 (unseeded) | 768 | 7.93 mm | 7.95 | 5.04 | 10.57 | 16.8 % |

So P32's 54–57 % against P26's `screen = 0` 63.7 % is not a batch-difficulty artifact. It is
most likely one more instance of the very phenomenon under study — P26 drew three poses, P32
drew six, and at a 12-point pose sd those means differ by chance. Their intervals overlap.
**No conclusion is drawn from it**, which is the discipline §2e's power analysis demands.

**Does anything measured predict the pose mean, on these 18 properly-paired poses?**

| predictor | r | r² |
|---|---|---|
| `o_align` | −0.243 | 0.059 |
| wrist-side *sign* | +0.220 | 0.048 |
| **\|wrist_y\| (lateral offset magnitude)** | **−0.804** | **0.646** |

`o_align` is now *negatively* signed, and the wrist *sign* — the thing P32 was built to test —
lands at r² = 0.048, consistent with the null it just produced.

But **|wrist_y|** is the strongest predictor found in this effort by a wide margin, over a
range of only 20.1 to 22.0 mm, and it has a mechanism: with the grasp axis fixed at
`o_hat = x̂`, a larger lateral offset leans the *forearm* further into the neighbouring
column — and the forearm is a far larger body than the finger blades P22 measured.

**It is recorded as a lead, not a finding.** It was found by looking, on the same data as
before, and the last thing found that way died this afternoon. It is registered as
**prediction 0 of P33**, written into that probe *before* the run, and will be tested on eight
candidates that had no part in generating it. Prediction: `r ≤ −0.5`. Falsifier: `|r| < 0.3`.

### One more data point against the `o_align` gate

Arm `−1` selection 1 has **`o_align = 0.9674`** — far below the 0.99 floor — and is the **best
pose in its arm at 77.7 %**, beating five gate-clearing poses by up to 44 points. Combined
with r² = 0.059 and the sign flip, the gate should now be described as *weakly protective at
the very low end* (P18's 0.848 and 0.885 both scored under 30 %) and **not** as a quality
signal anywhere near threshold. §2d's inference that the 0.9788 cell's deficit was *caused* by
its alignment is withdrawn.

---

## 2g. P33 — the pose tournament. **75.4 % verified on 512 fresh episodes, +18.8 points.**

*Run 2026-08-03, `runs/p33_tournament.json` / `p33.log`. 8 candidates × 2 selection batches +
3 candidates × 4 verification batches = **3 584 episodes**.*

### The design, and why it is the right one

Four attempts to find a statistic that predicts pose quality have failed (`o_align`, wrist
height, wrist side, screen score). P32 then showed the quantity is nonetheless **real and
stable** — ICC 0.82, true pose sd 12.0 points. So: stop predicting, start selecting.

```
8 candidates, each solved on its own seeded spawn draw   (reproducible: reset(seed) seeds torch)
SELECTION   all 8 on the SAME 2 batches, spawn-identical via snapshot/restore
VERIFY      top-2 AND the worst, on 4 batches that had no vote
```

Verifying the **worst** as well as the best is what turns this from a success story into a
falsifiable test of the ranking, and re-measuring on episodes that had no vote is the only way
to know how optimistic the pick was.

### The result

| cand | role | selection (256 eps) | **verified (512 eps)** | optimism |
|---|---|---|---|---|
| **5** | **winner** | 73.4 % | **75.4 %** | −2.0 |
| 2 | runner-up | 66.4 % | 73.2 % | −6.8 |
| 0 | worst | 34.4 % | 36.9 % | −2.5 |
| | *candidate mean* | *56.6 %* | | |

```
winner VERIFIED minus candidate mean        +18.8 pts
PREDICTION 1 (gain >= 10)                   HELD
PREDICTION 3 (worst verifies below mean)    HELD   (36.9 % vs 56.6 %)
PREDICTION 2 (optimism 0-8 pts)             NOT HELD -- it was NEGATIVE
```

**The ranking generalises at both ends.** The winner stayed on top, the runner-up stayed
second, and the worst candidate — 34.4 % on the selection batches — came back at 36.9 % on
four batches it had never seen, 38 points below the winner. Pose quality is not an artifact of
the batches used to rank it.

**On the negative optimism.** All three verified candidates shifted *up* by a similar amount
(+2.0, +6.8, +2.5; mean **+3.8**), which is a batch-difficulty offset — the verification
seeds are simply a little easier than the selection seeds — not evidence that selection is
anti-optimistic. Removing that common offset leaves the winner at about **+1.8 points** of
genuine selection optimism, comfortably inside the registered 0–8 band. Prediction 2 is scored
NOT HELD as written, and the honest reading is that the ranking was very nearly unbiased.

The predicted yield from P32's variance decomposition was `verified 70.4 %, gain +15.4` for
K = 8. Observed: **75.4 %, +18.8**. The model was slightly conservative and the mechanism it
assumed is the one that operated.

### Prediction 0 — the `|wrist_y|` lead is dead, and it died the right way

```
|wrist_y| vs selection score:  r = -0.223,  r2 = 0.050   (n = 8; P32 gave -0.804)
range here: 20.3 - 21.9 mm
NOT HELD    falsifier (|r| < 0.3): TRIGGERED
```

Registered in the probe's docstring **before the run**, on candidates that had no part in
generating it, and refuted. Two mined predictors have now been killed by prospective tests in
one day — wrist side (P32) and `|wrist_y|` (P33) — at a combined cost of about 8 000 episodes.
That is the correct price for not carrying a false mechanism into Stage 2.

**Five forward statistics have now been tested and none predicts pose quality.** The working
conclusion is that no cheap forward statistic will, and that measurement is the selection
mechanism.

### The at-goal side effect — replicated three times, and it is *not* the success effect

Worth separating out, because it looks like the refuted wrist-side finding and is not:

| | wrist +y | wrist −y |
|---|---|---|
| P32 arm means, at-goal | 90.6 % | 79.5 % |
| P33 run 1, at-goal (5 vs 3 cands) | 84–93 % | 64–78 % |
| P33 run 2, at-goal (5 vs 3 cands) | 84–94 % | 59–79 % |
| **P32 arm means, SUCCESS** | **57.0 %** | **54.2 %** |

The `−y` branch reaches the goal **~11 points less often** and this has now replicated three
times. It does **not** show up in success, because success is `at_goal ∧ ¬topple` and topple is
the binding constraint at ~38 %. So: a real, side-dependent placement effect, currently masked.
**If topple ever comes down, the wrist side becomes worth having** — which is a prediction, not
a finding, and it is registered here for whenever that happens.

### What this delivers to Stage 2

The winner is frozen in **`expert/pose_p33.json`** and `ClutterExpert` gained `pose_q=` to load
it, bypassing the CEM entirely via a new `_score_q` that rebuilds the candidate dict from
forward kinematics against the live scene — so a loaded pose is reported on exactly the same
footing as a solved one (same `o_align`, same keep-out penetration, same gate).

This matters for two independent reasons:

1. **It is worth ~19 points** over letting the expert re-solve, which is what every previous
   run did.
2. **It makes the data-generating process deterministic** — the thing BC actually needs. P28
   found six pose clusters in eight draws; a dataset mixing them is exactly the multimodality
   that a chunk policy resolves by picking a mode at random every refill, which is the failure
   mode upstream's EXP07 needed a whole RL stage to undo (`12_UPSTREAM_SYNC.md` §3.3).

`p34_pose_reload.py` is the positive control on that load path: replay P33's four verification
seeds and demand the *identical* 75.4 %, then re-measure on four never-used seeds.

---

## 2h. P34 — the reload control, which found two things nothing else would have

*Run three times, 2026-08-03. `runs/p34_reload{,_v2,_v3}.json`. 1 536 episodes per run.*

P33 gave Stage 2 a pose. Stage 2 will generate every demo from it, so before anything is
recorded the pose has to survive a round trip through `expert/pose_p33.json` and
`ClutterExpert(pose_q=…)`. A load path that quietly differed from the solve path would put a
bias into every demo in the dataset and would be **invisible** — the expert would still print
a plausible `o_align` and still succeed most of the time.

### Finding 1 — freezing the grasp pose does **not** freeze the manoeuvre

Run 1 loaded the pose and replayed P33's own verification seeds. The pose came back exact
(`o_align` identical to 12 digits, penetration 0.0, wrist to 1e-3 mm) and the **score did
not**: **74.2 % against P33's 75.4 %**, on episodes that should have been bit-identical.

The cause is not the load path. `plan()` builds the other 22 waypoints with `_dense`, which
calls `_solve(…, restarts=1, std0=0.08)` — **a CEM**, drawing `torch.randn` on every call:

```
grasp pose      1 waypoint    frozen
the manoeuvre  22 waypoints   DRAWN, every single run
```

Every measurement in this effort has been of a *different trajectory*. It was invisible for
Stage 0 and Stage 1 because **no experiment had ever replayed a batch** — pairing was always
across arms within a run, never across runs.

### Finding 2 — a seed does not fix it either, and the reason is instructive

`ClutterExpert` gained `chain_seed=` (global RNG saved and restored, so callers see no side
effect). Run 2 measured whether it worked instead of assuming:

```
two experts, SAME chain_seed:   max |dq| = 1.095e-01 rad over 23 waypoints,  39 episodes flipped
```

**Not a missed RNG source.** `_dense` runs 22 waypoints × 60 CEM iterations, and every
iteration reduces with `elite.mean(0)` / `elite.std(0)` on the GPU, whose summation order is
not bit-stable. Differences of ~1e-7 compound over 1 320 iterations into a different local
optimum. Seeding narrows the spread; it cannot close it.

So the chain is now **persisted, not re-derived** — `pose_p33.json` carries all 23 `pts` and
`qs` plus `i_grip`, and `chain=` loads them with no CEM at all. Run 3:

```
two experts, loaded chain:      max |dq| = 0.000e+00 rad
```

### Finding 3 — the stack is still not bit-reproducible, and here is the floor

With the manoeuvre identical to `0.000e+00` rad, **43 of 512 episodes still flipped** between
two sweeps of the same four seeds — while the aggregate means agreed to **0.2 points**
(73.6 % vs 73.4 %).

The residual is in `adapt()`'s per-env `refine()` (finite-difference Jacobians, same GPU
reduction issue) and in PhysX itself; the leading explanation is that the second sweep inherits
solver/contact state from the first, since both run in one process. It is symmetric churn, not
bias.

> **This is a number worth keeping: ~8 % episode-level churn is the paired-comparison noise
> floor in this environment.** A future experiment reporting "12 episodes changed" is
> reporting nothing. It also settles, for a second and independent reason, that
> `09_STAGE2_BC_PLAN.md`'s Gate 2a **cannot** demand bit-exactness between the teleport and
> `env.step` paths — §N1 said so because of the teleport; this says so because of the
> simulator.

### The pose's honest score

| arm | seeds | chain | episodes | success |
|---|---|---|---|---|
| P33 verification | 91000–3 | drawn (A) | 512 | 75.4 % |
| P34 v1 REPLAY | 91000–3 | drawn (B) | 512 | 74.2 % |
| P34 v2 REPLAY | 91000–3 | drawn (C) | 512 | 73.8 % |
| P34 v2 REPEAT | 91000–3 | drawn (D) | 512 | 74.0 % |
| P34 v3 REPLAY | 91000–3 | **frozen** | 512 | 73.6 % |
| P34 v3 REPEAT | 91000–3 | **frozen** | 512 | 73.4 % |
| P34 v1/v2 FRESH | 77000–3 | drawn | 512 ×2 | 72.5 %, 72.5 % |
| **P34 v3 FRESH** | **77000–3** | **frozen** | **512** | **72.1 %** |

On the 91xxx batches the six measurements give **74.1 % ± 0.7**; on 77xxx, **72.4 %**. The
1.7-point difference is batch difficulty. **Chain-draw variation is worth about ±0.7 points** —
small next to P32's 12-point pose sd, which is why every earlier result survives it, but not
zero.

**The number to quote is 72.1 %** — never-used spawns, frozen chain, exactly what Stage 2 will
run. **Central estimate ≈ 73 %.** P33's 75.4 % was the top of the range and should not be the
headline.

That is still **+16 points over the 56.6 % candidate mean**, and it is the first expert number
in this effort that is attached to a *specific, reproducible artifact* rather than to a
configuration that re-draws its own manoeuvre on every run.

---

## 3. P27 — hold duration

*Pending. Prediction registered: flat from 560 down to ~120 physics steps, then a cliff.
Falsifier: a gradual decline, which would mean the close is settling the row rather than
closing the gripper.*

---

## 4. P29 — the `home → chain[0]` approach segment

*Pending. Prediction registered: the joint-space lerp is clean — penetration 0.0 mm, approach
hazard 0 %, within ±5 points of the teleport baseline.*

Also answers `09_STAGE2_BC_PLAN.md` N7 (the action-magnitude question) by measurement.

---

## 5. Q7 — throughput and VRAM. **The card is compute-bound, not memory-bound.**

*Run 2026-08-03, `runs/q7_N*.json` / `q7.log`. Eight env counts, one process each, zero
actions, 30-step warm-up discarded, 200 timed steps. Outstanding since Stage 0.*

| N | env-steps/s | batch-steps/s | s per 690-step episode-batch | **episodes/s** | peak alloc | `nvidia-smi` used |
|---|---|---|---|---|---|---|
| 16 | 469 | 29.3 | 23.5 | 0.7 | 0.1 MiB | 1 563 MiB |
| 64 | 1 833 | 28.7 | 24.1 | 2.7 | 0.1 | 1 563 |
| 128 | 3 462 | 27.1 | 25.5 | 5.0 | 0.2 | 1 563 |
| 256 | 5 841 | 22.8 | 30.2 | 8.5 | 0.4 | 1 629 |
| 512 | 10 528 | 20.6 | 33.6 | 15.2 | 0.7 | 1 759 |
| 1 024 | 20 343 | 19.9 | 34.7 | 29.5 | 1.4 | 1 955 |
| **2 048** | **34 458** | 16.8 | 41.0 | **50.0** | 2.8 | **2 407** |
| **4 096** | **49 231** | 12.0 | 57.4 | **71.4** | 5.5 | **3 229** |

**No OOM at any size.** 4 096 envs uses **3.2 GiB of 10 GiB** — and the torch-side allocation
is 5.5 MiB, i.e. essentially all of it is the Isaac Sim baseline (1 563 MiB at N = 16) plus
PhysX scene data. *Memory was never the constraint on this card;* the ceiling is compute.

**Three planning consequences, all of which change a number written elsewhere:**

1. **Demo generation is free.** 1 024 episodes (8 spawn seeds × 128 envs) is **~3.4 minutes**
   of simulator time at N = 128, or **20 seconds** at N = 2 048. Nothing about the dataset
   size needs to be rationed.
2. **Clutter is ~3× faster than the task eva_bc benchmarked on.** Upstream measured ~11 k
   env-steps/s at 2 048 envs for its steering run on a *12 GB* card; clutter does **34.5 k** at
   the same env count on a 10 GiB one. The scene is simply smaller — five blocks and an arm.
3. **`12_UPSTREAM_SYNC.md` §5 item 4 is corrected.** It speculated that if only 512 envs fit,
   matching upstream's 9.8 M window-transitions would cost ~15 h. **2 048 envs fit with 7.8 GiB
   to spare**, and the same transition budget is ~147 M env-steps ≈ **1.2 h of pure simulator
   time** (policy inference on top). Stage 4's budget is *not* hardware-limited here.

**Scaling.** Per-env throughput falls monotonically — 29.3 batch-steps/s at N = 16 to 12.0 at
N = 4 096 — so the GPU is saturating. The knee is around **N = 1 024–2 048**: doubling 1 024 →
2 048 buys 1.69× total throughput, and 2 048 → 4 096 only 1.43×. **N = 2 048 is the efficient
operating point**; N = 4 096 is available if wall-clock matters more than efficiency.

For the standing ≥128-episode eval requirement, **N = 128 costs 25.5 s** — cheap enough that
no experiment in this effort ever needed to economise on episodes. In hindsight that is a small
indictment of the P26 family: the power problem in §2e was never a budget problem.

---

## 6. CORRECTIONS MADE WHILE REVIEWING THE PIPELINE

Not probe results — defects found by reading, listed so they are not re-found later.

1. **`HANDOFF.md` §12 printed the wrong `clip_actions` key path.** It is
   `agent.params.env.clip_actions` (`eva_rl/scripts/rl_games/train.py:166`), not
   `agent.params.config.env.clip_actions`. **Fixed in place.**
2. **`residual_core.py:159` garbles a quaternion.** `subtract_frame_transforms` already returns
   XYZW in Isaac Lab 3.x, so the `[1,2,3,0]` permutation maps `(x,y,z,w) → (y,z,w,x)`. Confirmed
   still present; **must be dropped in the port**, not copied. (Already recorded in
   `05_PORTING_MAP.md` L2 — re-verified, not re-discovered.)
3. **The section-4.2 flush trigger is dead code for clutter.** `EventCfg` is reset-only —
   clutter has no mid-episode perturbations at all, unlike pick-place's nudges and friction
   randomisation. The porting map's proposal to redesign the trigger around `up_z` is
   **withdrawn**: it would build a detector for an event with no useful response, since a
   toppling distractor ends the episode. Default `--no-flush`, gated on on≡off.
4. **`ClutterExpert` gained `holds` and `env_steps`.** Additive, defaults identical to the
   Stage-1 behaviour, so no Stage-1 number changes. Needed because P27 sweeps hold durations
   and because the demo horizon has to be sized from the schedule rather than guessed.
