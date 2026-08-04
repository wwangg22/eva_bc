# STAGE 1 RESULTS — cutting the topple rate

**Opened 2026-08-02 with the Stage-0 champion at 25 % success / 75 % topple.**
Probes P12–P23. Read `HANDOFF.md` §1–3 first for the manoeuvre; this document is the
evidence for everything that changed since.

Companion: `07_STAGE0_RESULTS.md` (probes P01–P11).

---

## 0. HEADLINE

| | Stage 0 exit | mid Stage 1 | **Stage 1 current** | source |
|---|---|---|---|---|
| enclosed at close | 100 % | 100 % | **100 %** | P26 |
| target at goal | 99 % | 89.5 % | **~90 %** | P26 |
| topple | 75 % | 35.3 % | **~22 %** | P26 |
| **success** | **25 %** | 57.7 % | **73.7 %** | P26, 768 episodes |
| per-cell spread | not measured | 38.3–72.7 %, sd 13.6 % | **67.2–80.5 %, sd 4.8 %** | P26 |

**Gate 1's ≥85 % does not pass. The revised 70 % floor does**, so decision rule `DR2`
applies: proceed to Stage 2 with the BC ceiling recorded in advance. The expert is also
already above the **mission** target of ≈70 %, which the *policy* must reach.

Two distinct kinds of gain got it there:

1. **Five trajectory defects fixed** (§2), each isolated by a paired experiment: dense
   Cartesian pathing (+44.5 pts on its own), whole-arm clearance in the pose search, lift
   height, yaw matching, and deleting a retreat that was mine. 25 % → 57.7 %.
2. **Simulator-screened pose selection** (§6e–6f): +10.0 pts, and the pose-draw variance —
   which had been worth ~40 points and was invisible to every forward-kinematic statistic
   tried — collapses from sd 17.5 % to 4.8 %. 57.7 % → **73.7 %**.

The residual failure mode is identified and unfixed: **the finger blades sweep the
neighbouring blocks during the close, independently of the target block entirely** (§4.3).

---

## 1. WHY EVERY EXPERIMENT IN THIS STAGE IS PAIRED

The first thing Stage 1 measured was its own noise floor, and it was enormous. Identical
code, identical configuration, different CEM draw and spawn batch:

| run | CEM err | `o_align` | enclosed | at goal | success |
|---|---|---|---|---|---|
| P10 | 0.6–1.0 mm | ~0.99 | 100 % | 99 % | 25 % |
| P12 run a | 0.83 mm | 0.998 | 94 % | 9 % | **0 %** |
| P12 run b | 0.98 mm | 0.996 | 100 % | 100 % | **37.5 %** |
| P13 | 0.55 mm | **1.000** | **32 %** | 12 % | **0 %** |

The best and the worst are separated by 0.45 mm of position error and 0.002 of `o_align`.
**The selection criteria in use could not see the variable that decided the outcome.**

Consequence, adopted for the whole stage: one `env.reset()`, settle, snapshot every block's
`root_state_w`, then run each variant from the restored snapshot with the same drawn pose.
Differences between arms are then attributable to the arm. Absolute levels still carry
pose-draw uncertainty, which is why P21 pools over three independent draws.

---

## 2. THE FOUR FIXES, IN ORDER OF SIZE

### 2.1 Dense Cartesian pathing — **+44.5 points** (P17)

The largest fix in the stage, and it was a defect rather than a tuning parameter.

Every waypoint from P05 onward was verified. **No segment between waypoints ever was.**
Waypoints were solved by an independent CEM with restarts from uniform draws over the joint
limits, then executed by interpolating in *joint* space. That is a straight Cartesian line
only when both endpoints lie in the same IK branch.

P17's audit sampled each segment at 41 points and measured the achieved TCP against the
straight Cartesian line:

| segment | Cartesian length | max TCP deviation |
|---|---|---|
| app0→app1 | 23 mm | 8.8 mm |
| app1→app2 | 23 mm | 2.3 mm |
| app2→grip | 23 mm | 0.8 mm |
| grip→lift | 150 mm | 8.2 mm |
| lift→carry | 196 mm | 18.1 mm |
| **carry→place** | **150 mm** | **108.0 mm** ← branch flip |

`carry→place` is 150 mm straight down at the goal, 160 mm from the nearest distractor. In
Cartesian terms it cannot touch anything. Its measured **contact hazard was 63/63 and 20/20
— 100 %**. A hazard of exactly 1.0 is not a clearance problem or a spawn-dependent problem;
it is a deterministic geometric fact, and it had been in the trajectory since P10.

Fix: rebuild every segment as a Cartesian polyline at ≤30 mm spacing, each waypoint solved
by a CEM seeded from the previous solution with `std0 = 0.08` and **`restarts = 1`**, so the
solution cannot leave the branch it starts in.

| | worst deviation | success | topple |
|---|---|---|---|
| coarse chain | 108.0 mm | 18.0 % | 82.0 % |
| dense chain (23 waypoints) | **1.1 mm** | **62.5 %** | **37.5 %** |

Paired: 59 episodes fixed, 2 broken, net **+57 of 128**.

The retreat hazard (§2.4) disappeared at the same time and for the same reason — the dense
chain's penultimate waypoint is 25 mm above the goal rather than back over the row.

### 2.2 Whole-arm clearance in the pose search — **at goal 28 % → 100 %** (P15)

`o_align` and TCP error pin down the **tool frame**. They say nothing about the elbow, the
forearm, or the wrist stub. The geometry at `phi = 90` is forced:

```
o_hat = x_hat  ⇒  a_hat lies in the y–z plane  ⇒  gripper_end = TCP − 0.0419·a_hat

 a_hat = (0.03, −0.94, +0.33)   wrist at y = +39 mm, z = 51 mm   INSIDE distractor_2   (P13)
 a_hat = (0.10, +0.47, +0.88)   wrist at y = −20 mm, z = 29 mm   in the −y free gap
```

P13's pose scored `pos_err = 0.55 mm, o_align = 1.000` and drove the wrist straight into
`distractor_2`: 63 of 128 envs contacted during the **descent**, enclosure collapsed to 32 %.

Fix: `ArmKin.box_penetration` prices the deepest intrusion of any body origin into any
keep-out box, wired into the CEM cost (`avoid=`), and pose selection became **lexicographic
— penetration first, then `o_align`, under a hard 1.5 mm position gate**. A weighted sum
would let 0.003 of `o_align` buy a 10 mm intrusion, which is exactly the trade P13 made.

| | descent contacts | enclosed | at goal |
|---|---|---|---|
| align-only selection | 34–36 / 128 | 100 % | 28 % |
| clearance selection | **0 / 128** | 100 % | **100 %** |

The keep-out set must include **the target**, not only the four distractors. P14's
"row-clear" poses were all sitting inside the target's own volume and it never noticed.

### 2.3 Lift height and the carry sweep (P12, P13)

The block is gripped near its top and hangs a long way below the TCP:

```
grip at 65 mm, block centre settles at 32 mm, half-height 35 mm
  ⇒ the block's bottom face is 68 mm below the TCP
  ⇒ a 75 mm lift puts it at 72 mm, against distractor tops at 67 mm  →  5 mm
```

and `GOAL_XY = (0.185, −0.185)` puts the carry diagonally **over `distractor_0` and
`distractor_1`**. P12 measured 48.4 % of all topple onsets inside the `carry` phase, with
those two as the victims in every case. Raised to 150 mm.

P13's paired lift sweep was run on a broken pose (enclosure 32 %) and its numbers are not
usable; the fix is retained on the geometry and on P15/P17, where 150 mm is the shipped
value throughout.

### 2.4 Yaw matching and the gratuitous retreat (P16)

`refine` corrected **position only**. The reset draws `yaw ~ U(−0.20, +0.20)` rad, so the
jaw arrived square at a block that was not, and the first face contact turned the block
instead of gripping it — measured at **3.72° of rotation during the close**, against free
gaps whose median is 8.3 mm.

`ArmKin.refine` gained an `o_des` channel: the residual is stacked as `[tcp ; k_rot·o_hat]`
against `[pos ; k_rot·o_des]`, sign-corrected per env because a parallel jaw is symmetric.
Target rotation during the close fell **3.72° → 0.28°**, jaw-vs-block alignment 0.9917 →
1.0000.

The retreat was mine: after releasing at the goal the script drove back to a pose above the
row with an empty gripper. **47 of 128 topples started there**, hazard 98.4 %, for a motion
the task never asked for.

**Neither fix showed a measurable effect in P16's own 2×2** (both ≤0.8 points) because at
that point the `place` segment's 100 % hazard was masking everything downstream of it. They
became visible only after §2.1. This is why the yaw and retreat fixes are justified by their
mechanism measurements (3.72° → 0.28°; hazard 98.4 % → 0 %) rather than by P16's headline.

---

## 3. WHAT THE HAZARD RATE IS AND WHY RAW COUNTS LIED

Contact attribution counts each env's **first** contact only. Once an early phase is bad,
every later phase looks harmless. Converting to a hazard — contacts in a phase, divided by
the envs that reached it still clean — changed the reading completely:

| | raw count | hazard |
|---|---|---|
| P16 arm A `close` | 65 / 128 | 51 % |
| P16 arm A `place` | 63 | **100 %** (63 of the 63 that reached it) |
| P16 arm C `place` | 20 | **100 %** (20 of 20) |

The raw counts made `place` look like a minor third-place contributor. The hazard made it
the only thing worth looking at, and it was right.

---

## 4. THE CLOSE PHASE: THREE ATTEMPTS, TWO WRONG

After §2 the trajectory had exactly one hazard left — `close`, at 48–52 %.

### 4.1 First attempt: the wrist (P18) — **wrong**

The victim was strongly asymmetric: `d1` (y = −42) in 76–90 of 128 envs across four arms,
`d3` never. The wrist sits in the −y gap at y = −20 mm, 7 mm from `d1`'s face, and is the
nearest body to it for the whole grasp. P18 ran a mirror control — same accuracy, wrist in
the **+y** gap — and the victim flipped from `d1` 51 / `d2` 31 to `d1` 28 / **`d2` 100**.

Reported as causal. **It was confounded.** The mirror pose differed in two variables at
once: wrist side *and* jaw alignment (`o_align` 0.885 against 0.995). The alignment is what
moved the victim, and the relationship holds across every arm measured:

| `o_align` | close hazard | source |
|---|---|---|
| 0.991 | 52 % | P19 A |
| 0.978 | 84 % | P20 A |
| 0.885 | 97 % | P18 B |
| 0.848 | 92 % | P19 D |

### 4.2 Second attempt: the closing impulse (P19/P20) — **partly wrong**

P19 traced the close at full physics-step resolution (2.5 ms). Two clean measurements:

- **The wrist travels 0.13 mm median, 0.21 mm max through the entire slam.** It is not
  lurching into anything. That alone should have retired §4.1.
- Ordering: jaw reaches 60 mm at step 3, `d1` first moves at step **4–5**, target at step 8.
  Slowing the jaw moved `d1`'s first motion to step 54.

Driving the finger joint target smoothly shut over 100 steps scored **97.7 % success, 2.3 %
topple**. But it is **not a legal action** — `ActionsCfg.gripper` is a
`BinaryJointPositionActionCfg` with no intermediate aperture — so it cannot appear in a
demonstration. P20 measured the legal approximations:

| arm | legal | success | topple | close hazard |
|---|---|---|---|---|
| grip 65 mm, binary | yes | 39.8 % | 43.0 % | 83.6 % |
| **grip 55 mm, binary** | **yes** | **82.0 %** | **13.3 %** | 55.5 % |
| grip 50 mm, binary | yes | 27.3 % | 53.1 % | 68.0 % |
| grip 65 mm, duty-cycled 1:1 | yes | 29.7 % | 58.6 % | 96.1 % |
| grip 65 mm, duty-cycled 1:2 | yes | 24.2 % | 64.1 % | 96.9 % |
| grip 65 mm, ramped | **no** | 71.9 % | 21.1 % | 79.7 % |

Duty-cycling the binary command is legal and made things **worse**. Grip height was the real
lever, and the story attached to it — "a strike 2 mm below the top face tips a 70 mm block,
a strike 23 mm down does not" — was about the *target*, which §4.3 shows is irrelevant.

The 82.0 % was a single favourable cell. P21 pooled 768 episodes at the same 55 mm and got
**57.7 %**.

### 4.3 Third attempt: the finger blades (P22) — **the control that settled it**

Teleport the target 2 m below the table, leave the distractors where they spawned, put the
arm in the identical grasp pose, close on empty air:

| arm | neighbour moved >1.5 mm | toppled |
|---|---|---|
| A target present, normal close | 74.2 % | 19.5 % |
| **B target REMOVED, same close** | **75.8 %** | **21.9 %** |
| D target present, jaw stays OPEN | **0.0 %** | **0.0 %** |

**The target is a bystander.** Removing it changes nothing. Holding the jaw open changes
everything. The finger blades sweep the neighbours directly, during the close.

The collision meshes — read at last through `Usd.TraverseInstanceProxies`, 17 298 points per
finger, the measurement missing since P01's retracted AABB:

```
gripper_left    x −19.2 .. +19.2 | y −41.9 .. +46.7 | z −58.7 .. +34.7   [mm, body frame]
gripper_right   x −19.2 .. +19.2 | y −46.7 .. +41.9 | z −58.7 .. +34.7
gripper_end     x −157.2 .. −73.2 | y −92.0 .. +92.0 | z −41.0 .. +40.9
```

The blades are ~88 mm across the opening axis. With the jaw open the finger origins sit at
x ≈ 205 and 295 mm — **outside** the row's 232–268 mm x-band, which is exactly why the
descent hazard has been 0 % since P15. Closing drags them 26.5 mm each **into** that band,
where the neighbours live at y = ±42 mm.

This is a different problem from every one solved so far. The previous fixes were about
where the arm *travels*; this is about the volume the gripper *sweeps while actuating*, and
no waypoint or path change touches it.

---

## 5. SUCCESS RISES AS THE GAP SHRINKS — and it is not a fluke

P21, 768 episodes, stratified by the per-episode minimum free gap:

| min free gap | n | topple | success |
|---|---|---|---|
| 0–4 mm | 17 | 17.6 % | **82.4 %** |
| 4–6 mm | 88 | 18.2 % | **77.3 %** |
| 6–8 mm | 248 | 37.1 % | 53.6 % |
| 8–10 mm | 309 | 39.8 % | 52.1 % |
| 10–12 mm | 92 | 33.7 % | 64.1 % |
| 12+ mm | 14 | 42.9 % | 57.1 % |

This is backwards from the Stage-0 reading, where success rose monotonically with clearance
(21.4 % at 0–8 mm up to 55.6 % at 12–14 mm). Both cannot be right, and the Stage-0 version
was measured through a trajectory that has since been shown to contain a 100 %-hazard
segment and a wrist inside a neighbour.

The plausible mechanism, consistent with §4.3: blocks are 70 mm tall and 30 mm wide, with
`h_crit = 15.8 mm` across the row. **A tightly packed row supports itself.** A nudged block
with 2 mm of room leans on its neighbour after 2 mm of top travel and stops; the same nudge
with 10 mm of room lets it accelerate past the 41.4° termination. Toppling needs somewhere
to fall.

This is a hypothesis fitted to 768 episodes after the fact, not a designed test, and it is
recorded as such. It also predicts that **`Tight-v0` (6 mm pitch) may be *easier* than
nominal**, which sharpens the Stage-0 prediction ("within 10 points") into something
falsifiable in the direction of a gain.

---

## 6. WHERE THIS LEAVES THE EXPERT

`clutter/expert/clutter_expert.py` now holds the whole recipe, with every constant carrying
the probe that set it. Its shape:

```
plan()   one CEM chain: clearance-gated grasp pose (o_align ≥ 0.99, penetration 0),
         then a dense Cartesian polyline outward in both directions, local solves only
adapt()  per-env DLS refine to the actual target position AND yaw, through the grasp
         and the lift
run_physics()  physics-only execution for measurement
schedule()     the single source of truth for the motion, so the physics-only evaluator
               and the (not yet written) env.step demo recorder are checkably equivalent
```

**Open, and blocking Gate 1: the finger-blade sweep (§4.3).** Everything else in the
trajectory is now at or near zero hazard:

| phase | hazard, P21 pooled |
|---|---|
| settle / descend / predwell | 0–1 % |
| **close** | **64–81 %** |
| carry | 50–89 % (of a small, already-selected remainder) |
| dwell / release / withdraw / final | 0 % |

---

## 6b. A PROBE BUG CAUGHT BY ITS OWN NONSENSE (P23, first run)

Recorded because the *detection* method is the reusable part.

P23's first run swept grip height on the isolated close, looping `plan → adapt → restore`.
`adapt()` reads the target's live position, and `restore()` (with the target dropped, P22's
control) ran at the *end* of each cell — so every cell after the first adapted its chain to
a target 2 m below the table.

The output said so plainly. The dominant victim was **`distractor_3`, at y = +84 mm**, in
62–80 % of envs, while `d2` at y = +42 mm was hit **0 %** of the time. No correct grasp at
`x = 250, |y| < 5` can reach past the near neighbour to topple the far one. The `o_align`
column also went incoherent — 0.8988 and 0.9955 next to each other at adjacent heights.

The full-trajectory confirmation in the same run was *not* affected (it calls `env.reset()`,
which restores the target) and is the one usable result from it:

| | enclosed | at goal | topple | success |
|---|---|---|---|---|
| grip 40 mm, batch 0 | 74.2 % | 51.6 % | 99.2 % | 0.0 % |
| grip 40 mm, batch 1 | 95.3 % | 71.1 % | 42.2 % | 39.8 % |
| **grip 55 mm, batch 0** | 100.0 % | 94.5 % | 21.9 % | **74.2 %** |
| **grip 55 mm, batch 1** | 100.0 % | 89.1 % | 18.0 % | **74.2 %** |

Two batches at 74.2 % on a draw with `o_align = 0.9988`, against P21's pooled 57.7 %. That
is consistent with alignment mattering a great deal and is *not* claimed as a result — it is
two cells, and P20's 82 % looked just as good before pooling flattened it.

Method note, now a convention: **a result that is geometrically impossible is a bug report,
not a finding.** The d3-without-d2 pattern could not be produced by any grasp, and that was
visible before any interpretation was attempted.

---

## 6c. GRIP HEIGHT IS NEARLY FLAT — and the real variable was hiding in the per-draw rows

P23, re-run after the §6b fix: nine heights × three pose draws, isolated close.

| grip z [mm] | isolated topple | spread over the three draws |
|---|---|---|
| 40 | 30.5 % | 67 %, 6 %, 18 % |
| 45 | 19.5 % | 11 %, 20 %, 28 % |
| 50 | 20.8 % | 16 %, 13 %, 33 % |
| 55 | 31.5 % | 23 %, 46 %, 25 % |
| 60 | 27.6 % | 23 %, 22 %, 38 % |
| 65 | 31.5 % | 30 %, 34 %, 30 % |
| 70 | 21.9 % | 10 %, 35 %, 20 % |
| 75 | 19.8 % | 32 %, 12 %, 15 % |
| 80 | **11.2 %** | **4 %, 27 %, 2 %** |

**The within-height spread is as large as the between-height variation.** P20's three-point
sample (65 → 43 %, 55 → 13 %, 50 → 53 %) was mostly pose-draw noise around a weak effect.
Retraction 3 in §7 is accordingly strengthened: it is not only the *explanation* for the
55 mm choice that was wrong, the *effect* is much smaller than reported.

End-to-end, 55 mm and 80 mm are indistinguishable:

| | enclosed | at goal | topple | success |
|---|---|---|---|---|
| grip 55 mm | 96.9 / 100 % | 89.8 / 89.8 % | 29.7 / 22.7 % | 68.8 / 71.9 % |
| grip 80 mm | 100 / 100 % | 96.9 / 93.8 % | 28.1 / 25.8 % | 68.8 / 68.0 % |

### The signal was in the rows, not the aggregate

```
grip 80 mm, draw 0   wrist at (−41, +72) mm   isolated topple   3.9 %
grip 80 mm, draw 2   wrist at (−42, +73) mm   isolated topple   2.3 %
grip 80 mm, draw 1   wrist at (−20, +44) mm   isolated topple  27.3 %
```

Block tops sit at ~67 mm. Two of those draws put `gripper_end` **above the row**; the third
left it in a 12 mm gap. Same height, same gates, an order of magnitude apart.

**This was believed impossible.** P14 concluded the wrist must thread a gap, from

```
z_wrist = grip_z − 41.9·cos(t) > 67 mm   requires   cos(t) < −0.05   at grip_z = 65 mm
```

— a downward approach axis, genuinely unattainable (`a_align = −0.11`). The algebra is
right; **the conclusion was over-generalised from a single grip height.** At `grip_z = 80 mm`
the same expression needs only `cos(t) < 0.31`, comfortably inside the attainable set. The
CEM reaches it unprompted in two draws of three.

So the wrist's height is not a fixed consequence of the geometry — it is **selectable**, and
unlike grip height it separates the cells cleanly. `ClutterExpert.wrist_min_z` now gates on
it, and P25 tests it properly: three arms × three pose draws × two spawn batches, pooled,
with a within-arm mechanism check that pools every cell by measured `wrist_z` regardless of
which arm produced it.

---

## 6d. THE WRIST-ABOVE-ROW HYPOTHESIS, FALSIFIED (P25) — and what the failure taught

P25 tested §6c's observation properly: three arms × three pose draws × two spawn batches,
768 episodes each, with a mechanism check that pools every cell from every arm by measured
`wrist_z` regardless of which arm produced it.

| arm | pooled | spread | sd |
|---|---|---|---|
| A grip 55, no gate | **53.3 %** | 32.8 – 74.2 % | 17.5 % |
| B grip 80, no gate | 40.1 % | 2.3 – 66.4 % | 30.0 % |
| C grip 80, wrist ≥ 70 mm | 24.9 % | 0.0 – 89.8 % | 37.7 % |

| pooled by measured wrist height | cells | success | topple |
|---|---|---|---|
| wrist **above** the row (>67 mm) | 6 | **2.0 %** | 43.2 % |
| wrist in a gap (≤67 mm) | 12 | **58.1 %** | 36.8 % |

**−56.2 points — the opposite of the prediction, and unambiguous.**

The reason is a hard geometric coupling. Lifting the wrist above the row requires
`cos(t) < 0.31`, i.e. an approach axis that is nearly **horizontal**. The finger blades
extend from `gripper_end` along that axis, so they end up lying flat instead of straddling
the block: **enclosure collapses to 19–27 %.**

And that is why the isolated-close proxy loved those poses. It scored *disturbance only*.
A jaw that never engages the block disturbs nothing — 2–4 % isolated topple, 2.0 % actual
success. **A proxy metric that omits the success condition will select precisely the
candidates that fail it.** The score is now `enclosed ∧ ¬toppled`.

P14's conclusion — the wrist must thread a gap — is **restored** for any pose that can
actually grasp, and retraction 6 in §7 is narrowed accordingly.

---

## 6e. THE REMAINING VARIANCE IS THE POSE DRAW, AND IT IS NOT PREDICTABLE FROM FK (P26)

Every trajectory defect is fixed and the expert sits at 53–58 % pooled. What is left is not
a failure mode — it is which pose the CEM happened to draw:

| P25 arm A, one configuration | batch 0 | batch 1 | `o_align` | wrist z |
|---|---|---|---|---|
| draw 0 | 68.0 % | 62.5 % | 0.9925 | 18 mm |
| draw 1 | **71.9 %** | **74.2 %** | 0.9916 | 20 mm |
| draw 2 | **39.1 %** | **32.8 %** | 0.9919 | 18 mm |

The good draw and the bad draw agree on `o_align` to three decimals, both have zero keep-out
penetration, and their wrists differ by 2 mm. **The selector is blind, and the spread it
cannot see is worth ~40 points — more than every fix in Stage 1 put together.**

Three attempts at a predictive forward-kinematic statistic have now failed:

| statistic | verdict | probe |
|---|---|---|
| `o_align` | necessary, nowhere near sufficient | P18 |
| wrist side (−y vs +y gap) | confounded, withdrawn | P18 |
| wrist height | actively **anti**-correlated, −56.2 points | P25 |

So P26 stops predicting and measures. The founding rule of this effort is that candidates are
scored by values **read back from the sim** rather than commanded — that is why the CEM scores
achieved FK instead of trusting an IK solver. It simply was not carried far enough: achieved
FK is still a *proxy* for whether the grasp works, and the thing itself is cheap to measure.
`ClutterExpert(screen=K)` solves K candidates, runs each one's close in the simulator, and
keeps the best on `enclosed ∧ ¬toppled`.

Screening runs on one spawn batch; every reported number comes from different batches, and the
screen score is printed next to the held-out score so any optimism is visible rather than
assumed away.

---

## 6f. RESULT: SCREENING WORKS, AND MORE OF IT OVERFITS (P26)

Three independent selections × two **held-out** spawn batches per arm, 768 episodes each.

| arm | pooled | spread | sd | screen scores |
|---|---|---|---|---|
| `screen = 0` | 63.7 % | 58.6 – 68.8 % | 4.1 % | — |
| **`screen = 4`** | **73.7 %** | 67.2 – 80.5 % | 4.8 % | 79 %, 84 %, 88 % |
| `screen = 8` | 67.2 % | 52.3 – 78.9 % | **10.2 %** | 84 %, 85 %, 91 % |

**+10.0 points from four candidates, and the pose-draw variance largely disappears** — the
40-point spread of §6e becomes 13 points.

**Eight candidates is worse than four.** The screen scores go *up* (84–91 % against 79–88 %)
while the held-out result goes *down*, and the optimism gap widens from **+9.9 to +19.3
points**. With a single 128-env batch as the selection sample, eight candidates is enough for
the winner to be one that got lucky on that batch. This is ordinary selection overfitting and
it is visible only because the screen score was reported next to the held-out score.

The fix is *not* fewer candidates. It is **more screening spawns per candidate** — average
each candidate over 2–3 batches so the selection statistic stops being noise-dominated, then
raise the candidate count again. Registered as the next experiment.

Confirmation that FK statistics really are blind, from the screen logs themselves:

| candidate | `o_align` | wrist z | screen score |
|---|---|---|---|
| best in its round | 0.9193 | 18.3 mm | **82.0 %** |
| worst in its round | 0.9968 | 18.9 mm | **61.7 %** |

The candidate with the *worst* alignment beat the one with the best by 20 points.

Stratified by minimum free gap (`screen = 4`, 768 episodes) — the tight end is now the
**strongest**, reinforcing §5:

| min free gap | n | success |
|---|---|---|
| 0–4 mm | 30 | **90.0 %** |
| 4–6 mm | 82 | **91.5 %** |
| 6–8 mm | 233 | 64.4 % |
| 8–10 mm | 303 | 71.0 % |
| 10–14 mm | 120 | 82.5 % |

---

## 7. RETRACTIONS ISSUED IN THIS STAGE

1. **"The wrist stub is the culprit" (P18).** Confounded control — the mirror pose changed
   jaw alignment as well as wrist side, and alignment is what moved the victim. The wrist
   travels 0.13 mm through the whole slam (P19). Withdrawn.
2. **"Fingers strike the target, the target strikes its neighbour" (P19/P20).** Removing the
   target changes nothing (P22). Withdrawn. P19's *ordering* measurement — neighbour at step
   4, target at step 8 — was correct and contradicted the story it was attached to; it was
   written down and not acted on.
3. **"Grip at 55 mm" (P20) — the reason AND the effect.** The stated mechanism concerned the
   target, which §4.3 shows is a bystander. And §6c's clean nine-height sweep at three draws
   each shows the effect is **far weaker than reported**: 19.5–31.5 % across 40–80 mm, with
   a within-height spread over pose draws (6–67 %) as large as the variation between
   heights. P20's three points were mostly noise. 55 mm and 80 mm are indistinguishable end
   to end (68.8 / 71.9 % vs 68.8 / 68.0 %). **Withdrawn as a tuned optimum**; retained only
   as a value that works.
6. **P14's "the wrist must thread a gap"** — retracted in §6c and then **restored** in §6d.
   The algebra was over-generalised from one grip height (at 80 mm the wrist *can* clear the
   row), but every pose that clears it has a near-horizontal approach axis and cannot grasp:
   enclosure 19–27 %, success 2.0 %. **The conclusion stands for any usable pose.** Recorded
   as a round trip rather than quietly reverted, because the intermediate claim was published
   in this document before P25 ran.
4. **P13's lift-height comparison.** Run on a pose with 32 % enclosure; the arms are not
   comparable. The 150 mm lift is retained on geometry and on P15/P17.
5. **Stage 0's "success rises monotonically with clearance".** Superseded by §5, on 768
   episodes through a trajectory without the defects.

---

## 7b. A PREDICTION REGISTERED BEFORE THE MEASUREMENT (P24)

Pre-registered here so the outcome cannot be rationalised afterwards, per the standing
convention on pre-registration.

P22's mesh read gives the blade geometry for the first time, and it makes a **quantitative**
claim about the yaw fix that §2.4 shipped:

```
perpendicular to the opening axis   blade half-width  19.2 mm
neighbour's near face                                 27.0 mm
                                    margin             7.8 mm

along the opening axis              blade reach       ~47   mm
full spawn yaw                                        11.4 deg
corner swing when the jaw is rotated to match   47 * sin(11.4°) =  9.3 mm   >  7.8 mm
```

**Prediction: matching the target's yaw improves the grip and makes the blade sweep worse,
and at full gain it is a net negative.** Two existing measurements already fit and neither
was recognised as such at the time:

- **P16:** matching yaw raised close-phase contacts from **65/128 to 93/128**, while
  improving every grip statistic (block turn during the close 3.72° → 0.28°). Recorded as
  an oddity.
- **P19:** the **square** jaw outscored the matched one end to end, **69.5 % vs 64.1 %**.
  Dismissed as noise.

P24 sweeps `yaw_gain ∈ {0, 0.5, 1}` × `phi ∈ {90°, 80°, 70°}` on the isolated close, three
pose draws per cell, then confirms the winner end to end. `yaw_gain` rotates the jaw that
fraction of the way from row-square to block-square, so the trade is a dial rather than an
assumption.

Falsifier: if `gain = 1.0` is not worse than `gain = 0.0` at `phi = 90`, the geometry
argument above is wrong and the yaw fix keeps its current justification.

---

## 8. CONVENTIONS EARNED IN STAGE 1

Added to the list in `HANDOFF.md` §9:

- **Pair everything.** One spawn, snapshotted and restored, one pose draw shared across
  arms. Unpaired comparisons at this noise level (sd 13.6 %) are unreadable.
- **Report hazard rates, not raw counts.** First-contact counts systematically understate
  every phase after a bad one.
- **Verify segments, not only waypoints.** A verified endpoint pair says nothing about the
  joint-space line between them. Audit the achieved TCP against the Cartesian line.
- **A control with two variables in it is not a control.** P18 changed wrist side and jaw
  alignment together and produced a confident wrong answer that survived two probes.
- **When a measurement contradicts the story, the story is wrong.** P19 measured 0.13 mm of
  wrist travel and neighbour-before-target ordering, then reported a mechanism inconsistent
  with both.
- **Remove the object to test whether the object matters.** P22 cost one probe and retired
  two mechanisms.
