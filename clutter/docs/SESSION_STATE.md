# SESSION STATE — resume here

**2026-08-04.** The environment is now finished. The expert is not.

## In one paragraph

Two changes landed. **2026-08-03:** success requires every neighbour to stay within **2 mm** of
its spawn, enforced by the environment itself. **2026-08-04:** the row spawns at a **random
heading** and the target is **any one of the five blocks**, not always the middle
(`eva_rl@d329ff3`). The frozen `pose_p33` expert scores **3.0 %** on the new task and **17.1 %**
on `-Fixed-v0`, the frozen-row control. Both env changes are validated — smoke test passes, P41
found no reachability wall, and `-Fixed-v0` reproduces the pre-refactor baseline within the
binomial CI. **The environment work is done. Everything from here is the expert.**

## Read

1. **`HANDOFF.md`** — ⭐ the full record. **§10 first**, then the rest.
2. **`17_ROW_RANDOMISATION.md`** — ⭐ the row randomisation: what changed, how it was validated,
   and the three things it opened up
3. **`REFERENCE.md`** — everything durable and metric-independent, and the commands
4. **`16_DISTURBANCE_ANATOMY.md`** — why the old expert fails, measured four ways
5. `15_STRICT_METRIC.md` — what the criterion is and why it is trustworthy

## The next action

**Task #26 — re-run P41 part B on jittered spawns.** It comes before P40 and may retire it.

P41 part B closed the jaw *in situ* at a per-cell solved pre-grasp and the neighbours moved
**0.8–1.6 mm median, p90 2.75 mm** — against the frozen expert's **4.8 mm median, 44 mm p90**
(P38). If that holds up, the 53-point deficit is **the pose**, not the in-situ close, and the
extract-then-grasp plan is unnecessary.

**It may well not hold up, and the reason is specific:** P41 spawned with *no per-block jitter*,
so every free gap was exactly 12 mm, while real spawns run 2.6–20.9 mm with a median near 8 —
and P36 measured that disturbed episodes have a 9.37 mm median gap against 12.93 mm for clean
ones. The test flattered the pose exactly where the expert actually fails. Settle it before
building anything on it.

Then, in order: P40 (the drag gate, `HANDOFF.md` §6.2) → design the manoeuvre → new pose family
→ measure → only then demos and BC. **Do not train**: ~50 min/seed and the ceiling is the
expert's rate.

## What is known about the new task

| | |
|---|---|
| **baseline to beat** | **3.0 %** (frozen expert, 768 eps) — *not* 16.4 %, that task is gone |
| **report per slot** | the heading is an isometry; the **slot** is not — an end slot has one adjacent neighbour, not two |
| **no reachability wall** | 25/27 cells solve; both worst corners at r = 0.3087 m, `pos_err 0.00`, `o_align 1.000` |
| **one pose family is not enough** | continuation from the row centre misses slot 0 at positive yaw *on position* by 1.5–5.9 mm; a global search finds a different branch there |
| **wrist band** | every accepted pose 17.2–23.8 mm; every pose that shoved the row ≥ 28 mm |
| failure mode, unchanged | `distractor_disturbed` **97.0 %**, `time_out` **0.0 %** — it reaches the row, it cannot aim |

## The four things that killed the old expert

Under the 2 mm rule the taxonomy has **one** non-zero bucket.

| | |
|---|---|
| **the arm is innocent** | same trajectory, gripper forced open → **0.0 %** disturbance |
| **it is the first close step** | p01 = 160, and step 160 *is* that step |
| **hooked and carried, not shoved** | **\|dx\| is 9.2× \|dy\|**; inner pair 100 %; d1 : d2 = **2.50 : 1** |
| **the jaw is too wide** | fouling reaches **33–39 mm** vs faces at 27 mm; 90 mm opening for a 36 mm block |

Dead ends, ruled out on evidence: a slower close, a shorter close hold, grip height.
Measured but small: jaw yaw-matching, **+2.9 pts**. Measured and deliberately declined:
narrowing the gripper, **+16.9 pts**.

## Constraints

- eva_bc work in `eva_bc/clutter/`; eva_rl's `challenge/` editable **for the clutter env only**.
- **Pull, validate, update docs, commit, push.** Both repos, every change.
- **The 90 mm gripper stays.** One GPU job at a time. `python -u`.
