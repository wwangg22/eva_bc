# SESSION STATE — resume here

**2026-08-03.** The task got harder, the old work was retired, and the expert is being started
over.

## In one paragraph

The success criterion now requires every neighbour to stay within **2 mm** of its spawn — it is
enforced by the environment itself (`eva_rl@ceeb24c`), not by the evaluator. The frozen expert
scores **16.4 %** under it, against 73.3 % under the old topple-only rule. Big Will decided to
**keep the 90 mm gripper** rather than narrow it, even though narrowing is worth 17 measured
points: the harder task is the more interesting one. Twelve stage-result documents and the
Stage-0/1 probes were deleted, because every success number in them describes a task that no
longer exists. **`REFERENCE.md` is what was kept.** Next action: **P40**.

## Read

1. **`REFERENCE.md`** — everything durable, and the commands
2. **`HANDOFF.md`** — the task, the plan, what is on disk
3. **`16_DISTURBANCE_ANATOMY.md`** — why the expert fails and what to do about it
4. `15_STRICT_METRIC.md` — what the criterion is now and why it is trustworthy

## The next action

**P40 — measure the drag step alone.** Reach in with the jaw open, pull the target −x by
30–40 mm, and watch the neighbours.

It is the cheapest decisive experiment available, because two things are already measured:

- an **open jaw disturbs the row 0.0 %** anywhere in it (384 episodes, p90 0.311 mm), and
- **`DISTURB_TOL` binds the four distractors and says nothing about the target.**

So if dragging the target out is also near 0 %, the whole manoeuvre — reach in open, pull the
target clear, close on it *outside* the row, carry — is viable, and the rest is engineering.
If dragging rakes the row, that is dead and the plan needs rethinking from §4 of `HANDOFF.md`.

Nothing else should start before it reports. In particular **do not train**: cloning costs
~50 min/seed and the ceiling is the expert's rate.

## The four things that killed the old expert

Under the 2 mm rule the taxonomy has **one** non-zero bucket — `distractor_disturbed` 83.6 %,
everything else **0.0 %**, including topple.

| | |
|---|---|
| **the arm is innocent** | same trajectory, gripper forced open → **0.0 %** disturbance |
| **it is the first close step** | p01 = 160, and step 160 *is* that step. Already 4.05 mm when it first exceeds 2 |
| **hooked and carried, not shoved** | **\|dx\| is 9.2× \|dy\|**; inner pair 100 %; d1 : d2 = **2.50 : 1** |
| **the jaw is too wide** | fouling reaches **33–39 mm** vs faces at 27 mm; 90 mm opening for a 36 mm block |

Dead ends, ruled out on evidence: a slower close, a shorter close hold, grip height.
Measured but small: jaw yaw-matching, **+2.9 pts**. Measured and deliberately declined:
narrowing the gripper, **+16.9 pts**.

## Constraints

- eva_bc work in `eva_bc/clutter/`; eva_rl's `challenge/` editable **for the clutter env only**.
- **Pull, validate, update docs, commit, push.** Both repos, every change.
- **The 90 mm gripper stays.** One GPU job at a time. `python -u`.
