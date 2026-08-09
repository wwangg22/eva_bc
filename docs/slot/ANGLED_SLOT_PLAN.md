# ANGLED SLOT — plan for making `Rebot-PrecisionSlot-v0` genuinely hard

*Written 2026-08-09, **before any code**. Big Will: "modify the exact env, dont make a new one…
We are now only having the policy really move the wedge into a place that is orthogonal to the
table. What if this wedge was angled arbitrarily? this would be a lot harder… we would need NEW
expert, new demos, etc. After implementing this new env and VERIFYING it works, please delete the
old data collection etc, but not the markdown files."*

---

## 0. The one thing to flag before anything else

**This edits `eva_rl`.** Every prior session held "`eva_rl` is a shared asset repo — never edit
it" as a hard rule, and all task-side work was contorted to route around it (post-parse patching
for `--slot-dx`, `--arm-jitter`, cameras). Big Will has explicitly asked for the env itself to
change, so that rule is now lifted **for this change**. Consequences, recorded so nobody has to
rediscover them:

* **Every number measured on the old task becomes a number about a former task.** The champion's
  0.979, all of `EXP_ROBUSTNESS`, all of `EXP_STEER`, the vision 0.804 and the DAgger −31.3 — none
  of them describe the new env. The markdown stays (Big Will's instruction) but each headline
  needs "measured on the **axis-aligned** slot" attached to it.
* **Other work may depend on this env.** `eva_rl` is shared with the `clutter/` effort and
  whatever else uses `challenge/`. A yaw field added to `mdp/common.py` must default to 0 so an
  unmodified caller sees exactly today's behaviour.

---

## 0b. SCOPING — Big Will: "only touch your assigned task (precision slot), and not any other task"

`challenge/mdp/` is **shared by four tasks**: `clutter`, `drawer`, `pregrasp` and
`precision_slot`. Editing it carelessly breaks three efforts that are not mine. Measured blast
radius (grep over the three sibling env cfgs):

| symbol | sibling tasks using it | verdict |
|---|---|---|
| `SLOT_CENTER`, `SLOT_DEPTH`, `SLOT_FLOOR_Z`, `WALL_HEIGHT`, `WALL_THICKNESS` | **0** | slot-only — **safe to change** |
| `SUCCESS_DEPTH`, `SUCCESS_YAW`, `slot_clearance_to_halfwidth` | **0** | slot-only — **safe** |
| `insertion_depth`, `lateral_error`, `yaw_error`, `is_inserted`, `slot_frame` | **0** | slot-only — **safe** |
| `BLOCK_HALF` | **2** | ⚠ **SHARED — do not change.** The block geometry stays exactly as it is. |
| `block_lifted`, `block_dropped`, `block_pose_in_root` | **3** | ⚠ **SHARED — do not touch.** |
| `object_pos_local`, `object_quat`, `yaw_of` | 0 direct, but used by the shared three | ⚠ **read-only.** May be *called*, never modified. |

**Rules this imposes on the implementation:**

1. Only `SLOT_*` constants and the five slot-only predicates may be edited.
2. The per-env `slot_yaw` is **new state**, added additively; nothing existing changes shape.
3. `slot_yaw` **must default to zeros**, so `clutter` / `drawer` / `pregrasp` — which never set it
   — see byte-identical behaviour. Gate A.1's bit-exactness check at θ = 0 covers this.
4. `precision_slot_env_cfg.py` is mine alone; the three sibling env cfgs are not touched at all.
5. A sibling smoke test runs before the work is called done: each of the other three tasks must
   still build and step.

## 1. What "angled arbitrarily" means, concretely

Today the slot is welded axis-aligned: `SLOT_CENTER = (0.245, 0.0)`, opening toward −x, walls
parallel to x. Three functions in `mdp/common.py` hard-code that assumption:

```python
insertion_depth = block_x - (SLOT_CENTER[0] - SLOT_DEPTH/2)     # assumes the axis is +x
lateral_error   = |block_y - SLOT_CENTER[1]|                    # assumes cross-axis is y
yaw_error       = |yaw_of(block_quat)|                          # assumes slot yaw = 0
```

**Proposal: a per-episode slot yaw θ about the table normal.** The slot keeps its footprint on
the table but points in a random direction; the block must be carried in along that direction and
squared to it.

| axis | feasible? | recommendation |
|---|---|---|
| **yaw θ** (in the table plane) | yes — the arm's wrist can rotate and the approach stays horizontal | **do this** |
| pitch (slot tilted out of plane) | **doubtful** — the arm has 0.00 % top-down capability below z = 0.19 m (`rebot-arm-no-topdown-grasp`), so a slot pitched upward may be unreachable | defer; escalate only after yaw works |
| slot position (x, y) | already partly explored — `EXP_ROBUSTNESS` §7/§8 moved it ±20 mm | fold in later as a second axis |

Start with **θ ∈ U(−θmax, +θmax)**, with θmax **set by measurement, not by choice** — see gate B.
My prior is θmax ≈ 30–40° before the arm envelope or the wrist limit binds.

**Why this is the right difficulty increase, and not just a harder number.** The single biggest
caveat on the vision result is that *the slot never moves* — which is why a **blind** policy still
scored 0.254 by aiming at the mean spawn. Randomising θ removes that: a policy that cannot
perceive the slot's orientation cannot insert. **Expected effect: blind → near 0.0**, and the
vision number becomes a clean measure of perception rather than of geometry.

---

## 2. What has to change

### 2a. `eva_rl` — `challenge/mdp/common.py`

* Add a **per-env slot pose**, not a module constant: `env.slot_yaw` (num_envs,) written at reset,
  defaulting to zeros. Module-level `SLOT_CENTER` stays as the centre.
* Rewrite the three predicates in the **slot frame**. With `c = cos θ`, `s = sin θ` and
  `d = block_xy − SLOT_CENTER`:

  ```
  along   =  d.x*c + d.y*s          # depth axis
  across  = -d.x*s + d.y*c          # cross axis
  insertion_depth = along - SLOT_DEPTH/2
  lateral_error   = |across|
  yaw_error       = |wrap(yaw_of(block_quat) - θ)|
  ```

  Note `wrap` to (−π, π] — a naive subtraction breaks at the branch cut and would score a
  perfectly aligned block as maximally misaligned near ±π.
* `is_inserted` needs no change once the three feed it slot-frame values.
* **Default θ = 0 must reproduce today's arithmetic exactly.** That is gate A's first check.

### 2b. `eva_rl` — `precision_slot_env_cfg.py`

The fixture is four **static** `AssetBaseCfg` boxes; static prims have no root pose to write per
reset. Options:

1. **One kinematic rigid body containing all four boxes** — a single prim whose pose is written at
   reset, so one write rotates the whole fixture and the parts cannot drift apart. **Preferred.**
2. Four kinematic rigid bodies posed individually — four times the writes and four chances to
   desynchronise.
3. Re-spawn geometry per episode — far too slow.

Then a reset event term samples θ, writes the fixture pose, and stores `env.slot_yaw`. Ordering
matters: it must run **after** any `reset_all`, the same trap that `--arm-jitter` hit
(`HANDOFF` §9).

### 2c. `eva_bc/slot` — the expert (`expert/plan.py`)

Currently plans in world axes: `stage_x = 0.165`, `insert_x = 0.2545`, y-targets of 0, and an
`axis_slot` that assumes slot yaw 0. All of it becomes slot-frame:

* staging point = `SLOT_CENTER − (stage_dist)·[cos θ, sin θ]`
* insertion target = `SLOT_CENTER + (insert_dist)·[cos θ, sin θ]`
* the `spin` phase must rotate the block's finger axis to **θ**, not to 0
* the `turn` phase (polar traverse) must reach the rotated staging point

The phase *structure* (reach → lift → back → spin → turn → push → release → retreat) should
survive; only the waypoints move. That is the cheapest possible rewrite and it is worth trying
before anything more ambitious.

### 2d. Everything downstream

`slot_mdp.py`, `eval_act.py`'s slot-shift flags, `collect_demos.py`, the vision stack. Most need
only to stop assuming yaw 0. The **34-D observation** gains nothing automatically — the privileged
`slot_frame` term already reports slot-relative quantities, so it keeps working once §2a lands.
The **23-D student** is untouched by construction, which is the point.

---

## 3. Gates — verify BEFORE deleting anything

**Gate A — geometry and predicate agree (minutes, no policy).**
1. θ = 0 reproduces today's `insertion_depth` / `lateral_error` / `yaw_error` **bit-for-bit** on a
   batch of random block poses.
2. For random θ, a block placed analytically at slot-frame `(along=+0.05, across=0, yaw=θ)` reads
   `is_inserted = True`; one at `across = half + 2 mm` reads False.
3. The rendered fixture actually rotates — save stills at θ = 0, ±20°, ±40° **for Big Will**.
   The predicate agreeing with itself proves nothing if the *geometry* did not move.

**Gate B — the arm can physically do it (the feasibility gate, and the one that sets θmax).**
Run the rewritten expert on a θ sweep: 0°, ±10°, ±20°, ±30°, ±40°, 128 episodes each.
* Success ≥ 0.95 at a given θ → that θ is in range.
* **θmax = the largest angle that holds ≥ 0.95.** If that is under ~15° the task is arm-limited,
  not perception-limited, and we should reconsider the axis (e.g. rotate the *block spawn* region
  with the slot to keep the approach inside the envelope).
* Watch for the failure *kind*: `never_lifted` means the grasp broke (yaw beyond wrist limit),
  `gross_miss` means the traverse left the envelope. They call for different fixes.

**Gate C — regression.** At θ ≡ 0 the new env must reproduce the old expert's success (≈1.00) and
the old champion's ≈0.979. If it does not, the rewrite changed something other than the angle.

**Only after A, B and C pass** does anything get deleted.

---

## 4. Deletion manifest — what goes, what stays

Big Will: *"delete the old data collection etc, but not the markdown files."*

**Delete** (all reproducible from scripts, all specific to the axis-aligned task):

```
slot/data/*.hdf5  slot/data/v2/            state demos          ~0.2 GB
slot/data/vision_bc/                       vision BC shards      13 GB
slot/data/vision_dagger/                   DAgger shards        0.9 GB
slot/runs/bc_arm*/                         6 BC arms + evals     ~7 GB
slot/runs/vision_bc/                       blind + v1 + v2       ~0.5 GB
slot/runs/vision_g0/  vision_render_probe/  vision_shimmer/      probe outputs
```

**Keep:** every `docs/slot/*.md`, all of `slot/scripts/`, `slot/analysis/`, `slot/slot_act/`,
`slot/expert/`.

**One judgement call I want Big Will's word on before executing:** the manifest above destroys
**the champion checkpoint** (`runs/bc_armB_seed0/ckpt_final.pt`) and **the 116 eval JSONs every
number in the markdown cites**. The markdown then makes claims nothing on disk can substantiate.
Archiving just those two things costs ~150 MB and keeps the record checkable. **Recommend: keep a
`slot/archive_axis_aligned/` with the champion ckpt, the vision v1 ckpt, and the eval JSONs;
delete everything else.** Say the word either way.

---

## 5. Then: mimic what worked, avoid what did not

The old sequence is a good template and it is worth reusing deliberately.

**What worked, reuse verbatim:**
* Flow-BC on scripted-expert demos, chunk 50 / execute 15 — cleared the state bar on the first
  honest attempt and there is no reason the angle changes that.
* DART noise **in free space only** — inside the channel a noised command levers the block out of
  the pads (`EXP_NOISE_SWEEP`). This applies unchanged; the channel is just rotated.
* The **blind control**, run *before* the visual number. On the angled task it should collapse to
  ~0, and that is the evidence that the task now requires perception.
* Two seeds before any headline; later-cohort scoring; the render-contract assert.
* **Supersample 4** and AA off — measured, not inherited.

**What did not work, do not repeat:**
* **DAgger with this teacher.** −31.3 pts, because a BC clone has no recovery behaviour to teach
  in the states the student reaches (`VISION_PLAN` §13). If DAgger is tried again it must first
  filter to states the teacher actually covers.
* **x0-steering as parameterised.** Its action space is a uniform joint-bias term; every broadcast
  latent scored 0.000/96 (`EXP_STEER` §12). Do not relaunch it.
* **Latent search.** Oracle headroom over 9 latents was +6.2 pts and did not grow with the pool.

**Open lever, untested, and now more attractive:** policy **resolution**. 160×90 was EXP08's pick
for dropping a can in a basket; a 1.5 mm clearance at an arbitrary angle plausibly needs more
pixels, and the v1 failures were already precision-shaped at 0.52 mm.

---

## 6. Order of work

1. §2a slot-frame predicates + per-env `slot_yaw` (default 0) → **Gate A.1** (bit-exact at θ = 0).
2. §2b kinematic fixture + reset event → **Gate A.2/A.3** (stills for Big Will).
3. §2c expert rewrite → **Gate B** θ sweep → fixes θmax.
4. **Gate C** regression at θ ≡ 0.
5. Deletion (§4), after Big Will's call on the archive.
6. Re-collect demos → flow-BC → state champion on the angled task.
7. Vision: cameras (unchanged), blind control **first**, then the visual policy.

**This is subject to change** — in particular gate B may force a smaller θmax, or reveal that the
block spawn region has to rotate with the slot to stay reachable.
