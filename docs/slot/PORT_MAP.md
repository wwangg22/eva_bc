# eva_bc -> `Rebot-PrecisionSlot-*` port map

Concrete interface facts, gathered by reading every file in `act/`. This is the checklist
for the port, not a summary of the design docs (those are in `PLAN.md`).

Everything below is `eva_bc`-relative unless the path says otherwise.

---

## 1. The good news

**The action space is identical.** Both tasks are 7-D: 6 arm joint position targets
(`joint[1-6]`, `scale=0.5`, `use_default_offset=True`) plus one binary gripper channel. So
`ACTION_DIM = 7`, `RES_ACTION_DIM = 6` (arm only; the grip channel passes through the residual
untouched), the `compose()` blend, and `make_fixed_x0(chunk, 7, seed)` all port **unchanged**.

The robot is the same 8-DOF articulation, so the joint block of the observation is also
identical: `joint_pos` 8 then `joint_vel` 8. That means `STATE_SLICE = slice(0, 16)` and
`FINGER_FEATURE_DIMS = [6, 7, 14, 15]` **survive the port verbatim**.

## 2. The observation edit

| | pick-place (41-D) | slot (34-D) |
|---|---|---|
| `OBS_DIM` | 41 | **34** |
| `STATE_SLICE` | `slice(0, 16)` | `slice(0, 16)` unchanged |
| `ENV_STATE_SLICE` | `slice(16, 41)` | **`slice(16, 34)`** |
| `STATE_DIM` | 16 | 16 unchanged |
| `ENV_STATE_DIM` | 25 | **18** |
| grip-command index | `40` | **`33`** (last dim of `last_action`) |
| `BIT_DIMS` | `[6, 7, 14, 15, 40]` | **`[6, 7, 14, 15, 33]`** |
| `RES_OBS_DIM` | 64 = 41+7+1+15 | **57** = 34+7+1+15 (if the 15-D tail is kept) |
| `STEER_OBS_DIM` | 56 = 41+15 | **49** = 34+15 |

The 18-D env-state block is `block_pose` 7 (pos + quat **XYZW**, robot-root frame) +
`slot_error` 4 (`depth, lateral, yaw, inserted`) + `actions` 7.

Files carrying these constants: `act/dataset.py:37-42` (plus the layout docstring at `:12-22`),
`act/residual_core.py:31-40` and `:61` (grip index) and `:194` (arm slice),
`act/steer_core.py:41-42`, `act/report_coverage.py:31,57,60-77` (entirely pick-place-specific,
needs rewriting or dropping).

## 3. The `mdp` touchpoint contract

`act/` imports the task's mdp module and calls exactly four things:

| call | pick-place | slot equivalent |
|---|---|---|
| `mdp.OBJECT_NAMES` | `("object_a","object_b")` | `("block",)` |
| `mdp.object_pos_local(env, name)` -> `(N,3)` | `mdp/common.py:46` | **already exists** in `challenge/mdp/common.py:66` |
| `mdp.placed_mask(env)` -> `(N, n_obj)` **bool** | `mdp/common.py:68` | wrap `mdp.is_inserted(env)` to `(N,1)` |
| `mdp.basket_centers_local(env)` -> `(N,2)` | `mdp/common.py:51` | **no analogue** — the slot is fixed at `(0.245, 0)` |

Note `placed_mask` returns `(N, n_objects)`, **not** `(N,)` as `README.md` claims; `eval_act.py:315`
reduces it with `.all(dim=1)`. Returning `(N,1)` for the single block keeps `.all(dim=1)` and
`.sum(dim=1)` valid with no further edits.

Also touched, indirectly, via `residual_core.task_features`:
`env.scene["ee_frame"].data.target_pos_w.torch[..., 0, :]` / `.target_quat_w.torch` (**wxyz**),
`env.scene[name].data.root_pos_w.torch` / `.root_quat_w.torch`, `env.scene.env_origins`.
The `.torch` accessor is the Isaac Lab 3.0 idiom and is required everywhere.

**Plan:** a single shim module `slot/slot_mdp.py` re-exports the challenge mdp plus the two
adapters, so nothing in `eva_rl` needs editing.

## 4. Things that must be REDESIGNED, not renamed

These are the places where the pick-place semantics do not carry over.

1. **Target-object selection** (`residual_core.py:141-159`) is hard-coded to two objects
   (`dists[1] < dists[0]`) with a nearest-unplaced rule. One block -> the whole block collapses
   to "the object". Simplification, not a problem.

2. **The basket-delta feature** (`residual_core.py:161-170`) is
   `[goal_x - obj_x, goal_y - obj_y, CAN_REST_Z_IN_BASKET - obj_z]`. For a horizontal insertion
   the meaningful error is along the insertion axis and includes yaw. **Replace** with the
   slot-frame error the env already computes: `(depth, lateral, yaw)` plus the block-to-mouth
   delta. `CAN_REST_Z_IN_BASKET = 0.015` has no analogue (the slot analogue is
   `SLOT_FLOOR_Z + BLOCK_HALF[2] = 0.055`).

3. **The flush trigger's z-drop half** (`FLUSH_Z_DROP=0.02` gated on `prev_z > FLUSH_Z_ABOVE=0.05`)
   is a *gravity-drop* slip proxy sized against the 40 mm basket rim. Here a slip is lateral, and
   the block legitimately sits at z = 0.055 inside the slot. Keep the position-jump half
   (`FLUSH_POS_JUMP = 0.03`) and re-derive the second half against the insertion axis — or drop it.

4. **The grasp bit must be retrained.** `experiments/exp06_grasp_bit.pt` encodes this gripper's
   aperture statistics *for a 24 mm can*. Our block is 30 mm across, so the closed-on-block finger
   value differs. Pipeline: `exp01_probe.py` -> `exp06_grasp_bit.py`, holding the **0 % FPR** gate.
   The probe is a 5 -> 128 -> 128 -> 1 MLP; runtime rule is `sigmoid > 0.5 AND commanded-closed`.
   The commanded-grip channel is **required** — physical finger joints alone score higher AUC
   (0.976) but 27.1 % FPR on real freeze states.

   *Slot-specific bonus:* C3 measured `separation = 2.000 * q` exactly, with calibration
   `gap = 1.0035*(q_l+q_r) - 1.25 mm` at 0.035 mm residual. A 30 mm block therefore pins the
   fingers at `q ≈ 0.0156` each, versus 0 on air — a **15.6 mm** separation in a channel the obs
   already carries. This is a much larger margin than the ±12 mm the pick-place task had.

5. **`taxonomy.py` buckets** are pick-place failure modes (placed-one-stuck, never-lifted,
   lifted-never-placed, dropped-after-place). Redefine for this task: never-grasped /
   grasped-never-lifted / lifted-never-engaged / engaged-but-shallow / toppled / dropped /
   inserted-then-lost.

## 5. Exact schemas the expert must produce

### HDF5 (`act/dataset.py:141-163`, written by `expert/run_expert_v1.py:787-801`)

```
/data                       group, attr: total (int)
/data/demo_{i}              group          <-- MUST be demo_<int>; loader sorts on int(k.split("_")[-1])
    obs/policy   (T, OBS_DIM) float32
    actions      (T, 7)       float32
    train_mask   (T,)         uint8        1 = trainable, 0 = loss-censored
  attrs: success (bool), num_samples (int), episode_kind (str),
         perturb_steps (JSON str), segments (JSON str), outcomes (JSON str)
```

Hard shape validation at `dataset.py:152-156` — mismatches raise rather than warn.

`train_mask` producer (`run_expert_v1.py:267-277`): zeros the span `[seg.t, next_seg.t)` for any
segment whose outcome is `"missed"` or `"lost"`. The mask boundary is failure **detection** —
post-detection segments (reopen, re-approach) are recovery skill and stay trainable.

Both the episode-end padding and `train_mask == 0` ride the **same** `action_is_pad` channel
(`dataset.py:168-187`), which the flow loss multiplies out and the decoder excludes from
self-attention via `key_padding_mask`.

### Checkpoint (`act/train_flow.py:64-83`)

```python
{"step": int,
 "policy_state_dict": ..., "normalizer_state_dict": ...,
 "config": {"policy_type": "flow", "chunk_size": 50, "n_action_steps": 15,
            "num_inference_steps": 10, "state_dim": 16, "env_state_dim": 25, "action_dim": 7}}
```

`eval_act.load_checkpoint` **asserts** `state_dim`/`env_state_dim`/`action_dim` against the
module constants (`eval_act.py:178-180`), so the dataset constants and the checkpoint must agree
— a useful tripwire during the port. Dispatch is on `config["policy_type"] == "flow"`.

Normalizer stats are persisted only as buffers and recovered by **string-parsing buffer names**
(`eval_act.py:182-186`, `key.replace("__", ".")`). Any obs key containing a literal `__` is
corrupted by that round-trip — ours don't.

## 6. The controller (ports unchanged)

`BatchedACTController` (`act/eval_act.py:53-158`) is fully vectorized: `_buf (N, n_action_steps, 7)`
+ `_idx (N,)`, where `_idx >= n_action_steps` means empty. API: `act()`, `peek()` (refill + read,
no advance), `pop()` (read + advance), `reset()/flush()`. `peek`/`pop` must be paired — `compose()`
asserts they returned the same tensor.

`predict_action_chunk(batch, generator=None, x0=None)`: `x0.dim()==3` is `(B, chunk, 7)` per-env
noise (**this is the x0-steering path**), `x0.dim()==2` is `(chunk, 7)` broadcast to the batch
(deterministic base). Precedence at the call site: `steer_x0` > `fixed_x0` > fresh draw.

`n_action_steps` is never read by the model — it lives only in the controller.

## 7. The RL wrappers

**Residual** (`act/residual_wrapper.py`): obs `Box(57,)`, action `Box(6,)`. Reward comes **from the
env**, plus one wrapper term `- res_penalty * ||applied||^2`.

**Steering** (`act/steer_wrapper.py`): obs `Box(49,)`, action `Box(7,)`. One rl_games step = one
full 15-env-step window; reward is the env reward **summed over the window**, nothing recomputed.
`set_steer(z)` sets `x0 = alpha_x0 * tanh(z)` broadcast across all chunk positions.

Both use `import gym as ogym` (**old** gym, not gymnasium) for the space objects — rl_games-facing.

**Window alignment is asserted at build time**: `max_episode_length % n_action_steps == 0`
(`train_steer.py:82-85`). Our episode is **600 steps** and the window is 15 -> **40 windows
exactly**. Clean, no episode-length override needed (pick-place had to force 30 s = 1500 steps).

`minibatch_size` is recomputed at runtime as `horizon_length * num_envs / 4` — the yaml value is
dead. Keep `horizon_length * num_envs` divisible by 4.

## 8. Non-negotiable config facts

- **`clip_actions: 1.0`** in both yamls. rl_games `preprocess_actions` clamps to [-1,1] and then
  **rescales to the action-space bounds** (`a2c_common.py:1181-1184`). The pick-place port had
  `100.0` and it multiplied every action by 100, saturating the tanh from step one. Cost two full
  training runs and two wrong post-hoc theories.
- **`mu_init: const 0.0`** zeroes the **weight only** — the mu *bias* stays at rl_games' default
  `U(±0.125)`. Measured harmless there; must be re-checked here, not assumed.
- **`sigma_init`**: `-2.5` (sigma ~ 0.08) for the additive residual — `-1.0` (sigma ~ 0.37,
  about 1 deg/joint/step) destroyed the mm-precision base outright. For steering, `-1.2`
  (sigma ~ 0.30) was pre-registered on the argument that z-space exploration is on-manifold.
  **This task is tighter than pick-place** (1.5 mm vs 50 mm placement tolerance), so the
  noise-tolerance of the frozen base must be measured before training, not inherited.
- `reward_shaper.scale_value: 0.01`, `normalize_input/value: True`, `horizon_length: 24`,
  `mini_epochs: 8`, `lr 1e-4` with adaptive schedule, `kl_threshold 0.01`.

## 9. Import quirks to normalize during the port

- `train_*.py` put argparse at **module level before `AppLauncher`**, which is why the wrappers
  were split into their own files.
- `sys.path.insert(0, parent.parent)` happens **after** `AppLauncher`.
- `report_coverage.py:29` uses a bare `from dataset import ...` — only works when run from inside
  `act/`.
- device is hardcoded `"cuda:0"` in every script and both yamls.
- default task id `"Rebot-PickPlace-Play-v1"` and mdp import path
  `reBot_RL.tasks.manager_based.pick_place.mdp` appear in six scripts each.

## 10. Reference values from the source task (for sanity-checking ours)

`decimation=8`, `sim.dt=1/400` -> 20 ms control step (**same here**).
Flush thresholds `0.03 / 0.02 / 0.05` m. `--alpha` 0.1 env action units = 0.05 rad.
`--alpha-x0` 1.0. `FIXED_X0_SEED` 7. `chunk 50 / execute 15 / 10 Euler steps / dim_model 512`.
Training: 100k steps, lr 1e-4, batch 64, AdamW, no LR schedule, no grad clipping, no val split.

---

# Addendum — findings from the actual port (2026-08-02, session 3)

Everything above was written by reading `act/`. This section records what changed once the
copy was made and exercised.

## A. The package is `slot_act`, not `act` — and this is not cosmetic

`act/` was copied to `slot/slot_act/` and every `from act.X import` / `import act.X` rewritten
(35 occurrences across 11 files; `modeling_act.py` and `modeling_flow.py` use *relative*
imports and were untouched).

Keeping the name `act` is a trap. Both directories are then importable as `act` and which one
wins is decided by `sys.path` order. Measured, not theorised:

```
cd /tmp && PYTHONPATH=/home/rei/Desktop/isaaclab/eva_bc python -c \
    "import act, act.dataset as D; print(act.__file__, D.OBS_DIM)"
/home/rei/Desktop/isaaclab/eva_bc/act/__init__.py 41
```

No exception, no warning — just the 41-D pick-place constants silently bound to 34-D slot
data. A guard placed *inside the copy* does not help, because in the failing case the copy is
never imported at all; the first attempt at this mitigation was wrong for exactly that reason.
Renaming removes the ambiguity rather than detecting it: there is no `slot_act` in `eva_bc`.

## B. Env-cfg attribute names that will raise on the slot task

The eval/train scripts null out pick-place terms by name. The slot task's names differ, so
these are `AttributeError`s waiting to happen — at `eval_act.py:262`, `eval_residual.py:137`,
`eval_steer.py:89`, `train_steer.py:77`:

| script writes | slot task actually has |
|---|---|
| `terminations.object_dropping` | **`terminations.block_dropped`** (`minimum_height=-0.05`) and **`terminations.block_toppled`** (`max_tilt=0.6`) |
| `rewards.dropping_penalty` | `rewards.dropping_penalty` **exists** (−30.0); there is also `toppling_penalty` (−10.0) |
| `terminations.time_out` | exists |

Whether to null them at eval is a *decision*, not a rename: on the slot task a dropped or
toppled block is a genuine failure that should end the episode, and demo collection already
measured that these fire (44/128 mid-episode resets at `noise_std = 0.02`).

## C. Horizon and window alignment — clean, verified

`episode_length_s = 12.0`, `decimation = 8`, `sim.dt = 1/400` ⇒ 20 ms control step ⇒
**600 steps**. `600 % 15 == 0`, so the build-time assert at `eval_steer.py:96-99` and
`train_steer.py:82-85` passes with **no episode-length override**. Every script's
`--episode-length-s` default of `30.0` must be changed to `12.0` or removed; pick-place needed
30 s and this task does not.

## D. `residual_core.py` hard-codes exactly two objects

Lines 154–156 select the target with `target_is_b = (dists[1] < dists[0])` and a two-way
`torch.where`. With one object this must be rewritten (or the whole block collapsed to "the
object"). It is an `IndexError` with `OBJECT_NAMES = ("block",)`, so it fails loudly — good.

## E. Files needing more than a rename

| file | status |
|---|---|
| `modeling_act.py`, `configuration_act.py`, `modeling_flow.py`, `normalize.py` | **verbatim** — zero task coupling; all dims flow from the config |
| `dataset.py` | **DONE** — constants remapped, layout docstring rewritten, pool filters verified to port unchanged |
| `train_act.py`, `train_flow.py` | no numeric edits at all; dims come from `dataset` constants |
| `residual_core.py` | heaviest: obs indices, the 2-object block, the basket delta, the grasp bit |
| `steer_core.py` | `STEER_OBS_DIM 56 → 49`, docstring layout |
| `eval_*.py`, `train_residual.py`, `train_steer.py`, `diag_training_env.py` | task id, mdp import, env-cfg names, horizon, `cuda:0` |
| `report_coverage.py` | **rewrite or drop** — hardest-coded obs layout in the tree (`for base in (16, 23)`, `o0[32:34]` basket slice, can-cylinder axis maths). Not on the critical path |

## F. The grasp bit must be retrained, and its file does not exist yet

Five scripts point at `<repo>/experiments/exp06_grasp_bit.pt`, which after the copy resolves to
`slot/experiments/exp06_grasp_bit.pt` — **absent**. The artefact is also numerically
pick-place-specific: its `mu`/`sd` were fit on obs dims `[6,7,14,15,40]` of the 41-D vector for
a 24 mm can. Our block is 30 mm, so the closed-on-block finger value differs.

## G. Pool filters port with **no logic change** — verified on real data

`collect_demos.py` emits eva_bc's outcome vocabulary (`reach:grasp:g0` → grasped/missed with a
`clean` flag, carry phases → held/lost, `release` → seated/unseated). Because noise-injected
episodes are tagged `episode_kind="dart"`, they fail `nominal_pool_filter`'s `== "nominal"`
test and land in `recovery_pool_filter` automatically. Checked against a collected file:
8 nominal / 0 recovery on a zero-noise run, and the dataset yields the right shapes —
state `(16,)`, env_state `(18,)`, action `(50,7)`, `action_is_pad` `(50,)`.

## H. Correction to section 2: the residual/steering dims are **58 / 50**, not 57 / 49

Section 2 predicted `RES_OBS_DIM = 57` and `STEER_OBS_DIM = 49` on the assumption that the
15-D `task_features` tail carries over unchanged. It does not, and the change is deliberate.

Pick-place's 3-D basket delta becomes a **4-D** slot goal delta, so the tail is 16-D:

```
RES_OBS_DIM   = 34 + 7 + 1 + 16 = 58
STEER_OBS_DIM = 34         + 16 = 50
```

The fourth channel is yaw, and the reason is that the env's own observation throws away the
information the policy needs. `obs[23:27]` already carries `(depth, lateral, yaw, inserted)`,
but `mdp.lateral_error` and `mdp.yaw_error` return **absolute values** — they tell the policy
how wrong it is and not which way to correct. `slot_mdp.goal_delta` returns the signed
quantities, and adds yaw because a horizontal insertion into a 1.5 mm per-side channel fails
on yaw (`SUCCESS_YAW` = 0.12 rad) in a way a top-down drop into a basket never did.

Asserted at import time (`slot/scripts/check_port.py`), because the two constants and the
tail width are related only by arithmetic in three separate files:

```
RES_OBS_DIM   == OBS_DIM + 7 + 1 + 16
STEER_OBS_DIM == OBS_DIM + 16
BIT_DIMS[-1]  == OBS_DIM - 1
```

## I. The one edit a mechanical rename misses

`residual_core.py:61` was `closed = obs41[:, 40] < 0` — a **bare integer** for the
commanded-grip channel, invisible to any `obs41 -> obs34` search-and-replace. With
`OBS_DIM = 34` it is an out-of-bounds read. It is now `obs34[:, BIT_DIMS[-1]]`, which cannot
drift from `OBS_DIM` again.

This channel is not optional: the 4 physical finger joints alone score a *better* AUC (0.976
vs 0.968) but **27.1 % false-positive rate** on real freeze states, against 0 % with the
command included.

## J. Flush rule: the z-drop half is removed on this task

`FLUSH_Z_DROP = 0.02` gated on `prev_z > FLUSH_Z_ABOVE = 0.05` was a gravity-drop slip proxy
sized against a 40 mm basket rim. Here the block **legitimately** descends ~8 mm the instant
the fingers open at the insert pose (measured: it rides at z = 63 mm while carried, above the
50 mm wall tops, and settles to 55 mm). The pick-place rule would fire a chunk flush exactly
there — the most precision-critical moment in the episode — and chunk commitment is
load-bearing: shortening the execution horizon collapsed success 59.4 → 32.8 → 3.1 → 0 %.

Only `FLUSH_POS_JUMP = 0.03` is kept, in both `residual_core.py` and `eval_act.py`.

## K. Success predicate wired into `eval_act.py`

`placed_mask` resolves to `slot_mdp.placed_mask` = `is_inserted AND seated`. Each episode
record additionally carries `inserted_raw`, `depth_mm`, `lateral_mm`, `yaw_rad`, so the gap
between the guarded and bare predicates stays visible in every eval file rather than being
silently closed. The `basket_xy` field is replaced by these (the slot is welded to the table).
