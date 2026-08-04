# 03 — `Rebot-ClutterExtract-v0` measured env facts

*Everything here is read off the eva_rl / IsaacLab source, with file:line. This is the
reference sheet to write code against — do not re-derive these.*

---

## 1. Gym registration

`challenge/__init__.py:19-46` registers all challenge tasks from one table:

```python
_AGENT = f"{agents.__name__}:rl_games_ppo_cfg.yaml"      # :17  -- SHARED by all tasks
gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
             kwargs={"env_cfg_entry_point": f"{__name__}.{_mod}:{_cls}",
                     "rl_games_cfg_entry_point": _AGENT},
             disable_env_checker=True)
```

| Gym ID | cfg class | default envs |
|---|---|---|
| `Rebot-ClutterExtract-v0` | `RebotClutterExtractEnvCfg` | 2048 |
| `Rebot-ClutterExtract-Play-v0` | `RebotClutterExtractEnvCfg_PLAY` | 16 |
| `Rebot-ClutterExtract-Tight-v0` | `RebotClutterExtractTightEnvCfg` | 2048, pitch 36 mm |

Only `rl_games_cfg_entry_point` exists — there is **no rsl_rl / skrl / sb3 entry point** for
challenge tasks.

## 2. ⚠ Three traps in the shipped rl_games config

`challenge/agents/rl_games_ppo_cfg.yaml`:

1. **`clip_actions: 100.0`** — this is **the exact bug that cost eva_bc two full training
   runs.** rl_games' `preprocess_actions` (`a2c_common.py:1181-84`) clamps the policy sample
   to [−1, 1] and then **rescales it to the action-space bounds**, so `100.0` multiplies every
   action by 100 and saturates the arm from step one. eva_bc's standing rule: **it must be
   1.0.** Any config we write starts from 1.0 with a comment.
2. **`name: rebot_precision_slot`** (`:51`) — shared across every challenge task, so a clutter
   run silently writes into `logs/rl_games/rebot_precision_slot/`. Always pass
   `agent.params.config.name=rebot_clutter...`.
3. `clip_observations: 100.0` — benign here (obs are small) but the same "read the consumer's
   source" rule applies.

Rest of the yaml: `a2c_continuous` / `continuous_a2c_logstd`, `separate: False`, MLP
`[256,128,64]` ELU, `fixed_sigma: True`, `normalize_input/value: True`, `gamma 0.99`,
`tau 0.95`, `lr 1e-4` adaptive with `kl_threshold 0.01`, `horizon_length 24`, `mini_epochs 8`,
`critic_coef 4`, `e_clip 0.2`, `entropy_coef 0.001`, `grad_norm 1.0`,
`bounds_loss_coef 1e-4`, `reward_shaper.scale_value 0.01`, `minibatch_size 6144`,
`max_epochs 30000`, `save_frequency 50`, `save_best_after 100`, `seed 42`.

Batch check: `horizon 24 × num_envs`, must be divisible by `minibatch_size 6144`
→ 1024 / 2048 / 4096 envs all divide evenly.

## 3. Robot

USD: `eva_rl/source/reBot_RL/data/RS-rebot-dev-arm/00-arm-rs_asm-v3.usda` (needs Isaac Sim 6.0+).
Chain: `base_link → link1…link6 → gripper_end → {gripper_left, gripper_right}`.

**Joint order (articulation order):** `joint1, joint2, joint3, joint4, joint5, joint6,
joint_left, joint_right` — 6 revolute + 2 prismatic.

| joint | type | lower [rad] | upper [rad] | USD stiffness | damping | maxForce |
|---|---|---|---|---|---|---|
| joint1 | rev | −2.80 | +2.80 | 500/deg | 60/deg | 36 N·m |
| joint2 | rev | −3.14 | 0.0 | 1500/deg | 96/deg | 36 |
| joint3 | rev | −3.14 | 0.0 | 1000/deg | 76/deg | 36 |
| joint4 | rev | −1.79 | +1.69 | 150/deg | 18/deg | 14 |
| joint5 | rev | −1.57 | +1.57 | 80/deg | 10/deg | 14 |
| joint6 | rev | −3.14 | +3.14 | 50/deg | 7/deg | 14 |
| joint_left | prism | 0 | **0.0500 m** | 100 N/m | 4 | 500 N |
| joint_right | prism | 0 | **0.0715 m** | 100 N/m | 4 | 500 N |

UsdPhysics authors angular gains **per degree**; Isaac Lab's parser converts to per-radian
(×57.2958). `rebot_arm.py:60-72` therefore passes `stiffness=None, damping=None` on purpose —
**never hardcode the USD numbers into an actuator cfg**, it de-scales every revolute gain 57×.

**Clutter overrides** (`clutter_env_cfg.py:228-234`): arm keeps `None/None`; fingers are
forced to `stiffness=2000.0, damping=40.0` (the authored 100 N/m caps squeeze at ~1.7 N,
25× too weak — CHALLENGE_SUITE C2).

### `_START_POSE` — the actual default joint pos (`lift/rebot_lift_env_cfg.py:40-49`)

```python
_START_POSE = {"joint1": 0.0, "joint2": -1.35, "joint3": -0.3, "joint4": -0.85,
               "joint5": 0.0, "joint6": 0.0, "joint_left": 0.04, "joint_right": 0.04}
```

Applied at `clutter_env_cfg.py:227`, so `default_joint_pos` **is** `_START_POSE`. Fingertips
hover near `(0.35, 0, 0.17)`; every joint ≥0.3 rad from a limit. (The USD's authored home pose
is unusable — it puts the gripper 0.135 m below the base, inside the table.)

### Links, TCP, gripper commands

```python
_BASE_LINK   = "Geometry/base_link"
_GRIPPER_END = "Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end"
_GRIPPER_OPEN  = 0.045      # per-finger prismatic target -> 89.1 mm actual opening
_GRIPPER_CLOSE = 0.0
TCP_OFFSET     = (-0.0419, 0.0, 0.0)     # challenge/mdp/common.py:38  -- MEASURED
```

Finger **body** names: `gripper_left`, `gripper_right`. Finger **joint** names: `joint_left`,
`joint_right`. Gap calibration (C3): `gap = 1.0035·(q_L + q_R) − 1.25 mm`.

> **Never carry `-0.075` into anything new.** That is `lift/`'s and `pick_place/`'s legacy TCP
> offset and it is 33.1 mm too far forward — fatal to a scripted grasp (fingers close 33 mm
> past the object). The challenge envs already use the measured value.

## 4. Action space — exact mapping

7-D float32, in cfg declaration order (`clutter_env_cfg.py:123-133`).

**Dims 0–5, `JointPositionActionCfg(scale=0.5, use_default_offset=True)`.**
`JointAction` computes `processed = offset + scale × raw` with
`offset = default_joint_pos` (`joint_actions.py:190-195`), then
`set_joint_position_target`:

```
q_target[i] = _START_POSE[joint_i] + 0.5 · a[i]        i = 1..6
offsets = (0.0, -1.35, -0.3, -0.85, 0.0, 0.0)
```

**Absolute, not relative** — a zero action commands the start pose. No per-joint `clip`;
limits are enforced by the solver. Target is written once per policy step and held for
`decimation = 8` physics substeps.

**Dim 6, `BinaryJointPositionActionCfg`.** `binary_joint_actions.py:131-146`:

```python
binary_mask = actions < 0            # true: CLOSE
processed   = torch.where(binary_mask, close_command, open_command)
```

**Threshold is exactly 0 on the sign, and `a[6] >= 0` means OPEN.** So `a[6] = 0.0` opens.
`reset()` zeroes `_raw_actions`, so the first step after a reset always reads gripper-open.

This is **identical to eva_bc's action encoding** (6 arm targets scaled `(dq − q_default)/0.5`
plus a ±1 grip channel), so the action plumbing ports with zero change.

## 5. Observations — the 42-D vector, dim by dim

| slice | dims | term | source | contents | frame/units |
|---|---|---|---|---|---|
| `0:8` | 8 | `joint_pos_rel` | `isaaclab/envs/mdp/observations.py:213` | `joint_pos − default_joint_pos`, articulation order | rad ×6, m ×2 |
| `8:16` | 8 | `joint_vel_rel` | `:260` | raw joint velocities (defaults are 0) | rad/s ×6, m/s ×2 |
| `16:19` | 3 | `target_pose` pos | `challenge/mdp/observations.py:18` | target root position | **robot root frame** ≡ env-local, table at z = 0 |
| `19:23` | 4 | `target_pose` quat | same | target orientation, **(x, y, z, w)** | — |
| `23:26` | 3 | `clutter` d0 | `challenge/mdp/clutter.py:90` | `(dx, dy, up_z)` for `distractor_0` | m, m, unit |
| `26:29` | 3 | d1 | " | | |
| `29:32` | 3 | d2 | " | | |
| `32:35` | 3 | d3 | " | | |
| `35:42` | 7 | `last_action` | `observations.py:740` | previous **raw** action, pre-scale | — |

`dx, dy` are *distractor minus target* in env-local xy. `up_z = quat_apply(q, [0,0,1])[2]` —
1.0 upright, 0.0 on its side, and `TOPPLE_DOT = 0.75` ⇒ 41.4° of tilt.

**Quaternions are XYZW everywhere** (`isaaclab/utils/math.py:891,640`), identity `(0,0,0,1)` —
same convention eva_bc used, so the normalizer and dataset port unchanged.

**Grasp-bit indices for this task** (eva_bc EXP01 variant D = finger pos + finger vel + last
grip command, the combination that scored AUC 0.968 at **0 % FPR**):

```
finger_pos  -> obs[6], obs[7]
finger_vel  -> obs[14], obs[15]
last_grip   -> obs[41]
```

(eva_bc's own indices were `(6, 7, 14, 15, 40)` on a 41-D obs — only the last index moves.)

## 6. `pick_place` (what eva_bc was built on) vs `clutter`

| | `Rebot-PickPlace-v0` | `Rebot-ClutterExtract-v0` |
|---|---|---|
| obs dim | 39 (`objects_canonical` 16-D, 2 objects) | **42** (`block_pose_in_root` 7 + `clutter_obs` 12) |
| actions | **identical** 7-D | identical |
| ee_frame offset | `(-0.075, 0, 0)` legacy | **`(-0.0419, 0, 0)` measured** |
| finger actuator | USD 100 N/m | **2000 / 40** |
| rewards | 12 terms incl. `gripper_toggle −50`, `time_penalty −0.3` | 9 terms, **no** toggle penalty, **no** time penalty |
| terminations | `time_out`, `any_object_dropped` | + **`any_distractor_toppled`** |
| curriculum | yes (action_rate/joint_vel tighten at 30 k) | **none** |
| events | mid-episode nudges, friction DR, reverse curriculum, lying spawns | **reset jitter only** |
| episode | 10.0 s = 500 steps | **14.0 s = 700 steps** |
| decimation / sim.dt | 8 / (1/400) → 50 Hz | identical |
| default envs | 4096 | 2048 |
| PhysX patch count | `2**19` | **`2**20`** |

Two consequences worth flagging:

- **No domain randomisation and no mid-episode perturbation in clutter.** pick_place had
  nudges + friction randomisation; clutter has only reset jitter. So a clutter policy will be
  *less* robustness-tested by construction, and the eva_bc "perturbed expert success caps what
  DAgger can teach" logic has no perturbation suite to run against yet. If we want a robustness
  number we have to add the suite ourselves.
- **No `gripper_toggle` penalty.** In pick_place that term (−50) suppressed grip chatter.
  Clutter has nothing stopping a policy from oscillating the binary gripper, which matters
  because grip chatter near a 12 mm gap is a topple risk.

## 7. MDP terms available in `challenge/mdp`

- `common.py`: `TCP_OFFSET`, `object_pos_local`, `object_quat`, `ee_pos_local`, `yaw_of`,
  plus slot-task geometry.
- `clutter.py`: `DISTRACTOR_NAMES`, `CL_BLOCK_HALF=(0.018,0.015,0.035)`, `TOPPLE_DOT=0.75`,
  `GOAL_XY=(0.185,-0.185)`, `GOAL_RADIUS=0.045`, `EXTRACT_Z=0.090`; `_up_z`,
  `any_distractor_toppled`, `distractors_disturbed`, `target_extracted`, **`target_at_goal`**,
  `reach_target`, `target_to_goal`, `clutter_obs`, `clutter_success` (logs
  `Metrics/clutter_success_rate`), `record_spawn_xy`.
- `rewards.py`: `block_lifted` (used by clutter), Factory `squashing_fn`, keypoint terms.
- `terminations.py`: `block_dropped` (used), `block_toppled`.
- `observations.py`: `block_pose_in_root` (used), `slot_frame`.

### eva_bc's five `mdp` touchpoints, mapped to clutter

eva_bc's README requires the task package to expose `placed_mask`, `object_pos_local`,
`basket_centers_local`, `OBJECT_NAMES`, and an `objects_canonical` obs term. Clutter has
none of those names. The mapping we will implement in our own adapter module (NOT by editing
`challenge/`):

| eva_bc touchpoint | clutter equivalent |
|---|---|
| `mdp.placed_mask(env) -> (N,) bool` | `clutter.target_at_goal(env)` |
| `mdp.object_pos_local(env, name)` | `common.object_pos_local` — same name, same semantics |
| `mdp.basket_centers_local(env) -> (N,2)` | constant `GOAL_XY` broadcast to `(N,2)` |
| `mdp.OBJECT_NAMES` | `("target",) + DISTRACTOR_NAMES` |
| `objects_canonical` obs term | not needed — there is exactly one target, no permutation |

The `objects_canonical` machinery (target-first canonical ordering, and the target-selection
logic eva_bc's residual/steer obs builders mirror *exactly*) simply **disappears** here, which
removes a whole class of eva_bc bug. Good news.

## 8. Launching

```bash
source /home/eva/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6
cd /home/eva/Desktop/isaacLab/eva_rl        # scripts use relative logs/ and data/ paths
```

```bash
# smoke test the env
python scripts/test_clutter_env.py
python scripts/test_clutter_env.py --task Rebot-ClutterExtract-Tight-v0 --num_envs 16

# rl_games training -- ALWAYS override the log name, and fix clip_actions
python scripts/rl_games/train.py --task Rebot-ClutterExtract-v0 --num_envs 1024 \
    agent.params.config.name=rebot_clutter

# play / eval
python scripts/rl_games/play.py --task Rebot-ClutterExtract-Play-v0 --num_envs 16 \
    --checkpoint <run>/nn/<name>.pth

# video of the scene
python scripts/challenge/record_env_video.py --task Rebot-ClutterExtract-Play-v0 \
    --cam_eye 0.10 -0.40 0.28 --cam_target 0.245 0.0 0.075
```

Hydra passthrough works for anything: `env.rewards.disturbance.weight=-1.0`,
`env.episode_length_s=20.0`, `agent.params.config.clip_actions=1.0`.

**`--headless` is deprecated** in Isaac Lab 3.0 — headless is the default; use `--viz kit`
to see the viewer. Camera rendering above ~256 envs has been seen to throw
`CUDA error: an illegal memory access`; keep video runs at ≤32 envs.

## 9. Starting assets: there are none

A full disk search found **no RL checkpoints anywhere** — `logs/rl_games/**` does not exist,
`logs/bc/`, `logs/distillation/`, `logs/analysis/`, `logs/videos/` are all absent. The only
`.pt` files in eva_rl are two pick-place lookup tables (`grasp_table.pt` 624 KB,
`carry_waypoints.pt` 2.5 KB), both built for **24 mm cylindrical cans at pick-place spawn
radii** — useless for 36×30×70 mm cuboids in a row at r = 0.25.

`data/pick_place_demos/*/` contains 10 directories holding only `meta.json`; every
`shard_*.pt` is gitignored and absent.

**No training run has ever been done on any challenge env.** We start from zero, which also
means: to use `bc_to_rlgames.py` (which needs a structurally correct rl_games `.pth` as a
transplant template) we must first run a few epochs of `train.py` on clutter to mint one.

## 10. Gotchas checklist

1. Override `agent.params.config.name` or every challenge task shares one log dir.
2. `clip_actions` must be **1.0**, not the shipped 100.0.
3. Run from the eva_rl repo root.
4. **`record_spawn_xy` must remain the last reset event** (`clutter_env_cfg.py:210`); the
   smoke test asserts disturbance is 0.00 at reset.
5. `distractors_disturbed` returns zeros until the first reset creates `env._clutter_spawn_xy`.
6. Zero action ⇒ start pose **and gripper open**.
7. TCP offset `-0.0419`, never `-0.075`.
8. Obs terms must return `(N, k)`, never `(N,)`. Isaac Lab 3.0 buffers need `.torch`.
   Quaternions are XYZW.
9. Keep `24 × num_envs` divisible by `minibatch_size`.
10. 10 GiB card — the 2048-env default is not a given; measure before trusting it.
</content>
