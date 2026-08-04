# 05 — Porting `eva_bc/act/` from pick-place (41-D) to ClutterExtract (42-D)

*Exact, line-referenced. Written against the eva_bc source; verify line numbers survive any
upstream change before editing.*

Target dims:

```
OBS_DIM        = 42     STATE_DIM  = 16     ENV_STATE_DIM = 26     ACTION_DIM = 7
RES_OBS_DIM    = 65     RES_ACTION_DIM   = 6
STEER_OBS_DIM  = 57     STEER_ACTION_DIM = 7
```

---

## 1. Five landmines, ranked

### L1 — There is no expert, and that is the whole project

Every stage downstream of Stage 1 consumes a frozen BC checkpoint, and BC consumes demos.
There are none, for this task, anywhere. Porting the code is ~2 days of mostly mechanical
work; **producing the demonstrations is the actual project**, and eva_rl's own doc concedes
the task has never been shown to be solvable at the tight end of its gap distribution.

### L2 — `residual_core.py:159` garbles a quaternion (latent bug, do not copy)

```python
158	rel_pos, rel_quat = subtract_frame_transforms(ee_w, ee_q, tpos, tquat)
159	rel_quat_xyzw = rel_quat[:, [1, 2, 3, 0]]  # obs convention is XYZW
```

`subtract_frame_transforms` in Isaac Lab 3.x **already returns XYZW**
(`isaaclab/utils/math.py:891-899`). The permutation therefore maps `(x,y,z,w) → (y,z,w,x)`,
feeding a garbled quaternion into both the 64-D residual obs (`obs64[57:61]`) and the 56-D
steer obs (`obs56[49:53]`). A related stale comment sits at `:147` (`ee_q = ... # wxyz` — it
is xyzw, and that input is fine).

It never showed up in EXP06/EXP07 because a randomly-initialised MLP just learns whatever
consistent bijection it is handed. **Drop the permutation in the port.**

### L3 — 700 is not a multiple of 15, so the steering protocol will not start

Clutter: `decimation = 8`, `sim.dt = 1/400` ⇒ `step_dt = 0.02`; `episode_length_s = 14.0`
⇒ `max_episode_length = 700`. **`700 % 15 = 10`**, so the window-alignment asserts at
`train_steer.py:82-85` and `eval_steer.py:96-99` fire immediately.

Options: `episode_length_s = 13.8` (690 = 46×15) or `14.1` (705 = 47×15); or choose an
`n_action_steps` that divides 700 (10, 14, 20, 25). **Note the tension with eva_bc EXP02** —
success collapsed monotonically as `n_action_steps` shrank 15 → 1, so dropping to 10 is not
free. Prefer changing the episode length; 13.8 s is the smaller perturbation.

### L4 — `topple_penalty` and `distractor_toppled` are inseparable

`clutter_env_cfg.py:173-175`:

```python
topple_penalty = RewTerm(func=mdp.is_terminated_term, weight=-40.0,
                         params={"term_keys": "distractor_toppled"})
```

The penalty is an `is_terminated_term` **on** that termination. Nulling the termination
without nulling the reward term leaves a term keyed to a name that no longer exists. This is
exactly the coupling `train_steer.py:8-11` warns about for pick-place's
`object_dropping`/`dropping_penalty` pair.

And there is a real design tension: the window-aligned steering protocol needs **no
mid-episode terminations**, but `distractor_toppled` *is the task*. Turning it off changes
the problem. Resolution has to be decided deliberately, not defaulted — see `02_PLAN.md`.

### L5 — The success predicate changes shape, `(N, K) → (N,)`

`mdp.placed_mask(env) → (N, num_objects)` becomes `clutter.target_at_goal(env) → (N,)`.
So `placed_now.all(dim=1)` collapses to the predicate itself, and `placed_final`/`placed_max`
have no analogue.

**Recommendation:** latch an `ever_at_goal` per episode, mirroring the env's own
`clutter_success._ever` (`clutter.py:100-118`). eva_bc samples success *pre-step* because
`ManagerBasedRLEnv` resets done envs before returning observations — that trick is fragile
when episodes terminate on `distractor_toppled` rather than `time_out`, so latch instead.
Also record `toppled` (`any_distractor_toppled`) and `extracted` (`target_extracted`) per
episode; the failure taxonomy needs them.

---

## 2. HDF5 demo format (what we must emit)

`dataset.py:2-23` is normative:

```
data/demo_{i}/
    obs/policy   (T, 42) float32     full privileged observation
    actions      (T, 7)  float32     expert actions
    train_mask   (T,)    uint8       1 = trainable, 0 = loss-censored
  attrs: success (bool), num_samples (int), episode_kind (str),
         segments (JSON str), outcomes (JSON str)
```

Hard requirements:

- **Group names are parsed, not just sorted**: `int(k.split("_")[-1])` at `dataset.py:143`.
  `demo_a` crashes.
- **No `dones`, no `rewards`, no episode-boundary dataset** — episode boundaries *are* group
  boundaries, and `T = obs.shape[0]`.
- **No phase-label array.** Phase lives only in the `segments` JSON and is consumed at write
  time to build `train_mask`.
- Everything loads fully into RAM; obs/actions gzip-compressed, `train_mask` uncompressed.
- **Obs/action alignment:** `ep_obs` gets the observation *before* the step, `ep_act` the
  action applied to it.
- All episodes are written; success filtering happens in `dataset.py`, not the collector.

Sample construction (`dataset.py:168-187`): every `(demo, t)` is one sample; the chunk is
edge-padded past episode end, and `action_is_pad` carries **two** meanings at once —
past-episode-end **and** `train_mask == 0`.

`train_mask` construction (`run_expert_v1.py:267-277`): zero over any segment whose outcome is
`missed` or `lost`, from that segment's start to the next segment's start. DAgger additionally
hard-zeroes the entire policy prefix (`collect_dagger.py:267-268`).

### The vocabulary we must design before writing a single demo

`dataset.py:45-98`'s pool filters key off pick-place's `{grasped, missed, lost, delivered,
recovery, misexec}`, the `":g{n}"` regrasp-id regex, `clean ≡ close_disp < 0.005`, and
`via.startswith("carry-direct")`. **None of that fits clutter.** The natural clutter taxonomy
is different — something like `{threaded, wedged, toppled, extracted, placed, dropped,
recovery}` — and it is baked into every HDF5 we produce. Decide it first.

---

## 3. Obs-layout hardcoding — every site

`dataset.py:37-42` is the single origin:

```python
OBS_DIM = 41              ->  42
STATE_SLICE = slice(0,16) ->  unchanged
ENV_STATE_SLICE = slice(16,41) -> slice(16,42)
STATE_DIM = 16            ->  unchanged
ENV_STATE_DIM = 25        ->  26
ACTION_DIM = 7            ->  unchanged
```

| file:line | what | change |
|---|---|---|
| `dataset.py:37-42` | the constants | as above |
| `dataset.py:2-23` | layout docstring | rewrite |
| `dataset.py:152,183-184,197-198` | assertions + slice application | automatic |
| `eval_act.py:41,106-108,178-180,290` | imports + split + assertions | automatic |
| `residual_core.py:31` | `RES_OBS_DIM = 64` | `65` |
| `residual_core.py:33` | `FINGER_FEATURE_DIMS = [6,7,14,15]` | **unchanged** — proprio block is identical |
| `residual_core.py:34` | `BIT_DIMS = [6,7,14,15,40]` | **`[6,7,14,15,41]`** |
| `residual_core.py:35` | `CAN_REST_Z_IN_BASKET = 0.015` | `BLOCK_REST_Z_AT_GOAL = 0.035` |
| `residual_core.py:61` | `closed = obs41[:, 40] < 0` | `obs42[:, 41] < 0` |
| `steer_core.py:41` | `STEER_OBS_DIM = 56` | `57` |
| `report_coverage.py:57-70` | can radii/quats at `[16:19]/[23:26]`, basket at `[32:34]` | **rewrite — none of it exists** |
| `experiments/exp01_probe.py:27,33-34,89,98,166-167` | `FINGER_DIMS`, `41` literals | `[6,7,14,15,41]`, `42` |

**Files needing zero edits:** `normalize.py`, `modeling_act.py`, `configuration_act.py`,
`modeling_flow.py`, `residual_wrapper.py`, `__init__.py`. All model widths flow from
`config.robot_state_feature.shape[0]` / `env_state_feature.shape[0]` /
`action_feature.shape[0]`, which `train_flow.make_config` derives from the dataset constants.

**Watch the interleaving.** `clutter_obs` is `[d0_dx, d0_dy, d0_upz, d1_dx, …]` — per-distractor
triples at obs indices `23,24,25 | 26,27,28 | 29,30,31 | 32,33,34`. It is **not**
`[all dx | all dy | all upz]`. Distractor up-axes therefore live at `[25, 28, 31, 34]`.

---

## 4. Action handling — a genuine no-op

Clutter's action terms are byte-for-byte identical to pick-place's. So:

- Collection encoding stays `a[:6] = (q_target − q_default)/0.5`, `a[6] = ±1`
  (`run_expert_v1.py:428-438`). **1 action unit = 0.5 rad.**
- Normalization: **all 7 dims mean/std normalized together, including the binary grip
  channel**, over all frames of kept demos (not masked by `train_mask`). Consequence worth
  remembering: the unnormalized output crosses zero at `mean`, not at normalized zero — so a
  hand-constructed "neutral" action in normalized space is *not* zeros.
- Inference applies no clamping and no sign quantisation; the env thresholds on `a[6] < 0`.
- The arm/grip split exists in exactly one place — `residual_core.py:194`,
  `full[:, :6] += applied`, grip passes through. `RES_ACTION_DIM`, `STEER_ACTION_DIM`,
  `ACTION_DIM` all unchanged.

---

## 5. `modeling_flow.py` — the x0 contract

```python
predict_action_chunk(batch, generator=None, x0=None) -> (B, chunk_size, action_dim)  # NORMALIZED
```

`x0` shape semantics (`modeling_flow.py:117`):

- `x0.dim() == 2` → `(chunk_size, action_dim)`, **broadcast across the batch** — the
  `fixed_x0` determinism path.
- `x0.dim() == 3` → `(B, chunk_size, action_dim)`, per-env — the `steer_x0` path.

`forward` is rectified flow: `x_tau = (1−τ)x0 + τx1`, target `v = x1 − x0`, elementwise MSE
masked by `action_is_pad`, with `key_padding_mask = pad & ~pad.all(dim=1, keepdim=True)`
threaded into decoder self-attention (the all-masked-row NaN guard).

Checkpoint (`train_flow.py:64-83`) stores `policy_state_dict`, `normalizer_state_dict`
(normalization stats ride as registered buffers, recovered by name), and a `config` dict
carrying `policy_type`, `chunk_size`, `n_action_steps`, `num_inference_steps`, `state_dim`,
`env_state_dim`, `action_dim`. `eval_act.py:178-180` asserts the stored dims against the
current `dataset.py` constants — so **old 41-D checkpoints hard-fail against a 42-D
`dataset.py`**, which is the intended behaviour but means the two cannot be mixed.

---

## 6. `eval_act.py` — controller and touchpoints

**Vectorized controller** (`eval_act.py:53-158`): `_buf` is `(N, n_action_steps, 7)`, `_idx`
is `(N,) long`; `_idx[i] == n_action_steps` means empty. `_refill` runs one batched forward
over exactly the empty envs; **`steer_x0` takes precedence over `fixed_x0`**. `peek` returns
`(base_action, phase)` without consuming; `pop` consumes exactly what `peek` returned;
`flush(ids)` is just `_idx[ids] = n_action_steps`.

**mdp touchpoints and their clutter equivalents:**

| eva_bc call | clutter replacement |
|---|---|
| `mdp.OBJECT_NAMES` | `("target",) + mdp.DISTRACTOR_NAMES` |
| `mdp.object_pos_local(env, name)` | same name, same semantics |
| `mdp.placed_mask(env) -> (N,K)` | `mdp.target_at_goal(env) -> (N,)` |
| `mdp.basket_centers_local(env)` | constant `mdp.GOAL_XY` |
| `objects_canonical` selection logic (`residual_core.py:145-156`) | **delete** — one target |

**Flush trigger** (`eval_act.py:45-50, 301-312`): thresholds `FLUSH_POS_JUMP = 0.03`,
`FLUSH_Z_DROP = 0.02`, `FLUSH_Z_ABOVE = 0.05`. For clutter these need redesign — the block's
resting root z is 0.035 and `EXTRACT_Z = 0.090`, so `FLUSH_Z_ABOVE = 0.05` no longer means
"above the rim". **The natural clutter discontinuity is a distractor's `up_z` starting to
drop** (imminent topple), which is directly available in the obs at `[25, 28, 31, 34]`.

---

## 7. The steering stack

**Control model** (`steer_core.py:1-31`): the controller stays **free-running** — z enters
only through the x0 of refills that happen while it is held. This is what made EXP07's gate 1
come back bit-exact with zero flips, and it **supersedes HANDOFF's synchronous-window
sketch**.

```
obs57 = core.build_obs(obs42, env)
z     = policy(obs57)                 # (N, 7), rl_games-clamped to [-1, 1]
core.set_steer(z)                     # x0 = alpha_x0 * tanh(z), broadcast over chunk
for _ in range(window):
    core.flush_check(env); action = core.controller.act(obs42); obs42 = env.step(action)
```

`set_steer` (`steer_core.py:54-62`): `x0 = alpha_x0 * tanh(z)`, then
`.unsqueeze(1).expand(-1, chunk_size, -1)` — one value per action dim, constant across all 50
chunk positions. With `clip_actions: 1.0` and `alpha_x0 = 1.0` the effective range is
`tanh(±1) ≈ ±0.76` per dim.

**Wrapper** (`steer_wrapper.py:45-78`): one rl_games step = one window; reward is a plain
**undiscounted sum** of the window's raw env rewards, so `gamma: 0.99` discounts *per window*
(15 env steps), not per env step. `extras` from the last inner step wins. Note `step()` fully
overrides the base wrapper, so it does **not** apply the base's `torch.clamp(actions, ±clip)`
— clamping happens only inside rl_games' `preprocess_actions`.

**The 15-D shared feature tail** (`residual_core.py:136-181`) is used by both the 64-D
residual obs and the 56-D steer obs: fingers 4, grasp bit 1, target-in-gripper pose 7,
target→goal delta 3.

For clutter, `task_features` needs: the two-object canonical selection deleted, the L2
permutation dropped, `basket_delta` replaced by `goal_delta = [GOAL_XY[0] − tx,
GOAL_XY[1] − ty, 0.035 − tz]`, and — **strongly recommended** — `min_i(up_z_i)` added as an
explicit topple-margin feature. The binding constraint of the task deserves to be surfaced
rather than left implicit, which is precisely the eva_bc EXP01 "salience, not information"
lesson applied one level up.

---

## 8. The PPO yamls

Shared: `a2c_continuous` / `continuous_a2c_logstd`, MLP `[256,128,64]` elu, `separate: False`,
`normalize_input/value: True`, `value_bootstrap: True`, `reward_shaper.scale_value: 0.01`,
`gamma 0.99`, `tau 0.95`, `lr 1e-4` adaptive (`kl_threshold 0.01`), `e_clip 0.2`,
`horizon_length 24`, `mini_epochs 8`, `critic_coef 4`, `entropy_coef 0.001`, `grad_norm 1.0`,
`bounds_loss_coef 1e-4`.

| key | residual | steer |
|---|---|---|
| `sigma_init.val` | **−2.5** | **−1.2** |
| `max_epochs` | 3000 | 200 |
| `minibatch_size` (yaml) | 768 | 12288 — **both dead**, overwritten at runtime as `horizon × num_envs // 4` |

The four load-bearing values: **`clip_actions: 1.0`** (with the EXP06 lesson written into the
yaml comment), **`sigma_init`** (chosen from gate S0, never from precedent), **`mu_init` const
0.0** (weight only — the bias stays `U(±0.125)`, which is why the diag harness probes a
`+0.125` bias condition), and **`horizon_length: 24`** which for steering means 24 *windows* =
360 env steps.

---

## 9. Ordered porting checklist

### Phase 0 — prerequisite, blocks everything
- **[THOUGHT]** Build a scripted expert for clutter extraction. Run a feasibility spike first
  (eva_bc README:250-253), before any full runner.
- **[THOUGHT]** Design the segment/outcome vocabulary. It is baked into every HDF5.
- **[MECH]** Write the HDF5 emitter — copy `run_expert_v1.py:785-802` verbatim.

### Phase 1 — dataset and dims
- **[MECH]** `dataset.py:37-42` constants; `:2-23` docstring.
- **[THOUGHT]** `dataset.py:45-98` pool filters against the new vocabulary. Keep
  `default_demo_filter`.
- **[MECH]** `dataset.py:113-207` and `normalize.py` — no edits.

### Phase 2 — model and training
- **[MECH]** `train_flow.py`, `train_act.py`, `modeling_flow.py`, `configuration_act.py` — no
  edits; verify only.
- **[THOUGHT]** Re-tune `chunk_size` / `n_action_steps`. Defaults 50/15 were for 1500-step
  pick-place episodes; clutter is 700 steps, so a 50-step chunk is 7 % of an episode vs 3.3 %.
  Constrained by L3.

### Phase 3 — eval
- **[MECH]** `--task` default → `Rebot-ClutterExtract-Play-v0`; import
  `challenge.mdp`; `--episode-length-s` → 13.8.
- **[THOUGHT]** Termination surgery (L4).
- **[THOUGHT]** Flush detector redesign (§6).
- **[THOUGHT]** Success predicate + per-episode record (L5).

### Phase 4 — residual stack
- **[MECH]** constants, `BIT_DIMS`, grip index, docstrings, `compose()` unchanged,
  `residual_wrapper.py` unchanged.
- **[THOUGHT]** `task_features` rewrite (§7), **including dropping the L2 permutation**.
- **[THOUGHT]** Decide whether the training env keeps `distractor_toppled` + `topple_penalty`
  ON. Sanity-check with `diag_training_env.py` before any full run.

### Phase 5 — steering stack
- **[MECH]** `STEER_OBS_DIM`, docstrings, `_obs41` → `_obs42` renames, `--task` defaults.
- **[THOUGHT]** Window alignment (L3) and its interaction with L4.
- **[MECH]** yamls: only `config.name` and `max_epochs`. **Keep `clip_actions: 1.0` and
  `mu_init` zero.** Re-run gate S0 to re-select `sigma_init` for this task.

### Phase 6 — support artifacts
- **[THOUGHT]** Retrain the grasp bit — needs expert post-close frames **and** on-policy
  closed-on-air negatives, so it comes after demos and a trained base. Hold the 0 %-FPR gate.
- **[THOUGHT]** `report_coverage.py` is effectively a rewrite; the meaningful clutter coverage
  axes are measured free gap, target yaw, and topple/extract/goal outcome counts. The
  `train_mask` accounting and pool counts are reusable.
- **[MECH]** `diag_training_env.py` defaults; its four fixed-action conditions are
  task-agnostic and worth keeping as the first sanity pass.

### Phase 7 — verification order (do not reorder)
1. `dataset.py` shape assertion passes on a real clutter HDF5.
2. `train_flow.py --steps 200` runs, loss decreases.
3. `eval_act.py` runs; `obs.shape == (n, 42)`; success rate finite.
4. **Residual gate 2a**: zero residual reproduces the plain eval episode-for-episode.
5. **Steering gate 1**: `--z-sigma 0 --z-bias 0` reproduces the x0-zeros base **bit-exactly**
   (`experiments/exp07_check_match.py`, exits nonzero on mismatch).
6. Only then train RL.
</content>
