# Provenance

Vendored from the official LeRobot repository (Apache-2.0, see `LICENSE` — verbatim copy
of the repo's LICENSE file).

- Repo: https://github.com/huggingface/lerobot
- Commit: `2aba372b4e217cc47db28e0f836859b20d1456c9` (commit date 2026-07-31)
- Vendored on: 2026-08-01
- Local clone kept at `reBot_ACT/third_party/lerobot` (shallow, same commit)

## Vendored files and modifications

### `modeling_act.py`
Source: `src/lerobot/policies/act/modeling_act.py`. Modifications (each one explicit):

1. Docstring: added a VENDORED provenance note and a note that normalization is NOT in
   the policy (upstream moved it to processor pipelines).
2. Removed `from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES,
   OBS_STATE` and `from ..pretrained import PreTrainedPolicy`; inlined the four string
   constants directly (identical values).
3. `class ACTPolicy(PreTrainedPolicy)` → `class ACTPolicy(nn.Module)`, and
   `super().__init__(config)` → `super().__init__()`. `PreTrainedPolicy` only stored
   `self.config` (which ACTPolicy re-sets itself) and provided HuggingFace Hub
   save/load — dropped to avoid the huggingface_hub/safetensors dependency stack.
   `config_class`/`name` class attrs kept.

Everything else (ACT, ACTEncoder/Decoder, ACTTemporalEnsembler, loss computation,
Apache header) is byte-identical to upstream.

Verified behaviors of this upstream version:
- **No-vision path:** the ResNet backbone is only constructed inside
  `if self.config.image_features:` (ACT.__init__), so with only
  `observation.state`/`observation.environment_state` inputs no vision tower exists and
  torchvision weights are never touched.
- **Pad semantics:** in `ACTPolicy.forward`, `valid_mask = ~batch["action_is_pad"]` and
  `l1_loss = (abs_err * valid_mask).sum() / num_valid` — i.e. `action_is_pad == True`
  means "ignore this timestep in the L1 loss" (and the same mask is fed to the VAE
  encoder as `key_padding_mask`). Our expert-failure `train_mask==0` censor rides this
  channel (PLAN.md §2.2/§4.1). Note upstream normalizes by the *valid* count, not by
  chunk length.

### `configuration_act.py`
Source: `src/lerobot/policies/act/configuration_act.py`. Modifications:

1. Header comment: added VENDORED provenance note.
2. Removed `from lerobot.configs import NormalizationMode, PreTrainedConfig` and
   `from lerobot.optim import AdamWConfig`.
3. Inlined `FeatureType`, `NormalizationMode`, `PolicyFeature` verbatim from
   `src/lerobot/configs/types.py`; inlined `OBS_STATE`/`ACTION` constants.
4. Replaced lerobot's `PreTrainedConfig` base (draccus choice registry + HubMixin +
   device auto-selection) with a minimal local dataclass keeping only what ACT uses:
   `n_obs_steps`, `input_features`, `output_features`, `device`, and the
   `robot_state_feature` / `env_state_feature` / `image_features` / `action_feature`
   properties (property bodies copied verbatim from `src/lerobot/configs/policies.py`).
   Its `__post_init__` is a no-op (no device auto-selection).
5. Removed `@PreTrainedConfig.register_subclass("act")` decorator and the
   `get_optimizer_preset()` / `get_scheduler_preset()` methods (draccus/`lerobot.optim`
   dependencies; `train_act.py` builds AdamW directly from `optimizer_lr` /
   `optimizer_weight_decay`).

The `ACTConfig` dataclass body (fields, defaults, `__post_init__` validation,
`validate_features`, delta-index properties, Apache header) is otherwise unmodified.

### `normalize.py` (adapted, not verbatim)
Source: MEAN_STD branch of `_NormalizationMixin._apply_transform` in
`src/lerobot/processor/normalize_processor.py`. Upstream applies normalization via
`PolicyProcessorPipeline` (heavy dependency surface); we keep only the math, with
identical semantics: `(x - mean) / (std + eps)`, `eps = 1e-8`, inverse `x * std + mean`.
Stats dict format matches upstream `dataset_stats`: `{key: {"mean": Tensor, "std":
Tensor}}` (see `make_act_pre_post_processors` in
`src/lerobot/policies/act/processor_act.py`).

### `LICENSE`
Verbatim copy of the repo's Apache-2.0 `LICENSE`.

## Original (non-vendored) files
- `dataset.py` — reBot HDF5 demo dataset + `compute_stats` (ours).
- `train_act.py` — minimal training skeleton (ours).
- `__init__.py` — package marker (ours).

## 41-D observation layout (read from the env config, not guessed)
Source of truth: `reBot_RL/source/reBot_RL/reBot_RL/tasks/manager_based/pick_place/
pick_place_v1_env_cfg.py`, `ObservationsV1Cfg.PolicyCfg` (used by
`Rebot-PickPlace-Play-v1` via `RebotPickPlaceV1EnvCfg_PLAY`; `concatenate_terms=True`
→ terms concatenate in declaration order). Robot has 8 joints (`joint[1-6]` +
`joint_left`, `joint_right`).

| slice    | term            | mdp function           | dim |
|----------|-----------------|------------------------|-----|
| `[0:8]`  | `joint_pos`     | `mdp.joint_pos_rel`    | 8   |
| `[8:16]` | `joint_vel`     | `mdp.joint_vel_rel`    | 8   |
| `[16:32]`| `objects`       | `mdp.objects_canonical`| 16  |
| `[32:34]`| `basket_center` | `mdp.basket_center_xy` | 2   |
| `[34:41]`| `actions`       | `mdp.last_action`      | 7   |

`observation.state` = `[0:16]` (proprio, 16), `observation.environment_state` =
`[16:41]` (25). Matches PLAN.md §2.3's 16/25 split.
