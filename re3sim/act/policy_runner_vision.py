# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Launcher-free runtime for the workstation vision student: load a checkpoint, drive it.

Split out of ``eval_flow_vision.py`` for the same reason ``policy_runner.py`` is split out of
``eval_flow.py``: those scripts build an ``AppLauncher`` and call ``parse_args`` at *module
level*, so importing one from another tool re-parses that tool's argv and starts a second Kit.
Anything that has to run the student from inside another program -- DAgger collection, most of
all -- imports this instead.

Nothing here touches the simulator, and nothing here reads a privileged observation:
``build_student_batch`` is the entire input surface, and it is two camera buffers plus
``obs41[0:16]`` (joint position and velocity) and ``obs41[34:41]`` (the policy's own last
action). Both of those the real rig publishes; ``obs41[16:34]`` -- cube pose, box pose,
clutter offsets, placed flag -- is never read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # eva_bc, for act/
sys.path.insert(0, str(Path(__file__).resolve().parent))       # for train_flow_vision

from act.configuration_act import ACTConfig, FeatureType, PolicyFeature  # noqa: E402
from act.dataset_vision import ACTION_DIM  # noqa: E402
from act.modeling_flow_vision import FlowMatchingVisionPolicy  # noqa: E402
from act.normalize import MeanStdNormalizer  # noqa: E402
from train_flow_vision import IMAGE_KEYS, TASK_TAG  # noqa: E402


def load_vision_checkpoint(path: str, device, n_action_steps: int | None = None):
    """-> (policy, normalizer, config, cam_shape). Refuses a checkpoint from another task."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    if cfg.get("task") != TASK_TAG:
        # The student dims here (23 state, 7 action, two cameras) are numerically identical to
        # pick-place's, so a foreign checkpoint loads without complaint and merely performs
        # badly -- which reads as a training problem rather than as the wrong file.
        raise SystemExit(f"{path} is task={cfg.get('task')!r}, expected {TASK_TAG!r}")
    cam_shape = tuple(int(v) for v in cfg["cam_shape"])
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE,
                                               shape=(cfg["state_dim"],)),
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=cam_shape) for k in IMAGE_KEYS},
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION,
                                                 shape=(cfg["action_dim"],))},
        chunk_size=cfg["chunk_size"],
        n_action_steps=n_action_steps or cfg["n_action_steps"],
        temporal_ensemble_coeff=None,
        use_vae=False,
        dim_model=512,
        device=str(device),
    )
    config.num_inference_steps = cfg.get("num_inference_steps", 10)
    policy = FlowMatchingVisionPolicy(config).to(device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()
    # The normalizer registers its stats as buffers named `<key>_mean` / `<key>_std` with dots
    # escaped to `__`; same reconstruction `policy_runner.load_checkpoint` does.
    stats = {}
    for name, tensor in ck["normalizer_state_dict"].items():
        if name.endswith("_mean"):
            key = name[: -len("_mean")].replace("__", ".")
            stats[key] = {"mean": tensor,
                          "std": ck["normalizer_state_dict"][name[: -len("_mean")] + "_std"]}
    normalizer = MeanStdNormalizer(stats).to(device)
    return policy, normalizer, config, cam_shape


def build_student_batch(obs41, cams: dict, device):
    """⭐ The no-privileged-info contract, in one function.

    ``cams`` maps the policy's camera name (``wrist`` / ``workspace``) to an Isaac Lab
    ``Camera``. The caller is responsible for having rendered and ``update``d them this step;
    doing it here would hide the render from the loop, and under DLSS *when* the render happens
    relative to the rest of the step changes the pixels.
    """
    state = torch.cat([obs41[:, 0:16], obs41[:, 34:41]], dim=1).float()
    out = {"observation.state": state.to(device)}
    for key, cam in cams.items():
        rgb = cam.data.output["rgb"][..., :3]
        out[f"observation.images.{key}"] = rgb.permute(0, 3, 1, 2).float().to(device) / 255.0
    return out


class VisionController:
    """Predict `chunk_size` actions, execute the first `n_action_steps`, re-predict.

    Chunk commitment is load-bearing, not a tunable: eva_bc measured 59.4 / 32.8 / 3.1 / 0 / 0 %
    at n_action_steps 15 / 8 / 4 / 2 / 1 (EXP02).
    """

    def __init__(self, policy, normalizer, config, device):
        self.policy, self.normalizer, self.config = policy, normalizer, config
        self.n_action_steps = config.n_action_steps
        self.device = torch.device(device)
        self._buf = None
        self._idx = None

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        n = batch["observation.state"].shape[0]
        if self._buf is None:
            self._buf = torch.zeros(n, self.n_action_steps, ACTION_DIM, device=self.device)
            self._idx = torch.full((n,), self.n_action_steps, dtype=torch.long,
                                   device=self.device)
        empty = (self._idx >= self.n_action_steps).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel():
            sub = self.normalizer.normalize({k: v[empty] for k, v in batch.items()})
            chunk = self.policy.predict_action_chunk(sub)[:, : self.n_action_steps]
            self._buf[empty] = self.normalizer.unnormalize("action", chunk)
            self._idx[empty] = 0
        a = self._buf[torch.arange(n, device=self.device), self._idx]
        self._idx += 1
        return a

    def reset(self, env_ids=None) -> None:
        if self._idx is None:
            return
        if env_ids is None:
            self._idx[:] = self.n_action_steps
        else:
            ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids)
            self._idx[ids.to(self.device)] = self.n_action_steps
