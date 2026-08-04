# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""Loading and running a trained clutter flow-BC checkpoint.

Shared by `eval_flow.py` (which measures) and `record_video.py` (which films). They must
drive the policy identically or the film is not of the thing that was measured, so the
controller lives in exactly one place. Nothing here parses arguments or launches the
simulator, so it is importable from a script that does.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))          # eva_bc, for the vendored act/ package
sys.path.insert(0, str(_HERE))
from act.modeling_flow import FlowMatchingPolicy  # noqa: E402
from act.normalize import MeanStdNormalizer  # noqa: E402
from dataset import ACTION_DIM, ENV_STATE_SLICE, STATE_SLICE  # noqa: E402
from train_flow import make_config  # noqa: E402


class ChunkController:
    """Per-env action queue over a chunk policy: predict `chunk_size`, commit `n_action_steps`.

    Vectorized rather than a python deque per env -- `_buf` (N, n_action_steps, 7) holds each
    env's committed window and `_idx` (N,) the position of its next action, with
    `_idx == n_action_steps` meaning empty. One batched forward refills every empty queue.
    Normalization happens here, outside the policy: this LeRobot version keeps it in an
    external processor (`act/PROVENANCE.md`).
    """

    def __init__(self, policy, stats, n_action_steps: int, chunk_size: int, device):
        assert 1 <= n_action_steps <= chunk_size, (n_action_steps, chunk_size)
        self.policy, self.n_action_steps, self.chunk_size = policy, n_action_steps, chunk_size
        self.device = torch.device(device)
        self.normalizer = MeanStdNormalizer(stats).to(self.device)
        self._buf = self._idx = None

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device, dtype=torch.float32)
        n = obs.shape[0]
        if self._buf is None:
            self._buf = torch.zeros(n, self.n_action_steps, ACTION_DIM, device=self.device)
            self._idx = torch.full((n,), self.n_action_steps, dtype=torch.long,
                                   device=self.device)
        empty = (self._idx >= self.n_action_steps).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel():
            sub = obs[empty]
            batch = self.normalizer.normalize({
                "observation.state": sub[:, STATE_SLICE],
                "observation.environment_state": sub[:, ENV_STATE_SLICE]})
            chunk = self.policy.predict_action_chunk(batch)[:, : self.n_action_steps]
            self._buf[empty] = self.normalizer.unnormalize("action", chunk)
            self._idx[empty] = 0
        a = self._buf[torch.arange(self._buf.shape[0], device=self.device), self._idx]
        self._idx += 1
        return a

    def reset(self, env_ids=None):
        if self._idx is not None:
            if env_ids is None:
                self._idx[:] = self.n_action_steps
            else:
                self._idx[env_ids.to(self.device)] = self.n_action_steps


def load_checkpoint(path, device):
    """`clutter/act/train_flow.py` checkpoint -> (policy, stats, config)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg.get("policy_type") == "flow", cfg
    assert cfg["state_dim"] == STATE_SLICE.stop - STATE_SLICE.start, cfg
    assert cfg["env_state_dim"] == ENV_STATE_SLICE.stop - ENV_STATE_SLICE.start, (
        f"{cfg} -- an env_state_dim of 25 means the checkpoint was trained on the "
        f"pick-place 41-D layout")
    stats = {}
    for name, tensor in ckpt["normalizer_state_dict"].items():
        if name.endswith("_mean"):
            key = name[: -len("_mean")].replace("__", ".")
            stats[key] = {"mean": tensor,
                          "std": ckpt["normalizer_state_dict"][name[: -len("_mean")] + "_std"]}
    config = make_config(SimpleNamespace(
        chunk_size=cfg["chunk_size"], n_action_steps=cfg["n_action_steps"],
        num_inference_steps=cfg.get("num_inference_steps", 10), device=str(device)))
    policy = FlowMatchingPolicy(config)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device).eval()
    return policy, stats, {**cfg, "step": ckpt.get("step")}
