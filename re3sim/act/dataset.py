# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""HDF5 demo dataset for ``Rebot-Workstation-PickPlace1-v0``.

A self-contained copy of ``eva_bc/clutter/act/dataset.py`` with the workstation observation
layout. It is a copy and not an import for the reason recorded in ``clutter/act/__init__.py``:
those modules hardcode their dimensions as module-level constants, and rebinding another
module's globals to change them is invisible at the call site -- a checkpoint trained against
the wrong split loads without complaint and simply performs badly.

Written by ``re3sim/expert/collect_demos.py``:

    /data/demo_<i>/obs/policy   (T, 41) float32   the full privileged observation
    /data/demo_<i>/actions      (T,  7) float32   the action actually submitted to env.step
    /data/demo_<i>/train_mask   (T,)    uint8     1 = trainable, 0 = loss-censored
    /data/demo_<i>/phase        (T,)    int8      PH_APPROACH .. PH_RETREAT
    attrs: success (bool), planned (bool), seed, env_index

**41-D observation layout -- read from the env config, NOT guessed.** Source of truth:
``eva_rl/source/reBot_RL/reBot_RL/tasks/manager_based/re3sim/workstation_env_cfg.py``
``ObservationsCfg.PolicyCfg`` (``concatenate_terms=True``, so terms concatenate in
declaration order), robot = 8 joints (joint1..joint6 + joint_left + joint_right)::

    [ 0: 8]  joint_pos    (mdp.joint_pos_rel, 8)                       \\
    [ 8:16]  joint_vel    (mdp.joint_vel_rel, 8)                       / observation.state (16)
    [16:23]  target_pose  (mdp.target_pose, 7: pos + quat xyzw)        \\
    [23:27]  box_pose     (mdp.box_pose, 4: centre xy + cos/sin yaw)    |
    [27:33]  clutter      (mdp.clutter_obs, 6: 2 bodies x               | observation.
                          (dx, dy relative to the cube, up-axis z))     | environment_state
    [33:34]  placed       (mdp.target_placed_flag, 1)                   | (25)
    [34:41]  actions      (mdp.last_action, 7)                         /

The env-state width is 25, which is **numerically identical to pick-place's**. That
coincidence is a trap, not a convenience: a pick-place checkpoint loads into this task
silently. `policy_runner.load_checkpoint` therefore asserts on a `task` tag rather than on
the widths alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

OBS_DIM = 41
STATE_SLICE = slice(0, 16)      # joint_pos_rel(8) + joint_vel_rel(8)
ENV_STATE_SLICE = slice(16, 41)  # target_pose(7) + box(4) + clutter(6) + placed(1) + last(7)
STATE_DIM = 16
ENV_STATE_DIM = 25
ACTION_DIM = 7
TASK_TAG = "re3sim_workstation"


def default_demo_filter(attrs: dict) -> bool:
    """Keep only successful episodes (the BC pool never sees failures).

    ``collect_demos.py`` writes the failures too, because the failure taxonomy is what tells
    the next session which leg of the manoeuvre to fix, and regenerating a dataset to answer a
    question about its failures is how evidence gets lost. They are dropped here, at training
    time, not at collection time.
    """
    return bool(attrs.get("success", False))


def _plain_attrs(h5_attrs) -> dict:
    out = {}
    for k, v in h5_attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


class WorkstationDemoDataset(Dataset):
    """Every ``(demo, t)`` is one sample; the chunk ``[t, t+chunk_size)`` is edge-padded past
    the episode end. ``action_is_pad[j]`` is True when ``t+j >= T`` (episode-end padding) OR
    ``train_mask[t+j] == 0`` (loss censor) -- both ride the same channel, and the vendored ACT
    loss multiplies by ``~action_is_pad``, so a censored step contributes no gradient.

    Demos are state-only and small, so everything is loaded into RAM and no HDF5 handle stays
    open: the dataset is multiprocessing-worker safe.
    """

    def __init__(
        self,
        h5_path: str | Path | Sequence[str | Path],
        chunk_size: int = 50,
        demo_filter: Callable[[dict], bool] | None = None,
    ):
        self.chunk_size = chunk_size
        demo_filter = default_demo_filter if demo_filter is None else demo_filter
        paths = [h5_path] if isinstance(h5_path, (str, Path)) else list(h5_path)

        self.demos: list[dict] = []
        self.index: list[tuple[int, int]] = []
        self.n_rejected = 0
        for path in paths:
            with h5py.File(str(path), "r") as f:
                data = f["data"]
                for key in sorted(data.keys(), key=lambda k: int(k.split("_")[-1])):
                    grp = data[key]
                    attrs = _plain_attrs(grp.attrs)
                    if not demo_filter(attrs):
                        self.n_rejected += 1
                        continue
                    obs = np.asarray(grp["obs/policy"], dtype=np.float32)
                    actions = np.asarray(grp["actions"], dtype=np.float32)
                    train_mask = np.asarray(grp["train_mask"], dtype=np.uint8)
                    T = obs.shape[0]
                    if obs.shape != (T, OBS_DIM) or actions.shape != (T, ACTION_DIM) \
                            or train_mask.shape != (T,):
                        raise ValueError(
                            f"{path}:{key}: unexpected shapes obs={obs.shape} "
                            f"actions={actions.shape} train_mask={train_mask.shape} "
                            f"(expected obs (T,{OBS_DIM}))")
                    d = len(self.demos)
                    self.demos.append({"name": key, "file": str(path), "obs": obs,
                                       "actions": actions, "train_mask": train_mask,
                                       "attrs": attrs})
                    self.index.extend((d, t) for t in range(T))
        if not self.demos:
            raise ValueError(f"{[str(p) for p in paths]}: no demos passed the filter")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        d, t = self.index[i]
        demo = self.demos[d]
        obs_t = demo["obs"][t]
        T = demo["obs"].shape[0]
        end = min(t + self.chunk_size, T)

        chunk = np.empty((self.chunk_size, ACTION_DIM), dtype=np.float32)
        chunk[: end - t] = demo["actions"][t:end]
        chunk[end - t:] = demo["actions"][T - 1]      # edge-pad (masked anyway)

        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[: end - t] = demo["train_mask"][t:end] == 0

        return {
            "observation.state": torch.from_numpy(obs_t[STATE_SLICE].copy()),
            "observation.environment_state": torch.from_numpy(obs_t[ENV_STATE_SLICE].copy()),
            "action": torch.from_numpy(chunk),
            "action_is_pad": torch.from_numpy(is_pad),
        }


def compute_stats(dataset: WorkstationDemoDataset) -> dict[str, dict[str, torch.Tensor]]:
    """Per-key mean/std over all frames of the kept demos, in the format the vendored
    normalizer expects: ``{key: {"mean": (dim,), "std": (dim,)}}`` float32 tensors."""
    obs = np.concatenate([d["obs"] for d in dataset.demos], axis=0)
    actions = np.concatenate([d["actions"] for d in dataset.demos], axis=0)
    arrays = {
        "observation.state": obs[:, STATE_SLICE],
        "observation.environment_state": obs[:, ENV_STATE_SLICE],
        "action": actions,
    }
    return {
        key: {
            "mean": torch.from_numpy(arr.mean(axis=0).astype(np.float32)),
            # A constant channel has std 0 and would divide to inf. `placed` is exactly that
            # in a pool of successes truncated before the drop, and `last_action[6]` is very
            # nearly it. Floor rather than drop, so the layout stays fixed.
            "std": torch.from_numpy(
                np.maximum(arr.std(axis=0), 1e-4).astype(np.float32)),
        }
        for key, arr in arrays.items()
    }
