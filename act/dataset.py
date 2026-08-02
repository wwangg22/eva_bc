#!/usr/bin/env python
"""HDF5 demo dataset for the reBot pick-place ACT pipeline (PLAN.md Stage 2.3).

File layout (written by the demo generator):
    data/demo_{i}/
        obs/policy   (T, 41) float32   full privileged observation
        actions      (T, 7)  float32   expert actions
        train_mask   (T,)    uint8     1 = trainable, 0 = loss-censored (expert miss/loss
                                       segments, PLAN.md section 2.2)
    attrs: success (bool), num_samples, episode_kind, segments / outcomes (JSON strings)

41-D observation layout — read from the env config, NOT guessed. Source of truth:
reBot_RL/source/reBot_RL/reBot_RL/tasks/manager_based/pick_place/pick_place_v1_env_cfg.py
ObservationsV1Cfg.PolicyCfg (concatenate_terms=True, so terms concat in declaration order),
robot = 8 joints (joint1..joint6 + joint_left + joint_right):
    [ 0: 8]  joint_pos      (mdp.joint_pos_rel, 8)   \
    [ 8:16]  joint_vel      (mdp.joint_vel_rel, 8)   / observation.state (proprio, 16)
    [16:32]  objects        (mdp.objects_canonical, 16: target pose 7 + other pose 7      \
                             + 2 placed flags, target-first canonical order)               |
    [32:34]  basket_center  (mdp.basket_center_xy, 2)                                      | observation.
    [34:41]  actions        (mdp.last_action, 7)                                           | environment_state
                                                                                          / (25)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

OBS_DIM = 41
STATE_SLICE = slice(0, 16)  # joint_pos_rel(8) + joint_vel_rel(8)
ENV_STATE_SLICE = slice(16, 41)  # objects_canonical(16) + basket_center_xy(2) + last_action(7)
STATE_DIM = 16
ENV_STATE_DIM = 25
ACTION_DIM = 7


def default_demo_filter(attrs: dict) -> bool:
    """Keep only successful episodes (the BC pool never sees failures)."""
    return bool(attrs.get("success", False))


_GRASP_ATTEMPT_RE = re.compile(r":g(\d+)$")  # seg ids look like "<name>:<tag>:g<n>"


def _nominal_clean(attrs: dict) -> bool:
    """PLAN.md section 2.2 clean-grasp criteria at attempt granularity, from the
    per-episode outcomes JSON: every first grasp attempt (":g0") that grasped did so
    with clean=True, there are no retry attempts (":g1"/":g2"...), no "missed"/"lost"
    outcomes anywhere, and no transport went via "carry-direct*"."""
    outcomes = json.loads(attrs.get("outcomes", "{}"))
    for seg_id, oc in outcomes.items():
        outcome = oc.get("outcome")
        if outcome in ("missed", "lost"):
            return False
        m = _GRASP_ATTEMPT_RE.search(seg_id)
        if m:
            if int(m.group(1)) > 0:  # any regrasp attempt disqualifies
                return False
            if outcome == "grasped" and not oc.get("clean", False):
                return False
        if str(oc.get("via", "")).startswith("carry-direct"):
            return False
    return True


def nominal_pool_filter(attrs: dict) -> bool:
    """Canonical nominal BC pool (PLAN.md section 2.2): successful nominal episodes
    where every grasped can was clean on the FIRST attempt (see _nominal_clean)."""
    return (
        bool(attrs.get("success", False))
        and attrs.get("episode_kind") == "nominal"
        and _nominal_clean(attrs)
    )


def recovery_pool_filter(attrs: dict) -> bool:
    """Recovery pool (PLAN.md sections 2.2 / 3): successful but NOT nominal-clean —
    regrasps, recovered drops, perturbation episodes. Complements nominal_pool_filter
    over successful episodes (the miss segments themselves stay loss-masked)."""
    return bool(attrs.get("success", False)) and not nominal_pool_filter(attrs)


def nominal_dagger_pool_filter(attrs: dict) -> bool:
    """DAgger training pool (PLAN.md section 3.2): the nominal-clean pool plus successful
    gated-takeover recoveries from expert/collect_dagger.py (episode_kind
    "recovery_dagger"; their policy-driven prefix is hard-zeroed in train_mask by the
    collector, so only the expert's recovery-to-success supervises)."""
    return nominal_pool_filter(attrs) or (
        attrs.get("episode_kind") == "recovery_dagger" and bool(attrs.get("success", False))
    )


def _plain_attrs(h5_attrs) -> dict:
    """h5py attrs -> plain dict (numpy scalars -> python, JSON strings left as strings)."""
    out = {}
    for k, v in h5_attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


class RebotDemoDataset(Dataset):
    """Every (demo, t) with t in [0, T) is one sample; the action chunk [t, t+chunk_size)
    is edge-padded past the episode end. ``action_is_pad[j]`` is True when t+j >= T
    (episode-end padding) OR train_mask[t+j] == 0 (expert-failure loss censor, PLAN.md
    sections 2.2 / 4.1) — both ride the same channel; the vendored ACT L1 loss multiplies
    by ~action_is_pad, so censored steps contribute no gradient.

    Demos are small (state-only) and loaded fully into RAM, so the dataset is
    multiprocessing-worker safe with no open HDF5 handles.

    ``h5_path`` may be a single path or a list of paths; the index space is the
    concatenation of all kept demos in path order (``demo_filter`` applies per demo
    across every file).
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
        self.index: list[tuple[int, int]] = []  # (demo_idx, t)
        for path in paths:
            with h5py.File(str(path), "r") as f:
                data = f["data"]
                for key in sorted(data.keys(), key=lambda k: int(k.split("_")[-1])):
                    grp = data[key]
                    attrs = _plain_attrs(grp.attrs)
                    if not demo_filter(attrs):
                        continue
                    obs = np.asarray(grp["obs/policy"], dtype=np.float32)
                    actions = np.asarray(grp["actions"], dtype=np.float32)
                    train_mask = np.asarray(grp["train_mask"], dtype=np.uint8)
                    T = obs.shape[0]
                    if obs.shape != (T, OBS_DIM) or actions.shape != (T, ACTION_DIM) or train_mask.shape != (T,):
                        raise ValueError(
                            f"{path}:{key}: unexpected shapes obs={obs.shape} "
                            f"actions={actions.shape} train_mask={train_mask.shape}"
                        )
                    d = len(self.demos)
                    self.demos.append(
                        {"name": key, "file": str(path), "obs": obs, "actions": actions, "train_mask": train_mask, "attrs": attrs}
                    )
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
        chunk[end - t :] = demo["actions"][T - 1]  # edge-pad past episode end (values masked anyway)

        is_pad = np.ones(self.chunk_size, dtype=bool)  # past-episode-end -> pad
        is_pad[: end - t] = demo["train_mask"][t:end] == 0  # expert-failure censor -> pad

        return {
            "observation.state": torch.from_numpy(obs_t[STATE_SLICE].copy()),
            "observation.environment_state": torch.from_numpy(obs_t[ENV_STATE_SLICE].copy()),
            "action": torch.from_numpy(chunk),
            "action_is_pad": torch.from_numpy(is_pad),
        }


def compute_stats(dataset: RebotDemoDataset) -> dict[str, dict[str, torch.Tensor]]:
    """Per-key mean/std over all frames of the kept demos, in the format lerobot's
    normalization expects: {key: {"mean": (dim,), "std": (dim,)}} float32 tensors
    (see act/normalize.py; upstream NormalizerProcessorStep takes the same dict)."""
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
            "std": torch.from_numpy(arr.std(axis=0).astype(np.float32)),
        }
        for key, arr in arrays.items()
    }


def load_segments(attrs: dict) -> tuple[list, dict]:
    """Decode the per-episode segments/outcomes JSON attrs (for pool selection/analysis)."""
    return json.loads(attrs.get("segments", "[]")), json.loads(attrs.get("outcomes", "{}"))
