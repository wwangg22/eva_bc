#!/usr/bin/env python
"""Vision-shard dataset for slot champion-distillation (VISION_PLAN section 8).

Ported from ``act/dataset_vision.py`` (eva_bc EXP08). The only substantive change is the
privileged array's name and width: pick-place carries a 41-D teacher obs, the slot task a 34-D
one. The student contract is identical at 23-D, so the asserts port unchanged.

Loads the per-episode .pt shards written by scripts/collect_vision.py:
    wrist_rgb / workspace_rgb  (T, 90, 160, 3) uint8
    proprio                    (T, 23) float32
    obs34                      (T, 34) float32   TEACHER-ONLY -- never loaded here
    actions                    (T, 7)  float32
    success                    bool

Samples are (obs_t, action chunk t:t+chunk_size) pairs over every t of every kept
episode, mirroring act/dataset.py's RebotDemoDataset:
    observation.state             (23,)  float32
    observation.images.wrist      (3, 90, 160) float32 in [0, 1]
    observation.images.workspace  (3, 90, 160) float32 in [0, 1]
    action                        (chunk_size, 7) float32
    action_is_pad                 (chunk_size,) bool  (past-episode-end padding)

The privileged obs34 array is deliberately NOT read (VISION_PLAN section 0: nothing
privileged may reach the student pipeline -- and for the slot task that includes
``slot_frame[3] = is_inserted``, which is the success predicate itself). Images stay uint8 in
RAM (~86 KB/step for both cameras) and are converted per sample.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

ACTION_DIM = 7
STUDENT_STATE_DIM = 23

# Keys a student sample is built from. obs34 is teacher-only and must never be here.
_STUDENT_KEYS = ("wrist_rgb", "workspace_rgb", "proprio", "actions")
# DAgger variant (scripts/collect_vision_dagger.py): champion chunk labels instead
# of executed actions; auto-detected via the label_chunks key.
_DAGGER_KEYS = ("wrist_rgb", "workspace_rgb", "proprio", "label_chunks")


class VisionShardDataset(Dataset):
    """RAM-preloaded (uint8 images) dataset over collect_vision.py shards.

    Two shard formats, mixable freely: executed-action episodes (chunks sliced from
    the action stream, success-filtered) and DAgger chunk-labeled episodes (champion
    labels used directly; ALL kept — the labels are champion-quality regardless of
    the student's outcome, and failed episodes are exactly where DAgger helps)."""

    def __init__(self, data_dirs: list[str], chunk_size: int = 50, success_only: bool = True):
        self.chunk_size = chunk_size
        self.render: dict | None = None
        self.episodes: list[dict[str, torch.Tensor]] = []
        n_skipped = n_dagger = 0
        for d in data_dirs:
            for shard_path in sorted(Path(d).glob("ep_*.pt")):
                shard = torch.load(shard_path, map_location="cpu")
                is_dagger = "label_chunks" in shard
                if not is_dagger and success_only and not shard["success"]:
                    n_skipped += 1
                    continue
                assert shard["proprio"].shape[1] == STUDENT_STATE_DIM, shard_path
                if is_dagger:
                    assert shard["label_chunks"].shape[1:] == (chunk_size, ACTION_DIM), shard_path
                    n_dagger += 1
                # The render config is the dataset's IDENTITY, not an input: EXP08 scored
                # 0.0%/1.6% evaluating a DLSS-trained student under AA-off. Carry it so the
                # checkpoint can record it and eval can refuse a mismatch.
                r = shard.get('render')
                if r is not None:
                    if self.render is None:
                        self.render = r
                    else:
                        assert r == self.render, f'render config differs: {shard_path}'
                # keep ONLY the student keys -- drop obs34 (privileged) immediately
                self.episodes.append({k: shard[k] for k in (_DAGGER_KEYS if is_dagger else _STUDENT_KEYS)})
        if not self.episodes:
            raise ValueError(f"no episodes loaded from {data_dirs}")
        self.index: list[tuple[int, int]] = [
            (e, t)
            for e, ep in enumerate(self.episodes)
            for t in range(ep["label_chunks" if "label_chunks" in ep else "actions"].shape[0])
        ]
        self._n_dagger = n_dagger
        n_bytes = sum(ep["wrist_rgb"].numel() + ep["workspace_rgb"].numel() for ep in self.episodes)
        print(
            f"[dataset_vision] {len(self.episodes)} episodes ({n_dagger} DAgger-labeled, "
            f"{n_skipped} filtered out), {len(self.index)} samples, images ~{n_bytes / 1e9:.1f} GB RAM"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        e, t = self.index[i]
        ep = self.episodes[e]
        if "label_chunks" in ep:  # DAgger shard: champion chunk label, nothing padded
            chunk = ep["label_chunks"][t]
            is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        else:
            T = ep["actions"].shape[0]
            end = min(t + self.chunk_size, T)
            chunk = torch.empty(self.chunk_size, ACTION_DIM, dtype=torch.float32)
            chunk[: end - t] = ep["actions"][t:end]
            chunk[end - t :] = ep["actions"][T - 1]  # edge-pad past episode end (masked anyway)
            is_pad = torch.ones(self.chunk_size, dtype=torch.bool)
            is_pad[: end - t] = False

        return {
            "observation.state": ep["proprio"][t],
            "observation.images.wrist": ep["wrist_rgb"][t].permute(2, 0, 1).float() / 255.0,
            "observation.images.workspace": ep["workspace_rgb"][t].permute(2, 0, 1).float() / 255.0,
            "action": chunk,
            "action_is_pad": is_pad,
        }


def compute_stats_vision(dataset: VisionShardDataset) -> dict[str, dict[str, torch.Tensor]]:
    """Mean/std for state and action only -- images use fixed ImageNet constants inside
    the model (act/modeling_flow_vision.py), so they never enter the normalizer."""
    proprio = torch.cat([ep["proprio"] for ep in dataset.episodes], dim=0)
    actions = torch.cat(
        [
            ep["actions"] if "actions" in ep else ep["label_chunks"].reshape(-1, ACTION_DIM)
            for ep in dataset.episodes
        ],
        dim=0,
    )
    return {
        "observation.state": {"mean": proprio.mean(dim=0), "std": proprio.std(dim=0)},
        "action": {"mean": actions.mean(dim=0), "std": actions.std(dim=0)},
    }
