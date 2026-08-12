#!/usr/bin/env python
"""Vision-shard dataset for EXP08 champion-distillation (step 3).

Loads the per-episode .pt shards written by experiments/exp08_collect.py:
    wrist_rgb / workspace_rgb  (T, 90, 160, 3) uint8
    proprio                    (T, 23) float32
    obs41                      (T, 41) float32   TEACHER-ONLY -- never loaded here
    actions                    (T, 7)  float32
    success                    bool

Samples are (obs_t, action chunk t:t+chunk_size) pairs over every t of every kept
episode, mirroring act/dataset.py's RebotDemoDataset:
    observation.state             (23,)  float32
    observation.images.wrist      (3, 90, 160) float32 in [0, 1]
    observation.images.workspace  (3, 90, 160) float32 in [0, 1]
    action                        (chunk_size, 7) float32
    action_is_pad                 (chunk_size,) bool  (past-episode-end padding)

The privileged obs41 array is deliberately NOT read (EXP08 section 4: nothing
privileged may reach the student pipeline). Images are held in RAM as JPEG (~12 KB/step
for both cameras vs ~86 KB raw) and decoded per sample (~0.1 ms/frame, cv2).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ACTION_DIM = 7
STUDENT_STATE_DIM = 23

#: In-RAM JPEG quality. Frames round-trip through cv2 without RGB<->BGR conversion —
#: symmetric, so the student sees its original channel order; only the (negligible)
#: chroma-subsampling error differs. 90 sits far above the train-time augmentation's
#: own JPEG floor (q~30 at strength 1), so it adds no distribution shift that matters.
_JPEG_Q = 90

_IMAGE_KEYS = ("wrist_rgb", "workspace_rgb")


def _compress_frames(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """(T,H,W,3) uint8 -> (one flat JPEG byte buffer, offsets, per-frame int sums).

    One numpy buffer per camera per episode, NOT a list of `bytes`: DataLoader workers
    fork, and python objects' refcounts write to their pages on every access, so a
    blob LIST would be copy-on-write-duplicated per worker over a long run. Numpy
    array pages are never written after init. The per-frame integer sums come free in
    this pass and serve the black-frame audit without decoding anything later."""
    sums = torch.from_numpy(arr.reshape(arr.shape[0], -1).sum(axis=1, dtype=np.int64))
    blobs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])[1] for f in arr]
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    np.cumsum([b.size for b in blobs], out=offsets[1:])
    buf = np.empty(int(offsets[-1]), dtype=np.uint8)
    for b, o in zip(blobs, offsets[:-1]):
        buf[o:o + b.size] = b.reshape(-1)
    return buf, offsets, sums


def _decode_frame(ep: dict, key: str, t: int) -> torch.Tensor:
    o = ep[key + "_off"]
    img = cv2.imdecode(ep[key][o[t]:o[t + 1]], cv2.IMREAD_COLOR)
    return torch.from_numpy(img)

# Keys a student sample is built from. obs41 is teacher-only and must never be here.
_STUDENT_KEYS = ("wrist_rgb", "workspace_rgb", "proprio", "actions")
# DAgger variant (experiments/exp08_dagger_collect.py): champion chunk labels instead
# of executed actions; auto-detected via the label_chunks key.
_DAGGER_KEYS = ("wrist_rgb", "workspace_rgb", "proprio", "label_chunks")


class VisionShardDataset(Dataset):
    """In-RAM JPEG-compressed dataset over exp08 collection shards (~7x smaller than raw).

    Two shard formats, mixable freely: executed-action episodes (chunks sliced from
    the action stream, success-filtered) and DAgger chunk-labeled episodes (champion
    labels used directly; ALL kept — the labels are champion-quality regardless of
    the student's outcome, and failed episodes are exactly where DAgger helps)."""

    def __init__(self, data_dirs: list[str], chunk_size: int = 50, success_only: bool = True,
                 augment=None):
        # Optional callable (H, W, 3) uint8 -> (H, W, 3) uint8 applied per sample per camera
        # (act/augment_vision.py). Train-time only — eval never augments.
        self.augment = augment
        self.chunk_size = chunk_size
        self.episodes: list[dict[str, torch.Tensor]] = []
        n_skipped = n_dagger = 0
        for d in data_dirs:
            for shard_path in sorted(Path(d).glob("ep_*.pt")):
                # mmap the read (no anon copy of the raw images), then re-store the frames
                # as in-RAM JPEG (~7x smaller). Both halves are load-bearing on the 31 GB
                # box (2026-08-11, 4 datasets = ~66 GB raw): anon-preloading OOM-killed the
                # trainer at the 26 GB cgroup cap, and TRAINING off the mmap thrashed page
                # cache hard enough that systemd-oomd pressure-killed it (85.8% > 50%).
                shard = torch.load(shard_path, map_location="cpu", mmap=True)
                is_dagger = "label_chunks" in shard
                if not is_dagger and success_only and not shard["success"]:
                    n_skipped += 1
                    continue
                assert shard["proprio"].shape[1] == STUDENT_STATE_DIM, shard_path
                if is_dagger:
                    assert shard["label_chunks"].shape[1:] == (chunk_size, ACTION_DIM), shard_path
                    n_dagger += 1
                # keep ONLY the student keys -- drop obs41 (privileged) immediately
                ep: dict = {}
                for k in (_DAGGER_KEYS if is_dagger else _STUDENT_KEYS):
                    if k in _IMAGE_KEYS:
                        arr = shard[k].numpy()
                        ep["frame_numel"] = int(arr[0].size)
                        ep[k], ep[k + "_off"], ep[k + "_sums"] = _compress_frames(arr)
                    else:
                        # clone: small, and drops the last reference into the mmap so the
                        # file mapping is released instead of accumulating per shard
                        ep[k] = shard[k].clone()
                self.episodes.append(ep)
        if not self.episodes:
            raise ValueError(f"no episodes loaded from {data_dirs}")
        self.index: list[tuple[int, int]] = [
            (e, t)
            for e, ep in enumerate(self.episodes)
            for t in range(ep["label_chunks" if "label_chunks" in ep else "actions"].shape[0])
        ]
        self._n_dagger = n_dagger
        n_bytes = sum(ep["wrist_rgb"].size + ep["workspace_rgb"].size for ep in self.episodes)
        print(
            f"[dataset_vision] {len(self.episodes)} episodes ({n_dagger} DAgger-labeled, "
            f"{n_skipped} filtered out), {len(self.index)} samples, "
            f"images ~{n_bytes / 1e9:.1f} GB as in-RAM JPEG q{_JPEG_Q}"
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

        wrist = _decode_frame(ep, "wrist_rgb", t)
        workspace = _decode_frame(ep, "workspace_rgb", t)
        if self.augment is not None:
            wrist, workspace = self.augment(wrist), self.augment(workspace)
        return {
            "observation.state": ep["proprio"][t],
            "observation.images.wrist": wrist.permute(2, 0, 1).float() / 255.0,
            "observation.images.workspace": workspace.permute(2, 0, 1).float() / 255.0,
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
