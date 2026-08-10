# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""The pick-place vision shard loader, plus the one filter the workstation needs.

The shard contract and both student dims (23 state, 7 action) are identical to pick-place's,
so ``act.dataset_vision.VisionShardDataset`` is *imported* rather than copied -- a copy would
only be a second thing to keep in sync. What is added here is workstation-specific and would
be wrong to push upstream.

⭐ **Black wrist frames.** The wrist camera sits 77 mm behind the fingertip and 67 mm above it,
and during the grasp and the carry it passes inside scene geometry -- the inside of a mesh
renders black. Measured over the 238-episode round-1 dataset:

    1.29 % of all wrist frames (2 556 / 197 824); workspace frames: 0.00 %
    78 of 238 episodes affected (32.8 %); per-episode share median 0 %, p90 2.8 %, max 37.8 %
    longest blackout per affected episode: median 20 steps (0.4 s), max 316 (6.3 s)
    by episode decile: [0, 0, 4, 31, 234, 1221, 408, 359, 299, 0]

It is bimodal: two thirds of episodes are clean, and a handful lose most of the manoeuvre. A
black frame paired with "now close on the cube" is contradictory supervision at the phase of
the task where BC already fails, so those samples are dropped.

**Dropped as SAMPLES, not as episodes.** A black frame only spoils the sample *anchored* at
it -- the observation. The actions at those timesteps are still expert actions and are still
reachable inside chunks anchored at neighbouring good frames, so removing them from the index
costs 1.3 % of the samples instead of 33 % of the episodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # eva_bc, for act/

from act.dataset_vision import (  # noqa: E402,F401
    ACTION_DIM,
    STUDENT_STATE_DIM,
    VisionShardDataset,
    compute_stats_vision,
)

#: A frame is "black" below this mean pixel value. Fully-black frames read 0; the threshold is
#: loose enough to catch the near-black ones without touching a legitimately dark close-up.
BLACK_MEAN = 3.0


class WorkstationVisionDataset(VisionShardDataset):
    """`VisionShardDataset` with black-camera samples removed from the index."""

    def __init__(self, data_dirs, chunk_size: int = 50, success_only: bool = True,
                 drop_black: bool = True):
        super().__init__(data_dirs, chunk_size=chunk_size, success_only=success_only)
        if not drop_black:
            return
        keep, n_black = [], 0
        for e, ep in enumerate(self.episodes):
            bad = torch.zeros(ep["wrist_rgb"].shape[0], dtype=torch.bool)
            for key in ("wrist_rgb", "workspace_rgb"):
                img = ep[key]
                # integer sum, not float mean: one pass over ~20 GB either way, but this does
                # not materialise a float copy of every frame
                per_frame = img.sum(dim=(1, 2, 3), dtype=torch.int64)
                bad |= per_frame < int(BLACK_MEAN * img[0].numel())
            n_black += int(bad.sum())
            keep.extend((e, t) for t in range(len(bad)) if not bad[t])
        dropped = len(self.index) - len(keep)
        self.index = keep
        print(f"[dataset_vision] dropped {dropped} samples on black camera frames "
              f"({dropped / max(1, dropped + len(keep)):.2%}); {len(keep)} remain")
