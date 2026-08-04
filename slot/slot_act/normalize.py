#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ADAPTED from https://github.com/huggingface/lerobot @ 2aba372b4e217cc47db28e0f836859b20d1456c9
# (src/lerobot/processor/normalize_processor.py, MEAN_STD branch of
# _NormalizationMixin._apply_transform). Upstream lerobot applies normalization in a
# processor pipeline outside the policy; we keep only the MEAN_STD math it uses:
#   normalize:   (x - mean) / (std + eps)      with eps = 1e-8
#   unnormalize:  x * std + mean
# Stats format matches upstream: {key: {"mean": Tensor, "std": Tensor}}.

import torch
from torch import Tensor, nn


class MeanStdNormalizer(nn.Module):
    """Per-key mean/std normalizer. Stats are registered as buffers so they ride along in
    ``state_dict()`` (checkpoints) and ``.to(device)``."""

    def __init__(self, stats: dict[str, dict[str, Tensor]], eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.keys = sorted(stats.keys())
        for key, s in stats.items():
            self.register_buffer(self._bname(key) + "_mean", torch.as_tensor(s["mean"], dtype=torch.float32))
            self.register_buffer(self._bname(key) + "_std", torch.as_tensor(s["std"], dtype=torch.float32))

    @staticmethod
    def _bname(key: str) -> str:
        return key.replace(".", "__")

    def _mean_std(self, key: str) -> tuple[Tensor, Tensor]:
        return getattr(self, self._bname(key) + "_mean"), getattr(self, self._bname(key) + "_std")

    def normalize(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Returns a shallow copy of ``batch`` with every key we have stats for normalized."""
        out = dict(batch)
        for key in self.keys:
            if key in out:
                mean, std = self._mean_std(key)
                out[key] = (out[key] - mean) / (std + self.eps)
        return out

    def unnormalize(self, key: str, tensor: Tensor) -> Tensor:
        mean, std = self._mean_std(key)
        return tensor * std + mean

    def stats_dict(self) -> dict[str, dict[str, Tensor]]:
        return {key: dict(zip(("mean", "std"), self._mean_std(key))) for key in self.keys}
