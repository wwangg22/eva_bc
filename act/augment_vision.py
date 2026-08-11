#!/usr/bin/env python
"""Train-time image corruption for the vision student (sensor-level domain randomisation).

The in-sim DR (reBot_RL `visual_dr.py`) owns everything that changes what the world IS —
camera pose, lighting, backgrounds. This module owns what the SENSOR does to it: defocus,
sensor noise, exposure error, compression, occlusion. Split that way deliberately: sensor
effects are pixel-exact, free and infinitely diverse when applied at train time, and
simulating them in RTX would buy nothing but render cost (05_VISUAL_DR.md §2).

Applied per SAMPLE, not per episode: the policy maps one frame to one chunk, so per-sample
independence is strictly more diversity at zero cost. Each camera draws independently — two
physical sensors do not share a noise process.

All magnitudes scale with ``strength`` (the sweep knob for the robustness matrix; 1.0 is
the plan of record). Operates on (H, W, 3) uint8 and returns the same, so
``VisionShardDataset.__getitem__`` applies it just before the float conversion.
"""

from __future__ import annotations

import torch


class VisionAugment:
    """Callable (H, W, 3) uint8 -> (H, W, 3) uint8. torch-only + torchvision codecs."""

    def __init__(self, strength: float = 1.0):
        self.s = float(strength)

    # each op draws its own gate so a sample can get any subset. Gate probabilities scale
    # with min(strength, 1) and every magnitude scales with strength, so --augment
    # converges to identity as strength -> 0 and the strength ablation measures ONE knob
    # (review finding 2026-08-11: JPEG used to fire at full severity at every strength).
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        s = self.s
        if s <= 0.0:
            return img
        g = min(1.0, s)  # gate scale
        x = img.permute(2, 0, 1).float()  # CHW, [0, 255]

        r = torch.rand(12)

        # geometric: random crop-shift + resize back (cheap extrinsics error)
        if r[0] < 0.5 * g:
            _, h, w = x.shape
            f = 1.0 - 0.15 * s * float(torch.rand(1))          # keep 85–100 %
            ch, cw = max(8, int(h * f)), max(8, int(w * f))
            y0 = int(float(torch.rand(1)) * (h - ch))
            x0 = int(float(torch.rand(1)) * (w - cw))
            x = torch.nn.functional.interpolate(
                x[None, :, y0:y0 + ch, x0:x0 + cw], size=(h, w),
                mode="bilinear", align_corners=False)[0]

        # photometric. Factors are floored at 0.05: past strength ~2.9 the raw draw goes
        # NEGATIVE (1 + u*0.35*s < 0) and would invert the image, which is not a stronger
        # corruption, it is a different (and unphysical) one — and the robustness matrix
        # sweeps strength upward.
        def _fac(scale):
            return max(0.05, 1.0 + (float(torch.rand(1)) * 2 - 1) * scale * s)

        if r[1] < 0.8 * g:  # brightness
            x = x * _fac(0.35)
        if r[2] < 0.8 * g:  # contrast
            m = x.mean()
            x = (x - m) * _fac(0.35) + m
        if r[3] < 0.6 * g:  # saturation
            gray = x.mean(dim=0, keepdim=True)
            x = gray + (x - gray) * _fac(0.5)
        if r[4] < 0.4 * g:  # channel tint (cheap hue-ish shift, avoids HSV round trip)
            x = x * (1.0 + (torch.rand(3, 1, 1) * 2 - 1) * 0.08 * s).clamp_min(0.05)
        if r[5] < 0.4 * g:  # gamma
            x = ((x.clamp(0, 255) / 255.0)
                 ** max(0.1, 1.0 + (float(torch.rand(1)) * 2 - 1) * 0.3 * s)) * 255.0

        # defocus blur
        if r[6] < 0.5 * g:
            sig = (0.1 + float(torch.rand(1)) * 1.9) * s
            k = max(3, int(2 * round(2 * sig) + 1))
            ax = torch.arange(k, dtype=torch.float32) - k // 2
            g1 = torch.exp(-(ax ** 2) / (2 * sig * sig))
            g1 = (g1 / g1.sum()).to(x.dtype)
            x = torch.nn.functional.conv2d(
                x[None], g1.view(1, 1, 1, k).expand(3, 1, 1, k), padding=(0, k // 2), groups=3)
            x = torch.nn.functional.conv2d(
                x, g1.view(1, 1, k, 1).expand(3, 1, k, 1), padding=(k // 2, 0), groups=3)[0]

        # sensor noise
        if r[7] < 0.5 * g:
            x = x + torch.randn_like(x) * (float(torch.rand(1)) * 10.0 * s)

        u8 = x.clamp(0, 255).to(torch.uint8)

        # JPEG compression artefacts
        if r[8] < 0.5 * g:
            from torchvision.io import decode_jpeg, encode_jpeg  # noqa: PLC0415

            q = max(15, int(95 - float(torch.rand(1)) * 65 * s))  # severity scales with strength
            u8 = decode_jpeg(encode_jpeg(u8.contiguous(), quality=q))

        # cutout occlusions
        if r[9] < 0.3 * g:
            _, h, w = u8.shape
            for _ in range(1 + int(float(torch.rand(1)) * 2)):
                rh = max(2, int(h * (0.05 + float(torch.rand(1)) * 0.15) * s))
                rw = max(2, int(w * (0.05 + float(torch.rand(1)) * 0.15) * s))
                y0 = int(float(torch.rand(1)) * (h - rh))
                x0 = int(float(torch.rand(1)) * (w - rw))
                u8[:, y0:y0 + rh, x0:x0 + rw] = int(float(torch.rand(1)) * 255)

        return u8.permute(1, 2, 0)
