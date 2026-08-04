#!/usr/bin/env python
"""Render-quality probe: why do the G0 stills look noisy, and what fixes it?

G0 proved the render is temporally DETERMINISTIC (two renders at an unchanged state differ by
0.0). That is not the same as clean, and Big Will's review caught the difference: Isaac Lab
defaults ``samples_per_pixel = 1``, and G0 also ran ``antialiasing_mode = "Off"`` (set to dodge
EXP08's DLSS suspicion). A 1-sample-per-pixel direct-lighting render with no AA is grainy and
aliased, and it renders the *same* grainy image every time -- which is exactly what the G0
numbers said and exactly what the PNGs show.

This script renders ONE fixed scene state under ONE render setting, so a shell loop can sweep
settings and Big Will can compare like for like. The pose is pinned by driving the champion with
``fixed_x0 = zeros`` for ``--steps`` steps, so every variant sees an identical robot and block.

Outputs per run, into ``<out>/<tag>/``:
  * ``<cam>_native.png``      -- the frame at training resolution (160x90), what the policy sees
  * ``<cam>_render.png``      -- the frame at render resolution (160*scale x 90*scale)
  * ``metrics.json``          -- numbers, since Claude does not view images

Metrics (all on the 160x90 frame the policy would consume):
  lap_var      variance of the Laplacian: high-frequency energy = detail AND noise together
  noise_proxy  mean |img - 3x3 box blur|: grain, mostly
  vs_ref       mean abs difference against a reference frame, if --ref is given. The reference
               is meant to be the highest-quality variant downsampled to 160x90; a variant that
               is close to it is clean, one that is far from it is noisy or aliased.

    python scripts/vision_render_probe.py --tag off_spp1 --aa Off --spp 1
    python scripts/vision_render_probe.py --tag ss4 --aa Off --spp 8 --scale 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tag", required=True, help="subdirectory name for this variant")
parser.add_argument("--aa", default="Off", choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"])
parser.add_argument("--spp", type=int, default=None, help="samples_per_pixel (Isaac default 1)")
parser.add_argument("--denoiser", action="store_true", help="enable the DL denoiser")
parser.add_argument("--mode", default=None, choices=["performance", "balanced", "quality"])
parser.add_argument("--scale", type=int, default=1,
                    help="render at scale*160 x scale*90 and box-downsample to 160x90 "
                         "(supersampling: scale=4 gives 16 samples per training pixel)")
parser.add_argument("--steps", type=int, default=150, help="pin the pose at this step")
parser.add_argument("--ckpt", default="runs/bc_armB_seed0/ckpt_final.pt")
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--out", default="runs/vision_render_probe")
parser.add_argument("--ref", default=None, help="path to a reference <cam>_native.png dir")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from slot_act.cameras import CAM_HEIGHT, CAM_WIDTH, CAMERA_NAMES, attach_cameras, rgb  # noqa: E402
from slot_act.eval_act import BatchedACTController, load_checkpoint  # noqa: E402


def to_native(img: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) uint8 at render res -> (90, 160, 3) uint8 by BOX average (true supersample)."""
    if img.shape[0] == CAM_HEIGHT and img.shape[1] == CAM_WIDTH:
        return img
    x = img.permute(2, 0, 1).float().unsqueeze(0)
    x = F.adaptive_avg_pool2d(x, (CAM_HEIGHT, CAM_WIDTH))
    return x.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)


def metrics(img: torch.Tensor) -> dict:
    g = img.float().mean(dim=-1).unsqueeze(0).unsqueeze(0)  # (1,1,H,W) luma
    lap_k = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3)
    lap = F.conv2d(g, lap_k, padding=1)
    blur = F.avg_pool2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    return {"lap_var": round(float(lap.var()), 2),
            "noise_proxy": round(float((g - blur).abs().mean()), 3),
            "mean": round(float(g.mean()), 2), "std": round(float(g.std()), 2)}


def main() -> None:
    device = "cuda:0"
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    policy, stats, cfg = load_checkpoint(Path(args.ckpt), device)
    # fixed x0 => the trajectory is a deterministic function of the observation, so every
    # variant of this sweep pins the SAME robot and block pose at --steps.
    controller = BatchedACTController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"],
                                      device, fixed_x0=torch.zeros(cfg["chunk_size"], 7))

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.seed = args.seed
    attach_cameras(env_cfg, width=CAM_WIDTH * args.scale, height=CAM_HEIGHT * args.scale,
                   antialiasing=args.aa)
    if args.spp is not None:
        env_cfg.sim.render.samples_per_pixel = args.spp
    if args.denoiser:
        env_cfg.sim.render.enable_dl_denoiser = True
    if args.mode is not None:
        env_cfg.sim.render.rendering_mode = args.mode

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    obs = env.reset()[0]["policy"]
    for _ in range(args.steps):
        obs = env.step(controller.act(obs).to(u.device))[0]["policy"]

    result = {"tag": args.tag, "aa": args.aa, "spp": args.spp, "denoiser": args.denoiser,
              "mode": args.mode, "scale": args.scale,
              "render_res": [CAM_WIDTH * args.scale, CAM_HEIGHT * args.scale], "cams": {}}
    for c in CAMERA_NAMES:
        raw = rgb(u, c)[0].cpu()
        nat = to_native(raw)
        short = c.split("_")[0]
        Image.fromarray(nat.numpy()).save(out / f"{short}_native.png")
        if args.scale > 1:
            Image.fromarray(raw.numpy()).save(out / f"{short}_render.png")
        m = metrics(nat)
        if args.ref:
            ref = torch.from_numpy(np.array(Image.open(Path(args.ref) / f"{short}_native.png")))
            m["vs_ref"] = round(float((nat.float() - ref[..., :3].float()).abs().mean()), 3)
        result["cams"][short] = m
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    line = f"[{args.tag:<16}] aa={args.aa:<5} spp={args.spp} scale={args.scale} "
    for short, m in result["cams"].items():
        line += f" | {short}: lap_var {m['lap_var']:>8.1f} noise {m['noise_proxy']:>6.3f}"
        if "vs_ref" in m:
            line += f" vs_ref {m['vs_ref']:>6.3f}"
    print(line)

    env.close()
    app.close()


if __name__ == "__main__":
    main()
