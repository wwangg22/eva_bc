#!/usr/bin/env python
"""GATE A.3 -- stills of the fixture at a ladder of slot yaws, for Big Will to eyeball.

The predicates agreeing with themselves proves nothing about what is on the table. This
renders one workstation per angle from the workspace camera and writes a PNG per angle plus a
contact sheet.

Two things fight image quality in this build and both are handled here:

* ``samples_per_pixel`` and FXAA are NO-OPS (measured, VISION_PLAN 11). The only lever that
  works is **supersampling**: render k x larger and box-average down.
* The raytracer's per-frame jitter has a measured temporal std of 35.24 / 255 on a static
  scene, so a single frame is visibly grainy no matter the resolution. Averaging N frames of
  the same static scene drops that by sqrt(N).

    python slot/scripts/slot_yaw_stills.py --out runs/angled/stills
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--out", default="runs/angled/stills")
parser.add_argument("--yaws", type=float, nargs="+", default=[-0.7, -0.35, 0.0, 0.35, 0.7])
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
# 2x, not 4x: five simultaneous viewports at 4x exhausted the 10.5 GB card ("Out of GPU
# memory allocating resource 'prevTargetMotion'"), and the wrist camera is not attached at all
# here for the same reason.
parser.add_argument("--supersample", type=int, default=2)
parser.add_argument("--frames", type=int, default=12, help="frames averaged to kill render noise")
parser.add_argument("--settle", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from reBot_RL.tasks.manager_based.challenge import mdp  # noqa: E402

from slot_act.cameras import attach_cameras, rgb  # noqa: E402


def main() -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(args.yaws)
    device = "cuda:0"

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=n)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.events.reset_slot.params["yaw_range"] = (0.0, 0.0)
    attach_cameras(env_cfg, width=args.width * args.supersample,
                   height=args.height * args.supersample, supersample=1)
    del env_cfg.scene.wrist_cam   # workspace view only; two viewports at this size OOM
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    ids = torch.arange(n, device=device)
    yaws = torch.tensor(args.yaws, device=device, dtype=torch.float32)
    mdp.set_slot_yaw(env, ids, yaws)

    act = torch.zeros(n, env.action_space.shape[1], device=device)
    for _ in range(args.settle):
        env.step(act)
        mdp.set_slot_yaw(env, ids, yaws)   # nothing resets mid-episode, but be explicit

    acc = torch.zeros(n, args.height * args.supersample, args.width * args.supersample, 3,
                      device=device, dtype=torch.float32)
    for _ in range(args.frames):
        env.step(act)
        acc += rgb(env, "workspace_cam").float()
    acc /= args.frames

    small = torch.nn.functional.adaptive_avg_pool2d(
        acc.permute(0, 3, 1, 2), (args.height, args.width)
    ).permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

    tiles = []
    for i, th in enumerate(args.yaws):
        img = Image.fromarray(small[i])
        d = ImageDraw.Draw(img)
        cap = f"slot yaw {th:+.2f} rad  ({np.degrees(th):+.1f} deg)"
        d.rectangle([0, 0, 260, 20], fill=(16, 16, 18))
        d.text((6, 5), cap, fill=(240, 240, 240))
        dest = out / f"slot_yaw_{th:+.2f}.png".replace("+", "p").replace("-", "m")
        img.save(dest)
        print(f"[stills] {cap} -> {dest}")
        tiles.append(np.asarray(img))

    sheet = Image.fromarray(np.concatenate(tiles, axis=0))
    sheet.thumbnail((args.width, 10_000))
    sheet.save(out / "contact_sheet.png")
    print(f"[stills] contact sheet -> {out / 'contact_sheet.png'}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
