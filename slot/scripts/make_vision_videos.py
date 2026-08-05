#!/usr/bin/env python
"""Stitched video of what the VISUAL policy actually sees while it drives.

Two panes side by side -- wrist D405 | workspace D455 -- at the **policy's own resolution**
(160x90), upscaled with NEAREST so the pixels stay visible as pixels. That is deliberate: a
smooth bilinear upscale would show a nicer image than the network ever receives, and the whole
point of this clip is to show the actual input. A caption strip carries the step, the block's
distance to the slot, and the running outcome.

The rollout writes nothing but CPU frame copies -- the video is encoded AFTER the episode ends.
EXP08 lost 40 points of student success to a loop that did optional GPU work between steps, and
that rule holds here even though nothing is being collected for training.

Outcomes at n = 1 are a coin flip, so each clip RETRIES over spawn seeds until the episode
genuinely has the outcome its filename claims -- a clip labelled "failure" showing a success
would be worse than no clip.

    python scripts/make_vision_videos.py --ckpt runs/vision_bc/v1/ckpt_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", default="runs/vision_bc/v1/ckpt_final.pt")
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--out", default="runs/vision_bc/v1/videos")
parser.add_argument("--scale", type=int, default=5, help="nearest upscale of the 160x90 panes")
parser.add_argument("--fps", type=int, default=50, help="control runs at 50 Hz; 1x real time")
# ONE SEED PER PROCESS. Isaac Sim cannot tear down and rebuild a camera-bearing env inside a
# single process -- the second gym.make raises "Unable to retrieve replicator graph". The first
# version of this script looped over seeds and died right after writing its first clip.
# scripts/make_vision_videos.sh drives the retry loop instead.
parser.add_argument("--seed", type=int, default=777)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
import slot_mdp as mdp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from slot_act.cameras import CAM_HEIGHT, CAM_WIDTH, attach_cameras, rgb_native  # noqa: E402
from slot_act.eval_flow_vision import VisionController, load_vision_checkpoint  # noqa: E402

BAR = 34  # caption strip height, px


def compose(wrist: np.ndarray, works: np.ndarray, scale: int, caption: str) -> np.ndarray:
    """(90,160,3) x2 -> one upscaled side-by-side frame with a caption strip."""
    pair = np.concatenate([wrist, works], axis=1)                     # (90, 320, 3)
    img = Image.fromarray(pair).resize(
        (pair.shape[1] * scale, pair.shape[0] * scale), Image.NEAREST)
    canvas = Image.new("RGB", (img.width, img.height + BAR), (16, 16, 18))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((8, 6), "WRIST D405  (policy input, 160x90)", fill=(235, 235, 235))
    d.text((CAM_WIDTH * scale + 8, 6), "WORKSPACE D455  (policy input, 160x90)", fill=(235, 235, 235))
    d.line([(CAM_WIDTH * scale, 0), (CAM_WIDTH * scale, img.height)], fill=(60, 60, 66), width=2)
    d.text((8, img.height + 10), caption, fill=(190, 220, 190))
    return np.asarray(canvas)


def main() -> None:
    device = "cuda:0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy, stats, cfg = load_vision_checkpoint(Path(args.ckpt), device)
    ss = int((cfg.get("render") or {}).get("supersample", 4))
    assert not cfg.get("blind"), "this is the sighted policy's video; the blind arm sees nothing"
    print(f"[video] render contract: {cfg.get('render')}")

    seed = args.seed
    if True:
        controller = VisionController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)

        env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
        env_cfg.terminations.block_dropped = None
        env_cfg.terminations.block_toppled = None
        env_cfg.rewards.dropping_penalty = None
        env_cfg.rewards.toppling_penalty = None
        env_cfg.seed = seed
        attach_cameras(env_cfg, supersample=ss)
        env = gym.make(args.task, cfg=env_cfg)
        u = env.unwrapped
        obs = env.reset()[0]["policy"]

        frames, caps = [], []
        success = False
        for step in range(599):
            wrist = rgb_native(u, "wrist_cam")[0].cpu().numpy()
            works = rgb_native(u, "workspace_cam")[0].cpu().numpy()
            placed = bool(mdp.placed_mask(u).all(dim=1)[0])
            depth_mm = float(mdp.insertion_depth(u)[0]) * 1000
            lat_mm = float(mdp.lateral_error(u)[0]) * 1000
            success = placed
            frames.append((wrist, works))
            caps.append(f"step {step:3d}/599   depth {depth_mm:+7.1f} mm   |lateral| {lat_mm:5.2f} mm"
                        f"   {'SEATED' if placed else ''}")
            stu = {"joint_pos": obs[:, 0:8], "joint_vel": obs[:, 8:16], "actions": obs[:, 27:34],
                   "wrist_rgb": rgb_native(u, "wrist_cam"),
                   "workspace_rgb": rgb_native(u, "workspace_cam")}
            obs = env.step(controller.act(stu).to(u.device))[0]["policy"]

        label = "success" if success else "failure"
        env.close()
        dest = out / f"vision_{label}_s{seed}.mp4"
        with imageio.get_writer(dest, fps=args.fps, quality=8) as w:
            for (wr, wo), cap in zip(frames, caps):
                w.append_data(compose(wr, wo, args.scale, f"{label.upper()}   {cap}"))
        print(f"[video] seed {seed}: {label} -> {dest}")
    print(f"[video] done -> {out}")
    app.close()


if __name__ == "__main__":
    main()
