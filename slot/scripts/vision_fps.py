#!/usr/bin/env python
"""Throughput probe: env-steps/s vs (supersample, num_envs). VISION_PLAN G1.

The 4-episode smoke ran at 3.7 vectorized steps/s with 2 envs at supersample 8, which puts a
256-episode pool somewhere between "an afternoon" and "not happening". Rendering 1280x720 twice
per env per step is the cost, and it is worth measuring the trade rather than guessing it:
supersample 4 has a quarter the pixels and still cuts temporal shimmer 35.2 -> 9.1.

Reports episode-steps/s (= vectorized steps/s x num_envs), which is what actually sets the wall
clock for a fixed episode budget.

    python scripts/vision_fps.py --supersample 8 --num-envs 16 --steps 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--supersample", type=int, default=8)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--warmup", type=int, default=15)
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--no-cameras", action="store_true", help="baseline: physics only")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = not args.no_cameras
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from slot_act.cameras import CAMERA_NAMES, attach_cameras, rgb_native  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.seed = 777
    if not args.no_cameras:
        attach_cameras(env_cfg, supersample=args.supersample)

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    act = torch.zeros(args.num_envs, 7, device=u.device)

    for _ in range(args.warmup):
        if not args.no_cameras:
            for c in CAMERA_NAMES:
                rgb_native(u, c)
        env.step(act)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args.steps):
        if not args.no_cameras:
            for c in CAMERA_NAMES:
                rgb_native(u, c)
        env.step(act)
    torch.cuda.synchronize()
    el = time.time() - t0

    vec = args.steps / el
    tag = "no-cameras" if args.no_cameras else f"ss{args.supersample}"
    # a 599-step episode budget, for the number that actually matters
    per_256 = 256 * 599 / (vec * args.num_envs) / 3600
    print(f"[fps] {tag:<12} envs {args.num_envs:>3}  {vec:6.2f} vec-steps/s  "
          f"{vec * args.num_envs:7.1f} ep-steps/s   256 eps = {per_256:5.2f} h")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
