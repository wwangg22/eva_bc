# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Film the VISION student's rollouts -- successes AND failures, from the student's own eyes.

Drives the same `VisionController` + `build_student_batch` loop as `eval_flow_vision.py`,
on the same `-Vision-` task cameras, so the film is of the thing that was measured. No
spectator camera is added: under DLSS the frame CONTENT depends on GPU work between steps
(EXP08 v1 lost 80% -> 40% to exactly this), so an extra render pass would film a slightly
different policy. Instead each episode's mp4 is the student's actual input, upscaled:
workspace view | wrist view, side by side.

Batches of `--num_envs` run until `--n-succ` successes and `--n-fail` failures are on film
(fresh seed per batch). Filenames carry the outcome.

.. code-block:: bash

    python -u re3sim/act/record_video_vision.py \\
        --ckpt re3sim/runs/vbc_stab_s2/ckpt_final.pt --out re3sim/runs/video_stab_s2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Film vision-student successes and failures.")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-Vision-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=88010, help="first batch seed; +1 per batch")
parser.add_argument("--horizon", type=int, default=600)
parser.add_argument("--warmup", type=int, default=120)
parser.add_argument("--n-succ", type=int, default=3)
parser.add_argument("--n-fail", type=int, default=3)
parser.add_argument("--max-batches", type=int, default=3)
parser.add_argument("--stride", type=int, default=2, help="record every Nth step (50Hz sim)")
parser.add_argument("--upscale", type=int, default=3)
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app = AppLauncher(args_cli).app

import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.re3sim import mdp  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))

from policy_runner_vision import (  # noqa: E402
    VisionController, build_student_batch, load_vision_checkpoint,
)


def main() -> None:
    device = torch.device("cuda")
    policy, normalizer, config, cam_shape = load_vision_checkpoint(args_cli.ckpt, device)
    h, w = int(cam_shape[1]), int(cam_shape[2])
    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    for nm in ("wrist_cam", "station_cam"):
        cam_cfg = getattr(env_cfg.scene, nm, None)
        if cam_cfg is None:
            raise SystemExit(f"{args_cli.task} has no {nm}; use a -Vision- task id")
        cam_cfg.width, cam_cfg.height = w, h
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    n, dev = e.num_envs, e.device
    cams = {"wrist": e.scene["wrist_cam"], "workspace": e.scene["station_cam"]}
    ctrl = VisionController(policy, normalizer, config, device)
    os.makedirs(args_cli.out, exist_ok=True)

    n_succ = n_fail = 0
    up = args_cli.upscale
    for b in range(args_cli.max_batches):
        seed = args_cli.seed + b
        env.reset(seed=seed)
        zero = torch.zeros((n, 7), device=dev)
        for _ in range(args_cli.warmup):
            env.step(zero)
            e.sim.render()
        env.reset(seed=seed)
        ctrl.reset()
        frames: list[list[np.ndarray]] = [[] for _ in range(n)]
        succ = torch.zeros(n, dtype=torch.bool, device=dev)
        obs = e.observation_manager.compute()["policy"]
        for t in range(args_cli.horizon):
            e.sim.render()
            for cam in cams.values():
                cam.update(dt=0.0)
            if t % args_cli.stride == 0:
                ws = cams["workspace"].data.output["rgb"][..., :3].cpu().numpy()
                wr = cams["wrist"].data.output["rgb"][..., :3].cpu().numpy()
                for i in range(n):
                    fr = np.concatenate([ws[i], wr[i]], axis=1).astype(np.uint8)
                    frames[i].append(np.kron(fr, np.ones((up, up, 1), dtype=np.uint8)))
            a = ctrl.act(build_student_batch(obs, cams, device))
            obs, _, _, _, _ = env.step(a.to(dev))
            obs = obs["policy"] if isinstance(obs, dict) else obs
            succ |= mdp.placed_mask(e)
        for i in range(n):
            ok = bool(succ[i])
            if ok and n_succ >= args_cli.n_succ:
                continue
            if not ok and n_fail >= args_cli.n_fail:
                continue
            tag = "ok" if ok else "fail"
            path = os.path.join(args_cli.out, f"s{seed}_env{i}_{tag}.mp4")
            wtr = imageio.get_writer(path, fps=50 // args_cli.stride, macro_block_size=1)
            for f in frames[i]:
                wtr.append_data(f)
            wtr.close()
            print(f"[video] {path} ({len(frames[i])} frames) -- for Big Will", flush=True)
            n_succ += int(ok)
            n_fail += int(not ok)
        print(f"[batch {b}] seed {seed}: {int(succ.sum())}/{n} succeeded; "
              f"filmed so far: {n_succ} ok, {n_fail} fail", flush=True)
        if n_succ >= args_cli.n_succ and n_fail >= args_cli.n_fail:
            break
    env.close()
    app.close()


if __name__ == "__main__":
    main()
