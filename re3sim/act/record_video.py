# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Film the trained policy: one wrist-camera view, one workstation view.

Driven through the SAME `ChunkController` that `eval_flow.py` uses, from `policy_runner`, so
the film is of the thing that was measured. A recorder that re-implements the control loop
ends up filming a slightly different policy, and the difference is invisible in the output.

Two cameras, neither of them invented here:

* **wrist** -- `WRIST_CAM_CFG` from `tasks/manager_based/lift/camera_cfg.py`. That mount was
  chosen by the user from rendered sweep grids (`tilt_x_m30`, optical axis 1.8 deg off the TCP,
  camera-to-TCP 0.171 m) and models a RealSense D405 at 84 deg HFOV. Re-deriving a mount here
  would film a camera that does not exist on the rig.
* **workstation** -- a fixed third-person view of the whole scene, placed to show the desk,
  the box and the object layout at once.

Episodes are **retried until one succeeds**, because a video of a failure is not what anyone
asks for when they ask to see the policy -- but the retry count is printed, so the film is
never mistaken for the success *rate*. That number is in `eval_flow.py`.

.. code-block:: bash

    python -u re3sim/act/record_video.py --headless \\
        --ckpt re3sim/runs/final2/bc/ckpt_final.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record the workstation policy from two views.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-Play-v0")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--out", type=str, default="re3sim/runs/video")
parser.add_argument("--seeds", type=str, default="88010,88011,88012,88013,88014,88015,88016,88017",
                    help="tried in order until one produces a success")
parser.add_argument("--stride", type=int, default=2, help="record every Nth env step")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--n-action-steps", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.lift.camera_cfg import WRIST_CAM_CFG  # noqa: E402
from reBot_RL.tasks.manager_based.re3sim import mdp  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE))
from policy_runner import ChunkController, load_checkpoint  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    # The user's own wrist mount, only re-sized for video. `.replace` keeps the offset and the
    # intrinsics exactly as measured.
    env_cfg.scene.wrist_cam = WRIST_CAM_CFG.replace(
        width=args_cli.width, height=args_cli.height, data_types=["rgb"], update_period=0.0)
    env_cfg.scene.station_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/StationCam",
        update_period=0.0,
        width=args_cli.width,
        height=args_cli.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=17.0, horizontal_aperture=20.955, clipping_range=(0.02, 20.0)),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    device = torch.device(args_cli.device if args_cli.device else "cuda")

    policy, stats, cfg = load_checkpoint(args_cli.ckpt, device)
    nas = args_cli.n_action_steps or cfg["n_action_steps"]
    ctrl = ChunkController(policy, stats, nas, cfg["chunk_size"], device)
    horizon = int(e.max_episode_length)

    wrist = e.scene["wrist_cam"]
    station = e.scene["station_cam"]
    origin = e.scene.env_origins[0]
    station.set_world_poses_from_view(
        (torch.tensor([[-0.62, -0.52, 0.46]], device=device) + origin),
        (torch.tensor([[0.24, 0.02, 0.04]], device=device) + origin))

    os.makedirs(args_cli.out, exist_ok=True)
    print(f"\nckpt {args_cli.ckpt}  (step {cfg.get('step')})")
    print(f"chunk {cfg['chunk_size']} / commit {nas}, horizon {horizon} env steps\n")

    for attempt, seed in enumerate([int(s) for s in args_cli.seeds.split(",")]):
        env.reset(seed=seed)
        station.set_world_poses_from_view(
            (torch.tensor([[-0.62, -0.52, 0.46]], device=device) + origin),
            (torch.tensor([[0.24, 0.02, 0.04]], device=device) + origin))
        ctrl.reset()
        obs = e.observation_manager.compute()["policy"]
        fw, fs, ok = [], [], False
        for t in range(horizon):
            obs, _, term, trunc, _ = env.step(ctrl.act(obs))
            obs = obs["policy"] if isinstance(obs, dict) else obs
            ok = ok or bool(mdp.placed_mask(e)[0])
            if t % args_cli.stride == 0:
                e.sim.render()
                wrist.update(dt=0.0)
                station.update(dt=0.0)
                fw.append(wrist.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
                fs.append(station.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            if bool(term[0]) or bool(trunc[0]):
                break
        print(f"  seed {seed}: {'SUCCESS' if ok else 'failed'}  ({len(fw)} frames)", flush=True)
        if ok:
            for name, frames in (("wrist", fw), ("workstation", fs)):
                path = os.path.join(args_cli.out, f"policy_{name}.mp4")
                imageio.mimwrite(path, frames, fps=args_cli.fps, quality=8,
                                 macro_block_size=1)
                print(f"  wrote {path}  ({len(frames)} frames, "
                      f"{len(frames) / args_cli.fps:.1f} s)")
            print(f"\nfilmed on attempt {attempt + 1} (seed {seed}). Attempts are printed so "
                  f"this is never read as a success RATE -- that is eval_flow.py's job.")
            break
    else:
        print("\nno seed produced a success; nothing written")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
