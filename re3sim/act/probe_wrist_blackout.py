# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Why does the wrist camera go black during the descent, and does the near plane fix it?

Measured on the round-1 vision shards: ~1 % of wrist frames are fully black, they are NOT
spread out -- they cluster in deciles 4-6 of the episode (the descent and grasp) in runs of up
to 37 steps (0.74 s). That is the wrist camera going blind at precisely the moment a wrist
camera exists to inform, so it is worth a diagnosis rather than a shrug.

The hypothesis this tests: as the gripper descends, the camera ends up inside / almost inside
the gaussian desk, and the splats within the near plane are not rasterised -- leaving nothing
but background. If that is it, a smaller near clip recovers the frames for free.

The replay is exact rather than approximate: the shard carries the joint positions
(`proprio[:, 0:8]` is `q - q_default`) and the cube pose (`obs41[16:23]`) for the very frame
that came out black, so the arm and the cube are put back exactly where they were and the only
thing varied is `clipping_range`.

Usage
-----
    python -u re3sim/act/probe_wrist_blackout.py --shard re3sim/expert/data/vision_r1/ep_21000_000.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wrist-camera blackout probe.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-Play-v0")
parser.add_argument("--shard", type=str, required=True)
parser.add_argument("--near", type=str, default="0.02,0.005,0.001",
                    help="near clip values to compare, metres")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_rl/docs/envs/re3sim/renders/wrist_nearclip.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.re3sim import mdp  # noqa: E402
from reBot_RL.tasks.manager_based.lift.camera_cfg import WRIST_CAM_CFG  # noqa: E402


def main() -> None:
    sh = torch.load(args_cli.shard, map_location="cpu", weights_only=False)
    frames = sh["wrist_rgb"].float().mean(dim=(1, 2, 3))
    black = (frames < 3).nonzero().flatten()
    if not len(black):
        raise SystemExit(f"{args_cli.shard} has no black wrist frames")
    # the middle of the longest blackout, not its edge
    t = int(black[len(black) // 2])
    print(f"[probe] replaying frame {t} of {len(frames)} "
          f"({len(black)} black frames in this episode)", flush=True)

    nears = [float(x) for x in args_cli.near.split(",")]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    h, w = int(sh["wrist_rgb"].shape[1]), int(sh["wrist_rgb"].shape[2])
    for k, nz in enumerate(nears):
        spawn = WRIST_CAM_CFG.spawn.replace(clipping_range=(nz, 5.0))
        setattr(env_cfg.scene, f"wrist_{k}", WRIST_CAM_CFG.replace(
            prim_path=WRIST_CAM_CFG.prim_path.replace("WristCam", f"WristCam{k}"),
            spawn=spawn, width=w, height=h, data_types=["rgb"], update_period=0.0))

    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()

    # Put the arm and the cube back exactly where the black frame found them.
    q = e.scene["robot"].data.default_joint_pos.clone()
    q[:, :8] = sh["proprio"][t, 0:8].to(q.device) + q[:, :8]
    e.scene["robot"].write_joint_state_to_sim(q, torch.zeros_like(q))
    obj = e.scene[mdp.TARGET_NAME]
    st = obj.data.root_state_w.torch.clone()
    st[:, :3] = sh["obs41"][t, 16:19].to(st.device) + e.scene.env_origins
    st[:, 3:7] = sh["obs41"][t, 19:23].to(st.device)
    st[:, 7:] = 0.0
    obj.write_root_state_to_sim(st)
    e.sim.forward()
    e.scene.update(e.physics_dt)
    # A render outside a simulation step produces no new frame, and the gaussian desk needs
    # ~100 stepped frames to become resident (see collect_demos.py's pre-roll).
    for _ in range(150):
        env.step(torch.zeros((1, 7), device=e.device))

    import imageio.v2 as imageio
    tiles = []
    for k, nz in enumerate(nears):
        cam = e.scene[f"wrist_{k}"]
        for _ in range(8):
            e.sim.render()
            cam.update(dt=0.0)
        rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        print(f"[probe] near {nz:.3f} m -> mean pixel {rgb.mean():6.1f}"
              f"   {'BLACK' if rgb.mean() < 3 else 'has content'}", flush=True)
        tiles.append(rgb)
    tiles.append(sh["wrist_rgb"][t].numpy())
    print("[probe] rightmost tile is the RECORDED frame, for comparison", flush=True)
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    imageio.imwrite(args_cli.out, np.concatenate(tiles, axis=1))
    print(f"[probe] wrote {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
