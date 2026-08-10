# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Batched simulator evaluation of the workstation VISION flow-BC student.

The sibling of ``eval_flow.py``, with the same failure taxonomy and the same
judged-against-the-expert framing, differing in exactly one thing: what the policy is allowed
to see.

    policy input   two rendered cameras + 23-D proprio, and nothing else
    metrics        privileged, deliberately -- `mdp.placed_mask`, cube height, `over_box`

Measurement is allowed to be privileged; policy input is not. `build_student_batch` is the
whole contract surface, and it is fed from the camera buffers plus `obs41[0:16]` and
`obs41[34:41]` -- the joint state and the policy's own last action, both of which the real rig
publishes. `obs41[16:34]` (cube pose, box pose, clutter, placed flag) is never touched here.

Two things this must keep byte-identical to collection or the numbers mean nothing:

* **the camera mount and resolution** -- the wrist camera's intrinsics are the D405's own,
  calibrated at 4:3; the workstation camera comes from `STATION_CAM_EYE/TARGET`. Both are
  imported, not re-declared.
* **the per-step render**. `e.sim.render()` runs every step whether or not the controller
  needs a new chunk, because under DLSS frame CONTENT depends on GPU work between steps
  (EXP08 v1 lost 80 % -> 40 % to exactly this). Do not make rendering conditional.

Usage
-----
    python -u re3sim/act/eval_flow_vision.py --ckpt re3sim/runs/vbc_r1/ckpt_final.pt \\
        --num_envs 32 --seeds 88000,88001
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/re3sim"

parser = argparse.ArgumentParser(description="Batched sim eval of the workstation vision student.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-Play-v0")
parser.add_argument("--num_envs", type=int, default=32,
                    help="Rendering is pixel-bound; 64 x two 160x120 cameras is fine, "
                         "64 x 640x480 renders blank.")
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--seeds", type=str, default="88000,88001",
                    help="held-out spawn seeds; NEVER the demo-collection seeds")
parser.add_argument("--horizon", type=int, default=0,
                    help="env steps per episode; 0 = the env's own episode length")
parser.add_argument("--n-action-steps", type=int, default=None)
parser.add_argument("--warmup", type=int, default=120,
                    help="throwaway steps after reset so the gaussian desk is resident")
parser.add_argument("--expert-rate", type=float, default=None,
                    help="the expert's rate on this protocol, for the retention line")
parser.add_argument("--out", type=str, default=f"{_ROOT}/runs/vbc_eval.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.re3sim import mdp  # noqa: E402
from reBot_RL.tasks.manager_based.lift.camera_cfg import WRIST_CAM_CFG  # noqa: E402
from reBot_RL.tasks.manager_based.re3sim.workstation_env_cfg import (  # noqa: E402
    STATION_CAM_EYE, STATION_CAM_TARGET,
)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))          # eva_bc, for the vendored act/ package
sys.path.insert(0, str(_HERE))

# The runtime lives in `policy_runner_vision`, not here: this module builds an AppLauncher at
# import time, so anything that needs to DRIVE the student from inside another program (DAgger)
# cannot import from it.
from policy_runner_vision import (  # noqa: E402
    VisionController, build_student_batch, load_vision_checkpoint,
)

#: "the grasp took" -- cube base clear of the desk by more than settle noise. Same constant as
#: eval_flow.py; a different threshold would make the two taxonomies incomparable.
LIFT_MIN = 0.060


def main() -> None:
    device = torch.device(args_cli.device if hasattr(args_cli, "device") else "cuda")
    policy, normalizer, config, cam_shape = load_vision_checkpoint(
        args_cli.ckpt, device, args_cli.n_action_steps)
    h, w = int(cam_shape[1]), int(cam_shape[2])

    env_cfg = parse_env_cfg(args_cli.task, device=str(device), num_envs=args_cli.num_envs)
    env_cfg.scene.wrist_cam = WRIST_CAM_CFG.replace(
        width=w, height=h, data_types=["rgb"], update_period=0.0)
    env_cfg.scene.station_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/StationCam", update_period=0.0, width=w, height=h,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=17.0, horizontal_aperture=20.955,
                                         clipping_range=(0.02, 20.0)))
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    n, dev = e.num_envs, e.device
    cams = {"wrist": e.scene["wrist_cam"], "workspace": e.scene["station_cam"]}
    ctrl = VisionController(policy, normalizer, config, device)
    horizon = args_cli.horizon or int(e.max_episode_length)

    if n > 1 and os.environ.get("RE3SIM_SPLATS_PER_ENV") != "1":
        raise SystemExit("eval at num_envs > 1 needs RE3SIM_SPLATS_PER_ENV=1, or every env "
                         "but one renders a bare floor -- which is not the distribution the "
                         "student was trained on")

    def aim_station():
        # After every reset: `reset_scene_to_default` restores prim poses, and one pose PER
        # camera -- `set_world_poses_from_view` does not broadcast.
        org = e.scene.env_origins
        cams["workspace"].set_world_poses_from_view(
            org + torch.tensor(STATION_CAM_EYE, device=dev),
            org + torch.tensor(STATION_CAM_TARGET, device=dev))

    rows, agg = [], {}
    for s in [int(x) for x in args_cli.seeds.split(",")]:
        env.reset(seed=s)
        aim_station()
        # The gaussian desk is not resident for the first ~100 rendered steps; a policy
        # scored on those frames is scored on a scene with no desk in it.
        zero = torch.zeros((n, 7), device=dev)
        for _ in range(args_cli.warmup):
            env.step(zero)
            e.sim.render()
        env.reset(seed=s)
        aim_station()
        ctrl.reset()

        tnames = list(e.termination_manager.active_terms)
        done_at = torch.full((n,), -1, dtype=torch.long, device=dev)
        why = torch.zeros((n, len(tnames)), dtype=torch.bool, device=dev)
        succ = torch.zeros(n, dtype=torch.bool, device=dev)
        lifted = torch.zeros(n, dtype=torch.bool, device=dev)
        over = torch.zeros(n, dtype=torch.bool, device=dev)

        obs = e.observation_manager.compute()["policy"]
        for t in range(horizon):
            # Unconditional: see the DLSS note in the module docstring.
            e.sim.render()
            for cam in cams.values():
                cam.update(dt=0.0)
            a = ctrl.act(build_student_batch(obs, cams, device))
            obs, _, terminated, truncated, _ = env.step(a.to(dev))
            obs = obs["policy"] if isinstance(obs, dict) else obs
            alive = done_at < 0
            succ |= mdp.placed_mask(e) & alive
            lifted |= (mdp.object_pos_local(e, mdp.TARGET_NAME)[:, 2] > LIFT_MIN) & alive
            over |= mdp.over_box(e) & alive
            newly = (terminated | truncated) & alive
            if bool(newly.any()):
                done_at[newly] = t
                for k, nm in enumerate(tnames):
                    why[:, k] |= newly & e.termination_manager.get_term(nm)
                ctrl.reset(newly.nonzero(as_tuple=False).squeeze(-1))
                aim_station()

        idx = {nm: k for k, nm in enumerate(tnames)}
        drop = why[:, idx["target_dropped"]] if "target_dropped" in idx \
            else torch.zeros(n, dtype=torch.bool, device=dev)
        row = {
            "seed": s, "n": n,
            "success": float(succ.float().mean()),
            "lifted": float(lifted.float().mean()),
            "over_box": float(over.float().mean()),
            "dropped": float(drop.float().mean()),
            "near_miss": float((~succ & lifted & over).float().mean()),
            "carried_astray": float((~succ & lifted & ~over).float().mean()),
            "no_lift": float((~succ & ~lifted).float().mean()),
        }
        rows.append(row)
        print(f"[seed {s}] success {row['success']:.1%}   lifted {row['lifted']:.1%}   "
              f"over_box {row['over_box']:.1%}   no_lift {row['no_lift']:.1%}   "
              f"near_miss {row['near_miss']:.1%}", flush=True)
        for k, v in row.items():
            if k not in ("seed", "n"):
                agg.setdefault(k, []).append(v)

    print("\n=== pooled over seeds ===")
    for k, v in agg.items():
        print(f"  {k:16s} {np.mean(v):.1%}")
    if args_cli.expert_rate:
        print(f"  retention        {np.mean(agg['success']) / args_cli.expert_rate:.1%} "
              f"of the expert's {args_cli.expert_rate:.1%}")
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump({"ckpt": args_cli.ckpt, "rows": rows,
                   "pooled": {k: float(np.mean(v)) for k, v in agg.items()}}, f, indent=2)
    print(f"wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
