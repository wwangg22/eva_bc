# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""Film the trained flow-BC policy: one environment, one episode per file, camera up close.

Numbers say the policy topples a neighbour in about 27 % of episodes and does nothing else
wrong. They do not say *how*. This renders the same policy, driven through the same
`ChunkController` the evaluator uses (`policy_runner.py` — shared deliberately, so the film
is of the thing that was measured), and writes one mp4 per episode.

Choices, and why
----------------
* **One env.** `--num_envs` is forced to 1. A tiled grid is useful for spotting spawn
  randomization and useless for watching a 12 mm gap.
* **A camera added at runtime.** `challenge/` spawns no cameras — that package is the
  benchmark under test and is not modified. `WORKSPACE_CAM_CFG` is attached to the scene cfg
  here, then re-aimed with `set_world_poses_from_view`; the same thing
  `eva_rl/scripts/challenge/record_env_video.py` does, for the same reason (its authored pose
  frames the lift scene and puts this row out of shot).
* **A narrowed lens.** The rig cfg is a 90 deg-HFOV D455 model, which at any sane working
  distance frames the whole table. `--hfov` overrides the focal length so the shot can be
  tight on the row without flying the camera into it.
* **Episodes end where they end.** A toppled episode is cut at the termination step rather
  than running on: `env.step` auto-resets internally, so the next frame would show a fresh
  scene and the film would quietly lie about what happened.
* **The filename carries the outcome**, decided from the termination manager at the step the
  env actually died — not from the end-of-episode scene, which has already been re-spawned
  (`REFERENCE.md` §5: env.step auto-resets from inside the step call).

Usage
-----
    python -u clutter/act/record_video.py --ckpt clutter/runs/bc_v3_s3/ckpt_final.pt \
        --seeds 88000,88001,88002,88003 --out-dir clutter/runs/videos
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/clutter"

parser = argparse.ArgumentParser(description="Record the clutter flow-BC policy.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--ckpt", type=str, default=f"{_ROOT}/runs/bc_v3_s3/ckpt_final.pt")
parser.add_argument("--seeds", type=str, default="88000,88001,88002,88003",
                    help="one spawn seed per video; each is a separate episode")
parser.add_argument("--steps", type=int, default=420,
                    help="max env steps per episode; a v3 demo is 325, so this leaves the "
                         "policy room to be late without filming 4 s of a settled scene")
parser.add_argument("--fps", type=int, default=25,
                    help="one frame per env step; the env runs at 50 Hz, so 25 plays at half "
                         "speed -- the close phase is over in ~5 steps at full speed")
parser.add_argument("--out-dir", type=str, default=f"{_ROOT}/runs/videos")
parser.add_argument("--cam-eye", type=float, nargs=3, default=[0.68, -0.47, 0.46],
                    help="camera position, ENV-relative metres. Chosen by rendering stills and looking at them, not by deriving it: 0.67 m out, aimed high enough that the wrist stays in shot through the lift")
parser.add_argument("--cam-target", type=float, nargs=3, default=[0.26, -0.06, 0.13],
                    help="look-at point, env-relative; the row is at x=0.25, the goal at "
                         "(0.185, -0.185), block tops at z=0.07")
parser.add_argument("--hfov", type=float, default=60.0, help="horizontal field of view, deg")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--stills", action="store_true",
                    help="also write a PNG per episode (framing checks without decoding mp4)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1          # one env at a time, not negotiable
args_cli.enable_cameras = True  # nothing renders without this

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl  # noqa: E402
from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_runner import ChunkController, load_checkpoint  # noqa: E402

#: USD's default 35 mm-film horizontal aperture, as `camera_cfg.py` uses it.
_APERTURE = 20.955


def grab(cam) -> np.ndarray:
    """One RGB frame for env 0. The `.torch` accessor may be an attribute or a callable."""
    out = cam.data.output["rgb"]
    out = getattr(out, "torch", out)
    if callable(out):
        out = out()
    return out[0, ..., :3].detach().cpu().numpy().astype("uint8")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # Narrow the lens: focal = aperture / (2 * tan(hfov / 2)).
    focal = _APERTURE / (2.0 * math.tan(math.radians(args_cli.hfov) / 2.0))
    env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(
        width=args_cli.width, height=args_cli.height,
        spawn=sim_utils.PinholeCameraCfg(focal_length=focal, horizontal_aperture=_APERTURE,
                                         clipping_range=(0.05, 20.0)),
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    cam = e.scene["workspace_cam"]
    device = torch.device(args_cli.device if args_cli.device else "cuda")

    # The cfg pose is env-relative; the look-at API is world, so add the env origin back in.
    eye = torch.tensor(args_cli.cam_eye, device=e.device).unsqueeze(0) + e.scene.env_origins
    tgt = torch.tensor(args_cli.cam_target, device=e.device).unsqueeze(0) + e.scene.env_origins
    cam.set_world_poses_from_view(eye, tgt)

    policy, stats, cfg = load_checkpoint(args_cli.ckpt, device)
    ctrl = ChunkController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)
    seeds = [int(s) for s in args_cli.seeds.split(",")]
    os.makedirs(args_cli.out_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print("RECORDING THE CLUTTER FLOW-BC POLICY")
    print("=" * 100)
    print(f"   ckpt {args_cli.ckpt}  (step {cfg.get('step')})")
    print(f"   chunk {cfg['chunk_size']} / commit {cfg['n_action_steps']}, 1 env, "
          f"{len(seeds)} episodes")
    print(f"   camera eye {args_cli.cam_eye} -> target {args_cli.cam_target}, "
          f"{math.dist(args_cli.cam_eye, args_cli.cam_target):.2f} m away, "
          f"{args_cli.hfov:.0f} deg HFOV ({focal:.1f} mm), "
          f"{args_cli.width}x{args_cli.height} @ {args_cli.fps} fps")

    tnames = list(e.termination_manager.active_terms)
    written = []
    for i, s in enumerate(seeds):
        env.reset(seed=s)
        ctrl.reset()
        obs = e.observation_manager.compute()["policy"]
        frames, outcome, succ, extracted = [], "TIMEOUT", False, False
        for t in range(args_cli.steps):
            obs, _, terminated, truncated, _ = env.step(ctrl.act(obs))
            obs = obs["policy"] if isinstance(obs, dict) else obs
            succ = succ or bool(mdp_cl.target_at_goal(e)[0])
            extracted = extracted or bool(mdp_cl.target_extracted(e)[0])
            frames.append(grab(cam))
            if bool(terminated[0]) or bool(truncated[0]):
                # Which term fired THIS step, read before the auto-reset can be believed --
                # the scene is already re-spawned by the time this line runs (R23).
                fired = [n for n in tnames if bool(e.termination_manager.get_term(n)[0])]
                outcome = "-".join(fired) if fired else "DONE"
                break
        if succ:
            outcome = "SUCCESS"
        elif outcome == "TIMEOUT":
            outcome = "STALLED" if extracted else "NO-GRASP"

        name = f"ep{i}_seed{s}_{outcome}"
        path = os.path.join(args_cli.out_dir, name + ".mp4")
        imageio.mimsave(path, frames, fps=args_cli.fps, macro_block_size=1)
        if args_cli.stills:
            imageio.imwrite(os.path.join(args_cli.out_dir, name + ".png"),
                            frames[len(frames) // 2])
        written.append((path, len(frames), outcome))
        print(f"   ep{i} seed {s}: {outcome:<20} {len(frames):4} frames "
              f"({len(frames) / args_cli.fps:5.1f} s) -> {os.path.basename(path)}")

    print(f"\n   wrote {len(written)} videos to {args_cli.out_dir}")
    n_ok = sum(1 for _, _, o in written if o == "SUCCESS")
    print(f"   {n_ok}/{len(written)} succeeded "
          f"(the checkpoint's measured rate is in runs/bc_eval_v3_s*.json)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
