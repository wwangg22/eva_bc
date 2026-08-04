#!/usr/bin/env python
"""Evaluate a vision flow student on the slot task (VISION_PLAN G3/G4).

The student sees the two cameras and 23-D proprioception. The privileged 34-D observation is
read **only** to score the episode -- success, depth, lateral -- and never reaches the policy;
`cameras.student_proprio` is the single place the student's input is built.

**The render config is checked, not assumed.** EXP08 evaluated a DLSS-trained student under
AA-off and scored 0.0 %/1.6 % -- floored, and silently, because nothing compared the two. The
training checkpoint carries the contract its data was collected under and this refuses to run on
a mismatch.

    python scripts/eval_vision.py --ckpt runs/vision_bc/v1/ckpt_final.pt \
        --episodes 128 --num-envs 16 --seed 777 --out runs/vision_bc/v1/eval_s777.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", required=True)
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--out", required=True)
parser.add_argument("--blind", action="store_true",
                    help="zero the images at eval too. Must match how the checkpoint was "
                         "trained; a blind checkpoint evaluated sighted (or vice versa) is the "
                         "same train/test mismatch that floored EXP08's student.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
import slot_mdp as mdp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from slot_act.cameras import attach_cameras, rgb_native  # noqa: E402
from slot_act.eval_flow_vision import VisionController, load_vision_checkpoint  # noqa: E402

IMAGE_KEYS = ("observation.images.wrist", "observation.images.workspace")


def main() -> None:
    device = "cuda:0"
    policy, stats, cfg = load_vision_checkpoint(Path(args.ckpt), device)
    render = cfg.get("render") or {}
    ss = int(render.get("supersample", 4))
    trained_blind = bool(cfg.get("blind", False))
    assert trained_blind == args.blind, (
        f"checkpoint was trained blind={trained_blind} but eval asked blind={args.blind}; "
        "that is a train/test mismatch, not a comparison")
    print(f"[eval_vision] render contract from checkpoint: {render}  blind={args.blind}")

    controller = VisionController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.seed = args.seed
    attach_cameras(env_cfg, supersample=ss)
    assert env_cfg.sim.render.antialiasing_mode == render.get("antialiasing", "Off")

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs
    obs = env.reset()[0]["policy"]

    ep_index = torch.zeros(n, dtype=torch.long)
    ep_len = torch.zeros(n, dtype=torch.long)
    ep_max_z = torch.zeros(n, len(mdp.OBJECT_NAMES))
    records: list[dict] = []

    zero_img = None
    while len(records) < args.episodes:
        # student view -- privileged obs is used below for METRICS only
        stu = {"joint_pos": obs[:, 0:8], "joint_vel": obs[:, 8:16], "actions": obs[:, 27:34],
               "wrist_rgb": rgb_native(u, "wrist_cam"),
               "workspace_rgb": rgb_native(u, "workspace_cam")}
        if args.blind:
            if zero_img is None:
                zero_img = torch.zeros_like(stu["wrist_rgb"])
            stu["wrist_rgb"] = zero_img
            stu["workspace_rgb"] = zero_img

        placed = mdp.placed_mask(u)
        last_placed = placed.all(dim=1)
        pos = torch.stack([mdp.object_pos_local(u, nm) for nm in mdp.OBJECT_NAMES], dim=1)
        ep_max_z = torch.maximum(ep_max_z, pos[..., 2].cpu())
        depth = mdp.insertion_depth(u).cpu()
        lat = mdp.lateral_error(u).cpu()
        yaw = mdp.yaw_error(u).cpu()

        action = controller.act(stu).to(u.device)
        obs, _, term, trunc, _ = env.step(action)
        obs = obs["policy"]
        ep_len += 1

        done = (term | trunc).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            for i in done.tolist():
                records.append({
                    "episode": len(records), "env": i,
                    "episode_index_in_env": int(ep_index[i]), "length": int(ep_len[i]),
                    "success": bool(last_placed[i]),
                    "max_obj_z": [round(float(z), 3) for z in ep_max_z[i]],
                    "final_obj_pos": [[round(float(v), 3) for v in pos[i, j].cpu()]
                                      for j in range(len(mdp.OBJECT_NAMES))],
                    "depth_mm": round(float(depth[i]) * 1000, 2),
                    "lateral_mm": round(float(lat[i]) * 1000, 2),
                    "yaw_rad": round(float(yaw[i]), 4),
                })
            controller.reset(done)
            ep_index[done] += 1
            ep_len[done] = 0
            ep_max_z[done] = 0.0

    records = records[: args.episodes]
    first = [r for r in records if r["episode_index_in_env"] == 0]
    later = [r for r in records if r["episode_index_in_env"] > 0]
    rate = lambda rs: (sum(r["success"] for r in rs) / len(rs)) if rs else None  # noqa: E731
    result = {
        "ckpt": args.ckpt, "task": args.task, "seed": args.seed, "blind": args.blind,
        "episodes": len(records),
        "success_rate": sum(r["success"] for r in records) / len(records),
        "success_rate_first_episode": rate(first), "success_rate_later": rate(later),
        "n_first_episode": len(first), "n_later": len(later),
        "config": {**cfg, "num_envs": n, "supersample": ss},
        "per_episode": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[eval_vision] {len(records)} eps: success={result['success_rate']:.3f}  "
          f"later={result['success_rate_later']:.3f} (n={len(later)})  -> {args.out}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
