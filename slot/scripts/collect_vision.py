#!/usr/bin/env python
"""Collect champion rollouts with cameras into per-episode shards (VISION_PLAN G1/G2).

**The rollout loop does no optional GPU work.** EXP08 lost 40 points of student success to a
collector that computed teacher features *inside* the driving loop and perturbed the frames it
was recording. Everything here is buffer reads plus one device->host copy per step; anything
derived is computed after the episode ends. That rule holds even though we run no temporal
filter -- it costs nothing and it removes the whole failure class.

**The render configuration is part of the dataset.** It is written into every shard and the
trainer/eval must refuse a mismatch: EXP08 evaluated a DLSS-trained student under AA-off and
scored 0.0 %/1.6 % -- floored, not degraded, and silently.

Shard (``ep_XXXX.pt``), matching slot_act/dataset_vision.py:
    wrist_rgb / workspace_rgb  (T, 90, 160, 3) uint8   <- box-averaged from the 8x render
    proprio                    (T, 23) float32          <- student, via cameras.student_proprio
    actions                    (T, 7)  float32
    obs34                      (T, 34) float32          TEACHER-ONLY, never read by the student
    success                    bool
    render                     dict                     the contract

    python scripts/collect_vision.py --episodes 4 --num-envs 2 --out data/vision_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", default="runs/bc_armB_seed0/ckpt_final.pt")
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--out", required=True)
parser.add_argument("--supersample", type=int, default=None, help="default: cameras.SUPERSAMPLE")
parser.add_argument("--warmup-episodes", type=int, default=1,
                    help="episodes per env discarded: PhysX warm start AND renderer cold start "
                         "(EXP08 measured a cold round 1 at 6/16 vs a warm 78-83%%)")
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

from slot_act.cameras import (  # noqa: E402
    CAMERA_NAMES,
    SUPERSAMPLE,
    attach_cameras,
    rgb_native,
    student_proprio,
)
from slot_act.eval_act import BatchedACTController, load_checkpoint  # noqa: E402


def main() -> None:
    device = "cuda:0"
    ss = SUPERSAMPLE if args.supersample is None else args.supersample
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy, stats, cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.seed = args.seed
    attach_cameras(env_cfg, supersample=ss)
    render = {"supersample": ss, "width": 160, "height": 90, "antialiasing": "Off",
              "update_period": env_cfg.decimation * env_cfg.sim.dt}

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs
    obs = env.reset()[0]["policy"]
    assert obs.shape == (n, 34), obs.shape

    ep_idx = torch.zeros(n, dtype=torch.long)
    buf = [{"wrist": [], "workspace": [], "proprio": [], "actions": [], "obs34": []}
           for _ in range(n)]
    kept = 0
    n_succ = 0
    t0 = time.time()
    steps = 0

    while kept < args.episodes:
        # ---- the driving loop. Buffer reads and one H2D-free clone per camera. Nothing else.
        wrist = rgb_native(u, "wrist_cam")
        works = rgb_native(u, "workspace_cam")
        success_now = mdp.placed_mask(u).all(dim=1)
        action = controller.act(obs)
        for i in range(n):
            b = buf[i]
            b["wrist"].append(wrist[i].cpu())
            b["workspace"].append(works[i].cpu())
            b["obs34"].append(obs[i].cpu())
            b["actions"].append(action[i].cpu())
        obs_next, _, term, trunc, _ = env.step(action.to(u.device))
        obs_next = obs_next["policy"]
        steps += 1

        done = (term | trunc).view(-1).nonzero(as_tuple=False).squeeze(-1)
        for i in done.tolist():
            b = buf[i]
            if ep_idx[i] >= args.warmup_episodes and kept < args.episodes:
                o34 = torch.stack(b["obs34"])
                shard = {
                    "wrist_rgb": torch.stack(b["wrist"]),
                    "workspace_rgb": torch.stack(b["workspace"]),
                    # computed AFTER the rollout of this episode, from recorded state
                    "proprio": student_proprio(o34),
                    "actions": torch.stack(b["actions"]),
                    "obs34": o34,
                    "success": bool(success_now[i]),
                    "render": render,
                    "env": i, "episode_index_in_env": int(ep_idx[i]), "seed": args.seed,
                }
                assert shard["proprio"].shape[1] == 23, shard["proprio"].shape
                assert shard["wrist_rgb"].shape[1:] == (90, 160, 3), shard["wrist_rgb"].shape
                torch.save(shard, out / f"ep_{kept:04d}.pt")
                kept += 1
                n_succ += int(shard["success"])
                if kept % 8 == 0 or kept == args.episodes:
                    el = time.time() - t0
                    print(f"[collect] {kept}/{args.episodes} eps  success {n_succ}/{kept} "
                          f"({n_succ / kept:.3f})  {steps / el:.1f} env-steps/s  "
                          f"{el / 60:.1f} min", flush=True)
            ep_idx[i] += 1
            for k in b:
                b[k].clear()
        obs = obs_next

    meta = {"ckpt": args.ckpt, "task": args.task, "seed": args.seed, "num_envs": n,
            "episodes": kept, "success_rate": n_succ / max(kept, 1), "render": render,
            "warmup_episodes": args.warmup_episodes,
            "env_steps_per_s": round(steps / (time.time() - t0), 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[collect] DONE {kept} eps, success {meta['success_rate']:.3f} "
          f"(champion state-only reference: 0.979), {meta['env_steps_per_s']} env-steps/s -> {out}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
