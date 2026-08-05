#!/usr/bin/env python
"""EXP09 R3' AWR collector: drive frozen v4 with per-boundary steering z, record
(images, proprio, z, window reward, placed delta, outcome) for offline AWR.

C2 discipline: EVERY iteration (random z or head-driven) uses the SAME loop
compute — one encode + manual Euler via model.velocity (mirrors
FlowMatchingPolicy.predict_action_chunk exactly), x0 = tanh(z) expanded. No
optional GPU work between steps; shards written after rolling stops.

    python experiments/exp09_awr_collect.py --episodes 64 --seed 3001 \
        --out-dir data/exp09_awr/it0_seed3001            # iteration 0: z ~ N(0,1)
    ... later: --head-ckpt runs/exp09/awr_it1/head.pt --explore-std 0.5
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-ckpt", default="runs/exp08_bc/v4_dagger3/ckpt_final.pt")
    parser.add_argument("--head-ckpt", default=None, help="steering head (tokens->z); absent = z~N(0,1)")
    parser.add_argument("--explore-std", type=float, default=1.0)
    parser.add_argument("--x0-scale", type=float, default=1.0,
                        help="alpha: x0 = alpha*tanh(z). Iteration-0 at alpha=1 drove 0.39-0.41 "
                             "vs the base's 0.64-0.69 -- chunk-shared full-magnitude x0 is "
                             "out-of-distribution for the vision base")
    parser.add_argument("--baseline", action="store_true",
                        help="transparent v4 baseline: x0 = per-element randn (what "
                             "predict_action_chunk draws internally), z recorded as zeros. "
                             "Same loop compute; for same-harness comparison runs only")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Vision-Play-v1")
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--out-dir", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import torch

    import reBot_RL.tasks  # noqa: F401
    import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
    from isaaclab_tasks.utils import parse_env_cfg

    from act.eval_flow_vision import VisionController, load_vision_checkpoint

    device = "cuda:0"
    root = Path(__file__).resolve().parent.parent
    policy, stats, cfg = load_vision_checkpoint(root / args.vision_ckpt, device)
    ctrl = VisionController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)
    window, chunk_size = cfg["n_action_steps"], cfg["chunk_size"]
    n_steps_int = policy.num_inference_steps if hasattr(policy, "num_inference_steps") else cfg["num_inference_steps"]

    head = None
    if args.head_ckpt:
        from experiments.exp09_awr_train import SteerHead  # same arch as trainer

        h = torch.load(root / args.head_ckpt, map_location="cpu")
        head = SteerHead(*h["tokens"]).to(device)
        head.load_state_dict(h["head_state_dict"])
        head.eval()

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs

    torch.manual_seed(args.seed)

    @torch.no_grad()
    def steered_chunk(stu):
        batch = ctrl.normalizer.normalize({k: v.to(device) for k, v in ctrl.build_batch(stu).items()})
        enc_out, enc_pos = policy.model.encode(policy._stack_images(batch))
        if args.baseline:
            z = torch.zeros(n, 7, device=device)
            x = torch.randn(n, chunk_size, 7, device=device)
        else:
            if head is not None:
                z = head(enc_out.permute(1, 0, 2)) + args.explore_std * torch.randn(n, 7, device=device)
            else:
                z = torch.randn(n, 7, device=device)
            x = args.x0_scale * torch.tanh(z).unsqueeze(1).expand(-1, chunk_size, -1).clone()
        dt = 1.0 / n_steps_int
        for i in range(n_steps_int):
            tau = torch.full((n,), i * dt, device=device)
            x = x + policy.model.velocity(enc_out, enc_pos, x, tau) * dt
        acts = ctrl.normalizer.unnormalize("action", x)[:, :window]
        return z, acts

    obs_dict, _ = env.reset()
    RAW = ("wrist_rgb", "workspace_rgb", "proprio", "z", "win_reward", "placed_before", "placed_after")
    bufs = [{k: [] for k in RAW} for _ in range(n)]
    ep_count = torch.zeros(n, dtype=torch.long)
    saved = success_count = step_i = 0
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[awr] {args.episodes} eps seed {args.seed} head={bool(head)} std={args.explore_std}", flush=True)

    win_reward = torch.zeros(n, device=u.device)
    while saved < args.episodes:
        stu = obs_dict["student"]
        if step_i % window == 0:
            placed_now = mdp.placed_mask(u).sum(dim=1)
            if step_i > 0:
                for i in range(n):
                    if bufs[i]["wrist_rgb"]:  # skip just-reset envs: no open window
                        bufs[i]["win_reward"].append(win_reward[i].clone())
                        bufs[i]["placed_after"].append(placed_now[i].clone())
            z, acts = steered_chunk(stu)
            for i in range(n):
                bufs[i]["wrist_rgb"].append(stu["wrist_rgb"][i].clone())
                bufs[i]["workspace_rgb"].append(stu["workspace_rgb"][i].clone())
                bufs[i]["proprio"].append(torch.cat([stu["joint_pos"][i], stu["joint_vel"][i], stu["actions"][i]]).clone())
                bufs[i]["z"].append(z[i].clone())
                bufs[i]["placed_before"].append(placed_now[i].clone())
            act_buf, act_idx = acts, 0
            win_reward = torch.zeros(n, device=u.device)

        placed_all = mdp.placed_mask(u).all(dim=1).cpu()
        obs_dict, rew, terminated, truncated, _ = env.step(act_buf[:, act_idx].to(u.device))
        win_reward += rew.view(-1)
        act_idx += 1
        step_i += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            placed_final = mdp.placed_mask(u).sum(dim=1)
            for i in done.tolist():
                if ep_count[i] > 0 and saved < args.episodes:
                    B = len(bufs[i]["win_reward"])  # complete windows only
                    torch.save(
                        {
                            "wrist_rgb": torch.stack(bufs[i]["wrist_rgb"][:B]).to(torch.uint8).cpu(),
                            "workspace_rgb": torch.stack(bufs[i]["workspace_rgb"][:B]).to(torch.uint8).cpu(),
                            "proprio": torch.stack(bufs[i]["proprio"][:B]).cpu(),
                            "z": torch.stack(bufs[i]["z"][:B]).cpu(),
                            "win_reward": torch.stack(bufs[i]["win_reward"]).cpu(),
                            "placed_before": torch.stack(bufs[i]["placed_before"][:B]).cpu(),
                            "placed_after": torch.stack(bufs[i]["placed_after"]).cpu(),
                            "success": bool(placed_all[i]),
                            "placed_final": int(placed_final[i]),
                            "seed": args.seed,
                            "head_ckpt": args.head_ckpt,
                            "explore_std": args.explore_std,
                        },
                        out_dir / f"ep_{saved:04d}.pt",
                    )
                    success_count += int(placed_all[i])
                    saved += 1
                    if saved % 16 == 0:
                        print(f"[awr] {saved}/{args.episodes}, success {success_count}/{saved}", flush=True)
                ep_count[i] += 1
                for v in bufs[i].values():
                    v.clear()

    rate = success_count / saved
    meta = {"vision_ckpt": args.vision_ckpt, "head_ckpt": args.head_ckpt,
            "explore_std": args.explore_std, "baseline": args.baseline,
            "x0_scale": args.x0_scale, "seed": args.seed, "episodes": saved,
            "driving_success_rate": rate, "num_envs": n, "window": window}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[awr] DONE {out_dir} driving={rate:.3f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
