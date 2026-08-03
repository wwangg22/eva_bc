#!/usr/bin/env python
"""EXP07 x0-steering evaluation (gates S0/1 + trained-steering evals).

Rolls the frozen flow base through the chunk-window steering path (act/steer_core.py)
under the ladder's exact eval protocol (object_dropping disabled, 30 s episodes,
section 4.2 flush, fixed spawn seed) so results are episode-comparable with
runs/exp06_residual/x0sweep_s-1_seed{42,123}.json (the 55.5%-pooled x0-zeros base).

z source per window (all envs share the boundary clock):
  --steer-ckpt PATH  -> rl_games checkpoint; deterministic z = clamp(mu, -1, 1).
  absent             -> fixed-z modes: z = bias_vec + sigma * N(0, I), resampled per
                        window per env from --z-seed; bias_vec ~ U(+/-z_bias) per dim,
                        drawn once. Defaults (sigma 0, bias 0) give z = 0, which must
                        reproduce the x0-zeros base BIT-EXACTLY (gate 1).

Run (env_isaaclab6):
    python eval_steer.py --ckpt ../runs/exp03_N3/ckpt_final.pt --seed 42 \
        --out ../runs/exp07_steer/gate1_seed42.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset import OBS_DIM
from act.eval_act import BatchedACTController, load_checkpoint
from act.eval_residual import load_residual_policy
from act.steer_core import STEER_ACTION_DIM, STEER_OBS_DIM, SteerCore


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="frozen base checkpoint (train_flow.py .pt)")
    parser.add_argument("--steer-ckpt", default=None, help="rl_games steering checkpoint (.pth); absent = fixed-z mode")
    parser.add_argument("--steer-cfg", default=str(Path(__file__).parent / "steer_ppo_cfg.yaml"))
    parser.add_argument("--alpha-x0", type=float, default=1.0, help="x0 bound: x0 = alpha_x0 * tanh(z)")
    parser.add_argument("--z-sigma", type=float, default=0.0,
                        help="fixed-z mode: z ~ N(0, sigma) resampled per window per env (gate S0)")
    parser.add_argument("--z-bias", type=float, default=0.0,
                        help="fixed-z mode: constant per-dim bias ~ U(+/-z_bias), drawn once (mu-bias condition)")
    parser.add_argument("--z-seed", type=int, default=0, help="fixed-z mode noise seed")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--out", required=True, help="results JSON path")
    parser.add_argument("--video", action="store_true", help="record an rgb_array video (offscreen render)")
    parser.add_argument("--video-length", type=int, default=3000, help="video length in env steps")
    parser.add_argument("--video-folder", default=None, help="output dir (default: videos/ next to --out)")
    parser.add_argument("--viewer-eye", type=float, nargs=3, default=None,
                        help="viewer camera eye xyz (close-up framing)")
    parser.add_argument("--viewer-lookat", type=float, nargs=3, default=None,
                        help="viewer camera lookat xyz")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.video:
        args.enable_cameras = True
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym

    import reBot_RL.tasks  # noqa: F401
    import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
    from isaaclab_tasks.utils import parse_env_cfg

    device = "cuda:0"
    policy, stats, ckpt_cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(
        policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device
    )
    core = SteerCore(controller, mdp, Path(__file__).parent.parent / "experiments" / "exp06_grasp_bit.pt",
                     alpha_x0=args.alpha_x0, flush=args.flush)
    window = controller.n_action_steps

    steer_act = None
    if args.steer_ckpt is not None:
        steer_act = load_residual_policy(Path(args.steer_ckpt), Path(args.steer_cfg), device,
                                         obs_dim=STEER_OBS_DIM, act_dim=STEER_ACTION_DIM)
    zgen = torch.Generator(device="cpu").manual_seed(args.z_seed)
    z_bias_vec = torch.empty(STEER_ACTION_DIM).uniform_(-args.z_bias, args.z_bias, generator=zgen)

    # --- env: EXACT ladder eval protocol (see act/eval_act.py) ---
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    assert env_cfg.terminations.time_out is not None
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    if args.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(args.viewer_eye)
    if args.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(args.viewer_lookat)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder or str(Path(args.out).parent / "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    u = env.unwrapped
    n = args.num_envs
    assert u.max_episode_length % window == 0, (
        f"episode length {u.max_episode_length} steps must be a multiple of the "
        f"{window}-step window (window-aligned protocol)"
    )

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    assert obs.shape == (n, OBS_DIM), obs.shape
    ep_len = torch.zeros(n, dtype=torch.long)
    ep_max_placed = torch.zeros(n, dtype=torch.long)
    ep_final_placed = torch.zeros(n, dtype=torch.long)
    ep_max_can_z = torch.zeros(n, len(mdp.OBJECT_NAMES))
    ep_z_mag = torch.zeros(n)
    ep_z_sum = torch.zeros(n, STEER_ACTION_DIM)
    ep_z_sq = torch.zeros(n, STEER_ACTION_DIM)
    ep_z_cnt = torch.zeros(n)
    records: list[dict] = []
    step_i = 0

    while len(records) < args.episodes:
        if step_i % window == 0:  # window boundary: choose z, hold for 15 steps
            obs56 = core.build_obs(obs, u)
            if steer_act is not None:
                z = steer_act(obs56)
            else:
                z = z_bias_vec + args.z_sigma * torch.randn((n, STEER_ACTION_DIM), generator=zgen)
            applied = core.set_steer(z).cpu()  # (N, 7) = alpha_x0 * tanh(z)
            ep_z_mag += applied.abs().mean(dim=1)
            ep_z_sum += applied
            ep_z_sq += applied.square()
            ep_z_cnt += 1

        placed_now = mdp.placed_mask(u)
        last_placed = placed_now.all(dim=1)
        ep_max_placed = torch.maximum(ep_max_placed, placed_now.sum(dim=1).cpu())
        ep_final_placed = placed_now.sum(dim=1).cpu()
        pos = core._can_pos_local(u)
        ep_max_can_z = torch.maximum(ep_max_can_z, pos[..., 2].cpu())

        core.flush_check(u)
        action = core.controller.act(obs)

        obs_dict, _, terminated, truncated, _ = env.step(action.to(u.device))
        obs = obs_dict["policy"]
        ep_len += 1
        step_i += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            for i in done.tolist():
                cnt = max(float(ep_z_cnt[i]), 1.0)
                var = (ep_z_sq[i] / cnt - (ep_z_sum[i] / cnt).square()).clamp_min(0.0)
                records.append(
                    {
                        "episode": len(records),
                        "env": i,
                        "length": int(ep_len[i]),
                        "success": bool(last_placed[i]),
                        "placed_final": int(ep_final_placed[i]),
                        "placed_max": int(ep_max_placed[i]),
                        "max_can_z": [round(float(z_), 3) for z_ in ep_max_can_z[i]],
                        "mean_z_mag": round(float(ep_z_mag[i] / cnt), 5),
                        "z_std": round(float(var.sqrt().mean()), 5),
                    }
                )
            core.reset(done.tolist())
            ep_len[done] = 0
            ep_max_placed[done] = 0
            ep_final_placed[done] = 0
            ep_max_can_z[done] = 0.0
            ep_z_mag[done] = 0.0
            ep_z_sum[done] = 0.0
            ep_z_sq[done] = 0.0
            ep_z_cnt[done] = 0.0

    records = records[: args.episodes]
    n_ep = len(records)
    result = {
        "ckpt": args.ckpt,
        "steer_ckpt": args.steer_ckpt,
        "alpha_x0": args.alpha_x0,
        "z_sigma": args.z_sigma,
        "z_bias": args.z_bias,
        "z_seed": args.z_seed,
        "task": args.task,
        "seed": args.seed,
        "episodes": n_ep,
        "success_rate": sum(r["success"] for r in records) / n_ep,
        "mean_ep_len": sum(r["length"] for r in records) / n_ep,
        "flush_count": core.flush_count,
        "per_episode": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[eval_steer] {n_ep} eps: success_rate={result['success_rate']:.3f} -> {args.out}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
