#!/usr/bin/env python
"""EXP06 residual-path evaluation (gate 2 + trained-residual evals).

Rolls the frozen flow base THROUGH the residual machinery (ResidualCore obs builder +
compose) under the ladder's exact eval protocol (object_dropping disabled, 30 s
episodes, section 4.2 flush, fixed spawn seed) so results are episode-comparable with
runs/exp03_N3/eval_64ep*.json.

Modes:
  --residual-ckpt absent  -> zero residual. With --x0-mode global this must reproduce
                             the base eval episode-for-episode (EXP06 gate 2a); with
                             --x0-mode fixed it measures the base under the residual
                             training condition (gate 2b baseline).
  --residual-ckpt PATH    -> rl_games checkpoint; deterministic action = mu.

Run (env_isaaclab6):
    python eval_residual.py --ckpt ../runs/exp03_N3/ckpt_final.pt --seed 42 \
        --x0-mode global --out ../runs/exp06/gate2a_seed42.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset import OBS_DIM
from act.eval_act import BatchedACTController, load_checkpoint
from act.residual_core import RES_ACTION_DIM, RES_OBS_DIM, ResidualCore

FIXED_X0_SEED = 7  # historical default; gate 2b showed x0 choice moves the base — the
#                    training x0 is now selected by pooled-suite sweep (EXP06 doc).


def make_fixed_x0(chunk_size: int, action_dim: int, seed: int = FIXED_X0_SEED) -> torch.Tensor:
    """seed >= 0: N(0,1) draw from that seed; seed == -1: zeros (distribution mean)."""
    if seed == -1:
        return torch.zeros((chunk_size, action_dim))
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn((chunk_size, action_dim), generator=g)


def load_residual_policy(ckpt_path: Path, cfg_yaml: Path, device,
                         obs_dim: int = RES_OBS_DIM, act_dim: int = RES_ACTION_DIM):
    """Build the rl_games a2c network from the training yaml and load a checkpoint.

    Returns act(obs) -> deterministic mu (N, act_dim). Defaults are the EXP06
    residual dims; EXP07 steering passes 56/7.
    """
    import yaml
    from rl_games.algos_torch.model_builder import ModelBuilder

    params = yaml.safe_load(cfg_yaml.read_text())["params"]
    builder = ModelBuilder()
    network = builder.load(params)
    config = params["config"]
    model = network.build(
        {
            "actions_num": act_dim,
            "input_shape": (obs_dim,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": config.get("normalize_value", False),
            "normalize_input": config.get("normalize_input", False),
        }
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    @torch.no_grad()
    def act(obs64: torch.Tensor) -> torch.Tensor:
        out = model({"is_train": False, "prev_actions": None, "obs": obs64})
        # match rl_games' training-time preprocess (clamp to [-1,1]; with
        # clip_actions 1.0 the subsequent rescale is the identity)
        return torch.clamp(out["mus"], -1.0, 1.0)

    return act


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="frozen base checkpoint (train_flow.py .pt)")
    parser.add_argument("--residual-ckpt", default=None, help="rl_games residual checkpoint (.pth); absent = zero residual")
    parser.add_argument("--residual-cfg", default=str(Path(__file__).parent / "residual_ppo_cfg.yaml"))
    parser.add_argument("--alpha", type=float, default=0.1, help="residual bound (env action units)")
    parser.add_argument("--x0-mode", choices=["global", "fixed"], default="fixed",
                        help="'global' = historical eval RNG (gate 2a comparability); "
                             "'fixed' = shared x0, deterministic base (training condition)")
    parser.add_argument("--x0-seed", type=int, default=FIXED_X0_SEED,
                        help="fixed-x0 noise seed; -1 = zeros (only used with --x0-mode fixed)")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--out", required=True, help="results JSON path")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym

    import reBot_RL.tasks  # noqa: F401
    import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
    from isaaclab_tasks.utils import parse_env_cfg

    device = "cuda:0"
    policy, stats, ckpt_cfg = load_checkpoint(Path(args.ckpt), device)
    fixed_x0 = (
        make_fixed_x0(ckpt_cfg["chunk_size"], 7, args.x0_seed) if args.x0_mode == "fixed" else None
    )
    controller = BatchedACTController(
        policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device,
        fixed_x0=fixed_x0,
    )
    core = ResidualCore(controller, mdp, Path(__file__).parent.parent / "experiments" / "exp06_grasp_bit.pt",
                        alpha=args.alpha, flush=args.flush)

    res_act = None
    if args.residual_ckpt is not None:
        res_act = load_residual_policy(Path(args.residual_ckpt), Path(args.residual_cfg), device)

    # --- env: EXACT ladder eval protocol (see act/eval_act.py) ---
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    assert env_cfg.terminations.time_out is not None
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    assert obs.shape == (n, OBS_DIM), obs.shape
    ep_len = torch.zeros(n, dtype=torch.long)
    ep_max_placed = torch.zeros(n, dtype=torch.long)
    ep_final_placed = torch.zeros(n, dtype=torch.long)
    ep_max_can_z = torch.zeros(n, len(mdp.OBJECT_NAMES))
    ep_res_mag = torch.zeros(n)
    records: list[dict] = []

    while len(records) < args.episodes:
        placed_now = mdp.placed_mask(u)
        last_placed = placed_now.all(dim=1)
        ep_max_placed = torch.maximum(ep_max_placed, placed_now.sum(dim=1).cpu())
        ep_final_placed = placed_now.sum(dim=1).cpu()
        pos = core._can_pos_local(u)
        ep_max_can_z = torch.maximum(ep_max_can_z, pos[..., 2].cpu())

        obs64 = core.advance(obs, u)
        if res_act is None:
            residual = torch.zeros(n, RES_ACTION_DIM, device=device)
        else:
            residual = res_act(obs64)
        full, applied = core.compose(residual)
        ep_res_mag += applied.abs().mean(dim=1).cpu()

        obs_dict, _, terminated, truncated, _ = env.step(full.to(u.device))
        obs = obs_dict["policy"]
        ep_len += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            for i in done.tolist():
                records.append(
                    {
                        "episode": len(records),
                        "env": i,
                        "length": int(ep_len[i]),
                        "success": bool(last_placed[i]),
                        "placed_final": int(ep_final_placed[i]),
                        "placed_max": int(ep_max_placed[i]),
                        "max_can_z": [round(float(z), 3) for z in ep_max_can_z[i]],
                        "mean_res_mag": round(float(ep_res_mag[i] / max(int(ep_len[i]), 1)), 5),
                    }
                )
            core.reset(done.tolist())
            ep_len[done] = 0
            ep_max_placed[done] = 0
            ep_final_placed[done] = 0
            ep_max_can_z[done] = 0.0
            ep_res_mag[done] = 0.0

    records = records[: args.episodes]
    n_ep = len(records)
    result = {
        "ckpt": args.ckpt,
        "residual_ckpt": args.residual_ckpt,
        "x0_mode": args.x0_mode,
        "x0_seed": args.x0_seed if args.x0_mode == "fixed" else None,
        "alpha": args.alpha,
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
    print(f"[eval_residual] {n_ep} eps: success_rate={result['success_rate']:.3f} -> {args.out}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
