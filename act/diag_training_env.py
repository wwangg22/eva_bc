#!/usr/bin/env python
"""EXP06 diagnostic — why do training rollouts collect negative reward?

Reproduces the EXACT training configuration (train_residual.py: Play-v1, drop
termination ON, 30 s horizon, fixed-x0-zeros base, ResidualRlGamesWrapper) and rolls
four fixed action conditions through it, no RL:

  zero   : residual = 0                      (the base itself, training env)
  bias   : residual = +0.125 on all joints   (rl_games default mu-BIAS magnitude —
                                              mu_init touches only the weight)
  noise8 : residual ~ N(0, 0.08) per step    (r2 exploration)
  noise37: residual ~ N(0, 0.37) per step    (r1 exploration)

Per condition: episodes done, success fraction (placed_mask pre-step), terminated-vs-
truncated split, mean episode length, mean episodic RAW reward (wrapper units — the
rl_games logs are these x0.01). Run: python diag_training_env.py --num_envs 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", default="../runs/exp03_N3/ckpt_final.pt")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--steps", type=int, default=1600, help="steps per condition (episode = 1500)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401
import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
from isaaclab_tasks.utils import parse_env_cfg

from act.eval_act import BatchedACTController, load_checkpoint
from act.eval_residual import make_fixed_x0
from act.residual_core import RES_ACTION_DIM, ResidualCore
from act.residual_wrapper import ResidualRlGamesWrapper


def main() -> None:
    device = "cuda:0"
    here = Path(__file__).resolve().parent
    policy, stats, ckpt_cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(
        policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device,
        fixed_x0=make_fixed_x0(ckpt_cfg["chunk_size"], 7, -1),
    )
    core = ResidualCore(controller, mdp, here.parent / "experiments" / "exp06_grasp_bit.pt",
                        alpha=0.1, flush=True)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.episode_length_s = 30.0
    env_cfg.seed = args.seed
    env = gymnasium.make(args.task, cfg=env_cfg)
    wrapped = ResidualRlGamesWrapper(env, device, 100.0, 100.0, core, res_penalty=0.01)

    n = args.num_envs
    g = torch.Generator(device=device).manual_seed(0)
    conditions = {
        "zero": lambda: torch.zeros(n, RES_ACTION_DIM, device=device),
        "bias": lambda: torch.full((n, RES_ACTION_DIM), 0.125, device=device),
        "noise8": lambda: torch.randn(n, RES_ACTION_DIM, device=device, generator=g) * 0.08,
        "noise37": lambda: torch.randn(n, RES_ACTION_DIM, device=device, generator=g) * 0.37,
    }

    for name, sample in conditions.items():
        obs = wrapped.reset()  # full env reset between conditions
        ep_rew = torch.zeros(n, device=device)
        ep_len = torch.zeros(n, device=device)
        done_rews, done_lens, done_succ, done_term = [], [], [], []
        u = env.unwrapped
        for _ in range(args.steps):
            placed_now = mdp.placed_mask(u).all(dim=1)  # pre-step, like eval
            o, rew, dones, extras = wrapped.step(sample())
            ep_rew += rew
            ep_len += 1
            d = dones.nonzero(as_tuple=False).squeeze(-1)
            if d.numel():
                timeout = extras.get("time_outs", torch.zeros_like(dones))
                done_rews.append(ep_rew[d].clone())
                done_lens.append(ep_len[d].clone())
                done_succ.append(placed_now[d].clone())
                done_term.append((~timeout.bool()[d]).clone())
                ep_rew[d] = 0.0
                ep_len[d] = 0.0
        if done_rews:
            r = torch.cat(done_rews); ln = torch.cat(done_lens)
            s = torch.cat(done_succ).float(); t = torch.cat(done_term).float()
            print(f"[{name}] eps={len(r)} success={s.mean():.3f} "
                  f"terminated_frac={t.mean():.3f} mean_len={ln.mean():.0f} "
                  f"mean_ep_raw_reward={r.mean():.2f}", flush=True)
        else:
            print(f"[{name}] NO episodes finished in {args.steps} steps", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
