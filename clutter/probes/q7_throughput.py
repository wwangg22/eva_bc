# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""Q7 -- Throughput and VRAM on THIS card, at the env counts every later stage assumes.

Why this is not optional
------------------------
Every env-count budget inherited from either repo was written for different hardware.
eva_bc's numbers come from a **12 GB** card (2048-env residual PPO at ~15 k env-steps/s,
steering PPO at ~740 windows/s). `challenge/agents/rl_games_ppo_cfg.yaml` defaults to
**2048 envs**. This machine has **10 GiB**. Nothing has ever measured what fits.

That matters in three specific places:

* **demo generation** -- batched rollouts at N envs; N sets the wall clock directly;
* **BC eval** -- pooled >=128-episode numbers are a standing requirement, and the eval cost
  is `episodes / N` rollouts of 690 steps;
* **steering PPO** -- `horizon_length 24 x num_envs` must stay divisible by the minibatch
  size, and an OOM at epoch 300 of a 3.7 h run costs the whole run.

This has been outstanding since Stage 0 and now blocks Stage 2's sizing.

Method
------
Zero actions (`a = 0` holds `_START_POSE` with the gripper OPEN -- see `03_ENV_FACTS.md` §10.6),
so nothing is measured except the simulator and the manager stack. A warm-up pass is discarded
because the first steps pay for CUDA-graph capture and lazy buffer allocation.

Reported per env count: env-steps/s, wall time for 690 steps (one eval episode), peak
allocated and peak reserved VRAM, and `nvidia-smi` used-memory as the number that actually
decides whether a second job fits.

**One env count per process.** A second `gym.make` in one process fails with
`Simulation context already exists`, which is what killed P21's `Tight-v0` attempt. The driver
loop is in the shell, and each run appends one record.

Usage
-----
    for N in 16 64 128 256 512 1024 2048; do
      python -u eva_bc/clutter/probes/q7_throughput.py --num_envs $N || break
    done
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Throughput and VRAM at one env count.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=200, help="timed env steps")
parser.add_argument("--warmup", type=int, default=30)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/q7_throughput.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import subprocess
import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401

EVAL_EPISODE_STEPS = 690   # episode_length_s 13.8 = 46 windows of 15


def smi_used_mib() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.strip().splitlines()[0])
    except Exception:
        return float("nan")


def main() -> None:
    n = args_cli.num_envs
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    env_cfg.episode_length_s = 13.8
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    torch.cuda.reset_peak_memory_stats()
    env.reset()

    a = torch.zeros((n, 7), device=e.device)
    for _ in range(args_cli.warmup):
        env.step(a)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args_cli.steps):
        env.step(a)
    torch.cuda.synchronize()
    dt = time.time() - t0

    sps = args_cli.steps / dt
    rec = {
        "task": args_cli.task,
        "num_envs": n,
        "steps": args_cli.steps,
        "sec": round(dt, 3),
        "iters_per_s": round(sps, 2),
        "env_steps_per_s": round(sps * n, 1),
        "sec_per_eval_episode": round(EVAL_EPISODE_STEPS / sps, 1),
        "peak_alloc_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "smi_used_MiB": smi_used_mib(),
    }

    print("\n" + "=" * 90)
    print(f"Q7 -- num_envs = {n}")
    print("=" * 90)
    print(f"   iterations/s        {rec['iters_per_s']:>10.2f}")
    print(f"   env-steps/s         {rec['env_steps_per_s']:>10.1f}")
    print(f"   one 690-step episode{rec['sec_per_eval_episode']:>10.1f} s  "
          f"(={n} episodes in parallel)")
    print(f"   torch peak alloc    {rec['peak_alloc_MiB']:>10.1f} MiB")
    print(f"   torch peak reserved {rec['peak_reserved_MiB']:>10.1f} MiB")
    print(f"   nvidia-smi used     {rec['smi_used_MiB']:>10.1f} MiB  of 10240")
    print("=" * 90)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    hist = []
    if os.path.exists(args_cli.out):
        try:
            hist = json.load(open(args_cli.out))
        except Exception:
            hist = []
    hist = [h for h in hist if not (h.get("num_envs") == n and h.get("task") == args_cli.task)]
    hist.append(rec)
    hist.sort(key=lambda h: (h.get("task", ""), h.get("num_envs", 0)))
    with open(args_cli.out, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[q7] appended to {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
