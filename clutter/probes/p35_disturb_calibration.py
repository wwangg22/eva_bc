# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P35 -- calibrate the 2 mm disturbance threshold against the solver's own noise.

`DISTURB_TOL = 0.002` is now a **termination**: move a neighbour more than 2 mm and the
episode ends in failure. That makes it a floor on what the environment can measure, and a
threshold below the noise floor would fail every episode regardless of what the policy does
-- the whole task would read 0 % and the number would mean nothing.

So this is the positive control the Stage-0 rule demands: **before trusting a constraint,
show that it does not fire when nothing violates it.**

The control
-----------
Reset, then submit a null action every step for a full episode. `a = 0` decodes to
`q_target = q_default + 0.5*0 = q_default`, so the arm holds its reset pose, which is well
clear of the row; the gripper closes on air (`a[6] = 0` is not `> 0`). Nothing in the scene
should move. Any displacement measured is therefore attributable to the solver: settling
after the reset write, contact jitter between neighbours whose jitter ranges brought them
close, and integration drift over ~700 steps.

Run on `-Lenient-v0` deliberately. The main task would terminate the instant the threshold
were crossed and auto-reset the scene from inside `env.step` (R23), which would both
truncate the measurement and corrupt it -- the post-step scene is already re-spawned. The
lenient variant has the same physics and no disturbance termination, so the displacement
can be watched for the whole episode.

What would refute the threshold
-------------------------------
* any env reaching 2 mm under a null action        -> the threshold is inside the noise
* p99 above ~0.5 mm                                -> less than 4x margin, too close
* drift growing without bound in step              -> the tolerance cannot be a constant

Registered prediction (before running): **max displacement < 0.2 mm over the whole episode,
0 of 128 envs disturbed.** The demo collector's spawn fingerprints round positions at 0.1 mm
and have always paired correctly across the 30-step settle, which bounds the settle at
< 0.1 mm; this extends that bound from 30 steps to the full episode.

Usage
-----
    python -u clutter/probes/p35_disturb_calibration.py --num_envs 128 --steps 700
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="P35 -- disturbance-threshold calibration.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Lenient-v0",
                    help="lenient by design: the main task terminates on the very quantity "
                         "being measured, and auto-resets the scene from inside env.step")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=700,
                    help="700 env steps = 14 s = one full episode at the cfg's length")
parser.add_argument("--seeds", type=str, default="88000,88001,88002,88003,88004,88005",
                    help="the held-out eval seeds, so the calibration covers exactly the "
                         "spawns every reported number is measured on")
parser.add_argument("--json", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl  # noqa: E402

#: the milestones the write-up quotes, in env steps
MARKS = (1, 5, 30, 100, 325, 700)


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    e = env.unwrapped
    n = e.num_envs
    seeds = [int(s) for s in args_cli.seeds.split(",")]

    print("\n" + "=" * 100)
    print("P35  DISTURBANCE-THRESHOLD CALIBRATION -- is 2 mm above the solver noise floor?")
    print("=" * 100)
    print(f"   task {args_cli.task}   {n} envs x {len(seeds)} seeds = {n * len(seeds)} episodes")
    print(f"   null action every step for {args_cli.steps} env steps")
    print(f"   DISTURB_TOL = {mdp_cl.DISTURB_TOL * 1e3:.1f} mm")
    print(f"   PREDICTION: max < 0.2 mm, 0 envs disturbed\n")

    zero = torch.zeros(n, e.action_manager.total_action_dim, device=e.device)
    per_seed, at_mark = [], {m: [] for m in MARKS}
    all_worst = []

    for s in seeds:
        env.reset(seed=s)
        worst = torch.zeros(n, device=e.device)   # latched, per env
        for t in range(1, args_cli.steps + 1):
            env.step(zero)
            worst = torch.maximum(worst, mdp_cl.max_distractor_displacement(e))
            if t in at_mark:
                at_mark[t].append(worst.clone())
        all_worst.append(worst.clone())
        n_bad = int((worst > mdp_cl.DISTURB_TOL).sum())
        per_seed.append({"seed": s, "max_mm": float(worst.max()) * 1e3,
                         "mean_mm": float(worst.mean()) * 1e3, "n_disturbed": n_bad})
        print(f"   seed {s}:  max {float(worst.max()) * 1e3:7.4f} mm   "
              f"mean {float(worst.mean()) * 1e3:7.4f} mm   disturbed {n_bad}/{n}")

    w = torch.cat(all_worst)
    q = torch.tensor([0.50, 0.90, 0.99, 1.00], device=w.device)
    p50, p90, p99, pmax = (torch.quantile(w, q) * 1e3).tolist()
    n_bad = int((w > mdp_cl.DISTURB_TOL).sum())

    print("\n" + "-" * 100)
    print("   DRIFT vs EPISODE STEP  (max over all envs of the latched worst displacement)")
    for m in MARKS:
        if at_mark[m]:
            v = torch.cat(at_mark[m])
            print(f"      step {m:4}:  max {float(v.max()) * 1e3:7.4f} mm   "
                  f"p99 {float(torch.quantile(v, 0.99)) * 1e3:7.4f} mm")

    print("\n" + "-" * 100)
    print(f"   NULL-ACTION DISPLACEMENT over {w.numel()} episodes")
    print(f"      p50 {p50:7.4f} mm   p90 {p90:7.4f} mm   p99 {p99:7.4f} mm   max {pmax:7.4f} mm")
    print(f"      disturbed at 2 mm: {n_bad}/{w.numel()} = {100.0 * n_bad / w.numel():.2f} %")
    margin = mdp_cl.DISTURB_TOL * 1e3 / max(pmax, 1e-9)
    print(f"      margin: the threshold is {margin:.1f}x the worst observed noise")
    verdict = ("THRESHOLD SOUND" if n_bad == 0 and margin >= 4.0 else
               "THRESHOLD TOO TIGHT -- it fires on solver noise")
    print(f"\n   VERDICT: {verdict}")
    print("=" * 100 + "\n")

    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump({"task": args_cli.task, "steps": args_cli.steps, "n_envs": n,
                       "tol_m": mdp_cl.DISTURB_TOL, "per_seed": per_seed,
                       "p50_mm": p50, "p90_mm": p90, "p99_mm": p99, "max_mm": pmax,
                       "n_disturbed": n_bad, "n_episodes": int(w.numel()),
                       "margin_x": margin, "verdict": verdict}, f, indent=2)
        print(f"   wrote {args_cli.json}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
