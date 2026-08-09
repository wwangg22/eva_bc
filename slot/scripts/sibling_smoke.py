#!/usr/bin/env python
"""Do the three OTHER challenge tasks still work after the slot edits to ``challenge/mdp``?

Big Will's constraint on the eva_rl change was "only touch your assigned task (precision slot),
and not any other task". ``challenge/mdp/`` is shared by four tasks, so the scoping analysis
(docs/slot/ANGLED_SLOT_PLAN.md 0b) restricted the edit to symbols with zero sibling users. This
script is the check on that analysis rather than a restatement of it: each sibling env is
built, reset and stepped, and its observation width and reward keys are printed.

One process per task -- Isaac Sim will not build a second env after the first is torn down.

    python slot/scripts/sibling_smoke.py --task Rebot-PreGrasp-v0
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg).unwrapped
    obs = env.reset()[0]["policy"]
    act = torch.zeros(args.num_envs, env.action_space.shape[1], device=env.device)
    rew_sum = torch.zeros(args.num_envs, device=env.device)
    for _ in range(args.steps):
        obs, rew, _, _, _ = env.step(act)
        obs = obs["policy"]
        rew_sum += rew
    print(f"\n[smoke] {args.task}: OK  obs {tuple(obs.shape)}  act {tuple(act.shape)}  "
          f"mean return over {args.steps} steps {float(rew_sum.mean()):+.4f}")
    print(f"[smoke] reward terms: {list(env.reward_manager.active_terms)}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
    sys.exit(0)
