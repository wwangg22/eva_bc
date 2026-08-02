#!/usr/bin/env python
"""EXP06 — PPO residual on the frozen flow base (rl_games), PLAN section 6.

The frozen ``BatchedACTController`` (deterministic base: fixed shared x0) lives inside
``ResidualRlGamesWrapper``; rl_games sees a 64-D obs / 6-D action env whose step
blends ``arm = base + alpha * tanh(res)`` (grip passes through from the base).
Training env keeps the task's own terminations/rewards (object_dropping ON, placed
+60 / drop -30) plus a small residual-magnitude penalty; episode horizon 30 s.

Run (env_isaaclab6, ONE GPU job at a time):
    python train_residual.py --ckpt ../runs/exp03_N3/ckpt_final.pt \
        --run-name r1_seed1 --seed 1 --num_envs 128 --max_iterations 3000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", required=True, help="frozen base checkpoint (train_flow.py .pt)")
parser.add_argument("--run-name", required=True, help="log dir name under runs/exp06_residual/")
parser.add_argument("--alpha", type=float, default=0.1, help="residual bound (env action units)")
parser.add_argument("--x0-seed", type=int, default=7,
                    help="fixed-x0 noise seed for the deterministic base (-1 = zeros); "
                         "MUST match the seed used when evaluating this run")
parser.add_argument("--res-penalty", type=float, default=0.01,
                    help="reward penalty coefficient on ||applied residual||^2 (pre reward_shaper)")
parser.add_argument("--num_envs", type=int, default=128)
# -Play-v1: the SAME task variant the demos and every ladder eval used — the residual
# must train on the base policy's own spawn distribution.
parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--episode-length-s", type=float, default=30.0)
parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--checkpoint", default=None, help="rl_games checkpoint to resume from")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium
import yaml
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401  (registers Rebot-PickPlace-*)
import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
from isaaclab_rl.rl_games import RlGamesGpuEnv
from isaaclab_tasks.utils import parse_env_cfg

from act.eval_act import BatchedACTController, load_checkpoint
from act.eval_residual import make_fixed_x0
from act.residual_core import ResidualCore
from act.residual_wrapper import ResidualRlGamesWrapper


def main() -> None:
    device = "cuda:0"
    here = Path(__file__).resolve().parent

    policy, stats, ckpt_cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(
        policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device,
        fixed_x0=make_fixed_x0(ckpt_cfg["chunk_size"], 7, args.x0_seed),
    )
    core = ResidualCore(controller, mdp, here.parent / "experiments" / "exp06_grasp_bit.pt",
                        alpha=args.alpha, flush=args.flush)

    # training env: task defaults KEPT (object_dropping termination + penalty stay ON),
    # longer horizon so full two-can episodes fit (matches demo/eval horizon).
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env = gymnasium.make(args.task, cfg=env_cfg)

    agent_cfg = yaml.safe_load((here / "residual_ppo_cfg.yaml").read_text())
    agent_cfg["params"]["seed"] = args.seed
    if args.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args.max_iterations
    # 4 minibatches per batch regardless of env count (teacher-config proportions)
    batch = agent_cfg["params"]["config"]["horizon_length"] * args.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = batch // 4
    if args.checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args.checkpoint

    log_dir = here.parent / "runs" / "exp06_residual"
    agent_cfg["params"]["config"]["train_dir"] = str(log_dir)
    agent_cfg["params"]["config"]["full_experiment_name"] = args.run_name
    (log_dir / args.run_name).mkdir(parents=True, exist_ok=True)
    (log_dir / args.run_name / "run_config.yaml").write_text(yaml.safe_dump({
        "base_ckpt": args.ckpt, "alpha": args.alpha, "res_penalty": args.res_penalty,
        "x0_seed": args.x0_seed,
        "num_envs": args.num_envs, "seed": args.seed, "task": args.task,
        "episode_length_s": args.episode_length_s, "flush": bool(args.flush),
        "agent": agent_cfg,
    }, sort_keys=False))

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 100.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 100.0)
    wrapped = ResidualRlGamesWrapper(env, rl_device, clip_obs, clip_actions, core, args.res_penalty)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: wrapped})
    agent_cfg["params"]["config"]["num_actors"] = wrapped.unwrapped.num_envs

    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    if args.checkpoint is not None:
        runner.run({"train": True, "play": False, "checkpoint": args.checkpoint})
    else:
        runner.run({"train": True, "play": False})

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
