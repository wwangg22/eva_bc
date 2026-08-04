#!/usr/bin/env python
"""EXP07 — PPO x0-steering on the frozen flow base (rl_games), chunk-level RL.

The frozen ``BatchedACTController`` lives inside ``SteerRlGamesWrapper``; rl_games
sees a 56-D obs / 7-D action env whose step runs one full 15-env-step execution
window: z -> x0 = alpha_x0 * tanh(z) held for the window, reward summed.

Training protocol = LADDER EVAL protocol (pre-registered EXP07 deviation from EXP06):
object_dropping termination + dropping_penalty OFF (they are inseparable — the
penalty is an is_terminated_term) so 30 s episodes are exactly 100 windows and every
reset lands on a window boundary. Zero train/eval mismatch.

Run (env_isaaclab6, ONE GPU job at a time):
    python train_steer.py --ckpt ../runs/exp03_N3/ckpt_final.pt \
        --run-name s1_seed1 --seed 1 --num_envs 2048 --max_iterations 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", required=True, help="frozen base checkpoint (train_flow.py .pt)")
parser.add_argument("--run-name", required=True, help="log dir name under runs/exp07_steer/")
parser.add_argument("--alpha-x0", type=float, default=1.0, help="x0 bound: x0 = alpha_x0 * tanh(z)")
parser.add_argument("--num_envs", type=int, default=2048)
# -Play-v1: the SAME task variant the demos and every ladder eval used.
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--episode-length-s", type=float, default=12.0)  # slot task native
parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--checkpoint", default=None, help="rl_games checkpoint to resume from")
# EXP_STEER: train the steerer under the condition the FROZEN BASE fails at. Magnitudes are
# fractions of each channel's own training std, identical in meaning to eval_act.py's flags --
# --action-noise 0.02 is the perturbation that took the base from 0.979 to 0.146.
parser.add_argument("--obs-noise", type=float, default=0.0, metavar="FRAC")
parser.add_argument("--action-noise", type=float, default=0.0, metavar="FRAC")
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

import reBot_RL.tasks  # noqa: F401  (registers Rebot-PrecisionSlot-*)
import slot_mdp as mdp  # the act/ mdp surface for the slot task; see slot/slot_mdp.py
from isaaclab_rl.rl_games import RlGamesGpuEnv
from isaaclab_tasks.utils import parse_env_cfg

from slot_act.eval_act import BatchedACTController, load_checkpoint
from slot_act.noise import noise_sigmas
from slot_act.steer_core import SteerCore
from slot_act.steer_wrapper import SteerRlGamesWrapper


def main() -> None:
    device = "cuda:0"
    here = Path(__file__).resolve().parent

    policy, stats, ckpt_cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(
        policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device
    )
    core = SteerCore(controller, mdp, alpha_x0=args.alpha_x0, flush=args.flush)

    # training env = eval protocol (see module docstring): drop termination + penalty
    # OFF, 30 s horizon -> exactly 100 windows/episode, resets on window boundaries.
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    assert env_cfg.terminations.time_out is not None
    # Slot task term names differ from pick-place's, and nulling them is a DECISION, not a
    # rename: a dropped or toppled block must FAIL the episode, not silently re-randomise the
    # scene and let the policy be scored on a second attempt.
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env = gymnasium.make(args.task, cfg=env_cfg)
    assert env.unwrapped.max_episode_length % controller.n_action_steps == 0, (
        f"episode length {env.unwrapped.max_episode_length} steps must be a multiple "
        f"of the {controller.n_action_steps}-step window"
    )

    agent_cfg = yaml.safe_load((here / "steer_ppo_cfg.yaml").read_text())
    agent_cfg["params"]["seed"] = args.seed
    if args.max_iterations is not None:
        agent_cfg["params"]["config"]["max_epochs"] = args.max_iterations
    # 4 minibatches per batch regardless of env count (teacher-config proportions)
    batch = agent_cfg["params"]["config"]["horizon_length"] * args.num_envs
    agent_cfg["params"]["config"]["minibatch_size"] = batch // 4
    if args.checkpoint is not None:
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = args.checkpoint

    log_dir = here.parent / "runs" / "exp07_steer"
    agent_cfg["params"]["config"]["train_dir"] = str(log_dir)
    agent_cfg["params"]["config"]["full_experiment_name"] = args.run_name
    (log_dir / args.run_name).mkdir(parents=True, exist_ok=True)
    (log_dir / args.run_name / "run_config.yaml").write_text(yaml.safe_dump({
        "base_ckpt": args.ckpt, "alpha_x0": args.alpha_x0,
        "num_envs": args.num_envs, "seed": args.seed, "task": args.task,
        "episode_length_s": args.episode_length_s, "flush": bool(args.flush),
        # provenance: a run trained under noise must not be indistinguishable from a clean one
        "obs_noise": args.obs_noise, "action_noise": args.action_noise,
        "agent": agent_cfg,
    }, sort_keys=False))

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", 100.0)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", 1.0)
    # Per-channel noise scales from the base checkpoint's own normaliser, via the SAME helper
    # eval_act.py uses, so --action-noise 0.02 here is the perturbation that scored 0.146 there.
    obs_sigma, act_sigma = noise_sigmas(stats, device, args.obs_noise, args.action_noise,
                                        tag="train_steer")

    wrapped = SteerRlGamesWrapper(env, rl_device, clip_obs, clip_actions, core,
                                  obs_noise=args.obs_noise, obs_sigma=obs_sigma,
                                  action_noise=args.action_noise, act_sigma=act_sigma)

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
