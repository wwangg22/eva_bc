#!/usr/bin/env python
"""EXP08 step 2: collect champion rollouts in the vision env (Gate B).

Drives the 91.4% champion (frozen flow base + x0-steering head, deterministic) through
``Rebot-PickPlace-Vision-Play-v1`` under the ladder eval protocol (30 s episodes,
object_dropping off, fixed spawn seed) and records, per step:

    wrist_rgb / workspace_rgb  (T, 90, 160, 3) uint8   student camera obs
    proprio                    (T, 23) float32         student proprio (jp8 + jv8 + act7)
    obs41                      (T, 41) float32         TEACHER-ONLY privileged obs
                                                       (audit + DAgger labels; never a
                                                       student input — EXP08 section 4)
    actions                    (T, 7)  float32         champion actions as executed

One .pt shard per episode under --out-dir. The success rate over recorded episodes IS
the Gate B audit (expect ~91.4%, accept 88-94%).

Frame-sync guard: each env's FIRST episode is discarded (cameras warm up / partial
episode after reset). Post-success tail is truncated at --tail steps (the champion
idles once both cans are placed; full-length idle frames are disk without information).

No-privileged-info contract enforcement: every step asserts that the recorded student
proprio equals the matching slices of the privileged obs ([0:16] and [34:41]) -- the
student stream is byte-derived from the same manager terms, nothing more.

Run (env_isaaclab6, from reBot_ACT/):
    python experiments/exp08_collect.py \
        --ckpt runs/exp03_N3/ckpt_final.pt \
        --steer-ckpt runs/exp07_steer/s1_seed1/nn/exp07_steer.pth \
        --episodes 64 --seed 42 --out-dir data/exp08_vision/seed42
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

STUDENT_PROPRIO_DIM = 23  # obs41[0:16] + obs41[34:41]


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="frozen flow base checkpoint (train_flow.py .pt)")
    parser.add_argument("--steer-ckpt", required=True, help="rl_games steering checkpoint (.pth)")
    parser.add_argument("--steer-cfg", default=str(Path(__file__).parent.parent / "act" / "steer_ppo_cfg.yaml"))
    parser.add_argument("--episodes", type=int, default=64, help="recorded episodes (after first-episode discard)")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Vision-Play-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--tail", type=int, default=100, help="steps kept after both cans are placed")
    parser.add_argument("--out-dir", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
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
    core = SteerCore(controller, mdp, Path(__file__).parent / "exp06_grasp_bit.pt", alpha_x0=1.0, flush=True)
    window = controller.n_action_steps
    steer_act = load_residual_policy(Path(args.steer_ckpt), Path(args.steer_cfg), device,
                                     obs_dim=STEER_OBS_DIM, act_dim=STEER_ACTION_DIM)

    # --- env: EXACT ladder eval protocol ---
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    assert env_cfg.terminations.time_out is not None
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs
    assert u.max_episode_length % window == 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    assert obs.shape == (n, OBS_DIM), obs.shape

    def student_proprio(stu: dict) -> torch.Tensor:
        return torch.cat([stu["joint_pos"], stu["joint_vel"], stu["actions"]], dim=1)

    # per-env step buffers (lists of per-step CPU tensors) + episode bookkeeping
    bufs: list[dict[str, list]] = [
        {"wrist_rgb": [], "workspace_rgb": [], "proprio": [], "obs41": [], "actions": []} for _ in range(n)
    ]
    ep_count = torch.zeros(n, dtype=torch.long)  # episodes finished per env (0 => discard)
    tail_left = torch.full((n,), -1, dtype=torch.long)  # -1 = not yet successful
    recording = torch.ones(n, dtype=torch.bool)  # false once tail exhausted this episode
    saved = 0
    success_count = 0
    step_i = 0

    def record_step(stu: dict, obs41: torch.Tensor, action: torch.Tensor) -> None:
        pro = student_proprio(stu).cpu()
        o41 = obs41.cpu()
        # contract self-audit: the student proprio must be exactly the deployable slices
        assert torch.allclose(pro, torch.cat([o41[:, :16], o41[:, 34:41]], dim=1), atol=1e-5), (
            "student proprio diverged from obs41 slices -- term order changed?"
        )
        for i in range(n):
            if not recording[i]:
                continue
            bufs[i]["wrist_rgb"].append(stu["wrist_rgb"][i].cpu().clone())
            bufs[i]["workspace_rgb"].append(stu["workspace_rgb"][i].cpu().clone())
            bufs[i]["proprio"].append(pro[i].clone())
            bufs[i]["obs41"].append(o41[i].clone())
            bufs[i]["actions"].append(action[i].cpu().clone())

    print(f"[collect] {args.episodes} episodes, task {args.task}, seed {args.seed}, "
          f"{n} envs, tail {args.tail}", flush=True)

    while saved < args.episodes:
        if step_i % window == 0:
            core.set_steer(steer_act(core.build_obs(obs, u)))

        placed_all = mdp.placed_mask(u).all(dim=1).cpu()
        newly = placed_all & (tail_left < 0)
        tail_left[newly] = args.tail
        active_tail = tail_left > 0
        tail_left[active_tail] -= 1
        recording &= ~((tail_left == 0) & placed_all)

        core.flush_check(u)
        action = core.controller.act(obs)

        record_step(obs_dict["student"], obs, action)

        obs_dict, _, terminated, truncated, _ = env.step(action.to(u.device))
        obs = obs_dict["policy"]
        step_i += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            final_placed = placed_all  # success = both placed at episode end
            for i in done.tolist():
                if ep_count[i] > 0 and saved < args.episodes:
                    ep = {k: torch.stack(v) for k, v in bufs[i].items()}
                    ep["wrist_rgb"] = ep["wrist_rgb"].to(torch.uint8)
                    ep["workspace_rgb"] = ep["workspace_rgb"].to(torch.uint8)
                    shard = {
                        **ep,
                        "success": bool(final_placed[i]),
                        "env": i,
                        "seed": args.seed,
                        "task": args.task,
                        "teacher_only_keys": ["obs41"],
                    }
                    torch.save(shard, out_dir / f"ep_{saved:04d}.pt")
                    success_count += int(final_placed[i])
                    saved += 1
                    if saved % 8 == 0:
                        print(f"[collect] {saved}/{args.episodes} saved, "
                              f"success so far {success_count}/{saved}", flush=True)
                ep_count[i] += 1
                for v in bufs[i].values():
                    v.clear()
                tail_left[i] = -1
                recording[i] = True
            core.reset(done.tolist())

    rate = success_count / saved
    meta = {
        "ckpt": args.ckpt,
        "steer_ckpt": args.steer_ckpt,
        "task": args.task,
        "seed": args.seed,
        "episodes": saved,
        "success_rate": rate,
        "tail": args.tail,
        "num_envs": n,
        "episode_length_s": args.episode_length_s,
        "student_proprio_dim": STUDENT_PROPRIO_DIM,
        "flush_count": core.flush_count,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] DONE: {saved} eps, success_rate={rate:.3f} (Gate B window 0.88-0.94) -> {out_dir}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
