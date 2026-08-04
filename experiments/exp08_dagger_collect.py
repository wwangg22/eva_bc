#!/usr/bin/env python
"""EXP08 step 4: champion-DAgger collection (Gate D).

The VISION STUDENT drives the env from its own (non-privileged) observations; the
CHAMPION labels every state the student visits at a chunk-commit boundary. Labels are
full 50-step action chunks computed exactly as the champion would act from that state:

    obs56 = SteerCore.build_obs(obs41, env)       # privileged, teacher-side only
    z     = steering_head(obs56)
    x0    = tanh(z) broadcast over the chunk      # alpha_x0 = 1
    chunk = frozen_base.predict_action_chunk(obs41_batch, x0=x0)   # unnormalized

Labeling happens ONLY at the student's 15-step window boundaries: with the
window-aligned protocol those are the only states where a deployed student ever
predicts, so they ARE the on-policy decision-point distribution (and labeling is 15x
cheaper than every-step). All episodes are kept regardless of student outcome — the
labels are champion-quality either way.

Shard format (one .pt per episode) — chunk-labeled variant of the exp08_collect
format, auto-detected by act/dataset_vision.py via the ``label_chunks`` key:
    wrist_rgb / workspace_rgb  (T', 90, 160, 3) uint8   student obs at boundaries
    proprio                    (T', 23) float32
    label_chunks               (T', 50, 7) float32      champion chunk labels
    success                    bool                     student's own outcome

Run (env_isaaclab6, from reBot_ACT/):
    python experiments/exp08_dagger_collect.py \
        --student-ckpt runs/exp08_bc/v1/ckpt_final.pt \
        --ckpt runs/exp03_N3/ckpt_final.pt \
        --steer-ckpt runs/exp07_steer/s1_seed1/nn/exp07_steer.pth \
        --episodes 64 --seed 42 --out-dir data/exp08_dagger/r1_seed42
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset import ENV_STATE_SLICE, OBS_DIM, STATE_SLICE
from act.eval_act import BatchedACTController, load_checkpoint
from act.eval_flow_vision import VisionController, load_vision_checkpoint
from act.eval_residual import load_residual_policy
from act.steer_core import STEER_ACTION_DIM, STEER_OBS_DIM, SteerCore


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-ckpt", required=True, help="train_flow_vision.py checkpoint (drives)")
    parser.add_argument("--ckpt", required=True, help="frozen flow base checkpoint (labels)")
    parser.add_argument("--steer-ckpt", required=True, help="rl_games steering checkpoint (labels)")
    parser.add_argument("--steer-cfg", default=str(Path(__file__).parent.parent / "act" / "steer_ppo_cfg.yaml"))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Vision-Play-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-label", action="store_true",
                        help="DIAGNOSTIC: skip champion loading + labeling entirely; run the "
                             "student rollout alone and report success (bisecting the "
                             "eval-vs-dagger success discrepancy, see EXP08 log)")
    parser.add_argument("--load-only", action="store_true",
                        help="DIAGNOSTIC: load champion machinery but never call it")
    parser.add_argument("--dummy-load", action="store_true",
                        help="DIAGNOSTIC: no labeling, but burn comparable GPU time in pure "
                             "matmuls at each boundary (isolates renderer contention)")
    parser.add_argument("--features-only", action="store_true",
                        help="DIAGNOSTIC: call build_obs + steering head at boundaries but "
                             "NOT the flow forward (isolates sensor-access side effects)")
    parser.add_argument("--aa-mode", default=None, choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"],
                        help="override renderer antialiasing (DLSS default is TEMPORAL -> "
                             "frame content depends on GPU load; see EXP08 log)")
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

    # --- student (drives; sees ONLY the student obs group) ---
    s_policy, s_stats, s_cfg = load_vision_checkpoint(Path(args.student_ckpt), device)
    student = VisionController(s_policy, s_stats, s_cfg["n_action_steps"], s_cfg["chunk_size"], device)
    window = s_cfg["n_action_steps"]

    # --- champion labeler (teacher-side, privileged; NO controller queue) ---
    labeling = not args.no_label
    if labeling:
        c_policy, c_stats, c_cfg = load_checkpoint(Path(args.ckpt), device)
        chunk_size = c_cfg["chunk_size"]
        c_controller = BatchedACTController(c_policy, c_stats, c_cfg["n_action_steps"], chunk_size, device)
        core = SteerCore(c_controller, mdp, Path(__file__).parent / "exp06_grasp_bit.pt", alpha_x0=1.0, flush=False)
        steer_act = load_residual_policy(Path(args.steer_ckpt), Path(args.steer_cfg), device,
                                         obs_dim=STEER_OBS_DIM, act_dim=STEER_ACTION_DIM)
        if args.load_only or args.features_only:
            labeling = False

    @torch.no_grad()
    def champion_chunk(obs41: torch.Tensor, u) -> torch.Tensor:
        """(N, 41) privileged obs -> (N, chunk_size, 7) champion label chunks."""
        z = steer_act(core.build_obs(obs41, u))
        x0 = torch.tanh(z).unsqueeze(1).expand(-1, chunk_size, -1)
        batch = c_controller.normalizer.normalize(
            {
                "observation.state": obs41[:, STATE_SLICE],
                "observation.environment_state": obs41[:, ENV_STATE_SLICE],
            }
        )
        chunk = c_policy.predict_action_chunk(batch, x0=x0)
        return c_controller.normalizer.unnormalize("action", chunk)

    # --- env: EXACT ladder eval protocol ---
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    assert env_cfg.terminations.time_out is not None
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    if args.aa_mode is not None:
        env_cfg.sim.render.antialiasing_mode = args.aa_mode
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs
    assert u.max_episode_length % window == 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_dict, _ = env.reset()
    assert obs_dict["policy"].shape == (n, OBS_DIM)

    bufs: list[dict[str, list]] = [
        {"wrist_rgb": [], "workspace_rgb": [], "proprio": [], "label_chunks": []} for _ in range(n)
    ]
    ep_count = torch.zeros(n, dtype=torch.long)
    saved = 0
    success_count = 0
    step_i = 0

    print(f"[dagger] {args.episodes} eps, student {args.student_ckpt}, seed {args.seed}", flush=True)

    while saved < args.episodes:
        stu = obs_dict["student"]
        obs41 = obs_dict["policy"]

        if args.dummy_load and step_i % window == 0:  # DIAGNOSTIC: contention without semantics
            m = torch.randn(2048, 2048, device=device)
            for _ in range(12):
                m = m @ m / m.norm()
        if args.features_only and step_i % window == 0:  # DIAGNOSTIC: sensor access, no flow
            steer_act(core.build_obs(obs41, u))
        if labeling and step_i % window == 0:  # student chunk-commit boundary: label + record
            labels = champion_chunk(obs41, u).cpu()
            pro = torch.cat([stu["joint_pos"], stu["joint_vel"], stu["actions"]], dim=1).cpu()
            assert torch.allclose(
                pro, torch.cat([obs41.cpu()[:, :16], obs41.cpu()[:, 34:41]], dim=1), atol=1e-5
            )
            for i in range(n):
                bufs[i]["wrist_rgb"].append(stu["wrist_rgb"][i].cpu().clone())
                bufs[i]["workspace_rgb"].append(stu["workspace_rgb"][i].cpu().clone())
                bufs[i]["proprio"].append(pro[i].clone())
                bufs[i]["label_chunks"].append(labels[i].clone())

        placed_all = mdp.placed_mask(u).all(dim=1).cpu()
        action = student.act(stu).to(u.device)
        obs_dict, _, terminated, truncated, _ = env.step(action)
        step_i += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            for i in done.tolist():
                if not labeling:  # diagnostic mode: count outcomes only, no shards
                    if ep_count[i] > 0 and saved < args.episodes:
                        success_count += int(placed_all[i])
                        saved += 1
                        if saved % 16 == 0:
                            print(f"[dagger-diag] {saved}/{args.episodes}, student success "
                                  f"{success_count}/{saved}", flush=True)
                    ep_count[i] += 1
                    continue
                if ep_count[i] > 0 and saved < args.episodes:
                    ep = {k: torch.stack(v) for k, v in bufs[i].items()}
                    ep["wrist_rgb"] = ep["wrist_rgb"].to(torch.uint8)
                    ep["workspace_rgb"] = ep["workspace_rgb"].to(torch.uint8)
                    torch.save(
                        {**ep, "success": bool(placed_all[i]), "env": i, "seed": args.seed,
                         "task": args.task, "student_ckpt": args.student_ckpt},
                        out_dir / f"ep_{saved:04d}.pt",
                    )
                    success_count += int(placed_all[i])
                    saved += 1
                    if saved % 8 == 0:
                        print(f"[dagger] {saved}/{args.episodes} saved, student success "
                              f"{success_count}/{saved}", flush=True)
                ep_count[i] += 1
                for v in bufs[i].values():
                    v.clear()
            student.reset(done.tolist())

    meta = {
        "student_ckpt": args.student_ckpt,
        "ckpt": args.ckpt,
        "steer_ckpt": args.steer_ckpt,
        "task": args.task,
        "seed": args.seed,
        "episodes": saved,
        "student_success_rate": success_count / saved,
        "num_envs": n,
        "window": window,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[dagger] DONE: {saved} eps, student success {success_count / saved:.3f} -> {out_dir}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
