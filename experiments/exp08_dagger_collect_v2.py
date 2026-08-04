#!/usr/bin/env python
"""EXP08 step 4 v2: champion-DAgger collection with POST-HOC labeling (Gate D).

v1 (exp08_dagger_collect.py) labeled in-loop and collapsed the student 80%->40%:
under the DLSS temporal renderer, frame CONTENT depends on GPU load between steps, so
ANY optional GPU work inside the driving loop degrades the pixels the student sees
(proven by pure-matmul bisection; EXP08 running log 2026-08-03).

v2 keeps the student-driving loop byte-equivalent to the eval loop: at each 15-step
chunk-commit boundary it records ONLY buffer reads + small device clones (student
images/proprio, obs41, ee pose, can poses, placed mask, basket centers). The champion
labels (steering z from obs56 -> x0 = tanh(z) -> frozen base flow chunks) are computed
AFTER the rollout ends, when frames no longer matter. The steering obs56 feature tail
is rebuilt from the recorded raw tensors by ``obs56_from_raw`` — a pure-tensor mirror
of ResidualCore.task_features, validated against the live build with
``--validate-features``.

Built-in audit: the student's success rate while driving is reported; it must match
the clean no-label baseline (~80% for the v1 student on episodes 2+; a collapse means
the loop is perturbing frames again).

Shard format: identical to v1 (label_chunks variant, act/dataset_vision.py).

Run (env_isaaclab6, from reBot_ACT/):
    python experiments/exp08_dagger_collect_v2.py \
        --student-ckpt runs/exp08_bc/v1/ckpt_final.pt \
        --ckpt runs/exp03_N3/ckpt_final.pt \
        --steer-ckpt runs/exp07_steer/s1_seed1/nn/exp07_steer.pth \
        --episodes 64 --seed 42 --out-dir data/exp08_dagger/r1v2_seed42
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
from act.residual_core import CAN_REST_Z_IN_BASKET, FINGER_FEATURE_DIMS
from act.steer_core import STEER_ACTION_DIM, STEER_OBS_DIM, SteerCore


def obs56_from_raw(core, obs41, ee_w, ee_q_wxyz, pos_a, quat_a, pos_b, quat_b,
                   placed, centers, env_origins) -> torch.Tensor:
    """Pure-tensor mirror of ResidualCore.task_features (world-frame raw inputs).

    All quats wxyz (Isaac buffer convention); the obs56 tail wants xyzw.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    d_a = torch.linalg.norm(pos_a - ee_w, dim=1)
    d_b = torch.linalg.norm(pos_b - ee_w, dim=1)
    d_a = torch.where(placed[:, 0], torch.full_like(d_a, torch.inf), d_a)
    d_b = torch.where(placed[:, 1], torch.full_like(d_b, torch.inf), d_b)
    target_is_b = (d_b < d_a).unsqueeze(1)
    tpos = torch.where(target_is_b, pos_b, pos_a)
    tquat = torch.where(target_is_b, quat_b, quat_a)

    rel_pos, rel_quat = subtract_frame_transforms(ee_w, ee_q_wxyz, tpos, tquat)
    rel_quat_xyzw = rel_quat[:, [1, 2, 3, 0]]

    tpos_local = tpos - env_origins
    basket_delta = torch.stack(
        [
            centers[:, 0] - tpos_local[:, 0],
            centers[:, 1] - tpos_local[:, 1],
            torch.full_like(tpos_local[:, 2], CAN_REST_Z_IN_BASKET) - tpos_local[:, 2],
        ],
        dim=1,
    )
    feats = torch.cat(
        [obs41[:, FINGER_FEATURE_DIMS], core.grasp_bit(obs41).unsqueeze(1), rel_pos, rel_quat_xyzw, basket_delta],
        dim=1,
    )
    return torch.cat([obs41, feats], dim=1)


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--ckpt", required=True, help="frozen flow base checkpoint (labels)")
    parser.add_argument("--steer-ckpt", required=True)
    parser.add_argument("--steer-cfg", default=str(Path(__file__).parent.parent / "act" / "steer_ppo_cfg.yaml"))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Vision-Play-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--validate-features", action="store_true",
                        help="VALIDATION: at each boundary also run the live core.build_obs and "
                             "compare with obs56_from_raw (perturbs frames -- discard the data)")
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

    s_policy, s_stats, s_cfg = load_vision_checkpoint(Path(args.student_ckpt), device)
    student = VisionController(s_policy, s_stats, s_cfg["n_action_steps"], s_cfg["chunk_size"], device)
    window = s_cfg["n_action_steps"]

    c_policy, c_stats, c_cfg = load_checkpoint(Path(args.ckpt), device)
    chunk_size = c_cfg["chunk_size"]
    c_controller = BatchedACTController(c_policy, c_stats, c_cfg["n_action_steps"], chunk_size, device)
    core = SteerCore(c_controller, mdp, Path(__file__).parent / "exp06_grasp_bit.pt", alpha_x0=1.0, flush=False)
    steer_act = load_residual_policy(Path(args.steer_ckpt), Path(args.steer_cfg), device,
                                     obs_dim=STEER_OBS_DIM, act_dim=STEER_ACTION_DIM)

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
    assert obs_dict["policy"].shape == (n, OBS_DIM)
    env_origins = u.scene.env_origins.clone()

    # raw boundary records kept ON DEVICE (small: ~90 KB/env/boundary); per-env lists
    RAW_KEYS = ("wrist_rgb", "workspace_rgb", "obs41", "ee_w", "ee_q", "pos_a", "quat_a",
                "pos_b", "quat_b", "placed", "centers")
    bufs: list[dict[str, list]] = [{k: [] for k in RAW_KEYS} for _ in range(n)]
    pending: list[dict] = []  # finished episodes awaiting post-hoc labeling
    ep_count = torch.zeros(n, dtype=torch.long)
    saved = 0
    success_count = 0
    step_i = 0
    val_max_diff = 0.0

    print(f"[dagger-v2] {args.episodes} eps, post-hoc labeling, seed {args.seed}", flush=True)

    while saved < args.episodes:
        stu = obs_dict["student"]
        obs41 = obs_dict["policy"]

        if step_i % window == 0:  # boundary: RECORD ONLY (buffer reads + clones)
            ee = u.scene["ee_frame"].data
            ee_w = ee.target_pos_w.torch[..., 0, :]
            ee_q = ee.target_quat_w.torch[..., 0, :]  # wxyz
            obj_a, obj_b = u.scene["object_a"].data, u.scene["object_b"].data
            placed = mdp.placed_mask(u)
            centers = mdp.basket_centers_local(u)
            snap = {
                "wrist_rgb": stu["wrist_rgb"].clone(),
                "workspace_rgb": stu["workspace_rgb"].clone(),
                "obs41": obs41.clone(),
                "ee_w": ee_w.clone(), "ee_q": ee_q.clone(),
                "pos_a": obj_a.root_pos_w.torch.clone(), "quat_a": obj_a.root_quat_w.torch.clone(),
                "pos_b": obj_b.root_pos_w.torch.clone(), "quat_b": obj_b.root_quat_w.torch.clone(),
                "placed": placed.clone(), "centers": centers.clone(),
            }
            for i in range(n):
                for k in RAW_KEYS:
                    bufs[i][k].append(snap[k][i])
            if args.validate_features:
                live = core.build_obs(obs41, u)
                rebuilt = obs56_from_raw(core, obs41, ee_w, ee_q, snap["pos_a"], snap["quat_a"],
                                         snap["pos_b"], snap["quat_b"], placed, centers, env_origins)
                val_max_diff = max(val_max_diff, (live - rebuilt).abs().max().item())

        placed_all = mdp.placed_mask(u).all(dim=1).cpu()
        action = student.act(stu).to(u.device)
        obs_dict, _, terminated, truncated, _ = env.step(action)
        step_i += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            for i in done.tolist():
                if ep_count[i] > 0 and saved < args.episodes:
                    pending.append(
                        {
                            **{k: torch.stack(v) for k, v in bufs[i].items()},
                            "success": bool(placed_all[i]),
                            "env": i,
                        }
                    )
                    success_count += int(placed_all[i])
                    saved += 1
                    if saved % 16 == 0:
                        print(f"[dagger-v2] {saved}/{args.episodes} rolled, student success "
                              f"{success_count}/{saved}", flush=True)
                ep_count[i] += 1
                for v in bufs[i].values():
                    v.clear()
            student.reset(done.tolist())

    student_rate = success_count / saved
    print(f"[dagger-v2] rollouts done: student success {student_rate:.3f} "
          f"(audit: expect ~0.80 for the v1 student; collapse => loop perturbs frames)", flush=True)

    # ---- POST-HOC labeling: sim no longer stepping, GPU work is free ----
    @torch.no_grad()
    def label_episode(ep: dict) -> torch.Tensor:
        obs56 = obs56_from_raw(core, ep["obs41"], ep["ee_w"], ep["ee_q"], ep["pos_a"], ep["quat_a"],
                               ep["pos_b"], ep["quat_b"], ep["placed"], ep["centers"],
                               env_origins[ep["env"]].unsqueeze(0))
        z = steer_act(obs56)
        x0 = torch.tanh(z).unsqueeze(1).expand(-1, chunk_size, -1)
        batch = c_controller.normalizer.normalize(
            {
                "observation.state": ep["obs41"][:, STATE_SLICE],
                "observation.environment_state": ep["obs41"][:, ENV_STATE_SLICE],
            }
        )
        chunk = c_policy.predict_action_chunk(batch, x0=x0)
        return c_controller.normalizer.unnormalize("action", chunk)

    for idx, ep in enumerate(pending):
        labels = label_episode(ep)
        proprio = torch.cat([ep["obs41"][:, :16], ep["obs41"][:, 34:41]], dim=1)
        torch.save(
            {
                "wrist_rgb": ep["wrist_rgb"].to(torch.uint8).cpu(),
                "workspace_rgb": ep["workspace_rgb"].to(torch.uint8).cpu(),
                "proprio": proprio.cpu(),
                "label_chunks": labels.cpu(),
                "success": ep["success"],
                "env": ep["env"],
                "seed": args.seed,
                "task": args.task,
                "student_ckpt": args.student_ckpt,
            },
            out_dir / f"ep_{idx:04d}.pt",
        )
    print(f"[dagger-v2] labeled + saved {len(pending)} shards", flush=True)

    meta = {
        "student_ckpt": args.student_ckpt,
        "ckpt": args.ckpt,
        "steer_ckpt": args.steer_ckpt,
        "task": args.task,
        "seed": args.seed,
        "episodes": saved,
        "student_success_rate": student_rate,
        "num_envs": n,
        "window": window,
        "labeling": "post-hoc",
    }
    if args.validate_features:
        meta["feature_validation_max_abs_diff"] = val_max_diff
        print(f"[dagger-v2] feature validation max abs diff: {val_max_diff:.2e}")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[dagger-v2] DONE -> {out_dir}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
