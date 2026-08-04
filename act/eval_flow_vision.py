#!/usr/bin/env python
"""Batched sim evaluation for the vision flow-BC student (EXP08 Gate C).

Protocol identical to the ladder evals (act/eval_act.py): 30 s episodes, time_out
kept, object_dropping disabled, fixed spawn seed, success = mdp.placed_mask sampled
one control step before episode end.

Two deliberate deltas from eval_act.py, both from the EXP08 section 4
no-privileged-info contract:

* The policy consumes ONLY the env's ``student`` obs group (two cameras + 23-D
  proprio). The privileged 41-D group is read exclusively for METRICS (success
  predicate, can heights) -- measurement is allowed to be privileged, policy input
  is not.
* The section 4.2 discontinuity flush is OFF by default: it clears action queues from
  privileged sim-side can positions, which no deployed stack has. ``--flush`` re-enables
  it for diagnostics only. NOTE for comparisons: the 55.5 / 64.1 / 91.4 state anchors
  were measured WITH flush; flush-free student numbers carry a small honest handicap.

Run (env_isaaclab6):
    python act/eval_flow_vision.py --ckpt runs/exp08_bc/v1/ckpt_final.pt \
        --episodes 64 --seed 42 --out runs/exp08_bc/v1/eval_seed42.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset_vision import ACTION_DIM, STUDENT_STATE_DIM
from act.eval_act import FLUSH_POS_JUMP, FLUSH_Z_ABOVE, FLUSH_Z_DROP
from act.normalize import MeanStdNormalizer


class VisionController:
    """Per-env action-queue manager around the vision flow policy (BatchedACTController
    with the batch built from the student obs group instead of obs41 slices)."""

    def __init__(self, policy, stats, n_action_steps: int, chunk_size: int, device):
        self.policy = policy
        self.normalizer = MeanStdNormalizer(stats).to(device)
        self.n_action_steps = n_action_steps
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self._buf: torch.Tensor | None = None
        self._idx: torch.Tensor | None = None

    @staticmethod
    def build_batch(stu: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Student obs group (env dict) -> policy batch. The ONLY inputs are the two
        cameras and the 23-D proprio; nothing else exists in the ``student`` group."""
        return {
            "observation.state": torch.cat([stu["joint_pos"], stu["joint_vel"], stu["actions"]], dim=1).float(),
            "observation.images.wrist": stu["wrist_rgb"].permute(0, 3, 1, 2).float() / 255.0,
            "observation.images.workspace": stu["workspace_rgb"].permute(0, 3, 1, 2).float() / 255.0,
        }

    @torch.no_grad()
    def act(self, stu: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_full = {k: v.to(self.device) for k, v in self.build_batch(stu).items()}
        n = batch_full["observation.state"].shape[0]
        if self._buf is None:
            self._buf = torch.zeros(n, self.n_action_steps, ACTION_DIM, device=self.device)
            self._idx = torch.full((n,), self.n_action_steps, dtype=torch.long, device=self.device)

        empty = (self._idx >= self.n_action_steps).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel():
            sub = self.normalizer.normalize({k: v[empty] for k, v in batch_full.items()})
            chunk = self.policy.predict_action_chunk(sub)[:, : self.n_action_steps]
            self._buf[empty] = self.normalizer.unnormalize("action", chunk)
            self._idx[empty] = 0

        actions = self._buf[torch.arange(n, device=self.device), self._idx]
        self._idx += 1
        return actions

    def reset(self, env_ids) -> None:
        if self._idx is None:
            return
        ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, dtype=torch.long)
        self._idx[ids.to(self.device)] = self.n_action_steps

    flush = reset


def load_vision_checkpoint(ckpt_path: Path, device):
    """Load a train_flow_vision.py checkpoint -> (policy, stats, config dict)."""
    from act.modeling_flow_vision import FlowMatchingVisionPolicy
    from act.train_flow_vision import make_config

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg["policy_type"] == "flow_vision", cfg
    assert cfg["state_dim"] == STUDENT_STATE_DIM and cfg["action_dim"] == ACTION_DIM, cfg

    stats = {}
    for name, tensor in ckpt["normalizer_state_dict"].items():
        if name.endswith("_mean"):
            key = name[: -len("_mean")].replace("__", ".")
            stats[key] = {"mean": tensor, "std": ckpt["normalizer_state_dict"][name[: -len("_mean")] + "_std"]}

    config = make_config(
        SimpleNamespace(
            chunk_size=cfg["chunk_size"],
            n_action_steps=cfg["n_action_steps"],
            num_inference_steps=cfg.get("num_inference_steps", 10),
            device=str(device),
        )
    )
    policy = FlowMatchingVisionPolicy(config)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device).eval()
    return policy, stats, {**cfg, "step": ckpt.get("step")}


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="train_flow_vision.py checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Vision-Play-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush", action="store_true",
                        help="privileged discontinuity flush (DIAGNOSTIC ONLY; default off "
                             "for the student -- see module docstring)")
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--burst-every", type=int, default=0,
                        help="EXP09 S2 DIAGNOSTIC: every N env steps run a dummy GPU burst "
                             "(PPO-update signature) to measure DLSS frame perturbation. 0 = off")
    parser.add_argument("--burst-iters", type=int, default=200,
                        help="matmul iterations per burst (200 x 2048^2 ~ 1-2 s, one PPO update)")
    parser.add_argument("--out", required=True, help="results JSON path")
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
    ckpt_path = Path(args.ckpt)
    policy, stats, ckpt_cfg = load_vision_checkpoint(ckpt_path, device)
    controller = VisionController(policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    assert env_cfg.terminations.time_out is not None
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs

    def can_pos_local() -> torch.Tensor:
        return torch.stack([mdp.object_pos_local(u, name) for name in mdp.OBJECT_NAMES], dim=1)

    obs_dict, _ = env.reset()
    prev_pos = can_pos_local()
    just_reset = torch.ones(n, dtype=torch.bool, device=u.device)
    ep_len = torch.zeros(n, dtype=torch.long)
    ep_max_placed = torch.zeros(n, dtype=torch.long)
    ep_final_placed = torch.zeros(n, dtype=torch.long)
    ep_max_can_z = torch.zeros(n, len(mdp.OBJECT_NAMES))
    records: list[dict] = []
    flush_count = 0

    while len(records) < args.episodes:
        pos = can_pos_local()
        if args.flush:
            jump = torch.linalg.norm(pos - prev_pos, dim=-1) > FLUSH_POS_JUMP
            z_drop = ((prev_pos[..., 2] - pos[..., 2]) > FLUSH_Z_DROP) & (prev_pos[..., 2] > FLUSH_Z_ABOVE)
            trig = ((jump | z_drop).any(dim=1) & ~just_reset).nonzero(as_tuple=False).squeeze(-1)
            if trig.numel():
                controller.flush(trig.tolist())
                flush_count += len(trig)
        prev_pos = pos
        placed_now = mdp.placed_mask(u)
        last_placed = placed_now.all(dim=1)
        ep_max_placed = torch.maximum(ep_max_placed, placed_now.sum(dim=1).cpu())
        ep_final_placed = placed_now.sum(dim=1).cpu()
        ep_max_can_z = torch.maximum(ep_max_can_z, pos[..., 2].cpu())
        just_reset[:] = False

        actions = controller.act(obs_dict["student"]).to(u.device)
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        ep_len += 1
        if args.burst_every and int(ep_len.sum()) // n % args.burst_every == 0:
            m = torch.randn(2048, 2048, device=device)
            for _ in range(args.burst_iters):
                m = m @ m * 1e-3
            torch.cuda.synchronize()

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
                    }
                )
            controller.reset(done.tolist())
            ep_len[done] = 0
            ep_max_placed[done] = 0
            ep_final_placed[done] = 0
            ep_max_can_z[done] = 0.0
            just_reset[done.to(u.device)] = True

    records = records[: args.episodes]
    n_ep = len(records)
    result = {
        "ckpt": str(ckpt_path),
        "task": args.task,
        "seed": args.seed,
        "episodes": n_ep,
        "success_rate": sum(r["success"] for r in records) / n_ep,
        "mean_ep_len": sum(r["length"] for r in records) / n_ep,
        "flush_enabled": bool(args.flush),
        "flush_count": flush_count,
        "config": {**ckpt_cfg, "num_envs": n},
        "per_episode": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[eval_flow_vision] {n_ep} eps: success_rate={result['success_rate']:.3f} -> {args.out}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
