#!/usr/bin/env python
"""Batched sim evaluation for the vendored ACT policy (PLAN.md sections 2.3 / 4.2).

Rolls a trained ACT checkpoint in Rebot-PickPlace-Play-v1 across N parallel envs with
per-env action queues (receding horizon, temporal ensembling OFF) and the section 4.2
privileged flush trigger: on a detected can discontinuity (teleport-size position jump,
or a fast z drop while the can is above the table) the env's queue is cleared so the
next control step re-predicts from fresh observations.

Env terminations vs the expert runner (expert/run_expert_v1.py): the runner disables
BOTH time_out and object_dropping for its scripted control. Here time_out is KEPT
(episodes must end on the fixed horizon) and object_dropping is DISABLED to match how
the demos were generated (a dropped can does not re-randomize the scene mid-episode;
the fixed horizon ends the episode instead).

Success predicate: the env's own placement predicate, ``mdp.placed_mask(env)`` (both
cans inside the per-env basket footprint and below the rim, env-local frame) — the
exact function behind the expert runner's ``placed()`` check and the env's
Metrics/success_rate logger. Because Isaac Lab's ManagerBasedRLEnv resets done envs
*before* computing the returned observations, the scene state at the exact termination
step is unreadable after ``env.step``; success is therefore sampled one control step
(20 ms) before episode end. On a 10 s / 500-step horizon this is inconsequential.

Run (env_isaaclab6):
    python eval_act.py --ckpt runs/act_nominal/ckpt_final.pt --episodes 64 --num-envs 16

The policy/queue/normalization logic lives in ``BatchedACTController`` (sim-free,
CPU-testable); all Isaac Sim imports happen inside ``main()`` after the AppLauncher.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset import ACTION_DIM, ENV_STATE_SLICE, OBS_DIM, STATE_SLICE
from act.normalize import MeanStdNormalizer

# section 4.2 flush thresholds (privileged, sim-side)
FLUSH_POS_JUMP = 0.03  # m, per-step can position jump (teleport / nudge detector)
FLUSH_Z_DROP = 0.02  # m, per-step z drop while above the table (slip proxy)
FLUSH_Z_ABOVE = 0.05  # "above the table": can root z before the drop (> basket rim,
#                       so a can settling inside the basket never triggers; a normal
#                       release from high above the rim can still fire one benign flush
#                       near the end of its fall — costs one extra re-predict)


class BatchedACTController:
    """Per-env action-queue manager around an ACT policy for batched sim eval.

    ``policy`` needs only ``predict_action_chunk(batch) -> (B, chunk_size, action_dim)``
    (see act/modeling_act.py); normalization happens HERE, outside the policy — this
    LeRobot version keeps normalization in an external processor (act/PROVENANCE.md):
    observations are mean/std-normalized before the forward and the predicted chunk is
    unnormalized before queueing.

    Each env has its own deque of upcoming actions. ``act`` refills every empty queue
    with ONE batched forward over exactly those envs, then pops one action per env.
    ``reset``/``flush`` clear only the targeted envs' queues (episode reset / section
    4.2 discontinuity trigger), forcing a fresh prediction on their next ``act``.
    """

    def __init__(self, policy, stats, n_action_steps: int, chunk_size: int, device,
                 fixed_x0: torch.Tensor | None = None):
        if not (1 <= n_action_steps <= chunk_size):
            raise ValueError(f"n_action_steps={n_action_steps} must be in [1, chunk_size={chunk_size}]")
        self.policy = policy
        self.normalizer = MeanStdNormalizer(stats).to(device)
        self.n_action_steps = n_action_steps
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        # Optional (chunk_size, action_dim) noise shared by every prediction: makes the
        # base chunk a deterministic function of the observation (residual-RL stages,
        # PLAN section 6). None = draw x0 fresh per refill (historical eval behavior).
        self.fixed_x0 = fixed_x0.to(self.device) if fixed_x0 is not None else None
        # Optional per-env steered noise (EXP07): (N, chunk_size, action_dim), set by
        # the steering wrapper each RL window. Takes precedence over fixed_x0.
        self.steer_x0: torch.Tensor | None = None
        # Vectorized queue (no per-env python containers — required for 1000+ envs):
        # _buf (N, n_action_steps, 7) holds each env's committed window, _idx (N,) the
        # position of its NEXT action; _idx == n_action_steps means empty.
        self._buf: torch.Tensor | None = None  # lazily sized on first act()/peek()
        self._idx: torch.Tensor | None = None

    @torch.no_grad()
    def _refill(self, obs_batch: torch.Tensor) -> None:
        """Refill every empty queue with ONE batched forward over exactly those envs
        (ascending env order — preserves the historical forward-batch composition)."""
        n = obs_batch.shape[0]
        if self._buf is None:
            self._buf = torch.zeros(n, self.n_action_steps, ACTION_DIM, device=self.device)
            self._idx = torch.full((n,), self.n_action_steps, dtype=torch.long, device=self.device)
        if self._buf.shape[0] != n:
            raise ValueError(f"obs batch size {n} != controller size {self._buf.shape[0]}")

        empty = (self._idx >= self.n_action_steps).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel():
            sub = obs_batch[empty]
            batch = self.normalizer.normalize(
                {
                    "observation.state": sub[:, STATE_SLICE],
                    "observation.environment_state": sub[:, ENV_STATE_SLICE],
                }
            )
            if self.steer_x0 is not None:
                chunk = self.policy.predict_action_chunk(batch, x0=self.steer_x0[empty])
            elif self.fixed_x0 is not None:
                chunk = self.policy.predict_action_chunk(batch, x0=self.fixed_x0)
            else:
                chunk = self.policy.predict_action_chunk(batch)
            chunk = chunk[:, : self.n_action_steps]
            chunk = self.normalizer.unnormalize("action", chunk)  # (B, n_action_steps, 7)
            self._buf[empty] = chunk
            self._idx[empty] = 0

    @torch.no_grad()
    def act(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """(N, 41) observation batch -> (N, 7) actions, refilling empty queues first."""
        obs_batch = obs_batch.to(self.device, dtype=torch.float32)
        self._refill(obs_batch)
        actions = self._buf[torch.arange(self._buf.shape[0], device=self.device), self._idx]
        self._idx += 1
        return actions

    @torch.no_grad()
    def peek(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refill empties, then return (base_actions (N, 7), phase (N,)) WITHOUT popping.

        ``phase`` is the 0-based index of the returned action inside its execution
        window (0 = fresh chunk, n_action_steps-1 = last committed step). Call
        ``pop()`` afterwards to consume exactly these actions (residual-RL wrapper).
        """
        obs_batch = obs_batch.to(self.device, dtype=torch.float32)
        self._refill(obs_batch)
        base = self._buf[torch.arange(self._buf.shape[0], device=self.device), self._idx]
        return base, self._idx.clone()

    def pop(self) -> torch.Tensor:
        """Consume the actions the last ``peek`` returned (one per env)."""
        actions = self._buf[torch.arange(self._buf.shape[0], device=self.device), self._idx]
        self._idx += 1
        return actions

    def reset(self, env_ids) -> None:
        """Clear the action queues of the given envs (episode reset)."""
        if self._idx is None:
            return
        ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, dtype=torch.long)
        self._idx[ids.to(self.device)] = self.n_action_steps

    def flush(self, env_ids) -> None:
        """Section 4.2 discontinuity flush — same mechanics as reset."""
        self.reset(env_ids)


def load_checkpoint(ckpt_path: Path, device):
    """Load a train_act.py checkpoint -> (policy, stats, config dict).

    Checkpoint format (act/train_act.py save_checkpoint): {step, policy_state_dict,
    normalizer_state_dict, config:{chunk_size, n_action_steps, state_dim, env_state_dim,
    action_dim}}. Normalizer stats are recovered from the buffer names MeanStdNormalizer
    registers ("<key with . -> __>_mean"/"_std"). Checkpoints from act/train_flow.py
    additionally carry config["policy_type"] == "flow" (+ num_inference_steps) and are
    dispatched to FlowMatchingPolicy (same predict_action_chunk contract).
    """
    from types import SimpleNamespace

    from act.modeling_act import ACTPolicy
    from act.train_act import make_config

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg["state_dim"] == STATE_SLICE.stop - STATE_SLICE.start, cfg
    assert cfg["env_state_dim"] == ENV_STATE_SLICE.stop - ENV_STATE_SLICE.start, cfg
    assert cfg["action_dim"] == ACTION_DIM, cfg

    stats = {}
    for name, tensor in ckpt["normalizer_state_dict"].items():
        if name.endswith("_mean"):
            key = name[: -len("_mean")].replace("__", ".")
            stats[key] = {"mean": tensor, "std": ckpt["normalizer_state_dict"][name[: -len("_mean")] + "_std"]}

    if cfg.get("policy_type") == "flow":
        # identical architecture construction to training (act/train_flow.py make_config)
        from act.modeling_flow import FlowMatchingPolicy
        from act.train_flow import make_config as make_flow_config

        config = make_flow_config(
            SimpleNamespace(
                chunk_size=cfg["chunk_size"],
                n_action_steps=cfg["n_action_steps"],
                num_inference_steps=cfg.get("num_inference_steps", 10),
                device=str(device),
            )
        )
        policy = FlowMatchingPolicy(config)
    else:
        # identical architecture construction to training (act/train_act.py make_config)
        config = make_config(
            SimpleNamespace(chunk_size=cfg["chunk_size"], n_action_steps=cfg["n_action_steps"], device=str(device))
        )
        policy = ACTPolicy(config)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device).eval()
    return policy, stats, {**cfg, "step": ckpt.get("step")}


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="train_act.py checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-action-steps", type=int, default=None, help="override ckpt config's n_action_steps")
    parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True,
                        help="section 4.2 discontinuity flush trigger (default on)")
    parser.add_argument("--out", default=None, help="results JSON (default: eval_act_results.json next to ckpt)")
    parser.add_argument("--episode-length-s", type=float, default=30.0,
                        help="episode horizon override; expert demos run up to ~1234 steps (~25 s), "
                             "the task default (500 steps) times out 92%% of demo-length episodes")
    parser.add_argument("--video", action="store_true", help="record an rgb_array video (offscreen render)")
    parser.add_argument("--video-length", type=int, default=3000, help="video length in env steps")
    parser.add_argument("--viewer-eye", type=float, nargs=3, default=None,
                        help="viewer camera eye xyz (close-up framing for failure review)")
    parser.add_argument("--viewer-lookat", type=float, nargs=3, default=None,
                        help="viewer camera lookat xyz")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.video:
        args.enable_cameras = True
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym

    import reBot_RL.tasks  # noqa: F401  (registers Rebot-PickPlace-*)
    import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
    from isaaclab_tasks.utils import parse_env_cfg

    device = "cuda:0"
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out) if args.out else ckpt_path.parent / "eval_act_results.json"

    policy, stats, ckpt_cfg = load_checkpoint(ckpt_path, device)
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else ckpt_cfg["n_action_steps"]
    controller = BatchedACTController(policy, stats, n_action_steps, ckpt_cfg["chunk_size"], device)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    # KEEP time_out (fixed horizon ends episodes); DISABLE object_dropping + its reward
    # hook to match demo generation (see module docstring / run_expert_v1.py Driver).
    assert env_cfg.terminations.time_out is not None
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    if args.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(args.viewer_eye)
    if args.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(args.viewer_lookat)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(ckpt_path.parent / "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    u = env.unwrapped
    n = args.num_envs

    def can_pos_local() -> torch.Tensor:
        """(N, num_cans, 3) env-local can root positions (fixed a/b order — the obs'
        canonical target-first ordering can swap mid-episode and would fake jumps)."""
        return torch.stack([mdp.object_pos_local(u, name) for name in mdp.OBJECT_NAMES], dim=1)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    assert obs.shape == (n, OBS_DIM), obs.shape
    prev_pos = can_pos_local()
    just_reset = torch.ones(n, dtype=torch.bool, device=u.device)  # no flush across resets
    ep_len = torch.zeros(n, dtype=torch.long)
    ep_flushes = torch.zeros(n, dtype=torch.long)
    ep_max_placed = torch.zeros(n, dtype=torch.long)
    ep_final_placed = torch.zeros(n, dtype=torch.long)
    ep_max_can_z = torch.zeros(n, len(mdp.OBJECT_NAMES))
    records: list[dict] = []
    flush_count = 0

    while len(records) < args.episodes:
        pos = can_pos_local()
        if args.flush:
            jump = torch.linalg.norm(pos - prev_pos, dim=-1) > FLUSH_POS_JUMP  # (N, cans)
            z_drop = ((prev_pos[..., 2] - pos[..., 2]) > FLUSH_Z_DROP) & (prev_pos[..., 2] > FLUSH_Z_ABOVE)
            trig = ((jump | z_drop).any(dim=1) & ~just_reset).nonzero(as_tuple=False).squeeze(-1)
            if trig.numel():
                ids = trig.tolist()
                controller.flush(ids)
                flush_count += len(ids)
                ep_flushes[ids] += 1
        prev_pos = pos
        # env's own success predicate, sampled pre-step (see module docstring)
        placed_now = mdp.placed_mask(u)
        last_placed = placed_now.all(dim=1)
        ep_max_placed = torch.maximum(ep_max_placed, placed_now.sum(dim=1).cpu())
        ep_final_placed = placed_now.sum(dim=1).cpu()
        ep_max_can_z = torch.maximum(ep_max_can_z, pos[..., 2].cpu())
        just_reset[:] = False

        actions = controller.act(obs).to(u.device)
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        ep_len += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            bxy = mdp.basket_centers_local(u).cpu()
            for i in done.tolist():
                records.append(
                    {
                        "episode": len(records),
                        "env": i,
                        "length": int(ep_len[i]),
                        "success": bool(last_placed[i]),
                        "flushes": int(ep_flushes[i]),
                        "placed_final": int(ep_final_placed[i]),
                        "placed_max": int(ep_max_placed[i]),
                        "max_can_z": [round(float(z), 3) for z in ep_max_can_z[i]],
                        # last pre-step sample (one control step stale — diagnostics only)
                        "final_can_pos": [[round(float(v), 3) for v in c] for c in pos[i].cpu()],
                        "final_placed_per_can": [bool(x) for x in placed_now[i].cpu()],
                        "basket_xy": [round(float(v), 3) for v in bxy[i]],
                    }
                )
            controller.reset(done.tolist())
            ep_len[done] = 0
            ep_flushes[done] = 0
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
        "flush_count": flush_count,
        "flush_enabled": bool(args.flush),
        "config": {**ckpt_cfg, "n_action_steps_effective": n_action_steps, "num_envs": n},
        "per_episode": records,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"[eval_act] {n_ep} eps: success_rate={result['success_rate']:.3f} "
        f"mean_ep_len={result['mean_ep_len']:.1f} flushes={flush_count} -> {out_path}"
    )

    env.close()
    app.close()


if __name__ == "__main__":
    main()
