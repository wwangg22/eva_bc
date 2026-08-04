#!/usr/bin/env python
"""Is the trained policy closed-loop, or has it learned an open-loop clock?

The concern
-----------
`analysis/label_consistency.py` measured the label-noise floor of the demo pools at **0.1 mrad**
-- the recorded action is an essentially deterministic, smooth function of the observation. That
is good news for fitting, but it exposes a specific degenerate solution. Every demo follows the
same phase schedule with the same step counts (cross-demo nearest neighbours land a median of
1-2 timesteps apart), and observation dims 27:34 carry `last_action`. A policy can therefore
integrate its own previous output and replay a trajectory indexed by *time* and by the *initial*
block pose, never looking at where the block is now. On the demo distribution that strategy
scores very well. It is also not a manipulation policy.

The test
--------
At a chosen step during `reach` -- before the fingers close, while the block is still free on
the table -- teleport the block to a **new pose drawn from the same reset distribution**. Then
let the episode run.

* A **clock** policy continues to where the block *was* and closes on nothing.
* A **feedback** policy re-targets and grasps where the block *is*.

The perturbation is drawn from the training reset distribution, so the new pose is always one
the policy has seen at t=0. Nothing out-of-distribution is being asked of it except the timing.

Controls, because a bare number here is not interpretable:

1. `--perturb-step -1` runs the identical harness with no teleport at all. Any drop from this
   baseline is caused by the perturbation and not by the harness.
2. `--resample-only` teleports the block to the pose it *already has* (a no-op write through the
   same code path, same physics flush). If this differs from control 1, the write itself is
   disturbing the episode and the main result is confounded.

Success is `slot_mdp.placed_mask`, the same predicate as everywhere else in this project.

.. code-block:: bash

    python slot/scripts/diag_feedback.py --ckpt runs/bc_armA_seed0/ckpt_final.pt \
        --perturb-step 40 --num-envs 32 --episodes 64
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="Rebot-PrecisionSlot-v0")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--perturb-step", type=int, default=20,
                   help="control step at which to re-randomise the block. The expert's gripper "
                        "channel flips closed at EXACTLY step 40 (measured from the demo "
                        "actions; `lift` begins at 71), so the block is only free to move over "
                        "steps 0-39. 20 is mid-reach and leaves ~20 steps (0.4 s) to "
                        "re-approach; sweep 5/15/25/35 for the reaction-time curve. -1 disables "
                        "the teleport entirely, which is control 1.")
    p.add_argument("--resample-only", action="store_true",
                   help="control 2: write the block's CURRENT pose back through the same code "
                        "path instead of a new one. Isolates the cost of the write itself.")
    p.add_argument("--episode-length-s", type=float, default=12.0)
    p.add_argument("--out", default=None)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    if a.perturb_step >= 40:
        raise SystemExit(
            f"--perturb-step {a.perturb_step}: the expert commands the gripper closed at step "
            f"40, so the block is not free after that. Teleporting it out of a closing gripper "
            f"tests nothing about feedback. Use a step in [0, 39].")
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841

    import gymnasium as gym

    import reBot_RL.tasks  # noqa: F401
    import slot_mdp as mdp
    from isaaclab_tasks.utils import parse_env_cfg
    from slot_act.dataset import OBS_DIM
    from slot_act.eval_act import BatchedACTController, load_checkpoint

    device = "cuda:0"
    policy, stats, cfg = load_checkpoint(Path(a.ckpt), device)
    ctrl = BatchedACTController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)

    env_cfg = parse_env_cfg(a.task, device=device, num_envs=a.num_envs)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.episode_length_s = a.episode_length_s
    env_cfg.seed = a.seed
    env = gym.make(a.task, cfg=env_cfg)
    u = env.unwrapped
    n = a.num_envs
    block = u.scene["block"]

    # The reset distribution, read from the env config rather than restated, so this can never
    # drift away from what the policy was trained on.
    pr = env_cfg.events.reset_block.params["pose_range"]
    print(f"[diag] perturbation drawn from the env's own reset pose_range: {pr}")

    gen = torch.Generator(device="cpu").manual_seed(a.seed + 12345)

    def teleport(ids: torch.Tensor, resample: bool) -> torch.Tensor:
        """Move the given envs' blocks. Returns the (m, 3) x/y/yaw delta actually applied."""
        m = ids.numel()
        st = block.data.root_state_w[ids].clone()
        if resample:
            return torch.zeros(m, 3)  # control 2: write the same pose back, delta is exactly 0
        base = u.scene.env_origins[ids]
        old_xy = st[:, :2] - base[:, :2]
        old_yaw = mdp.yaw_of(st[:, 3:7])
        dx = torch.empty(m).uniform_(*pr["x"], generator=gen).to(u.device)
        dy = torch.empty(m).uniform_(*pr["y"], generator=gen).to(u.device)
        dyaw = torch.empty(m).uniform_(*pr["yaw"], generator=gen).to(u.device)
        # The default spawn is the centre of the range, so a fresh draw is default + delta.
        new_x = u.scene.rigid_objects["block"].data.default_root_state[ids, 0] + dx
        new_y = u.scene.rigid_objects["block"].data.default_root_state[ids, 1] + dy
        st[:, 0] = base[:, 0] + new_x
        st[:, 1] = base[:, 1] + new_y
        # XYZW, matching collect_demos.py and the "pos 3 + quat 4 XYZW" layout documented in
        # slot_act/dataset.py. Writing WXYZ here would silently tip the block onto its side.
        st[:, 3] = 0.0
        st[:, 4] = 0.0
        st[:, 5] = torch.sin(dyaw / 2)
        st[:, 6] = torch.cos(dyaw / 2)
        st[:, 7:] = 0.0
        block.write_root_state_to_sim(st, env_ids=ids)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(u.sim.get_physics_dt())
        moved = torch.stack([new_x - old_xy[:, 0], new_y - old_xy[:, 1],
                             dyaw - old_yaw], dim=-1)
        return moved.cpu()

    obs = env.reset()[0]["policy"]
    assert obs.shape == (n, OBS_DIM)
    step_in_ep = torch.zeros(n, dtype=torch.long, device=u.device)
    ep_index = torch.zeros(n, dtype=torch.long)
    records: list[dict] = []
    moved_mm = torch.zeros(n, 3)

    while len(records) < a.episodes:
        if a.perturb_step >= 0:
            hit = (step_in_ep == a.perturb_step).nonzero(as_tuple=False).squeeze(-1)
            if hit.numel():
                d = teleport(hit, a.resample_only)
                moved_mm[hit.cpu()] = d
                # the queue holds a chunk predicted before the teleport; clearing it is the
                # BEST case for the policy, so not clearing it would confound "cannot react"
                # with "was not asked to react yet".
                ctrl.flush(hit.tolist())
                obs = u.observation_manager.compute()["policy"]

        # sampled pre-step: env.step resets done envs internally, so anything read after it
        # describes the next episode's spawn (see eval_act.py -- this cost a wrong diagnostic
        # table once already)
        placed = mdp.placed_mask(u).all(dim=1)
        pos = mdp.object_pos_local(u, "block")
        obs = obs.to(u.device)
        actions = ctrl.act(obs).to(u.device)
        obs_d, _, terminated, truncated, _ = env.step(actions)
        obs = obs_d["policy"]
        step_in_ep += 1

        done = (terminated | truncated).view(-1).nonzero(as_tuple=False).squeeze(-1)
        # step_in_ep lives on the sim device, ep_index/moved_mm on the CPU. Indexing a CPU
        # tensor with a CUDA index tensor raises; keep both spellings rather than moving
        # everything, so the sim-side update stays on-device.
        done_cpu = done.cpu()
        if done.numel():
            for i in done.tolist():
                records.append({
                    "episode": len(records), "env": i,
                    "episode_index_in_env": int(ep_index[i]),
                    "success": bool(placed[i]),
                    "moved_dx_mm": round(float(moved_mm[i, 0]) * 1000, 2),
                    "moved_dy_mm": round(float(moved_mm[i, 1]) * 1000, 2),
                    "moved_dyaw": round(float(moved_mm[i, 2]), 4),
                    "final_block": [round(float(v), 4) for v in pos[i].cpu()],
                })
            ctrl.reset(done.tolist())
            step_in_ep[done] = 0
            ep_index[done_cpu] += 1
            moved_mm[done_cpu] = 0.0

    records = records[: a.episodes]
    later = [r for r in records if r["episode_index_in_env"] > 0]
    res = {
        "ckpt": a.ckpt, "task": a.task, "seed": a.seed,
        "perturb_step": a.perturb_step, "resample_only": a.resample_only,
        "episodes": len(records),
        "success_rate": sum(r["success"] for r in records) / len(records),
        "success_rate_later": (sum(r["success"] for r in later) / len(later)) if later else None,
        "mean_abs_move_mm": sum(abs(r["moved_dx_mm"]) + abs(r["moved_dy_mm"])
                                for r in records) / (2 * len(records)),
        "per_episode": records,
    }
    out = Path(a.out) if a.out else Path(a.ckpt).parent / (
        f"diag_feedback_p{a.perturb_step}{'_resample' if a.resample_only else ''}.json")
    out.write_text(json.dumps(res, indent=2))
    print(f"[diag] perturb_step={a.perturb_step} resample_only={a.resample_only}  "
          f"success={res['success_rate']:.3f} later={res['success_rate_later']}  "
          f"mean |move| {res['mean_abs_move_mm']:.1f} mm -> {out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
