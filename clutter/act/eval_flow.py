# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""Batched simulator evaluation of a clutter flow-BC checkpoint, with a failure taxonomy.

Two things this reports, and the second is not optional
-------------------------------------------------------
1. **the success rate**, pooled over held-out spawn seeds, judged against the band registered
   in `HANDOFF.md` §10.1 before any policy existed:

       >= 58 %      better transfer than eva_bc ever achieved -- Gate 2 passes outright
       50 - 58 %    ON TREND; judge on the taxonomy, not the number alone
       43 - 50 %    below trend; suspect the static-observation ambiguity first
       <  43 %      a PORTING DEFECT, not a learning limit. Stop and find it.

2. **the failure taxonomy.** Upstream's EXP06 looked flat-but-fine on the number and it was
   the taxonomy that revealed a symmetric 26/26 churn underneath -- two failure modes trading
   places while the aggregate sat still. A success rate alone cannot distinguish "never
   grasped" from "grasped and toppled", and those want opposite fixes.

The buckets are the env's own predicates, latched per env before its first termination,
because `env.step` auto-resets a terminated env and a toppled scene re-spawns upright -- the
same evidence-erasing trap `run_physics` exists to avoid (`07_STAGE0_RESULTS.md` §7.5):

    success        `mdp.target_at_goal` ever true            <- the headline
    toppled        terminated by `distractor_toppled`
    dropped        terminated by `target_dropped`
    stalled        extracted (z > 90 mm, nothing toppled) but never placed
    no-grasp       ran out of episode without ever extracting

Chunk commitment is the controller's whole job: predict 50, execute the first 15, then
re-predict. eva_bc measured 59.4 / 32.8 / 3.1 / 0 / 0 % at n_action_steps 15 / 8 / 4 / 2 / 1
(EXP02), so `--n-action-steps` exists to reproduce that curve, not to tune.

Usage
-----
    python -u clutter/act/eval_flow.py --ckpt runs/bc_s1/ckpt_final.pt --num_envs 128 \
        --seeds 88000,88001
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/clutter"

parser = argparse.ArgumentParser(description="Batched sim eval of a clutter flow-BC policy.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--seeds", type=str, default="88000,88001",
                    help="held-out spawn batch seeds; NEVER the demo seeds (30000-30007)")
parser.add_argument("--horizon", type=int, default=0,
                    help="env steps per episode; 0 = the env's own episode length")
parser.add_argument("--n-action-steps", type=int, default=None,
                    help="override the checkpoint's n_action_steps (EXP02 replication only)")
parser.add_argument("--policy-seed", type=int, default=None,
                    help="reseed torch AFTER env.reset, decoupling the flow's x0 draw from "
                         "the spawn seed. Isaac Lab's env.reset(seed=) calls torch.manual_seed, "
                         "so without this the whole evaluation is deterministic given the "
                         "spawn seeds -- verified: two runs of the same checkpoint agreed "
                         "episode for episode on all 768. That makes checkpoint comparisons "
                         "exactly paired, and it also means one run samples ONE x0 sequence. "
                         "Vary this to measure the policy's own sampling variance.")
parser.add_argument("--out", type=str, default=f"{_ROOT}/runs/bc_eval.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))          # eva_bc, for the vendored act/ package
sys.path.insert(0, str(_HERE))
from policy_runner import ChunkController, load_checkpoint  # noqa: E402


def rollout(env, ctrl, horizon):
    """One spawn batch. Returns latched per-env outcomes, frozen at each env's first done."""
    e = env.unwrapped
    n, dev = e.num_envs, e.device
    tnames = list(e.termination_manager.active_terms)
    done_at = torch.full((n,), -1, dtype=torch.long, device=dev)
    why = torch.zeros((n, len(tnames)), dtype=torch.bool, device=dev)
    succ = torch.zeros(n, dtype=torch.bool, device=dev)
    extr = torch.zeros(n, dtype=torch.bool, device=dev)
    # Worst planar displacement of ANY non-target block from where it spawned, latched over
    # the episode. The env tracks this quantity (`mdp.distractors_disturbed`) but wires it
    # only to a reward term -- `target_at_goal` asks whether a neighbour TOPPLED, never
    # whether it MOVED. An episode that drags a neighbour 3 cm and sets it down upright is a
    # full success by the benchmark's own predicate. Measured here so it can be reported.
    disturb = torch.zeros(n, device=dev)
    last_goal = torch.zeros(n, dtype=torch.bool, device=dev)

    ctrl.reset()
    obs = e.observation_manager.compute()["policy"]
    for t in range(horizon):
        obs, _, terminated, truncated, _ = env.step(ctrl.act(obs))
        obs = obs["policy"] if isinstance(obs, dict) else obs
        alive = done_at < 0
        # read the predicates BEFORE recording the done, so a step that both places the
        # block and times out still counts as a success
        at_goal = mdp_cl.target_at_goal(e)
        succ |= at_goal & alive
        extr |= mdp_cl.target_extracted(e) & alive
        # `last_goal` is a last-write-wins quantity, so it must exclude the step on which the
        # env dies: Isaac Lab auto-resets a done env INSIDE `env.step`, so everything read
        # after that call describes a freshly re-spawned scene. Every episode ends on a
        # `time_out`, so writing on the terminal step gives **0.0 % for every seed** -- which
        # reads exactly like a finding ("the policy never ends at the goal") and is not one.
        # This is R23 a second time, in a place the first fix did not reach.
        #
        # `succ` and `extr` are latches and are safe as written: a re-spawned target sits at
        # (250, 0) mm, 196 mm from the goal and 35 mm high, so it satisfies neither predicate.
        last_goal = torch.where(alive & ~(terminated | truncated), at_goal, last_goal)
        if hasattr(e, "_clutter_spawn_xy"):
            cur = torch.stack([mdp_cl.common.object_pos_local(e, nm)[:, :2]
                               for nm in mdp_cl.DISTRACTOR_NAMES], dim=1)
            d = (cur - e._clutter_spawn_xy).norm(dim=-1).max(dim=1).values
            disturb = torch.where(alive & ~(terminated | truncated),
                                  torch.maximum(disturb, d), disturb)
        newly = (terminated | truncated) & alive
        if bool(newly.any()):
            done_at[newly] = t
            for k, nm in enumerate(tnames):
                why[:, k] |= newly & e.termination_manager.get_term(nm)
            ctrl.reset(newly.nonzero(as_tuple=False).squeeze(-1))
    idx = {nm: k for k, nm in enumerate(tnames)}
    top = why[:, idx["distractor_toppled"]]
    drop = why[:, idx["target_dropped"]]
    # `success` latches over the whole episode; `success_final` reads the last step only.
    # The expert's collector reports both and they agree exactly (71.4 % / 71.4 %), so the
    # latch costs it nothing -- but a policy that carries a block THROUGH the goal circle and
    # out again would be flattered by the latch, and the comparison would silently stop being
    # like for like. Measured rather than assumed, in both directions.
    return {"success": succ, "success_final": last_goal, "extracted": extr, "toppled": top,
            "disturb": disturb,
            "dropped": drop,
            "stalled": ~succ & ~top & ~drop & extr,
            "no_grasp": ~succ & ~top & ~drop & ~extr,
            "done_at": done_at}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    device = torch.device(args_cli.device if args_cli.device else "cuda")

    policy, stats, cfg = load_checkpoint(args_cli.ckpt, device)
    nas = args_cli.n_action_steps if args_cli.n_action_steps is not None else cfg["n_action_steps"]
    ctrl = ChunkController(policy, stats, nas, cfg["chunk_size"], device)
    horizon = args_cli.horizon or int(e.max_episode_length)
    seeds = [int(s) for s in args_cli.seeds.split(",")]

    print("\n" + "=" * 100)
    print("CLUTTER FLOW-BC EVALUATION")
    print("=" * 100)
    print(f"   ckpt {args_cli.ckpt}  (step {cfg.get('step')})")
    print(f"   chunk {cfg['chunk_size']} / commit {nas}"
          + ("" if nas == cfg["n_action_steps"] else f"  (OVERRIDDEN from {cfg['n_action_steps']})")
          + f", {cfg.get('num_inference_steps', 10)} Euler steps")
    print(f"   {args_cli.num_envs} envs x {len(seeds)} seeds = "
          f"{args_cli.num_envs * len(seeds)} episodes, horizon {horizon} env steps")

    BUCKETS = ("success", "success_final", "toppled", "dropped", "stalled", "no_grasp")
    #: strict success also requires every neighbour to stay within this much of its spawn [m]
    STRICT = (0.002, 0.005, 0.010)
    cells, agg = [], {b: 0 for b in BUCKETS}
    for s in seeds:
        env.reset(seed=s)
        if args_cli.policy_seed is not None:
            torch.manual_seed(args_cli.policy_seed * 100003 + s)
        r = rollout(env, ctrl, horizon)
        c = {"seed": s, **{b: float(r[b].float().mean()) for b in BUCKETS},
             "extracted": float(r["extracted"].float().mean()),
             "succ_mask": r["success"].tolist(),
             "disturb_mm": (r["disturb"] * 1000).tolist(),
             **{f"strict_{int(t * 1000)}mm": float((r["success"] & (r["disturb"] < t))
                                                   .float().mean()) for t in STRICT}}
        cells.append(c)
        for b in BUCKETS:
            agg[b] += int(r[b].sum())
        print(f"      seed {s}: SUCCESS {c['success']:6.1%} (final {c['success_final']:6.1%}) "
              f"| extracted {c['extracted']:6.1%} "
              f"| toppled {c['toppled']:6.1%} | dropped {c['dropped']:6.1%} "
              f"| stalled {c['stalled']:6.1%} | no-grasp {c['no_grasp']:6.1%}")

    N = args_cli.num_envs * len(seeds)
    m = agg["success"] / N
    print("\n   " + "-" * 90)
    print(f"   POOLED SUCCESS {m:.1%}  ({agg['success']}/{N} episodes)   "
          f"last-step-only {agg['success_final'] / N:.1%} "
          f"(delta {(m - agg['success_final'] / N) * 100:+.2f} pts -- the latch's own effect)")
    print("   FAILURE TAXONOMY")
    for b in BUCKETS[2:]:
        print(f"      {b:<10} {agg[b]:5d}  {agg[b] / N:6.1%}")
    # --- how much of that "success" moved a neighbour? -------------------------------------
    dm = np.concatenate([np.asarray(c["disturb_mm"]) for c in cells])
    sm = np.concatenate([np.asarray(c["succ_mask"], dtype=bool) for c in cells])
    print("\n   NEIGHBOUR DISPLACEMENT -- worst non-target block, distance from its spawn")
    print("   `target_at_goal` does not look at this. Among the episodes it calls SUCCESS:")
    print(f"      median {np.median(dm[sm]):6.2f} mm   p90 {np.percentile(dm[sm], 90):6.2f} mm"
          f"   max {dm[sm].max():6.2f} mm")
    print(f"      {'threshold':>10} {'STRICT success':>15} {'lost vs headline':>18}")
    for t in STRICT:
        k = float((sm & (dm < t * 1000)).mean())
        print(f"      {t * 1000:7.0f} mm {k:15.1%} {(k - m) * 100:17.2f} pts")
    band = ("PASS -- better transfer than eva_bc achieved" if m >= 0.58 else
            "ON TREND -- judge on the taxonomy" if m >= 0.50 else
            "BELOW TREND -- suspect the static-observation ambiguity first" if m >= 0.43 else
            "PORTING DEFECT -- stop and find it")
    print(f"\n   vs the band registered in HANDOFF.md 10.1 before any policy existed: {band}")
    print("   expert under env.step on the same protocol: 73.6 % (Gate 2, runs/gate2.json)")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump({"args": vars(args_cli), "ckpt_config": cfg, "n_action_steps": nas,
                   "cells": cells, "pooled": {b: agg[b] / N for b in BUCKETS},
                   "strict": {f"{int(t * 1000)}mm": float((sm & (dm < t * 1000)).mean())
                              for t in STRICT},
                   "disturb_mm_success": {"median": float(np.median(dm[sm])),
                                          "p90": float(np.percentile(dm[sm], 90)),
                                          "max": float(dm[sm].max())},
                   "counts": agg, "n": N, "verdict": band}, f, indent=2)
    print(f"   wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
