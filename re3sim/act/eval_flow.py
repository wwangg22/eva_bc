# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Batched simulator evaluation of a workstation flow-BC checkpoint, with a failure taxonomy.

Two things this reports, and the second is not optional
-------------------------------------------------------
1. **the success rate**, pooled over held-out spawn seeds, judged against the *expert's own*
   rate on the same protocol rather than an absolute number. A BC policy that reaches ~80 % of
   its demonstrator on a state observation is a healthy port; the absolute figure moves with
   the expert and says nothing on its own.

2. **the failure taxonomy.** Aggregate success hides real change: upstream's EXP06 looked
   flat-but-fine on the number while two failure modes traded places 26-for-26 underneath. A
   success rate cannot distinguish "never closed on the cube" from "carried it and missed the
   box", and those want opposite fixes.

The buckets are latched per env **before** its first termination, because `env.step`
auto-resets a terminated env and everything read after that call describes a freshly
re-spawned scene:

    success     `mdp.placed_mask` ever true                   <- the headline
    lifted      cube ever above LIFT_MIN                       (did the grasp take?)
    over_box    `mdp.over_box` ever true                       (did the carry arrive?)
    dropped     terminated by `target_dropped`
    near_miss   lifted and over the box, but never placed      <- release/settle problem
    no_lift     never lifted at all                            <- grasp problem

`clutter_mm` is reported separately: the success predicate deliberately does not ask whether
the arm shoved the tape measure across the desk on its way, and an episode that does is not
the same quality of success as one that does not.

Chunk commitment is the controller's whole job: predict 50, execute the first 15, re-predict.
eva_bc measured 59.4 / 32.8 / 3.1 / 0 / 0 % at n_action_steps 15 / 8 / 4 / 2 / 1 (EXP02), so
`--n-action-steps` exists to reproduce that curve, not to tune.

Usage
-----
    python -u re3sim/act/eval_flow.py --ckpt re3sim/runs/bc_s1/ckpt_final.pt \
        --num_envs 128 --seeds 88000,88001
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/re3sim"

parser = argparse.ArgumentParser(description="Batched sim eval of a workstation flow-BC policy.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--seeds", type=str, default="88000,88001",
                    help="held-out spawn seeds; NEVER the demo-collection seeds")
parser.add_argument("--horizon", type=int, default=0,
                    help="env steps per episode; 0 = the env's own episode length")
parser.add_argument("--n-action-steps", type=int, default=None,
                    help="override the checkpoint's n_action_steps (EXP02 replication only)")
parser.add_argument("--policy-seed", type=int, default=None,
                    help="reseed torch AFTER env.reset, decoupling the flow's x0 draw from the "
                         "spawn seed. Isaac Lab's env.reset(seed=) calls torch.manual_seed, so "
                         "without this an evaluation is deterministic given the spawn seeds -- "
                         "which makes checkpoint comparisons exactly paired, and also means one "
                         "run samples ONE x0 sequence. Vary this to measure sampling variance.")
parser.add_argument("--match-demo-start", action="store_true",
                    help="Start each episode at the expert's pre-grasp, as the demonstrations "
                         "do, instead of at the env's reset pose. Required to measure a "
                         "checkpoint trained from `collect_demos.py --teleport-pregrasp`: "
                         "such a policy has never seen the reset pose, so scoring it from "
                         "there measures the distribution gap and nothing about the policy. "
                         "A run with this flag is a measurement of the GRASP-AND-PLACE "
                         "MANOEUVRE, not of a deployable pick-and-place policy, and the "
                         "printed header says so.")
parser.add_argument("--expert-rate", type=float, default=None,
                    help="the expert's success on this protocol, for the retention figure")
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
from reBot_RL.tasks.manager_based.re3sim import mdp  # noqa: E402

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))          # eva_bc, for the vendored act/ package
sys.path.insert(0, str(_HERE))
from policy_runner import ChunkController, load_checkpoint  # noqa: E402

sys.path.insert(0, str(_HERE.parent / "expert"))
sys.path.insert(0, str(_HERE.parent / "probes"))
from _kin import ArmKin, Q_OPEN  # noqa: E402
from workstation_expert import plan_episode  # noqa: E402

#: "the grasp took" -- cube base clear of the desk by more than settle noise
LIFT_MIN = 0.060


def teleport_to_demo_start(env, kin):
    """Put every env at the expert's pre-grasp, exactly as `collect_demos --teleport-pregrasp`.

    Envs the planner cannot solve are left at the reset pose and reported, rather than being
    given some other env's plan -- that would place the arm relative to a different layout.
    """
    import torch as _t
    e = env.unwrapped
    cube = mdp.object_pos_local(e, mdp.TARGET_NAME).clone()
    yaw = mdp.yaw_of(mdp.object_quat(e, mdp.TARGET_NAME)).clone()
    boxc = mdp.box_centers_local(e).clone()
    clut = {c: mdp.object_pos_local(e, c).clone() for c in mdp.CLUTTER_NAMES}
    q = kin.q_arm0.unsqueeze(0).repeat(e.num_envs, 1).clone()
    ok = _t.zeros(e.num_envs, dtype=_t.bool, device=e.device)
    for i in range(e.num_envs):
        obstacles = [(clut["rolloftape"][i], mdp.TAPE_DIAMETER / 2, mdp.TAPE_HEIGHT),
                     (clut["tapemeasure"][i], mdp.TAPEMEASURE_LONG / 2, mdp.TAPEMEASURE_HEIGHT)]
        pl = plan_episode(kin, cube[i, :2], float(yaw[i]), boxc[i], obstacles)
        if pl is not None:
            q[i], ok[i] = pl["pre"], True
    kin.teleport_arm(q, q_fing=Q_OPEN)
    # planning teleported the arm all over the workspace; restore the layout it solved against
    for name, p0 in [(mdp.TARGET_NAME, cube)] + [(c, clut[c]) for c in mdp.CLUTTER_NAMES]:
        obj = e.scene[name]
        st = obj.data.root_state_w.torch.clone()
        st[:, :3] = p0 + e.scene.env_origins
        st[:, 7:] = 0.0
        obj.write_root_state_to_sim(st)
    e.sim.forward()
    e.scene.update(e.physics_dt)
    print(f"      demo-start protocol: planner solved {int(ok.sum())}/{e.num_envs}", flush=True)
    return ok


def rollout(env, ctrl, horizon):
    """One spawn batch. Returns latched per-env outcomes, frozen at each env's first done."""
    e = env.unwrapped
    n, dev = e.num_envs, e.device
    tnames = list(e.termination_manager.active_terms)
    done_at = torch.full((n,), -1, dtype=torch.long, device=dev)
    why = torch.zeros((n, len(tnames)), dtype=torch.bool, device=dev)
    succ = torch.zeros(n, dtype=torch.bool, device=dev)
    lifted = torch.zeros(n, dtype=torch.bool, device=dev)
    over = torch.zeros(n, dtype=torch.bool, device=dev)
    disturb = torch.zeros(n, device=dev)
    last_placed = torch.zeros(n, dtype=torch.bool, device=dev)

    ctrl.reset()
    obs = e.observation_manager.compute()["policy"]
    for t in range(horizon):
        obs, _, terminated, truncated, _ = env.step(ctrl.act(obs))
        obs = obs["policy"] if isinstance(obs, dict) else obs
        alive = done_at < 0
        # Read the predicates BEFORE recording the done, so a step that both places the cube
        # and times out still counts.
        placed = mdp.placed_mask(e)
        succ |= placed & alive
        lifted |= (mdp.object_pos_local(e, mdp.TARGET_NAME)[:, 2] > LIFT_MIN) & alive
        over |= mdp.over_box(e) & alive
        # `last_placed` is last-write-wins and must EXCLUDE the step the env dies on: Isaac Lab
        # auto-resets inside `env.step`, so a value read after a done describes a new scene.
        # Every episode here ends on a time_out, so writing on the terminal step reports 0.0 %
        # for every seed -- which reads exactly like a finding and is not one.
        live = alive & ~(terminated | truncated)
        last_placed = torch.where(live, placed, last_placed)
        if hasattr(e, "_clutter_spawn_xy"):
            cur = torch.stack([mdp.object_pos_local(e, nm)[:, :2]
                               for nm in mdp.CLUTTER_NAMES], dim=1)
            d = (cur - e._clutter_spawn_xy).norm(dim=-1).max(dim=1).values
            disturb = torch.where(live, torch.maximum(disturb, d), disturb)
        newly = (terminated | truncated) & alive
        if bool(newly.any()):
            done_at[newly] = t
            for k, nm in enumerate(tnames):
                why[:, k] |= newly & e.termination_manager.get_term(nm)
            ctrl.reset(newly.nonzero(as_tuple=False).squeeze(-1))
    idx = {nm: k for k, nm in enumerate(tnames)}
    drop = why[:, idx["target_dropped"]] if "target_dropped" in idx \
        else torch.zeros(n, dtype=torch.bool, device=dev)
    return {"success": succ, "success_final": last_placed, "lifted": lifted, "over_box": over,
            "dropped": drop, "disturb": disturb,
            "near_miss": ~succ & lifted & over,
            "carried_astray": ~succ & lifted & ~over,
            "no_lift": ~succ & ~lifted,
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
    kin = ArmKin(env) if args_cli.match_demo_start else None
    seeds = [int(s) for s in args_cli.seeds.split(",")]

    print("\n" + "=" * 100)
    print("WORKSTATION FLOW-BC EVALUATION"
          + ("  --  DEMO-START PROTOCOL (grasp-and-place manoeuvre, NOT from a cold reset)"
             if args_cli.match_demo_start else "  --  from the env's own reset pose"))
    print("=" * 100)
    print(f"   ckpt {args_cli.ckpt}  (step {cfg.get('step')})")
    print(f"   chunk {cfg['chunk_size']} / commit {nas}"
          + ("" if nas == cfg["n_action_steps"] else f"  (OVERRIDDEN from {cfg['n_action_steps']})")
          + f", {cfg.get('num_inference_steps', 10)} Euler steps")
    print(f"   {args_cli.num_envs} envs x {len(seeds)} seeds = "
          f"{args_cli.num_envs * len(seeds)} episodes, horizon {horizon} env steps")

    BUCKETS = ("success", "success_final", "lifted", "over_box", "dropped",
               "near_miss", "carried_astray", "no_lift")
    STRICT = (0.002, 0.005, 0.010)
    cells, agg = [], {b: 0 for b in BUCKETS}
    for s in seeds:
        env.reset(seed=s)
        if kin is not None:
            teleport_to_demo_start(env, kin)
        if args_cli.policy_seed is not None:
            torch.manual_seed(args_cli.policy_seed * 100003 + s)
        r = rollout(env, ctrl, horizon)
        c = {"seed": s, **{b: float(r[b].float().mean()) for b in BUCKETS},
             "succ_mask": r["success"].tolist(),
             "disturb_mm": (r["disturb"] * 1000).tolist(),
             **{f"strict_{int(t * 1000)}mm": float((r["success"] & (r["disturb"] < t))
                                                   .float().mean()) for t in STRICT}}
        cells.append(c)
        for b in BUCKETS:
            agg[b] += int(r[b].sum())
        print(f"      seed {s}: SUCCESS {c['success']:6.1%} (final {c['success_final']:6.1%}) "
              f"| lifted {c['lifted']:6.1%} | over-box {c['over_box']:6.1%} "
              f"|| near-miss {c['near_miss']:6.1%} | astray {c['carried_astray']:6.1%} "
              f"| no-lift {c['no_lift']:6.1%} | dropped {c['dropped']:6.1%}")

    N = args_cli.num_envs * len(seeds)
    m = agg["success"] / N
    print("\n   " + "-" * 90)
    print(f"   POOLED SUCCESS {m:.1%}  ({agg['success']}/{N} episodes)   "
          f"last-step-only {agg['success_final'] / N:.1%} "
          f"(delta {(m - agg['success_final'] / N) * 100:+.2f} pts -- the latch's own effect)")
    print("   FAILURE TAXONOMY (mutually exclusive, and they partition the failures)")
    for b in ("near_miss", "carried_astray", "no_lift"):
        print(f"      {b:<15} {agg[b]:5d}  {agg[b] / N:6.1%}")
    print(f"      {'(dropped)':<15} {agg['dropped']:5d}  {agg['dropped'] / N:6.1%}"
          "   overlaps the above -- a termination cause, not a bucket")
    dm = np.concatenate([np.asarray(c["disturb_mm"]) for c in cells])
    sm = np.concatenate([np.asarray(c["succ_mask"], dtype=bool) for c in cells])
    print("\n   CLUTTER DISPLACEMENT -- worst clutter body, distance from its spawn.")
    print("   `placed_mask` does not look at this. Among the episodes it calls SUCCESS:")
    if sm.any():
        print(f"      median {np.median(dm[sm]):6.2f} mm   p90 {np.percentile(dm[sm], 90):6.2f} mm"
              f"   max {dm[sm].max():6.2f} mm")
        for t in STRICT:
            k = float((sm & (dm < t * 1000)).mean())
            print(f"      strict at {t * 1000:4.0f} mm: {k:6.1%}  ({(k - m) * 100:+.2f} pts)")
    else:
        print("      (no successes to describe)")

    verdict = "no expert rate supplied -- pass --expert-rate to get the retention figure"
    if args_cli.expert_rate:
        ret = m / args_cli.expert_rate
        verdict = (f"retains {ret:.0%} of the expert ({args_cli.expert_rate:.1%}). "
                   + ("HEALTHY PORT" if ret >= 0.80 else
                      "ON TREND -- judge on the taxonomy" if ret >= 0.65 else
                      "BELOW TREND -- suspect the observation, not the optimiser" if ret >= 0.45
                      else "PORTING DEFECT -- stop and find it"))
    print(f"\n   vs the expert: {verdict}")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump({"args": vars(args_cli), "ckpt_config": cfg, "n_action_steps": nas,
                   "cells": cells, "pooled": {b: agg[b] / N for b in BUCKETS},
                   "counts": agg, "n": N, "verdict": verdict}, f, indent=2)
    print(f"   wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
