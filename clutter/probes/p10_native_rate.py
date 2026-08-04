# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P10 -- What does the validated grasp score on genuinely random spawns, unaided?

Why this is the number that matters
-----------------------------------
Stage 0 has established that the grasp itself is sound (100 % held on an isolated target) and
that the row blocks it (0 % at nominal pitch), and it has ruled out every mechanism for
creating clearance: no lateral push has a contact surface, wedging yields 4.2 mm against the
~31 mm needed, and every reachable +x push topples the block.

But all of those were measured at the **nominal** row. The env's reset events jitter the
target by x in +/-12 mm and each distractor independently by x in +/-10 mm and y in +/-5 mm.
So a random spawn already supplies up to **22 mm of relative x offset** and a free gap
anywhere in the measured 7.0-18.8 mm range, for free. P08 put the x-clearance threshold at
65 mm for a single pose, and P06 showed the threshold moves ~37 mm with opening-axis alignment
alone -- so the tail of the spawn distribution is not obviously empty.

This probe measures that tail directly: reset normally, adapt the grasp to where the target
actually is, execute, and score. It is a Stage-1 expert in miniature, and its output is the
honest floor on what any policy could achieve without solving the clutter problem at all.

* If the rate is ~0 %, Gate 0 is closed on evidence and there is nothing to build on.
* If it is materially above 0 %, there is a graspable sub-population, the task has a base
  case, and Stage 1 has somewhere to start -- the expert's job becomes converting the rest.

Either way the run yields the **conditioning data** that Gate 1 requires anyway: success
stratified by the per-episode minimum free gap and by the target's relative x offset.

Design notes
------------
* Per-env adaptation uses `refine` around one CEM-verified nominal pose, then re-reads the
  achieved TCP from the sim. Nothing is trusted as commanded.
* Execution steps **physics only** (`_kin.hold_phys`/`run_phys`). Stepping the MDP would let
  `distractor_toppled` reset the scene mid-trial and re-spawn the blocks upright, which is the
  exact defect that made P09's first run unreadable. Success and topple are evaluated here
  against the same predicates the env uses (`mdp.clutter.target_at_goal`,
  `any_distractor_toppled`), just read at the end rather than enforced during.
* The whole plan -- descent, lift, carry, place -- is solved BEFORE the fingers close.

Usage
-----
    python eva_bc/clutter/probes/p10_native_rate.py --num_envs 128 --reps 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Native grasp rate on random spawns.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--reps", type=int, default=2, help="independent reset batches")
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=10)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--tilt", type=float, default=15.0)
parser.add_argument("--phi", type=float, default=90.0,
                    help="opening-axis azimuth [deg]: 0 = across the row (y), 90 = fore/aft "
                         "(x, the orthogonal grasp P11 validated at 100%%)")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p10_native.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kin import ArmKin, Q_CLOSE, Q_OPEN, lerp_pts  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
BLOCK_W = 2 * HY
DIST = mdp_cl.DISTRACTOR_NAMES
STANDOFF = 0.070


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    K = ArmKin(env)
    dev, n = K.dev, K.n
    org = e.scene.env_origins
    goal = torch.tensor(mdp_cl.GOAL_XY, device=dev)
    # Opening-axis direction. phi = 90 is the ORTHOGONAL grasp: the fingers straddle the
    # target fore and aft, sit at |y| < 5 mm, and never enter the 12 mm gaps that block the
    # cross-row grasp. P11 measured 100 % held / 0 % topple with it on the nominal row.
    _p = math.radians(args_cli.phi)
    Y = torch.tensor([math.sin(_p), math.cos(_p), 0.0], device=dev)
    # An orthogonal jaw closes on the block's 36 mm x-faces, a cross-row jaw on its 30 mm
    # y-faces. Scoring with the wrong width marks every real grasp as a failure.
    width = 2 * HX if args_cli.phi >= 45.0 else 2 * HY
    # P11 left the approach axis unconstrained and that is what found these poses; forcing
    # a_des here re-imposes the assumption that produced the earlier false negatives.
    a_des = None

    print("\n" + "=" * 100)
    print("P10 -- NATIVE GRASP RATE ON RANDOM SPAWNS (no clutter manipulation)")
    print("=" * 100)

    def best_pose(pos, seed, tries=6, std0=0.6, restarts=None, pos_max=0.0015):
        """Pick the pose with the highest `o_align`, not the first the CEM happens to return.

        The CEM is stochastic and its draws vary a lot in jaw alignment: P11's winning pose
        had `o_align = 0.998` while a later run of the same call drew 0.848 -- a jaw 32 deg off
        axis, which swings the fingers from |y| < 5 mm out to +/-24 mm and straight back into
        the 12 mm gaps. P06 already quantified this: alignment is worth ~37 mm of required
        clearance. So alignment is a **selection criterion**, not a soft cost term, and any
        expert built on this must gate on it explicitly.
        """
        best = None
        for _ in range(tries):
            c = K.cem(pos, seed, o_des=Y, a_des=a_des, w_o=0.60, iters=args_cli.iters,
                      std0=std0, restarts=restarts or args_cli.restarts)
            if c["pos_err"] > pos_max:
                continue
            if best is None or c["o_align"] > best["o_align"]:
                best = c
        return best or c

    # ---------- one nominal plan, solved once, entirely before anything closes
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    r = best_pose(grip, K.q_arm0)
    q_grip = r["q"]
    print(f"   nominal grasp pose: CEM err {r['pos_err'] * 1000:.2f} mm, "
          f"o_align {r['o_align']:.3f}, a_hat ({r['a_hat'][0]:+.2f},{r['a_hat'][1]:+.2f},"
          f"{r['a_hat'][2]:+.2f})")

    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    qq, appr = q_grip, []
    for t in lerp_pts(grip, grip - STANDOFF * down, 3):
        qq = best_pose(t, qq, tries=3, std0=0.15)["q"]
        appr.append(qq)
    approach_nom = list(reversed(appr))            # high -> low, grasp appended per-env

    lift = grip + torch.tensor([0.0, 0.0, 0.075], device=dev)
    carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + 0.075], device=dev)
    place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
    qq, post = q_grip, []
    for t in [lift, carry, place]:
        qq = best_pose(t, qq, tries=3, std0=0.30, restarts=3)["q"]
        post.append(qq)
    print(f"   carry chain solved to goal ({float(goal[0]) * 1000:.0f}, "
          f"{float(goal[1]) * 1000:.0f}) mm")

    all_held, all_succ, all_topp, all_gap, all_dx = [], [], [], [], []
    for rep in range(args_cli.reps):
        env.reset()
        for _ in range(30):                        # let the spawn settle
            e.sim.step()
            e.scene.update(e.physics_dt)

        tpos = (e.scene["target"].data.root_pos_w.torch - org).clone()
        dpos = torch.stack([(e.scene[d].data.root_pos_w.torch - org) for d in DIST], dim=1)
        # per-episode minimum free gap between adjacent bodies, measured off the settled row
        ys = torch.cat([dpos[:, :2, 1], tpos[:, 1:2], dpos[:, 2:, 1]], dim=1)
        ys, _ = ys.sort(dim=1)
        min_gap = (ys[:, 1:] - ys[:, :-1] - 2 * HY).min(dim=1).values
        # how far the target sits in FRONT of its two neighbours (the P08 axis)
        rel_dx = dpos[:, 1:3, 0].mean(dim=1) - tpos[:, 0]

        # per-env grasp target = where the target actually is
        tgt = torch.stack([tpos[:, 0], tpos[:, 1],
                           torch.full((n,), args_cli.grip_z, device=dev)], dim=1)
        q_g = K.refine(q_grip.unsqueeze(0).repeat(n, 1), tgt, iters=4)
        ach = K.fk(q_g)["tcp"]
        perr = (ach - tgt).norm(dim=1)
        q_app = [K.refine(q.unsqueeze(0).repeat(n, 1),
                          tgt + torch.tensor([0.0, 0.0, dz], device=dev), iters=3)
                 for q, dz in zip(approach_nom, (STANDOFF, 2 * STANDOFF / 3, STANDOFF / 3))]

        # ---- execute, physics only
        K.teleport_arm(q_app[0], Q_OPEN)
        K.hold_phys(q_app[0], 80)
        seq = q_app + [q_g]
        for i in range(len(seq) - 1):
            K.run_phys(seq[i], seq[i + 1], 25)
        K.hold_phys(q_g, 160)
        tcp_grasp = K.tcp_now().clone()
        K.hold_phys(q_g, 560, q_fing=Q_CLOSE)
        gap_stall = K.gap().clone()
        chain = [q_g] + [p.unsqueeze(0).repeat(n, 1) for p in post]
        for i in range(len(chain) - 1):
            K.run_phys(chain[i], chain[i + 1], 30, q_fing=Q_CLOSE)
        K.hold_phys(chain[-1], 160, q_fing=Q_CLOSE)
        K.hold_phys(chain[-1], 240, q_fing=Q_OPEN)      # release
        K.run_phys(chain[-1], chain[1], 25, q_fing=Q_OPEN)
        K.hold_phys(chain[1], 240, q_fing=Q_OPEN)

        bpos = e.scene["target"].data.root_pos_w.torch - org
        up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
        topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)
        held = (gap_stall - width).abs() < 0.012
        at_goal = ((bpos[:, :2] - goal).norm(dim=1) < mdp_cl.GOAL_RADIUS) & (bpos[:, 2] < 0.055)
        succ = at_goal & ~topp

        print(f"\n   rep {rep}: refine err median {float(perr.median()) * 1000:.2f} mm | "
              f"min free gap {float(min_gap.min()) * 1000:.1f}..{float(min_gap.max()) * 1000:.1f} mm "
              f"(median {float(min_gap.median()) * 1000:.1f}) | "
              f"rel dx {float(rel_dx.min()) * 1000:+.1f}..{float(rel_dx.max()) * 1000:+.1f} mm")
        print(f"           enclosed-at-close {float(held.float().mean()):.1%} | "
              f"at goal {float(at_goal.float().mean()):.1%} | "
              f"topple {float(topp.float().mean()):.1%} | "
              f"SUCCESS {float(succ.float().mean()):.1%}")
        all_held.append(held)
        all_succ.append(succ)
        all_topp.append(topp)
        all_gap.append(min_gap)
        all_dx.append(rel_dx)

    held = torch.cat(all_held)
    succ = torch.cat(all_succ)
    topp = torch.cat(all_topp)
    gapv = torch.cat(all_gap)
    dxv = torch.cat(all_dx)
    N = len(succ)

    print("\n" + "=" * 100)
    print(f"POOLED OVER {N} EPISODES")
    print("=" * 100)
    print(f"   enclosed at close : {float(held.float().mean()):.1%}")
    print(f"   target at goal    : {float(succ.float().mean()):.1%}")
    print(f"   topple rate       : {float(topp.float().mean()):.1%}")

    print("\n   stratified by per-episode minimum free gap:")
    print(f"      {'gap band [mm]':>16} | {'n':>5} | {'enclosed':>9} | {'success':>8}")
    edges = [0, 8, 10, 12, 14, 100]
    strat = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gapv * 1000 >= lo) & (gapv * 1000 < hi)
        if int(m.sum()) == 0:
            continue
        strat.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                      "encl": float(held[m].float().mean()),
                      "succ": float(succ[m].float().mean())})
        print(f"      {f'{lo}-{hi}':>16} | {int(m.sum()):5d} | "
              f"{float(held[m].float().mean()):8.1%} | {float(succ[m].float().mean()):7.1%}")

    print("\n   stratified by target's forward offset relative to its neighbours:")
    print(f"      {'dx band [mm]':>16} | {'n':>5} | {'enclosed':>9} | {'success':>8}")
    dstrat = []
    for lo, hi in ((-100, -10), (-10, 0), (0, 10), (10, 100)):
        m = (dxv * 1000 >= lo) & (dxv * 1000 < hi)
        if int(m.sum()) == 0:
            continue
        dstrat.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                       "encl": float(held[m].float().mean()),
                       "succ": float(succ[m].float().mean())})
        print(f"      {f'{lo}..{hi}':>16} | {int(m.sum()):5d} | "
              f"{float(held[m].float().mean()):8.1%} | {float(succ[m].float().mean()):7.1%}")

    print()
    if float(succ.float().mean()) > 0.02:
        print("   -> There IS a graspable sub-population. The task has a base case: some random")
        print("      spawns are solvable with no clutter manipulation at all. Stage 1's job")
        print("      becomes converting the rest, and this rate is the floor to beat.")
    elif float(held.float().mean()) > 0.02:
        print("   -> The grasp closes on the target in some spawns but the episode is not")
        print("      completed. The blocker is downstream of the grasp (extract/carry/place),")
        print("      which is a much better problem to have than a blocked approach.")
    else:
        print("   -> No random spawn is graspable unaided. Combined with P06-P09 there is no")
        print("      measured path to the goal. Gate 0 fails on evidence; report and stop.")

    out = {"n": N, "reps": args_cli.reps, "grip_z": args_cli.grip_z,
           "encl_rate": float(held.float().mean()),
           "success_rate": float(succ.float().mean()),
           "topple_rate": float(topp.float().mean()),
           "min_gap_mm": (gapv * 1000).tolist(), "rel_dx_mm": (dxv * 1000).tolist(),
           "held": held.tolist(), "succ": succ.tolist(), "topple": topp.tolist(),
           "by_gap": strat, "by_dx": dstrat}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p10] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
