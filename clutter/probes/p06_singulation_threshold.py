# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P06 -- How far must the neighbours be pushed before the target can be grasped?

The one number Stage 0 actually needs
-------------------------------------
P03 measured, with a grasp primitive validated at **100 % on an isolated target**:

    solo      HELD 100 %   (gap stalls at 30.0 mm, encl 100 %, rose 100 %)
    clutter   HELD   0 %   (gap -> -1.2 mm: shut on air)   topple 0 %
    gap +25mm HELD   0 %   (same)                          topple 0 %

Two things in that table are informative beyond the obvious. First, **topple 0 %** -- the
gripper is not knocking the row over, it is being *stopped* by it. Second, the `gap` condition
gave 104 mm of clear space between d1 and d2 and still failed, which rules out the comfortable
explanation that the fingers merely clip the neighbours.

That points back at P01. Its axis-aligned finger bound was dismissed as unusable because it
implies a 51.7 mm clear gap where C3 *measured* 89.07 mm. But the two disagreements are not
the same kind: an AABB overestimates *inward* extent (it fills a concave jaw with air) while
its *outer* extreme is attained by a real mesh vertex. Reading it that way:

    outward thickness  t  = (128.59 - 89.07) / 2 = 19.8 mm per side
    open gripper outer width                     = 128.6 mm

and 104 mm of clearance is simply not enough room. That is a falsifiable prediction, so
falsify it: sweep the singulation distance and find where the grasp starts working.

Method
------
One env per push distance -- the grasp trajectory does not depend on where the distractors
are, so a single shared execution tests the whole sweep at once. d1/d2 move out by `push`,
d0/d3 by `2*push` so they stay ahead and the inner pair never simply collides with them.

Reported per env: the **achieved** TCP z at the grasp phase (a descent blocked by a neighbour
stalls high, which separates "blocked" from "reached but missed"), the stall gap, whether the
target rose, and whether anything toppled.

The output is a threshold in millimetres. It is the difference between strategy B being a
20 mm nudge and being a demolition job -- and it is also, directly, the number that says
whether `Tight-v0` is reachable at all.

Usage
-----
    python eva_bc/clutter/probes/p06_singulation_threshold.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Singulation distance needed to allow a grasp.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128, help="one env per push distance")
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--tilt", type=float, default=15.0)
parser.add_argument("--max_push", type=float, default=0.070)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p06_singulation.json")
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
from _kin import ArmKin, Q_OPEN, lerp_pts  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
ROW_PITCH = 0.042
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
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
    Y = torch.tensor([0.0, 1.0, 0.0], device=dev)

    push = torch.linspace(0.0, args_cli.max_push, n, device=dev)

    def put(name, y_per_env, x=ROW_X, z=HZ):
        p = torch.stack([torch.full((n,), x, device=dev), y_per_env,
                         torch.full((n,), z, device=dev)], dim=1) + org
        e.scene[name].write_root_state_to_sim(
            torch.cat([p, ident, torch.zeros((n, 6), device=dev)], dim=1))

    def scene():
        put("target", torch.zeros(n, device=dev))
        put(DIST[0], -2 * ROW_PITCH - 2 * push)
        put(DIST[1], -ROW_PITCH - push)
        put(DIST[2], ROW_PITCH + push)
        put(DIST[3], 2 * ROW_PITCH + 2 * push)
        e.sim.forward()
        e.scene.update(e.physics_dt)

    print("\n" + "=" * 96)
    print("P06 -- SINGULATION THRESHOLD")
    print("=" * 96)
    print(f"   grip z {args_cli.grip_z * 1000:.0f} mm, approach tilt {args_cli.tilt:.0f} deg, "
          f"vertical descent from +{STANDOFF * 1000:.0f} mm")
    print(f"   push swept 0 .. {args_cli.max_push * 1000:.0f} mm over {n} envs "
          f"({args_cli.max_push * 1000 / (n - 1):.2f} mm per env)")
    print(f"   free space between d1 and d2 inner faces = 54 + 2*push mm")

    # ---- plan once, with the clutter present (it does not affect the kinematics)
    scene()
    g = math.radians(args_cli.tilt)
    a_des = torch.tensor([-math.cos(g), 0.0, -math.sin(g)], device=dev)
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    r = K.cem(grip, K.q_arm0, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
              iters=args_cli.iters, std0=0.6, restarts=args_cli.restarts)
    q_grip = r["q"]
    print(f"\n   grasp pose: CEM err {r['pos_err'] * 1000:.2f} mm, o_align {r['o_align']:.3f}, "
          f"a_hat ({r['a_hat'][0]:+.2f},{r['a_hat'][1]:+.2f},{r['a_hat'][2]:+.2f}), "
          f"gripper low_z {r['low_z'] * 1000:.1f} mm")

    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    qq, back = q_grip, []
    for t in lerp_pts(grip, grip - STANDOFF * down, 3):
        qq = K.cem(t, qq, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
                   iters=args_cli.iters, std0=0.12)["q"]
        back.append(qq)
    approach = list(reversed(back)) + [q_grip]
    # lift planned BEFORE the close -- see 07_STAGE0_RESULTS.md 4.3
    qq, up_seq = q_grip, []
    for t in lerp_pts(grip, grip + torch.tensor([0.0, 0.0, 0.09], device=dev), 3):
        qq = K.cem(t, qq, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
                   iters=args_cli.iters, std0=0.12)["q"]
        up_seq.append(qq)

    # ---- execute (identical for every env; only the scene differs)
    scene()
    K.teleport_arm(approach[0].unsqueeze(0).repeat(n, 1), Q_OPEN)
    K.hold(approach[0].unsqueeze(0).repeat(n, 1), 15, close=False)
    for i in range(len(approach) - 1):
        K.run(approach[i].unsqueeze(0).repeat(n, 1), approach[i + 1].unsqueeze(0).repeat(n, 1),
              25, close=False)
    K.hold(q_grip.unsqueeze(0).repeat(n, 1), 25, close=False)
    tcp_at_grasp = K.tcp_now().clone()          # achieved, per env -- the blocked/reached test
    up_at_grasp = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1).clone()
    # WHAT is stopping the descent? Record every block and every gripper body at the moment
    # the arm has finished trying to reach the grip pose. A threshold that implies a 190 mm
    # gripper is a sign the blocking model is wrong, not that the gripper is 190 mm wide.
    bp_all = (K.robot.data.body_pos_w.torch - org.unsqueeze(1)).clone()
    blocks_at_grasp = {b: (e.scene[b].data.root_pos_w.torch - org).clone()
                       for b in ("target",) + DIST}
    gbodies = {"link6": K.robot.body_names.index("link6"), "end": K.i_end,
               "fL": K.i_left, "fR": K.i_right}
    print("\n" + "-" * 96)
    print("   CONTACT DIAGNOSTIC AT THE GRASP PHASE (achieved positions, mm)")
    print("-" * 96)
    print(f"   {'push':>6} | {'TCP z':>6} | " + " ".join(f"{k + ' y,z':>14}" for k in gbodies)
          + f" | {'tgt z':>6} | {'d1 y,z':>13} | {'d2 y,z':>13}")
    for i in range(0, n, max(1, n // 12)):
        cells = " ".join(f"{bp_all[i, j, 1] * 1000:6.1f},{bp_all[i, j, 2] * 1000:6.1f} "
                         for j in gbodies.values())
        d1 = blocks_at_grasp[DIST[1]][i]
        d2 = blocks_at_grasp[DIST[2]][i]
        print(f"   {float(push[i]) * 1000:5.1f}m | {float(tcp_at_grasp[i, 2]) * 1000:6.1f} | "
              f"{cells}| {float(blocks_at_grasp['target'][i, 2]) * 1000:6.1f} | "
              f"{float(d1[1]) * 1000:6.1f},{float(d1[2]) * 1000:5.1f} | "
              f"{float(d2[1]) * 1000:6.1f},{float(d2[2]) * 1000:5.1f}")
    K.hold(q_grip.unsqueeze(0).repeat(n, 1), 70, close=True)
    gap_stall = K.gap().clone()
    seq = [q_grip] + up_seq
    for i in range(len(seq) - 1):
        K.run(seq[i].unsqueeze(0).repeat(n, 1), seq[i + 1].unsqueeze(0).repeat(n, 1),
              25, close=True)
    K.hold(up_seq[-1].unsqueeze(0).repeat(n, 1), 50, close=True)

    bpos = e.scene["target"].data.root_pos_w.torch - org
    tcp = K.tcp_now()
    gap = K.gap()
    encl = (gap - BLOCK_W).abs() < 0.012
    rose = bpos[:, 2] > HZ + 0.045
    near = (tcp - bpos).norm(dim=1) < 0.09
    held = rose & near & encl
    up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
    topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)
    # a descent stopped by a neighbour never reaches the commanded grip height
    blocked = (tcp_at_grasp[:, 2] - args_cli.grip_z) > 0.004

    print("\n" + "-" * 96)
    print(f"   {'push':>6} {'free':>6} | {'TCP z @grasp':>12} | {'blocked':>7} | "
          f"{'stall gap':>9} | {'encl':>5} | {'rose':>5} | {'HELD':>5} | {'topple':>6}")
    print("   " + "-" * 88)
    step = max(1, n // 28)
    rows = []
    for i in range(0, n, step):
        rows.append({"push_mm": float(push[i]) * 1000,
                     "free_mm": 54.0 + 2000 * float(push[i]),
                     "tcp_z_at_grasp_mm": float(tcp_at_grasp[i, 2]) * 1000,
                     "blocked": bool(blocked[i]), "stall_gap_mm": float(gap_stall[i]) * 1000,
                     "encl": bool(encl[i]), "rose": bool(rose[i]), "held": bool(held[i]),
                     "topple": bool(topp[i])})
        print(f"   {float(push[i]) * 1000:5.1f}mm {54.0 + 2000 * float(push[i]):5.0f} | "
              f"{float(tcp_at_grasp[i, 2]) * 1000:11.2f} | "
              f"{'YES' if blocked[i] else '-':>7} | {float(gap_stall[i]) * 1000:8.2f} | "
              f"{'Y' if encl[i] else '.':>5} | {'Y' if rose[i] else '.':>5} | "
              f"{'HELD' if held[i] else '.':>5} | {'YES' if topp[i] else '-':>6}")

    # ---- threshold
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    idx = torch.nonzero(held).squeeze(-1)
    out = {"grip_z": args_cli.grip_z, "tilt_deg": args_cli.tilt, "n": n,
           "push_mm": (push * 1000).tolist(),
           "tcp_z_at_grasp_mm": (tcp_at_grasp[:, 2] * 1000).tolist(),
           "blocked": blocked.tolist(), "stall_gap_mm": (gap_stall * 1000).tolist(),
           "held": held.tolist(), "topple": topp.tolist(), "rows": rows,
           "plan": {k: (float(v) if isinstance(v, float) else
                        [float(x) for x in v]) for k, v in r.items()
                    if k in ("pos_err", "o_align", "a_align", "low_z", "a_hat", "o_hat")}}
    if len(idx) == 0:
        print(f"   NO push distance up to {args_cli.max_push * 1000:.0f} mm allows the grasp.")
        print("   Either the gripper is wider than 54 + 2*max_push mm, or something other")
        print("   than the neighbours is blocking the descent. Check `blocked` and TCP z.")
        nb = int((~blocked).sum())
        print(f"   envs where the descent DID reach grip height: {nb}/{n}")
        out["threshold_mm"] = None
    else:
        thr = float(push[idx.min()]) * 1000
        print(f"   FIRST success at push = {thr:.2f} mm  "
              f"(free space {54.0 + 2 * thr:.1f} mm between d1/d2 inner faces)")
        print(f"   implied open-gripper outer width  ~ {54.0 + 2 * thr:.1f} mm")
        print(f"   success rate above threshold: "
              f"{float(held[idx.min():].float().mean()):.0%}")
        print(f"   topple rate over the whole sweep: {float(topp.float().mean()):.0%}")
        out["threshold_mm"] = thr
        print()
        print(f"   Consequence for strategy B (singulate then grasp): each neighbour must be")
        print(f"   moved {thr:.1f} mm outward. The nominal free gap is 12 mm, so d1 must also")
        print(f"   displace d0 (and d2 displace d3) unless the push is staged outside-in.")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p06] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
