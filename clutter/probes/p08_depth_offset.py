# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P08 -- Would moving the neighbours BACKWARD (+x) unblock the grasp?

Testing a precondition before building its mechanism
----------------------------------------------------
`07_STAGE0_RESULTS.md` §7 dismissed strategy C ("plow the neighbours back along x") on the
grounds that it "creates no lateral room". That reasoning was wrong, and the measurement that
shows why is already in hand: P04 established that this arm's fingers point **back toward the
robot** (`a_hat . x = -0.98` at home; ~(-0.77, ., +0.56) at a grasp pose). The jaw therefore
extends from the wrist in **-x**, and the wrist sits *behind* the grasp point. So the volume
the gripper needs is not a y-slab through the row -- it is a wedge that leans toward the robot.
Neighbours displaced in **+x** might simply fall outside it, even though the row's y-pitch is
untouched.

That is a cheap thing to test and an expensive thing to build. P07's push mechanism would need
its own pose chain, its own contact-height sweep and its own topple accounting; there is no
point writing any of it if the precondition fails. So this probe **teleports** d1 and d2
backward by a swept offset and runs the already-validated grasp. Privileged, deliberately: it
measures whether the geometry admits a solution, not whether a policy could reach it.

Reads directly against P06, which is the same experiment with the offset applied along y:

    P06 (lateral, y):  first grasp at ~31 mm/side, reliable at ~48 mm/side, 0 % topple
    P08 (depth,   x):  this file

If a modest +x offset unblocks the grasp, strategy C is alive and the next question is whether
the neighbours can be pushed there without toppling (their -x faces ARE exposed, unlike their
inner faces -- §7). If no offset works, then combined with P06 and P07 there is no measured
mechanism at all and Gate 0 fails on evidence.

Usage
-----
    python eva_bc/clutter/probes/p08_depth_offset.py --num_envs 128 --max_back 0.090
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Does a +x offset of the neighbours unblock it?")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--tilt", type=float, default=15.0)
parser.add_argument("--max_back", type=float, default=0.090)
parser.add_argument("--forward", action="store_true",
                    help="offset the neighbours toward the robot (-x) instead of away")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p08_depth.json")
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
NOM_Y = (-2 * ROW_PITCH, -ROW_PITCH, ROW_PITCH, 2 * ROW_PITCH)
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

    sign = -1.0 if args_cli.forward else 1.0
    back = torch.linspace(0.0, args_cli.max_back, n, device=dev) * sign

    def put(name, y, x_per_env=None, z=HZ):
        x = torch.full((n,), ROW_X, device=dev) if x_per_env is None else x_per_env
        p = torch.stack([x, torch.full((n,), y, device=dev),
                         torch.full((n,), z, device=dev)], dim=1) + org
        e.scene[name].write_root_state_to_sim(
            torch.cat([p, ident, torch.zeros((n, 6), device=dev)], dim=1))

    def scene():
        put("target", 0.0)
        for d, y in zip(DIST, NOM_Y):
            put(d, y, ROW_X + back)
        e.sim.forward()
        e.scene.update(e.physics_dt)

    print("\n" + "=" * 96)
    print(f"P08 -- NEIGHBOUR DEPTH OFFSET ({'-x, toward robot' if sign < 0 else '+x, away'})")
    print("=" * 96)
    print(f"   all four distractors offset 0 .. {args_cli.max_back * 1000:.0f} mm along "
          f"{'-x' if sign < 0 else '+x'}; the target stays at x = {ROW_X}")
    print(f"   row y-pitch is UNCHANGED -- this tests depth clearance, not lateral room")

    scene()
    g = math.radians(args_cli.tilt)
    a_des = torch.tensor([-math.cos(g), 0.0, -math.sin(g)], device=dev)
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    r = K.cem(grip, K.q_arm0, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
              iters=args_cli.iters, std0=0.6, restarts=args_cli.restarts)
    q_grip = r["q"]
    print(f"\n   grasp pose: CEM err {r['pos_err'] * 1000:.2f} mm, o_align {r['o_align']:.3f}, "
          f"a_hat ({r['a_hat'][0]:+.2f},{r['a_hat'][1]:+.2f},{r['a_hat'][2]:+.2f})")

    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    qq, appr = q_grip, []
    for t in lerp_pts(grip, grip - STANDOFF * down, 3):
        qq = K.cem(t, qq, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
                   iters=args_cli.iters, std0=0.12)["q"]
        appr.append(qq)
    approach = list(reversed(appr)) + [q_grip]
    qq, up_seq = q_grip, []          # lift planned BEFORE the close
    for t in lerp_pts(grip, grip + torch.tensor([0.0, 0.0, 0.09], device=dev), 3):
        qq = K.cem(t, qq, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
                   iters=args_cli.iters, std0=0.12)["q"]
        up_seq.append(qq)

    scene()
    K.teleport_arm(approach[0].unsqueeze(0).repeat(n, 1), Q_OPEN)
    K.hold(approach[0].unsqueeze(0).repeat(n, 1), 15, close=False)
    for i in range(len(approach) - 1):
        K.run(approach[i].unsqueeze(0).repeat(n, 1), approach[i + 1].unsqueeze(0).repeat(n, 1),
              25, close=False)
    K.hold(q_grip.unsqueeze(0).repeat(n, 1), 25, close=False)
    tcp_grasp = K.tcp_now().clone()
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
    blocked = (tcp_grasp[:, 2] - args_cli.grip_z) > 0.004

    print("\n" + "-" * 96)
    print(f"   {'offset':>8} | {'TCP z @grasp':>12} | {'blocked':>7} | {'stall gap':>9} | "
          f"{'HELD':>5} | {'topple':>6} | {'min up_z':>8}")
    print("   " + "-" * 84)
    rows = []
    for i in range(0, n, max(1, n // 26)):
        rows.append({"offset_mm": float(back[i]) * 1000,
                     "tcp_z_mm": float(tcp_grasp[i, 2]) * 1000, "blocked": bool(blocked[i]),
                     "stall_gap_mm": float(gap_stall[i]) * 1000, "held": bool(held[i]),
                     "topple": bool(topp[i]), "min_up_z": float(up[i].min())})
        print(f"   {float(back[i]) * 1000:7.1f}m | {float(tcp_grasp[i, 2]) * 1000:11.2f} | "
              f"{'YES' if blocked[i] else '-':>7} | {float(gap_stall[i]) * 1000:8.2f} | "
              f"{'HELD' if held[i] else '.':>5} | {'YES' if topp[i] else '-':>6} | "
              f"{float(up[i].min()):8.3f}")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    idx = torch.nonzero(held).squeeze(-1)
    out = {"grip_z": args_cli.grip_z, "sign": sign, "n": n,
           "offset_mm": (back * 1000).tolist(),
           "tcp_z_mm": (tcp_grasp[:, 2] * 1000).tolist(), "held": held.tolist(),
           "topple": topp.tolist(), "blocked": blocked.tolist(),
           "stall_gap_mm": (gap_stall * 1000).tolist(), "rows": rows}
    if len(idx) == 0:
        print(f"   NO depth offset up to {args_cli.max_back * 1000:.0f} mm unblocks the grasp.")
        print(f"   envs where the descent reached grip height: {int((~blocked).sum())}/{n}")
        print("   -> Strategy C is dead on its precondition: even with the neighbours")
        print("      teleported out of the way along x, the grasp does not happen. No push")
        print("      mechanism needs to be built.")
        out["threshold_mm"] = None
    else:
        thr = float(back[idx.min()]) * 1000
        nb = int((~blocked).float().sum())
        print(f"   FIRST success at offset = {thr:.2f} mm ({int(len(idx))}/{n} envs held)")
        print(f"   envs with an unblocked descent: {nb}/{n}")
        print(f"   topple rate: {float(topp.float().mean()):.0%}")
        print("   -> Strategy C's PRECONDITION HOLDS. Displacing the neighbours along x by")
        print(f"      {thr:.0f} mm is enough. Next: can they be pushed there without toppling?")
        print("      Their -x faces ARE exposed, so unlike a lateral push this one has a")
        print("      contact surface. Contact height vs h_crit = 19.0 mm (x axis) is the risk.")
        out["threshold_mm"] = thr

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p08] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
