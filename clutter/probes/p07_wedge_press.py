# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P07 -- Can the fingers WEDGE into the gaps and spread the row?

Why this is now the decisive probe
----------------------------------
P06 established, with a grasp primitive validated at 100 % on an isolated target:

* the descent onto the nominal row is **blocked** (TCP stalls at z ~ 88 mm against a
  commanded 55 mm) and the fingers then shut on air;
* the grasp starts working once each neighbour is **~31 mm** out of the way and is reliable
  from **~48 mm**;
* **nothing topples** anywhere in the sweep -- the row stops the gripper, it does not fall over.

That reduces the task to one question: how does anything move sideways? And here geometry
bites. To push a distractor **outward** you must contact its **inner** face, which lives in a
12 mm gap; the closed gripper is far wider than that. Every outward-facing surface in the row
can only be pushed *inward*. **So a lateral push can compress the row but can never spread
it.** Strategy B ("singulate, then grasp") has no contact surface to work with, and strategy C
("plow back") moves blocks along x, which creates no lateral room either.

What is left is the mechanism the env's own docstring describes:

    "The gripper's fingers have to come down that gap, or the policy has to push a
     neighbour aside first."  -- clutter_env_cfg.py:11-13

i.e. the fingers act as **wedges**. A finger entering a 12 mm gap tapers, and pressing down
forces the two blocks apart. P06 only ever held the grasp pose for 25 steps against a
position drive that had already given up 33 mm of tracking error. This probe presses
properly: command the TCP *below* the grasp height so the drive keeps pushing, and dwell.

Sweep
-----
One env per commanded overshoot depth (how far below the grip height the pose is commanded),
which sets how hard the drive presses. Reported per env:

* achieved TCP z -- did the gripper actually descend, and how far;
* d1/d2 lateral displacement -- did the row spread, and by how much;
* `up_z` of all four distractors -- did wedging cost a topple (the episode-ending constraint);
* target displacement -- did the target get shoved out of position instead.

A spread of >= ~31 mm per side with nothing toppled would make the task solvable by wedging
alone. A spread that stalls, or that only arrives together with a topple, closes that door and
forces the Gate 0 decision in `02_PLAN.md`.

Usage
-----
    python eva_bc/clutter/probes/p07_wedge_press.py --num_envs 128 --dwell 300
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Can the fingers wedge the row apart?")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--tilt", type=float, default=15.0)
parser.add_argument("--min_press_z", type=float, default=-0.020,
                    help="deepest commanded TCP z; negative means below the table top")
parser.add_argument("--dwell", type=int, default=300, help="steps held at the pressed pose")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p07_wedge.json")
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

    def put(name, y, x=ROW_X, z=HZ):
        p = torch.tensor([x, y, z], device=dev).repeat(n, 1) + org
        e.scene[name].write_root_state_to_sim(
            torch.cat([p, ident, torch.zeros((n, 6), device=dev)], dim=1))

    def scene():
        put("target", 0.0)
        for d, y in zip(DIST, NOM_Y):
            put(d, y)
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def blocks():
        return {b: (e.scene[b].data.root_pos_w.torch - org).clone()
                for b in ("target",) + DIST}

    print("\n" + "=" * 100)
    print("P07 -- WEDGE PRESS: can the fingers force the row apart?")
    print("=" * 100)
    print(f"   commanded press depth swept {args_cli.grip_z * 1000:.0f} -> "
          f"{args_cli.min_press_z * 1000:.0f} mm over {n} envs, dwell {args_cli.dwell} steps")

    # ---- plan: descend to grip height, then press to a per-env depth
    scene()
    g = math.radians(args_cli.tilt)
    a_des = torch.tensor([-math.cos(g), 0.0, -math.sin(g)], device=dev)
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    r = K.cem(grip, K.q_arm0, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
              iters=args_cli.iters, std0=0.6, restarts=args_cli.restarts)
    q_grip = r["q"]
    print(f"   grasp pose: CEM err {r['pos_err'] * 1000:.2f} mm, o_align {r['o_align']:.3f}, "
          f"a_hat ({r['a_hat'][0]:+.2f},{r['a_hat'][1]:+.2f},{r['a_hat'][2]:+.2f})")

    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    qq, back = q_grip, []
    for t in lerp_pts(grip, grip - STANDOFF * down, 3):
        qq = K.cem(t, qq, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
                   iters=args_cli.iters, std0=0.12)["q"]
        back.append(qq)
    approach = list(reversed(back)) + [q_grip]

    # per-env pressed pose: same wrist, TCP driven below the grip height.
    # `refine` gives per-env joint targets around the CEM-verified pose, and the achieved TCP
    # is read back afterwards so nothing here is a trusted commanded value.
    press_z = torch.linspace(args_cli.grip_z, args_cli.min_press_z, n, device=dev)
    tgt = torch.stack([torch.full((n,), ROW_X, device=dev),
                       torch.zeros(n, device=dev), press_z], dim=1)
    q_press = K.refine(q_grip.unsqueeze(0).repeat(n, 1), tgt, iters=4)
    ach = K.fk(q_press)["tcp"]
    print(f"   per-env refine: median |dz| "
          f"{float((ach[:, 2] - press_z).abs().median()) * 1000:.2f} mm, "
          f"max {float((ach[:, 2] - press_z).abs().max()) * 1000:.2f} mm")

    # ---- execute
    scene()
    b0 = blocks()
    K.teleport_arm(approach[0].unsqueeze(0).repeat(n, 1), Q_OPEN)
    K.hold(approach[0].unsqueeze(0).repeat(n, 1), 15, close=False)
    for i in range(len(approach) - 1):
        K.run(approach[i].unsqueeze(0).repeat(n, 1), approach[i + 1].unsqueeze(0).repeat(n, 1),
              25, close=False)
    K.run(q_grip.unsqueeze(0).repeat(n, 1), q_press, 60, close=False)
    K.hold(q_press, args_cli.dwell, close=False)

    tcp = K.tcp_now()
    b1 = blocks()
    up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
    topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)
    # outward spread: d1 more negative in y, d2 more positive
    d1_out = -(b1[DIST[1]][:, 1] - b0[DIST[1]][:, 1])
    d2_out = (b1[DIST[2]][:, 1] - b0[DIST[2]][:, 1])
    tgt_move = (b1["target"][:, :2] - b0["target"][:, :2]).norm(dim=1)

    print("\n" + "-" * 100)
    print(f"   {'cmd z':>7} {'ach z':>7} | {'d1 out':>7} {'d2 out':>7} | {'min up_z':>8} | "
          f"{'topple':>6} | {'tgt move':>8} | {'tgt z':>6}")
    print("   " + "-" * 92)
    rows = []
    for i in range(0, n, max(1, n // 26)):
        rows.append({"cmd_z_mm": float(press_z[i]) * 1000,
                     "ach_z_mm": float(tcp[i, 2]) * 1000,
                     "d1_out_mm": float(d1_out[i]) * 1000,
                     "d2_out_mm": float(d2_out[i]) * 1000,
                     "min_up_z": float(up[i].min()), "topple": bool(topp[i]),
                     "tgt_move_mm": float(tgt_move[i]) * 1000,
                     "tgt_z_mm": float(b1["target"][i, 2]) * 1000})
        print(f"   {float(press_z[i]) * 1000:6.1f} {float(tcp[i, 2]) * 1000:7.1f} | "
              f"{float(d1_out[i]) * 1000:6.2f} {float(d2_out[i]) * 1000:7.2f} | "
              f"{float(up[i].min()):8.3f} | {'YES' if topp[i] else '-':>6} | "
              f"{float(tgt_move[i]) * 1000:7.2f} | {float(b1['target'][i, 2]) * 1000:6.1f}")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    spread = torch.minimum(d1_out, d2_out)
    ok = (~topp) & (spread > 0.031)
    print(f"   best single-side spread : d1 {float(d1_out.max()) * 1000:.2f} mm, "
          f"d2 {float(d2_out.max()) * 1000:.2f} mm")
    print(f"   best symmetric spread   : {float(spread.max()) * 1000:.2f} mm")
    print(f"   deepest achieved TCP z  : {float(tcp[:, 2].min()) * 1000:.2f} mm "
          f"(commanded down to {args_cli.min_press_z * 1000:.0f} mm)")
    print(f"   topple rate over sweep  : {float(topp.float().mean()):.0%}")
    print(f"   envs reaching the ~31 mm spread with nothing toppled: {int(ok.sum())}/{n}")
    if int(ok.sum()) > 0:
        j = int(torch.nonzero(ok).squeeze(-1)[0])
        print(f"   -> WEDGING WORKS. First at commanded z = {float(press_z[j]) * 1000:.1f} mm "
              f"(achieved {float(tcp[j, 2]) * 1000:.1f} mm), spread "
              f"{float(spread[j]) * 1000:.1f} mm.")
        print("      The row can be opened without a lateral push, which is the only")
        print("      mechanism geometry leaves available. Strategy B is back, by wedging.")
    else:
        print("   -> WEDGING DOES NOT OPEN THE ROW. Combined with the fact that no distractor's")
        print("      inner face is reachable for an outward push, there is no measured")
        print("      mechanism for creating the ~31 mm of lateral room the grasp needs.")
        print("      This is the Gate 0 decision point in 02_PLAN.md -- report it.")

    out = {"grip_z": args_cli.grip_z, "dwell": args_cli.dwell, "n": n,
           "cmd_z_mm": (press_z * 1000).tolist(),
           "ach_z_mm": (tcp[:, 2] * 1000).tolist(),
           "d1_out_mm": (d1_out * 1000).tolist(), "d2_out_mm": (d2_out * 1000).tolist(),
           "min_up_z": up.min(dim=1).values.tolist(), "topple": topp.tolist(),
           "tgt_move_mm": (tgt_move * 1000).tolist(), "rows": rows,
           "best_symmetric_spread_mm": float(spread.max()) * 1000,
           "n_ok": int(ok.sum())}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p07] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
