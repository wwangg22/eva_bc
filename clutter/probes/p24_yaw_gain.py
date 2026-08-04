# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P24 -- Does matching the target's yaw make the blade sweep WORSE? A 2-D sweep.

The prediction this tests
-------------------------
P22 established that the finger blades sweep the neighbours directly during the close, and
gave the geometry to reason about it for the first time:

    gripper_left   x -19.2 .. +19.2 | y -41.9 .. +46.7 | z -58.7 .. +34.7   [mm, body frame]

Perpendicular to the opening axis the blade is only **+/-19.2 mm**, against neighbour faces
at +/-27 mm: **7.8 mm of margin**. Along the opening axis it reaches **~47 mm**. Rotating
the opening axis to match the block's spawn yaw therefore swings the blade's own corner:

    47 mm * sin(11.4 deg) = 9.3 mm    against 7.8 mm of margin

A fully yaw-matched jaw is predicted to put its corners into the neighbours, and the
prediction is retrodictive as well as forward-looking -- two measurements already fit it and
neither was recognised:

    P16   matching yaw raised close-phase contacts from 65/128 to 93/128, while improving
          every grip statistic (block turn during close 3.72 deg -> 0.28 deg)
    P19   the SQUARE jaw outscored the matched one end to end, 69.5 % vs 64.1 %, and it was
          written off as noise

So the yaw fix (`08_STAGE1_RESULTS` 2.4) may be buying grip quality at the cost of exactly
the clearance that now dominates. This probe makes the trade explicit instead of assumed.

The sweep
---------
`yaw_gain` rotates the jaw `gain` of the way from the row-square axis toward the block's own:
0 is the square jaw, 1 is the full match currently shipped. Crossed with `phi`, the opening
azimuth, because it moves the same geometry from the other side -- at `phi < 90` the fingers
sit partly across the row, trading blade-corner clearance for gap intrusion. P11 found
`phi = 45/70/80` attainable and row-clear.

Every cell is measured on the **isolated close** (settle at the grasp pose, close, measure).
No descent, no carry, no place, so nothing depends on the path or on downstream luck -- and
with `plan_full=False` a cell costs a grasp-pose solve instead of a 23-waypoint chain, which
is what makes a 2-D sweep affordable at three pose draws per cell.

The winner is then confirmed end-to-end, because an isolated result that does not transfer
is not a result.

Usage
-----
    python eva_bc/clutter/probes/p24_yaw_gain.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Yaw-gain x phi sweep on the isolated close.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--gains", type=str, default="0.0,0.5,1.0")
parser.add_argument("--phis", type=str, default="90,80,70")
parser.add_argument("--pose_reps", type=int, default=3)
parser.add_argument("--confirm", type=int, default=1)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p24_yaw.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from _kin import Q_CLOSE, Q_OPEN, _t  # noqa: E402
from clutter_expert import ClutterExpert, DIST, HY  # noqa: E402

ROW_Y = (-0.084, -0.042, 0.042, 0.084)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    dev, n = e.device, e.num_envs
    org = e.scene.env_origins
    gains = [float(x) for x in args_cli.gains.split(",")]
    phis = [float(x) for x in args_cli.phis.split(",")]

    print("\n" + "=" * 104)
    print("P24 -- YAW GAIN x PHI ON THE ISOLATED CLOSE")
    print("=" * 104)
    print("   blade: +/-19.2 mm perpendicular to the opening axis, ~47 mm along it")
    print("   margin to a neighbour's face: 27 - 19.2 = 7.8 mm")
    print("   corner swing at full yaw match: 47 * sin(11.4 deg) = 9.3 mm   -> predicted to "
          "intrude")

    env.reset()
    for _ in range(30):
        e.sim.step()
        e.scene.update(e.physics_dt)
    snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}
    tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
    dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
    ys = torch.cat([dpos0[:, :2, 1], tpos0[:, 1:2], dpos0[:, 2:, 1]], dim=1)
    ys, _ = ys.sort(dim=1)
    min_gap = (ys[:, 1:] - ys[:, :-1] - 2 * HY).min(dim=1).values
    print(f"   spawn: min free gap median {float(min_gap.median()) * 1000:.1f} mm\n")

    def restore():
        for k, v in snap.items():
            e.scene[k].write_root_state_to_sim(v.clone())
        e.sim.forward()
        e.scene.update(e.physics_dt)

    print(f"   {'phi':>4} | {'gain':>5} | {'draw':>4} | {'o_align':>8} | {'moved':>7} | "
          f"{'toppled':>8} | {'stall':>7} | victims")
    print("   " + "-" * 96)
    rows = []
    for phi in phis:
        for g in gains:
            for pr in range(args_cli.pose_reps):
                restore()                       # spawn back BEFORE adapt reads the target
                ex = ClutterExpert(env, grip_z=args_cli.grip_z, phi=phi,
                                   verbose=False, plan_full=False)
                K = ex.K
                q = ex.adapt(yaw_gain=g)[0]
                restore()
                K.teleport_arm(q, Q_OPEN)
                for _ in range(160):
                    K.robot.set_joint_position_target(K._drive(q, Q_OPEN))
                    K.robot.write_data_to_sim()
                    e.sim.step()
                    e.scene.update(e.physics_dt)
                pre = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST],
                                  dim=1).clone()
                for _ in range(560):
                    K.robot.set_joint_position_target(K._drive(q, Q_CLOSE))
                    K.robot.write_data_to_sim()
                    e.sim.step()
                    e.scene.update(e.physics_dt)
                post = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST],
                                   dim=1)
                up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
                stall = K.gap().clone()
                disp = (post[:, :, :2] - pre[:, :, :2]).norm(dim=2)
                moved = float((disp > 0.0015).any(dim=1).float().mean())
                topp = float((up < mdp_cl.TOPPLE_DOT).any(dim=1).float().mean())
                per = [float((up[:, k] < mdp_cl.TOPPLE_DOT).float().mean()) for k in range(4)]
                held = float(((stall - ex.width).abs() < 0.012).float().mean())
                print(f"   {phi:4.0f} | {g:5.2f} | {pr:4d} | {ex.pose['o_align']:8.4f} | "
                      f"{moved:6.1%} | {topp:7.1%} | {float(stall.median()) * 1000:5.1f}mm | "
                      + " ".join(f"d{k}:{per[k]:.0%}" for k in range(4)))
                rows.append({"phi": phi, "gain": g, "pose_rep": pr,
                             "o_align": ex.pose["o_align"], "moved": moved, "topple": topp,
                             "held": held, "stall_mm": float(stall.median()) * 1000,
                             "per_block": per})

    print("\n" + "=" * 104)
    print("   ISOLATED-CLOSE TOPPLE, MEAN OVER POSE DRAWS")
    print(f"   {'phi \\ gain':>12} | " + " | ".join(f"{g:>10.2f}" for g in gains))
    agg = {}
    for phi in phis:
        cells = []
        for g in gains:
            r = [x for x in rows if x["phi"] == phi and x["gain"] == g]
            v = sum(x["topple"] for x in r) / len(r)
            agg[(phi, g)] = v
            cells.append(f"{v:10.1%}")
        print(f"   {phi:12.0f} | " + " | ".join(cells))
    print(f"\n   {'phi \\ gain':>12} | " + " | ".join(f"{g:>10.2f}" for g in gains)
          + "     (enclosure)")
    for phi in phis:
        cells = []
        for g in gains:
            r = [x for x in rows if x["phi"] == phi and x["gain"] == g]
            cells.append(f"{sum(x['held'] for x in r) / len(r):10.1%}")
        print(f"   {phi:12.0f} | " + " | ".join(cells))

    best = min(agg, key=agg.get)
    print(f"\n   lowest isolated topple: phi = {best[0]:.0f}, yaw_gain = {best[1]:.2f} "
          f"({agg[best]:.1%})")
    cur = agg.get((90.0, 1.0))
    if cur is not None:
        print(f"   the expert currently ships phi = 90, yaw_gain = 1.00 ({cur:.1%})")

    # ---- does it transfer? enclosure is the constraint the isolated test cannot see fully
    conf = []
    if args_cli.confirm:
        print("\n" + "=" * 104)
        print("   FULL-TRAJECTORY CONFIRMATION")
        cands = sorted({best, (90.0, 1.0)})
        for phi, g in cands:
            env.reset()
            for _ in range(30):
                e.sim.step()
                e.scene.update(e.physics_dt)
            ex = ClutterExpert(env, grip_z=args_cli.grip_z, phi=phi, verbose=False)
            for rep in range(2):
                env.reset()
                for _ in range(30):
                    e.sim.step()
                    e.scene.update(e.physics_dt)
                r = ex.run_physics(ex.adapt(yaw_gain=g))
                print(f"      phi {phi:.0f}, gain {g:.2f}, batch {rep}: enclosed "
                      f"{float(r['held'].float().mean()):6.1%} | at goal "
                      f"{float(r['at_goal'].float().mean()):6.1%} | topple "
                      f"{float(r['topple'].float().mean()):6.1%} | SUCCESS "
                      f"{float(r['success'].float().mean()):6.1%}")
                conf.append({"phi": phi, "gain": g, "rep": rep,
                             "encl": float(r["held"].float().mean()),
                             "at_goal": float(r["at_goal"].float().mean()),
                             "topple": float(r["topple"].float().mean()),
                             "success": float(r["success"].float().mean())})

    out = {"n": n, "grip_z": args_cli.grip_z, "gains": gains, "phis": phis,
           "min_gap_mm": (min_gap * 1000).tolist(), "rows": rows,
           "agg": {f"{k[0]}_{k[1]}": v for k, v in agg.items()},
           "best": list(best), "confirm": conf}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p24] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
