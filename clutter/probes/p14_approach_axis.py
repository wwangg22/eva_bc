# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P14 -- Where does the rest of the arm go? Approach-axis sweep with whole-arm clearance.

The defect this probe exists to close
-------------------------------------
Every pose used from P11 through P13 was selected on two criteria: TCP error < 1.5 mm and
maximum `o_align`. Those two pin down the **tool frame** and nothing else. P13 then drew a
pose scoring `pos_err = 0.55 mm, o_align = 1.000` -- perfect on both -- for which enclosure
collapsed from 100 % to 32 % and 63 of 128 envs made contact with a distractor during the
*descent*. Its approach axis was `a_hat = (+0.03, -0.94, +0.33)`: almost horizontal along
-y.

That single number decides everything, because the geometry is forced. With `o_hat = x_hat`
the approach axis is confined to the y-z plane, `a_hat = (0, sin t, cos t)`, and the wrist
stub sits a fixed 41.9 mm behind the TCP along it:

    gripper_end = TCP - 0.0419 * a_hat = (250, -41.9 sin t, 65 - 41.9 cos t)   [mm]

    t = -70 deg  (P13)        wrist at y = +39, z = 51   -> INSIDE distractor_2
    t = -22 deg  (P12 run 2)  wrist at y = +15, z = 26   -> on the target's +y face, marginal
    t =   0 deg               wrist at y =   0, z = 23   -> inside the TARGET block
    t = 180 deg  (top-down)   wrist at y =   0, z = 107  -> clear of everything

The free gaps either side of the target are 12 mm wide. There is no value of `sin t` that
places a 40 mm-scale wrist stub inside one reliably. **The only wrist placements that clear
the row are the ones with `cos t < 0` -- an approach axis pointing downward.** Which is
exactly the family `CHALLENGE_SUITE` C1 reports as unavailable ("0.00 % top-down-capable
voxels at table height").

So this probe asks the question directly, at the pose that actually matters rather than over
a generic voxel grid: **sweep t through the full circle and measure what is attainable, and
where the whole arm ends up.** It reports, per t:

    * whether the TCP is reachable at all with `o_hat = x_hat`
    * the achieved `o_align` and `a_align`
    * the wrist stub's position
    * the deepest penetration of ANY robot body into ANY distractor's volume

The last column is the one that was missing. It is computed from body origins, so it
understates true penetration -- a body's shell reaches beyond its origin -- which makes a
reported penetration of zero a necessary condition, not a sufficient one. Execution still
decides; this only stops the search proposing poses that were never going to work.

Usage
-----
    python eva_bc/clutter/probes/p14_approach_axis.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Approach-axis sweep with whole-arm clearance.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=14)
parser.add_argument("--tries", type=int, default=3)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--step", type=float, default=15.0, help="theta step [deg]")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p14_axis.json")
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
from _kin import ArmKin  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
ROW_Y = (-0.084, -0.042, 0.042, 0.084)


def distractor_boxes(dev, z_c: float = 0.032):
    """Keep-out volumes for the four neighbours at nominal spawn, `[c, h]` rows."""
    return torch.tensor([[ROW_X, y, z_c, HX, HY, HZ] for y in ROW_Y], device=dev)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    K = ArmKin(env)
    dev, n = K.dev, K.n
    names = list(K.robot.body_names)
    boxes = distractor_boxes(dev)

    _p = math.radians(args_cli.phi)
    O = torch.tensor([math.sin(_p), math.cos(_p), 0.0], device=dev)
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)

    print("\n" + "=" * 108)
    print("P14 -- APPROACH-AXIS SWEEP, WHOLE-ARM CLEARANCE")
    print("=" * 108)
    print(f"   o_hat = x_hat (phi = {args_cli.phi:.0f}), TCP = "
          f"({ROW_X * 1000:.0f}, 0, {args_cli.grip_z * 1000:.0f}) mm")
    print("   a_hat = (0, sin t, cos t);  t = 0 is straight UP, t = 180 is straight DOWN")
    print("   penetration = deepest intrusion of any BODY ORIGIN into any distractor volume")
    print("   (origins understate real penetration: 0 is necessary, not sufficient)\n")
    print(f"   {'t':>5} | {'a_hat':>20} | {'err':>6} | {'o_al':>5} | {'a_al':>5} | "
          f"{'wrist x,y,z [mm]':>22} | {'pen':>6} | worst body")
    print("   " + "-" * 104)

    rows = []
    for deg in range(0, 360, int(args_cli.step)):
        t = math.radians(deg)
        a_des = torch.tensor([0.0, math.sin(t), math.cos(t)], device=dev)
        best = None
        for _ in range(args_cli.tries):
            c = K.cem(grip, K.q_arm0, o_des=O, a_des=a_des, w_o=0.60, w_a=0.60,
                      iters=args_cli.iters, std0=0.6, restarts=args_cli.restarts)
            if best is None or (c["pos_err"], -c["a_align"]) < (best["pos_err"], -best["a_align"]):
                best = c
        # achieved geometry of the winning pose, all bodies
        g = K.fk(best["q"].unsqueeze(0).repeat(n, 1))
        bod = g["bodies"][0]                                        # (B,3)
        pen_body = torch.stack([
            K.box_penetration(bod[b].view(1, 1, 3), boxes, 0.0)[0] for b in range(len(names))
        ])
        pen = float(pen_body.max())
        worst = names[int(pen_body.argmax())] if pen > 0 else "-"
        wr = bod[K.i_end]
        ok = best["pos_err"] < 0.0015 and best["o_align"] > 0.97
        flag = "" if ok else "   (unattainable)"
        print(f"   {deg:5d} | ({0.0:+.2f},{math.sin(t):+.2f},{math.cos(t):+.2f})".ljust(32)
              + f"| {best['pos_err'] * 1000:5.2f} | {best['o_align']:5.3f} | "
                f"{best['a_align']:+5.2f} | "
                f"{float(wr[0]) * 1000:6.1f},{float(wr[1]) * 1000:+6.1f},"
                f"{float(wr[2]) * 1000:6.1f} | {pen * 1000:5.1f} | {worst}{flag}")
        rows.append({"deg": deg, "pos_err_mm": best["pos_err"] * 1000,
                     "o_align": best["o_align"], "a_align": best["a_align"],
                     "wrist_mm": (wr * 1000).tolist(), "pen_mm": pen * 1000,
                     "worst_body": worst, "attainable": bool(ok),
                     "q": best["q"].tolist()})

    good = [r for r in rows if r["attainable"] and r["pen_mm"] <= 0.01]
    print("\n" + "=" * 108)
    if good:
        print(f"   {len(good)} of {len(rows)} approach axes are BOTH attainable and "
              f"row-clear (by body origin):")
        for r in sorted(good, key=lambda x: -x["o_align"]):
            print(f"      t = {r['deg']:3d} deg | err {r['pos_err_mm']:.2f} mm | "
                  f"o_align {r['o_align']:.3f} | wrist z "
                  f"{r['wrist_mm'][2]:.1f} mm, y {r['wrist_mm'][1]:+.1f} mm")
        print("\n   -> C1's 'no top-down approach' does NOT close this task: the axis only has")
        print("      to clear the row, and it is measured here at the pose that matters.")
    else:
        att = [r for r in rows if r["attainable"]]
        print(f"   NO approach axis is both attainable and row-clear. {len(att)} of "
              f"{len(rows)} are attainable at all.")
        if att:
            b = min(att, key=lambda x: x["pen_mm"])
            print(f"   Shallowest intrusion: t = {b['deg']} deg, {b['pen_mm']:.1f} mm into "
                  f"{b['worst_body']}, wrist at "
                  f"({b['wrist_mm'][0]:.0f},{b['wrist_mm'][1]:+.0f},{b['wrist_mm'][2]:.0f}) mm")
        print("   -> The wrist stub cannot be kept out of the neighbours at this grip height.")
        print("      Next lever is grip_z: raising the grip raises the whole tool frame.")

    # --------- the same sweep is much cheaper to reason about as a table of wrist heights
    print("\n   predicted vs achieved wrist placement (sanity check on the 41.9 mm offset):")
    print(f"      {'t':>5} | {'pred y':>7} | {'meas y':>7} | {'pred z':>7} | {'meas z':>7}")
    for r in rows[::3]:
        t = math.radians(r["deg"])
        py, pz = -41.9 * math.sin(t), args_cli.grip_z * 1000 - 41.9 * math.cos(t)
        print(f"      {r['deg']:5d} | {py:+7.1f} | {r['wrist_mm'][1]:+7.1f} | "
              f"{pz:7.1f} | {r['wrist_mm'][2]:7.1f}")

    out = {"grip_z": args_cli.grip_z, "phi": args_cli.phi, "rows": rows,
           "n_attainable": sum(r["attainable"] for r in rows), "n_clear": len(good)}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p14] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
