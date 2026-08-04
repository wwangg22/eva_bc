# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P11 -- The orthogonal (front-back) grasp, tested with a WORKING instrument.

Why this exists
---------------
Big Will asked whether the gripper had been tried **orthogonal to the block** -- rotated so the
fingers straddle the target fore and aft rather than across the row -- approaching from above,
clear of the neighbours, then descending. Checking the record honestly: **it had not.**

* `p02_orientation_envelope.py` swept exactly this family ("G2", `o_des = x_hat`) and returned
  0 of 48 cells attainable. That probe is marked **void** in `07_STAGE0_RESULTS.md` S3: it ran
  on a CEM with three defects (weighted-sum cost that let orientation outrank position, a floor
  penalty made inert by `base_link` sitting at z = 0, and no restarts from the folded home
  pose). It was never re-run after those were fixed.
* `p04_reach_global.py` reported "G2: 0 samples", but only **3** uniform draws out of 819 200
  landed within 10 mm of the grasp point at all -- a density artifact, flagged as such at the
  time, not evidence.
* Every probe that produced a number -- P03, P05, P06, P08, P10 -- used `o_des = y_hat`, i.e.
  fingers at +/-44.5 mm **across** the row. That is exactly the configuration that lands them
  on d1/d2's tops.

The geometric case for the orthogonal grasp is strong. With `o_hat = x_hat` the fingers sit at
x ~ 0.205 and ~0.295 -- in front of and behind the row -- and never enter a 12 mm gap. The
block is 36 mm deep in x, comfortably inside the 89 mm opening. The row's y-pitch, which is the
entire difficulty of the task, stops being the binding constraint.

The objection I previously raised against it does not survive contact with P05's own data. A
parallel jaw's opening axis is perpendicular to its approach axis, so `o_hat = x_hat` forces
`a_hat` into the y-z plane, and C1 says a downward *approach axis* is unavailable at table
height. But the validated grasp already **descends vertically while `a_hat = (-0.75, +0.26,
+0.61)`** -- direction of travel and approach-axis orientation are independent. C1 constrains
the latter; the descent uses the former.

What this probe does
--------------------
1. Sweeps the opening axis from `y_hat` (phi = 0, the cross-row grasp that fails) to `x_hat`
   (phi = 90, orthogonal), at several grip heights, using the **corrected** CEM: hinge cost,
   floor over gripper bodies only, multi-restart. Reports achieved alignment and the finger
   positions in x and y, so "does the jaw sit clear of the row" is read rather than argued.
2. For every attainable orientation, runs the full validated recipe -- vertical descent from
   above the block tops, whole chain solved **before** the fingers close -- against the real
   nominal row, stepping **physics only** so a topple cannot reset the scene and hide itself.
3. Scores with the enclosure check against the correct face width: an orthogonal grasp closes
   on the block's **36 mm** x-faces, not its 30 mm y-faces. Using the wrong width here would
   silently mark every real grasp as a failure.

Usage
-----
    python eva_bc/clutter/probes/p11_orthogonal_grasp.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Orthogonal (front-back) grasp of the target.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=70)
parser.add_argument("--restarts", type=int, default=12)
parser.add_argument("--standoff", type=float, default=0.075)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p11_orthogonal.json")
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
BLOCK_W_Y = 2 * HY          # 30 mm -- the faces a CROSS-ROW grasp closes on
BLOCK_W_X = 2 * HX          # 36 mm -- the faces an ORTHOGONAL grasp closes on
ROW_PITCH = 0.042
DIST = mdp_cl.DISTRACTOR_NAMES
NOM_Y = (-2 * ROW_PITCH, -ROW_PITCH, ROW_PITCH, 2 * ROW_PITCH)

GRIP_ZS = (0.045, 0.055, 0.065)
PHIS = (0.0, 45.0, 70.0, 80.0, 90.0)
#: a pose is attainable only if all of these hold
OK_POS, OK_O, OK_FLOOR = 0.003, 0.95, 0.010


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

    print("\n" + "=" * 104)
    print("P11 -- ORTHOGONAL (FRONT-BACK) GRASP, corrected CEM")
    print("=" * 104)
    print("   phi = 0  -> opening axis along y  (across the row; fingers land on d1/d2 tops)")
    print("   phi = 90 -> opening axis along x  (fore/aft; fingers clear the row entirely)")
    print(f"   block is {BLOCK_W_Y * 1000:.0f} mm across y and {BLOCK_W_X * 1000:.0f} mm across x;"
          f" the enclosure check uses the width that matches the orientation")
    print()

    # ------------------------------------------------------- 1. attainability, corrected CEM
    hdr = (f"   {'z[mm]':>6} {'phi':>4} | {'pos err':>8} | {'o_align':>7} | {'low_z':>7} | "
           f"{'fL x,y':>14} | {'fR x,y':>14} | {'clear row?':>10} | {'OK':>4}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))

    def row_clear(fx, fy):
        """Is a finger at (x, y) outside every distractor's footprint?"""
        for y0 in NOM_Y:
            if abs(fy - y0) < HY and abs(fx - ROW_X) < HX:
                return False
        return True

    scene()
    cells = []
    for gz in GRIP_ZS:
        pos = torch.tensor([ROW_X, 0.0, gz], device=dev)
        for phi in PHIS:
            p = math.radians(phi)
            o_des = torch.tensor([math.sin(p), math.cos(p), 0.0], device=dev)
            r = K.cem(pos, K.q_arm0, o_des=o_des, w_o=0.60, iters=args_cli.iters,
                      std0=0.6, restarts=args_cli.restarts, floor_z=0.012, w_floor=20.0)
            K.fk(r["q"].unsqueeze(0).repeat(n, 1))
            fl, fr = K.finger_pos()
            fl, fr = fl[0].cpu().numpy(), fr[0].cpu().numpy()
            clear = row_clear(fl[0], fl[1]) and row_clear(fr[0], fr[1])
            ok = (r["pos_err"] < OK_POS and r["o_align"] > OK_O and r["low_z"] > OK_FLOOR)
            cells.append({"grip_z": gz, "phi": phi, "pos_err_m": r["pos_err"],
                          "o_align": r["o_align"], "low_z_m": r["low_z"],
                          "finger_l": fl.tolist(), "finger_r": fr.tolist(),
                          "row_clear": bool(clear), "attainable": bool(ok),
                          "q": [float(v) for v in r["q"]],
                          "a_hat": [float(v) for v in r["a_hat"]]})
            print(f"   {gz * 1000:6.0f} {phi:4.0f} | {r['pos_err'] * 1000:7.2f}mm | "
                  f"{r['o_align']:7.3f} | {r['low_z'] * 1000:6.1f}mm | "
                  f"{fl[0] * 1000:6.1f},{fl[1] * 1000:6.1f} | "
                  f"{fr[0] * 1000:6.1f},{fr[1] * 1000:6.1f} | "
                  f"{'YES' if clear else 'no':>10} | {'YES' if ok else '.':>4}")
        print("   " + "-" * (len(hdr) - 3))

    # ------------------------------------------------------------- 2. execute what is viable
    viable = [c for c in cells if c["attainable"] and c["phi"] >= 45.0]
    print(f"\n   attainable cells with phi >= 45 deg: {len(viable)}")
    if not viable:
        print("   No orthogonal-ish orientation is attainable even with the corrected CEM.")
        print("   -> The arm cannot hold the opening axis away from y at this point. That is")
        print("      now a measured kinematic fact rather than a search artifact.")

    results = []
    for c in sorted(viable, key=lambda c: (-c["phi"], -c["o_align"]))[:6]:
        gz, phi = c["grip_z"], c["phi"]
        # the width the fingers will actually close on, given how far the jaw is rotated
        wid = BLOCK_W_X if phi >= 45.0 else BLOCK_W_Y
        q_grip = torch.tensor(c["q"], device=dev)
        p = math.radians(phi)
        o_des = torch.tensor([math.sin(p), math.cos(p), 0.0], device=dev)
        grip = torch.tensor([ROW_X, 0.0, gz], device=dev)
        down = torch.tensor([0.0, 0.0, -1.0], device=dev)

        # whole chain solved BEFORE anything closes (07_STAGE0_RESULTS.md 4.3)
        qq, appr = q_grip, []
        for t in lerp_pts(grip, grip - args_cli.standoff * down, 3):
            qq = K.cem(t, qq, o_des=o_des, w_o=0.60, iters=args_cli.iters, std0=0.15,
                       floor_z=0.012, w_floor=20.0)["q"]
            appr.append(qq)
        approach = list(reversed(appr)) + [q_grip]
        qq, up_seq = q_grip, []
        for t in lerp_pts(grip, grip + torch.tensor([0.0, 0.0, 0.09], device=dev), 3):
            qq = K.cem(t, qq, o_des=o_des, w_o=0.60, iters=args_cli.iters, std0=0.15,
                       floor_z=0.012, w_floor=20.0)["q"]
            up_seq.append(qq)

        # physics-only execution: env.step would reset on topple and hide it (S10.1)
        scene()
        K.teleport_arm(approach[0].unsqueeze(0).repeat(n, 1), Q_OPEN)
        K.hold_phys(approach[0].unsqueeze(0).repeat(n, 1), 80)
        for i in range(len(approach) - 1):
            K.run_phys(approach[i].unsqueeze(0).repeat(n, 1),
                       approach[i + 1].unsqueeze(0).repeat(n, 1), 25)
        K.hold_phys(q_grip.unsqueeze(0).repeat(n, 1), 160)
        tcp_grasp = K.tcp_now()[0].clone()
        K.hold_phys(q_grip.unsqueeze(0).repeat(n, 1), 560, q_fing=Q_CLOSE)
        gap_stall = K.gap().clone()
        seq = [q_grip] + up_seq
        for i in range(len(seq) - 1):
            K.run_phys(seq[i].unsqueeze(0).repeat(n, 1), seq[i + 1].unsqueeze(0).repeat(n, 1),
                       25, q_fing=Q_CLOSE)
        K.hold_phys(up_seq[-1].unsqueeze(0).repeat(n, 1), 240, q_fing=Q_CLOSE)

        bpos = e.scene["target"].data.root_pos_w.torch - org
        tcp = K.tcp_now()
        encl = (gap_stall - wid).abs() < 0.012
        rose = bpos[:, 2] > HZ + 0.045
        near = (tcp - bpos).norm(dim=1) < 0.09
        held = rose & near & encl
        up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
        topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)
        blocked = (tcp_grasp[2] - gz) > 0.004

        rec = {"grip_z": gz, "phi": phi, "width_used_m": wid,
               "tcp_z_at_grasp_mm": float(tcp_grasp[2]) * 1000, "blocked": bool(blocked),
               "gap_stall_mm": float(gap_stall.median()) * 1000,
               "encl": float(encl.float().mean()), "rose": float(rose.float().mean()),
               "held": float(held.float().mean()), "topple": float(topp.float().mean()),
               "min_up_z": float(up.min())}
        results.append(rec)
        print(f"\n   EXEC z={gz * 1000:.0f} phi={phi:.0f}  (closing on {wid * 1000:.0f} mm faces)")
        print(f"        TCP z at grasp {float(tcp_grasp[2]) * 1000:6.2f} mm  "
              f"{'BLOCKED' if blocked else 'reached'}   stall gap "
              f"{float(gap_stall.median()) * 1000:6.2f} mm")
        print(f"        encl {rec['encl']:4.0%}  rose {rec['rose']:4.0%}  "
              f"HELD {rec['held']:4.0%}  topple {rec['topple']:4.0%}  "
              f"min up_z {rec['min_up_z']:.3f}")

    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    best = max(results, key=lambda r: r["held"]) if results else None
    if best and best["held"] > 0.5 and best["topple"] < 0.1:
        print(f"   ORTHOGONAL GRASP WORKS: HELD {best['held']:.0%} at phi = {best['phi']:.0f} deg,"
              f" z = {best['grip_z'] * 1000:.0f} mm, topple {best['topple']:.0%}.")
        print("   The row's y-pitch is NOT the binding constraint after all -- rotating the")
        print("   jaw sidesteps it entirely. Gate 0 PASSES and Stage 1 has its primitive.")
    elif best and best["held"] > 0.0:
        print(f"   Partial: best HELD {best['held']:.0%} at phi = {best['phi']:.0f} deg "
              f"(topple {best['topple']:.0%}). Worth tuning before concluding either way.")
    elif results:
        print("   Attainable, but no orthogonal orientation completes a grasp of the target.")
        print("   Check `blocked` and the stall gap above to see whether the descent was")
        print("   stopped or the jaw simply missed.")
    else:
        print("   Nothing executable to report -- see section 1.")

    out = {"cells": cells, "results": results,
           "block_w_y_m": BLOCK_W_Y, "block_w_x_m": BLOCK_W_X}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p11] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
