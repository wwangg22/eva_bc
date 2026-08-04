# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P03 -- CONTROL: can this code grasp the target with the clutter taken away?

Why a control comes before any clutter measurement
--------------------------------------------------
P02's kinematic gate rejected all 48 orientation cells, with achieved `a_align` **negative**
throughout -- the approach axis pointing back toward the robot instead of at the block. Two
explanations fit that equally well:

  (i)  the arm genuinely cannot hold a grasp orientation at the row, or
  (ii) `ArmKin` has a convention bug, or its CEM never escapes the folded home pose.

Those have opposite consequences and the difference is not visible from the numbers alone.
So: take the clutter away and grasp the target on its own. eva_rl already measured that this
gripper picks up free-standing blocks (`grasp_geometry.py`), so a failure here is a bug in
**our** code, and a success validates the whole convention chain -- TCP offset, approach axis,
opening axis, action encoding, waypoint execution -- before any of it is used to make a claim
about the task.

This is eva_bc's house rule applied to a probe rather than a wrapper: *gate every new
component on bit-exact reproduction of a known-good result before trusting it on the hard
case.* A probe that has never reproduced a positive control cannot produce a credible
negative.

Three conditions, same code path, same grip heights:

  ============  ===========================================================
  ``solo``      distractors parked far away; target alone. **Must succeed.**
  ``clutter``   distractors at their nominal row positions. The real task.
  ``gap``       d1/d2 pushed 25 mm outward by hand (privileged), i.e. what a
                perfect singulation would leave behind. Bounds strategy B.
  ============  ===========================================================

The `clutter` - `solo` difference is exactly what the row costs. The `gap` condition says
whether singulating first would be enough, without needing a working push yet.

Success is `grasp_geometry.py:207-211`'s enclosure check, which eva_rl's handoff says
"cannot be faked": the clear finger gap must end up near the block width (fingers resting on
its faces), the block must have risen, and the TCP must still be near it. Fingers that closed
on air go to ~0 mm; fingers jammed outside the block sit near 89 mm; a block merely shoved
along scores `rose = False`.

Usage
-----
    python eva_bc/clutter/probes/p03_grasp_control.py --num_envs 64
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Control grasp with and without the clutter.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=6,
                    help="independent CEM restarts; >1 is what escapes the home-pose basin")
parser.add_argument("--conds", type=str, default="solo,clutter,gap")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p03_grasp_control.json")
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
BLOCK_W = 2 * HY  # 30 mm across the fingers
DIST = mdp_cl.DISTRACTOR_NAMES
GRIP_ZS = (0.045, 0.055, 0.065)
STANDOFF = 0.070
#: keep every gripper body this far above the table during the CEM search
FLOOR_Z, W_FLOOR = 0.012, 5.0
#: Approach tilts below horizontal [deg]. The sign convention is read off P04's measurement
#: of the home pose: `a_hat . x = -0.98`, i.e. this arm's fingers point BACK toward the robot,
#: and it grasps by reaching past an object and closing as it comes back in -x. So the family
#: is `a_hat = (-cos g, 0, -sin g)`; g > 0 lifts the wrist ABOVE the grasp point, which is
#: what keeps `gripper_end` out of the table.
APPROACH_TILTS = (0.0, 15.0, 30.0, 45.0)
#: somewhere on the table, well clear of the arm's working volume
PARK = ((0.45, 0.30), (0.45, -0.30), (0.62, 0.30), (0.62, -0.30))
#: what a perfect singulation would leave: d1/d2 shoved 25 mm further out
GAP_PUSH = 0.025


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5  # no timeouts during a scripted trial
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    K = ArmKin(env)
    dev, n = K.dev, K.n
    org = e.scene.env_origins
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
    Y = torch.tensor([0.0, 1.0, 0.0], device=dev)

    out: dict = {"task": args_cli.task, "num_envs": n, "grip_zs": list(GRIP_ZS)}
    print("\n" + "=" * 92)
    print("P03 -- CONTROL GRASP: solo vs clutter vs pre-singulated")
    print("=" * 92)

    def put(name: str, x: float, y: float, z: float = HZ):
        p = torch.tensor([x, y, z], device=dev).repeat(n, 1) + org
        e.scene[name].write_root_state_to_sim(
            torch.cat([p, ident, torch.zeros((n, 6), device=dev)], dim=1))

    def scene_for(cond: str):
        put("target", ROW_X, 0.0)
        for i, d in enumerate(DIST):
            if cond == "solo":
                put(d, PARK[i][0], PARK[i][1])
            elif cond == "gap":
                s = (-1, -1, 1, 1)[i]
                extra = GAP_PUSH if i in (1, 2) else 2 * GAP_PUSH
                put(d, ROW_X, (-2, -1, 1, 2)[i] * ROW_PITCH + s * extra)
            else:
                put(d, ROW_X, (-2, -1, 1, 2)[i] * ROW_PITCH)
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def solve(t, seed, a_des, std0):
        """One CEM solve with the constraint set that P04 says this arm actually admits.

        `w_pos = 1.0` is `grasp_geometry.py`'s proven scaling, not the 20.0 that P02 needed:
        there the orientation was possibly unattainable and had to be prevented from dragging
        the search away; here the orientation is one the arm demonstrably holds, so letting it
        outweigh a millimetre is what puts the search in the right basin.

        `floor_z`/`w_floor` are the fix for P03's first failure: with no floor term the CEM
        returned poses with `gripper_end` at z = 17 mm, i.e. below the table, and the fingers
        closed on air in every condition including the control.
        Position enters `cem` as a hinge with `pos_tol = 1 mm`, so the orientation terms can
        buy a few millimetres of slack and no more -- see `_kin.ArmKin.cem`. `restarts > 1`
        is what lets the search leave this arm's folded home-pose basin.
        """
        return K.cem(t, seed, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
                     iters=args_cli.iters, std0=std0, floor_z=FLOOR_Z, w_floor=W_FLOOR,
                     restarts=args_cli.restarts)

    def path(targets, seed, a_des, std0):
        qs, q, r = [], seed, None
        for i, t in enumerate(targets):
            r = solve(t, q, a_des, std0 if i == 0 else 0.12)
            q = r["q"]
            qs.append(q)
        return qs, r

    rows = []
    for cond in args_cli.conds.split(","):
        print("\n" + "-" * 92)
        print(f"CONDITION: {cond}")
        print("-" * 92)
        for gz, td in [(z, t) for z in GRIP_ZS for t in APPROACH_TILTS]:
            # --- plan (CEM runs with the arm teleporting through everything; blocks are
            #     re-placed afterwards, exactly as grasp_geometry.py does)
            scene_for(cond)
            g = math.radians(td)
            a_des = torch.tensor([-math.cos(g), 0.0, -math.sin(g)], device=dev)
            grip = torch.tensor([ROW_X, 0.0, gz], device=dev)
            r_grip = solve(grip, K.q_arm0, a_des, 0.6)
            q_grip = r_grip["q"]
            # VERTICAL descent, not a standoff along the jaw axis. P05 traced both: the
            # jaw-axis standoff lands at x ~ 0.31, z ~ 0.055 -- C9's documented unusable
            # region -- where tracking error ran to 211 mrad and the arm ploughed the block
            # 76 mm out of place before the fingers ever closed. Descending from directly
            # above holds the wrist orientation fixed, never leaves the reliable x band, and
            # left the block undisturbed to within 0.0 mm through the whole approach.
            f = torch.tensor([0.0, 0.0, -1.0], device=dev)
            q_back, _ = path(lerp_pts(grip, grip - STANDOFF * f, 3), q_grip, a_des, 0.45)
            approach = list(reversed(q_back)) + [q_grip]
            # PLAN THE LIFT BEFORE ANYTHING CLOSES: `cem` -> `fk` -> `write_joint_state_to_sim`
            # with the fingers OPEN, hundreds of times. Searching after the close silently
            # drops the block (measured: clean 30.0 mm grasp, then gap -> -1.2 mm on the lift).
            q_up, _ = path(lerp_pts(grip, grip + torch.tensor([0.0, 0.0, 0.09], device=dev), 3),
                           q_grip, a_des, 0.45)

            # --- execute
            scene_for(cond)
            K.teleport_arm(approach[0].unsqueeze(0).repeat(n, 1), Q_OPEN)
            z0 = HZ
            up0 = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)

            K.hold(approach[0].unsqueeze(0).repeat(n, 1), 15, close=False)
            for i in range(len(approach) - 1):
                K.run(approach[i].unsqueeze(0).repeat(n, 1),
                      approach[i + 1].unsqueeze(0).repeat(n, 1), 25, close=False)
            K.hold(q_grip.unsqueeze(0).repeat(n, 1), 70, close=True)
            seq = [q_grip] + q_up
            for i in range(len(seq) - 1):
                K.run(seq[i].unsqueeze(0).repeat(n, 1), seq[i + 1].unsqueeze(0).repeat(n, 1),
                      25, close=True)
            K.hold(q_up[-1].unsqueeze(0).repeat(n, 1), 50, close=True)

            # --- score (grasp_geometry.py:207-211, the check eva_rl says cannot be faked)
            bpos = e.scene["target"].data.root_pos_w.torch - org
            tcp = K.tcp_now()
            gap = K.gap()
            encl = (gap - BLOCK_W).abs() < 0.012
            rose = bpos[:, 2] > z0 + 0.045
            near = (tcp - bpos).norm(dim=1) < 0.09
            held = rose & near & encl
            up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
            topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)

            rec = {"cond": cond, "grip_z": gz, "tilt_deg": td, "cem_err_m": r_grip["pos_err"],
                   "a_hat": [float(v) for v in r_grip["a_hat"]],
                   "o_align": r_grip["o_align"], "a_align": r_grip["a_align"],
                   "grip_low_z_m": r_grip["low_z"],
                   "gap_med_m": float(gap.median()),
                   "encl_rate": float(encl.float().mean()),
                   "rose_rate": float(rose.float().mean()),
                   "held_rate": float(held.float().mean()),
                   "topple_rate": float(topp.float().mean()),
                   "up0_min": float(up0.min()), "up_min": float(up.min())}
            rows.append(rec)
            print(f"  z={gz * 1000:3.0f} tilt={td:4.0f}  CEM {r_grip['pos_err'] * 1000:5.2f}mm "
                  f"a=({r_grip['a_hat'][0]:+.2f},{r_grip['a_hat'][1]:+.2f},"
                  f"{r_grip['a_hat'][2]:+.2f}) oA={r_grip['o_align']:.2f} "
                  f"lowz={r_grip['low_z'] * 1000:5.1f} | gap {float(gap.median()) * 1000:5.1f} "
                  f"encl {rec['encl_rate']:4.0%} rose {rec['rose_rate']:4.0%} "
                  f"HELD {rec['held_rate']:4.0%} topp {rec['topple_rate']:4.0%}")
    out["rows"] = rows

    # -------------------------------------------------------------------------- verdict
    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    best = {}
    for cond in args_cli.conds.split(","):
        sub = [r for r in rows if r["cond"] == cond]
        if not sub:
            continue
        b = max(sub, key=lambda r: r["held_rate"])
        best[cond] = b
        print(f"   {cond:>8}: best HELD {b['held_rate']:5.0%} at z = {b['grip_z'] * 1000:.0f} mm"
              f"   (topple {b['topple_rate']:.0%})")
    out["best"] = best

    print()
    if "solo" in best:
        if best["solo"]["held_rate"] < 0.5:
            print("   [p03] THE CONTROL FAILED. With the clutter removed entirely, this code")
            print("         cannot pick up a free-standing block that eva_rl has already shown")
            print("         to be graspable. Every negative result from P02 is therefore")
            print("         uninterpretable -- fix the primitive before measuring the task.")
        else:
            print("   [p03] Control PASSES: the convention chain (TCP offset, approach axis,")
            print("         opening axis, action encoding, waypoints) is validated end-to-end.")
            print("         Negative results on the clutter conditions can now be believed.")
            for cond in ("clutter", "gap"):
                if cond in best:
                    d = best[cond]["held_rate"] - best["solo"]["held_rate"]
                    print(f"         {cond:>8}: {best[cond]['held_rate']:.0%} "
                          f"({d:+.0%} vs solo, topple {best[cond]['topple_rate']:.0%})")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p03] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
