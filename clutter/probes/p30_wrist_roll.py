# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P30 -- Is `joint6 -> joint6 +/- pi` a symmetry, or a second physical pose?

Where this question came from
-----------------------------
P28 solved the grasp pose eight times, independently, and the joint vectors did not agree.
The first four draws:

    draw 0   [-0.310, -1.767, -1.035, +1.043, -0.492, -1.946]   o_align 0.9996
    draw 1   [-0.353, -1.653, -0.898, +0.908, -0.566, +1.209]   o_align 0.9936
    draw 2   [+0.304, -1.755, -1.026, +1.032, +0.489, -1.284]   o_align 0.9974
    draw 3   [-0.271, -1.762, -1.052, +1.056, -0.438, +1.367]   o_align 0.9906

Draws 0, 1 and 3 agree on joints 1-5 to within a few hundredths of a radian and disagree on
**joint6 by about pi** (-1.946 + pi = +1.196). Draw 2 is something else again -- joints 1 and 5
sign-flipped, which is the mirrored-wrist family.

Both matter for Stage 2 and they matter for different reasons, so they get separate treatment.
This probe is only about the joint6 one.

Why it is not obviously an alias
---------------------------------
The CEM cannot tell the two apart. Its orientation term is `|o_hat . o_des|` -- **sign-free**,
because a parallel jaw is symmetric and demanding a sign would reject half the valid solutions
(`_kin.py` cem docstring). `box_penetration` takes a max over body origins, and a pi roll
merely swaps which finger is where, so that is blind to it too. Both representatives therefore
score identically on everything `plan()` uses to choose.

But the measured finger meshes are **not** symmetric about the roll axis:

    gripper_left    x -19.2 .. +19.2 | y -41.9 .. +46.7 | z -58.7 .. +34.7   [mm, body frame]
    gripper_right   x -19.2 .. +19.2 | y -46.7 .. +41.9 | z -58.7 .. +34.7

In y the two are exact mirrors, so swapping them is free. **In z they are identical, not
mirrored** -- the blade runs 58.7 mm one way and 34.7 mm the other. A roll that maps left onto
right's position also flips that z profile, and the two configurations then sweep **different
volumes**, by as much as 24 mm.

Which is to say: the pi roll is a symmetry of everything the *search* scores and possibly not
a symmetry of the thing that actually fails. P22 established that the finger blades sweeping
the neighbours is the entire remaining failure mode, with 7.8 mm of margin. 24 mm of blade
profile is not a rounding error against 7.8 mm of margin.

So this is measured, not argued -- the same discipline that produced the mesh read in the first
place, after an AABB assumption had been confidently wrong for eleven probes.

What is measured
----------------
1. **Kinematic.** `fk(q)` vs `fk(q_rolled)`: TCP error, `a_hat` dot, `|o_hat . o_hat'|`, the
   per-body position differences, and the keep-out penetration of each.
2. **Physical, paired.** Snapshot / restore, one spawn, both representatives run end to end.
   Enclosure, at-goal, topple, success -- the conjunction.
3. **The consequence for P28.** If the two score the same, `plan()` should canonicalise joint6
   so repeated planning is comparable and demos are consistent, and P28's draws 0/1/3 collapse
   to one pose. If they score differently, the roll is a **free, unexploited degree of freedom**
   that the pose search currently picks at random -- and screening it is worth as much as
   `screen = 4` was.

Pre-registered prediction
-------------------------
**They differ.** The z profile is not mirror-symmetric, and the failure mode is a blade sweep
with 7.8 mm of margin. **Predicted: TCP and axis agreement to within a millimetre and a few
thousandths, and an end-to-end success difference of more than 5 points.**

Falsifier: identical scores. Then the roll IS an alias, the fix is canonicalisation, and P28's
apparent branch diversity was an artifact of the search rather than a property of the task.

Usage
-----
    python -u eva_bc/clutter/probes/p30_wrist_roll.py --num_envs 128 --reps 3
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Is a pi roll of joint6 a symmetry?")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--screen", type=int, default=4)
parser.add_argument("--reps", type=int, default=3, help="independent pose draws / spawn batches")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p30_wrist_roll.json")
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from _kin import _t  # noqa: E402
from clutter_expert import DIST, ClutterExpert  # noqa: E402


def roll(q: torch.Tensor, lo: float, hi: float) -> torch.Tensor | None:
    """`q` with joint6 rolled by +/- pi, whichever lands inside the joint limits."""
    out = q.clone()
    for s in (+1.0, -1.0):
        v = float(q[5]) + s * math.pi
        if lo <= v <= hi:
            out[5] = v
            return out
    return None


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    n = e.num_envs

    print("\n" + "=" * 100)
    print("P30 -- IS A pi ROLL OF joint6 A SYMMETRY, OR A SECOND PHYSICAL POSE?")
    print("=" * 100)
    print("   PREDICTION (registered): they DIFFER -- TCP and axes agree to <1 mm / <0.005,")
    print("   but end-to-end success differs by more than 5 points, because the finger blade")
    print("   z-profile (-58.7 .. +34.7 mm) is not mirror-symmetric about the roll axis and")
    print("   the residual failure has only 7.8 mm of margin.")

    cells = []
    for rep in range(args_cli.reps):
        env.reset()
        for _ in range(30):
            e.sim.step()
            e.scene.update(e.physics_dt)
        ex = ClutterExpert(env, grip_z=args_cli.grip_z, screen=args_cli.screen, verbose=False)
        K = ex.K
        q0 = ex.pose["q"]
        q1 = roll(q0, float(K.lo[5]), float(K.hi[5]))
        if q1 is None:
            print(f"\n   draw {rep}: joint6 = {float(q0[5]):+.3f} -- neither +pi nor -pi is "
                  f"inside [{float(K.lo[5]):+.2f}, {float(K.hi[5]):+.2f}]. SKIPPED.")
            continue

        # ------------------------------------------------------------------ 1. kinematic
        g0 = K.fk(q0.unsqueeze(0).repeat(n, 1))
        b0 = g0["bodies"][0].clone()
        t0, a0, o0 = g0["tcp"][0].clone(), g0["a_hat"][0].clone(), g0["o_hat"][0].clone()
        p0 = float(K.box_penetration(g0["bodies"][:1], ex.boxes, ex.margin)[0])
        g1 = K.fk(q1.unsqueeze(0).repeat(n, 1))
        b1 = g1["bodies"][0].clone()
        t1, a1, o1 = g1["tcp"][0].clone(), g1["a_hat"][0].clone(), g1["o_hat"][0].clone()
        p1 = float(K.box_penetration(g1["bodies"][:1], ex.boxes, ex.margin)[0])

        print(f"\n   --- draw {rep}: joint6 {float(q0[5]):+.3f} -> {float(q1[5]):+.3f} ---")
        print(f"      TCP           ({t0[0]*1000:6.1f},{t0[1]*1000:+6.1f},{t0[2]*1000:6.1f}) vs "
              f"({t1[0]*1000:6.1f},{t1[1]*1000:+6.1f},{t1[2]*1000:6.1f}) mm   "
              f"|d| {float((t1 - t0).norm())*1000:.2f} mm")
        print(f"      a_hat . a_hat' {float(a0 @ a1):+.5f}     |o_hat . o_hat'| "
              f"{abs(float(o0 @ o1)):.5f}")
        print(f"      penetration    {p0*1000:.2f} mm vs {p1*1000:.2f} mm")
        for i, nm in ((K.i_end, "gripper_end"), (K.i_left, "gripper_left"),
                      (K.i_right, "gripper_right")):
            d = b1[i] - b0[i]
            print(f"      {nm:<14} moves ({d[0]*1000:+6.1f},{d[1]*1000:+6.1f},"
                  f"{d[2]*1000:+6.1f}) mm")
        swap = float((b1[K.i_left] - b0[K.i_right]).norm()) * 1000
        print(f"      left' vs right: {swap:.2f} mm apart "
              + ("(the fingers SWAPPED -- consistent with a pure roll)" if swap < 3.0
                 else "(NOT a clean swap)"))

        # -------------------------------------------------------------------- 2. physical
        snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}

        def restore():
            for k, v in snap.items():
                e.scene[k].write_root_state_to_sim(v.clone())
            e.sim.forward()
            e.scene.update(e.physics_dt)

        row = {"rep": rep, "j6": [float(q0[5]), float(q1[5])],
               "tcp_delta_mm": float((t1 - t0).norm()) * 1000,
               "a_dot": float(a0 @ a1), "o_abs_dot": abs(float(o0 @ o1)),
               "pen_mm": [p0 * 1000, p1 * 1000], "finger_swap_mm": swap,
               "o_align": ex.pose["o_align"], "screen_score": ex.pose.get("screen_score")}
        print(f"      {'arm':<10} {'encl':>7} {'goal':>7} {'topple':>7} {'SUCCESS':>8}")
        for tag, q in (("as solved", q0), ("rolled", q1)):
            restore()
            ex.pose = dict(ex.pose)
            ex.qs[ex.i_grip] = q
            # rebuild the chain around the new grasp joint vector: every waypoint is refined
            # per env anyway, and the neighbours of the grasp must start from it or the
            # descent lands in the OTHER roll and the comparison is meaningless
            for i in range(ex.i_grip - 1, -1, -1):
                ex.qs[i] = ex._solve(ex.pts[i], ex.qs[i + 1], 2, 0.08, 1)["q"]
            for i in range(ex.i_grip + 1, len(ex.qs)):
                ex.qs[i] = ex._solve(ex.pts[i], ex.qs[i - 1], 2, 0.08, 1)["q"]
            r = ex.run_physics(ex.adapt())
            row[tag] = {"encl": float(r["held"].float().mean()),
                        "at_goal": float(r["at_goal"].float().mean()),
                        "topple": float(r["topple"].float().mean()),
                        "success": float(r["success"].float().mean()),
                        "succ_mask": r["success"].tolist()}
            print(f"      {tag:<10} {row[tag]['encl']:>7.1%} {row[tag]['at_goal']:>7.1%} "
                  f"{row[tag]['topple']:>7.1%} {row[tag]['success']:>8.1%}")
        restore()
        d = row["rolled"]["success"] - row["as solved"]["success"]
        print(f"      -> rolled - as solved = {d:+.1%}")
        cells.append(row)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    if not cells:
        print("   no usable draws (joint6 never admitted a +/- pi roll inside its limits)")
    else:
        S0 = torch.tensor([x for c in cells for x in c["as solved"]["succ_mask"]],
                          dtype=torch.float32)
        S1 = torch.tensor([x for c in cells for x in c["rolled"]["succ_mask"]],
                          dtype=torch.float32)
        d = float(S1.mean() - S0.mean())
        per = [c["rolled"]["success"] - c["as solved"]["success"] for c in cells]
        print(f"   pooled {len(S0)} episodes per arm")
        print(f"      as solved  {float(S0.mean()):6.1%}")
        print(f"      rolled     {float(S1.mean()):6.1%}    difference {d:+.1%}")
        print(f"      per draw   " + ", ".join(f"{x:+.1%}" for x in per))
        print(f"      max TCP disagreement {max(c['tcp_delta_mm'] for c in cells):.2f} mm; "
              f"min |o.o'| {min(c['o_abs_dot'] for c in cells):.5f}")
        print("\n   " + "=" * 90)
        if max(abs(x) for x in per) > 0.05:
            print("   VERDICT: the roll is NOT an alias -- prediction holds. joint6's roll is a")
            print("   FREE, UNEXPLOITED DEGREE OF FREEDOM the pose search currently picks at")
            print("   random. It must be screened, not canonicalised away.")
        else:
            print("   VERDICT: the roll IS an alias -- prediction FALSIFIED. Canonicalise joint6")
            print("   in plan(); P28's apparent branch diversity was an artifact of the search.")
        print("   " + "=" * 90)

    out = {"num_envs": n, "reps": args_cli.reps, "grip_z": args_cli.grip_z,
           "screen": args_cli.screen, "cells": cells}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p30] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
