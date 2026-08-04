# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P09 -- The last Gate-0 unknown: can a distractor be pushed 65 mm along +x without toppling?

Where this sits
---------------
P08 established strategy C's precondition: displace the neighbours **65 mm along +x** and the
validated grasp works, 100 % held above threshold with **0 % topple**. The transition is sharp
(62.4 mm blocked -> 65.2 mm clean), so it is a genuine clearance boundary, not a soft trend.

Everything now rests on one question. The distractors' **-x faces are exposed** -- unlike their
inner faces, which is what killed a lateral push (`07_STAGE0_RESULTS.md` §7) -- so a push along
+x has a contact surface. But quasi-static theory says it should topple:

    h_crit(x) = b / (2*mu) = 36 mm / (2 * 0.95) = 19.0 mm       (P01 measured mu_eff = 0.95)
    lowest achievable contact height ~ the TCP floor, ~44 mm

44 mm is 2.3x the threshold, and a block tipping backward has nothing behind it to lean on, so
past 23.2 deg (centre of mass over the edge) it falls on its own and `TOPPLE_DOT = 0.75` fires
at 41.4 deg. P07 already saw the quasi-static version of this: pressing down harder tilted the
neighbours (`up_z` 0.934 -> 0.765) without translating them.

The escape, if there is one, is **dynamics**. Tip-vs-slide is a quasi-static criterion. A
sufficiently fast strike delivers its impulse in less time than the block needs to rotate, and
the block can skid instead of rotating. Whether that happens here at achievable speeds and
contact heights is exactly the measurement `02_PLAN.md` pre-registered as Q2, and belief 1 bet
against it ("a push at any reachable height topples more often than it slides", moderate-high
confidence).

Method
------
One env per contact height (44-90 mm). The closed gripper is placed in front of d1 and driven
+x through it. `--push_steps` sets the speed; run the probe two or three times to sweep it,
since all envs must step together.

Reported per env: d1's x displacement and `up_z`, plus the same for the **target and d0** --
the pusher is ~38.6 mm wide against a 30 mm block in a row pitched at 42 mm, so whether it
contacts only d1 is itself a measurement, not an assumption.

Usage
-----
    python eva_bc/clutter/probes/p09_push_topple.py --push_steps 40    # ~100 mm/s
    python eva_bc/clutter/probes/p09_push_topple.py --push_steps 8     # ~500 mm/s
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Push a distractor along +x: slide or topple?")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--push_steps", type=int, default=40,
                    help="control steps for the 100 mm stroke; 40 ~ 125 mm/s, 8 ~ 625 mm/s")
parser.add_argument("--x_from", type=float, default=0.160)
parser.add_argument("--x_to", type=float, default=0.265)
parser.add_argument("--min_z", type=float, default=0.044)
parser.add_argument("--max_z", type=float, default=0.090)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p09_push.json")
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
from _kin import ArmKin, Q_CLOSE, lerp_pts  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
ROW_PITCH = 0.042
DIST = mdp_cl.DISTRACTOR_NAMES
NOM_Y = (-2 * ROW_PITCH, -ROW_PITCH, ROW_PITCH, 2 * ROW_PITCH)
PUSH_Y = -ROW_PITCH          # push d1
#: measured in P01 from the block material (0.9) averaged with the table's (1.0)
MU_EFF = 0.95
H_CRIT_X = 2 * HX / (2 * MU_EFF)


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

    zs = torch.linspace(args_cli.min_z, args_cli.max_z, n, device=dev)

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

    def state():
        return {b: (e.scene[b].data.root_pos_w.torch - org).clone()
                for b in ("target",) + DIST}

    stroke = (args_cli.x_to - args_cli.x_from) * 1000
    speed = stroke / (args_cli.push_steps * 0.02)
    print("\n" + "=" * 100)
    print("P09 -- PUSH A DISTRACTOR ALONG +x: SLIDE OR TOPPLE?")
    print("=" * 100)
    print(f"   pushing {DIST[1]} at y = {PUSH_Y * 1000:.0f} mm, stroke {stroke:.0f} mm in "
          f"{args_cli.push_steps} steps  ->  ~{speed:.0f} mm/s")
    print(f"   contact height swept {args_cli.min_z * 1000:.0f}..{args_cli.max_z * 1000:.0f} mm "
          f"over {n} envs")
    print(f"   quasi-static tip threshold h_crit(x) = {H_CRIT_X * 1000:.1f} mm at mu = {MU_EFF}")
    print(f"   -> every reachable contact height is ABOVE it; only dynamics can save the push")

    # nominal pose in front of d1, gripper closed
    scene()
    a_des = torch.tensor([-math.cos(math.radians(15.0)), 0.0,
                          -math.sin(math.radians(15.0))], device=dev)
    z0 = float(zs.median())
    p0 = torch.tensor([args_cli.x_from, PUSH_Y, z0], device=dev)
    r = K.cem(p0, K.q_arm0, o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
              iters=args_cli.iters, std0=0.6, restarts=args_cli.restarts)
    print(f"\n   start pose: CEM err {r['pos_err'] * 1000:.2f} mm, o_align {r['o_align']:.3f}, "
          f"a_hat ({r['a_hat'][0]:+.2f},{r['a_hat'][1]:+.2f},{r['a_hat'][2]:+.2f})")
    p1 = torch.tensor([args_cli.x_to, PUSH_Y, z0], device=dev)
    r1 = K.cem(p1, r["q"], o_des=Y, a_des=a_des, w_o=0.60, w_a=0.25,
               iters=args_cli.iters, std0=0.20)
    print(f"   end   pose: CEM err {r1['pos_err'] * 1000:.2f} mm, o_align {r1['o_align']:.3f}")

    # per-env contact heights around the two nominal poses
    def per_env(q_nom, x):
        tgt = torch.stack([torch.full((n,), x, device=dev),
                           torch.full((n,), PUSH_Y, device=dev), zs], dim=1)
        q = K.refine(q_nom.unsqueeze(0).repeat(n, 1), tgt, iters=4)
        err = (K.fk(q)["tcp"] - tgt).norm(dim=1)
        return q, err

    q_start, e_start = per_env(r["q"], args_cli.x_from)
    q_end, e_end = per_env(r1["q"], args_cli.x_to)
    print(f"   per-env refine err: start median {float(e_start.median()) * 1000:.2f} mm, "
          f"end median {float(e_end.median()) * 1000:.2f} mm")

    # execute: closed gripper, straight drive through d1
    scene()
    b0 = state()
    # physics-only stepping: `env.step` would fire `distractor_toppled` and reset the scene,
    # re-spawning the toppled block upright with fresh jitter and erasing the very event this
    # probe exists to detect. See `_kin.ArmKin.hold_phys`.
    K.teleport_arm(q_start, Q_CLOSE)
    K.hold_phys(q_start, 160, q_fing=Q_CLOSE)
    K.run_phys(q_start, q_end, args_cli.push_steps, q_fing=Q_CLOSE)
    K.hold_phys(q_end, 480, q_fing=Q_CLOSE)
    b_mid = state()
    up_mid = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1).clone()
    # retreat and let everything settle -- a block still rocking is not yet a verdict
    K.run_phys(q_end, q_start, 40, q_fing=Q_CLOSE)
    K.hold_phys(q_start, 960, q_fing=Q_CLOSE)
    b1 = state()
    up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
    topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)

    d1x = b1[DIST[1]][:, 0] - b0[DIST[1]][:, 0]
    d1up = up[:, 1]
    tgtx = b1["target"][:, 0] - b0["target"][:, 0]
    d0x = b1[DIST[0]][:, 0] - b0[DIST[0]][:, 0]
    ok = (d1x > 0.065) & ~topp

    print("\n" + "-" * 100)
    print(f"   {'contact z':>10} | {'d1 dx':>8} {'d1 up_z':>8} | {'tgt dx':>8} {'d0 dx':>8} | "
          f"{'min up_z':>8} | {'topple':>6} | {'>=65mm & upright':>16}")
    print("   " + "-" * 92)
    rows = []
    for i in range(0, n, max(1, n // 24)):
        rows.append({"z_mm": float(zs[i]) * 1000, "d1_dx_mm": float(d1x[i]) * 1000,
                     "d1_up_z": float(d1up[i]), "tgt_dx_mm": float(tgtx[i]) * 1000,
                     "d0_dx_mm": float(d0x[i]) * 1000, "min_up_z": float(up[i].min()),
                     "topple": bool(topp[i]), "ok": bool(ok[i])})
        print(f"   {float(zs[i]) * 1000:9.1f} | {float(d1x[i]) * 1000:7.2f} "
              f"{float(d1up[i]):8.3f} | {float(tgtx[i]) * 1000:7.2f} "
              f"{float(d0x[i]) * 1000:7.2f} | {float(up[i].min()):8.3f} | "
              f"{'YES' if topp[i] else '-':>6} | {'YES' if ok[i] else '.':>16}")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"   max d1 displacement : {float(d1x.max()) * 1000:.1f} mm "
          f"(needed >= 65 mm, P08)")
    print(f"   topple rate         : {float(topp.float().mean()):.0%}")
    print(f"   target dragged along: median {float(tgtx.median()) * 1000:.2f} mm, "
          f"max {float(tgtx.abs().max()) * 1000:.2f} mm")
    print(f"   envs achieving >=65 mm with NOTHING toppled: {int(ok.sum())}/{n}")
    if int(ok.sum()) > 0:
        j = int(torch.nonzero(ok).squeeze(-1)[0])
        print(f"   -> A NON-TOPPLING PUSH EXISTS at contact z = {float(zs[j]) * 1000:.1f} mm, "
              f"speed ~{speed:.0f} mm/s.")
        print("      Belief 1 (pre-registered: pushes topple rather than slide) is WRONG, and")
        print("      strategy C is a complete, measured solution path. Gate 0 PASSES.")
    else:
        print("   -> No contact height at this speed both moves d1 far enough and leaves the")
        print("      row standing. Re-run with a shorter --push_steps before concluding: this")
        print("      is a dynamics question and one speed is not a sweep.")

    out = {"push_steps": args_cli.push_steps, "speed_mm_s": speed, "n": n,
           "z_mm": (zs * 1000).tolist(), "d1_dx_mm": (d1x * 1000).tolist(),
           "d1_up_z": d1up.tolist(), "tgt_dx_mm": (tgtx * 1000).tolist(),
           "d0_dx_mm": (d0x * 1000).tolist(), "topple": topp.tolist(),
           "up_mid_min": up_mid.min(dim=1).values.tolist(),
           "n_ok": int(ok.sum()), "rows": rows, "h_crit_x_mm": H_CRIT_X * 1000}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p09] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
