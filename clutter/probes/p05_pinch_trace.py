# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P05 -- Step-by-step trace of one grasp: commanded vs ACHIEVED at every phase.

P03's control grasp fails with `gap = -1.2 mm` (fingers shut on air) in every condition,
including the one with no clutter at all, while the CEM reports sub-millimetre TCP error and
the distractors are demonstrably never disturbed (`up_min = 1.0`, so no hidden reset). Three
candidates remain and the aggregate numbers cannot separate them:

  A. the planned pose is right but the arm never gets there (tracking error -- C9 measured
     commands below the floor landing 5-16 mm off, and x = 0.30 carrying 14-48 mm);
  B. the arm gets there but the block is not between the fingers (a TCP-convention error --
     precisely the failure `mdp/common.TCP_OFFSET` was written to prevent, and one that
     eva_bc paid a full GPU run for);
  C. the approach path sweeps the block away before the fingers ever close.

So stop aggregating and print one trial. At each phase this reports the **achieved** TCP read
back from the sim next to the commanded one, the two finger body positions, the block pose,
and the clear gap. eva_bc's rule -- *executed-state checks at every phase boundary, never
trust a commanded pose* -- applied as a diagnostic instead of a guard.

The decisive line is the one just before the fingers close: if `|finger.y - block.y|` is not
about 44.5 mm on both sides, the block was never in the jaw and no amount of closing force
will help.

Usage
-----
    python eva_bc/clutter/probes/p05_pinch_trace.py --grip_z 0.055
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace one grasp, commanded vs achieved.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--teleport", action="store_true",
                    help="skip the approach path: teleport straight to the grasp pose. "
                         "Isolates candidate C from A/B.")
parser.add_argument("--approach", type=str, default="axis", choices=("axis", "vertical"),
                    help="'axis' = grasp_geometry.py's standoff along the (z-flattened) jaw "
                         "axis. 'vertical' = descend from directly above the grasp pose, "
                         "holding the wrist orientation fixed.")
parser.add_argument("--standoff", type=float, default=0.070)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p05_pinch_trace.json")
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kin import ArmKin, Q_OPEN, lerp_pts  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
PARK = ((0.45, 0.30), (0.45, -0.30), (0.62, 0.30), (0.62, -0.30))
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
    trace = []

    def put(name, x, y, z=HZ):
        p = torch.tensor([x, y, z], device=dev).repeat(n, 1) + org
        e.scene[name].write_root_state_to_sim(
            torch.cat([p, ident, torch.zeros((n, 6), device=dev)], dim=1))

    def scene():
        put("target", ROW_X, 0.0)
        for i, d in enumerate(mdp_cl.DISTRACTOR_NAMES):
            put(d, PARK[i][0], PARK[i][1])
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def report(tag: str, q_cmd: torch.Tensor | None = None):
        tcp = K.tcp_now()[0]
        fl, fr = K.finger_pos()
        fl, fr = fl[0], fr[0]
        bp = (e.scene["target"].data.root_pos_w.torch - org)[0]
        gap = float(K.gap()[0])
        jt = K.robot.data.joint_pos.torch[0, K.arm_dof]
        jerr = float((jt - q_cmd).abs().max()) if q_cmd is not None else float("nan")
        print(f"  {tag:<22} TCP ({tcp[0] * 1000:7.2f},{tcp[1] * 1000:7.2f},{tcp[2] * 1000:7.2f}) "
              f"| blk ({bp[0] * 1000:7.2f},{bp[1] * 1000:7.2f},{bp[2] * 1000:7.2f}) "
              f"| fL.y {fl[1] * 1000:7.2f} fR.y {fr[1] * 1000:7.2f} "
              f"| dL {abs(float(fl[1] - bp[1])) * 1000:6.2f} dR {abs(float(fr[1] - bp[1])) * 1000:6.2f} "
              f"| gap {gap * 1000:6.2f} | qerr {jerr * 1000:6.2f}mrad")
        trace.append({"tag": tag, "tcp_mm": (tcp * 1000).tolist(),
                      "block_mm": (bp * 1000).tolist(),
                      "fl_mm": (fl * 1000).tolist(), "fr_mm": (fr * 1000).tolist(),
                      "gap_mm": gap * 1000, "joint_err_mrad": jerr * 1000})

    print("\n" + "=" * 118)
    print(f"P05 -- GRASP TRACE at grip z = {args_cli.grip_z * 1000:.0f} mm"
          f"{'  (teleport mode)' if args_cli.teleport else ''}")
    print("=" * 118)
    print("  dL/dR = |finger.y - block.y|. For the block to be IN the jaw both must be")
    print("  about 44.5 mm with the fingers open. gap is the clear finger opening.")
    print()

    scene()
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    a_des = torch.tensor([-1.0, 0.0, 0.0], device=dev)
    r = K.cem(grip, K.q_arm0, o_des=Y, a_des=a_des, w_o=0.25, w_a=0.35,
              iters=args_cli.iters, std0=0.6, restarts=args_cli.restarts)
    q_grip = r["q"]
    print(f"  PLAN: CEM err {r['pos_err'] * 1000:.2f} mm, o_align {r['o_align']:.3f}, "
          f"a_align {r['a_align']:+.3f}, a_hat "
          f"({r['a_hat'][0]:+.3f},{r['a_hat'][1]:+.3f},{r['a_hat'][2]:+.3f}), "
          f"gripper low_z {r['low_z'] * 1000:.1f} mm")
    print(f"  PLAN: commanded TCP ({grip[0] * 1000:.1f},{grip[1] * 1000:.1f},"
          f"{grip[2] * 1000:.1f})  achieved-by-FK ({r['tcp'][0] * 1000:.2f},"
          f"{r['tcp'][1] * 1000:.2f},{r['tcp'][2] * 1000:.2f})")

    # what the FK pose alone says about the jaw, before any physics
    K.fk(q_grip.unsqueeze(0).repeat(n, 1))
    scene()
    report("FK pose (no physics)")

    if args_cli.approach == "vertical":
        # Descend from directly above, wrist orientation unchanged. For this arm the jaw axis
        # runs up-and-back (a_hat ~ (-0.78, +0.44, +0.44)), so the wrist sits BEHIND and BELOW
        # the grasp point while the fingers reach back over it. A vertical descent therefore
        # brings the two fingers down either side of the block and the wrist down behind it --
        # and, decisively, it never visits x ~ 0.31, which C9 measured as unusable below
        # z ~ 0.10 (14-48 mm tracking error). That standoff is what wrecked the `axis` run.
        f = torch.tensor([0.0, 0.0, -1.0], device=dev)
    else:
        f = r["a_hat"].clone()
        f[2] = 0.0
        f = f / f.norm().clamp(min=1e-9)
    STANDOFF = args_cli.standoff
    q_seq = [q_grip]
    if not args_cli.teleport:
        back, _ = [], None
        qq = q_grip
        for t in lerp_pts(grip, grip - STANDOFF * f, 3):
            rr = K.cem(t, qq, o_des=Y, a_des=a_des, w_o=0.25, w_a=0.35,
                       iters=args_cli.iters, std0=0.12)
            qq = rr["q"]
            back.append(qq)
        q_seq = list(reversed(back)) + [q_grip]
        print(f"  PLAN: standoff at TCP ({(grip - STANDOFF * f)[0] * 1000:.1f},"
              f"{(grip - STANDOFF * f)[1] * 1000:.1f},{(grip - STANDOFF * f)[2] * 1000:.1f}) mm"
              f"  (approach direction f = ({f[0]:+.2f},{f[1]:+.2f},{f[2]:+.2f}))")
    # PLAN THE LIFT NOW, BEFORE ANYTHING CLOSES.
    # eva_rl's handoff calls this the single most expensive mistake of that effort, and the
    # first version of this probe reproduced it exactly: `cem()` calls `fk()`, which does
    # `write_joint_state_to_sim` with `q_fing = Q_OPEN` -- hundreds of times. Searching for a
    # lift path after the fingers have closed teleports the arm around with the jaw wide open
    # and silently drops the block. The symptom is a clean grasp (gap stalls at the block
    # width) followed by `gap -> -1.2 mm` on the lift.
    up = grip + torch.tensor([0.0, 0.0, 0.09], device=dev)
    q_up = K.cem(up, q_grip, o_des=Y, a_des=a_des, w_o=0.25, w_a=0.35,
                 iters=args_cli.iters, std0=0.25)["q"]
    print()

    # ---- execute
    scene()
    K.teleport_arm(q_seq[0].unsqueeze(0).repeat(n, 1), Q_OPEN)
    report("after teleport", q_seq[0])
    K.hold(q_seq[0].unsqueeze(0).repeat(n, 1), 20, close=False)
    report("settled at start", q_seq[0])
    for i in range(len(q_seq) - 1):
        K.run(q_seq[i].unsqueeze(0).repeat(n, 1), q_seq[i + 1].unsqueeze(0).repeat(n, 1),
              25, close=False)
        report(f"waypoint {i + 1}", q_seq[i + 1])
    K.hold(q_grip.unsqueeze(0).repeat(n, 1), 30, close=False)
    report("AT GRASP, open", q_grip)
    K.hold(q_grip.unsqueeze(0).repeat(n, 1), 70, close=True)
    report("after close", q_grip)

    K.run(q_grip.unsqueeze(0).repeat(n, 1), q_up.unsqueeze(0).repeat(n, 1), 40, close=True)
    K.hold(q_up.unsqueeze(0).repeat(n, 1), 40, close=True)
    report("after lift", q_up)

    bp = (e.scene["target"].data.root_pos_w.torch - org)
    rose = float((bp[:, 2] > HZ + 0.045).float().mean())
    gap = K.gap()
    encl = float(((gap - 2 * HY).abs() < 0.012).float().mean())
    print(f"\n  rose {rose:.0%}   enclosed {encl:.0%}   median gap {float(gap.median()) * 1000:.2f} mm")

    print("\n" + "=" * 118)
    print("READING")
    print("=" * 118)
    g = next((t for t in trace if t["tag"] == "AT GRASP, open"), None)
    if g:
        dL = abs(g["fl_mm"][1] - g["block_mm"][1])
        dR = abs(g["fr_mm"][1] - g["block_mm"][1])
        dz = g["tcp_mm"][2] - g["block_mm"][2]
        dx = g["tcp_mm"][0] - g["block_mm"][0]
        print(f"  at the grasp, open: dL {dL:.2f} mm, dR {dR:.2f} mm  "
              f"(both should be ~44.5)")
        print(f"                      TCP - block: dx {dx:+.2f} mm, dz {dz:+.2f} mm")
        if abs(dL - 44.5) > 8 or abs(dR - 44.5) > 8:
            print("  -> The block is NOT between the fingers. This is a pose/convention")
            print("     problem, not a force problem. Candidate B.")
        elif abs(dx) > 15:
            print("  -> The fingers straddle it in y but the TCP is off along x: the jaw is")
            print("     in front of or behind the block. Check the approach depth.")
        else:
            print("  -> The block IS in the jaw with the fingers open. If it is still not")
            print("     held after closing, the problem is grip force or the lift, not the")
            print("     pose. Candidate A.")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f2:
        json.dump({"grip_z": args_cli.grip_z, "teleport": args_cli.teleport,
                   "plan": {k: (v if isinstance(v, (int, float, str)) else
                                [float(x) for x in v]) for k, v in r.items() if k != "q"},
                   "trace": trace, "rose": rose, "encl": encl}, f2, indent=2)
    print(f"\n[p05] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
