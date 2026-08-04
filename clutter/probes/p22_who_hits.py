# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P22 -- Remove the target and close the gripper on nothing. What still moves?

The contradiction that forced this
----------------------------------
P19 traced the closing slam at full physics-step resolution and reported the ordering:

    jaw reaches 60 mm at step 3   |   neighbour first moves at step 4-5   |   target at step 8
    "target moved FIRST in 7 % of the 30 envs where both moved"

The neighbour moves **before** the target does. That is incompatible with the story P19
itself told -- fingers strike the target, target strikes its neighbour -- and it was written
down anyway. This probe exists to settle it with a control instead of a narrative.

The control
-----------
Teleport the target 2 m below the table, leave the four distractors exactly where they
spawned, put the arm in the identical grasp pose, and close the gripper on empty air.

    * If distractors still move -> **the finger blades are sweeping them directly.** The
      target is a bystander and every fix aimed at the grip is aimed at the wrong thing.
    * If nothing moves -> the target really is the intermediary, and P19's ordering
      measurement is what is wrong.

Either way something currently believed gets retired. Three further arms bound it:

    A  normal            target present, grasp pose, close        the reference
    B  no target         target removed, same pose, same close    the control
    C  no target, high   target removed, pose raised 40 mm        clear of the row entirely;
                                                                  if THIS moves anything the
                                                                  problem is not geometric
    D  no close          target present, pose held, jaw stays open  isolates the pose itself

Alongside, the measurement that has been missing since P01: the **finger collision meshes**,
read from USD through instance proxies, expressed in the gripper frame, and used to compute
the blades' true reach in y at the row's height. P01's finger geometry came from an AABB and
was retracted (`HANDOFF` 7.1); nothing replaced it. Body origins have been standing in for
contact surfaces ever since, and P18 built a wrong causal claim on exactly that gap.

Usage
-----
    python eva_bc/clutter/probes/p22_who_hits.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Does the close disturb the row without a target?")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p22_who.json")
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
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from _kin import Q_CLOSE, Q_OPEN, _t  # noqa: E402
from clutter_expert import ClutterExpert, DIST, HX, HY, HZ  # noqa: E402

ROW_Y = (-0.084, -0.042, 0.042, 0.084)


def finger_mesh_extents(K, env):
    """Collision-mesh extents of each gripper body, in that body's own frame [mm].

    Read through `Usd.TraverseInstanceProxies` -- a plain `Usd.PrimRange` stops at the
    instance boundary and reports a confident zero, which is what made P01's first pass
    report 'collision prims: 0' (`HANDOFF` 7.1).
    """
    try:
        from pxr import Usd, UsdGeom
        from isaaclab.sim.utils.stage import get_current_stage
    except Exception as exc:                                     # noqa: BLE001
        print(f"   [mesh] unavailable: {exc}")
        return {}
    stage = get_current_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not root or not root.IsValid():
        print("   [mesh] robot prim not found")
        return {}
    out = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        name = prim.GetName()
        if not prim.IsA(UsdGeom.Mesh):
            continue
        parent = prim.GetPath().pathString
        which = next((b for b in ("gripper_left", "gripper_right", "gripper_end")
                      if f"/{b}" in parent), None)
        if which is None:
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        p = torch.tensor([[q[0], q[1], q[2]] for q in pts], dtype=torch.float32)
        lo, hi = p.amin(0) * 1000, p.amax(0) * 1000
        cur = out.setdefault(which, {"lo": lo, "hi": hi, "n": 0})
        cur["lo"] = torch.minimum(cur["lo"], lo)
        cur["hi"] = torch.maximum(cur["hi"], hi)
        cur["n"] += len(p)
    return out


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    dev, n = e.device, e.num_envs
    org = e.scene.env_origins

    print("\n" + "=" * 100)
    print("P22 -- WHO ACTUALLY HITS THE NEIGHBOURS?")
    print("=" * 100)

    ex = ClutterExpert(env, grip_z=args_cli.grip_z)
    K = ex.K

    # ---------------------------------------------------------------- finger geometry
    print("\n   FINGER COLLISION-MESH EXTENTS (body frame, mm) -- missing since P01")
    ext = finger_mesh_extents(K, env)
    for name, d in ext.items():
        print(f"      {name:>14}: x {float(d['lo'][0]):7.1f}..{float(d['hi'][0]):6.1f} | "
              f"y {float(d['lo'][1]):7.1f}..{float(d['hi'][1]):6.1f} | "
              f"z {float(d['lo'][2]):7.1f}..{float(d['hi'][2]):6.1f} | "
              f"{d['n']} pts")
    if not ext:
        print("      (unavailable -- the controls below stand on their own)")

    # --------------------------------------------------- one spawn, restored between arms
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
    chain = ex.adapt()
    print(f"\n   spawn: min free gap median {float(min_gap.median()) * 1000:.1f} mm | "
          f"grasp pose o_align {ex.pose['o_align']:.4f}")

    def restore(drop_target: bool):
        for k, v in snap.items():
            s = v.clone()
            if drop_target and k == "target":
                s[:, 2] -= 2.0                     # out of the scene, not merely displaced
                s[:, 7:] = 0.0
            e.scene[k].write_root_state_to_sim(s)
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def arm(tag, drop_target, dz, do_close):
        restore(drop_target)
        q = chain[ex.i_grip]
        if dz:
            tgt = torch.stack([tpos0[:, 0], tpos0[:, 1],
                               torch.full((n,), args_cli.grip_z + dz, device=dev)], dim=1)
            q = K.refine(q, tgt, iters=4)
        # settle at the grasp pose with the jaw OPEN, exactly as the expert does
        K.teleport_arm(q, Q_OPEN)
        for _ in range(160):
            K.robot.set_joint_position_target(K._drive(q, Q_OPEN))
            K.robot.write_data_to_sim()
            e.sim.step()
            e.scene.update(e.physics_dt)
        pre = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
        qf = Q_CLOSE if do_close else Q_OPEN
        for _ in range(560):
            K.robot.set_joint_position_target(K._drive(q, qf))
            K.robot.write_data_to_sim()
            e.sim.step()
            e.scene.update(e.physics_dt)
        post = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1)
        up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
        disp = (post[:, :, :2] - pre[:, :, :2]).norm(dim=2)          # (n,4)
        moved = (disp > 0.0015).any(dim=1)
        topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)
        per = [float((disp[:, k] > 0.0015).float().mean()) for k in range(4)]
        print(f"\n   {tag}")
        print(f"      distractor moved >1.5 mm : {float(moved.float().mean()):6.1%}   "
              f"toppled: {float(topp.float().mean()):6.1%}")
        print(f"      per block                : "
              + " | ".join(f"d{k}(y={ROW_Y[k] * 1000:+.0f}) {per[k]:5.1%}" for k in range(4)))
        print(f"      median displacement      : "
              + " | ".join(f"{float(disp[:, k].median()) * 1000:5.2f}" for k in range(4))
              + "  mm")
        return {"arm": tag, "moved": float(moved.float().mean()),
                "topple": float(topp.float().mean()), "per_block": per,
                "median_disp_mm": [float(disp[:, k].median()) * 1000 for k in range(4)]}

    res = [
        arm("A  target present, normal close", False, 0.0, True),
        arm("B  TARGET REMOVED, same close  ", True, 0.0, True),
        arm("C  TARGET REMOVED, pose +40 mm ", True, 0.040, True),
        arm("D  target present, jaw stays OPEN", False, 0.0, False),
    ]

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    a, b, c, d = res
    print(f"   with a target      : {a['moved']:.1%} moved, {a['topple']:.1%} toppled")
    print(f"   with NO target     : {b['moved']:.1%} moved, {b['topple']:.1%} toppled")
    print(f"   no target, 40 mm up: {c['moved']:.1%} moved, {c['topple']:.1%} toppled")
    print(f"   no close at all    : {d['moved']:.1%} moved, {d['topple']:.1%} toppled")
    if b["topple"] > 0.5 * a["topple"] and b["topple"] > 0.05:
        print("\n   -> The blades sweep the neighbours DIRECTLY. The target is a bystander,")
        print("      and every fix aimed at the grip has been aimed at the wrong object.")
        print("      P19's mechanism ('fingers strike target, target strikes neighbour') is")
        print("      withdrawn; its ORDERING measurement was right and its story was wrong.")
    elif d["topple"] > 0.05:
        print("\n   -> The pose alone disturbs the row before anything closes. The grasp")
        print("      pose, not the close, is what needs to move.")
    else:
        print("\n   -> Removing the target removes the disturbance: the target really is the")
        print("      intermediary, and the closing impulse propagates through it.")

    out = {"n": n, "grip_z": args_cli.grip_z, "o_align": ex.pose["o_align"],
           "mesh": {k: {"lo": v["lo"].tolist(), "hi": v["hi"].tolist(), "n": v["n"]}
                    for k, v in ext.items()},
           "min_gap_mm": (min_gap * 1000).tolist(), "arms": res}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p22] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
