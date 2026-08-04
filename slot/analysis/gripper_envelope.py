# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the gripper's *collision envelope* relative to the TCP, then sweep grip height.

Why this exists
---------------
``eva_rl/docs/envs/precision-slot.md`` **withdraws** the insertion probe's achievability
figures: they were tuned against ``TCP_OFFSET = -0.075`` and, with the corrected -0.0419,
"the fingers now clip the slot wall tops". Nobody measured by how much. That number sets the
grip height, and the grip height has to satisfy three constraints at once:

* the hand must clear the **slot wall tops at z = 0.050** while the block is inserted,
* the hand must clear the **table at z = 0.000** while grasping off the table,
* the TCP must stay above the measured **44 mm TCP floor** (CHALLENGE_SUITE C9).

Method
------
Geometry comes from USD, pose comes from PhysX -- deliberately not sharing assumptions (the
lesson from the withdrawn probe: a self-consistent harness proves nothing; and the standing
warning that ``BBoxCache`` never sees physics poses is respected by taking only the *shape*
from USD).

**Per-collider attribution.** v1 of this script called ``ComputeLocalBound`` on each body
prim, which returns the bound of the whole *subtree* -- so ``link5``'s box swallowed the
entire downstream arm (254 mm across) and the posed result claimed the hand reached 60 mm
below the table. v2 enumerates every prim carrying ``UsdPhysics.CollisionAPI``, attributes it
to its nearest rigid-body ancestor, and stores its bound **in that body's frame** via
``ComputeRelativeBound``. Each collider is then posed by its own body's PhysX transform, so
articulated children move with their joints instead of riding a frozen parent AABB.

.. code-block:: bash

    python slot/analysis/gripper_envelope.py --num_envs 256
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gripper collision envelope + grip-height sweep.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=256, help="CEM population size")
parser.add_argument("--cem_iters", type=int, default=80)
parser.add_argument("--out_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge import mdp

#: bodies whose geometry can plausibly foul the fixture or the table during the task
HAND_BODIES = ("gripper_end", "gripper_left", "gripper_right", "link6")

OUT_DIR = args_cli.out_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "envelope"
)


def collect_colliders(stage: Usd.Stage, robot_root: str, body_names: list[str]) -> tuple[dict, str]:
    """Map body name -> list of (prim_path, local AABB in that body's frame).

    Every geometry prim is attributed to its *nearest* rigid-body ancestor, so a collider
    belonging to a finger is never counted against the wrist. v1 of this function called
    ``ComputeLocalBound`` on the body prims themselves, which returns the whole *subtree*
    bound -- ``link5``'s box then swallowed the entire downstream arm (254 mm across) and the
    posed result claimed the hand reached 60 mm below the table.

    The asset is authored with payloads and instancing (``payloads/instances.usda``), so the
    traversal MUST use ``TraverseInstanceProxies`` or it never descends into the geometry at
    all -- that is what made the first run report zero colliders.

    Prefers prims carrying ``UsdPhysics.CollisionAPI``; falls back to render geometry (a
    strict over-estimate of the collider, which is the safe direction for a clearance bound).
    """
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = stage.GetPrimAtPath(robot_root)
    if not root.IsValid():
        return {}, "robot prim not found"
    predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    def gather(want_collision: bool) -> dict:
        out: dict[str, list] = {b: [] for b in body_names}
        for prim in Usd.PrimRange(root, predicate):
            if want_collision:
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
            elif not prim.IsA(UsdGeom.Gprim):
                continue
            owner, p = None, prim
            while p.IsValid() and p.GetPath() != root.GetPath().GetParentPath():
                if p.GetName() in out:
                    owner = p
                    break
                p = p.GetParent()
            if owner is None:
                continue
            try:
                box = cache.ComputeRelativeBound(prim, owner).ComputeAlignedRange()
            except Exception:  # noqa: BLE001 - instance proxies can refuse; fall through
                continue
            if box.IsEmpty():
                continue
            out[owner.GetName()].append((str(prim.GetPath()), np.array(box.GetMin()), np.array(box.GetMax())))
        return out

    got = gather(want_collision=True)
    if any(got.values()):
        return got, "UsdPhysics.CollisionAPI"
    got = gather(want_collision=False)
    return got, "render geometry (no CollisionAPI prims found -- OVER-estimate)"


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    n = e.num_envs

    arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
    fing_dof = [robot.joint_names.index(x) for x in ("joint_left", "joint_right")]
    lo = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)[arm_dof, 0]
    hi = torch.as_tensor(robot.data.joint_pos_limits[0], device=dev)[arm_dof, 1]
    q_default = torch.as_tensor(robot.data.default_joint_pos[0], device=dev).clone()

    end_idx = robot.body_names.index("gripper_end")
    left_idx = robot.body_names.index("gripper_left")
    right_idx = robot.body_names.index("gripper_right")
    offs = torch.tensor(mdp.TCP_OFFSET, device=dev).repeat(n, 1)

    report: dict = {"tcp_offset": list(mdp.TCP_OFFSET), "body_names": list(robot.body_names)}

    # ---------------------------------------------------------------- part 1: static USD
    import omni.usd  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    cols, source = collect_colliders(stage, "/World/envs/env_0/Robot", list(robot.body_names))

    print("\n" + "=" * 78)
    print("PART 1 -- per-body COLLIDER bounds (metres, in that body's own frame)")
    print("=" * 78)
    print(f"  geometry source: {source}")
    report["geometry_source"] = source
    if not any(cols.values()):
        raise SystemExit("[envelope] FATAL: no geometry found under the robot prim -- the "
                         "traversal or the prim path is wrong. Nothing below would be valid.")
    report["colliders"] = {}
    for name in robot.body_names:
        items = cols.get(name, [])
        if not items:
            continue
        mn = np.min([a for _, a, _ in items], axis=0)
        mx = np.max([b for _, _, b in items], axis=0)
        report["colliders"][name] = {"n": len(items), "min": mn.tolist(), "max": mx.tolist()}
        print(f"  {name:14s} n={len(items):2d}  min={np.round(mn, 4)}  max={np.round(mx, 4)}")

    # per-collider corner sets, kept per body so each is posed by its own PhysX transform
    corner_sets = {}
    for name in HAND_BODIES:
        items = cols.get(name, [])
        if not items:
            continue
        cs = []
        for _, a, b in items:
            cs += [[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])]
        corner_sets[name] = torch.tensor(np.array(cs), dtype=torch.float32, device=dev)
        print(f"  [hand] {name}: {len(cs)} corners tracked")

    def fk(q_arm: torch.Tensor, q_fing: float) -> dict:
        """Batched FK: TCP (env-local), finger axis, and the hand's minimum world z."""
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, arm_dof] = q_arm
        q[:, fing_dof] = q_fing
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)
        bp = torch.as_tensor(robot.data.body_pos_w.torch, device=dev) - e.scene.env_origins.unsqueeze(1)
        bq = torch.as_tensor(robot.data.body_quat_w.torch, device=dev)
        tcp = bp[:, end_idx, :] + quat_apply(bq[:, end_idx, :], offs)
        sep = bp[:, left_idx, :] - bp[:, right_idx, :]
        sep = sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9)
        mins, ys = [], []
        for name, c in corner_sets.items():
            bi = robot.body_names.index(name)
            k = c.shape[0]
            w = bp[:, bi, :].unsqueeze(1) + quat_apply(bq[:, bi, :].unsqueeze(1).expand(-1, k, -1), c.expand(n, -1, -1))
            mins.append(w[..., 2].min(dim=1).values)
            ys.append(w[..., 1].abs().max(dim=1).values)
        return {"tcp": tcp, "sep": sep,
                "hand_min_z": torch.stack(mins, 1).min(1).values,
                "hand_max_absy": torch.stack(ys, 1).max(1).values}

    def cem(target: torch.Tensor, seed: torch.Tensor, q_fing: float) -> dict:
        """CEM over the 6 arm joints: TCP at ``target`` with the finger axis along world y.

        Position alone is not a sufficient specification for this arm -- an unconstrained
        wrist arrives holding the block along its 45 mm length instead of across its 30 mm
        width. Constraining the separation axis took the measured insert rate 0 % -> 55-81 %.
        """
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)
        mean, std = seed.clone(), torch.full((6,), 0.45, device=dev)
        best = {"cost": 1e9}
        for _ in range(args_cli.cem_iters):
            q = (mean + std * torch.randn((n, 6), device=dev)).clamp(lo, hi)
            r = fk(q, q_fing)
            pos_err = (r["tcp"] - target).norm(dim=1)
            align_err = 1.0 - (r["sep"] @ y_axis).abs()
            cost = pos_err + 0.25 * align_err
            elite = q[cost.argsort()[: max(8, n // 20)]]
            mean, std = elite.mean(0), elite.std(0).clamp(min=0.01)
            i = int(cost.argmin())
            if float(cost[i]) < best["cost"]:
                best = {"cost": float(cost[i]), "q": q[i].clone(), "pos_err": float(pos_err[i]),
                        "align_err": float(align_err[i]), "hand_min_z": float(r["hand_min_z"][i]),
                        "hand_max_absy": float(r["hand_max_absy"][i])}
        return best

    wall_top = mdp.SLOT_FLOOR_Z + mdp.WALL_HEIGHT
    mouth_x = mdp.SLOT_CENTER[0] - mdp.SLOT_DEPTH / 2
    home_x = mouth_x + mdp.SUCCESS_DEPTH
    wall_inner_y = mdp.BLOCK_HALF[1] + getattr(e.cfg, "clearance", 0.0015)

    with torch.inference_mode():
        env.reset()
        seed = q_default[arm_dof].clone()

        # ------------------------------------------------- part 2: hand extent below TCP
        print("\n" + "=" * 78)
        print("PART 2 -- hand extent below the TCP, at a representative insertion posture")
        print("=" * 78)
        probe = cem(torch.tensor([home_x, 0.0, 0.080], device=dev), seed, q_fing=0.015)
        r = fk(probe["q"].unsqueeze(0).repeat(n, 1), q_fing=0.015)
        tcp_z = float(r["tcp"][0, 2])
        drop = tcp_z - probe["hand_min_z"]
        report["hand_below_tcp_m"] = drop
        report["min_grip_z_for_walls_m"] = wall_top + drop
        print(f"  TCP at z = {tcp_z * 1000:.1f} mm  (pos err {probe['pos_err'] * 1000:.2f} mm, "
              f"align err {probe['align_err']:.4f})")
        print(f"  lowest hand collider at z = {probe['hand_min_z'] * 1000:.1f} mm")
        print(f"  --> the hand hangs {drop * 1000:.1f} mm BELOW the TCP")
        print(f"  wall tops z = {wall_top * 1000:.1f} mm, wall inner face |y| = {wall_inner_y * 1000:.1f} mm")
        print(f"  --> minimum wall-clearing grip height z >= {(wall_top + drop) * 1000:.1f} mm")
        print(f"  --> TCP floor is 44 mm, so the feasible grip band is "
              f"[{max(0.044, wall_top + drop) * 1000:.1f}, 100] mm")

        # ------------------------------------------- part 3: grip-height x depth sweep
        print("\n" + "=" * 78)
        print("PART 3 -- CEM sweep: TCP at (x, 0, z), finger axis along world y")
        print("=" * 78)
        print(f"  mouth x = {mouth_x:.3f}   home x = {home_x:.3f}   wall top z = {wall_top:.3f}")
        print(f"  {'x [m]':>7} {'z [m]':>7} {'tcp_err':>9} {'align':>8} {'hand_min_z':>11} {'wall_clr':>9}  ok")
        xs = [round(mouth_x - 0.030, 4), round(mouth_x, 4), round(mouth_x + 0.020, 4), round(home_x, 4)]
        zs = [0.060, 0.066, 0.072, 0.078, 0.084, 0.090, 0.096]
        grid = []
        for z in zs:
            warm = seed.clone()
            for x in xs:
                b = cem(torch.tensor([x, 0.0, z], device=dev), warm, q_fing=0.015)
                warm = b["q"].clone()
                clr = b["hand_min_z"] - wall_top
                ok = b["pos_err"] < 0.004 and b["align_err"] < 0.05 and clr > 0.002
                grid.append({"x": x, "z": z, "tcp_err_mm": b["pos_err"] * 1000, "align_err": b["align_err"],
                             "hand_min_z_mm": b["hand_min_z"] * 1000, "wall_clearance_mm": clr * 1000,
                             "ok": bool(ok), "q": b["q"].cpu().tolist()})
                print(f"  {x:7.3f} {z:7.3f} {b['pos_err'] * 1000:8.2f}m {b['align_err']:8.4f} "
                      f"{b['hand_min_z'] * 1000:10.1f}m {clr * 1000:+8.1f}m  {'OK' if ok else '--'}")
        report["sweep"] = grid

        # ------------------------------------------- part 4: grasp pose over the table
        print("\n" + "=" * 78)
        print("PART 4 -- grasp pose at the block spawn, fingers OPEN (table clearance, z = 0)")
        print("=" * 78)
        print(f"  {'z [m]':>7} {'tcp_err':>9} {'align':>8} {'hand_min_z':>11}  ok")
        spawn = (0.220, -0.130)
        grasp_grid = []
        warm = seed.clone()
        for z in [0.046, 0.052, 0.058, 0.064, 0.070, 0.076]:
            b = cem(torch.tensor([spawn[0], spawn[1], z], device=dev), warm, q_fing=0.045)
            warm = b["q"].clone()
            ok = b["pos_err"] < 0.004 and b["align_err"] < 0.05 and b["hand_min_z"] > 0.001
            grasp_grid.append({"z": z, "tcp_err_mm": b["pos_err"] * 1000, "align_err": b["align_err"],
                               "hand_min_z_mm": b["hand_min_z"] * 1000, "ok": bool(ok),
                               "q": b["q"].cpu().tolist()})
            print(f"  {z:7.3f} {b['pos_err'] * 1000:8.2f}m {b['align_err']:8.4f} "
                  f"{b['hand_min_z'] * 1000:10.1f}m  {'OK' if ok else '--'}")
        report["grasp_sweep"] = grasp_grid

        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "gripper_envelope.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[envelope] wrote {path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
