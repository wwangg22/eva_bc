# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Validation rung V4, redone: can this arm actually execute the insertion stroke?

Background
----------
``eva_rl/docs/envs/precision-slot.md`` **withdrew** the original probe's numbers -- they were
tuned against ``TCP_OFFSET = -0.075`` and the corrected -0.0419 "makes the fingers clip the
slot wall tops". The original probe also started the block at 42 mm of depth against a 40 mm
success threshold, so it measured a ~1 mm stroke and its 68.8 % meant nothing.

``gripper_envelope.py`` then measured the hand hanging **40.8 mm below the TCP**, implying a
minimum wall-clearing grip height of 90.8 mm -- but a block sitting on the slot floor has its
top at **z = 0.090**, so that bound would make the task impossible. That bound is
over-conservative in two ways, and this script removes both:

1. It took the minimum z over **all** hand geometry regardless of where it is in xy. Only
   geometry that is actually *over a wall* can hit one. The walls occupy a thin band
   (``x in [0.210, 0.280]``, ``|y| in [half, half+0.008]``, ``z in [0.020, 0.050]``); the
   fingers press on the block faces at ``|y| = 0.015``, which is **inboard** of the walls.
2. It fell back to render geometry because no ``UsdPhysics.CollisionAPI`` prims were found
   by that traversal. Render meshes are an over-estimate of the collider.

So this script reports the **footprint-restricted** clearance, and then -- because a
kinematic bound is still only a bound -- runs the stroke **in physics** and measures the
insert rate. A self-consistent harness proves nothing; the physics is the arbiter.

Ordering discipline
-------------------
**Every CEM search finishes before the block is placed.** These searches call
``write_joint_state_to_sim`` hundreds of times, which teleports the arm and re-opens the
fingers; running one after closing the gripper silently drops the block and reads as a slip.
That bug cost this project more time than anything else, so the two stages are separated
here by construction.

.. code-block:: bash

    python slot/analysis/insertion_feasibility.py --num_envs 256
    python slot/analysis/insertion_feasibility.py --task Rebot-PrecisionSlot-Loose-v0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="V4 achievability: footprint clearance + physical stroke.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--cem_iters", type=int, default=100)
parser.add_argument("--stroke_steps", type=int, default=150)
parser.add_argument("--grip_zs", type=float, nargs="+", default=[0.066, 0.072, 0.078, 0.084, 0.090])
parser.add_argument("--start_depth", type=float, default=0.006,
                    help="block CENTRE depth past the mouth at stroke start [m]")
parser.add_argument("--end_depth", type=float, default=0.045,
                    help="block CENTRE depth commanded at the end of the stroke [m]")
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

HAND_BODIES = ("gripper_end", "gripper_left", "gripper_right", "link6")
CORNERS_PER_BOX = 8

OUT_DIR = args_cli.out_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "envelope"
)


def box_samples(lo: np.ndarray, hi: np.ndarray, k: int = 3) -> np.ndarray:
    """A k^3 lattice over a box -- corners alone miss the middle of a long face."""
    g = [np.linspace(lo[i], hi[i], k) for i in range(3)]
    return np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)


def collect_geometry(stage: Usd.Stage, robot_root: str, body_names: list[str]) -> tuple[dict, str]:
    """body name -> list of local AABBs, attributed to the nearest rigid-body ancestor."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    root = stage.GetPrimAtPath(robot_root)
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
            except Exception:  # noqa: BLE001
                continue
            if not box.IsEmpty():
                out[owner.GetName()].append((np.array(box.GetMin()), np.array(box.GetMax())))
        return out

    got = gather(True)
    if any(got.values()):
        return got, "UsdPhysics.CollisionAPI"
    return gather(False), "render geometry (OVER-estimate)"


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    block = e.scene["block"]
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

    clearance = getattr(e.cfg, "clearance", 0.0015)
    half = mdp.BLOCK_HALF[1] + clearance
    wall_top = mdp.SLOT_FLOOR_Z + mdp.WALL_HEIGHT              # 0.050
    wall_y_in, wall_y_out = half, half + mdp.WALL_THICKNESS    # e.g. 0.0165 .. 0.0245
    mouth_x = mdp.SLOT_CENTER[0] - mdp.SLOT_DEPTH / 2          # 0.210
    back_x = mdp.SLOT_CENTER[0] + mdp.SLOT_DEPTH / 2           # 0.280
    # Depth is measured on the block CENTRE (mdp.insertion_depth reads root_pos), so the
    # stroke endpoints are centre positions. Two bugs in v1 of this script, both of which
    # the original withdrawn probe also had:
    #   * start x was computed as mouth + depth + BLOCK_HALF[0], i.e. treating the argument
    #     as a NOSE depth -- the block actually started at 28.5 mm of a 40 mm threshold, so
    #     it measured an 11.5 mm stroke, not 34 mm;
    #   * the home target was exactly mouth + SUCCESS_DEPTH, i.e. sitting precisely ON a
    #     ">= 40 mm" predicate, so any tracking error or in-hand slip reads as a failure.
    # Now: start at centre depth --start_depth, finish at --end_depth with margin to spare.
    start_x = mouth_x + args_cli.start_depth
    home_x = mouth_x + args_cli.end_depth
    max_center_x = back_x - mdp.BLOCK_HALF[0]                  # 0.2575: nose against the back stop
    if home_x > max_center_x:
        raise SystemExit(f"--end_depth {args_cli.end_depth} puts the block centre at "
                         f"{home_x:.4f} > {max_center_x:.4f} (nose through the back stop)")

    report = {"task": args_cli.task, "clearance_mm": clearance * 1000,
              "wall_top_m": wall_top, "wall_band_y_m": [wall_y_in, wall_y_out],
              "mouth_x": mouth_x, "home_x": home_x}

    import omni.usd  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    geom, source = collect_geometry(stage, "/World/envs/env_0/Robot", list(robot.body_names))
    report["geometry_source"] = source

    print("\n" + "=" * 78)
    print(f"SETUP  task={args_cli.task}  clearance={clearance * 1000:.1f} mm")
    print("=" * 78)
    print(f"  geometry source : {source}")
    print(f"  wall footprint  : x in [{mouth_x:.3f}, {back_x:.3f}], "
          f"|y| in [{wall_y_in:.4f}, {wall_y_out:.4f}], top z = {wall_top:.3f}")
    print(f"  block in slot   : centre z = {mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2]:.3f}, "
          f"top z = {mdp.SLOT_FLOOR_Z + 2 * mdp.BLOCK_HALF[2]:.3f}")

    sample_sets = {}
    for name in HAND_BODIES:
        items = geom.get(name, [])
        if not items:
            continue
        pts = np.concatenate([box_samples(a, b, 3) for a, b in items], axis=0)
        sample_sets[name] = torch.tensor(pts, dtype=torch.float32, device=dev)
    print(f"  hand sample pts : {sum(v.shape[0] for v in sample_sets.values())} "
          f"across {len(sample_sets)} bodies")

    def fk(q_arm: torch.Tensor, q_fing: float) -> dict:
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

        glob, over = [], []
        for name, pts in sample_sets.items():
            bi = robot.body_names.index(name)
            k = pts.shape[0]
            w = bp[:, bi, :].unsqueeze(1) + quat_apply(
                bq[:, bi, :].unsqueeze(1).expand(-1, k, -1), pts.expand(n, -1, -1))
            glob.append(w[..., 2].min(dim=1).values)
            # only geometry standing over a wall can hit one
            inside = ((w[..., 0] >= mouth_x) & (w[..., 0] <= back_x)
                      & (w[..., 1].abs() >= wall_y_in) & (w[..., 1].abs() <= wall_y_out))
            z = torch.where(inside, w[..., 2], torch.full_like(w[..., 2], 1e3))
            over.append(z.min(dim=1).values)
        return {"tcp": tcp, "sep": sep,
                "min_z_global": torch.stack(glob, 1).min(1).values,
                "min_z_over_wall": torch.stack(over, 1).min(1).values}

    def cem(target: torch.Tensor, seed: torch.Tensor, q_fing: float) -> dict:
        """CEM over the 6 arm joints. Position alone is NOT a sufficient specification for
        this arm -- an unconstrained wrist arrives holding the block along its 45 mm length
        instead of across its 30 mm width (measured: insert rate 0 % -> 55-81 % when the
        finger-separation axis is constrained to world y)."""
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)
        mean, std = seed.clone(), torch.full((6,), 0.40, device=dev)
        best = {"cost": 1e9}
        for _ in range(args_cli.cem_iters):
            q = (mean + std * torch.randn((n, 6), device=dev)).clamp(lo, hi)
            r = fk(q, q_fing)
            pos_err = (r["tcp"] - target).norm(dim=1)
            align_err = 1.0 - (r["sep"] @ y_axis).abs()
            cost = pos_err + 0.25 * align_err
            elite = q[cost.argsort()[: max(8, n // 20)]]
            mean, std = elite.mean(0), elite.std(0).clamp(min=0.008)
            i = int(cost.argmin())
            if float(cost[i]) < best["cost"]:
                best = {"cost": float(cost[i]), "q": q[i].clone(), "pos_err": float(pos_err[i]),
                        "align_err": float(align_err[i]),
                        "min_z_global": float(r["min_z_global"][i]),
                        "min_z_over_wall": float(r["min_z_over_wall"][i])}
        return best

    with torch.inference_mode():
        env.reset()
        seed = q_default[arm_dof].clone()

        # ---------------- STAGE 1: all CEM searches, BEFORE any block placement -----------
        print("\n" + "=" * 78)
        print("STAGE 1 -- kinematics: footprint-restricted wall clearance vs grip height")
        print("=" * 78)
        print("  'global' = min z of any hand geometry;  'over wall' = min z of hand geometry")
        print("  that is actually standing inside the wall footprint (the one that matters).")
        print(f"  {'grip z':>8} {'where':>6} {'tcp_err':>9} {'align':>8} {'global':>9} {'over wall':>11} {'clr':>8}")
        plans, kin = {}, []
        for gz in args_cli.grip_zs:
            warm = seed.clone()
            row = {}
            for tag, x in (("pre", start_x), ("home", home_x)):
                tgt = torch.tensor([x, 0.0, gz], device=dev)
                b = cem(tgt, warm, q_fing=0.016)
                warm = b["q"].clone()
                b["target"] = tgt.cpu().tolist()
                row[tag] = b
                clr = b["min_z_over_wall"] - wall_top
                free = b["min_z_over_wall"] > 999
                kin.append({"grip_z": gz, "where": tag, "tcp_err_mm": b["pos_err"] * 1000,
                            "align_err": b["align_err"], "min_z_global_mm": b["min_z_global"] * 1000,
                            "over_wall": None if free else b["min_z_over_wall"] * 1000,
                            "clearance_mm": None if free else clr * 1000})
                ow = "  none " if free else f"{b['min_z_over_wall'] * 1000:9.1f}m"
                cl = "   n/a " if free else f"{clr * 1000:+7.1f}m"
                print(f"  {gz:8.3f} {tag:>6} {b['pos_err'] * 1000:8.2f}m {b['align_err']:8.4f} "
                      f"{b['min_z_global'] * 1000:8.1f}m {ow} {cl}")
            plans[gz] = row
        report["kinematics"] = kin

        def act(q_arm: torch.Tensor, close: bool) -> torch.Tensor:
            a = torch.zeros((n, 7), device=dev)
            a[:, :6] = (q_arm - q_default[arm_dof].unsqueeze(0)) / 0.5
            a[:, 6] = -1.0 if close else 1.0
            return a

        def read_tcp() -> torch.Tensor:
            """Achieved TCP from physics -- no joint writes, so the sim state survives."""
            bp = torch.as_tensor(robot.data.body_pos_w.torch, device=dev) - e.scene.env_origins.unsqueeze(1)
            bq = torch.as_tensor(robot.data.body_quat_w.torch, device=dev)
            return bp[:, end_idx, :] + quat_apply(bq[:, end_idx, :], offs)

        def park(q_arm: torch.Tensor, q_fing: float) -> None:
            q = q_default.unsqueeze(0).repeat(n, 1)
            q[:, arm_dof] = q_arm
            q[:, fing_dof] = q_fing
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            e.sim.forward()
            robot.update(0.0)

        def run_stroke(q_pre: torch.Tensor, q_home: torch.Tensor) -> None:
            for s in range(args_cli.stroke_steps):
                f = min(1.0, s / (args_cli.stroke_steps * 0.7))
                env.step(act(((1 - f) * q_pre + f * q_home).unsqueeze(0).repeat(n, 1), close=True))

        # ------------- STAGE 2a: does the EMPTY hand fit through the slot? ---------------
        # The 'over wall' column above came from RENDER meshes (no CollisionAPI prims were
        # found), which is an over-estimate of the collider -- and it claims a negative
        # clearance at every grip height, which cannot be reconciled with a block that
        # physically travelled 40 mm. So settle it in physics instead: park the block far
        # away, stroke the closed empty gripper through the slot, and measure how well the
        # TCP tracks. A hand jammed on a wall cannot track its commanded pose.
        print("\n" + "=" * 78)
        print("STAGE 2a -- physics collision test: stroke the EMPTY hand through the slot")
        print("=" * 78)
        print("  large tracking error => the hand is fouling the fixture at that grip height")
        print(f"  {'grip z':>8} {'tcp err':>10} {'dz':>9} {'verdict':>12}")
        far = torch.tensor([0.45, 0.30, 0.035], device=dev).repeat(n, 1)
        ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
        empty = []
        for gz in args_cli.grip_zs:
            block.write_root_state_to_sim(
                torch.cat([far + e.scene.env_origins, ident, torch.zeros((n, 6), device=dev)], dim=1))
            park(plans[gz]["pre"]["q"], 0.0)
            run_stroke(plans[gz]["pre"]["q"], plans[gz]["home"]["q"])
            tcp = read_tcp()
            tgt = torch.tensor(plans[gz]["home"]["target"], device=dev)
            err = (tcp - tgt).norm(dim=1)
            dz = (tcp[:, 2] - tgt[2])
            clean = float(err.mean()) < 0.004
            empty.append({"grip_z": gz, "tcp_err_mm": float(err.mean()) * 1000,
                          "dz_mm": float(dz.mean()) * 1000, "clean": clean})
            print(f"  {gz:8.3f} {float(err.mean()) * 1000:9.2f}m {float(dz.mean()) * 1000:+8.2f}m "
                  f"{'clear' if clean else 'FOULING':>12}")
        report["empty_hand"] = empty

        # ---------------- STAGE 2b: physics with the block held ------------------------
        print("\n" + "=" * 78)
        print("STAGE 2b -- physics: hold the block and execute the stroke")
        print("=" * 78)
        print(f"  block CENTRE starts at depth {args_cli.start_depth * 1000:.0f} mm and is "
              f"commanded to {args_cli.end_depth * 1000:.0f} mm; success needs "
              f"{mdp.SUCCESS_DEPTH * 1000:.0f} mm -> a real "
              f"{(args_cli.end_depth - args_cli.start_depth) * 1000:.0f} mm stroke")
        print(f"  {'grip z':>8} {'gap mm':>8} {'held':>9} {'ins':>9} {'rate':>7} "
              f"{'depth mm':>9} {'lat mm':>8} {'yaw':>7} {'tcp err':>9}")
        strokes = []
        for gz in args_cli.grip_zs:
            q_pre, q_home = plans[gz]["pre"]["q"], plans[gz]["home"]["q"]
            park(q_pre, 0.045)
            tcp0 = read_tcp()

            # block centred under the grip point, bottom just clear of the slot floor, with a
            # little lateral + yaw scatter so this is not one lucky alignment
            bpos = tcp0.clone()
            bpos[:, 0] = start_x
            bpos[:, 2] = mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2] + 0.002
            bpos[:, 1] += (torch.rand(n, device=dev) - 0.5) * 0.002
            yaw = (torch.rand(n, device=dev) - 0.5) * 0.04
            quat = torch.zeros((n, 4), device=dev)
            quat[:, 2], quat[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
            block.write_root_state_to_sim(
                torch.cat([bpos + e.scene.env_origins, quat, torch.zeros((n, 6), device=dev)], dim=1))

            # close immediately: any open-gripper settle just lets the block topple or slide
            for _ in range(40):
                env.step(act(q_pre.unsqueeze(0).repeat(n, 1), close=True))

            jp = torch.as_tensor(robot.data.joint_pos.torch, device=dev)
            gap_mm = (1.0035 * (jp[:, fing_dof[0]] + jp[:, fing_dof[1]]) - 0.00125) * 1000
            bp_b = torch.as_tensor(block.data.root_pos_w.torch, device=dev) - e.scene.env_origins
            # A real retention test. "Object within 80 mm of the TCP" once scored an
            # untouched 107 mm block in an 89 mm gripper at 100 %; the finger gap cannot be
            # faked -- 30 mm of block between the pads reads as 30 mm.
            held = ((gap_mm > 26.0) & (gap_mm < 34.0)
                    & ((bp_b[:, :2] - tcp0[:, :2]).norm(dim=1) < 0.02))

            run_stroke(q_pre, q_home)
            depth, lat = mdp.insertion_depth(e), mdp.lateral_error(e)
            yaw_e, ok = mdp.yaw_error(e), mdp.is_inserted(e)
            tgt = torch.tensor(plans[gz]["home"]["target"], device=dev)
            terr = (read_tcp() - tgt).norm(dim=1)
            rec = {"grip_z": gz, "held": int(held.sum()), "n": n, "inserted": int(ok.sum()),
                   "insert_rate": float(ok.float().mean()),
                   "gap_mm_mean": float(gap_mm.mean()), "gap_mm_std": float(gap_mm.std()),
                   "gap_mm_p10": float(torch.quantile(gap_mm, 0.1)),
                   "gap_mm_p90": float(torch.quantile(gap_mm, 0.9)),
                   "depth_mm_mean": float(depth.mean()) * 1000,
                   "depth_mm_p10": float(torch.quantile(depth, 0.1)) * 1000,
                   "lateral_mm_mean": float(lat.mean()) * 1000,
                   "yaw_rad_mean": float(yaw_e.mean()),
                   "tcp_err_mm": float(terr.mean()) * 1000}
            strokes.append(rec)
            print(f"  {gz:8.3f} {rec['gap_mm_mean']:8.1f} {rec['held']:5d}/{n:<3d} "
                  f"{rec['inserted']:5d}/{n:<3d} {rec['insert_rate']:7.3f} "
                  f"{rec['depth_mm_mean']:9.1f} {rec['lateral_mm_mean']:8.2f} "
                  f"{rec['yaw_rad_mean']:7.3f} {rec['tcp_err_mm']:8.2f}m")
        report["strokes"] = strokes

        os.makedirs(OUT_DIR, exist_ok=True)
        tag = args_cli.task.replace("/", "_")
        path = os.path.join(OUT_DIR, f"insertion_feasibility_{tag}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        best = max(strokes, key=lambda r: r["insert_rate"]) if strokes else None
        print("\n" + "=" * 78)
        if best and best["insert_rate"] > 0.5:
            print(f"  VERDICT: the arm CAN execute the insertion stroke -- best "
                  f"{best['insert_rate'] * 100:.1f}% at grip z = {best['grip_z']:.3f} m.")
            print("           Remaining difficulty is the approach and the grasp, not the geometry.")
        else:
            r = best["insert_rate"] * 100 if best else 0.0
            d = best["depth_mm_mean"] if best else 0.0
            print(f"  VERDICT: the stroke FAILS (best {r:.1f}%, mean depth {d:.1f} mm vs "
                  f"{mdp.SUCCESS_DEPTH * 1000:.0f} mm needed).")
            print("           Check STAGE 1's 'over wall' column: if it is negative at every")
            print("           grip height, the fixture geometry is infeasible as authored.")
        print("=" * 78)
        print(f"[feas] wrote {path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
