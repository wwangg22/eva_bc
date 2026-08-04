# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P01 -- Finger geometry: the numbers that decide which strategies are even possible.

REWRITTEN 2026-08-02 against the binary-aperture finding (`06_EXPERT_DESIGN.md S3`). The
first version asked "is the finger thin enough to thread a 12 mm gap while holding a 30 mm
block?", which presumes an intermediate aperture. `BinaryJointPositionActionCfg` offers only
q = 0.045 (89.07 mm clear) or q = 0.000. There is no such command. The questions that
actually matter are different, and sharper:

**Q1a -- outward finger thickness `t`.** The open gripper's inner faces are pinned at
+/-44.53 mm. Treating the fingers as zero-thickness planes, the two alignment windows where
both land in free gaps are each 6.94 mm wide (nominal row). A real finger of outward
thickness `t` shrinks each window to `6.94 - 2t`, so:

        **the pair-capture strategy (A) dies outright if t > 3.47 mm.**

That is a hard, falsifiable threshold and it is not recorded anywhere in either repo.

**Q1b -- closed-gripper blade width.** A shut gripper is a blade of some width `w_closed`.
If `w_closed` clears a 12 mm free gap it can be inserted between blocks, which is what
strategies C/D/E need for a pusher. If it does not, no in-row pushing is possible at all.

**Q1c -- how far below the TCP does finger geometry reach?** C9 puts the *TCP* floor at
~44 mm, but the fingers are what touch things. Tip-vs-slide is `h > b/(2 mu)`: 16.7 mm for a
push across the row, 20.0 mm for a push along it. If the fingertips reach far enough below
the TCP, a push can land *below* the tipping threshold and slide the block instead of
toppling it -- which would license strategies B and C. This single number is the difference
between "every reachable push topples" and "pushing is a usable primitive".

**Q5 -- the friction actually resolved on the blocks and the table**, which sets every
`h_crit` above. The blocks author `static_friction = 0.9`; the table is a stock Nucleus USD
and its material has never been read.

Sections
--------
1. Finger collider extents in the link frame (a *local* bound is safe to read from USD --
   C8's warning is about *world* bounds, which PhysX never writes back).
2. Outer width vs joint value, composed from (1) and the physics-view body poses. -> Q1a/Q1b
3. Finger geometry below the TCP. -> Q1c
4. Resolved physics materials. -> Q5
5. Verdict against the three thresholds above.

Usage
-----
    python eva_bc/clutter/probes/p01_gripper_geometry.py --num_envs 4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the gripper's finger geometry.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p01_gripper_geometry.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kin import TCP_OFFSET, ArmKin, Q_CLOSE, Q_OPEN, clear_gap  # noqa: E402

FINGER_BODIES = ("gripper_left", "gripper_right")
BLOCK_W = 0.030       # block width across the row (y)
BLOCK_D = 0.036       # block depth toward the robot (x)
FREE_GAP = 0.012      # nominal free gap between neighbours
#: authored on the blocks (clutter_env_cfg._block)
MU_BLOCK = 0.9
#: the two alignment windows are 6.94 mm wide for zero-thickness fingers; see the module docstring
WINDOW_0 = 0.00694


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def walk(root: Usd.Prim):
    """Traverse a subtree **including instance proxies**.

    The arm USD splits `Geometry` and `Physics` scopes and the stock table is
    `table_instanceable.usd`; a plain `Usd.PrimRange` stops at the instance boundary and
    silently reports zero colliders and zero materials. That is not "there are none" -- it
    is "you did not look inside". Getting this wrong is what made the first run of this
    probe report a 19.76 mm finger thickness measured off render geometry.
    """
    return Usd.PrimRange(root, Usd.TraverseInstanceProxies())


def find_prim(root: Usd.Prim, name: str) -> Usd.Prim | None:
    for p in walk(root):
        if p.GetName() == name:
            return p
    return None


def colliders_under(root: Usd.Prim, key: str) -> list[Usd.Prim]:
    """Every collision-enabled prim whose path contains `key`, anywhere under `root`."""
    return [p for p in walk(root)
            if p.HasAPI(UsdPhysics.CollisionAPI) and key in str(p.GetPath())]


def rel_bound(cache: UsdGeom.BBoxCache, prim: Usd.Prim,
              ancestor: Usd.Prim | None = None) -> np.ndarray | None:
    """Axis-aligned bound of `prim`'s subtree, expressed in `ancestor`'s frame.

    With `ancestor=None` this is `ComputeLocalBound`, i.e. the subtree geometry in `prim`'s
    own frame with `prim`'s local-to-parent transform stripped -- exactly the link-frame
    extents we want, and independent of wherever PhysX has since moved the body.
    """
    try:
        box = cache.ComputeRelativeBound(prim, ancestor) if ancestor is not None \
            else cache.ComputeLocalBound(prim)
    except Exception:
        return None
    box = box.ComputeAlignedRange()
    if box.IsEmpty():
        return None
    return np.array([[*box.GetMin()], [*box.GetMax()]], dtype=float)


def quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    K = ArmKin(env)
    robot = K.robot
    n = K.n

    stage = get_current_stage()
    R: dict = {"task": args_cli.task}

    print("\n" + "=" * 92)
    print("P01 -- FINGER GEOMETRY (rewritten against the BINARY aperture)")
    print("=" * 92)

    lim = _t(robot.data.joint_pos_limits)[0]
    print(f"\n  joint_left  limits  {lim[K.fing_dof[0], 0]:+.4f} .. {lim[K.fing_dof[0], 1]:+.4f} m")
    print(f"  joint_right limits  {lim[K.fing_dof[1], 0]:+.4f} .. {lim[K.fing_dof[1], 1]:+.4f} m")
    print(f"  commandable apertures: OPEN q={Q_OPEN} (gap {clear_gap(2 * Q_OPEN) * 1000:.2f} mm), "
          f"CLOSE q={Q_CLOSE}. Nothing in between.")
    print(f"  body_names: {list(robot.body_names)}")
    R["joint_limits"] = {"joint_left": [float(lim[K.fing_dof[0], 0]), float(lim[K.fing_dof[0], 1])],
                         "joint_right": [float(lim[K.fing_dof[1], 0]), float(lim[K.fing_dof[1], 1])]}
    R["body_names"] = list(robot.body_names)

    # ------------------------------------------------------------- 1. link-frame extents
    print("\n" + "-" * 92)
    print("1. FINGER COLLIDER EXTENTS IN THE LINK FRAME")
    print("-" * 92)

    root_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    if not root_prim or not root_prim.IsValid():
        root_prim = stage.GetPrimAtPath("/World/envs/env_0")
    cache_d = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
                               useExtentsHint=False)
    cache_g = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide,
                                UsdGeom.Tokens.proxy, UsdGeom.Tokens.render],
                               useExtentsHint=False)

    R["finger_local_bounds"] = {}
    for b in FINGER_BODIES:
        prim = find_prim(root_prim, b)
        if prim is None:
            print(f"  {b}: PRIM NOT FOUND under {root_prim.GetPath()}")
            continue
        print(f"\n  {b}  ->  {prim.GetPath()}")
        entry = {"path": str(prim.GetPath())}
        for tag, cache in (("default", cache_d), ("all_purposes", cache_g)):
            lb = rel_bound(cache, prim)
            if lb is None:
                continue
            size = lb[1] - lb[0]
            print(f"     {tag:>13}: min {np.round(lb[0] * 1000, 2)}  max {np.round(lb[1] * 1000, 2)}"
                  f"   size {np.round(size * 1000, 2)} mm")
            entry[tag] = {"min_mm": (lb[0] * 1000).tolist(), "max_mm": (lb[1] * 1000).tolist()}
        # collision-enabled prims for this finger, wherever they live in the tree
        # (the arm USD keeps a separate /Physics scope), expressed in the BODY frame
        cols = []
        for p in colliders_under(root_prim, b):
            lb = rel_bound(cache_d, p, prim)
            cols.append({"path": str(p.GetPath()), "type": p.GetTypeName(),
                         "min_mm": None if lb is None else (lb[0] * 1000).tolist(),
                         "max_mm": None if lb is None else (lb[1] * 1000).tolist()})
        print(f"     collision prims: {len(cols)}")
        for c in cols:
            print(f"        [{c['type']:>10}] {c['path'].split('/')[-1]:>22}  "
                  f"min {None if c['min_mm'] is None else np.round(c['min_mm'], 2)}  "
                  f"max {None if c['max_mm'] is None else np.round(c['max_mm'], 2)}")
        entry["collision_prims"] = cols
        # --- 1b. the AABB above is the bound of a MESH, and an AABB of a non-box shape
        # includes air. C3 measured a clear gap of 89.07 mm at q = 0.045 while the AABB
        # implies 51.7 mm, so the AABB is definitively not the jaw. Read the points.
        pts_all = []
        for p in colliders_under(root_prim, b):
            m = UsdGeom.Mesh(p)
            pa = m.GetPointsAttr() if m else None
            pts = pa.Get() if pa else None
            approx = p.GetAttribute("physics:approximation")
            approx = approx.Get() if approx and approx.IsValid() else None
            if pts is None:
                print(f"     [1b] {p.GetPath().name}: no points (type {p.GetTypeName()})")
                continue
            a = np.array([[q[0], q[1], q[2]] for q in pts], dtype=float)
            pts_all.append(a)
            print(f"     [1b] {p.GetPath().name}: {len(a)} pts, physics:approximation="
                  f"{approx!r}")
            print(f"          point AABB min {np.round(a.min(0) * 1000, 2)}  "
                  f"max {np.round(a.max(0) * 1000, 2)} mm")
            entry["mesh_points"] = {"n": len(a), "approximation": str(approx),
                                    "min_mm": (a.min(0) * 1000).tolist(),
                                    "max_mm": (a.max(0) * 1000).tolist()}
        if pts_all:
            a = np.concatenate(pts_all, 0)
            # slice along each axis to expose the shape: an L-shaped finger has most of its
            # mass on one side, and a slab has none of this structure
            for ax, nm in ((0, "x"), (1, "y"), (2, "z")):
                hist, edges = np.histogram(a[:, ax] * 1000, bins=8)
                entry[f"hist_{nm}"] = {"counts": hist.tolist(),
                                       "edges_mm": np.round(edges, 2).tolist()}
                print(f"          {nm}: " + "  ".join(
                    f"[{edges[i]:.0f},{edges[i + 1]:.0f}):{hist[i]}" for i in range(8)))
        R["finger_local_bounds"][b] = entry

    def body_box(b: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Union of the collision-prim bounds in the body frame; falls back to the subtree."""
        ent = R["finger_local_bounds"].get(b)
        if not ent:
            return None
        mins = [np.array(c["min_mm"]) / 1000 for c in ent.get("collision_prims", []) if c["min_mm"]]
        maxs = [np.array(c["max_mm"]) / 1000 for c in ent.get("collision_prims", []) if c["max_mm"]]
        if not mins:
            d = ent.get("default") or ent.get("all_purposes")
            if not d:
                return None
            return np.array(d["min_mm"]) / 1000, np.array(d["max_mm"]) / 1000
        return np.min(np.stack(mins), 0), np.max(np.stack(maxs), 0)

    boxes = {b: body_box(b) for b in FINGER_BODIES}
    if any(v is None for v in boxes.values()):
        print("\n  [p01] FATAL: could not read finger collider bounds from USD. "
              "Sections 2/3/5 cannot run.")
        R["error"] = "no finger bounds"

    # --------------------------------------------------- 2. world outer width vs aperture
    print("\n" + "-" * 92)
    print("2. WORLD OUTER WIDTH vs APERTURE  ->  Q1a (window survival) and Q1b (blade width)")
    print("-" * 92)
    print("   Intermediate q values are NOT commandable; they are written directly here only")
    print("   to expose the linear geometry. The two rows that matter are q=0.045 and q=0.000.")
    print()
    print(f"   {'q [m]':>7} | {'origin sep':>10} | {'clear gap':>10} | {'outer y':>9} | "
          f"{'t/finger':>9} | {'span x':>8} | {'span z':>8}")
    print("   " + "-" * 78)

    def world_span(q: float) -> dict | None:
        js = K.q_default.unsqueeze(0).repeat(n, 1)
        js[:, K.fing_dof] = q
        robot.write_joint_state_to_sim(js, torch.zeros_like(js))
        e.sim.forward()
        robot.update(0.0)
        bp = _t(robot.data.body_pos_w)[0].cpu().numpy()
        bq = _t(robot.data.body_quat_w)[0].cpu().numpy()
        org = e.scene.env_origins[0].cpu().numpy()
        lo, hi, per = [], [], {}
        for b in FINGER_BODIES:
            if boxes[b] is None:
                return None
            mn, mx = boxes[b]
            corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                                for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
            i = K.i_left if b == "gripper_left" else K.i_right
            w = corners @ quat_to_R(bq[i]).T + bp[i] - org
            lo.append(w.min(0))
            hi.append(w.max(0))
            per[b] = {"origin": (bp[i] - org).tolist(), "lo": w.min(0).tolist(),
                      "hi": w.max(0).tolist()}
        return {"lo": np.min(np.stack(lo), 0), "hi": np.max(np.stack(hi), 0), "per": per}

    rows = []
    for q in (0.0, 0.010, 0.020, 0.030, 0.045):
        ws = world_span(q)
        if ws is None:
            break
        bp = _t(robot.data.body_pos_w)[0].cpu().numpy()
        sep = float(np.linalg.norm(bp[K.i_left] - bp[K.i_right]))
        gap = clear_gap(2 * q)
        span = ws["hi"] - ws["lo"]
        # outward thickness per finger: (outer width - clear opening) / 2
        t_out = (span[1] - gap) / 2.0
        rows.append({"q": q, "origin_sep_m": sep, "clear_gap_m": float(gap),
                     "outer_y_m": float(span[1]), "t_out_m": float(t_out),
                     "span_x_m": float(span[0]), "span_z_m": float(span[2]),
                     "lo_m": ws["lo"].tolist(), "hi_m": ws["hi"].tolist(), "per": ws["per"]})
        print(f"   {q:7.3f} | {sep * 1000:10.3f} | {gap * 1000:10.3f} | {span[1] * 1000:9.2f} | "
              f"{t_out * 1000:9.2f} | {span[0] * 1000:8.2f} | {span[2] * 1000:8.2f}")
    R["aperture_rows"] = rows

    # ------------------------------------------------ 3. finger geometry below the TCP (Q1c)
    print("\n" + "-" * 92)
    print("3. HOW FAR BELOW THE TCP DOES FINGER GEOMETRY REACH?  ->  Q1c (push contact height)")
    print("-" * 92)
    print("   Reported in the `gripper_end` LINK frame, which is orientation-independent:")
    print("   local -X is the approach axis, local +Y the opening axis. The TCP sits at")
    print(f"   ({TCP_OFFSET[0] * 1000:.1f}, 0, 0) mm in this frame. Combine with the achieved")
    print("   approach tilt (P02) to get the true contact height above the table.")
    print()
    R["below_tcp"] = {}
    for q, tag in ((Q_OPEN, "open"), (Q_CLOSE, "closed")):
        ws = world_span(q)
        if ws is None:
            break
        bp = _t(robot.data.body_pos_w)[0].cpu().numpy()
        bq = _t(robot.data.body_quat_w)[0].cpu().numpy()
        org = e.scene.env_origins[0].cpu().numpy()
        Rend = quat_to_R(bq[K.i_end])
        p_end = bp[K.i_end] - org
        # finger collider corners -> gripper_end link frame
        loc = []
        for b in FINGER_BODIES:
            mn, mx = boxes[b]
            corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                                for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
            i = K.i_left if b == "gripper_left" else K.i_right
            w = corners @ quat_to_R(bq[i]).T + bp[i] - org
            loc.append((w - p_end) @ Rend)
        loc = np.concatenate(loc, 0)
        lmin, lmax = loc.min(0), loc.max(0)
        tcp_l = np.array(TCP_OFFSET)
        print(f"   {tag:>6}: finger geometry in the end-link frame [mm]")
        print(f"           x {lmin[0] * 1000:7.2f} .. {lmax[0] * 1000:7.2f}   "
              f"y {lmin[1] * 1000:7.2f} .. {lmax[1] * 1000:7.2f}   "
              f"z {lmin[2] * 1000:7.2f} .. {lmax[2] * 1000:7.2f}")
        print(f"           relative to the TCP: fingertips reach "
              f"{(lmax[0] - tcp_l[0]) * 1000:6.2f} mm forward along -(-X), and the jaw spans "
              f"{(lmax[1] - lmin[1]) * 1000:.2f} mm across the opening axis")
        R["below_tcp"][tag] = {"link_min_m": lmin.tolist(), "link_max_m": lmax.tolist(),
                               "tcp_link_m": tcp_l.tolist()}
        # world-frame reading at the home pose, kept for continuity
        tcp = K.tcp_now()[0].cpu().numpy()
        R["below_tcp"][tag]["home_tcp_z_m"] = float(tcp[2])
        R["below_tcp"][tag]["home_low_z_m"] = float(ws["lo"][2])
        R["below_tcp"][tag]["home_drop_m"] = float(tcp[2] - ws["lo"][2])

    # ------------------------------------------------------------ 4. physics materials (Q5)
    print("\n" + "-" * 92)
    print("4. RESOLVED PHYSICS MATERIALS  ->  Q5 (friction sets every h_crit)")
    print("-" * 92)
    mats: dict = {}
    for path in ("/World/envs/env_0/Table", "/World/envs/env_0/Target",
                 "/World/envs/env_0/Distractor0", "/World/GroundPlane"):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            print(f"   {path}: NOT FOUND")
            continue
        found = []
        seen = set()

        def add(p, src):
            if not p or not p.IsValid() or str(p.GetPath()) in seen:
                return
            if not p.HasAPI(UsdPhysics.MaterialAPI):
                return
            seen.add(str(p.GetPath()))
            m = UsdPhysics.MaterialAPI(p)
            found.append({"path": str(p.GetPath()), "via": src,
                          "static_friction": m.GetStaticFrictionAttr().Get(),
                          "dynamic_friction": m.GetDynamicFrictionAttr().Get(),
                          "restitution": m.GetRestitutionAttr().Get()})

        for p in walk(prim):
            add(p, "inline")
            rel = p.GetRelationship("material:binding:physics")
            if rel:
                for tgt in rel.GetTargets():
                    add(stage.GetPrimAtPath(tgt), f"bound<-{p.GetName()}")
        print(f"   {path}: {len(found)} material(s)")
        for f in found[:8]:
            print(f"      mu_s={f['static_friction']}  mu_d={f['dynamic_friction']}  "
                  f"e={f['restitution']}   [{f['via']}]  {f['path'].split('/')[-1]}")
        if not found:
            print("      (none -- PhysX default material applies: mu_s = mu_d = 0.5, e = 0.0)")
        mats[path] = found
    R["materials"] = mats

    # ------------------------------------------------------------------------ 5. verdict
    print("\n" + "=" * 92)
    print("5. VERDICT")
    print("=" * 92)
    V: dict = {}
    open_row = next((r for r in rows if r["q"] == Q_OPEN), None)
    shut_row = next((r for r in rows if r["q"] == Q_CLOSE), None)

    if open_row:
        t = open_row["t_out_m"]
        V["t_out_m"] = t
        win = WINDOW_0 - 2 * t
        V["window_nominal_m"] = win
        print(f"\n   Q1a  outward finger thickness t = {t * 1000:.2f} mm per side")
        print(f"        zero-thickness alignment window : {WINDOW_0 * 1000:.2f} mm")
        print(f"        window with the real fingers    : {win * 1000:+.2f} mm"
              f"   (= 6.94 - 2t)")
        if win > 0.001:
            print(f"        -> STRATEGY A (pair-capture) SURVIVES with {win * 1000:.2f} mm of")
            print("           alignment tolerance at the NOMINAL row. That is the whole budget:")
            print("           spawn jitter, yaw jitter and TCP error all come out of it.")
        else:
            print("        -> STRATEGY A (pair-capture) IS DEAD. The fingers are too thick for")
            print("           any placement that puts both of them in free gaps. No open-gripper")
            print("           placement at the nominal row is collision-free, at any alignment.")
        # how much the row would have to open up for the window to be usable
        need = (2 * t + 0.002 - WINDOW_0) / 2.0  # +2 mm of usable tolerance
        if need > 0:
            print(f"        -> each free gap would have to widen by {need * 1000:.2f} mm "
                  f"(to {(FREE_GAP + need) * 1000:.2f} mm) for a 2 mm-tolerance window.")
            V["gap_widening_needed_m"] = float(need)

    if shut_row:
        w_closed = shut_row["outer_y_m"]
        V["w_closed_m"] = w_closed
        print(f"\n   Q1b  closed-gripper blade width = {w_closed * 1000:.2f} mm")
        for tag, g in (("nominal 12 mm gap", FREE_GAP), ("worst measured 7 mm", 0.007),
                       ("Tight-v0 6 mm", 0.006)):
            fits = w_closed < g
            print(f"        vs {tag:<20}: {'FITS' if fits else 'does NOT fit':<12} "
                  f"(margin {(g - w_closed) * 1000:+.2f} mm)")
        V["blade_fits_12mm"] = bool(w_closed < FREE_GAP)
        print("        A blade that fits can be inserted BETWEEN blocks (strategies C/D/E).")
        print("        A blade that does not fit can still push a block's exposed face.")

    # ---- Q1c: tip-vs-slide, with the friction actually resolved rather than assumed
    tab = mats.get("/World/envs/env_0/Table", [])
    mu_table = min((m["static_friction"] for m in tab if m["static_friction"] is not None),
                   default=None)
    if mu_table is None:
        mu_table = 0.5  # PhysX default material
        mu_src = "PhysX default (the table authors NO physics material)"
    else:
        mu_src = "read off the table's bound material"
    #: PhysX default friction combine mode is `average`
    mu_eff = 0.5 * (MU_BLOCK + mu_table)
    hc_y, hc_x = BLOCK_W / (2 * mu_eff), BLOCK_D / (2 * mu_eff)
    V.update({"mu_block": MU_BLOCK, "mu_table": float(mu_table), "mu_eff_avg": float(mu_eff),
              "h_crit_y_m": float(hc_y), "h_crit_x_m": float(hc_x)})
    print(f"\n   Q5   block mu_s = {MU_BLOCK}, table mu_s = {mu_table}  [{mu_src}]")
    print(f"        PhysX combines by AVERAGE by default -> effective mu = {mu_eff:.3f}")
    print(f"        This is NOT the 0.9 the analysis assumed. Every h_crit moves:")
    print(f"        tip-vs-slide across the row (b={BLOCK_W * 1000:.0f} mm): "
          f"{hc_y * 1000:.2f} mm   (was {BLOCK_W / (2 * MU_BLOCK) * 1000:.2f} mm at mu=0.9)")
    print(f"        tip-vs-slide along the row  (b={BLOCK_D * 1000:.0f} mm): "
          f"{hc_x * 1000:.2f} mm   (was {BLOCK_D / (2 * MU_BLOCK) * 1000:.2f} mm at mu=0.9)")

    if "closed" in R.get("below_tcp", {}):
        drop = R["below_tcp"]["closed"]["home_drop_m"]
        print(f"\n   Q1c  at the HOME wrist pose the lowest finger geometry sits "
              f"{drop * 1000:.2f} mm below the TCP.")
        print("        That drop is orientation-dependent, so it is NOT a contact height. The")
        print("        real number is P02's achieved `low_z` at a grasp-height pose; this run")
        print("        only bounds the jaw's own extent. Do not compare it to h_crit here.")
        V["home_drop_m"] = float(drop)

    R["verdict"] = V
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(R, f, indent=2, default=str)
    print(f"\n[p01] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
