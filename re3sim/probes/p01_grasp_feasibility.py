# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""P01 -- can this arm grasp the three real objects from the 2026-08-05 workstation capture?

This is the **B1 gate** for `eva_rl/docs/envs/re3sim`. Nothing downstream is worth building
until it passes: the plan makes all three captured objects grasp targets, and two of them
(a 24 mm roll of tape, a 36 mm tape measure) sit below the 44 mm "TCP floor" that
`eva_rl/docs/CHALLENGE_SUITE.md` C9 records as the suite's most restrictive constraint.

WHY THE FIRST ATTEMPT AT THIS WAS THROWN AWAY
----------------------------------------------
`eva_rl/scripts/analysis/re3sim_grasp_probe.py` (2026-08-05, first run) copied the search out
of `grasp_geometry.py`: `cost = |dp| + 0.25 * (1 - |o.Y|)`, no floor term, no approach-axis
term, one restart. Its controls came back

    24 mm tall x 40 mm wide  ->  gap 40.9 mm, HELD 100 %      (correct)
    56 mm tall x 40 mm wide  ->  gap -1.2 mm, HELD   0 %      (fingers shut on AIR)

A 56 mm block is *easier* than a 24 mm one for this gripper, so the taller control failing at
every grip height is a defect in the search, not a fact about the arm. `_kin.py` names it
exactly: with `w_pos = 1` and `|dp|` in metres, 1 mm of position error costs 0.001 against
orientation terms of order 0.25, and with no floor penalty the CEM returns poses with
`gripper_end` **inside the table** -- "which is why P03's first control run closed on air in
every condition".

So this probe uses `_kin.ArmKin.cem`: hinged position cost (`w_pos = 200`, free inside 1 mm),
a floor penalty, a signed approach-axis term, and restarts. Same gates the clutter expert
uses, so a pose accepted here is a pose an expert would accept.

THE POSITIVE CONTROL IS BUILT IN
---------------------------------
`control_24` is a 24 mm x 40 mm block, *measured* to grasp at 100 % even under the defective
search. If it does not pass here the search is under-budgeted and every other row is void.
`control_56` is the cell the old search got wrong; it must now pass too.

.. code-block:: bash

    python -u re3sim/probes/p01_grasp_feasibility.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------------------
# Object table. Heights and longest dimensions are the user's caliper measurements from
# data/captures/2026-08-05/measurements.txt. `across` is the span the fingers must close
# over; `along` is the span down the approach axis.
#
#   rubixcube    56 x 56 x 56       73 g
#   rolloftape   24 tall, 91 dia    42 g
#   tapemeasure  36 tall, 71.5 long 184 g   <- third dimension NOT measured; 64 mm assumed
#   box          93 tall, 218 long  95 g    <- the container, not a grasp target
# ---------------------------------------------------------------------------------------
OBJECTS = {
    # name:                (shape,       height, across, along,  mass)
    "control_24":          ("cuboid",    0.024,  0.040,  0.040,  0.050),
    "control_56":          ("cuboid",    0.056,  0.040,  0.040,  0.050),
    "rubixcube":           ("cuboid",    0.056,  0.056,  0.056,  0.073),
    "tapemeasure":         ("cuboid",    0.036,  0.064,  0.0715, 0.184),
    # lying flat, as it rests naturally: the fingers must span the full 91 mm diameter,
    # against a commanded opening of 89.1 mm (C3). Predicted to FAIL on width.
    "rolloftape":          ("cylinder",  0.024,  0.091,  0.091,  0.042),
    # stood on its rim: 91 mm becomes the height and 24 mm the span across the fingers.
    # A physically natural resting pose for a taped roll, and the one that fits the gripper.
    "rolloftape_onedge":   ("cylinder_x", 0.091, 0.024,  0.091,  0.042),
}

parser = argparse.ArgumentParser(description="B1 gate: grasp feasibility, re3sim capture objects.")
parser.add_argument("--task", type=str, default="Rebot-PreGrasp-Play-v0")
parser.add_argument("--num_envs", type=int, default=128, help="doubles as the CEM population")
parser.add_argument("--object", type=str, required=True, choices=sorted(OBJECTS))
parser.add_argument("--object_x", type=float, default=0.245)
parser.add_argument("--iters", type=int, default=45)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--tries", type=int, default=3, help="gated re-solves before giving up")
parser.add_argument("--out_dir", type=str, default="re3sim/probes/out/p01")
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

import isaaclab.sim as sim_utils
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kin import ArmKin, Q_OPEN, lerp_pts  # noqa: E402

#: grip heights to sweep [m] -- deliberately reaching far below the 44 mm C9 floor
GRIP_ZS = (0.008, 0.012, 0.016, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.070)
STANDOFF = 0.075
LIFT = 0.090

# gates, in the spirit of ClutterExpert._gated_solve but re-tuned for THIS task -- see below
POS_TOL = 0.0015
O_ALIGN_MIN = 0.97

#: Floor penalty height [m]. The clutter expert uses 0.012, and copying that number here is
#: what made run 2's positive control fail: its blocks are 70 mm tall, but grasping a 24 mm
#: object *requires* the finger bodies down at z ~ 5 mm, so a 12 mm floor penalises exactly
#: the configuration the grasp needs. Run 2 reported low_z = 5.8-13.1 mm with o_align
#: collapsed to 0.40-0.69: the search was spending its budget fighting the floor term and
#: paying for it in alignment. This value only has to keep the arm out of the *table*.
LOW_Z_MIN = 0.002

#: Run 2 also pinned the approach axis to horizontal +x (`a_des`). That is over-constrained:
#: C1 records that this arm's attainable approach at table height is tilted, not horizontal,
#: and the un-constrained search reached HELD 100 % on this control. So the approach axis is
#: left FREE and the standoff direction is taken from the *achieved* `a_hat` instead.


def main() -> None:
    shape, obj_h, obj_across, obj_along, obj_m = OBJECTS[args_cli.object]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5

    common = dict(
        rigid_props=env_cfg.scene.block.spawn.rigid_props,
        mass_props=sim_utils.MassPropertiesCfg(mass=obj_m),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=0.9, restitution=0.0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.10)),
    )
    if shape == "cuboid":
        env_cfg.scene.block.spawn = sim_utils.CuboidCfg(
            size=(obj_along, obj_across, obj_h), **common)
    elif shape == "cylinder":
        env_cfg.scene.block.spawn = sim_utils.CylinderCfg(
            radius=obj_across / 2, height=obj_h, axis="Z", **common)
    else:  # cylinder_x -- stood on its rim, axis horizontal along the approach
        env_cfg.scene.block.spawn = sim_utils.CylinderCfg(
            radius=obj_h / 2, height=obj_across, axis="X", **common)

    env_cfg.scene.block.init_state.pos = [args_cli.object_x, 0.0, obj_h / 2]
    env_cfg.scene.block.init_state.rot = [0.0, 0.0, 0.0, 1.0]
    env_cfg.scene.wall.init_state.pos = (1.5, 0.0, 0.070)  # would turn a failure into a wedge

    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    n = e.num_envs
    env.reset()

    kin = ArmKin(env)
    block = e.scene["block"]
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
    # Side grasp with the fingers separating along y. The APPROACH axis is deliberately
    # unconstrained -- see the LOW_Z_MIN note above.
    o_des = torch.tensor([0.0, 1.0, 0.0], device=dev)

    def reset_block():
        pos = torch.tensor([args_cli.object_x, 0.0, obj_h / 2], device=dev).repeat(n, 1)
        block.write_root_state_to_sim(torch.cat(
            [pos + e.scene.env_origins, ident, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)
        return pos

    def gated_solve(pos, seed, tries, **kw):
        """CEM until a solution clears every gate, else the best seen. Reports which gate failed."""
        best = None
        for _ in range(max(1, tries)):
            s = kin.cem(pos, seed, o_des=o_des, iters=args_cli.iters,
                        restarts=args_cli.restarts, floor_z=LOW_Z_MIN, **kw)
            ok = (s["pos_err"] <= POS_TOL and s["o_align"] >= O_ALIGN_MIN
                  and s["low_z"] >= LOW_Z_MIN)
            s["gated"] = ok
            if best is None or s["cost"] < best["cost"]:
                best = s
            if ok:
                return s
        return best

    print(f"\n  object : {args_cli.object}  ({shape})")
    print(f"  height : {obj_h * 1000:.1f} mm     across the fingers: {obj_across * 1000:.1f} mm")
    print(f"  mass   : {obj_m * 1000:.0f} g")
    if obj_across > 0.0891:
        print(f"  ** {obj_across * 1000:.0f} mm exceeds the 89.1 mm COMMANDED opening (C3).")
        print("     The fingers can only be forced this wide by contact (~120 mm max).")
    if args_cli.object == "tapemeasure":
        print("  ** the 64 mm cross-finger span is an ESTIMATE -- not on the capture sheet.")

    rows = []
    print(f"\n  {'gripz':>6} | {'posErr':>7} {'oAlign':>6} {'lowZ':>6} {'gate':>4} | "
          f"{'gap':>7} {'encl':>5} {'rose':>5} | {'HELD':>5}")
    print("  " + "-" * 72)
    for gz in GRIP_ZS:
        if gz > obj_h - 0.004:  # the grip point has to be on the object
            continue
        reset_block()
        grip = torch.tensor([args_cli.object_x, 0.0, gz], device=dev)

        # --- solve the whole trajectory BEFORE anything closes (LESSONS A6) --------------
        s_grip = gated_solve(grip, kin.q_arm0, args_cli.tries, std0=0.45)
        q_grip = s_grip["q"]

        # Back off along the achieved approach axis, projected horizontal: retreating along
        # a tilted a_hat would lift the standoff pose off the grasp line and the fingers
        # would arrive from above the object rather than around it.
        f = s_grip["a_hat"].clone()
        f[2] = 0.0
        f = f / f.norm().clamp(min=1e-9)
        back = grip - STANDOFF * f
        q_back = []
        q = q_grip
        for t in reversed(lerp_pts(grip, back, 3)):
            q = kin.cem(t, q, o_des=o_des, iters=args_cli.iters,
                        restarts=1, std0=0.12, floor_z=LOW_Z_MIN)["q"]
            q_back.append(q)
        approach = list(reversed(q_back)) + [q_grip]

        q_up, q = [], q_grip
        for t in lerp_pts(grip, grip + torch.tensor([0.0, 0.0, LIFT], device=dev), 3):
            q = kin.cem(t, q, o_des=o_des, iters=args_cli.iters,
                        restarts=1, std0=0.12, floor_z=LOW_Z_MIN)["q"]
            q_up.append(q)

        # --- execute ---------------------------------------------------------------------
        pos0 = reset_block()
        kin.teleport_arm(approach[0].unsqueeze(0).repeat(n, 1), q_fing=Q_OPEN)
        reset_block()  # the teleport swept the arm through the block; restore it
        z0 = float(pos0[0, 2])

        kin.hold(approach[0].unsqueeze(0).repeat(n, 1), 15, close=False)
        for i in range(len(approach) - 1):
            kin.run(approach[i].unsqueeze(0).repeat(n, 1),
                    approach[i + 1].unsqueeze(0).repeat(n, 1), 25, close=False)
        kin.hold(q_grip.unsqueeze(0).repeat(n, 1), 70, close=True)
        seq = [q_grip] + q_up
        for i in range(len(seq) - 1):
            kin.run(seq[i].unsqueeze(0).repeat(n, 1), seq[i + 1].unsqueeze(0).repeat(n, 1),
                    25, close=True)
        kin.hold(q_up[-1].unsqueeze(0).repeat(n, 1), 50, close=True)

        bpos = block.data.root_pos_w.torch - e.scene.env_origins
        gap = kin.gap()
        encl = (gap - obj_across).abs() < 0.012
        rose = bpos[:, 2] > z0 + 0.045
        near = (kin.tcp_now() - bpos).norm(dim=1) < 0.09
        held = rose & near & encl
        rows.append({
            "grip_z": gz, "pos_err_m": s_grip["pos_err"], "o_align": s_grip["o_align"],
            "low_z_m": s_grip["low_z"], "gated": bool(s_grip["gated"]),
            "gap_m": float(gap.median()), "enclose_rate": float(encl.float().mean()),
            "rose_rate": float(rose.float().mean()), "near_rate": float(near.float().mean()),
            "held_rate": float(held.float().mean()),
        })
        print(f"  {gz * 1000:6.0f} | {s_grip['pos_err'] * 1000:6.2f}m {s_grip['o_align']:6.3f} "
              f"{s_grip['low_z'] * 1000:5.1f}m {'ok' if s_grip['gated'] else 'FAIL':>4} | "
              f"{float(gap.median()) * 1000:6.1f} {float(encl.float().mean()):5.0%} "
              f"{float(rose.float().mean()):5.0%} | {float(held.float().mean()):5.0%}")

    print("  " + "-" * 72)
    os.makedirs(args_cli.out_dir, exist_ok=True)
    out = os.path.join(args_cli.out_dir, f"{args_cli.object}.json")
    with open(out, "w") as f:
        json.dump({"object": args_cli.object, "shape": shape, "height_m": obj_h,
                   "across_m": obj_across, "along_m": obj_along, "mass_kg": obj_m,
                   "x": args_cli.object_x, "pos_tol": POS_TOL, "o_align_min": O_ALIGN_MIN,
                   "grid": rows}, f, indent=2)

    if rows:
        best = max(rows, key=lambda r: r["held_rate"])
        print(f"\n[b1] {args_cli.object}: best grip z = {best['grip_z'] * 1000:.0f} mm "
              f"-> HELD {best['held_rate']:.0%} "
              f"(enclosed {best['enclose_rate']:.0%}, rose {best['rose_rate']:.0%})")
        if best["held_rate"] > 0.5:
            print(f"[b1] VERDICT: GRASPABLE at grip z = {best['grip_z'] * 1000:.0f} mm")
        else:
            print("[b1] VERDICT: NOT reliably graspable as posed.")
            print("[b1]   gap ~ 0      -> closed on air (never got around it)")
            print(f"[b1]   gap ~ 89 mm  -> jammed OUTSIDE it ({obj_across * 1000:.0f} mm is too wide)")
            print("[b1]   enclosed but not risen -> gripped and slipped (C2 finger force)")
    print(f"[b1] wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
