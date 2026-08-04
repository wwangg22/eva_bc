# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P02 -- Which grasp orientations exist at the row, and which of them physically fit?

REWRITTEN 2026-08-02 after a first run produced kinematically valid but physically useless
poses. Two defects, both worth recording because both are easy to repeat:

1. **The CEM cost let orientation outrank position.** With `cost = |dp| + 0.25*(1-|o.o_des|)`
   and `|dp|` in metres, a 1 mm position error costs 0.001 while the orientation term costs up
   to 0.25 -- so the search happily walked 500 mm away to perfect an axis it could not
   otherwise achieve. The approach-tilt sweep returned 340-520 mm position errors. Position is
   now weighted `W_POS = 20.0`, making 1 mm worth 0.02 against orientation terms of order 0.1.
2. **Leaving the approach axis free is not a free lunch.** Unconstrained, the CEM returned
   poses with the *fingers pointing upward* (achieved tilt 118-166 deg off straight-down, i.e.
   `a_hat_z > 0`). The TCP was in the right place and the opening axis was right, but the jaw
   was above the block rather than around it. `reachability_map.py:62` is explicit -- "fingers
   extend along gripper_end local -X" -- so a pose is only a grasp if `a_hat` points roughly
   at the object.

What this probe now measures
---------------------------
Only two opening-axis families can grasp an upright cuboid with a parallel jaw:

* **G1 -- cross-row.** `o_hat = y_hat`: squeeze the block's 30 mm faces, fingers at +/-44.5 mm
  across the row. The standard grasp, and the one the task is built to block.
* **G2 -- front-back (strategy F).** `o_hat = x_hat`: squeeze the 36 mm faces, fingers fore and
  aft of the block. The row's y-pitch would stop being the binding constraint. Requires
  `a_hat` perpendicular to x_hat, i.e. approaching from along the row or from above -- and C1
  says straight-down is unavailable at table height. Whether some intermediate angle works has
  never been measured.

For each family we sweep the approach tilt and the grip height, keep what is kinematically
attainable, and then -- the part that cannot be faked -- **place the arm there and step
physics**. USD archaeology cannot answer whether the gripper fits: section 1 of P01 shows the
finger collider is an 8649-point `convexDecomposition` mesh whose axis-aligned bound implies a
51.7 mm clear gap where C3 *measured* 89.07 mm. So we stop reading geometry and start pushing
blocks: a pose that overlaps the row throws the blocks on the first physics step, and a pose
that fits leaves them where they are.

Section 3 turns that into a **caliper**: sweep the TCP across the row in 1 mm steps, test every
offset at once (one env per offset), and read off the interference profile directly. That is
the effective outer width of the open gripper, measured the only way that counts.

Usage
-----
    python eva_bc/clutter/probes/p02_orientation_envelope.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grasp-orientation envelope at the clutter row.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128, help="CEM population / caliper width")
parser.add_argument("--iters", type=int, default=50)
parser.add_argument("--settle", type=int, default=120, help="physics substeps per placement test")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p02_orientation.json")
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
from _kin import ArmKin, Q_OPEN  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
ROW_PITCH = 0.042
BLOCKS = ("target",) + mdp_cl.DISTRACTOR_NAMES
NOM_Y = {"target": 0.0, "distractor_0": -2 * ROW_PITCH, "distractor_1": -ROW_PITCH,
         "distractor_2": ROW_PITCH, "distractor_3": 2 * ROW_PITCH}

#: position weight -- see the module docstring. 1 mm of error costs 0.02.
W_POS = 20.0
GRIP_ZS = (0.045, 0.055, 0.065, 0.080)
#: approach tilt below horizontal [deg]; 0 = level, 90 = straight down (C1 says unavailable)
TILTS = (0.0, 20.0, 40.0, 60.0, 80.0, 90.0)
#: a pose counts as attainable only if all three hold
OK_POS, OK_O, OK_A = 0.003, 0.98, 0.95


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    K = ArmKin(env)
    dev, n = K.dev, K.n
    robot = K.robot
    ident = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).repeat(n, 1)
    org = e.scene.env_origins

    out: dict = {"task": args_cli.task, "num_envs": n, "w_pos": W_POS,
                 "block_half": list(mdp_cl.CL_BLOCK_HALF), "row_x": ROW_X}

    print("\n" + "=" * 100)
    print("P02 -- GRASP-ORIENTATION ENVELOPE AT THE CLUTTER ROW  (rewritten)")
    print("=" * 100)

    # gripper-only floor proxy: base_link sits at z = 0, so a whole-arm minimum is always 0
    grip_bodies = [K.i_end, K.i_left, K.i_right, robot.body_names.index("link6")]

    def grip_low_z(q_arm: torch.Tensor) -> torch.Tensor:
        K.fk(q_arm)
        bp = robot.data.body_pos_w.torch if hasattr(robot.data.body_pos_w, "torch") \
            else robot.data.body_pos_w
        return (bp[:, grip_bodies, 2] - org[:, 2:3]).min(dim=1).values

    def place_blocks(y_of: dict | None = None):
        """Put every block at its nominal pose, upright and at rest."""
        for b in BLOCKS:
            y = (y_of or NOM_Y)[b]
            p = torch.tensor([ROW_X, y, HZ], device=dev).repeat(n, 1) + org
            e.scene[b].write_root_state_to_sim(
                torch.cat([p, ident, torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def block_state() -> dict:
        st = {}
        for b in BLOCKS:
            st[b] = (e.scene[b].data.root_pos_w.torch - org).clone()
        return st

    def settle(q_arm: torch.Tensor, steps: int, q_fing: float = Q_OPEN):
        """Hold `q_arm` and step **physics only**.

        Deliberately not `env.step`: the env would fire `distractor_toppled` and reset the
        scene out from under the measurement. This isolates contact physics from the MDP.
        """
        q = K.q_default.unsqueeze(0).repeat(n, 1)
        q[:, K.arm_dof] = q_arm
        q[:, K.fing_dof] = q_fing
        for _ in range(steps):
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
            e.sim.step()
            e.scene.update(e.physics_dt)

    # ------------------------------------------------------- 1. kinematic orientation sweep
    print("\n" + "-" * 100)
    print("1. KINEMATIC SWEEP -- is the grasp orientation attainable at the target at all?")
    print("-" * 100)
    print("   G1  o_hat = y  (cross-row: squeeze the 30 mm faces; fingers at +/-44.5 mm in y)")
    print("   G2  o_hat = x  (front-back / strategy F: squeeze the 36 mm faces)")
    print("   a_hat tilt: 0 = level approach, 90 = straight down. C1: >=42.3 deg off vertical,")
    print("   i.e. tilt <= 47.7 deg, is the most this arm can do at table height.")
    print()
    hdr = (f"   {'fam':>3} {'z[mm]':>6} {'tilt':>5} | {'pos err':>8} | {'o_align':>7} | "
           f"{'a_align':>7} | {'grip low_z':>10} | {'ATTAINABLE':>10}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))

    X = torch.tensor([1.0, 0.0, 0.0], device=dev)
    Y = torch.tensor([0.0, 1.0, 0.0], device=dev)
    fams = {
        # o_des,  a_des(tilt) -> unit vector. G1 approaches from the robot (+x) and downward.
        "G1": (Y, lambda t: torch.tensor([math.cos(t), 0.0, -math.sin(t)], device=dev)),
        # G2 must approach perpendicular to x: from +y (over d2/d3) and downward.
        "G2": (X, lambda t: torch.tensor([0.0, -math.cos(t), -math.sin(t)], device=dev)),
    }

    kin_rows = []
    for fam, (o_des, a_of) in fams.items():
        for gz in GRIP_ZS:
            pos = torch.tensor([ROW_X, 0.0, gz], device=dev)
            seed = K.q_arm0.clone()
            for td in TILTS:
                a_des = a_of(math.radians(td))
                r = K.cem(pos, seed, o_des=o_des, a_des=a_des, w_o=0.35, w_a=0.35,
                          iters=args_cli.iters, std0=0.55, w_floor=0.0, w_pos=W_POS)
                lz = float(grip_low_z(r["q"].unsqueeze(0).repeat(n, 1))[0])
                ok = (r["pos_err"] < OK_POS and r["o_align"] > OK_O
                      and r["a_align"] > OK_A and lz > 0.0)
                if ok:
                    seed = r["q"].clone()
                kin_rows.append({"family": fam, "grip_z": gz, "tilt_deg": td,
                                 "pos_err_m": r["pos_err"], "o_align": r["o_align"],
                                 "a_align": r["a_align"], "grip_low_z_m": lz,
                                 "attainable": bool(ok),
                                 "q": [float(v) for v in r["q"]],
                                 "a_hat": [float(v) for v in r["a_hat"]],
                                 "o_hat": [float(v) for v in r["o_hat"]]})
                print(f"   {fam:>3} {gz * 1000:6.0f} {td:5.0f} | {r['pos_err'] * 1000:7.2f}mm | "
                      f"{r['o_align']:7.3f} | {r['a_align']:+7.3f} | {lz * 1000:9.1f}mm | "
                      f"{'YES' if ok else 'no':>10}")
            print("   " + "-" * (len(hdr) - 3))
    out["kinematic_sweep"] = kin_rows

    for fam in fams:
        good = [r for r in kin_rows if r["family"] == fam and r["attainable"]]
        print(f"   {fam}: {len(good)}/{len([r for r in kin_rows if r['family'] == fam])} "
              f"cells attainable")
        if good:
            b = min(good, key=lambda r: r["pos_err_m"])
            print(f"       best: z = {b['grip_z'] * 1000:.0f} mm, tilt = {b['tilt_deg']:.0f} deg, "
                  f"pos err = {b['pos_err_m'] * 1000:.2f} mm, a_hat = "
                  f"{[round(v, 3) for v in b['a_hat']]}")

    # ------------------------------------------------------ 2/3. PHYSICAL CALIPER ACROSS y
    print("\n" + "-" * 100)
    print("2. PHYSICAL CALIPER -- sweep the TCP across the row and see what actually moves")
    print("-" * 100)
    print("   One env per y-offset. The arm is teleported to the pose with the fingers OPEN,")
    print("   then physics runs for "
          f"{args_cli.settle} substeps ({args_cli.settle / 400:.2f} s) with the pose held.")
    print("   A pose that overlaps the row throws the blocks; a pose that fits leaves them.")
    print("   This is the effective outer width of the gripper, measured rather than derived.")
    print()

    caliper = {}
    for fam in fams:
        good = [r for r in kin_rows if r["family"] == fam and r["attainable"]]
        if not good:
            print(f"   {fam}: no attainable pose -- nothing to test physically.")
            caliper[fam] = None
            continue
        base = min(good, key=lambda r: (abs(r["grip_z"] - 0.055), r["pos_err_m"]))
        q0 = torch.tensor(base["q"], device=dev).unsqueeze(0).repeat(n, 1)
        # one y-offset per env, spanning the whole row
        ys = torch.linspace(-0.060, 0.060, n, device=dev)
        tgt = torch.stack([torch.full((n,), ROW_X, device=dev), ys,
                           torch.full((n,), base["grip_z"], device=dev)], dim=1)
        q = K.refine(q0, tgt, iters=4)
        ach = K.fk(q)["tcp"]
        err = (ach - tgt).norm(dim=1)
        print(f"   {fam}: base pose z = {base['grip_z'] * 1000:.0f} mm, "
              f"tilt = {base['tilt_deg']:.0f} deg; per-env refine err "
              f"median {float(err.median()) * 1000:.2f} mm, max {float(err.max()) * 1000:.2f} mm")

        place_blocks()
        K.teleport_arm(q, Q_OPEN)
        before = block_state()
        settle(q, args_cli.settle)
        after = block_state()
        moved = {b: (after[b][:, :2] - before[b][:, :2]).norm(dim=1) for b in BLOCKS}
        worst = torch.stack([moved[b] for b in BLOCKS], dim=1).max(dim=1).values
        up = torch.stack([mdp_cl._up_z(e, b) for b in mdp_cl.DISTRACTOR_NAMES], dim=1)
        topp = (up < mdp_cl.TOPPLE_DOT).any(dim=1)

        clean = (worst < 0.002) & ~topp & (err < 0.003)
        rec = {"base": base, "y_mm": (ys * 1000).tolist(),
               "refine_err_mm": (err * 1000).tolist(),
               "worst_move_mm": (worst * 1000).tolist(),
               "toppled": topp.tolist(), "clean": clean.tolist()}
        caliper[fam] = rec

        print(f"        {'y[mm]':>7} | {'refine':>7} | {'worst move':>10} | {'topple':>6} | ok")
        step = max(1, n // 32)
        for i in range(0, n, step):
            print(f"        {ys[i] * 1000:7.1f} | {err[i] * 1000:6.2f}mm | "
                  f"{worst[i] * 1000:9.2f}mm | {'YES' if topp[i] else '-':>6} | "
                  f"{'CLEAR' if clean[i] else '.'}")
        idx = [i for i in range(n) if clean[i]]
        if idx:
            runs, s = [], idx[0]
            for a, b in zip(idx, idx[1:] + [None]):
                if b is None or b != a + 1:
                    runs.append((float(ys[s]) * 1000, float(ys[a]) * 1000))
                    if b is not None:
                        s = b
            print(f"        -> CLEAR y-bands [mm]: "
                  + ", ".join(f"({a:.1f} .. {b:.1f}) w={b - a:.1f}" for a, b in runs))
            rec["clear_bands_mm"] = runs
        else:
            print("        -> NO y-offset leaves the row undisturbed with the fingers open.")
            rec["clear_bands_mm"] = []
    out["caliper"] = caliper

    # ------------------------------------------------------------------------- 4. verdict
    print("\n" + "=" * 100)
    print("4. VERDICT")
    print("=" * 100)
    for fam, lbl in (("G1", "cross-row (o_hat = y)"), ("G2", "front-back / strategy F (o_hat = x)")):
        rec = caliper.get(fam)
        print(f"\n   {fam}  {lbl}")
        if rec is None:
            print("        NOT KINEMATICALLY ATTAINABLE at the row. Ruled out on kinematics")
            print("        alone -- no physics needed.")
            continue
        bands = rec.get("clear_bands_mm", [])
        if not bands:
            print("        attainable, but NO placement leaves the row undisturbed. The open")
            print("        gripper cannot be put anywhere along the row without contact.")
        else:
            widest = max(bands, key=lambda ab: ab[1] - ab[0])
            print(f"        attainable, and {len(bands)} clear band(s); widest "
                  f"{widest[1] - widest[0]:.1f} mm at y in ({widest[0]:.1f}, {widest[1]:.1f}) mm.")
            print("        Whether the TARGET is between the fingers there is P03's question.")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p02] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
