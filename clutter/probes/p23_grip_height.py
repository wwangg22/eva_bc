# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P23 -- Grip height, measured on the isolated mechanism instead of the whole trajectory.

What P22 established
--------------------
Remove the target block entirely, put the arm in the identical grasp pose, and close the
gripper on empty air:

    target present   74.2 % of neighbours moved, 19.5 % toppled
    TARGET REMOVED   75.8 % moved,               21.9 % toppled
    jaw kept OPEN     0.0 % moved,                0.0 % toppled

The target is a bystander. **The finger blades sweep the neighbours directly**, and they do
it during the close, not the descent. The collision meshes -- read at last, through instance
proxies, 17 298 points per finger -- say why:

    gripper_left   x -19.2..19.2 | y -41.9..46.7 | z -58.7..34.7   [mm, body frame]

The blades are ~88 mm across the opening axis. With the jaw open the finger origins sit at
x ~ 205 and 295 mm, outside the row's 232-268 mm x-band, and nothing touches -- which is
exactly why the descent hazard has been 0 % since P15. Closing drags them 26.5 mm each into
that band, where the neighbours live at y = +/-42 mm.

Why sweep on the isolated mechanism
-----------------------------------
Grip height has been the largest single lever measured (P20: 65 mm -> 43.0 % topple,
55 mm -> 13.3 %, 50 mm -> 53.1 %) and it was measured **once**, on one spawn, inside a full
trajectory whose success also depends on the pose draw. P21 then pooled 768 episodes at
55 mm and got 35.3 % topple, not 13.3 %. The single-cell number was noise around a real
effect, and the effect is worth locating properly.

P22's control gives a way to do that cheaply and without confounds: settle at the grasp
pose, remove the target, close, measure. No descent, no carry, no place, no pose-dependent
success -- just the mechanism, at 128 envs per height, in a few seconds each. The trajectory
is then run end-to-end only at the winner, to confirm the isolated measurement transfers.

The sweep is deliberately wider than the plausible range (40-80 mm against a block whose top
is at ~67 mm and whose centre of mass is at 32 mm), because the empirical curve was
non-monotonic and a three-point sample cannot distinguish a real optimum from two bad draws.

Usage
-----
    python eva_bc/clutter/probes/p23_grip_height.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grip-height sweep on the isolated close.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--heights", type=str, default="0.040,0.045,0.050,0.055,0.060,0.065,"
                                                   "0.070,0.075,0.080")
parser.add_argument("--pose_reps", type=int, default=2,
                    help="independent pose draws per height -- the confound P21 exposed")
parser.add_argument("--confirm", type=int, default=1,
                    help="run the full trajectory at the winning height")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p23_grip.json")
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from _kin import Q_CLOSE, Q_OPEN, _t  # noqa: E402
from clutter_expert import ClutterExpert, DIST, HY  # noqa: E402

ROW_Y = (-0.084, -0.042, 0.042, 0.084)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    dev, n = e.device, e.num_envs
    org = e.scene.env_origins
    heights = [float(x) for x in args_cli.heights.split(",")]

    print("\n" + "=" * 100)
    print("P23 -- GRIP HEIGHT ON THE ISOLATED CLOSE")
    print("=" * 100)
    print("   block: 70 mm tall, settles with centre at 32 mm and top at ~67 mm")
    print("   isolated = settle at the grasp pose, close, measure. No descent, no carry,")
    print("   no place -- so nothing here depends on the path or on the pose draw's luck")
    print("   downstream. The target is KEPT (P22 showed it makes no difference to what the")
    print("   blades hit, and keeping it preserves the real 36 mm stall instead of letting")
    print("   the jaw slam shut on nothing).")

    # ---- one spawn, restored before every cell
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
    print(f"   spawn: min free gap median {float(min_gap.median()) * 1000:.1f} mm\n")

    def restore(drop_target=True):
        for k, v in snap.items():
            s = v.clone()
            if drop_target and k == "target":
                s[:, 2] -= 2.0
                s[:, 7:] = 0.0
            e.scene[k].write_root_state_to_sim(s)
        e.sim.forward()
        e.scene.update(e.physics_dt)

    print(f"   {'grip z':>7} | {'draw':>4} | {'o_align':>8} | {'wrist y,z':>12} | "
          f"{'moved':>7} | {'toppled':>8} | per-block topple")
    print("   " + "-" * 92)
    rows = []
    experts = {}
    for gz in heights:
        for pr in range(args_cli.pose_reps):
            # ORDER MATTERS. `adapt()` reads the target's live position, so the spawn has to
            # be restored BEFORE it is called. The first version of this probe adapted after
            # the previous cell had already dropped the target 2 m below the table, so every
            # cell but the first aimed a chain at nothing -- which showed up as `d3`
            # (y = +84 mm, the far end of the row) becoming the dominant victim, geometry
            # that no correct grasp can produce. The nonsense in the output is what caught it.
            restore(False)
            ex = ClutterExpert(env, grip_z=gz, verbose=False, plan_full=False)
            K = ex.K
            q = ex.adapt()[0]
            restore(False)
            K.teleport_arm(q, Q_OPEN)
            for _ in range(160):
                K.robot.set_joint_position_target(K._drive(q, Q_OPEN))
                K.robot.write_data_to_sim()
                e.sim.step()
                e.scene.update(e.physics_dt)
            pre = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST],
                              dim=1).clone()
            for _ in range(560):
                K.robot.set_joint_position_target(K._drive(q, Q_CLOSE))
                K.robot.write_data_to_sim()
                e.sim.step()
                e.scene.update(e.physics_dt)
            post = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1)
            up = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
            disp = (post[:, :, :2] - pre[:, :, :2]).norm(dim=2)
            moved = float((disp > 0.0015).any(dim=1).float().mean())
            topp_b = [float((up[:, k] < mdp_cl.TOPPLE_DOT).float().mean()) for k in range(4)]
            topp = float((up < mdp_cl.TOPPLE_DOT).any(dim=1).float().mean())
            w = K.fk(ex.pose["q"].unsqueeze(0).repeat(n, 1))["bodies"][0, K.i_end]
            print(f"   {gz * 1000:6.0f}  | {pr:4d} | {ex.pose['o_align']:8.4f} | "
                  f"{float(w[1]) * 1000:+5.0f},{float(w[2]) * 1000:5.0f} | {moved:6.1%} | "
                  f"{topp:7.1%} | "
                  + " ".join(f"d{k}:{topp_b[k]:.0%}" for k in range(4)))
            rows.append({"grip_z": gz, "pose_rep": pr, "o_align": ex.pose["o_align"],
                         "wrist_y_mm": float(w[1]) * 1000, "wrist_z_mm": float(w[2]) * 1000,
                         "moved": moved, "topple": topp, "per_block": topp_b})

    # ---- aggregate over pose draws: the height effect, separated from the draw effect
    print("\n" + "=" * 100)
    print("   ISOLATED-CLOSE TOPPLE RATE BY GRIP HEIGHT (mean over pose draws)")
    print(f"   {'grip z [mm]':>12} | {'moved':>8} | {'toppled':>9} | spread over draws")
    agg = {}
    for gz in heights:
        r = [x for x in rows if x["grip_z"] == gz]
        m = sum(x["moved"] for x in r) / len(r)
        t = sum(x["topple"] for x in r) / len(r)
        agg[gz] = t
        print(f"   {gz * 1000:12.0f} | {m:7.1%} | {t:8.1%} | "
              + ", ".join(f"{x['topple']:.0%}" for x in r))
    best = min(agg, key=agg.get)
    print(f"\n   minimum at grip_z = {best * 1000:.0f} mm ({agg[best]:.1%} isolated topple)")
    cur = agg.get(0.055)
    if cur is not None and best != 0.055:
        print(f"   the expert currently ships 55 mm ({cur:.1%}); this run prefers "
              f"{best * 1000:.0f} mm")

    # ---- does the isolated result transfer to the full trajectory?
    conf = []
    if args_cli.confirm:
        print("\n" + "=" * 100)
        print("   FULL-TRAJECTORY CONFIRMATION (whole chain, does the isolated result "
              "transfer?)")
        for gz in sorted({best, 0.055}):
            env.reset()
            for _ in range(30):
                e.sim.step()
                e.scene.update(e.physics_dt)
            ex = ClutterExpert(env, grip_z=gz, verbose=False)
            for rep in range(2):
                env.reset()
                for _ in range(30):
                    e.sim.step()
                    e.scene.update(e.physics_dt)
                r = ex.run_physics(ex.adapt())
                print(f"      grip {gz * 1000:.0f} mm, batch {rep}: enclosed "
                      f"{float(r['held'].float().mean()):6.1%} | at goal "
                      f"{float(r['at_goal'].float().mean()):6.1%} | topple "
                      f"{float(r['topple'].float().mean()):6.1%} | SUCCESS "
                      f"{float(r['success'].float().mean()):6.1%}")
                conf.append({"grip_z": gz, "rep": rep,
                             "encl": float(r["held"].float().mean()),
                             "at_goal": float(r["at_goal"].float().mean()),
                             "topple": float(r["topple"].float().mean()),
                             "success": float(r["success"].float().mean())})

    out = {"n": n, "heights": heights, "pose_reps": args_cli.pose_reps,
           "min_gap_mm": (min_gap * 1000).tolist(), "rows": rows,
           "agg_topple": {str(k): v for k, v in agg.items()}, "best": best,
           "confirm": conf}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p23] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
