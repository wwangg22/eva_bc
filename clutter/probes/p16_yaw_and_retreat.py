# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P16 -- The last two topple modes: my own retreat sweep, and the target's spawn yaw.

Where P15 left it
-----------------
Adding whole-arm clearance to the pose search worked on exactly what it was aimed at:
descent contact went from 34/128 envs to **zero**, enclosure held at 100 %, and at-goal rose
from 28 % to **100 %** at a 150 mm lift. The manipulation is now essentially perfect. But
topple stayed at 100 %, and the phase attribution moved to two new places:

    D  clear + lift150   topple onset:  close 67 | place 9 | dwell 5 | retreat 47
                         contact onset: close 73 | lift 2  | place 38 | dwell 15

Two mechanisms, both nothing to do with the grasp:

**1. The retreat is mine, and it is gratuitous.** After releasing at the goal the script ran
`place -> lift`, travelling from (185, -185) back to a pose above the row at (250, 0). The
task never asked for that. 47 of 128 topples start there -- an empty gripper sweeping back
across the blocks it just successfully avoided. The fix is to delete the motion: withdraw
*vertically at the goal* instead.

**2. Closing on a yawed block turns it.** The reset draws `yaw ~ U(-0.20, +0.20)` rad for the
target -- up to 11.5 deg. The orthogonal jaw closes on the block's 36 mm x-faces, and
`refine` corrects **position only**, so the jaw arrives square while the block is not. The
first face contact then torques the block back into alignment rather than gripping it. A
36 x 30 mm block rotating 11.5 deg sweeps its corners out to

    sqrt(18^2 + 15^2) * sin(atan(15/18) + 11.5 deg) - 15  =  ~8.4 mm in y

against a measured median free gap of **8.3 mm**. The block does not have room to turn, so
it turns its neighbour instead -- and the victim is `distractor_1` in 66 of 128 envs.

`ArmKin.refine` now takes an `o_des` channel that steers the opening axis per env, so the
jaw can arrive already matched to the block it is about to grip.

The 2x2
-------
All arms use the P15 clearance pose and a 150 mm lift; only the two fixes vary. One spawn,
snapshotted and restored, so every cell sees identical blocks.

    A  retreat  + square jaw     P15 arm D, reproduced as the reference
    B  withdraw + square jaw     retreat fix alone
    C  retreat  + yawed jaw      yaw fix alone
    D  withdraw + yawed jaw      both

The yaw fix is applied to the three approach waypoints, the grasp and **the lift** -- the
lift matters because snapping the wrist back to a square nominal pose while the block is
still inside the row twists it against exactly the neighbours the grasp just avoided.

Usage
-----
    python eva_bc/clutter/probes/p16_yaw_and_retreat.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Retreat and yaw fixes, paired 2x2.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=12)
parser.add_argument("--tries", type=int, default=8)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--lift_dz", type=float, default=0.150)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--margin", type=float, default=0.005)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p16_yaw.json")
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
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kin import ArmKin, Q_CLOSE, Q_OPEN, lerp_pts, _t  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
DIST = mdp_cl.DISTRACTOR_NAMES
STANDOFF = 0.070
ROW_Y = (-0.084, -0.042, 0.042, 0.084)
DISP_ON = 0.0015
TILT_ON = 0.999
PHASES = ["descend", "predwell", "close", "lift", "carry", "place",
          "dwell", "release", "retreat", "final"]


class Tape:
    """Per-control-step recorder, plus the target's yaw so the close mode is checkable."""

    def __init__(self, e, dpos0, yaw0):
        self.e, self.dpos0, self.yaw0 = e, dpos0, yaw0
        self.up, self.disp, self.dyaw, self.phase = [], [], [], []
        self.label = "descend"

    def sample(self):
        org = self.e.scene.env_origins
        up = torch.stack([mdp_cl._up_z(self.e, d) for d in DIST], dim=1)
        dp = torch.stack([(_t(self.e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1)
        self.up.append(up)
        self.disp.append((dp[:, :, :2] - self.dpos0[:, :, :2]).norm(dim=2))
        y = target_yaw(self.e)
        self.dyaw.append(torch.atan2(torch.sin(y - self.yaw0), torch.cos(y - self.yaw0)))
        self.phase.append(self.label)


def target_yaw(e) -> torch.Tensor:
    """Heading of the target's local +x axis (the 36 mm face normal) in the table plane."""
    q = _t(e.scene["target"].data.root_quat_w)
    ax = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3))
    return torch.atan2(ax[:, 1], ax[:, 0])


def target_axis(e) -> torch.Tensor:
    """Unit horizontal direction the jaw must open along to face the 36 mm faces squarely."""
    q = _t(e.scene["target"].data.root_quat_w)
    ax = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3))
    ax = ax.clone()
    ax[:, 2] = 0.0
    return ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9)


def hold_t(K, tp, q, steps, q_fing=Q_OPEN, every=8):
    qd = K._drive(q, q_fing)
    for s in range(steps):
        K.robot.set_joint_position_target(qd)
        K.robot.write_data_to_sim()
        K.e.sim.step()
        K.e.scene.update(K.e.physics_dt)
        if (s + 1) % every == 0:
            tp.sample()


def run_t(K, tp, q_from, q_to, steps, q_fing=Q_OPEN, substeps=8):
    for s in range(steps):
        f = (s + 1) / steps
        qd = K._drive((1 - f) * q_from + f * q_to, q_fing)
        for _ in range(substeps):
            K.robot.set_joint_position_target(qd)
            K.robot.write_data_to_sim()
            K.e.sim.step()
            K.e.scene.update(K.e.physics_dt)
        tp.sample()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    K = ArmKin(env)
    dev, n = K.dev, K.n
    org = e.scene.env_origins
    goal = torch.tensor(mdp_cl.GOAL_XY, device=dev)

    _p = math.radians(args_cli.phi)
    Y = torch.tensor([math.sin(_p), math.cos(_p), 0.0], device=dev)
    width = 2 * HX if args_cli.phi >= 45.0 else 2 * HY
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    dz = args_cli.lift_dz
    boxes = torch.tensor([[ROW_X, y, 0.032, HX, HY, HZ] for y in ROW_Y]
                         + [[ROW_X, 0.0, 0.032, HX, HY, HZ]], device=dev)

    print("\n" + "=" * 100)
    print("P16 -- RETREAT AND YAW FIXES (paired 2x2)")
    print("=" * 100)

    def pose_clear(pos, seed, tries, std0, restarts, pos_max=0.0015):
        best = None
        for _ in range(tries):
            c = K.cem(pos, seed, o_des=Y, w_o=0.60, iters=args_cli.iters, std0=std0,
                      restarts=restarts, avoid=boxes, avoid_margin=args_cli.margin)
            if c["pos_err"] > pos_max:
                continue
            key = (round(c["pen"], 4), -c["o_align"])
            if best is None or key < (round(best["pen"], 4), -best["o_align"]):
                best = c
        return best or c

    # ---------------- nominal chain, solved once, before anything closes
    pc = pose_clear(grip, K.q_arm0, args_cli.tries, 0.6, args_cli.restarts)
    q_grip = pc["q"]
    print(f"   grasp pose: err {pc['pos_err'] * 1000:.2f} mm | o_align {pc['o_align']:.3f} | "
          f"pen {pc['pen'] * 1000:.1f} mm | a_hat ({pc['a_hat'][0]:+.2f},"
          f"{pc['a_hat'][1]:+.2f},{pc['a_hat'][2]:+.2f})")

    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    qq, appr = q_grip, []
    for t in lerp_pts(grip, grip - STANDOFF * down, 3):
        qq = pose_clear(t, qq, 3, 0.15, args_cli.restarts)["q"]
        appr.append(qq)
    approach_nom = list(reversed(appr))

    lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
    carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + dz], device=dev)
    place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
    qq, post = q_grip, []
    for t in [lift, carry, place]:
        qq = pose_clear(t, qq, 3, 0.30, 3)["q"]
        post.append(qq)

    # ---------------- one spawn for every arm
    env.reset()
    for _ in range(30):
        e.sim.step()
        e.scene.update(e.physics_dt)
    snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}
    tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
    dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
    yaw0 = target_yaw(e).clone()
    o_env = target_axis(e).clone()
    ys = torch.cat([dpos0[:, :2, 1], tpos0[:, 1:2], dpos0[:, 2:, 1]], dim=1)
    ys, _ = ys.sort(dim=1)
    min_gap = (ys[:, 1:] - ys[:, :-1] - 2 * HY).min(dim=1).values
    yaw_deg = yaw0 * 180.0 / math.pi
    print(f"   spawn: min free gap {float(min_gap.min()) * 1000:.1f}.."
          f"{float(min_gap.max()) * 1000:.1f} mm (median {float(min_gap.median()) * 1000:.1f})"
          f" | target yaw {float(yaw_deg.min()):+.1f}..{float(yaw_deg.max()):+.1f} deg "
          f"(median |yaw| {float(yaw_deg.abs().median()):.1f})")

    def restore():
        for k, v in snap.items():
            e.scene[k].write_root_state_to_sim(v.clone())
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def execute(yaw_match: bool, withdraw: bool):
        restore()
        od = o_env if yaw_match else None
        tgt = torch.stack([tpos0[:, 0], tpos0[:, 1],
                           torch.full((n,), args_cli.grip_z, device=dev)], dim=1)
        q_g = K.refine(q_grip.unsqueeze(0).repeat(n, 1), tgt, iters=4, o_des=od)
        q_app = [K.refine(q.unsqueeze(0).repeat(n, 1),
                          tgt + torch.tensor([0.0, 0.0, d], device=dev), iters=3, o_des=od)
                 for q, d in zip(approach_nom, (STANDOFF, 2 * STANDOFF / 3, STANDOFF / 3))]
        # the lift must carry the same jaw yaw: snapping back to a square nominal pose while
        # the block is still between its neighbours twists it into them
        q_lift = K.refine(post[0].unsqueeze(0).repeat(n, 1),
                          tgt + torch.tensor([0.0, 0.0, dz], device=dev), iters=3, o_des=od)

        # what the jaw actually achieved, measured not assumed
        oa = (K.fk(q_g)["o_hat"] * o_env).sum(dim=1).abs()
        perr = (K.fk(q_g)["tcp"] - tgt).norm(dim=1)

        tp = Tape(e, dpos0, yaw0)
        K.teleport_arm(q_app[0], Q_OPEN)
        tp.label = "descend"
        hold_t(K, tp, q_app[0], 80)
        seq = q_app + [q_g]
        for i in range(len(seq) - 1):
            run_t(K, tp, seq[i], seq[i + 1], 25)
        tp.label = "predwell"
        hold_t(K, tp, q_g, 160)
        tp.label = "close"
        hold_t(K, tp, q_g, 560, q_fing=Q_CLOSE)
        gap_stall = K.gap().clone()
        chain = [q_g, q_lift] + [p.unsqueeze(0).repeat(n, 1) for p in post[1:]]
        for i, lab in enumerate(("lift", "carry", "place")):
            tp.label = lab
            run_t(K, tp, chain[i], chain[i + 1], 30, q_fing=Q_CLOSE)
        tp.label = "dwell"
        hold_t(K, tp, chain[-1], 160, q_fing=Q_CLOSE)
        tp.label = "release"
        hold_t(K, tp, chain[-1], 240, q_fing=Q_OPEN)
        tp.label = "retreat"
        # withdraw: straight up at the GOAL. retreat: back across the row, as before.
        run_t(K, tp, chain[-1], chain[2] if withdraw else chain[1], 25, q_fing=Q_OPEN)
        tp.label = "final"
        hold_t(K, tp, chain[2] if withdraw else chain[1], 240, q_fing=Q_OPEN)

        bpos = _t(e.scene["target"].data.root_pos_w) - org
        up_end = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
        topp = (up_end < mdp_cl.TOPPLE_DOT).any(dim=1)
        held = (gap_stall - width).abs() < 0.012
        at_goal = (((bpos[:, :2] - goal).norm(dim=1) < mdp_cl.GOAL_RADIUS)
                   & (bpos[:, 2] < 0.055))
        return held, at_goal, topp, at_goal & ~topp, tp, oa, perr

    def attribute(tp):
        UP = torch.stack(tp.up)
        DS = torch.stack(tp.disp)
        DY = torch.stack(tp.dyaw)
        ph, T = tp.phase, UP.shape[0]
        hit = (DS > DISP_ON) | (UP < TILT_ON)
        anyhit = hit.any(dim=2)
        ever = anyhit.any(dim=0)
        idx = torch.where(ever, anyhit.float().argmax(dim=0),
                          torch.full((n,), -1, device=dev, dtype=torch.long))
        ar = torch.arange(n, device=dev)
        victim = hit[idx.clamp(min=0), ar].float().argmax(dim=1)
        bad = (UP < mdp_cl.TOPPLE_DOT).any(dim=2)
        ever_t = bad.any(dim=0)
        idx_t = torch.where(ever_t, bad.float().argmax(dim=0),
                            torch.full((n,), -1, device=dev, dtype=torch.long))

        def by_phase(ix, ok):
            out = {}
            for p in PHASES:
                st = [t for t in range(T) if ph[t] == p]
                if st:
                    k = int((ok & (ix >= st[0]) & (ix <= st[-1])).sum())
                    if k:
                        out[p] = k
            return out

        cs = [t for t in range(T) if ph[t] == "close"]
        turn = DY[cs[-1]].abs() * 180 / math.pi if cs else torch.zeros(n, device=dev)
        vic = [int((ever & (victim == k)).sum()) for k in range(4)]
        return by_phase(idx_t, ever_t), by_phase(idx, ever), vic, turn

    ARMS = [("A retreat+square", False, False), ("B withdraw+square", False, True),
            ("C retreat+yawed", True, False), ("D withdraw+yawed", True, True)]
    results, masks = [], {}
    for tag, yawm, wd in ARMS:
        held, at_goal, topp, succ, tp, oa, perr = execute(yawm, wd)
        hp, hc, vic, turn = attribute(tp)
        print(f"\n   {tag}")
        print(f"      jaw-vs-block alignment {float(oa.median()):.4f} (median) | "
              f"TCP err {float(perr.median()) * 1000:.2f} mm | "
              f"target turned {float(turn.median()):.2f} deg during close")
        print(f"      enclosed {float(held.float().mean()):6.1%} | at goal "
              f"{float(at_goal.float().mean()):6.1%} | topple "
              f"{float(topp.float().mean()):6.1%} | SUCCESS "
              f"{float(succ.float().mean()):6.1%}")
        print(f"      topple onset : " + (", ".join(f"{k} {v}" for k, v in hp.items()) or "none"))
        print(f"      contact onset: " + (", ".join(f"{k} {v}" for k, v in hc.items()) or "none"))
        print(f"      first victim : "
              + (", ".join(f"d{k}(y={ROW_Y[k] * 1000:+.0f}) {vic[k]}"
                           for k in range(4) if vic[k]) or "none"))
        masks[tag] = succ
        results.append({"arm": tag, "yaw_match": yawm, "withdraw": wd,
                        "jaw_align": float(oa.median()),
                        "turn_deg": float(turn.median()),
                        "encl": float(held.float().mean()),
                        "at_goal": float(at_goal.float().mean()),
                        "topple": float(topp.float().mean()),
                        "success": float(succ.float().mean()),
                        "topple_by_phase": hp, "contact_by_phase": hc, "victim": vic,
                        "succ_mask": succ.tolist()})

    print("\n" + "=" * 100)
    print("PAIRED 2x2 (identical spawn in every cell)")
    print("=" * 100)
    print(f"   {'':>14} | {'retreat':>10} | {'withdraw':>10}   (success rate)")
    print(f"   {'square jaw':>14} | {results[0]['success']:9.1%} | {results[1]['success']:9.1%}")
    print(f"   {'yawed jaw':>14} | {results[2]['success']:9.1%} | {results[3]['success']:9.1%}")
    print(f"\n   main effect of WITHDRAW (B-A, D-C): "
          f"{results[1]['success'] - results[0]['success']:+.1%}, "
          f"{results[3]['success'] - results[2]['success']:+.1%}")
    print(f"   main effect of YAW      (C-A, D-B): "
          f"{results[2]['success'] - results[0]['success']:+.1%}, "
          f"{results[3]['success'] - results[1]['success']:+.1%}")
    base = masks["A retreat+square"]
    print(f"\n   {'arm':>18} | {'fixed':>6} | {'broke':>6} | net vs A")
    for tag in masks:
        m = masks[tag]
        f_, b_ = int((~base & m).sum()), int((base & ~m).sum())
        print(f"   {tag:>18} | {f_:6d} | {b_:6d} | {f_ - b_:+d}")

    # success against the two conditioning variables the gate will require anyway
    best = max(results, key=lambda r: r["success"])
    bm = torch.tensor(best["succ_mask"], device=dev)
    print(f"\n   champion arm '{best['arm']}' stratified:")
    print(f"      {'min free gap [mm]':>18} | {'n':>4} | {'success':>8}")
    for lo, hi in ((0, 6), (6, 8), (8, 10), (10, 14)):
        mm = (min_gap * 1000 >= lo) & (min_gap * 1000 < hi)
        if int(mm.sum()):
            print(f"      {f'{lo}-{hi}':>18} | {int(mm.sum()):4d} | "
                  f"{float(bm[mm].float().mean()):7.1%}")
    print(f"      {'|target yaw| [deg]':>18} | {'n':>4} | {'success':>8}")
    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 12)):
        mm = (yaw_deg.abs() >= lo) & (yaw_deg.abs() < hi)
        if int(mm.sum()):
            print(f"      {f'{lo}-{hi}':>18} | {int(mm.sum()):4d} | "
                  f"{float(bm[mm].float().mean()):7.1%}")

    out = {"n": n, "grip_z": args_cli.grip_z, "lift_dz": dz, "phi": args_cli.phi,
           "pose": {"err_mm": pc["pos_err"] * 1000, "o_align": pc["o_align"],
                    "pen_mm": pc["pen"] * 1000, "q": pc["q"].tolist()},
           "min_gap_mm": (min_gap * 1000).tolist(),
           "yaw_deg": yaw_deg.tolist(), "arms": results}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p16] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
