# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P15 -- Does whole-arm clearance-aware pose selection fix the descent contact?

The problem, stated precisely
-----------------------------
Pose selection from P11 through P13 used two gates: `pos_err < 1.5 mm` and maximum
`o_align`. Three runs of that selector, all passing both gates comfortably, produced:

    P10   err 0.6-1.0 mm, o_align ~0.99   -> enclosed 100 %, at goal  99 %, success 25 %
    P12a  err 0.83 mm,    o_align 0.998   -> enclosed  94 %, at goal   9 %, success  0 %
    P12b  err 0.98 mm,    o_align 0.996   -> enclosed 100 %, at goal 100 %, success 37.5 %
    P13   err 0.55 mm,    o_align 1.000   -> enclosed  32 %, at goal  12 %, success  0 %

The best and the worst runs are separated by 0.45 mm of position error and 0.002 of
`o_align`. **The selector cannot see the variable that decides the outcome.** P13's pose,
the one with the *perfect* scores, had `a_hat = (+0.03, -0.94, +0.33)` -- and because
`o_hat = x_hat` confines the approach axis to the y-z plane and the wrist stub trails the
TCP by 41.9 mm along it, that put `gripper_end` at y = +39 mm, z = 51 mm, inside
`distractor_2`. 63 of 128 envs made contact during the descent; d2 was the first victim in
50 of them.

P14 then swept the approach axis over the full circle and established two things:

  1. **A downward approach axis is genuinely unattainable here.** At `a_des = (0,0,-1)` the
     CEM achieves `a_align = -0.11`; nothing past t = 105 deg is reached at all. C1 is
     correct at the pose that matters, not just on its voxel grid. So the wrist cannot be
     parked above the row, and it must thread one of the 12 mm gaps.
  2. The poses P14 scored as row-clear put the wrist at |y| < 20 mm, z ~ 26 mm -- inside the
     **target's** own volume, which P14's keep-out set did not contain.

So the question this probe answers: with the target included and the penetration term wired
into the CEM cost, **can the search find a pose that is simultaneously accurate, aligned and
clear -- and does it hold up in execution?**

Four arms, all paired
---------------------
One reset, one settled spawn, snapshotted and restored between arms, so nothing here is
confounded by spawn draw:

    A  align_only  + lift 75    the P10/P12/P13 baseline, reproduced for reference
    B  clearance   + lift 75    isolates the pose fix
    C  align_only  + lift 150   isolates the lift fix
    D  clearance   + lift 150   both

A vs B and C vs D measure the pose fix; A vs C and B vs D measure the lift fix; the 2x2
shows whether they interact. Each arm reports its own phase-resolved contact and topple
attribution, so a change in the headline can be traced to the phase it came from.

Usage
-----
    python eva_bc/clutter/probes/p15_clearance_pose.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Clearance-aware pose selection, paired 2x2.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=12)
parser.add_argument("--tries", type=int, default=8)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--margin", type=float, default=0.005,
                    help="keep-out inflation [m]; body origins understate real extent")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p15_clearance.json")
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
    def __init__(self, e, dpos0):
        self.e, self.dpos0 = e, dpos0
        self.up, self.disp, self.phase = [], [], []
        self.label = "descend"

    def sample(self):
        org = self.e.scene.env_origins
        up = torch.stack([mdp_cl._up_z(self.e, d) for d in DIST], dim=1)
        dp = torch.stack([(_t(self.e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1)
        self.up.append(up)
        self.disp.append((dp[:, :, :2] - self.dpos0[:, :, :2]).norm(dim=2))
        self.phase.append(self.label)


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
    names = list(K.robot.body_names)

    _p = math.radians(args_cli.phi)
    Y = torch.tensor([math.sin(_p), math.cos(_p), 0.0], device=dev)
    width = 2 * HX if args_cli.phi >= 45.0 else 2 * HY
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)

    # keep-out volumes: the four neighbours AND the target. The target belongs here even
    # though it is the thing being grasped -- the fingers straddle it in x (205 / 295 mm)
    # and are outside its 232-268 mm span, so nothing legitimate is inside it at the moment
    # of the grasp. P14's "row-clear" poses were all inside the target and it never noticed.
    boxes = torch.tensor([[ROW_X, y, 0.032, HX, HY, HZ] for y in ROW_Y]
                         + [[ROW_X, 0.0, 0.032, HX, HY, HZ]], device=dev)

    print("\n" + "=" * 100)
    print("P15 -- CLEARANCE-AWARE POSE SELECTION (paired 2x2 against lift height)")
    print("=" * 100)

    def pose_align(pos, seed, tries, std0, restarts, pos_max=0.0015):
        """Baseline selector: max `o_align` subject to a position gate. Blind to the arm."""
        best = None
        for _ in range(tries):
            c = K.cem(pos, seed, o_des=Y, w_o=0.60, iters=args_cli.iters,
                      std0=std0, restarts=restarts)
            if c["pos_err"] > pos_max:
                continue
            if best is None or c["o_align"] > best["o_align"]:
                best = c
        return best or c

    def pose_clear(pos, seed, tries, std0, restarts, pos_max=0.0015):
        """Clearance selector: penetration first, then alignment, subject to the same gate.

        Lexicographic, not weighted. A pose that puts a link inside a neighbour is not
        "slightly worse" than one that does not -- it is a different kind of object, and a
        weighted sum lets a 0.003 gain in `o_align` buy a 10 mm intrusion, which is exactly
        the trade P13 made.
        """
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

    def describe(tag, c):
        g = K.fk(c["q"].unsqueeze(0).repeat(n, 1))
        wr = g["bodies"][0, K.i_end]
        pen_b = torch.stack([K.box_penetration(g["bodies"][0, b].view(1, 1, 3), boxes, 0.0)[0]
                             for b in range(len(names))])
        worst = names[int(pen_b.argmax())] if float(pen_b.max()) > 0 else "-"
        print(f"   {tag:>12}: err {c['pos_err'] * 1000:5.2f} mm | o_align {c['o_align']:.3f} | "
              f"a_hat ({c['a_hat'][0]:+.2f},{c['a_hat'][1]:+.2f},{c['a_hat'][2]:+.2f}) | "
              f"wrist ({float(wr[0]) * 1000:.0f},{float(wr[1]) * 1000:+.0f},"
              f"{float(wr[2]) * 1000:.0f}) mm | pen {float(pen_b.max()) * 1000:.1f} mm "
              f"({worst})")
        return float(pen_b.max())

    # ---------------- the two candidate grasp poses
    pa = pose_align(grip, K.q_arm0, args_cli.tries, 0.6, args_cli.restarts)
    pc = pose_clear(grip, K.q_arm0, args_cli.tries, 0.6, args_cli.restarts)
    print("\n   GRASP POSE CANDIDATES")
    pen_a = describe("align_only", pa)
    pen_c = describe("clearance", pc)

    # ---------------- one reset, one snapshot, restored between arms
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
    print(f"\n   spawn: min free gap {float(min_gap.min()) * 1000:.1f}.."
          f"{float(min_gap.max()) * 1000:.1f} mm (median {float(min_gap.median()) * 1000:.1f})")

    def restore():
        for k, v in snap.items():
            e.scene[k].write_root_state_to_sim(v.clone())
        e.sim.forward()
        e.scene.update(e.physics_dt)

    # ---------------- per-policy chains, all solved before anything closes
    def build(sel, q_grip, dz):
        down = torch.tensor([0.0, 0.0, -1.0], device=dev)
        qq, appr = q_grip, []
        for t in lerp_pts(grip, grip - STANDOFF * down, 3):
            qq = sel(t, qq, 3, 0.15, args_cli.restarts)["q"]
            appr.append(qq)
        approach = list(reversed(appr))
        lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
        carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + dz], device=dev)
        place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
        qq, post = q_grip, []
        for t in [lift, carry, place]:
            qq = sel(t, qq, 3, 0.30, 3)["q"]
            post.append(qq)
        return approach, post

    def execute(approach, q_grip, post, dz):
        restore()
        tgt = torch.stack([tpos0[:, 0], tpos0[:, 1],
                           torch.full((n,), args_cli.grip_z, device=dev)], dim=1)
        q_g = K.refine(q_grip.unsqueeze(0).repeat(n, 1), tgt, iters=4)
        q_app = [K.refine(q.unsqueeze(0).repeat(n, 1),
                          tgt + torch.tensor([0.0, 0.0, d], device=dev), iters=3)
                 for q, d in zip(approach, (STANDOFF, 2 * STANDOFF / 3, STANDOFF / 3))]
        tp = Tape(e, dpos0)
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
        chain = [q_g] + [p.unsqueeze(0).repeat(n, 1) for p in post]
        for i, lab in enumerate(("lift", "carry", "place")):
            tp.label = lab
            run_t(K, tp, chain[i], chain[i + 1], 30, q_fing=Q_CLOSE)
        tp.label = "dwell"
        hold_t(K, tp, chain[-1], 160, q_fing=Q_CLOSE)
        tp.label = "release"
        hold_t(K, tp, chain[-1], 240, q_fing=Q_OPEN)
        tp.label = "retreat"
        run_t(K, tp, chain[-1], chain[1], 25, q_fing=Q_OPEN)
        tp.label = "final"
        hold_t(K, tp, chain[1], 240, q_fing=Q_OPEN)

        bpos = _t(e.scene["target"].data.root_pos_w) - org
        up_end = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
        topp = (up_end < mdp_cl.TOPPLE_DOT).any(dim=1)
        held = (gap_stall - width).abs() < 0.012
        at_goal = (((bpos[:, :2] - goal).norm(dim=1) < mdp_cl.GOAL_RADIUS)
                   & (bpos[:, 2] < 0.055))
        return held, at_goal, topp, at_goal & ~topp, tp

    def attribute(tp):
        UP = torch.stack(tp.up)
        DS = torch.stack(tp.disp)
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

        vic = [int((ever & (victim == k)).sum()) for k in range(4)]
        return by_phase(idx_t, ever_t), by_phase(idx, ever), vic

    ARMS = [("A align+lift75", pose_align, pa["q"], 0.075),
            ("B clear+lift75", pose_clear, pc["q"], 0.075),
            ("C align+lift150", pose_align, pa["q"], 0.150),
            ("D clear+lift150", pose_clear, pc["q"], 0.150)]

    results, masks = [], {}
    for tag, sel, qg, dz in ARMS:
        approach, post = build(sel, qg, dz)
        held, at_goal, topp, succ, tp = execute(approach, qg, post, dz)
        hp, hc, vic = attribute(tp)
        print(f"\n   {tag}")
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
        results.append({"arm": tag, "lift_dz": dz,
                        "encl": float(held.float().mean()),
                        "at_goal": float(at_goal.float().mean()),
                        "topple": float(topp.float().mean()),
                        "success": float(succ.float().mean()),
                        "topple_by_phase": hp, "contact_by_phase": hc, "victim": vic,
                        "succ_mask": succ.tolist()})

    print("\n" + "=" * 100)
    print("PAIRED 2x2 (identical spawn in every cell)")
    print("=" * 100)
    print(f"   {'':>16} | {'lift 75':>10} | {'lift 150':>10}   (success rate)")
    print(f"   {'align_only':>16} | {results[0]['success']:9.1%} | {results[2]['success']:9.1%}")
    print(f"   {'clearance':>16} | {results[1]['success']:9.1%} | {results[3]['success']:9.1%}")
    print(f"\n   main effect of POSE fix  (B-A, D-C): "
          f"{results[1]['success'] - results[0]['success']:+.1%}, "
          f"{results[3]['success'] - results[2]['success']:+.1%}")
    print(f"   main effect of LIFT fix  (C-A, D-B): "
          f"{results[2]['success'] - results[0]['success']:+.1%}, "
          f"{results[3]['success'] - results[1]['success']:+.1%}")
    base = masks["A align+lift75"]
    print(f"\n   {'arm':>16} | {'fixed':>6} | {'broke':>6} | net vs A")
    for tag in masks:
        m = masks[tag]
        f_, b_ = int((~base & m).sum()), int((base & ~m).sum())
        print(f"   {tag:>16} | {f_:6d} | {b_:6d} | {f_ - b_:+d}")

    out = {"n": n, "grip_z": args_cli.grip_z, "phi": args_cli.phi,
           "pen_align_mm": pen_a * 1000, "pen_clear_mm": pen_c * 1000,
           "pose_align": {"err_mm": pa["pos_err"] * 1000, "o_align": pa["o_align"],
                          "a_hat": pa["a_hat"].tolist(), "q": pa["q"].tolist()},
           "pose_clear": {"err_mm": pc["pos_err"] * 1000, "o_align": pc["o_align"],
                          "a_hat": pc["a_hat"].tolist(), "q": pc["q"].tolist()},
           "min_gap_mm": (min_gap * 1000).tolist(), "arms": results}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p15] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
