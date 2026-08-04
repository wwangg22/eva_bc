# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P17 -- Nothing has ever checked the path BETWEEN the waypoints.

The number that forced this probe
---------------------------------
P16's contact attribution counts each env's *first* contact only, so raw counts understate
late phases. Converted to a hazard rate -- contacts in a phase, divided by the envs that
reached that phase still clean -- the trajectory reads:

    arm A (square jaw)   close  65/128 = 51 %     place  63/63  = **100 %**
    arm C (yawed jaw)    close  93/128 = 73 %     place  20/20  = **100 %**

Every env that survives to the `place` segment makes contact during it. Not most: all. A
100 % hazard is not a clearance problem, a precision problem or a spawn-dependent problem.
It is a deterministic geometric fact about that segment, and it has been there since P10.

`place` runs from `carry` = (185, -185, 215) mm to `place` = (185, -185, 65) mm -- the same
xy, 150 mm straight down, 160 mm away from the nearest distractor. In Cartesian terms it
cannot touch anything. So the arm is not travelling in Cartesian terms.

The cause
---------
Every waypoint in this chain is solved independently by a stochastic CEM with restarts from
uniform draws over the joint limits, and then **executed by interpolating in joint space**:

    q(f) = (1 - f) q_from + f q_to

That is only a straight Cartesian line when both endpoints lie in the same IK branch and the
map is near-linear between them. A 6-DOF arm has multiple branches -- elbow up/down, wrist
flipped -- and a CEM restarting from uniform draws will happily return a `place` pose in a
different branch from its `carry` pose. Both are individually verified, both are row-clear,
and the straight line between them in *joint* space swings the whole arm through an arc that
was never checked. Seeding each solve from the previous one reduces this but does not
prevent it: `std0 = 0.30` with `restarts = 3` is more than enough to jump.

Every waypoint in every probe from P05 onward has been verified. No segment ever has.

What this probe does
--------------------
**Part A -- audit.** For the existing chain, sample each joint-space segment at 40 points,
run FK on each, and report:
    * the maximum deviation of the achieved TCP from the straight Cartesian segment
    * the deepest penetration of any body origin into any block volume, along the segment
A well-behaved segment deviates by a few millimetres. A branch flip shows up as hundreds.

**Part B -- fix.** Rebuild the same chain as a dense Cartesian path: waypoints every
`--seg` metres, each solved by a CEM seeded from the previous solution with a small `std0`
and **no restarts**, so the solution cannot leave the branch it starts in. Audit again.

**Part C -- paired execution.** Same spawn, both chains, full phase forensics with hazard
rates rather than raw counts.

Usage
-----
    python eva_bc/clutter/probes/p17_path_verify.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Segment audit and Cartesian densification.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=12)
parser.add_argument("--tries", type=int, default=8)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--lift_dz", type=float, default=0.150)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--margin", type=float, default=0.005)
parser.add_argument("--seg", type=float, default=0.030,
                    help="max Cartesian spacing between solved waypoints [m]")
parser.add_argument("--yaw", type=int, default=1, help="1 = match the target's spawn yaw")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p17_path.json")
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
from _kin import ArmKin, Q_CLOSE, Q_OPEN, _t  # noqa: E402

ROW_X = 0.250
HX, HY, HZ = mdp_cl.CL_BLOCK_HALF
DIST = mdp_cl.DISTRACTOR_NAMES
STANDOFF = 0.070
ROW_Y = (-0.084, -0.042, 0.042, 0.084)
DISP_ON = 0.0015
TILT_ON = 0.999


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


def target_axis(e) -> torch.Tensor:
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
    print("P17 -- SEGMENT AUDIT AND CARTESIAN DENSIFICATION")
    print("=" * 100)

    def solve(pos, seed, tries, std0, restarts, pos_max=0.0015):
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

    # ------------------------------------------------------------------ Part A: the audit
    def audit(pts, qs, label):
        """Sample every joint-space segment and report what the arm actually does."""
        print(f"\n   {label}")
        print(f"      {'segment':>26} | {'cart len':>9} | {'max TCP dev':>12} | "
              f"{'max pen':>8} | {'min body z':>10}")
        worst = 0.0
        rows = []
        for i in range(len(qs) - 1):
            a, b = pts[i], pts[i + 1]
            L = float((b - a).norm())
            dev_max, pen_max, zmin = 0.0, 0.0, 9.0
            for s in range(41):
                f = s / 40.0
                q = ((1 - f) * qs[i] + f * qs[i + 1]).unsqueeze(0).repeat(n, 1)
                g = K.fk(q)
                dev = float((g["tcp"][0] - ((1 - f) * a + f * b)).norm())
                pen = float(K.box_penetration(g["bodies"][:1], boxes, 0.0)[0])
                zmin = min(zmin, float(g["bodies"][0, K.floor_bodies, 2].min()))
                dev_max, pen_max = max(dev_max, dev), max(pen_max, pen)
            flag = "  <-- BRANCH FLIP" if dev_max > 0.030 else ""
            print(f"      {label_of(i, len(qs)):>26} | {L * 1000:8.0f} | "
                  f"{dev_max * 1000:11.1f} | {pen_max * 1000:7.1f} | {zmin * 1000:9.1f}{flag}")
            worst = max(worst, dev_max)
            rows.append({"seg": i, "len_mm": L * 1000, "dev_mm": dev_max * 1000,
                         "pen_mm": pen_max * 1000})
        return worst, rows

    names_cache = {}

    def label_of(i, tot):
        return names_cache.get(i, f"wp{i}->wp{i + 1}")

    # ---- the P16 chain: five Cartesian waypoints, each solved independently
    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
    carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + dz], device=dev)
    place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
    approach_pts = [grip + STANDOFF * down * (-(3 - k) / 3.0) for k in range(3)]
    coarse_pts = approach_pts + [grip, lift, carry, place]
    labels = ["app0->app1", "app1->app2", "app2->grip", "grip->lift",
              "lift->carry", "carry->place"]
    names_cache.update({i: labels[i] for i in range(len(labels))})

    qq, coarse_qs = K.q_arm0, []
    for k, t in enumerate(coarse_pts):
        c = solve(t, qq, args_cli.tries if k == 3 else 3,
                  0.6 if k == 3 else (0.15 if k < 3 else 0.30),
                  args_cli.restarts if k <= 3 else 3)
        qq = c["q"]
        coarse_qs.append(qq)
    q_grip_coarse = coarse_qs[3]
    w_coarse, rows_coarse = audit(coarse_pts, coarse_qs, "COARSE chain (P10-P16, as shipped)")

    # ------------------------------------------------- Part B: dense Cartesian, one branch
    def dense_chain(pts, seed):
        """Solve a Cartesian polyline with <= `seg` spacing, never leaving the branch.

        `restarts = 1` and a small `std0` are the whole point: the CEM must stay a *local*
        refinement of the previous solution. Restarting from uniform draws is what lets a
        waypoint land in a different IK branch from its neighbour, and the joint-space lerp
        between two branches is the arc that P16's `place` segment was sweeping.
        """
        qs, out_pts, qq = [], [], seed
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            k = max(1, int(math.ceil(float((b - a).norm()) / args_cli.seg)))
            for j in range(k):
                t = a + (b - a) * (j + 1) / k
                c = solve(t, qq, 2, 0.08, 1)
                qq = c["q"]
                qs.append(qq)
                out_pts.append(t)
        return [pts[0]] + out_pts, [seed] + qs

    # Seed the dense chain from the SAME verified grasp pose the coarse chain uses, then
    # walk outward in both directions. The grasp is then bit-identical between the two arms
    # and anything the comparison shows is attributable to the path, not the pose.
    up_pts = [grip, lift, carry, place]
    dn_pts = [grip, approach_pts[2], approach_pts[1], approach_pts[0]]
    up_full, q_up = dense_chain(up_pts, q_grip_coarse)
    dn_full, q_dn = dense_chain(dn_pts, q_grip_coarse)
    dense_pts = list(reversed(dn_full)) + up_full[1:]
    dense_qs = list(reversed(q_dn)) + q_up[1:]
    names_cache.clear()
    names_cache.update({i: f"seg{i:02d}" for i in range(len(dense_qs))})
    w_dense, rows_dense = audit(dense_pts, dense_qs, f"DENSE chain ({len(dense_qs)} waypoints, "
                                                     f"<= {args_cli.seg * 1000:.0f} mm apart)")
    print(f"\n   worst TCP deviation: coarse {w_coarse * 1000:.1f} mm  ->  dense "
          f"{w_dense * 1000:.1f} mm")

    i_grip_dense = len(dn_full) - 1     # index of the grasp waypoint in the dense chain

    # ------------------------------------------------------- Part C: paired execution
    env.reset()
    for _ in range(30):
        e.sim.step()
        e.scene.update(e.physics_dt)
    snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}
    tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
    dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
    o_env = target_axis(e).clone() if args_cli.yaw else None
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

    def per_env(pts, qs, i_grip):
        """Shift the whole chain to this env's target, matching yaw where it matters."""
        d = torch.stack([tpos0[:, 0] - ROW_X, tpos0[:, 1], torch.zeros(n, device=dev)], dim=1)
        out = []
        for i, (p, q) in enumerate(zip(pts, qs)):
            # only the pre-grasp half tracks the target; the goal half is absolute
            shift = d if i <= i_grip else torch.zeros_like(d)
            od = o_env if (o_env is not None and i <= i_grip + 1) else None
            out.append(K.refine(q.unsqueeze(0).repeat(n, 1),
                                p.unsqueeze(0) + shift, iters=3, o_des=od))
        return out

    def execute(chain, i_grip, tag):
        restore()
        tp = Tape(e, dpos0)
        K.teleport_arm(chain[0], Q_OPEN)
        tp.label = "descend"
        hold_t(K, tp, chain[0], 80)
        steps = max(6, int(round(25 * 3 / max(1, i_grip))))
        for i in range(i_grip):
            run_t(K, tp, chain[i], chain[i + 1], steps)
        tp.label = "predwell"
        hold_t(K, tp, chain[i_grip], 160)
        tp.label = "close"
        hold_t(K, tp, chain[i_grip], 560, q_fing=Q_CLOSE)
        gap_stall = K.gap().clone()
        nseg = len(chain) - 1 - i_grip
        steps = max(6, int(round(90 / max(1, nseg))))
        for i in range(i_grip, len(chain) - 1):
            tp.label = "carry"
            run_t(K, tp, chain[i], chain[i + 1], steps, q_fing=Q_CLOSE)
        tp.label = "dwell"
        hold_t(K, tp, chain[-1], 160, q_fing=Q_CLOSE)
        tp.label = "release"
        hold_t(K, tp, chain[-1], 240, q_fing=Q_OPEN)
        tp.label = "retreat"
        run_t(K, tp, chain[-1], chain[-2], 25, q_fing=Q_OPEN)
        tp.label = "final"
        hold_t(K, tp, chain[-2], 240, q_fing=Q_OPEN)

        bpos = _t(e.scene["target"].data.root_pos_w) - org
        up_end = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
        topp = (up_end < mdp_cl.TOPPLE_DOT).any(dim=1)
        held = (gap_stall - width).abs() < 0.012
        at_goal = (((bpos[:, :2] - goal).norm(dim=1) < mdp_cl.GOAL_RADIUS)
                   & (bpos[:, 2] < 0.055))
        succ = at_goal & ~topp

        UP = torch.stack(tp.up)
        DS = torch.stack(tp.disp)
        ph, T = tp.phase, UP.shape[0]
        hit = ((DS > DISP_ON) | (UP < TILT_ON)).any(dim=2)
        ever = hit.any(dim=0)
        idx = torch.where(ever, hit.float().argmax(dim=0),
                          torch.full((n,), -1, device=dev, dtype=torch.long))
        bad = (UP < mdp_cl.TOPPLE_DOT).any(dim=2)
        ever_t = bad.any(dim=0)
        idx_t = torch.where(ever_t, bad.float().argmax(dim=0),
                            torch.full((n,), -1, device=dev, dtype=torch.long))
        order = ["descend", "predwell", "close", "carry", "dwell", "release",
                 "retreat", "final"]
        print(f"\n   {tag}")
        print(f"      enclosed {float(held.float().mean()):6.1%} | at goal "
              f"{float(at_goal.float().mean()):6.1%} | topple "
              f"{float(topp.float().mean()):6.1%} | SUCCESS "
              f"{float(succ.float().mean()):6.1%}")
        # HAZARD rate, not raw count: raw counts only ever show the first contact, which
        # makes any late phase look harmless once an early one is bad.
        print(f"      {'phase':>10} | {'reached clean':>13} | {'contacts':>9} | {'hazard':>8}")
        clean = torch.ones(n, dtype=torch.bool, device=dev)
        haz = {}
        for p in order:
            st = [t for t in range(T) if ph[t] == p]
            if not st:
                continue
            inp = ever & (idx >= st[0]) & (idx <= st[-1])
            k, r = int(inp.sum()), int(clean.sum())
            if r:
                haz[p] = k / r
                if k:
                    print(f"      {p:>10} | {r:13d} | {k:9d} | {k / r:7.1%}")
            clean = clean & ~inp
        return succ, topp, held, at_goal, haz

    coarse_chain = per_env(coarse_pts, coarse_qs, 3)
    dense_chain_q = per_env(dense_pts, dense_qs, i_grip_dense)
    s1, t1, h1, g1, hz1 = execute(coarse_chain, 3, "COARSE chain")
    s2, t2, h2, g2, hz2 = execute(dense_chain_q, i_grip_dense, "DENSE chain")

    print("\n" + "=" * 100)
    print(f"   coarse success {float(s1.float().mean()):.1%} | topple "
          f"{float(t1.float().mean()):.1%}")
    print(f"   dense  success {float(s2.float().mean()):.1%} | topple "
          f"{float(t2.float().mean()):.1%}")
    print(f"   paired: fixed {int((~s1 & s2).sum())}, broke {int((s1 & ~s2).sum())}, "
          f"net {int((~s1 & s2).sum()) - int((s1 & ~s2).sum()):+d}")
    print(f"\n   dense chain stratified by min free gap:")
    for lo, hi in ((0, 6), (6, 8), (8, 10), (10, 14)):
        mm = (min_gap * 1000 >= lo) & (min_gap * 1000 < hi)
        if int(mm.sum()):
            print(f"      {f'{lo}-{hi} mm':>10} | n {int(mm.sum()):4d} | "
                  f"success {float(s2[mm].float().mean()):6.1%}")

    out = {"n": n, "seg_mm": args_cli.seg * 1000, "yaw": bool(args_cli.yaw),
           "worst_dev_coarse_mm": w_coarse * 1000, "worst_dev_dense_mm": w_dense * 1000,
           "audit_coarse": rows_coarse, "audit_dense": rows_dense,
           "coarse": {"success": float(s1.float().mean()), "topple": float(t1.float().mean()),
                      "encl": float(h1.float().mean()), "at_goal": float(g1.float().mean()),
                      "hazard": hz1, "succ_mask": s1.tolist()},
           "dense": {"success": float(s2.float().mean()), "topple": float(t2.float().mean()),
                     "encl": float(h2.float().mean()), "at_goal": float(g2.float().mean()),
                     "hazard": hz2, "succ_mask": s2.tolist()},
           "min_gap_mm": (min_gap * 1000).tolist()}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p17] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
