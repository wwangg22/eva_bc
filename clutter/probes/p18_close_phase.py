# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P18 -- The last mode: 48 % hazard during the finger close, and it is the wrist.

State of the expert
-------------------
P17's dense-Cartesian chain reaches **enclosed 100 %, at goal 100 %, success 62.5 %** on 128
random spawns. Its per-phase hazard -- contacts in a phase over the envs that reached it
clean -- has exactly one term left:

    descend 0 % | predwell 0 % | **close 48.4 %** | carry 3.0 % | dwell 0 % | release 0 %
    retreat 0 % | final 0 %

The hypothesis, and why it is not a guess
-----------------------------------------
The victim is not symmetric. Across four independent P16 arms:

    d0 (y = -84)  8-16      d1 (y = -42)  76-90      d2 (y = +42)  24-44      d3 (y = +84)  0

`distractor_1` in roughly two thirds of every batch, and `distractor_3` never. Something
directional is at fault, on the -y side.

The clearance pose has `a_hat = (+0.08, +0.47, +0.88)`. With `o_hat = x_hat` the wrist stub
trails the TCP by 41.9 mm along `a_hat`, so it sits at

    y = -41.9 * 0.467 = **-19.6 mm**,  z = 65 - 41.9 * 0.880 = **28 mm**

which is inside the -y free gap: the target's face is at y = -15 mm, `distractor_1`'s at
y = -27 mm. **The wrist has 7 mm of room on each side, at a height well below the block
tops, and it is the nearest thing to d1 for the entire grasp.** It is statically clear --
`pen = 0.0` -- and clear for the whole descent, which is why the descent hazard is zero.
Then the fingers close, the drive reacts against the grip force, and 7 mm is not enough.

The control that decides it
---------------------------
This is testable rather than arguable. Solve the **mirror** pose -- same accuracy, same
alignment, same zero penetration, wrist in the **+y** gap -- and re-run. If the wrist is the
culprit the victim distribution must flip from d1 to d2. If it does not flip, the wrist is
exonerated and the cause is something symmetric (blade width, or the block tipping under a
grip taken 2 mm below its top face), and this probe says so.

A control that can come out either way is the only kind worth running; Stage 0 produced six
confident wrong answers and the positive control caught five of them.

Three arms, paired
------------------
    A  wrist in -y gap    the P17 pose, reproduced
    B  wrist in +y gap    the mirror -- the control
    C  per-env choice     wrist placed in whichever gap this spawn actually made wider

C is the fix if the control confirms the mechanism: the reset jitters each distractor's y by
+/-5 mm independently, so the two gaps are rarely equal, and the side with more room is known
before the arm moves.

Usage
-----
    python eva_bc/clutter/probes/p18_close_phase.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Close-phase forensics and the wrist control.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=12)
parser.add_argument("--tries", type=int, default=14)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--lift_dz", type=float, default=0.150)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--margin", type=float, default=0.005)
parser.add_argument("--seg", type=float, default=0.030)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p18_close.json")
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
ORDER = ["descend", "predwell", "close", "carry", "dwell", "release", "retreat", "final"]


class Tape:
    """Adds gripper-body and target geometry, so the close phase can be dissected."""

    def __init__(self, K, e, dpos0):
        self.K, self.e, self.dpos0 = K, e, dpos0
        self.up, self.disp, self.phase = [], [], []
        self.grip_bodies, self.tgt, self.gap = [], [], []
        self.label = "descend"

    def sample(self):
        org = self.e.scene.env_origins
        up = torch.stack([mdp_cl._up_z(self.e, d) for d in DIST], dim=1)
        dp = torch.stack([(_t(self.e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1)
        self.up.append(up)
        self.disp.append((dp[:, :, :2] - self.dpos0[:, :, :2]).norm(dim=2))
        bp = _t(self.K.robot.data.body_pos_w) - org.unsqueeze(1)
        self.grip_bodies.append(
            bp[:, [self.K.i_end, self.K.i_left, self.K.i_right], :].clone())
        self.tgt.append((_t(self.e.scene["target"].data.root_pos_w) - org).clone())
        self.gap.append(self.K.gap().clone())
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
    print("P18 -- CLOSE-PHASE FORENSICS AND THE WRIST CONTROL")
    print("=" * 100)

    def wrist_of(q):
        return K.fk(q.unsqueeze(0).repeat(n, 1))["bodies"][0, K.i_end].clone()

    def solve(pos, seed, tries, std0, restarts, pos_max=0.0015, side=0):
        """`side` = -1 / +1 restricts the wrist stub to that side of the target, 0 = free."""
        best = None
        for _ in range(tries):
            c = K.cem(pos, seed, o_des=Y, w_o=0.60, iters=args_cli.iters, std0=std0,
                      restarts=restarts, avoid=boxes, avoid_margin=args_cli.margin)
            if c["pos_err"] > pos_max:
                continue
            if side and float(wrist_of(c["q"])[1]) * side <= 0:
                continue
            key = (round(c["pen"], 4), -c["o_align"])
            if best is None or key < (round(best["pen"], 4), -best["o_align"]):
                best = c
        return best

    print("\n   GRASP POSES (wrist stub = gripper_end origin, the deepest body in the row)")
    poses = {}
    for tag, side in (("-y gap", -1), ("+y gap", +1)):
        c = solve(grip, K.q_arm0, args_cli.tries, 0.6, args_cli.restarts, side=side)
        if c is None:
            print(f"   {tag:>8}: no pose found with the wrist on this side")
            continue
        w = wrist_of(c["q"])
        poses[side] = c
        print(f"   {tag:>8}: err {c['pos_err'] * 1000:5.2f} mm | o_align {c['o_align']:.3f} | "
              f"pen {c['pen'] * 1000:4.1f} mm | a_hat ({c['a_hat'][0]:+.2f},"
              f"{c['a_hat'][1]:+.2f},{c['a_hat'][2]:+.2f}) | wrist "
              f"({float(w[0]) * 1000:.0f},{float(w[1]) * 1000:+.0f},{float(w[2]) * 1000:.0f}) mm")
    if len(poses) < 2:
        print("\n   Cannot run the control: only one wrist side is attainable. That is itself")
        print("   a finding -- the mechanism cannot be tested by mirroring at this grip height.")
        env.close()
        return

    # --------------------------------------------------- dense Cartesian chains, per side
    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
    carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + dz], device=dev)
    place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
    approach_pts = [grip + STANDOFF * down * (-(3 - k) / 3.0) for k in range(3)]

    def dense(pts, seed):
        qs, out, qq = [], [], seed
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            k = max(1, int(math.ceil(float((b - a).norm()) / args_cli.seg)))
            for j in range(k):
                t = a + (b - a) * (j + 1) / k
                c = solve(t, qq, 2, 0.08, 1)
                qq = c["q"] if c else qq
                qs.append(qq)
                out.append(t)
        return [pts[0]] + out, [seed] + qs

    chains = {}
    for side, c in poses.items():
        up_full, q_up = dense([grip, lift, carry, place], c["q"])
        dn_full, q_dn = dense([grip, approach_pts[2], approach_pts[1], approach_pts[0]], c["q"])
        chains[side] = (list(reversed(dn_full)) + up_full[1:],
                        list(reversed(q_dn)) + q_up[1:], len(dn_full) - 1)
    assert chains[-1][2] == chains[1][2] and len(chains[-1][1]) == len(chains[1][1])
    i_grip = chains[-1][2]
    print(f"\n   dense chains: {len(chains[-1][1])} waypoints, grasp at index {i_grip}")

    # --------------------------------------------------- one spawn for every arm
    env.reset()
    for _ in range(30):
        e.sim.step()
        e.scene.update(e.physics_dt)
    snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}
    tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
    dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
    o_env = target_axis(e).clone()
    # the two gaps flanking the target, measured off THIS spawn
    gap_m = (tpos0[:, 1] - HY) - (dpos0[:, 1, 1] + HY)      # target's -y face to d1's +y face
    gap_p = (dpos0[:, 2, 1] - HY) - (tpos0[:, 1] + HY)      # target's +y face to d2's -y face
    ys = torch.cat([dpos0[:, :2, 1], tpos0[:, 1:2], dpos0[:, 2:, 1]], dim=1)
    ys, _ = ys.sort(dim=1)
    min_gap = (ys[:, 1:] - ys[:, :-1] - 2 * HY).min(dim=1).values
    print(f"   spawn: -y gap {float(gap_m.median()) * 1000:.1f} mm median, +y gap "
          f"{float(gap_p.median()) * 1000:.1f} mm median | wider side is +y in "
          f"{float((gap_p > gap_m).float().mean()):.0%} of envs")

    def restore():
        for k, v in snap.items():
            e.scene[k].write_root_state_to_sim(v.clone())
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def per_env(pts, qs, sel=None, alt=None):
        """Shift to this env's target and match its yaw; `sel` picks between two chains."""
        d = torch.stack([tpos0[:, 0] - ROW_X, tpos0[:, 1], torch.zeros(n, device=dev)], dim=1)
        out = []
        for i, (p, q) in enumerate(zip(pts, qs)):
            shift = d if i <= i_grip else torch.zeros_like(d)
            od = o_env if i <= i_grip + 1 else None
            q0 = q.unsqueeze(0).repeat(n, 1)
            if sel is not None:
                q0 = torch.where(sel.unsqueeze(1), alt[i].unsqueeze(0).repeat(n, 1), q0)
            out.append(K.refine(q0, p.unsqueeze(0) + shift, iters=3, o_des=od))
        return out

    def execute(chain, tag):
        restore()
        tp = Tape(K, e, dpos0)
        K.teleport_arm(chain[0], Q_OPEN)
        tp.label = "descend"
        hold_t(K, tp, chain[0], 80)
        st = max(6, int(round(75 / max(1, i_grip))))
        for i in range(i_grip):
            run_t(K, tp, chain[i], chain[i + 1], st)
        tp.label = "predwell"
        hold_t(K, tp, chain[i_grip], 160)
        tp.label = "close"
        hold_t(K, tp, chain[i_grip], 560, q_fing=Q_CLOSE)
        gap_stall = K.gap().clone()
        st = max(6, int(round(90 / max(1, len(chain) - 1 - i_grip))))
        tp.label = "carry"
        for i in range(i_grip, len(chain) - 1):
            run_t(K, tp, chain[i], chain[i + 1], st, q_fing=Q_CLOSE)
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

        UP, DS = torch.stack(tp.up), torch.stack(tp.disp)
        GB, GP = torch.stack(tp.grip_bodies), torch.stack(tp.gap)
        ph, T = tp.phase, UP.shape[0]
        hitd = (DS > DISP_ON) | (UP < TILT_ON)              # (T,n,4)
        hit = hitd.any(dim=2)
        ever = hit.any(dim=0)
        idx = torch.where(ever, hit.float().argmax(dim=0),
                          torch.full((n,), -1, device=dev, dtype=torch.long))
        ar = torch.arange(n, device=dev)
        victim = hitd[idx.clamp(min=0), ar].float().argmax(dim=1)

        print(f"\n   {tag}")
        print(f"      enclosed {float(held.float().mean()):6.1%} | at goal "
              f"{float(at_goal.float().mean()):6.1%} | topple "
              f"{float(topp.float().mean()):6.1%} | SUCCESS "
              f"{float(succ.float().mean()):6.1%}")
        print(f"      {'phase':>10} | {'reached clean':>13} | {'contacts':>9} | {'hazard':>8}")
        clean = torch.ones(n, dtype=torch.bool, device=dev)
        haz = {}
        for p in ORDER:
            s = [t for t in range(T) if ph[t] == p]
            if not s:
                continue
            inp = ever & (idx >= s[0]) & (idx <= s[-1])
            k, r = int(inp.sum()), int(clean.sum())
            if r:
                haz[p] = k / r
                if k:
                    print(f"      {p:>10} | {r:13d} | {k:9d} | {k / r:7.1%}")
            clean = clean & ~inp
        vic = [int((ever & (victim == k)).sum()) for k in range(4)]
        print(f"      victims      : "
              + ", ".join(f"d{k}(y={ROW_Y[k] * 1000:+.0f}) {vic[k]}" for k in range(4)))

        # ---- inside the close: where in the squeeze, and what was nearest?
        cs = [t for t in range(T) if ph[t] == "close"]
        inclose = ever & (idx >= cs[0]) & (idx <= cs[-1])
        cdet = {}
        if int(inclose.sum()):
            j = idx[inclose]
            frac = (j - cs[0]).float() / max(1, len(cs) - 1)
            gp = GP[j, ar[inclose]]
            vy = torch.stack(tp.disp)[0].new_zeros(1)  # placeholder, unused
            bodies = GB[j, ar[inclose]]                       # (m,3,3) end/left/right
            vic_pos = dpos0[ar[inclose], victim[inclose]]
            dd = (bodies - vic_pos.unsqueeze(1)).norm(dim=2)
            near = dd.argmin(dim=1)
            nm = ["gripper_end", "gripper_left", "gripper_right"]
            print(f"      close detail : onset at {float(frac.median()):.0%} of the squeeze | "
                  f"finger gap then {float(gp.median()) * 1000:.1f} mm "
                  f"(open 89.1, stall {float(gap_stall.median()) * 1000:.1f})")
            for b in range(3):
                mb = near == b
                if int(mb.sum()):
                    print(f"                   nearest {nm[b]:>14}: {int(mb.sum()):3d} envs, "
                          f"median {float(dd[mb, b].median()) * 1000:5.1f} mm from the victim")
            cdet = {"frac": float(frac.median()), "gap_mm": float(gp.median()) * 1000}
        return succ, topp, haz, vic, cdet

    pm, pp, im = chains[-1]
    pp2, qp2, _ = chains[1]
    res, masks = [], {}
    for tag, sel in (("A wrist -y", None), ("B wrist +y", "mirror"),
                     ("C per-env side", "adaptive")):
        if sel is None:
            chain = per_env(pm, pp)
        elif sel == "mirror":
            chain = per_env(pp2, qp2)
        else:
            # put the wrist in whichever gap this spawn actually made wider
            chain = per_env(pm, pp, sel=(gap_p > gap_m), alt=qp2)
        s, t, h, v, cd = execute(chain, tag)
        masks[tag] = s
        res.append({"arm": tag, "success": float(s.float().mean()),
                    "topple": float(t.float().mean()), "hazard": h, "victim": v,
                    "close": cd, "succ_mask": s.tolist()})

    print("\n" + "=" * 100)
    print("THE CONTROL")
    print("=" * 100)
    a, b = res[0]["victim"], res[1]["victim"]
    print(f"   wrist in -y gap -> victims d0..d3 = {a}")
    print(f"   wrist in +y gap -> victims d0..d3 = {b}")
    flipped = (a[1] > a[2]) != (b[1] > b[2])
    if flipped:
        print("   -> The victim FLIPPED with the wrist. The wrist stub is the culprit,")
        print("      confirmed causally rather than by geometry alone.")
    else:
        print("   -> The victim did NOT flip. The wrist is exonerated; the close-phase")
        print("      contact is symmetric and must come from the blades or the block itself.")
    print(f"\n   {'arm':>16} | {'success':>8} | {'topple':>8} | {'close hazard':>13} | net vs A")
    base = masks["A wrist -y"]
    for r in res:
        m = torch.tensor(r["succ_mask"], device=dev)
        print(f"   {r['arm']:>16} | {r['success']:7.1%} | {r['topple']:7.1%} | "
              f"{r['hazard'].get('close', 0):12.1%} | "
              f"{int((~base & m).sum()) - int((base & ~m).sum()):+d}")

    out = {"n": n, "grip_z": args_cli.grip_z, "flipped": bool(flipped),
           "gap_minus_mm": (gap_m * 1000).tolist(), "gap_plus_mm": (gap_p * 1000).tolist(),
           "min_gap_mm": (min_gap * 1000).tolist(), "arms": res}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p18] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
