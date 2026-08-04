# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P12 -- WHAT touches WHAT, and WHEN? Phase-resolved forensics on the 75 % topple rate.

Why this probe exists
---------------------
P10 measured the orthogonal grasp on 256 random spawns: **enclosed 100 %, target at goal
99 %, topple 75 %, success 25 %**. At-goal is essentially saturated, so the topple rate *is*
the residual -- every point removed converts almost one-for-one into success.

75 % is far too high to be a precision problem. A precision problem produces a rate that
tracks clearance; this one is nearly universal. So before tuning anything (grip height,
approach speed, yaw matching), the question to answer is the forensic one:

    which body contacts which distractor, during which phase of the trajectory?

Everything downstream depends on the answer, and the answers imply completely different
fixes:

    * during DESCENT   -> the fingers are too wide / mis-aligned; fix pose selection
    * during CLOSE     -> the finger blades are wider in y than assumed; fix grip geometry
    * during LIFT      -> the target is dragging a neighbour; fix the lift direction
    * during CARRY     -> the *carried target* is sweeping the row; fix the lift HEIGHT
    * during RETREAT   -> an empty gripper is sweeping back over the row; delete the motion

Two of those are free fixes to a trajectory that is already 99 % accurate at the goal.

The specific hypothesis this probe was written to test
-----------------------------------------------------
Reading `clutter_env_cfg.py` after P10:

    GOAL_XY = (0.185, -0.185)          distractor_0 at y = -84 mm, distractor_1 at y = -42 mm

so the carry runs diagonally **over distractors 0 and 1**, not away from the row. And the
arithmetic of the lift is uncomfortably tight:

    grip point            z = 65 mm      (chosen by P11 to clear the neighbours on the way IN)
    block centre settles  z = 32 mm      -> the block hangs 33 mm BELOW the TCP
    block half-height     35 mm          -> its bottom face is 68 mm below the TCP
    lift is               +75 mm         -> TCP 140 mm, carried block's bottom at **72 mm**
    distractor top        67 mm          -> clearance = **5 mm**

Five millimetres, over a joint-space interpolation that does not travel in a straight
Cartesian line and is free to sag. The grip height that makes the *approach* work is the
same choice that makes the *carry* skim the row. If that is the mechanism, the topple is
being caused by the block the arm is holding, and no amount of approach tuning touches it.

The probe does not assume that. It records the geometry and lets the data pick.

Method
------
Same trajectory as P10 (phi = 90, grip_z = 0.065), executed **physics-only** so a topple
does not reset the scene and erase itself (retraction 7.5). Every control step samples:

    * `up_z` of all four distractors      -> topple and disturbance onset
    * all robot body positions             -> which body was nearest at onset
    * all distractor positions             -> pushed vs struck from above
    * the target's position                -> is the CARRIED block the culprit?

Post-processing finds, per env, the first step at which any distractor is disturbed
(`up_z < 0.98`, well before the 0.75 termination) and the first at which it actually
topples, attributes each to a phase, and reports what was closest to the victim at that
instant. Disturbance onset is the informative one: by the time `up_z < 0.75` the block is
already falling and the culprit has moved on.

Usage
-----
    python eva_bc/clutter/probes/p12_topple_forensics.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase-resolved topple forensics.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=10)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--lift_dz", type=float, default=0.075,
                    help="lift height above the grip point [m]; the P10 baseline is 0.075")
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p12_forensics.json")
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

#: `up_z` below this counts as "disturbed" -- 0.98 is 11.5 deg of tilt, far short of the
#: 0.75 (41.4 deg) termination. Onset is what identifies the culprit; by the time the block
#: is past 0.75 it is falling under gravity and the arm has moved on.
DISTURB_DOT = 0.98

PHASES = ["settle", "descend", "predwell", "close", "lift", "carry",
          "place", "dwell", "release", "retreat", "final"]


class Trace:
    """Per-control-step recorder. One row per sample, phase-labelled."""

    def __init__(self, K, e):
        self.K, self.e = K, e
        self.up, self.which, self.phase = [], [], []
        self.bp, self.dp, self.tp = [], [], []
        self.label = "settle"

    def sample(self):
        org = self.e.scene.env_origins
        up = torch.stack([mdp_cl._up_z(self.e, d) for d in DIST], dim=1)      # (n, 4)
        m, w = up.min(dim=1)
        self.up.append(m)
        self.which.append(w)
        self.phase.append(self.label)
        self.bp.append((_t(self.K.robot.data.body_pos_w) - org.unsqueeze(1)).clone())
        self.dp.append(torch.stack(
            [(_t(self.e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone())
        self.tp.append((_t(self.e.scene["target"].data.root_pos_w) - org).clone())


def hold_t(K, tr, q, steps, q_fing=Q_OPEN, every=8):
    """`hold_phys` with a trace sample every `every` physics steps."""
    qd = K._drive(q, q_fing)
    for s in range(steps):
        K.robot.set_joint_position_target(qd)
        K.robot.write_data_to_sim()
        K.e.sim.step()
        K.e.scene.update(K.e.physics_dt)
        if (s + 1) % every == 0:
            tr.sample()


def run_t(K, tr, q_from, q_to, steps, q_fing=Q_OPEN, substeps=8):
    """`run_phys` with a trace sample after every control step."""
    for s in range(steps):
        f = (s + 1) / steps
        qd = K._drive((1 - f) * q_from + f * q_to, q_fing)
        for _ in range(substeps):
            K.robot.set_joint_position_target(qd)
            K.robot.write_data_to_sim()
            K.e.sim.step()
            K.e.scene.update(K.e.physics_dt)
        tr.sample()


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
    body_names = list(K.robot.body_names)

    _p = math.radians(args_cli.phi)
    Y = torch.tensor([math.sin(_p), math.cos(_p), 0.0], device=dev)
    width = 2 * HX if args_cli.phi >= 45.0 else 2 * HY

    print("\n" + "=" * 100)
    print("P12 -- TOPPLE FORENSICS: which body, which distractor, which phase")
    print("=" * 100)

    def best_pose(pos, seed, tries=6, std0=0.6, restarts=None, pos_max=0.0015):
        """Highest `o_align` subject to a hard position gate -- see P10's docstring."""
        best = None
        for _ in range(tries):
            c = K.cem(pos, seed, o_des=Y, a_des=None, w_o=0.60, iters=args_cli.iters,
                      std0=std0, restarts=restarts or args_cli.restarts)
            if c["pos_err"] > pos_max:
                continue
            if best is None or c["o_align"] > best["o_align"]:
                best = c
        return best or c

    # ---------------- the whole plan, solved before anything closes (convention S9)
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    r = best_pose(grip, K.q_arm0)
    q_grip = r["q"]
    print(f"   nominal grasp: err {r['pos_err'] * 1000:.2f} mm, o_align {r['o_align']:.3f}")

    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    qq, appr = q_grip, []
    for t in lerp_pts(grip, grip - STANDOFF * down, 3):
        qq = best_pose(t, qq, tries=3, std0=0.15)["q"]
        appr.append(qq)
    approach_nom = list(reversed(appr))

    dz = args_cli.lift_dz
    lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
    carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + dz], device=dev)
    place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
    qq, post = q_grip, []
    for t in [lift, carry, place]:
        qq = best_pose(t, qq, tries=3, std0=0.30, restarts=3)["q"]
        post.append(qq)

    # The arithmetic that motivated this probe, printed from the ACHIEVED poses rather than
    # the commanded ones, so the number in the log is the one physics will see. The block is
    # gripped at `grip_z` while its centre sits at ~32 mm, so it hangs (grip_z - 0.032 + HZ)
    # below the TCP -- 68 mm at the default. That is the number that decides the carry.
    g_lift = K.fk(post[0].unsqueeze(0).repeat(n, 1))
    hang = args_cli.grip_z - 0.032 + HZ
    print(f"   lift TCP z = {float(g_lift['tcp'][0, 2]) * 1000:.1f} mm; block hangs "
          f"{hang * 1000:.0f} mm below the TCP -> carried bottom "
          f"{(float(g_lift['tcp'][0, 2]) - hang) * 1000:.1f} mm vs distractor tops ~67 mm")

    # ---------------- STATIC ROW CLEARANCE OF THE WHOLE ARM AT THE GRASP POSE
    # P11 checked that the FINGERS clear the row. It never checked the rest of the arm.
    # `o_align` and TCP error together pin down the tool frame; they say nothing about the
    # elbow, the forearm or the wrist, any of which the CEM is free to route straight
    # through the neighbours. This table is the check that was missing.
    g_grip = K.fk(q_grip.unsqueeze(0).repeat(n, 1))
    bpg = (_t(K.robot.data.body_pos_w) - org.unsqueeze(1))[0]         # (B,3), env frame
    print("\n   ARM GEOMETRY AT THE NOMINAL GRASP POSE (env frame, mm)")
    print(f"      {'body':>16} | {'x':>7} | {'y':>7} | {'z':>7} | clearance to nearest block")
    row_y = torch.tensor([-0.084, -0.042, 0.0, 0.042, 0.084], device=dev)
    for b, name in enumerate(body_names):
        p = bpg[b]
        # horizontal distance to each row block's centre-line, and height above the row
        dy = (p[1] - row_y).abs().min()
        note = ""
        if float(p[2]) < 0.075 and float(dy) < 0.030 and abs(float(p[0]) - ROW_X) < 0.060:
            note = "  <-- INSIDE THE ROW ENVELOPE"
        print(f"      {name:>16} | {float(p[0]) * 1000:7.1f} | {float(p[1]) * 1000:7.1f} | "
              f"{float(p[2]) * 1000:7.1f} |{note}")

    # ---------------- reset, settle, adapt
    env.reset()
    for _ in range(30):
        e.sim.step()
        e.scene.update(e.physics_dt)

    tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
    dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
    ys = torch.cat([dpos0[:, :2, 1], tpos0[:, 1:2], dpos0[:, 2:, 1]], dim=1)
    ys, _ = ys.sort(dim=1)
    min_gap = (ys[:, 1:] - ys[:, :-1] - 2 * HY).min(dim=1).values
    rel_dx = dpos0[:, 1:3, 0].mean(dim=1) - tpos0[:, 0]

    tgt = torch.stack([tpos0[:, 0], tpos0[:, 1],
                       torch.full((n,), args_cli.grip_z, device=dev)], dim=1)
    q_g = K.refine(q_grip.unsqueeze(0).repeat(n, 1), tgt, iters=4)
    q_app = [K.refine(q.unsqueeze(0).repeat(n, 1),
                      tgt + torch.tensor([0.0, 0.0, d], device=dev), iters=3)
             for q, d in zip(approach_nom, (STANDOFF, 2 * STANDOFF / 3, STANDOFF / 3))]

    # ---------------- execute with tracing
    tr = Trace(K, e)
    K.teleport_arm(q_app[0], Q_OPEN)
    tr.label = "settle"
    hold_t(K, tr, q_app[0], 80)

    tr.label = "descend"
    seq = q_app + [q_g]
    for i in range(len(seq) - 1):
        run_t(K, tr, seq[i], seq[i + 1], 25)

    tr.label = "predwell"
    hold_t(K, tr, q_g, 160)
    tcp_grasp = K.tcp_now().clone()

    tr.label = "close"
    hold_t(K, tr, q_g, 560, q_fing=Q_CLOSE)
    gap_stall = K.gap().clone()

    chain = [q_g] + [p.unsqueeze(0).repeat(n, 1) for p in post]
    for i, lab in enumerate(("lift", "carry", "place")):
        tr.label = lab
        run_t(K, tr, chain[i], chain[i + 1], 30, q_fing=Q_CLOSE)

    tr.label = "dwell"
    hold_t(K, tr, chain[-1], 160, q_fing=Q_CLOSE)
    tr.label = "release"
    hold_t(K, tr, chain[-1], 240, q_fing=Q_OPEN)
    tr.label = "retreat"
    run_t(K, tr, chain[-1], chain[1], 25, q_fing=Q_OPEN)
    tr.label = "final"
    hold_t(K, tr, chain[1], 240, q_fing=Q_OPEN)

    # ---------------- outcome, scored exactly as P10 scored it
    bpos = _t(e.scene["target"].data.root_pos_w) - org
    up_end = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
    topp = (up_end < mdp_cl.TOPPLE_DOT).any(dim=1)
    held = (gap_stall - width).abs() < 0.012
    at_goal = ((bpos[:, :2] - goal).norm(dim=1) < mdp_cl.GOAL_RADIUS) & (bpos[:, 2] < 0.055)
    succ = at_goal & ~topp
    print(f"\n   enclosed {float(held.float().mean()):.1%} | at goal "
          f"{float(at_goal.float().mean()):.1%} | topple {float(topp.float().mean()):.1%} | "
          f"SUCCESS {float(succ.float().mean()):.1%}   (P10 baseline: 100 / 99 / 75 / 25)")

    # ---------------- forensics
    UP = torch.stack(tr.up)                       # (T, n)
    WH = torch.stack(tr.which)                    # (T, n)
    BP = torch.stack(tr.bp)                       # (T, n, B, 3)
    DP = torch.stack(tr.dp)                       # (T, n, 4, 3)
    TP = torch.stack(tr.tp)                       # (T, n, 3)
    T = UP.shape[0]
    ph = tr.phase
    print(f"\n   trace: {T} samples over {len(set(ph))} phases")

    def onset(mask):
        """First sample index where `mask` (T,n) is true; -1 where it never is."""
        ever = mask.any(dim=0)
        idx = mask.float().argmax(dim=0)
        return torch.where(ever, idx, torch.full_like(idx, -1)), ever

    i_top, ever_top = onset(UP < mdp_cl.TOPPLE_DOT)
    i_dis, ever_dis = onset(UP < DISTURB_DOT)

    def phase_hist(idx, ever, title):
        print(f"\n   {title}")
        print(f"      {'phase':>10} | {'n':>5} | {'share of all envs':>18}")
        rows = []
        for p in PHASES:
            steps = [t for t in range(T) if ph[t] == p]
            if not steps:
                continue
            m = ever & (idx >= steps[0]) & (idx <= steps[-1])
            k = int(m.sum())
            if k:
                print(f"      {p:>10} | {k:5d} | {k / n:17.1%}")
            rows.append({"phase": p, "n": k})
        print(f"      {'NEVER':>10} | {int((~ever).sum()):5d} | {float((~ever).float().mean()):17.1%}")
        return rows

    hist_top = phase_hist(i_top, ever_top, "TOPPLE onset (up_z < 0.75) by phase:")
    hist_dis = phase_hist(i_dis, ever_dis, f"DISTURBANCE onset (up_z < {DISTURB_DOT}) by phase:")

    # ---- at disturbance onset: what was closest to the victim?
    idx = i_dis.clamp(min=0)
    ar = torch.arange(n, device=dev)
    victim = WH[idx, ar]                                    # (n,) distractor index
    vpos = DP[idx, ar, victim]                              # (n,3) victim position
    bodies = BP[idx, ar]                                    # (n,B,3)
    tgt_at = TP[idx, ar]                                    # (n,3) carried target
    d_body = (bodies - vpos.unsqueeze(1)).norm(dim=2)       # (n,B)
    d_min, b_min = d_body.min(dim=1)
    d_tgt = (tgt_at - vpos).norm(dim=1)
    # is the culprit the arm, or the block the arm is carrying?
    culprit_is_target = d_tgt < d_min

    m = ever_dis
    print(f"\n   AT DISTURBANCE ONSET ({int(m.sum())} envs):")
    print(f"      {'victim':>14} | {'n':>5} |  (row y-offset)")
    for k, dname in enumerate(DIST):
        c = int((m & (victim == k)).sum())
        if c:
            print(f"      {dname:>14} | {c:5d} |  y = {[-84, -42, 42, 84][k]:+d} mm")

    print(f"\n      {'nearest robot body':>22} | {'n':>5} | {'median dist':>12}")
    for b in sorted(set(int(x) for x in b_min[m].tolist())):
        mm = m & (b_min == b)
        print(f"      {body_names[b]:>22} | {int(mm.sum()):5d} | "
              f"{float(d_body[mm, b].median()) * 1000:9.1f} mm")

    # Body ORIGINS are not contact surfaces -- a finger blade or a forearm shell reaches far
    # from its origin -- so "nearest origin" alone cannot name a culprit. Print the whole
    # arm's pose at the instant of onset instead, and let the geometry speak.
    print(f"\n      full arm pose at onset (median over disturbed envs, mm), victim at "
          f"({float(vpos[m][:, 0].median()) * 1000:.0f}, {float(vpos[m][:, 1].median()) * 1000:+.0f}, "
          f"{float(vpos[m][:, 2].median()) * 1000:.0f}):")
    print(f"      {'body':>16} | {'x':>7} | {'y':>7} | {'z':>7} | {'dist to victim':>14} |"
          f" {'dy':>7}")
    for b, name in enumerate(body_names):
        p = bodies[m][:, b, :]
        dyv = (p[:, 1] - vpos[m][:, 1]).abs()
        print(f"      {name:>16} | {float(p[:, 0].median()) * 1000:7.1f} | "
              f"{float(p[:, 1].median()) * 1000:7.1f} | {float(p[:, 2].median()) * 1000:7.1f} | "
              f"{float(d_body[m, b].median()) * 1000:11.1f} mm | "
              f"{float(dyv.median()) * 1000:7.1f}")

    # When in the descent, and at what height? Onset on the first descent step implicates a
    # high link; onset on the last implicates the fingers.
    desc = [t for t in range(T) if ph[t] == "descend"]
    if desc:
        md = m & (i_dis >= desc[0]) & (i_dis <= desc[-1])
        if int(md.sum()):
            frac = (i_dis[md] - desc[0]).float() / max(1, len(desc) - 1)
            tcpz = bodies[md][:, K.i_end, 2]
            print(f"\n      onset within the descent: median at "
                  f"{float(frac.median()):.0%} of the way down "
                  f"(gripper_end z = {float(tcpz.median()) * 1000:.1f} mm; "
                  f"range {float(tcpz.min()) * 1000:.0f}..{float(tcpz.max()) * 1000:.0f})")

    print(f"\n      culprit is the CARRIED TARGET : {int((m & culprit_is_target).sum()):4d} "
          f"({float((m & culprit_is_target).float().sum() / m.float().sum()):.1%} of disturbed)")
    print(f"      culprit is a ROBOT BODY       : {int((m & ~culprit_is_target).sum()):4d} "
          f"({float((m & ~culprit_is_target).float().sum() / m.float().sum()):.1%})")
    print(f"      median target->victim distance : {float(d_tgt[m].median()) * 1000:.1f} mm")
    print(f"      median  body ->victim distance : {float(d_min[m].median()) * 1000:.1f} mm")

    # ---- the specific hypothesis: does the carried block skim the row during the carry?
    car = [t for t in range(T) if ph[t] in ("lift", "carry")]
    if car:
        sl = slice(car[0], car[-1] + 1)
        tb = TP[sl, :, 2] - HZ                              # (t,n) carried block bottom
        dt = DP[sl, :, :, 2] + HZ                           # (t,n,4) distractor tops
        hxy = (TP[sl, :, None, :2] - DP[sl, :, :, :2]).norm(dim=3)   # (t,n,4) horizontal sep
        # only count a pair as a near-miss when it is horizontally overlapping-ish
        over = hxy < 0.045
        vclr = torch.where(over, tb.unsqueeze(2) - dt, torch.full_like(dt, 9.0))
        vmin = vclr.amin(dim=(0, 2))                        # (n,) worst vertical clearance
        print(f"\n   CARRIED-BLOCK SWEEP (lift+carry, pairs within 45 mm horizontally):")
        print(f"      worst vertical clearance, block bottom vs distractor top:")
        print(f"         median {float(vmin.median()) * 1000:+.1f} mm | "
              f"min {float(vmin.min()) * 1000:+.1f} mm | "
              f"share NEGATIVE (overlapping) {float((vmin < 0).float().mean()):.1%}")
    else:
        vmin = torch.zeros(n, device=dev)

    # ---- was the victim pushed sideways, or struck from above?
    dxy = (DP[idx, ar, victim, :2] - dpos0[ar, victim, :2]).norm(dim=1)
    print(f"\n      victim planar displacement at onset: median "
          f"{float(dxy[m].median()) * 1000:.2f} mm  "
          f"(large => pushed; ~0 => tipped in place / struck from above)")

    # ---- does the topple correlate with clearance at all?
    print(f"\n   topple vs per-episode minimum free gap:")
    print(f"      {'gap band [mm]':>16} | {'n':>5} | {'topple':>8} | {'success':>8}")
    for lo, hi in ((0, 8), (8, 10), (10, 12), (12, 100)):
        mm = (min_gap * 1000 >= lo) & (min_gap * 1000 < hi)
        if int(mm.sum()):
            print(f"      {f'{lo}-{hi}':>16} | {int(mm.sum()):5d} | "
                  f"{float(topp[mm].float().mean()):7.1%} | "
                  f"{float(succ[mm].float().mean()):7.1%}")

    out = {
        "n": n, "grip_z": args_cli.grip_z, "lift_dz": dz, "phi": args_cli.phi,
        "encl_rate": float(held.float().mean()),
        "at_goal_rate": float(at_goal.float().mean()),
        "topple_rate": float(topp.float().mean()),
        "success_rate": float(succ.float().mean()),
        "phase_hist_topple": hist_top, "phase_hist_disturb": hist_dis,
        "victim": victim.tolist(), "nearest_body": [body_names[int(b)] for b in b_min.tolist()],
        "d_body_mm": (d_min * 1000).tolist(), "d_target_mm": (d_tgt * 1000).tolist(),
        "culprit_is_target": culprit_is_target.tolist(),
        "carry_vclear_mm": (vmin * 1000).tolist(),
        "min_gap_mm": (min_gap * 1000).tolist(), "rel_dx_mm": (rel_dx * 1000).tolist(),
        "topple": topp.tolist(), "succ": succ.tolist(),
        "tcp_grasp_z_mm": (tcp_grasp[:, 2] * 1000).tolist(),
    }
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p12] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
