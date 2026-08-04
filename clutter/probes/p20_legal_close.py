# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P20 -- The ramped close is not policy-legal. What is?

The result that has to be converted
-----------------------------------
P19 ran four paired arms on one spawn. Driving the finger joint target smoothly from 45 mm
to 0 over 100 physics steps instead of commanding it shut in one:

    A  binary close   enclosed 100 % | at goal  99.2 % | topple 35.9 % | success 64.1 %
    C  ramped close   enclosed 100 % | at goal 100.0 % | topple  2.3 % | **success 97.7 %**

That is the whole remaining problem, solved -- by an action the benchmark does not offer.
`ActionsCfg.gripper` is a `BinaryJointPositionActionCfg`: `a[6] >= 0` commands 45 mm,
`a[6] < 0` commands 0, and there is nothing in between. A policy trained on demonstrations
that ramp the finger target is being taught to emit an action it cannot emit.

It did, however, settle the mechanism, and correct P18 in the process:

  * The wrist stub travels **0.13 mm median, 0.21 mm max** through the entire slam. It is not
    lurching into anything. **P18's causal claim -- "the wrist is the culprit" -- is wrong.**
    Its mirror arm differed in two variables at once, wrist side *and* jaw alignment
    (`o_align` 0.885 vs 0.995), and the second is what moved the victim.
  * The ordering is unambiguous: the jaw bites at step 3, `d1` moves at step 5. Slowing the
    jaw moves `d1`'s first motion to step 54. The fingers strike the target and the target
    strikes its neighbour.
  * Alignment dominates the hazard. Across every arm measured so far:
        o_align 0.991 -> close hazard 52 %      o_align 0.885 -> 97 %
        o_align 0.848 -> 92 %                   (P18 B, P19 D)

Candidates that ARE legal
-------------------------
    A  binary, grip 65 mm      the P17/P19 champion, reproduced as the reference
    B  binary, grip 55 mm      the impulse arrives 2 mm below the block's top face at 65 mm;
                               a strike near the top tips a 70 mm block that a strike nearer
                               its middle does not. The wrist rides down with the TCP
                               (`z_wrist = grip_z - 41.9 * a_z`), so 55 mm puts it at ~19 mm,
                               still above the 12 mm floor guard.
    C  binary, grip 50 mm      the same lever, as far as the floor guard allows
    D  duty-cycled close       `a[6]` alternated closed/open every control step. Fully legal
                               -- it is just a bit pattern in the action sequence, which a
                               chunked BC policy reproduces natively -- and it halves the
                               mean drive travel per unit time.
    E  ramped close            P19's arm C, kept as the upper bound. NOT legal; it is the
                               control that says how much of the gap is closable in principle.

Reporting E alongside the legal arms is deliberate. If the legal arms land far below it, the
gap is a property of the benchmark's action space and belongs in the findings, not in a
quietly ramped expert.

Usage
-----
    python eva_bc/clutter/probes/p20_legal_close.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Policy-legal replacements for the ramped close.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=12)
parser.add_argument("--tries", type=int, default=10)
parser.add_argument("--lift_dz", type=float, default=0.150)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--margin", type=float, default=0.005)
parser.add_argument("--seg", type=float, default=0.030)
parser.add_argument("--fine", type=int, default=120)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p20_legal.json")
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
DECIM = 8            # the env's control:physics ratio -- a legal action lasts this long


class Tape:
    def __init__(self, K, e, dpos0):
        self.K, self.e, self.dpos0 = K, e, dpos0
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
    dz = args_cli.lift_dz
    boxes = torch.tensor([[ROW_X, y, 0.032, HX, HY, HZ] for y in ROW_Y]
                         + [[ROW_X, 0.0, 0.032, HX, HY, HZ]], device=dev)

    print("\n" + "=" * 100)
    print("P20 -- POLICY-LEGAL REPLACEMENTS FOR THE RAMPED CLOSE")
    print("=" * 100)
    print(f"   physics dt {e.physics_dt * 1000:.2f} ms, decimation {DECIM} -> one legal "
          f"action lasts {DECIM * e.physics_dt * 1000:.0f} ms")

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

    def build(grip_z):
        grip = torch.tensor([ROW_X, 0.0, grip_z], device=dev)
        down = torch.tensor([0.0, 0.0, -1.0], device=dev)
        lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
        carry = torch.tensor([goal[0], goal[1], grip_z + dz], device=dev)
        place = torch.tensor([goal[0], goal[1], grip_z], device=dev)
        appr = [grip + STANDOFF * down * (-(3 - k) / 3.0) for k in range(3)]
        c0 = solve(grip, K.q_arm0, args_cli.tries, 0.6, args_cli.restarts)

        def dn(pts, seed):
            qs, out, qq = [], [], seed
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                k = max(1, int(math.ceil(float((b - a).norm()) / args_cli.seg)))
                for j in range(k):
                    t = a + (b - a) * (j + 1) / k
                    qq = solve(t, qq, 2, 0.08, 1)["q"]
                    qs.append(qq)
                    out.append(t)
            return [pts[0]] + out, [seed] + qs

        up_full, q_up = dn([grip, lift, carry, place], c0["q"])
        dn_full, q_dn = dn([grip, appr[2], appr[1], appr[0]], c0["q"])
        g = K.fk(c0["q"].unsqueeze(0).repeat(n, 1))
        w = g["bodies"][0, K.i_end]
        lo = float(g["bodies"][0, K.floor_bodies, 2].min())
        print(f"   grip {grip_z * 1000:.0f} mm: err {c0['pos_err'] * 1000:.2f} | o_align "
              f"{c0['o_align']:.3f} | pen {c0['pen'] * 1000:.1f} | wrist "
              f"({float(w[0]) * 1000:.0f},{float(w[1]) * 1000:+.0f},"
              f"{float(w[2]) * 1000:.0f}) | lowest body z {lo * 1000:.1f} mm")
        return {"pts": list(reversed(dn_full)) + up_full[1:],
                "qs": list(reversed(q_dn)) + q_up[1:], "i": len(dn_full) - 1,
                "grip_z": grip_z, "o_align": c0["o_align"]}

    print()
    chains = {z: build(z) for z in (0.065, 0.055, 0.050)}

    # --------------------------------------------------------------- one spawn, all arms
    env.reset()
    for _ in range(30):
        e.sim.step()
        e.scene.update(e.physics_dt)
    snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}
    tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
    dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
    o_env = target_axis(e).clone()
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

    def per_env(ch, grip_z):
        d = torch.stack([tpos0[:, 0] - ROW_X, tpos0[:, 1], torch.zeros(n, device=dev)], dim=1)
        out = []
        for i, (p, q) in enumerate(zip(ch["pts"], ch["qs"])):
            shift = d if i <= ch["i"] else torch.zeros_like(d)
            od = o_env if i <= ch["i"] + 1 else None
            out.append(K.refine(q.unsqueeze(0).repeat(n, 1), p.unsqueeze(0) + shift,
                                iters=3, o_des=od))
        return out

    def finger_schedule(mode, n_phys):
        """Per-physics-step finger target. Legal modes hold one value for a whole DECIM block."""
        if mode == "binary":
            return [Q_CLOSE] * n_phys
        if mode == "duty":                       # closed / open, alternating control steps
            out = []
            for c in range(n_phys // DECIM):
                v = Q_CLOSE if (c % 2 == 0 or c > 30) else Q_OPEN
                out += [v] * DECIM
            return out + [Q_CLOSE] * (n_phys - len(out))
        if mode == "duty3":                      # closed 1, open 2 -- a gentler duty cycle
            out = []
            for c in range(n_phys // DECIM):
                v = Q_CLOSE if (c % 3 == 0 or c > 40) else Q_OPEN
                out += [v] * DECIM
            return out + [Q_CLOSE] * (n_phys - len(out))
        if mode == "ramp":                       # NOT legal: intermediate joint targets
            return [Q_OPEN + (Q_CLOSE - Q_OPEN) * min(1.0, (s + 1) / 100.0)
                    for s in range(n_phys)]
        raise ValueError(mode)

    def execute(ch, mode, tag):
        restore()
        cq = per_env(ch, ch["grip_z"])
        i_grip = ch["i"]
        tp = Tape(K, e, dpos0)
        K.teleport_arm(cq[0], Q_OPEN)
        tp.label = "descend"
        hold_t(K, tp, cq[0], 80)
        st = max(6, int(round(75 / max(1, i_grip))))
        for i in range(i_grip):
            run_t(K, tp, cq[i], cq[i + 1], st)
        tp.label = "predwell"
        hold_t(K, tp, cq[i_grip], 160)

        # ---- the close, with a per-physics-step finger schedule
        tp.label = "close"
        sched = finger_schedule(mode, 560)
        pre_t = (_t(e.scene["target"].data.root_pos_w) - org).clone()
        fine_t, fine_d, fine_g = [], [], []
        for s, qf in enumerate(sched):
            K.robot.set_joint_position_target(K._drive(cq[i_grip], qf))
            K.robot.write_data_to_sim()
            K.e.sim.step()
            K.e.scene.update(K.e.physics_dt)
            if s < args_cli.fine:
                o = e.scene.env_origins
                fine_g.append(K.gap().clone())
                fine_t.append((_t(e.scene["target"].data.root_pos_w) - o).clone())
                fine_d.append(torch.stack(
                    [(_t(e.scene[d].data.root_pos_w) - o) for d in DIST], dim=1).clone())
            if (s + 1) % 8 == 0:
                tp.sample()
        gap_stall = K.gap().clone()

        st = max(6, int(round(90 / max(1, len(cq) - 1 - i_grip))))
        tp.label = "carry"
        for i in range(i_grip, len(cq) - 1):
            run_t(K, tp, cq[i], cq[i + 1], st, q_fing=Q_CLOSE)
        tp.label = "dwell"
        hold_t(K, tp, cq[-1], 160, q_fing=Q_CLOSE)
        tp.label = "release"
        hold_t(K, tp, cq[-1], 240, q_fing=Q_OPEN)
        tp.label = "retreat"
        run_t(K, tp, cq[-1], cq[-2], 25, q_fing=Q_OPEN)
        tp.label = "final"
        hold_t(K, tp, cq[-2], 240, q_fing=Q_OPEN)

        bpos = _t(e.scene["target"].data.root_pos_w) - org
        up_end = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
        topp = (up_end < mdp_cl.TOPPLE_DOT).any(dim=1)
        held = (gap_stall - width).abs() < 0.012
        at_goal = (((bpos[:, :2] - goal).norm(dim=1) < mdp_cl.GOAL_RADIUS)
                   & (bpos[:, 2] < 0.055))
        succ = at_goal & ~topp

        UP, DS = torch.stack(tp.up), torch.stack(tp.disp)
        ph, T = tp.phase, UP.shape[0]
        hitd = (DS > DISP_ON) | (UP < TILT_ON)
        hit = hitd.any(dim=2)
        ever = hit.any(dim=0)
        idx = torch.where(ever, hit.float().argmax(dim=0),
                          torch.full((n,), -1, device=dev, dtype=torch.long))
        print(f"\n   {tag}")
        print(f"      enclosed {float(held.float().mean()):6.1%} | at goal "
              f"{float(at_goal.float().mean()):6.1%} | topple "
              f"{float(topp.float().mean()):6.1%} | SUCCESS "
              f"{float(succ.float().mean()):6.1%}")
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
                    print(f"      hazard {p:>9}: {k:3d}/{r:3d} = {k / r:6.1%}")
            clean = clean & ~inp

        # ---- ordering: does the TARGET move before its neighbour does?
        G, TT = torch.stack(fine_g), torch.stack(fine_t)
        DD = torch.stack(fine_d)
        dt_ = (TT[:, :, :2] - pre_t[:, :2].unsqueeze(0)).norm(dim=2)
        dn_ = (DD[:, :, :, :2] - dpos0[:, :, :2].unsqueeze(0)).norm(dim=3).amax(dim=2)
        mt, mn = (dt_ > 0.001), (dn_ > DISP_ON)
        t_t = torch.where(mt.any(0), mt.float().argmax(0), torch.full((n,), -1, device=dev,
                                                                     dtype=torch.long))
        t_n = torch.where(mn.any(0), mn.float().argmax(0), torch.full((n,), -1, device=dev,
                                                                     dtype=torch.long))
        both = (t_t >= 0) & (t_n >= 0)
        print(f"      close trace  : jaw 89->60 mm by step "
              f"{float((G < 0.060).float().argmax(dim=0).median()):.0f}, stall "
              f"{float(gap_stall.median()) * 1000:.1f} mm | target moves 1 mm at step "
              f"{float(t_t[t_t >= 0].float().median()) if int((t_t >= 0).sum()) else -1:.0f}"
              f" | neighbour at step "
              f"{float(t_n[t_n >= 0].float().median()) if int((t_n >= 0).sum()) else -1:.0f}")
        if int(both.sum()):
            print(f"                     target moved FIRST in "
                  f"{float((t_t[both] <= t_n[both]).float().mean()):.0%} of the "
                  f"{int(both.sum())} envs where both moved | peak target excursion "
                  f"{float(dt_.amax(0).median()) * 1000:.2f} mm")
        return succ, topp, haz

    ARMS = [("A grip65 binary   (legal)", 0.065, "binary"),
            ("B grip55 binary   (legal)", 0.055, "binary"),
            ("C grip50 binary   (legal)", 0.050, "binary"),
            ("D grip65 duty 1:1 (legal)", 0.065, "duty"),
            ("E grip65 duty 1:2 (legal)", 0.065, "duty3"),
            ("F grip65 ramped   (ILLEGAL, upper bound)", 0.065, "ramp")]
    res, masks = [], {}
    for tag, gz, mode in ARMS:
        s, t, h = execute(chains[gz], mode, tag)
        masks[tag] = s
        res.append({"arm": tag, "grip_z": gz, "mode": mode, "legal": mode != "ramp",
                    "success": float(s.float().mean()), "topple": float(t.float().mean()),
                    "hazard": h, "succ_mask": s.tolist()})

    print("\n" + "=" * 100)
    print(f"   {'arm':>42} | {'success':>8} | {'topple':>8} | {'close haz':>10} | net vs A")
    base = masks[ARMS[0][0]]
    for r in res:
        m = torch.tensor(r["succ_mask"], device=dev)
        print(f"   {r['arm']:>42} | {r['success']:7.1%} | {r['topple']:7.1%} | "
              f"{r['hazard'].get('close', 0):9.1%} | "
              f"{int((~base & m).sum()) - int((base & ~m).sum()):+d}")
    legal = [r for r in res if r["legal"]]
    bl = max(legal, key=lambda r: r["success"])
    up = max(res, key=lambda r: r["success"])
    print(f"\n   best LEGAL arm: {bl['arm'].strip()} at {bl['success']:.1%}")
    print(f"   upper bound   : {up['arm'].strip()} at {up['success']:.1%}")
    if up["success"] - bl["success"] > 0.10:
        print("   -> The residual gap is a property of the BINARY gripper, not of the plan.")
        print("      Report it as an env finding; do not ship an expert that ramps.")

    out = {"n": n, "min_gap_mm": (min_gap * 1000).tolist(),
           "o_align": {str(z): chains[z]["o_align"] for z in chains}, "arms": res}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p20] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
