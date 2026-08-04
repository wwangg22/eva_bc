# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P19 -- Inside the closing slam: what moves first, and can the jolt be removed?

What P18 proved
---------------
Mirroring the wrist stub from the -y gap to the +y gap flipped the victim distribution from
`d1` 51 / `d2` 31 to `d1` 28 / `d2` **100**. The wrist is the culprit, established causally
rather than by geometry. Two further measurements came out of the same run:

  * **Contact fires at 0 % of the squeeze.** The close phase was sampled every 8 physics
    steps and the very first sample already shows the finger gap at its stall value and the
    distractor already displaced. Whatever happens, happens inside ~66 ms of commanding the
    close. This is the *slam*, not the grip.
  * **The wrist cannot be moved.** The mirror pose exists but is far worse (`o_align` 0.885
    against 0.995, at-goal 21 % against 98 %), because with `o_hat = x_hat` the approach axis
    is confined to the y-z plane and `a_x ~ 0`, which pins `gripper_end` to
    `x ~ 250 mm` -- inside the row's 232-268 mm band -- and forces it to thread one of the
    12 mm gaps. Only a badly misaligned jaw buys `|a_x| > 0.43` and lifts the wrist clear in
    x, and that jaw cannot hold the block.

So the wrist's position is not negotiable at `phi = 90`. What might be negotiable is the
**jolt**: the wrist sits ~5 mm from `d1`'s face, and the question is whether it is already
touching or whether the closing transient drives it there.

Four arms, one spawn
--------------------
    A  yaw-matched jaw, binary close      the P17 champion (62.5 %)
    B  square jaw,      binary close      P16 measured a LOWER close hazard without yaw
                                          matching (51 % vs 73 %); P17 never tested it
    C  yaw-matched jaw, ramped close      finger target driven 45 -> 0 mm over 100 steps
                                          instead of commanded shut in one step
    D  yaw-matched jaw, binary close,     the keep-out boxes inflated in y by the distractors'
       jitter-aware keep-out              own +/-5 mm spawn jitter. The nominal boxes assume
                                          d1 sits at y = -42; the reset can put it at -37,
                                          which eats the wrist's entire margin.

C is diagnostic first and a candidate fix second. The env's gripper is
`BinaryJointPositionActionCfg` -- there is no intermediate aperture and no rate limit a
policy can request -- so if only the ramp works, that is a **property of the benchmark to
report**, not something to paper over. Recording it either way is the point.

The high-resolution trace
-------------------------
During the first `--fine` physics steps of the close, every single step is sampled: finger
gap, wrist position, target position, and all four distractor positions. The ordering is
what matters -- if the wrist lurches toward `d1` before `d1` moves, the transient is the
cause; if `d1` moves first, the wrist was already in contact and the pose is the cause.

Usage
-----
    python eva_bc/clutter/probes/p19_close_dynamics.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Close-phase dynamics at physics-step resolution.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=12)
parser.add_argument("--tries", type=int, default=10)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--lift_dz", type=float, default=0.150)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--margin", type=float, default=0.005)
parser.add_argument("--seg", type=float, default=0.030)
parser.add_argument("--fine", type=int, default=120, help="physics steps traced at full rate")
parser.add_argument("--ramp", type=int, default=100, help="steps for the ramped close")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p19_close_dyn.json")
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
    grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    dz = args_cli.lift_dz

    def make_boxes(dy: float):
        """Keep-out set; `dy` inflates the distractors by their own y spawn jitter."""
        return torch.tensor([[ROW_X, y, 0.032, HX, HY + dy, HZ] for y in ROW_Y]
                            + [[ROW_X, 0.0, 0.032, HX, HY, HZ]], device=dev)

    print("\n" + "=" * 100)
    print("P19 -- CLOSING-SLAM DYNAMICS")
    print("=" * 100)
    print(f"   physics dt {e.physics_dt * 1000:.2f} ms | {args_cli.fine} fine steps = "
          f"{args_cli.fine * e.physics_dt * 1000:.0f} ms of the close traced per step")

    def make_solver(boxes):
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
        return solve

    down = torch.tensor([0.0, 0.0, -1.0], device=dev)
    lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
    carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + dz], device=dev)
    place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
    approach_pts = [grip + STANDOFF * down * (-(3 - k) / 3.0) for k in range(3)]

    def build(dy):
        solve = make_solver(make_boxes(dy))
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
        dn_full, q_dn = dn([grip, approach_pts[2], approach_pts[1], approach_pts[0]], c0["q"])
        w = K.fk(c0["q"].unsqueeze(0).repeat(n, 1))["bodies"][0, K.i_end]
        print(f"   chain(dy={dy * 1000:.0f} mm): err {c0['pos_err'] * 1000:.2f} mm | o_align "
              f"{c0['o_align']:.3f} | pen {c0['pen'] * 1000:.1f} mm | wrist "
              f"({float(w[0]) * 1000:.0f},{float(w[1]) * 1000:+.0f},"
              f"{float(w[2]) * 1000:.0f}) mm")
        return (list(reversed(dn_full)) + up_full[1:],
                list(reversed(q_dn)) + q_up[1:], len(dn_full) - 1)

    chain_std = build(0.000)
    chain_jit = build(0.005)

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
    gap_m = (tpos0[:, 1] - HY) - (dpos0[:, 1, 1] + HY)
    print(f"\n   spawn: min free gap {float(min_gap.median()) * 1000:.1f} mm median | "
          f"-y gap {float(gap_m.min()) * 1000:.1f}..{float(gap_m.max()) * 1000:.1f} mm")

    def restore():
        for k, v in snap.items():
            e.scene[k].write_root_state_to_sim(v.clone())
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def per_env(pts, qs, i_grip, yaw):
        d = torch.stack([tpos0[:, 0] - ROW_X, tpos0[:, 1], torch.zeros(n, device=dev)], dim=1)
        out = []
        for i, (p, q) in enumerate(zip(pts, qs)):
            shift = d if i <= i_grip else torch.zeros_like(d)
            od = o_env if (yaw and i <= i_grip + 1) else None
            out.append(K.refine(q.unsqueeze(0).repeat(n, 1), p.unsqueeze(0) + shift,
                                iters=3, o_des=od))
        return out

    def close_fine(q, ramp):
        """Trace the first `fine` physics steps of the close, one sample per step."""
        rec = {"gap": [], "wrist": [], "tgt": [], "dst": []}
        for s in range(args_cli.fine):
            f = min(1.0, (s + 1) / ramp) if ramp else 1.0
            qf = Q_OPEN + (Q_CLOSE - Q_OPEN) * f
            qd = K._drive(q, qf)
            K.robot.set_joint_position_target(qd)
            K.robot.write_data_to_sim()
            K.e.sim.step()
            K.e.scene.update(K.e.physics_dt)
            o = e.scene.env_origins
            rec["gap"].append(K.gap().clone())
            rec["wrist"].append((_t(K.robot.data.body_pos_w)[:, K.i_end] - o).clone())
            rec["tgt"].append((_t(e.scene["target"].data.root_pos_w) - o).clone())
            rec["dst"].append(torch.stack(
                [(_t(e.scene[d].data.root_pos_w) - o) for d in DIST], dim=1).clone())
        return rec

    def execute(chain, i_grip, yaw, ramp, tag):
        restore()
        cq = per_env(chain[0], chain[1], i_grip, yaw)
        tp = Tape(K, e, dpos0)
        K.teleport_arm(cq[0], Q_OPEN)
        tp.label = "descend"
        hold_t(K, tp, cq[0], 80)
        st = max(6, int(round(75 / max(1, i_grip))))
        for i in range(i_grip):
            run_t(K, tp, cq[i], cq[i + 1], st)
        tp.label = "predwell"
        hold_t(K, tp, cq[i_grip], 160)

        # ---- the close, traced at full rate for the first `fine` steps
        tp.label = "close"
        pre_w = (_t(K.robot.data.body_pos_w)[:, K.i_end] - org).clone()
        pre_d = dpos0.clone()
        rec = close_fine(cq[i_grip], ramp if ramp else 0)
        tp.sample()
        hold_t(K, tp, cq[i_grip], max(0, 560 - args_cli.fine), q_fing=Q_CLOSE)
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

        # ---- the fine trace: ordering of events
        G = torch.stack(rec["gap"])                      # (F,n)
        W = torch.stack(rec["wrist"])                    # (F,n,3)
        D = torch.stack(rec["dst"])                      # (F,n,4,3)
        dw = (W - pre_w.unsqueeze(0)).norm(dim=2)        # wrist travel
        dd1 = (D[:, :, 1, :2] - pre_d[:, 1, :2].unsqueeze(0)).norm(dim=2)
        t_gap = (G < 0.060).float().argmax(dim=0)        # step the jaw first bites
        t_w = (dw > 0.0005).float().argmax(dim=0)        # wrist moved 0.5 mm
        t_d = (dd1 > DISP_ON).float().argmax(dim=0)      # d1 moved 1.5 mm
        moved = (dd1 > DISP_ON).any(dim=0)
        print(f"      close trace  : jaw reaches 60 mm at step "
              f"{float(t_gap.float().median()):.0f} | wrist first moves 0.5 mm at step "
              f"{float(t_w.float().median()):.0f} | d1 first moves at step "
              f"{float(t_d[moved].float().median()) if int(moved.sum()) else float('nan'):.0f}"
              f" ({int(moved.sum())}/{n} envs)")
        print(f"      wrist travel during the traced window: median "
              f"{float(dw[-1].median()) * 1000:.2f} mm, max "
              f"{float(dw[-1].max()) * 1000:.2f} mm")
        return succ, topp, haz, {
            "t_gap": float(t_gap.float().median()), "t_w": float(t_w.float().median()),
            "t_d": float(t_d[moved].float().median()) if int(moved.sum()) else -1,
            "n_moved": int(moved.sum()),
            "wrist_travel_mm": float(dw[-1].median()) * 1000}

    ARMS = [("A yaw + binary close", chain_std, True, 0),
            ("B square + binary close", chain_std, False, 0),
            ("C yaw + ramped close", chain_std, True, args_cli.ramp),
            ("D yaw + jitter-aware keepout", chain_jit, True, 0)]
    res, masks = [], {}
    for tag, ch, yaw, ramp in ARMS:
        s, t, h, fine = execute((ch[0], ch[1]), ch[2], yaw, ramp, tag)
        masks[tag] = s
        res.append({"arm": tag, "success": float(s.float().mean()),
                    "topple": float(t.float().mean()), "hazard": h, "fine": fine,
                    "succ_mask": s.tolist()})

    print("\n" + "=" * 100)
    print(f"   {'arm':>30} | {'success':>8} | {'topple':>8} | {'close hazard':>13} | net vs A")
    base = masks["A yaw + binary close"]
    for r in res:
        m = torch.tensor(r["succ_mask"], device=dev)
        print(f"   {r['arm']:>30} | {r['success']:7.1%} | {r['topple']:7.1%} | "
              f"{r['hazard'].get('close', 0):12.1%} | "
              f"{int((~base & m).sum()) - int((base & ~m).sum()):+d}")

    out = {"n": n, "grip_z": args_cli.grip_z, "fine": args_cli.fine, "ramp": args_cli.ramp,
           "min_gap_mm": (min_gap * 1000).tolist(), "arms": res}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p19] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
