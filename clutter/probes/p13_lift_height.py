# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P13 -- Paired A/B harness for trajectory variants. First question: how high to lift?

What P12 established
--------------------
Phase-resolved forensics on 128 random spawns put **48.4 % of all topple onsets inside the
`carry` phase** -- more than the descent, the close and the place put together -- and the
victims were `distractor_0` (y = -84 mm) and `distractor_1` (y = -42 mm) in every single
case. Those are precisely the two blocks between the row and the goal at (185, -185) mm.
The arm is not fumbling the approach; it is **sweeping the row with the block it is already
holding**, on the way out.

The arithmetic says the same thing. The block is gripped at z = 65 mm while its centre sits
at 32 mm, so it hangs 68 mm below the TCP. A 75 mm lift puts the TCP at 140 mm and the
carried block's bottom face at **72 mm, against distractor tops at 67 mm** -- five
millimetres, across a joint-space interpolation that does not travel in a Cartesian straight
line and is free to sag. The grip height chosen to clear the neighbours on the way *in* is
the same choice that makes the exit skim them.

Why this has to be a PAIRED experiment
--------------------------------------
Three runs of the *same code* have now produced 0 %, 25 % and 37.5 % success. The variance
comes from two sources that a single-arm comparison cannot separate from a real effect:

  * **spawn draw** -- the reset jitters the target and all four distractors independently;
  * **pose draw**  -- the CEM is stochastic, and `pos_err < 1.5 mm` plus `o_align > 0.99`
    still leaves the elbow, forearm and wrist free. P12's two runs both cleared those gates
    (0.83 mm/0.998 and 0.98 mm/0.996) and scored 0 % and 37.5 %.

So this harness holds **both** fixed: one reset, one settle, snapshot every block's root
state, then run each variant from the identical spawn with the identical drawn pose,
restoring the snapshot in between. Differences between variants are then attributable to
the variant. The absolute level still carries pose-draw uncertainty, and `--pose_reps`
exists to bound that once the ordering is known.

This is deliberately built as a reusable harness, not a one-off: grip height, approach
speed, yaw matching and `phi` all get asked the same way later.

Usage
-----
    python eva_bc/clutter/probes/p13_lift_height.py --num_envs 128
    python eva_bc/clutter/probes/p13_lift_height.py --num_envs 128 --pose_reps 3
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Paired variant sweep: lift height.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--restarts", type=int, default=10)
parser.add_argument("--grip_z", type=float, default=0.065)
parser.add_argument("--phi", type=float, default=90.0)
parser.add_argument("--pose_reps", type=int, default=1,
                    help="independent CEM pose draws; every variant is run under each")
parser.add_argument("--lifts", type=str, default="0.075,0.110,0.150,0.200",
                    help="comma-separated lift heights above the grip point [m]")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p13_lift.json")
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

#: onset detectors. A push registers in planar displacement long before it registers as
#: tilt -- P12 used `up_z < 0.98` and found the victim had ALREADY moved 9.3 mm by then,
#: which is why its "nearest body" attribution named bodies 50-65 mm away. Displacement is
#: the earlier and therefore the more honest signal.
DISP_ON = 0.0015
TILT_ON = 0.999

PHASES = ["descend", "predwell", "close", "lift", "carry", "place",
          "dwell", "release", "retreat", "final"]


class Tape:
    """Compact per-control-step recorder: enough to attribute a topple, nothing more."""

    def __init__(self, e, dev, n, dpos0):
        self.e, self.dev, self.n, self.dpos0 = e, dev, n, dpos0
        self.up, self.disp, self.phase = [], [], []
        self.label = "descend"

    def sample(self):
        org = self.e.scene.env_origins
        up = torch.stack([mdp_cl._up_z(self.e, d) for d in DIST], dim=1)          # (n,4)
        dp = torch.stack([(_t(self.e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1)
        self.up.append(up)
        self.disp.append((dp[:, :, :2] - self.dpos0[:, :, :2]).norm(dim=2))       # (n,4)
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
    lifts = [float(x) for x in args_cli.lifts.split(",")]

    _p = math.radians(args_cli.phi)
    Y = torch.tensor([math.sin(_p), math.cos(_p), 0.0], device=dev)
    width = 2 * HX if args_cli.phi >= 45.0 else 2 * HY

    print("\n" + "=" * 100)
    print("P13 -- PAIRED VARIANT SWEEP: LIFT HEIGHT")
    print("=" * 100)
    print(f"   grip_z {args_cli.grip_z * 1000:.0f} mm | block hangs "
          f"{(args_cli.grip_z - 0.032 + HZ) * 1000:.0f} mm below the TCP | "
          f"distractor tops ~67 mm")
    print(f"   lifts tested: {['%.0f mm' % (x * 1000) for x in lifts]}")

    def best_pose(pos, seed, tries=6, std0=0.6, restarts=None, pos_max=0.0015):
        best = None
        for _ in range(tries):
            c = K.cem(pos, seed, o_des=Y, a_des=None, w_o=0.60, iters=args_cli.iters,
                      std0=std0, restarts=restarts or args_cli.restarts)
            if c["pos_err"] > pos_max:
                continue
            if best is None or c["o_align"] > best["o_align"]:
                best = c
        return best or c

    # ---------------- one reset, one settle, one snapshot: every variant sees this spawn
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
    print(f"   spawn: min free gap {float(min_gap.min()) * 1000:.1f}.."
          f"{float(min_gap.max()) * 1000:.1f} mm (median {float(min_gap.median()) * 1000:.1f})")

    def restore():
        for k, v in snap.items():
            e.scene[k].write_root_state_to_sim(v.clone())
        e.sim.forward()
        e.scene.update(e.physics_dt)

    results = []
    for prep in range(args_cli.pose_reps):
        # ------------ one pose draw, shared by every variant in this rep
        grip = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
        r = best_pose(grip, K.q_arm0)
        q_grip = r["q"]
        print(f"\n   -- pose draw {prep}: CEM err {r['pos_err'] * 1000:.2f} mm, "
              f"o_align {r['o_align']:.3f}, a_hat ({r['a_hat'][0]:+.2f},"
              f"{r['a_hat'][1]:+.2f},{r['a_hat'][2]:+.2f})")

        down = torch.tensor([0.0, 0.0, -1.0], device=dev)
        qq, appr = q_grip, []
        for t in lerp_pts(grip, grip - STANDOFF * down, 3):
            qq = best_pose(t, qq, tries=3, std0=0.15)["q"]
            appr.append(qq)
        approach_nom = list(reversed(appr))

        tgt = torch.stack([tpos0[:, 0], tpos0[:, 1],
                           torch.full((n,), args_cli.grip_z, device=dev)], dim=1)
        q_g = K.refine(q_grip.unsqueeze(0).repeat(n, 1), tgt, iters=4)
        q_app = [K.refine(q.unsqueeze(0).repeat(n, 1),
                          tgt + torch.tensor([0.0, 0.0, d], device=dev), iters=3)
                 for q, d in zip(approach_nom, (STANDOFF, 2 * STANDOFF / 3, STANDOFF / 3))]

        for dz in lifts:
            # ---- variant-specific carry chain, solved BEFORE anything closes
            lift = grip + torch.tensor([0.0, 0.0, dz], device=dev)
            carry = torch.tensor([goal[0], goal[1], args_cli.grip_z + dz], device=dev)
            place = torch.tensor([goal[0], goal[1], args_cli.grip_z], device=dev)
            qq, post, errs = q_grip, [], []
            for t in [lift, carry, place]:
                c = best_pose(t, qq, tries=3, std0=0.30, restarts=3)
                qq = c["q"]
                post.append(qq)
                errs.append(c["pos_err"])
            g_lift = K.fk(post[0].unsqueeze(0).repeat(n, 1))
            tcp_lift = float(g_lift["tcp"][0, 2])
            hang = args_cli.grip_z - 0.032 + HZ
            clr = (tcp_lift - hang) - (0.032 + HZ)

            # ---- execute from the identical spawn
            restore()
            tp = Tape(e, dev, n, dpos0)
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

            # ---- score exactly as the env would
            bpos = _t(e.scene["target"].data.root_pos_w) - org
            up_end = torch.stack([mdp_cl._up_z(e, d) for d in DIST], dim=1)
            topp = (up_end < mdp_cl.TOPPLE_DOT).any(dim=1)
            held = (gap_stall - width).abs() < 0.012
            at_goal = (((bpos[:, :2] - goal).norm(dim=1) < mdp_cl.GOAL_RADIUS)
                       & (bpos[:, 2] < 0.055))
            succ = at_goal & ~topp

            # ---- attribute: first contact, by phase and by victim
            UP = torch.stack(tp.up)                       # (T,n,4)
            DS = torch.stack(tp.disp)                     # (T,n,4)
            ph = tp.phase
            T = UP.shape[0]
            hit = (DS > DISP_ON) | (UP < TILT_ON)          # (T,n,4) per-distractor contact
            anyhit = hit.any(dim=2)                        # (T,n)
            ever = anyhit.any(dim=0)
            idx = torch.where(ever, anyhit.float().argmax(dim=0),
                              torch.full((n,), -1, device=dev, dtype=torch.long))
            # victim = first distractor flagged at the onset step
            ar = torch.arange(n, device=dev)
            victim = hit[idx.clamp(min=0), ar].float().argmax(dim=1)

            # topple onset (the terminal event) separately -- contact is not always fatal
            bad = (UP < mdp_cl.TOPPLE_DOT).any(dim=2)
            ever_t = bad.any(dim=0)
            idx_t = torch.where(ever_t, bad.float().argmax(dim=0),
                                torch.full((n,), -1, device=dev, dtype=torch.long))

            def by_phase(ix, ok):
                out = {}
                for p in PHASES:
                    st = [t for t in range(T) if ph[t] == p]
                    if not st:
                        continue
                    out[p] = int((ok & (ix >= st[0]) & (ix <= st[-1])).sum())
                return out

            hp = by_phase(idx_t, ever_t)
            hc = by_phase(idx, ever)
            vic = [int((ever & (victim == k)).sum()) for k in range(4)]

            print(f"\n   lift {dz * 1000:5.0f} mm | TCP {tcp_lift * 1000:6.1f} mm | carried "
                  f"bottom {(tcp_lift - hang) * 1000:6.1f} mm | clearance over row "
                  f"{clr * 1000:+6.1f} mm | chain CEM err "
                  f"{max(errs) * 1000:.1f} mm")
            print(f"        enclosed {float(held.float().mean()):6.1%} | at goal "
                  f"{float(at_goal.float().mean()):6.1%} | topple "
                  f"{float(topp.float().mean()):6.1%} | SUCCESS "
                  f"{float(succ.float().mean()):6.1%}")
            print(f"        topple onset by phase : "
                  + ", ".join(f"{k} {v}" for k, v in hp.items() if v))
            print(f"        contact onset by phase: "
                  + ", ".join(f"{k} {v}" for k, v in hc.items() if v))
            print(f"        first victim          : "
                  + ", ".join(f"d{k}(y={ROW_Y[k] * 1000:+.0f}) {vic[k]}" for k in range(4)
                              if vic[k]))

            results.append({
                "pose_rep": prep, "lift_dz": dz, "tcp_lift_mm": tcp_lift * 1000,
                "row_clearance_mm": clr * 1000, "chain_err_mm": max(errs) * 1000,
                "o_align": r["o_align"], "pose_err_mm": r["pos_err"] * 1000,
                "encl": float(held.float().mean()),
                "at_goal": float(at_goal.float().mean()),
                "topple": float(topp.float().mean()),
                "success": float(succ.float().mean()),
                "topple_by_phase": hp, "contact_by_phase": hc, "victim": vic,
                "succ_mask": succ.tolist(), "topple_mask": topp.tolist(),
            })

    # ---------------- paired comparison against the first lift in the list
    print("\n" + "=" * 100)
    print("PAIRED COMPARISON (same spawn, same pose draw, per pose_rep)")
    print("=" * 100)
    print(f"   {'variant':>12} | {'success':>8} | {'topple':>8} | {'fixed':>6} | "
          f"{'broke':>6} | net")
    for prep in range(args_cli.pose_reps):
        rs = [x for x in results if x["pose_rep"] == prep]
        base = rs[0]
        bs = torch.tensor(base["succ_mask"], device=dev)
        for x in rs:
            xs = torch.tensor(x["succ_mask"], device=dev)
            fixed = int((~bs & xs).sum())
            broke = int((bs & ~xs).sum())
            tag = f"lift{x['lift_dz'] * 1000:.0f}"
            print(f"   {tag:>12} | {x['success']:7.1%} | {x['topple']:7.1%} | "
                  f"{fixed:6d} | {broke:6d} | {fixed - broke:+d}")

    out = {"n": n, "grip_z": args_cli.grip_z, "phi": args_cli.phi,
           "min_gap_mm": (min_gap * 1000).tolist(), "results": results}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p13] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
