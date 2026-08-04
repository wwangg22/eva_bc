# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P29 -- The `home -> chain[0]` approach: the one segment that has never existed.

The gap this closes
-------------------
`ClutterExpert.run_physics` opens with

    K.teleport_arm(chain[0], Q_OPEN)

`chain[0]` is the top of the approach, TCP at about (250, 0, 125) mm. `env.reset()` leaves the
arm at `_START_POSE`, TCP near (350, 0, 170) mm. **The motion between those two poses has never
been planned, never been executed and never been measured** -- every Stage-0 and Stage-1 number
was collected downstream of a teleport.

It cannot be skipped and it cannot stay a teleport. A demo has to begin where `env.reset()`
leaves the arm; a demo that begins mid-air teaches a policy nothing about its own first step,
which is the worst possible place for covariate shift. So Stage 2 begins by planning and
verifying a new trajectory segment, and it gets the treatment P17 established: audit the
**segment**, not just its endpoints.

What is measured
----------------
Kinematically, for each candidate approach:
  * whole-arm keep-out penetration at every interpolation step (`box_penetration`)
  * lowest body-origin z (the table proxy)
  * TCP deviation from the straight line between the endpoints -- reported, NOT a pass
    criterion. Above the row a curved path is harmless, and P14's most expensive error was
    scoring a pose on a proxy that omitted what actually mattered.

Physically, paired on one spawn (snapshot / restore between arms):
  * per-phase hazard rate with `approach` as a new first phase
  * end-to-end enclosure / at-goal / topple / success against the teleport baseline

Candidates
----------
    T    teleport to chain[0]                    the Stage-1 baseline, for pairing
    A20  joint-space lerp, 20 env steps          ~5.5 mm/step
    A40  joint-space lerp, 40 env steps          ~2.8 mm/step
    B40  dense Cartesian, 40 env steps           through the same solver as every other segment

Pre-registered prediction
-------------------------
**The joint-space lerp is clean and is the one we keep.** The entire path lies above
z = 125 mm while the row tops out at 67 mm, so no interpolation excursion can reach the
blocks; the dense Cartesian solve exists as the fallback, not the default.
**Predicted: penetration 0.0 mm, approach hazard 0 %, and A40 within +/-5 points of T.**

Falsifier: any candidate showing a non-zero `approach` hazard, or A/B differing from T by more
than 5 points. If the *lerp* is dirty and the *Cartesian* one is clean, the arm swings out on
its way in and the expert gains a real waypoint.

**Already falsified before this probe ran.** `act/collect_demos.py`'s first smoke run
(32 envs, one spawn batch, 2026-08-03) executed a 60-step joint-space lerp and got
**9.4 % success against 65.6 % for the teleport, with 90.6 % of episodes terminating early**
and an FK audit reporting **7.3 mm of keep-out penetration**. The prediction's premise -- "the
entire path lies above z = 125 mm" -- was an assumption about the TCP, and the TCP is not the
arm. It is the same error P17 cost a GPU day for: independently specified endpoints, a
joint-space line between them, and a Cartesian excursion nobody bounded. This probe now runs
to say *which candidate ships* and to give the mechanism, not to test a prediction that the
collector has already refuted.

Run 1 of this probe therefore uses the FROZEN pose and chain (`expert/pose_p33.json`), not a
fresh screened draw: a re-solved pose would move the endpoint being approached, and the
12-point pose sd would swamp a 5-point pass band.

Also answered here (`09_STAGE2_BC_PLAN.md` N7)
----------------------------------------------
The action magnitude over the whole adapted chain, `a = (q - q_default) / 0.5`. The goal sits
at azimuth -45 deg = -0.785 rad and `joint1` starts at 0, so the carry should need
`a[0] ~ -1.57` -- **outside the [-1, 1] box that `clip_actions = 1.0` gives an rl_games agent.**
Harmless for BC (`eval_act.py` applies no clamping); a hard wall for a from-scratch PPO
baseline. Measured here rather than asserted.

Usage
-----
    python -u eva_bc/clutter/probes/p29_approach_segment.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plan and verify the home -> chain[0] approach.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--pose", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/expert/pose_p33.json",
                    help="frozen pose+chain; '' re-solves one (Stage-1 behaviour, NOT paired)")
parser.add_argument("--screen", type=int, default=4, help="only used when --pose is empty")
parser.add_argument("--reps", type=int, default=2, help="spawn batches per arm")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p29_approach.json")
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
from clutter_expert import DIST, ClutterExpert, _hold, _move  # noqa: E402

ORDER = ["approach", "settle", "descend", "predwell", "close", "carry", "dwell",
         "release", "withdraw", "final"]
DISP_ON = 0.0015
TILT_ON = 0.999


class Tape:
    def __init__(self, e, dpos0):
        self.e, self.dpos0 = e, dpos0
        self.up, self.disp, self.phase = [], [], []
        self.label = "approach"

    def sample(self):
        org = self.e.scene.env_origins
        up = torch.stack([mdp_cl._up_z(self.e, d) for d in DIST], dim=1)
        dp = torch.stack([(_t(self.e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1)
        self.up.append(up)
        self.disp.append((dp[:, :, :2] - self.dpos0[:, :, :2]).norm(dim=2))
        self.phase.append(self.label)


def hazards(tape, n, dev):
    UP, DS = torch.stack(tape.up), torch.stack(tape.disp)
    ph, T = tape.phase, UP.shape[0]
    hit = ((DS > DISP_ON) | (UP < TILT_ON)).any(dim=2)
    ever = hit.any(dim=0)
    idx = torch.where(ever, hit.float().argmax(dim=0),
                      torch.full((n,), -1, device=dev, dtype=torch.long))
    clean, out = torch.ones(n, dtype=torch.bool, device=dev), {}
    for p in ORDER:
        s = [t for t in range(T) if ph[t] == p]
        if not s:
            continue
        inp = ever & (idx >= s[0]) & (idx <= s[-1])
        r = int(clean.sum())
        if r and int(inp.sum()):
            out[p] = int(inp.sum()) / r
        clean = clean & ~inp
    return out


BOX_NAMES = ["distractor_0", "distractor_1", "distractor_2", "distractor_3", "target"]


def audit(K, qs, boxes, margin, a, b, body_names=None):
    """Kinematic audit of a joint-space path: penetration, floor, TCP line deviation [mm].

    Also returns WHERE the worst penetration is -- which body, which keep-out box, how far
    along the path, and the TCP there. "The path is dirty" is not a mechanism and cannot be
    fixed; "link4 enters distractor_1 at 45 % of the way in, with the TCP 90 mm off the
    straight line" is. The inner loop repeats `box_penetration`'s arithmetic rather than
    calling it because that function takes a max and throws the indices away.
    """
    pen, low, dev, worst = 0.0, 1e9, 0.0, None
    ab = b - a
    L2 = max(float(ab @ ab), 1e-12)
    c, h = boxes[:, :3], boxes[:, 3:] + margin
    for i, q in enumerate(qs):
        g = K.fk(q if q.dim() == 2 else q.unsqueeze(0).repeat(K.n, 1))
        bod = g["bodies"][0]                                    # (B, 3)
        pmat = (-(((bod.unsqueeze(1) - c).abs() - h).amax(dim=2))).clamp(min=0.0)   # (B, M)
        v = float(pmat.max())
        p = g["tcp"][0]
        t = float(((p - a) @ ab) / L2)
        d_line = float((p - (a + max(0.0, min(1.0, t)) * ab)).norm())
        if v > pen:
            pen = v
            j = int(pmat.argmax())
            worst = {"step": i, "frac": round(i / max(1, len(qs) - 1), 3),
                     "body": body_names[j // pmat.shape[1]] if body_names else j // pmat.shape[1],
                     "box": BOX_NAMES[j % pmat.shape[1]], "pen_mm": round(v * 1000, 2),
                     "tcp_mm": [round(float(x) * 1000, 1) for x in p],
                     "line_dev_mm": round(d_line * 1000, 1)}
        low = min(low, float(g["low_z"][0]))
        dev = max(dev, d_line)
    return pen * 1000.0, low * 1000.0, dev * 1000.0, worst


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    for _ in range(30):
        e.sim.step()
        e.scene.update(e.physics_dt)
    dev, n = e.device, e.num_envs

    print("\n" + "=" * 100)
    print("P29 -- THE home -> chain[0] APPROACH SEGMENT")
    print("=" * 100)
    print("   PREDICTION (registered): joint-space lerp is clean; penetration 0.0 mm,")
    print("   approach hazard 0 %, A40 within +/-5 points of the teleport baseline.")

    if args_cli.pose:
        spec = json.load(open(args_cli.pose))
        print(f"   frozen pose {spec['name']} ({spec['source']})")
        ex = ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"],
                           chain=spec.get("chain"), verbose=True)
    else:
        print("   *** NO frozen pose: re-solving. The 12-point pose sd swamps a 5-point band.")
        ex = ClutterExpert(env, grip_z=args_cli.grip_z, screen=args_cli.screen, verbose=True)
    K = ex.K
    q_home = K.q_arm0.unsqueeze(0).repeat(n, 1)
    tcp_home = K.fk(q_home)["tcp"][0].clone()
    tcp_0 = K.fk(ex.qs[0].unsqueeze(0).repeat(n, 1))["tcp"][0].clone()
    print(f"\n   home TCP     ({tcp_home[0]*1000:6.1f},{tcp_home[1]*1000:+6.1f},"
          f"{tcp_home[2]*1000:6.1f}) mm")
    print(f"   chain[0] TCP ({tcp_0[0]*1000:6.1f},{tcp_0[1]*1000:+6.1f},{tcp_0[2]*1000:6.1f}) mm"
          f"   |  straight-line distance {float((tcp_0 - tcp_home).norm())*1000:.1f} mm")

    # ------------------------------------------------------------------ action magnitudes (N7)
    A = torch.stack([(q - K.q_arm0) / 0.5 for q in ex.qs])          # (W, 6) nominal chain
    print("\n   ACTION MAGNITUDE OVER THE NOMINAL CHAIN   a = (q_target - q_default) / 0.5")
    print("      joint |   min    max   |a|max")
    over = []
    for j in range(6):
        lo, hi = float(A[:, j].min()), float(A[:, j].max())
        m = max(abs(lo), abs(hi))
        over.append(m > 1.0)
        print(f"      j{j+1}    | {lo:+6.3f} {hi:+6.3f}  {m:6.3f}" + ("   <-- OUTSIDE [-1,1]" if m > 1.0 else ""))
    n_over = sum(over)
    print(f"\n   -> {n_over}/6 joints leave the [-1, 1] action box. "
          + ("With clip_actions = 1.0 an rl_games agent CANNOT reach this chain."
             if n_over else "clip_actions = 1.0 is sufficient for this chain -- N7 falsified."))

    # ------------------------------------------------------------------- candidate approaches
    boxes, margin = ex.boxes, ex.margin
    bn = list(K.robot.body_names)
    lerp_q = [K.q_arm0 + (ex.qs[0] - K.q_arm0) * (i / 40.0) for i in range(41)]
    pen_a, low_a, dev_a, w_a = audit(K, lerp_q, boxes, margin, tcp_home, tcp_0, bn)
    cart_pts, cart_qs = ex._dense([tcp_home, tcp_0], K.q_arm0)
    pen_b, low_b, dev_b, w_b = audit(K, cart_qs, boxes, margin, tcp_home, tcp_0, bn)

    # C -- the same dense Cartesian path solved BACKWARD, seeded from the frozen grasp chain.
    #
    # Run 1 measured B's approach at 0.00 mm penetration and still scored 2.7 %, with the
    # hazard landing in `settle` (98 %) -- the phase AFTER the approach, holding still. The
    # reason is the seam: `_dense([tcp_home, tcp_0], q_arm0)` ends at a joint vector 1.95 rad
    # away on `joint6` from the frozen chain's `qs[0]`, i.e. a different IK branch at the same
    # TCP. The schedule's first hold then commands that 1.95 rad as a step change while the
    # gripper is directly above the row.
    #
    # This is P17's defect at a new location. `_dense` fixed branch flips WITHIN a path by
    # solving locally from the previous waypoint; nothing enforced the same rule ACROSS the
    # join between two independently solved paths. Solving backward from `ex.qs[0]` makes the
    # seam the identity by construction -- which is exactly what `plan()` already does for the
    # descent, building `dn_pts` outward from the grasp and reversing it.
    back_pts, back_qs = ex._dense([tcp_0, tcp_home], ex.qs[0])
    c_qs = list(reversed(back_qs))
    seam = float((c_qs[-1] - ex.qs[0]).abs().max())
    join = float((c_qs[0] - K.q_arm0).abs().max())
    c_path = [K.q_arm0 + (c_qs[0] - K.q_arm0) * (i / 10.0) for i in range(11)] + c_qs
    pen_c, low_c, dev_c, w_c = audit(K, c_path, boxes, margin, tcp_home, tcp_0, bn)

    print("\n   KINEMATIC AUDIT OF THE APPROACH")
    print(f"      {'candidate':<22} {'penetration':>12} {'min body z':>12} {'max line dev':>14}")
    print(f"      {'A  joint-space lerp':<22} {pen_a:9.2f} mm {low_a:9.1f} mm {dev_a:11.1f} mm")
    print(f"      {'B  dense Cartesian':<22} {pen_b:9.2f} mm {low_b:9.1f} mm {dev_b:11.1f} mm"
          f"   ({len(cart_qs)} waypoints)")
    print(f"      {'C  backward Cartesian':<22} {pen_c:9.2f} mm {low_c:9.1f} mm {dev_c:11.1f} mm"
          f"   ({len(c_qs)} waypoints)")
    print(f"\n      SEAM AT chain[0]  -- max |dq| between the approach's last waypoint and the")
    print(f"      frozen chain's qs[0], which the schedule then holds:")
    print(f"         B forward  {float((cart_qs[-1] - ex.qs[0]).abs().max()):.4f} rad"
          f"   per joint " + " ".join(f"{float(x):+.2f}" for x in (cart_qs[-1] - ex.qs[0])))
    print(f"         C backward {seam:.4f} rad   <-- identity by construction")
    print(f"      JOIN AT home -- max |dq| from the reset pose to C's first waypoint: {join:.4f} rad")
    for lbl, w in (("A", w_a), ("B", w_b), ("C", w_c)):
        if w:
            print(f"      {lbl} worst: {w['body']} into {w['box']} at {w['frac']:.0%} of the "
                  f"path, {w['pen_mm']:.2f} mm | TCP {w['tcp_mm']} mm, "
                  f"{w['line_dev_mm']:.0f} mm off the straight line")
    print("      (line deviation is REPORTED, not a pass criterion -- but the TCP is not the")
    print("       arm, and it was assuming otherwise that made the lerp look safe.)")

    # ------------------------------------------------------------------------------- physical
    snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}

    def restore():
        for k, v in snap.items():
            e.scene[k].write_root_state_to_sim(v.clone())
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def run(kind, steps, tape):
        """Execute one arm end to end, physics-only. Returns the env's own predicates."""
        chain = ex.adapt()
        if kind == "T":
            K.teleport_arm(chain[0], Q_OPEN)
        else:
            K.teleport_arm(q_home, Q_OPEN)
            if tape is not None:
                tape.label = "approach"
            if kind == "A":
                _move(K, q_home, chain[0], steps, Q_OPEN, tape)
            else:
                # bind the nominal Cartesian approach to this reset: ramp the per-env
                # correction linearly from 0 at home to the full offset at chain[0], so the
                # last target is exactly `chain[0]` whenever the nominal path ends on qs[0].
                nom = cart_qs if kind == "B" else c_qs
                d = chain[0] - ex.qs[0].unsqueeze(0)
                seq = [nom[i].unsqueeze(0) + d * (i / (len(nom) - 1)) for i in range(len(nom))]
                if kind == "B":
                    seq = seq[1:]          # B's nom[0] IS q_home; do not re-command it
                prev, sub = q_home, max(1, steps // len(seq))
                for tgt in seq:
                    _move(K, prev, tgt, sub, Q_OPEN, tape)
                    prev = tgt
        gap_stall = None
        for item in ex.schedule(chain):
            phase, k = item[0], item[1]
            if tape is not None:
                tape.label = phase
            qf = Q_CLOSE if item[-1] else Q_OPEN
            if k == "hold":
                _hold(K, item[2], item[3], qf, tape)
            else:
                _move(K, item[2], item[3], item[4], qf, tape)
            if phase == "close":
                gap_stall = K.gap().clone()
        return ex.score(gap_stall)

    ARMS = [("T   teleport (baseline)", "T", 0),
            ("A40 lerp, 40 steps", "A", 40),
            ("B40 fwd cartesian, 40", "B", 40),
            ("C40 bwd cartesian, 40", "C", 40),
            ("C80 bwd cartesian, 80", "C", 80)]

    print("\n   PHYSICAL, PAIRED -- snapshot restored between arms, "
          f"{args_cli.reps} spawn batches\n")
    results = {}
    for rep in range(args_cli.reps):
        if rep:
            env.reset()
            for _ in range(30):
                e.sim.step()
                e.scene.update(e.physics_dt)
            snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}
        org = e.scene.env_origins
        dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST], dim=1).clone()
        for label, kind, steps in ARMS:
            restore()
            tape = Tape(e, dpos0)
            r = run(kind, steps, tape)
            hz = hazards(tape, n, dev)
            print(f"      batch {rep} | {label:<26} encl {float(r['held'].float().mean()):6.1%}"
                  f" | goal {float(r['at_goal'].float().mean()):6.1%}"
                  f" | topple {float(r['topple'].float().mean()):6.1%}"
                  f" | SUCCESS {float(r['success'].float().mean()):6.1%}")
            print(f"                {'':<26} hazards: "
                  + (", ".join(f"{k} {v:.0%}" for k, v in hz.items()) or "none"))
            results.setdefault(label, []).append(
                {"rep": rep, "encl": float(r["held"].float().mean()),
                 "at_goal": float(r["at_goal"].float().mean()),
                 "topple": float(r["topple"].float().mean()),
                 "success": float(r["success"].float().mean()),
                 "approach_hazard": hz.get("approach", 0.0), "hazard": hz,
                 "succ_mask": r["success"].tolist()})
        restore()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    base = None
    for label, _, _ in ARMS:
        S = torch.tensor([x for c in results[label] for x in c["succ_mask"]], dtype=torch.float32)
        ah = max(c["approach_hazard"] for c in results[label])
        m = float(S.mean())
        if base is None:
            base = m
        print(f"   {label:<26} pooled {m:6.1%}  ({len(S)} eps)  vs teleport {m - base:+6.1%}"
              f"   approach hazard max {ah:.1%}")
        results[label + "__pooled"] = m

    print("\n   " + "=" * 90)
    print("   VERDICT: the registered prediction is FALSIFIED -- the joint-space lerp is dirty")
    print(f"   ({pen_a:.2f} mm, {w_a['body'] if w_a else '?'} into "
          f"{w_a['box'] if w_a else '?'}). What ships is decided by C:")
    for lbl in ("C40 bwd cartesian, 40", "C80 bwd cartesian, 80"):
        m = results[lbl + "__pooled"]
        ah = max(c["approach_hazard"] for c in results[lbl])
        ok = abs(m - base) <= 0.05 and ah <= 1e-9
        print(f"      {lbl}: {m:.1%} vs teleport {base:.1%}  delta {(m - base) * 100:+.2f} pts"
              f"   approach hazard {ah:.1%}   {'PASS' if ok else '*** FAIL ***'}")
    print("   " + "=" * 90)

    out = {"num_envs": n, "reps": args_cli.reps, "grip_z": args_cli.grip_z,
           "tcp_home": [float(x) for x in tcp_home], "tcp_chain0": [float(x) for x in tcp_0],
           "action_abs_max": [float(A[:, j].abs().max()) for j in range(6)],
           "joints_outside_unit_box": int(n_over),
           "audit": {"lerp": {"pen_mm": pen_a, "low_mm": low_a, "dev_mm": dev_a, "worst": w_a},
                     "cartesian": {"pen_mm": pen_b, "low_mm": low_b, "dev_mm": dev_b,
                                   "waypoints": len(cart_qs), "worst": w_b,
                                   "seam_rad": float((cart_qs[-1] - ex.qs[0]).abs().max()),
                                   "pts": [[float(x) for x in p] for p in cart_pts],
                                   "qs": [[float(x) for x in q] for q in cart_qs]},
                     "backward": {"pen_mm": pen_c, "low_mm": low_c, "dev_mm": dev_c,
                                  "waypoints": len(c_qs), "worst": w_c,
                                  "seam_rad": seam, "join_rad": join,
                                  "pts": [[float(x) for x in p] for p in reversed(back_pts)],
                                  "qs": [[float(x) for x in q] for q in c_qs]}},
           "arms": {k: v for k, v in results.items() if not k.endswith("__pooled")},
           "pooled": {k[:-8]: v for k, v in results.items() if k.endswith("__pooled")}}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p29] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
