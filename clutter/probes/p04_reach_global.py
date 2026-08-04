# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P04 -- The attainable orientation set at the target, by GLOBAL sampling.

Why this replaces the CEM for the feasibility question
------------------------------------------------------
P02 and P03 both returned poses with `a_hat` pointing back toward the robot and upward
(`a_hat ~ (-0.7, ., +0.7)`), and P03's control grasp closed on air (`gap = -1.2 mm`) even
with the clutter removed. A CEM seeded at the folded home pose (`joint2 = -1.35,
joint3 = -0.3, joint4 = -0.85`) and stepped with `std0 ~ 0.5` is a *local* search: it can only
report what is reachable from where it started. It cannot distinguish "this orientation is
unattainable" from "I never left this branch", and those have opposite consequences.

So do what `reachability_map.py` does and sample the joint space **globally**: uniform draws
over all six joint limits, forward kinematics in one batch, keep the draws whose TCP lands
near the point of interest, and look at the distribution of achieved orientations. No seed, no
local optimum, no way to mistake a search failure for a kinematic fact.

What it answers
---------------
1. **The home pose**, printed in full -- TCP, approach axis, opening axis, finger positions.
   This should have been the first measurement of the whole effort; both earlier probes
   assumed it rather than reading it.
2. **The attainable approach directions** with the TCP within a tolerance of the grasp point,
   as a histogram over `a_hat`. Directly reproduces C1's "top-down capable" statistic at the
   one workspace point this task cares about, and says whether `a_hat . x > 0` -- a gripper
   reaching *forward* onto the block -- exists at all.
3. **The best sampled grasp candidates** for each orientation family, written out as joint
   vectors so P03 can be re-run from a seed that is in the right basin.

Usage
-----
    python eva_bc/clutter/probes/p04_reach_global.py --num_envs 4096 --batches 400
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Global reachability/orientation set at the row.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--batches", type=int, default=400)
parser.add_argument("--tol", type=float, default=0.010, help="TCP capture radius [m]")
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p04_reach_global.json")
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _kin import TCP_OFFSET, ArmKin  # noqa: E402

ROW_X = 0.250


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    K = ArmKin(env)
    dev, n = K.dev, K.n

    out: dict = {"task": args_cli.task, "num_envs": n, "batches": args_cli.batches,
                 "tol_m": args_cli.tol, "grip_z": args_cli.grip_z,
                 "tcp_offset": list(TCP_OFFSET)}

    print("\n" + "=" * 92)
    print("P04 -- GLOBAL ATTAINABLE ORIENTATION SET AT THE TARGET")
    print("=" * 92)

    # ----------------------------------------------------------------- 1. the home pose
    print("\n" + "-" * 92)
    print("1. THE HOME POSE, READ RATHER THAN ASSUMED")
    print("-" * 92)
    g = K.fk(K.q_arm0.unsqueeze(0).repeat(n, 1))
    fl, fr = K.finger_pos()
    home = {"q_arm": [float(v) for v in K.q_arm0],
            "tcp_mm": (g["tcp"][0] * 1000).tolist(),
            "end_mm": (g["end"][0] * 1000).tolist(),
            "a_hat": (g["a_hat"][0]).tolist(),
            "o_hat": (g["o_hat"][0]).tolist(),
            "finger_l_mm": (fl[0] * 1000).tolist(),
            "finger_r_mm": (fr[0] * 1000).tolist()}
    for k, v in home.items():
        print(f"   {k:>14}: {[round(x, 3) for x in v] if isinstance(v, list) else v}")
    ah = g["a_hat"][0]
    print(f"\n   approach tilt off straight-down: "
          f"{math.degrees(math.acos(max(-1.0, min(1.0, float(-ah[2]))))):.1f} deg")
    print(f"   a_hat . x_hat = {float(ah[0]):+.3f}  "
          f"({'points AWAY from the robot' if ah[0] > 0 else 'points BACK toward the robot'})")
    out["home"] = home

    # -------------------------------------------------- 2. global sample of the reach set
    print("\n" + "-" * 92)
    print(f"2. GLOBAL SAMPLING -- {n * args_cli.batches:,} uniform joint draws")
    print("-" * 92)
    tgt = torch.tensor([ROW_X, 0.0, args_cli.grip_z], device=dev)
    lo, hi = K.lo, K.hi
    keep_q, keep_a, keep_o, keep_d = [], [], [], []
    n_tot = 0
    for b in range(args_cli.batches):
        q = lo + (hi - lo) * torch.rand((n, 6), device=dev)
        gg = K.fk(q)
        d = (gg["tcp"] - tgt).norm(dim=1)
        m = d < args_cli.tol
        n_tot += n
        if int(m.sum()) == 0:
            continue
        keep_q.append(q[m].clone())
        keep_a.append(gg["a_hat"][m].clone())
        keep_o.append(gg["o_hat"][m].clone())
        keep_d.append(d[m].clone())
    if not keep_q:
        print(f"   NO sample landed within {args_cli.tol * 1000:.0f} mm of "
              f"({ROW_X}, 0, {args_cli.grip_z}).")
        print("   The grasp point itself is outside the sampled reach set -- report and stop.")
        out["n_hits"] = 0
        with open(args_cli.out, "w") as f:
            json.dump(out, f, indent=2)
        env.close()
        return

    Q = torch.cat(keep_q)
    A = torch.cat(keep_a)
    O = torch.cat(keep_o)
    D = torch.cat(keep_d)
    print(f"   {len(Q):,} of {n_tot:,} draws land within {args_cli.tol * 1000:.0f} mm "
          f"({100.0 * len(Q) / n_tot:.4f} %)")
    out["n_hits"], out["n_samples"] = int(len(Q)), int(n_tot)

    tilt = torch.rad2deg(torch.acos(A[:, 2].clamp(-1, 1) * -1.0))
    print(f"\n   approach tilt off straight-down [deg]: min {float(tilt.min()):.1f}, "
          f"p5 {float(tilt.quantile(0.05)):.1f}, median {float(tilt.median()):.1f}, "
          f"max {float(tilt.max()):.1f}")
    print(f"   a_hat . x_hat : min {float(A[:, 0].min()):+.3f}  max {float(A[:, 0].max()):+.3f}"
          f"   ({int((A[:, 0] > 0).sum()):,} of {len(A):,} point away from the robot)")
    print(f"   a_hat . z_hat : min {float(A[:, 2].min()):+.3f}  max {float(A[:, 2].max()):+.3f}")
    print(f"   |o_hat . y|   : max {float(O[:, 1].abs().max()):.3f}   "
          f"|o_hat . x| max {float(O[:, 0].abs().max()):.3f}")
    out["stats"] = {
        "tilt_min_deg": float(tilt.min()), "tilt_med_deg": float(tilt.median()),
        "ax_min": float(A[:, 0].min()), "ax_max": float(A[:, 0].max()),
        "az_min": float(A[:, 2].min()), "az_max": float(A[:, 2].max()),
        "oy_absmax": float(O[:, 1].abs().max()), "ox_absmax": float(O[:, 0].abs().max()),
        "n_forward": int((A[:, 0] > 0).sum()),
    }

    print("\n   approach-direction histogram (rows: a_hat.x sign; cols: tilt band)")
    bands = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 181)]
    print("      " + "".join(f"{f'{a}-{b}d':>10}" for a, b in bands))
    for lbl, msk in (("a.x > 0", A[:, 0] > 0), ("a.x <= 0", A[:, 0] <= 0)):
        cells = [int(((tilt >= a) & (tilt < b) & msk).sum()) for a, b in bands]
        print(f"   {lbl:>8}" + "".join(f"{c:>10,}" for c in cells))
    out["hist"] = {lbl: [int(((tilt >= a) & (tilt < b) & msk).sum()) for a, b in bands]
                   for lbl, msk in (("ax_pos", A[:, 0] > 0), ("ax_neg", A[:, 0] <= 0))}

    # ---------------------------------------------------- 3. best candidates per family
    print("\n" + "-" * 92)
    print("3. BEST SAMPLED CANDIDATES PER GRASP FAMILY")
    print("-" * 92)
    print("   G1 cross-row : |o_hat . y| > 0.95   (fingers straddle the block across the row)")
    print("   G2 front-back: |o_hat . x| > 0.95   (fingers straddle it fore and aft)")
    print("   Scored by TCP error; the approach direction is reported, not constrained.")
    print()
    cands = {}
    for fam, msk in (("G1", O[:, 1].abs() > 0.95), ("G2", O[:, 0].abs() > 0.95)):
        k = int(msk.sum())
        print(f"   {fam}: {k:,} samples")
        if k == 0:
            cands[fam] = []
            continue
        idx = torch.nonzero(msk).squeeze(-1)
        order = idx[D[idx].argsort()][:5]
        rec = []
        for j in order:
            a = A[j]
            t = math.degrees(math.acos(max(-1.0, min(1.0, float(-a[2])))))
            rec.append({"q": [float(v) for v in Q[j]], "tcp_err_mm": float(D[j]) * 1000,
                        "a_hat": [float(v) for v in a], "tilt_deg": t,
                        "o_hat": [float(v) for v in O[j]]})
            print(f"      err {float(D[j]) * 1000:5.2f}mm  a_hat=({a[0]:+.2f},{a[1]:+.2f},"
                  f"{a[2]:+.2f})  tilt {t:5.1f}d  q={[round(float(v), 3) for v in Q[j]]}")
        # also the candidate whose approach points most forward-and-down
        score = A[idx, 0] - A[idx, 2]
        j = idx[int(score.argmax())]
        a = A[j]
        t = math.degrees(math.acos(max(-1.0, min(1.0, float(-a[2])))))
        print(f"      most forward-and-down: a_hat=({a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f})  "
              f"tilt {t:5.1f}d  err {float(D[j]) * 1000:5.2f}mm")
        rec.append({"q": [float(v) for v in Q[j]], "tcp_err_mm": float(D[j]) * 1000,
                    "a_hat": [float(v) for v in a], "tilt_deg": t, "tag": "forward_down",
                    "o_hat": [float(v) for v in O[j]]})
        cands[fam] = rec
    out["candidates"] = cands

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    fwd_down = ((A[:, 0] > 0.3) & (A[:, 2] < -0.3)).sum()
    print(f"   samples with a_hat pointing forward AND down (a.x>0.3, a.z<-0.3): {int(fwd_down):,}")
    if int(fwd_down) == 0:
        print("   -> At this grasp point the arm CANNOT point its fingers forward-and-down.")
        print("      That is a kinematic fact about the arm, not a search artifact, and it")
        print("      explains P03's control failure. Any grasp here must use whatever")
        print("      orientations DO exist above -- re-plan against the measured set.")
    else:
        print("   -> Forward-and-down approaches DO exist. P02/P03's negative results were")
        print("      local-search artifacts. Re-seed the CEM from the candidates above.")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p04] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
