# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P28 -- Do independently screened grasp poses live in the same IK branch?

Why this has to be answered before a single demo is written
-----------------------------------------------------------
Stage 2 wants demos from several spawn seeds, and each seed screens its own grasp pose
(`ClutterExpert.plan` -> `_screen`). That is good for coverage: the pose draw is the single
largest remaining source of variance (P25 measured 32.8 %-74.2 % between draws of the same
configuration, sd 17.5 %; screening cut it to 4.8 % but did not remove it).

It is dangerous for **behaviour cloning**. `plan()` seeds the CEM from the folded home pose
with `std0 = 0.6, restarts = 12` -- a deliberately wide global search -- and nothing in the
cost constrains two independent solutions to lie in the same IK branch. If they do not, the
demo set contains two very different joint trajectories for near-identical observations, and
a chunk policy that interpolates between them reproduces **exactly the failure P17 found in
the CEM**: a joint-space path that leaves its Cartesian line. P17's version deviated 108 mm
and carried a 100 % contact hazard. Inside a neural network there is no segment audit that
can see it.

So: measure the branch structure, then decide. Decision rule fixed in advance
(`09_STAGE2_BC_PLAN.md` N3):

    all draws within 0.35 rad per joint of a common centre  ->  one branch, use all of them
    otherwise                                               ->  keep the largest cluster only

Pre-registered prediction
-------------------------
**They are one branch.** The CEM is seeded from the home pose, the hinge cost is dominated by
position, and with `o_hat = x_hat` the approach axis is confined to the y-z plane so the wrist
stub must sit in one of the 12 mm row gaps at y ~ -20 mm. That is a narrow basin.
**Predicted max pairwise L-inf < 0.35 rad.**

The falsifier is a bimodal distance histogram -- for instance a mirrored solution putting the
wrist in the +y gap. P18 showed such poses exist (`o_align` 0.885, at-goal 21 %); the question
is whether screening ever selects one.

Cost: `plan_full = False`, so each draw solves the grasp pose and screens it, and skips the
dense chain. About a tenth of a full plan.

Usage
-----
    python -u eva_bc/clutter/probes/p28_pose_branches.py --num_envs 128 --draws 8
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="IK-branch structure of screened grasp poses.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--draws", type=int, default=8, help="independent spawn batches / poses")
parser.add_argument("--screen", type=int, default=4)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--tol", type=float, default=0.35, help="per-joint L-inf branch tolerance [rad]")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p28_branches.json")
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from clutter_expert import ClutterExpert  # noqa: E402


def cluster(Q: torch.Tensor, tol: float) -> list[list[int]]:
    """Single-link clustering of (K, 6) joint vectors under the per-joint L-inf metric.

    Single-link, not centroid: two poses in the same branch are connected by a continuous
    family, so a chain of nearby draws is one branch even if its ends are far apart. Using a
    centroid radius here would split a genuinely connected branch and produce a false alarm.
    """
    k = Q.shape[0]
    D = (Q.unsqueeze(0) - Q.unsqueeze(1)).abs().amax(dim=2)   # (K, K)
    seen, out = set(), []
    for i in range(k):
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            j = stack.pop()
            comp.append(j)
            for m in range(k):
                if m not in seen and float(D[j, m]) <= tol:
                    seen.add(m)
                    stack.append(m)
        out.append(sorted(comp))
    return out


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()

    print("\n" + "=" * 100)
    print("P28 -- IK-BRANCH STRUCTURE OF SCREENED GRASP POSES")
    print("=" * 100)
    print(f"   {args_cli.draws} independent spawn batches, screen = {args_cli.screen}, "
          f"branch tolerance {args_cli.tol:.2f} rad (per-joint L-inf, single-link)")
    print("   PREDICTION (registered before the run): one branch, max pairwise L-inf < 0.35 rad.")

    rows = []
    for d in range(args_cli.draws):
        env.reset()
        for _ in range(30):
            e.sim.step()
            e.scene.update(e.physics_dt)
        ex = ClutterExpert(env, grip_z=args_cli.grip_z, screen=args_cli.screen,
                           plan_full=False, verbose=False)
        p = ex.pose
        rows.append({"draw": d, "q": [float(x) for x in p["q"]],
                     "o_align": p["o_align"], "pen_mm": p["pen"] * 1000.0,
                     "pos_err_mm": p["pos_err"] * 1000.0,
                     "wrist_z_mm": p.get("wrist_z", float("nan")) * 1000.0,
                     "screen_score": p.get("screen_score"),
                     "tcp": [float(x) for x in p["tcp"]],
                     "o_hat": [float(x) for x in p["o_hat"]],
                     "a_hat": [float(x) for x in p["a_hat"]]})
        q = rows[-1]["q"]
        print(f"   draw {d}: q = [" + ", ".join(f"{x:+.3f}" for x in q) + "]"
              f" | o_align {p['o_align']:.4f} | wrist z {rows[-1]['wrist_z_mm']:5.1f} mm"
              f" | screen {p.get('screen_score', float('nan')):.1%}")

    Q = torch.tensor([r["q"] for r in rows])
    D = (Q.unsqueeze(0) - Q.unsqueeze(1)).abs().amax(dim=2)
    comps = cluster(Q, args_cli.tol)

    print("\n   pairwise per-joint L-inf distance [rad]")
    print("        " + " ".join(f"{j:>6d}" for j in range(len(rows))))
    for i in range(len(rows)):
        print(f"    {i:>3d} " + " ".join(f"{float(D[i, j]):6.3f}" for j in range(len(rows))))

    off = D[~torch.eye(len(rows), dtype=torch.bool)]
    print(f"\n   max pairwise L-inf : {float(off.max()):.3f} rad")
    print(f"   median             : {float(off.median()):.3f} rad")
    print(f"   per-joint spread (max-min over draws): "
          + " ".join(f"j{j + 1} {float(Q[:, j].max() - Q[:, j].min()):.3f}" for j in range(6)))

    print(f"\n   clusters at tol = {args_cli.tol:.2f}: {len(comps)}")
    for c in comps:
        print(f"      {c}  (n = {len(c)})")

    one = len(comps) == 1
    print("\n   " + "=" * 90)
    if one:
        print("   VERDICT: ONE BRANCH. Prediction holds -- all draws are usable for demos.")
    else:
        big = max(comps, key=len)
        print(f"   VERDICT: {len(comps)} BRANCHES. Prediction FALSIFIED.")
        print(f"   Decision rule N3 fires: keep cluster {big} only "
              f"({len(big)}/{len(rows)} draws, {100 * (1 - len(big) / len(rows)):.0f} % of "
              f"planned demo diversity discarded).")
    print("   " + "=" * 90)

    out = {"draws": args_cli.draws, "screen": args_cli.screen, "tol": args_cli.tol,
           "num_envs": args_cli.num_envs, "poses": rows,
           "dist": D.tolist(), "clusters": comps, "one_branch": one,
           "max_linf": float(off.max()), "median_linf": float(off.median())}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p28] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
