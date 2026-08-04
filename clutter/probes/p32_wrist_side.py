# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P32 -- Is the wrist side the pose variable that has been driving everything?

Where this came from
--------------------
P26 has now been run five times (screen 0/4/8, v2 both-rolls, v3 roll-rule, v4 leak-fix),
18 pose selections in total. Pooling every selection ever measured and asking which
recorded property predicts the held-out score gives an uncomfortable answer:

    screen score  ->  held-out       r = +0.126   r2 = 0.016      n = 15
    o_align       ->  held-out       r = +0.065   r2 = 0.004      n = 15
    wrist side    ->  held-out       +y 76.8 % (n=6) vs -y 63.8 % (n=9)
                                     +13.0 pts, se 3.7, t = 3.54
                                     6/6 of the +y selections beat 7/9 of the -y ones
                                     Fisher one-sided p = 0.0056

The simulator screen -- four extra CEM solves and four extra physical closes per
selection, credited with +10 points in Stage 1 -- explains **1.6 %** of the variance in
the thing it selects for. The wrist side, which `plan()` does not control at all and
which P18 withdrew as "confounded", separates the selections almost perfectly.

And the within-run check does not depend on pooling across configurations:

    v1s8   +y 78.1  -y 61.7   delta +16.4
    v2     +y 78.9  -y 60.4   delta +18.5
    v3     +y 83.2  -y 70.3   delta +12.9
    v4     +y 78.1  -y 55.1   delta +23.0
    v1s4   +y 71.1  -y 78.9   delta  -7.8      <- the one exception, n=1 on the -y side

Why P26 could never have settled this
-------------------------------------
Selection-level sd across the screened runs is **10.1 points**. Every P26 arm used
`pose_reps = 3`, so the standard error of a run mean is ~6 points and its 95 % CI is ~+/-12.
Resolving a 5-point effect at that sd needs **63** selections per arm; resolving 10 needs
**16**. Every P26 verdict -- including "screening is worth +10" and "v4 is worse than v3" --
sits inside its own noise. P26 compared *different poses* across arms and never paired
anything, which is precisely the Stage-1 rule it was violating.

The design
----------
Three arms, and the variable is forced rather than observed. `ClutterExpert` already
accepts `wrist_side`; it has simply never been set.

    A  wrist_side = +1     reject any candidate whose wrist lands at -y
    B  wrist_side = -1     reject any candidate whose wrist lands at +y
    C  wrist_side =  0     free -- the current default, as a same-power control

`screen = 0` for all three: an r2 of 0.016 does not justify 4x the solve cost, and dropping
it removes a second, uncontrolled selection stage from the comparison. `--screen` re-enables
it if that needs re-testing later.

`pose_reps = 6` per arm, not 3. At the within-side sd (+y 4.8, -y 9.3, pooled ~7.5) six
selections give se ~3, enough to resolve the claimed 13 points at ~4 sigma. This is the
first P26-family experiment that is powered for the effect it is measuring.

**The evaluation batches are paired.** Every arm evaluates on the same spawn batches, drawn
from `--batch-seed0 + 1000*sel + rep`, so batch variance -- which reached 18 points inside a
single v4 cell (61.7 % vs 43.8 %) -- is differenced out instead of being absorbed into the
error term. Pairing is **verified, not assumed**: each batch's spawn fingerprint is recorded
and the run asserts the arms saw identical scenes.

Registered predictions
----------------------
1. Arm A > arm B by **>= 8 points**. Falsifier: |A - B| < 5 points, which would mean the
   pooled P26 pattern was an artifact of confounding with run configuration.
2. Arm C lands between them, near the draw-weighted mean. P28 drew +y 5 / -y 3, so
   C ~ 0.6*A + 0.4*B.
3. The gap is carried by **topple**, not by enclosure or at-goal -- every mechanism found in
   this effort so far has been. Falsifier: enclosure differs by more than 3 points.

Usage
-----
    python eva_bc/clutter/probes/p32_wrist_side.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Forced wrist-side comparison, batch-paired.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--sides", type=str, default="1,-1,0")
parser.add_argument("--screen", type=int, default=0)
parser.add_argument("--pose_reps", type=int, default=6)
parser.add_argument("--reps", type=int, default=2)
parser.add_argument("--batch-seed0", type=int, default=7000)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p32_wrist_side.json")
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
from _kin import _t  # noqa: E402
from clutter_expert import ClutterExpert, DIST, HY  # noqa: E402


def settle(e, k=30):
    for _ in range(k):
        e.sim.step()
        e.scene.update(e.physics_dt)


def fingerprint(e):
    """A scene identity hash: every object's planar position, to 0.1 mm."""
    org = e.scene.env_origins
    p = [(_t(e.scene[k].data.root_pos_w) - org)[:, :2] for k in ("target",) + DIST]
    return round(float(torch.stack(p).mul(1e4).round().sum()), 1)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    dev, n = e.device, e.num_envs
    sides = [int(x) for x in args_cli.sides.split(",")]

    print("\n" + "=" * 100)
    print("P32 -- FORCED WRIST SIDE, EVALUATION BATCHES PAIRED ACROSS ARMS")
    print("=" * 100)
    print(f"   {len(sides)} arms x {args_cli.pose_reps} selections x {args_cli.reps} batches "
          f"x {n} envs = {len(sides) * args_cli.pose_reps * args_cli.reps * n} episodes")
    print(f"   screen = {args_cli.screen} (the P26 meta-analysis put its r2 at 0.016)")

    results, prints = {}, {}
    for side in sides:
        tag = {1: "wrist +y", -1: "wrist -y", 0: "wrist free"}[side]
        print(f"\n   ARM side={side:+d}  ({tag})")
        masks, cells, sels = [], [], []
        for prep in range(args_cli.pose_reps):
            # The SOLVE batch is deliberately unseeded: independent pose draws are the
            # sampling distribution being estimated. Only the EVALUATION is paired.
            env.reset()
            settle(e)
            ex = ClutterExpert(env, grip_z=args_cli.grip_z, screen=args_cli.screen,
                               wrist_side=side, verbose=False)
            wy = float(ex.pose["wrist"][1]) * 1000.0 if "wrist" in ex.pose else float("nan")
            oa, j6 = float(ex.pose["o_align"]), float(ex.pose["q"][5])
            got = 0 if side == 0 else (1 if wy > 0 else -1)
            ok = "OK " if (side == 0 or got == side) else "*** SIDE NOT HONOURED ***"
            print(f"      sel {prep}: o_align {oa:.4f} | wrist_y {wy:+6.1f} mm | "
                  f"j6 {j6:+.3f} | roll {ex.pose.get('roll', 0):+d} | {ok}")
            sels.append({"sel": prep, "o_align": oa, "wrist_y_mm": wy, "j6": j6,
                         "roll": int(ex.pose.get("roll", 0)), "side_honoured": ok == "OK "})
            for rep in range(args_cli.reps):
                seed = args_cli.batch_seed0 + 1000 * prep + rep
                env.reset(seed=seed)
                settle(e)
                fp = fingerprint(e)
                prints.setdefault((prep, rep), []).append((side, fp))
                org = e.scene.env_origins
                tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
                dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST],
                                    dim=1).clone()
                ys = torch.cat([dpos0[:, :2, 1], tpos0[:, 1:2], dpos0[:, 2:, 1]], dim=1)
                ys, _ = ys.sort(dim=1)
                min_gap = (ys[:, 1:] - ys[:, :-1] - 2 * HY).min(dim=1).values
                r = ex.run_physics(ex.adapt(), None)
                enc = float(r["held"].float().mean())
                goal = float(r["at_goal"].float().mean())
                top = float(r["topple"].float().mean())
                suc = float(r["success"].float().mean())
                print(f"            batch {rep} (fp {fp:.1f}): encl {enc:6.1%} | goal "
                      f"{goal:6.1%} | topple {top:6.1%} | SUCCESS {suc:6.1%}")
                masks.append(r["success"])
                cells.append({"sel": prep, "batch": rep, "seed": seed, "fingerprint": fp,
                              "o_align": oa, "wrist_y_mm": wy, "encl": enc, "at_goal": goal,
                              "topple": top, "success": suc,
                              "min_gap": min_gap.tolist(),
                              "succ_mask": r["success"].tolist()})
        S = torch.cat(masks)
        per_sel = [sum(c["success"] for c in cells if c["sel"] == k) / args_cli.reps
                   for k in range(args_cli.pose_reps)]
        m = sum(per_sel) / len(per_sel)
        sd = (sum((x - m) ** 2 for x in per_sel) / max(len(per_sel) - 1, 1)) ** 0.5
        se = sd / len(per_sel) ** 0.5
        print(f"      ARM side={side:+d}: POOLED {float(S.float().mean()):6.1%} over "
              f"{S.numel()} episodes | selection mean {m:.1%} sd {sd:.1%} se {se:.1%}")
        print(f"         encl {sum(c['encl'] for c in cells)/len(cells):6.1%} | "
              f"goal {sum(c['at_goal'] for c in cells)/len(cells):6.1%} | "
              f"topple {sum(c['topple'] for c in cells)/len(cells):6.1%}")
        results[str(side)] = {"pooled": float(S.float().mean()), "n": int(S.numel()),
                              "sel_mean": m, "sel_sd": sd, "sel_se": se,
                              "per_sel": per_sel, "selections": sels, "cells": cells}

    # ---- pairing verification: every arm must have seen identical scenes -------------
    bad = {k: v for k, v in prints.items() if len({f for _, f in v}) > 1}
    print("\n" + "=" * 100)
    print("PAIRING CHECK")
    print("=" * 100)
    if bad:
        print(f"   *** {len(bad)} of {len(prints)} (sel,batch) slots DIFFER across arms.")
        print("   *** env.reset(seed=) does NOT reproduce spawns -- the arms are NOT paired,")
        print("   *** and the paired statistics below must be read as unpaired.")
        for k, v in list(bad.items())[:4]:
            print(f"       sel {k[0]} batch {k[1]}: " + ", ".join(f"{s:+d}->{f:.1f}" for s, f in v))
    else:
        print(f"   OK: all {len(prints)} (sel,batch) slots identical across all arms.")
    paired = not bad

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    hdr = f"   {'arm':>10} {'pooled':>8} {'sel mean':>9} {'se':>6} {'encl':>7} {'goal':>7} {'topple':>7}"
    print(hdr)
    for side in sides:
        d, c = results[str(side)], results[str(side)]["cells"]
        print(f"   {side:>+10d} {d['pooled']:8.1%} {d['sel_mean']:9.1%} {d['sel_se']:6.1%} "
              f"{sum(x['encl'] for x in c)/len(c):7.1%} {sum(x['at_goal'] for x in c)/len(c):7.1%} "
              f"{sum(x['topple'] for x in c)/len(c):7.1%}")
    if "1" in results and "-1" in results:
        a, b = results["1"], results["-1"]
        d = (a["sel_mean"] - b["sel_mean"]) * 100
        sed = ((a["sel_se"] * 100) ** 2 + (b["sel_se"] * 100) ** 2) ** 0.5
        print(f"\n   +y minus -y = {d:+.1f} pts, se {sed:.1f}, t = {d/sed if sed else 0:.2f}")
        print(f"   PREDICTION 1 (>= 8 pts): {'HELD' if d >= 8 else 'NOT HELD'}"
              f"   falsifier (|d| < 5): {'TRIGGERED' if abs(d) < 5 else 'not triggered'}")
        ca = [x for x in a["cells"]]; cb = [x for x in b["cells"]]
        de = (sum(x["encl"] for x in ca) / len(ca) - sum(x["encl"] for x in cb) / len(cb)) * 100
        dt = (sum(x["topple"] for x in ca) / len(ca) - sum(x["topple"] for x in cb) / len(cb)) * 100
        print(f"   PREDICTION 3 (carried by topple): d_topple {dt:+.1f} pts, "
              f"d_encl {de:+.1f} pts -> {'HELD' if abs(de) <= 3 else 'NOT HELD'}")

    with open(args_cli.out, "w") as f:
        json.dump({"args": vars(args_cli), "paired": paired, "arms": results}, f)
    print(f"\n   wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
