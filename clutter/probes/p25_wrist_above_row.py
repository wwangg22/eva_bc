# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P25 -- Put the wrist ABOVE the row instead of threading a 12 mm gap.

The observation, and why it was nearly missed
---------------------------------------------
P23 swept grip height on the isolated close, three pose draws per height. The headline was
that **grip height barely matters**: 19.5-31.5 % isolated topple across 40-80 mm, with a
within-height spread over pose draws (6-67 %) as large as the between-height variation.
P20's earlier three-point result (65 -> 43 %, 55 -> 13 %, 50 -> 53 %) was mostly pose noise.

The signal was in the per-draw rows, not the aggregate:

    grip 80 mm, draw 0   wrist at (-41, **+72**)   isolated topple  **3.9 %**
    grip 80 mm, draw 2   wrist at (-42, **+73**)   isolated topple  **2.3 %**
    grip 80 mm, draw 1   wrist at (-20, **+44**)   isolated topple    27.3 %

The blocks' tops are at ~67 mm. Two of those three draws put `gripper_end` **above the
row**; the third left it in the gap. Same height, same everything else, an order of
magnitude apart.

Why this was thought impossible
-------------------------------
P14 swept the approach axis over the full circle at `grip_z = 65 mm` and concluded the wrist
must thread a gap, because `gripper_end = TCP - 0.0419 * a_hat` and

    z_wrist = grip_z - 41.9 * cos(t)   >  67 mm   requires   cos(t) < -0.05

i.e. a downward approach axis, which is genuinely unattainable (`a_align = -0.11` at
`a_des = (0,0,-1)`). That algebra is correct **and its conclusion was over-generalised from
one grip height**. At `grip_z = 80 mm` the same expression needs only `cos(t) < 0.31`, which
is well inside the attainable set — the CEM reaches it without being asked, in two draws out
of three.

So the wrist's height is not a fixed consequence of the geometry. It is a **selectable**
property, and unlike grip height it separates the cells cleanly.

The experiment
--------------
`ClutterExpert` gained `wrist_min_z`: the grasp-pose search retries until it finds a pose
that clears the gate as well as the `o_align` and penetration gates, keeping the highest
wrist seen as a fallback so a missed gate is reported rather than hidden.

    A  grip 55 mm, no gate           the current expert, as shipped
    B  grip 80 mm, no gate           height alone, to separate it from the gate
    C  grip 80 mm, wrist >= 70 mm    the gate

Three pose draws each, two spawn batches per draw, full trajectory, pooled — because that is
the only kind of number this effort has learned to trust. Reporting per-draw `wrist_z`
alongside makes the mechanism checkable rather than assumed: if C beats B, the *within*-B
draws that happened to clear 70 mm should already look like C.

Usage
-----
    python eva_bc/clutter/probes/p25_wrist_above_row.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Wrist-above-row pose gate.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--pose_reps", type=int, default=3)
parser.add_argument("--reps", type=int, default=2)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p25_wrist.json")
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

ARMS = [("A grip55, no gate", 0.055, 0.000),
        ("B grip80, no gate", 0.080, 0.000),
        ("C grip80, wrist>=70", 0.080, 0.070)]
ORDER = ["settle", "descend", "predwell", "close", "carry", "dwell", "release",
         "withdraw", "final"]
DISP_ON = 0.0015
TILT_ON = 0.999


class Tape:
    def __init__(self, e, dpos0):
        self.e, self.dpos0 = e, dpos0
        self.up, self.disp, self.phase = [], [], []
        self.label = "settle"

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


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    dev, n = e.device, e.num_envs

    print("\n" + "=" * 100)
    print("P25 -- WRIST ABOVE THE ROW  (block tops ~67 mm)")
    print("=" * 100)

    results = {}
    for tag, gz, wmin in ARMS:
        print(f"\n   {tag}")
        masks, cells = [], []
        for prep in range(args_cli.pose_reps):
            env.reset()
            for _ in range(30):
                e.sim.step()
                e.scene.update(e.physics_dt)
            ex = ClutterExpert(env, grip_z=gz, wrist_min_z=wmin, verbose=False)
            wz = ex.pose["wrist_z"] * 1000
            for rep in range(args_cli.reps):
                env.reset()
                for _ in range(30):
                    e.sim.step()
                    e.scene.update(e.physics_dt)
                org = e.scene.env_origins
                tpos0 = (_t(e.scene["target"].data.root_pos_w) - org).clone()
                dpos0 = torch.stack([(_t(e.scene[d].data.root_pos_w) - org) for d in DIST],
                                    dim=1).clone()
                ys = torch.cat([dpos0[:, :2, 1], tpos0[:, 1:2], dpos0[:, 2:, 1]], dim=1)
                ys, _ = ys.sort(dim=1)
                min_gap = (ys[:, 1:] - ys[:, :-1] - 2 * HY).min(dim=1).values
                tape = Tape(e, dpos0)
                r = ex.run_physics(ex.adapt(), tape)
                hz = hazards(tape, n, dev)
                print(f"      draw {prep} (o_align {ex.pose['o_align']:.4f}, wrist z "
                      f"{wz:5.1f} mm{'  ABOVE ROW' if wz > 67 else '  in gap'}) batch {rep}: "
                      f"encl {float(r['held'].float().mean()):6.1%} | goal "
                      f"{float(r['at_goal'].float().mean()):6.1%} | topple "
                      f"{float(r['topple'].float().mean()):6.1%} | SUCCESS "
                      f"{float(r['success'].float().mean()):6.1%}")
                print(f"            hazards: "
                      + (", ".join(f"{k} {v:.0%}" for k, v in hz.items()) or "none"))
                masks.append(r["success"])
                cells.append({"pose_rep": prep, "spawn_rep": rep, "wrist_z_mm": wz,
                              "o_align": ex.pose["o_align"],
                              "encl": float(r["held"].float().mean()),
                              "at_goal": float(r["at_goal"].float().mean()),
                              "topple": float(r["topple"].float().mean()),
                              "success": float(r["success"].float().mean()),
                              "hazard": hz,
                              "min_gap_mm": (min_gap * 1000).tolist(),
                              "succ_mask": r["success"].tolist()})
        S = torch.cat(masks)
        rr = [c["success"] for c in cells]
        print(f"      POOLED {len(S)} episodes: SUCCESS {float(S.float().mean()):6.1%}  "
              f"(cells {min(rr):.1%}..{max(rr):.1%}, sd {torch.tensor(rr).std():.1%})")
        results[tag] = {"grip_z": gz, "wrist_min_z": wmin,
                        "success": float(S.float().mean()),
                        "cells": cells, "cell_min": min(rr), "cell_max": max(rr),
                        "sd": float(torch.tensor(rr).std())}

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"   {'arm':>22} | {'pooled':>8} | {'spread':>16} | {'sd':>6} | wrist z per draw")
    for tag, _, _ in ARMS:
        r = results[tag]
        wz = sorted({round(c["wrist_z_mm"]) for c in r["cells"]})
        print(f"   {tag:>22} | {r['success']:7.1%} | "
              f"{r['cell_min']:6.1%}..{r['cell_max']:6.1%} | {r['sd']:5.1%} | "
              + ", ".join(f"{x:.0f}" for x in wz))

    # the mechanism check: within every arm, do the high-wrist draws outscore the low ones?
    hi = [c for r in results.values() for c in r["cells"] if c["wrist_z_mm"] > 67]
    lo = [c for r in results.values() for c in r["cells"] if c["wrist_z_mm"] <= 67]
    print(f"\n   MECHANISM CHECK, pooling every cell from every arm by wrist height:")
    for name, grp in (("wrist ABOVE the row (>67 mm)", hi), ("wrist in a gap (<=67 mm)", lo)):
        if grp:
            s = sum(c["success"] for c in grp) / len(grp)
            t = sum(c["topple"] for c in grp) / len(grp)
            print(f"      {name:>30}: {len(grp):2d} cells | success {s:6.1%} | "
                  f"topple {t:6.1%}")
    if hi and lo:
        d = (sum(c["success"] for c in hi) / len(hi)
             - sum(c["success"] for c in lo) / len(lo))
        print(f"      difference: {d:+.1%}")
        print("      -> If this is large and the arm-level difference is not, the WRIST "
              "HEIGHT")
        print("         is the variable and grip height was only a way of reaching it.")

    out = {"n": n, "pose_reps": args_cli.pose_reps, "reps": args_cli.reps,
           "arms": results}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p25] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
