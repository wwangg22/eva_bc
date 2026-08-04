# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P26 -- Stop predicting which pose works. Measure it.

The variance that is now the whole problem
------------------------------------------
Every trajectory defect found in Stage 1 has been fixed, and the expert sits at 53-58 %
pooled. The dominant remaining term is not a failure mode -- it is **which pose the CEM
happened to draw**:

    P25 arm A, one configuration, three draws, two spawn batches each:
        draw 0    68.0 %, 62.5 %          o_align 0.9925, wrist z 18 mm
        draw 1    71.9 %, 74.2 %          o_align 0.9916, wrist z 20 mm
        draw 2    39.1 %, 32.8 %          o_align 0.9919, wrist z 18 mm
                                          -> pooled 53.3 %, sd 17.5 %

The good draw and the bad draw are indistinguishable on every statistic used to select them.
`o_align` agrees to three decimal places. Keep-out penetration is zero for both. Wrist height
differs by 2 mm. **The selector is blind, and the spread it cannot see is worth 40 points --
more than every fix in this stage put together.**

Three attempts to find a predictive forward-kinematic statistic have now failed:
`o_align` (P18, it is necessary but far from sufficient), wrist side (P18, confounded and
withdrawn), wrist height (P25, actively **anti**-correlated: −56.2 points).

The change of approach
----------------------
This effort's founding rule is that candidates are scored by values **read back from the
sim**, never by commanded ones — that is why the CEM scores achieved FK instead of trusting
an IK solver. The rule has simply not been carried far enough. Achieved FK is still a
*proxy* for whether the grasp works. The thing itself is measurable, and cheaply: settle at
the candidate pose, close, and look.

`ClutterExpert(screen=K)` solves K candidate poses and runs each one's close in the
simulator, scoring **`enclosed AND NOT toppled`**. The conjunction is not decoration. P25
scored candidates on disturbance alone and selected poses that disturb nothing because they
grasp nothing — enclosure 19-27 %, isolated topple 2-4 %, end-to-end success **2.0 %**. A
proxy that omits the success condition finds exactly the poses that fail it.

Honest evaluation
-----------------
Screening happens on **one** spawn batch; every reported number comes from **different**
batches. The pose is nominal — one chain for all envs, adapted per env by `refine` — so
screening on one draw of the spawn distribution and testing on others is ordinary
generalisation, not leakage. Reporting the screen score alongside the held-out score makes
any gap visible.

    A  screen = 0    the current expert (equivalent to P25 arm A)
    B  screen = 4
    C  screen = 8

Three independent selections per arm, two held-out spawn batches each: 768 episodes per arm.

Usage
-----
    python eva_bc/clutter/probes/p26_screened_pose.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simulator-screened grasp-pose selection.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--screens", type=str, default="0,4,8")
parser.add_argument("--pose_reps", type=int, default=3)
parser.add_argument("--reps", type=int, default=2)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p26_screen.json")
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
    screens = [int(x) for x in args_cli.screens.split(",")]

    print("\n" + "=" * 100)
    print("P26 -- SIMULATOR-SCREENED POSE SELECTION")
    print("=" * 100)
    print("   score = enclosed AND NOT toppled, measured on the close. Screening uses one")
    print("   spawn batch; every reported number uses different batches.")

    results = {}
    for sc in screens:
        tag = f"screen={sc}"
        print(f"\n   {tag}")
        masks, cells, gaps = [], [], []
        for prep in range(args_cli.pose_reps):
            env.reset()                                    # the SCREENING batch
            for _ in range(30):
                e.sim.step()
                e.scene.update(e.physics_dt)
            ex = ClutterExpert(env, grip_z=args_cli.grip_z, screen=sc, verbose=(sc > 0))
            ss = ex.pose.get("screen_score")
            for rep in range(args_cli.reps):
                env.reset()                                # HELD-OUT batches
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
                print(f"      sel {prep} (o_align {ex.pose['o_align']:.4f}"
                      + (f", screen {ss:.1%}" if ss is not None else "")
                      + f") batch {rep}: encl {float(r['held'].float().mean()):6.1%} | goal "
                      f"{float(r['at_goal'].float().mean()):6.1%} | topple "
                      f"{float(r['topple'].float().mean()):6.1%} | SUCCESS "
                      f"{float(r['success'].float().mean()):6.1%}")
                print(f"            hazards: "
                      + (", ".join(f"{k} {v:.0%}" for k, v in hz.items()) or "none"))
                masks.append(r["success"])
                gaps.append(min_gap)
                cells.append({"sel": prep, "batch": rep, "screen_score": ss,
                              "o_align": ex.pose["o_align"],
                              "encl": float(r["held"].float().mean()),
                              "at_goal": float(r["at_goal"].float().mean()),
                              "topple": float(r["topple"].float().mean()),
                              "success": float(r["success"].float().mean()),
                              "hazard": hz, "succ_mask": r["success"].tolist()})
        S = torch.cat(masks)
        G = torch.cat(gaps)
        rr = [c["success"] for c in cells]
        print(f"      POOLED {len(S)} episodes: SUCCESS {float(S.float().mean()):6.1%}  "
              f"(cells {min(rr):.1%}..{max(rr):.1%}, sd {torch.tensor(rr).std():.1%})")
        results[tag] = {"screen": sc, "success": float(S.float().mean()),
                        "cell_min": min(rr), "cell_max": max(rr),
                        "sd": float(torch.tensor(rr).std()), "cells": cells,
                        "min_gap_mm": (G * 1000).tolist(),
                        "succ_all": S.tolist()}

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"   {'arm':>12} | {'pooled':>8} | {'spread':>16} | {'sd':>7} | screen scores")
    for sc in screens:
        r = results[f"screen={sc}"]
        ss = sorted({c["screen_score"] for c in r["cells"] if c["screen_score"] is not None})
        print(f"   {f'screen={sc}':>12} | {r['success']:7.1%} | "
              f"{r['cell_min']:6.1%}..{r['cell_max']:6.1%} | {r['sd']:6.1%} | "
              + (", ".join(f"{x:.0%}" for x in ss) or "-"))

    base = results[f"screen={screens[0]}"]
    print(f"\n   {'arm':>12} | vs screen=0 | does the screen score predict the held-out one?")
    for sc in screens[1:]:
        r = results[f"screen={sc}"]
        pairs = [(c["screen_score"], c["success"]) for c in r["cells"]
                 if c["screen_score"] is not None]
        gap = (sum(p[0] for p in pairs) / len(pairs)
               - sum(p[1] for p in pairs) / len(pairs)) if pairs else float("nan")
        print(f"   {f'screen={sc}':>12} | {r['success'] - base['success']:+9.1%} | "
              f"screen mean − held-out mean = {gap:+.1%}"
              + ("   (optimistic, as expected for a selection score)" if gap > 0.03 else ""))

    print("\n   stratified by minimum free gap, best arm:")
    best = max(results.values(), key=lambda r: r["success"])
    S = torch.tensor(best["succ_all"], device=dev)
    G = torch.tensor(best["min_gap_mm"], device=dev)
    for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 10), (10, 14)):
        m = (G >= lo) & (G < hi)
        if int(m.sum()):
            print(f"      {f'{lo}-{hi} mm':>10} | n {int(m.sum()):4d} | "
                  f"success {float(S[m].float().mean()):6.1%}")

    out = {"n": n, "grip_z": args_cli.grip_z, "screens": screens,
           "pose_reps": args_cli.pose_reps, "reps": args_cli.reps, "arms": results}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p26] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
