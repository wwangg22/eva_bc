# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P21 -- Gate 1: what does the finished expert actually score, and how much does it vary?

Why this is not just "run it once more"
---------------------------------------
Every headline in Stage 1 has been a single (pose draw, spawn batch) pair, and both terms
have turned out to matter as much as the fixes being tested:

    identical config, different draws     P19 arm A 64.1 %   vs   P20 arm A 39.8 %
    o_align of those two draws                    0.991                     0.978

A 0.013 difference in one selection statistic moved the headline by 24 points. eva_bc paid
for this lesson once already -- it measured **26.6 points of seed variance** and made pooled
multi-seed numbers mandatory. So the expert's number has to be reported the same way:
several independent pose draws, several independent spawn batches, pooled, with the spread
stated rather than the best cell.

This run also carries the `o_align >= 0.99` gate that the expert now applies at plan time,
which is itself a hypothesis: if alignment is what drives the variance, gating on it should
*shrink* the spread, not just raise the mean. That is checkable here and is reported.

Gate 1 (pre-registered in `02_PLAN.md`)
---------------------------------------
    >= 85 % on nominal over >= 128 episodes, stratified by measured minimum free gap.
    Revised expectation recorded at the end of Stage 0: 85 % may be optimistic; 70-85 %
    means proceed but record the BC ceiling in advance; below 70 % means return to
    diagnosis rather than spend a demo chain.

`Tight-v0` is measured in the same run. A prediction was registered before any of this
work: **Tight lands within 10 points of nominal**, because the orthogonal grasp does not
use the row gaps at all. This is where that gets settled.

Usage
-----
    python eva_bc/clutter/probes/p21_gate1.py --num_envs 128 --pose_reps 3 --reps 2
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Gate 1 evaluation of the scripted expert.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--pose_reps", type=int, default=3, help="independent CEM pose draws")
parser.add_argument("--reps", type=int, default=2, help="independent spawn batches per draw")
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--lift_dz", type=float, default=0.150)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p21_gate1.json")
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from _kin import _t  # noqa: E402
from clutter_expert import ClutterExpert, DIST, HY  # noqa: E402

ROW_Y = (-0.084, -0.042, 0.042, 0.084)
DISP_ON = 0.0015
TILT_ON = 0.999
ORDER = ["settle", "descend", "predwell", "close", "carry", "dwell", "release",
         "withdraw", "final"]


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
    clean = torch.ones(n, dtype=torch.bool, device=dev)
    out = {}
    for p in ORDER:
        s = [t for t in range(T) if ph[t] == p]
        if not s:
            continue
        inp = ever & (idx >= s[0]) & (idx <= s[-1])
        r = int(clean.sum())
        if r:
            out[p] = (int(inp.sum()), r, int(inp.sum()) / r)
        clean = clean & ~inp
    return out


def evaluate(task: str, label: str) -> dict:
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    dev, n = e.device, e.num_envs
    goal = torch.tensor(mdp_cl.GOAL_XY, device=dev)

    print("\n" + "=" * 100)
    print(f"P21 -- GATE 1: {label}  ({task})")
    print("=" * 100)

    cells, all_s, all_t, all_h, all_g, all_gap = [], [], [], [], [], []
    for prep in range(args_cli.pose_reps):
        print(f"\n   pose draw {prep}")
        ex = ClutterExpert(env, grip_z=args_cli.grip_z, lift_dz=args_cli.lift_dz)
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

            chain = ex.adapt()
            tape = Tape(e, dpos0)
            r = ex.run_physics(chain, tape)
            hz = hazards(tape, n, dev)
            s = r["success"]
            print(f"      spawn batch {rep}: enclosed {float(r['held'].float().mean()):6.1%}"
                  f" | at goal {float(r['at_goal'].float().mean()):6.1%} | topple "
                  f"{float(r['topple'].float().mean()):6.1%} | SUCCESS "
                  f"{float(s.float().mean()):6.1%}   (gap median "
                  f"{float(min_gap.median()) * 1000:.1f} mm)")
            print("         hazards: " + ", ".join(
                f"{k} {v[0]}/{v[1]}={v[2]:.0%}" for k, v in hz.items() if v[0]))
            cells.append({"pose_rep": prep, "spawn_rep": rep,
                          "o_align": ex.pose["o_align"], "pen_mm": ex.pose["pen"] * 1000,
                          "encl": float(r["held"].float().mean()),
                          "at_goal": float(r["at_goal"].float().mean()),
                          "topple": float(r["topple"].float().mean()),
                          "success": float(s.float().mean()),
                          "hazard": {k: v[2] for k, v in hz.items()}})
            all_s.append(s)
            all_t.append(r["topple"])
            all_h.append(r["held"])
            all_g.append(r["at_goal"])
            all_gap.append(min_gap)

    S, T_, H, G = (torch.cat(x) for x in (all_s, all_t, all_h, all_g))
    GAP = torch.cat(all_gap)
    N = len(S)
    rates = [c["success"] for c in cells]
    print("\n" + "-" * 100)
    print(f"   POOLED OVER {N} EPISODES  ({args_cli.pose_reps} pose draws x "
          f"{args_cli.reps} spawn batches x {args_cli.num_envs} envs)")
    print(f"      enclosed  {float(H.float().mean()):6.1%}")
    print(f"      at goal   {float(G.float().mean()):6.1%}")
    print(f"      topple    {float(T_.float().mean()):6.1%}")
    print(f"      SUCCESS   {float(S.float().mean()):6.1%}   "
          f"(per-cell {min(rates):.1%}..{max(rates):.1%}, "
          f"sd {torch.tensor(rates).std():.1%})")
    print(f"\n      {'min free gap [mm]':>18} | {'n':>5} | {'topple':>8} | {'success':>8}")
    strat = []
    for lo, hi in ((0, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 100)):
        m = (GAP * 1000 >= lo) & (GAP * 1000 < hi)
        if int(m.sum()):
            strat.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                          "topple": float(T_[m].float().mean()),
                          "success": float(S[m].float().mean())})
            print(f"      {f'{lo}-{hi}':>18} | {int(m.sum()):5d} | "
                  f"{float(T_[m].float().mean()):7.1%} | {float(S[m].float().mean()):7.1%}")
    return {"task": task, "label": label, "n": N,
            "encl": float(H.float().mean()), "at_goal": float(G.float().mean()),
            "topple": float(T_.float().mean()), "success": float(S.float().mean()),
            "cells": cells, "by_gap": strat,
            "cell_min": min(rates), "cell_max": max(rates)}


def main() -> None:
    out = [evaluate(args_cli.task, "NOMINAL (12 mm pitch)")]
    try:
        out.append(evaluate("Rebot-ClutterExtract-Tight-v0", "TIGHT (6 mm pitch)"))
    except Exception as exc:                     # noqa: BLE001
        print(f"\n   [tight] skipped: {exc}")

    print("\n" + "=" * 100)
    print("GATE 1 VERDICT")
    print("=" * 100)
    for r in out:
        print(f"   {r['label']:<24} success {r['success']:6.1%} over {r['n']} episodes "
              f"(cells {r['cell_min']:.1%}..{r['cell_max']:.1%})")
    nom = out[0]["success"]
    if nom >= 0.85:
        print("\n   -> Gate 1 PASSES at the pre-registered 85 %. Proceed to Stage 2.")
    elif nom >= 0.70:
        print("\n   -> Gate 1 clears the revised 70 % floor but not 85 %. Per DR2, proceed")
        print("      to Stage 2 with the BC ceiling recorded in advance.")
    else:
        print("\n   -> Below the 70 % floor. Per DR3, return to diagnosis rather than")
        print("      spend a demo chain on this expert.")
    if len(out) > 1:
        d = (out[1]["success"] - nom) * 100
        print(f"\n   Tight vs nominal: {d:+.1f} points. Prediction registered in Stage 0 was")
        print(f"   'within 10 points' -- {'CONFIRMED' if abs(d) <= 10 else 'REFUTED'}.")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump({"grip_z": args_cli.grip_z, "lift_dz": args_cli.lift_dz,
                   "pose_reps": args_cli.pose_reps, "reps": args_cli.reps,
                   "results": out}, f, indent=2)
    print(f"\n[p21] wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
