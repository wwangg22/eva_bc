# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P27 -- How short can the holds be? The static-observation problem in behaviour cloning.

The problem, which is a BC problem and not an expert problem
------------------------------------------------------------
The expert's schedule, in env steps (one env step = `decimation` 8 physics steps):

    settle 10 | descend 75 | predwell 20 | CLOSE 70 | carry 102 | dwell 20 |
    release 30 | withdraw 25 | final 30                       = 382 total

**180 of those 382 steps -- 47.1 % -- are a held pose**, and 70 of them are the close alone.
Those durations were chosen for a physics-only measurement, where waiting is nearly free.

For behaviour cloning they are not free, and the reason is specific. During a hold the arm
holds one joint target, the fingers stall within the first few physics steps (P19 traced the
slam at 2.5 ms resolution), the block is gripped and still, and the neighbours are still. The
42-D observation is then **constant to numerical noise** for tens of consecutive env steps --
while the correct action chunk is **different at every one of them**, because the lift begins
at a fixed offset from the *start* of the close.

A memoryless chunk policy cannot recover phase from a constant observation. What it can do is
learn the distribution over "how many steps until the lift" -- which is exactly what a
rectified-flow head is for -- and at inference it draws **one sample** from that distribution.
So the policy will begin its lift at *some* point inside the close, not at step 70.

That is survivable if and only if lifting early is safe, i.e. if the close is not much longer
than the fingers actually need. **Nobody has ever asked how long they need.**

Design
------
Paired, as everything since P13 has been: one reset, snapshot every block's `root_state_w`,
restore between arms, one screened pose shared across all arms of a cell. Only the hold
duration changes.

    ARM 1   close   560 -> 400 -> 280 -> 200 -> 140 -> 80 -> 40 physics steps
    ARM 2   the best close from ARM 1, then predwell/dwell/release scaled 1.0 -> 0.5 -> 0.25

Scored on `enclosed AND at_goal AND NOT toppled` -- the **conjunction**. P25's most expensive
lesson: a proxy metric that omits the success condition selects the candidates that fail it.

Pre-registered prediction
-------------------------
The fingers travel 26.5 mm each at `stiffness = 2000, damping = 40`. **Predicted: flat from
560 down to about 120 physics steps (15 env steps), then enclosure falls off a cliff as the
gripper is still travelling when the lift starts.** If that holds, the close drops from 70 env
steps to ~15 and the ambiguous-label count falls by 55 steps per demo -- a 14 % cut in the
whole demo.

Falsifier: **a gradual** decline from 560 rather than a plateau-then-cliff. That would mean the
close is doing something other than closing -- most plausibly letting a disturbed row settle
before the lift -- and shortening it would trade expert success against BC learnability. That
is a real trade and would be reported as one, not resolved silently.

Usage
-----
    python -u eva_bc/clutter/probes/p27_hold_duration.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hold-duration sweep: how short can the close be?")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--pose", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/expert/pose_p33.json",
                    help="frozen pose+chain; '' re-solves per batch (Stage-1 behaviour). "
                         "MUST be set: pose-draw sd is 12 points (P32) and would swamp a "
                         "5-point pass band; with it, the sweep is a pure single variable")
parser.add_argument("--screen", type=int, default=4, help="only used when --pose is empty")
parser.add_argument("--seed0", type=int, default=66000, help="batch seeds seed0 + rep")
parser.add_argument("--closes", type=str, default="560,400,280,200,140,80,40")
parser.add_argument("--scales", type=str, default="1.0,0.5,0.25",
                    help="ARM 2: multiplier on predwell/dwell/release/final")
parser.add_argument("--reps", type=int, default=2, help="spawn batches (one pose each)")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p27_holds.json")
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
from _kin import _t  # noqa: E402
from clutter_expert import DIST, HOLDS, ClutterExpert  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    n = e.num_envs
    closes = [int(x) for x in args_cli.closes.split(",")]
    scales = [float(x) for x in args_cli.scales.split(",")]

    print("\n" + "=" * 100)
    print("P27 -- HOLD DURATION: HOW SHORT CAN THE CLOSE BE?")
    print("=" * 100)
    print("   PREDICTION (registered): flat from 560 down to ~120 physics steps, then a cliff.")
    print("   Falsifier: a GRADUAL decline, which would mean the close is settling the row,")
    print("   not closing the gripper.")

    spec = json.load(open(args_cli.pose)) if args_cli.pose else None
    if spec:
        print(f"   frozen pose {spec['name']} -- every arm and every batch shares one pose "
              f"and one chain, so the ONLY variable is the hold duration.")
    else:
        print("   *** NO frozen pose: re-solving per batch. Pose-draw sd is 12 points (P32)")
        print("   *** and will swamp this sweep. This mode is for reproducing Stage 1 only.")

    results = {"arm1": {}, "arm2": {}}
    ex = None
    for rep in range(args_cli.reps):
        env.reset(seed=args_cli.seed0 + rep)
        for _ in range(30):
            e.sim.step()
            e.scene.update(e.physics_dt)
        if ex is None or spec is None:
            ex = (ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"],
                                chain=spec.get("chain"), verbose=False) if spec else
                  ClutterExpert(env, grip_z=args_cli.grip_z, screen=args_cli.screen,
                                verbose=False))
        snap = {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}

        def restore():
            for k, v in snap.items():
                e.scene[k].write_root_state_to_sim(v.clone())
            e.sim.forward()
            e.scene.update(e.physics_dt)

        def cell(holds):
            restore()
            ex.holds = dict(HOLDS)
            ex.holds.update(holds)
            chain = ex.adapt()
            r = ex.run_physics(chain)
            steps = ex.env_steps(chain)
            return r, steps

        print(f"\n   --- batch {rep} (seed {args_cli.seed0 + rep}) | pose o_align "
              f"{ex.pose['o_align']:.4f} ---")
        print(f"   ARM 1: close duration")
        print(f"      {'close':>8} {'env steps':>10} {'demo len':>9} {'static %':>9} "
              f"{'encl':>7} {'goal':>7} {'topple':>7} {'SUCCESS':>8}")
        for c in closes:
            r, st = cell({"close": c})
            row = {"rep": rep, "close": c, "close_env_steps": c // 8,
                   "demo_len": st["TOTAL"], "static_frac": st["STATIC"] / st["TOTAL"],
                   "encl": float(r["held"].float().mean()),
                   "at_goal": float(r["at_goal"].float().mean()),
                   "topple": float(r["topple"].float().mean()),
                   "success": float(r["success"].float().mean()),
                   "succ_mask": r["success"].tolist()}
            results["arm1"].setdefault(c, []).append(row)
            print(f"      {c:>8} {c // 8:>10} {st['TOTAL']:>9} "
                  f"{row['static_frac']:>8.1%} {row['encl']:>7.1%} {row['at_goal']:>7.1%} "
                  f"{row['topple']:>7.1%} {row['success']:>8.1%}")

        # ARM 2 uses the shortest close that is still within 5 points of the 560 baseline,
        # decided per batch from THIS batch's own numbers -- reported, so the selection is
        # visible next to the held-out effect (the P26 rule).
        b = results["arm1"][closes[0]][-1]["success"]
        keep = [c for c in closes if results["arm1"][c][-1]["success"] >= b - 0.05]
        c_best = min(keep) if keep else closes[0]
        print(f"   ARM 2: other holds, with close = {c_best} "
              f"(shortest within 5 points of the {closes[0]} baseline)")
        print(f"      {'scale':>8} {'demo len':>9} {'static %':>9} "
              f"{'encl':>7} {'goal':>7} {'topple':>7} {'SUCCESS':>8}")
        for s in scales:
            h = {"close": c_best}
            for k in ("predwell", "dwell", "release", "final", "settle"):
                h[k] = max(8, int(round(HOLDS[k] * s / 8)) * 8)
            r, st = cell(h)
            row = {"rep": rep, "scale": s, "close": c_best, "holds": h,
                   "demo_len": st["TOTAL"], "static_frac": st["STATIC"] / st["TOTAL"],
                   "encl": float(r["held"].float().mean()),
                   "at_goal": float(r["at_goal"].float().mean()),
                   "topple": float(r["topple"].float().mean()),
                   "success": float(r["success"].float().mean()),
                   "succ_mask": r["success"].tolist()}
            results["arm2"].setdefault(s, []).append(row)
            print(f"      {s:>8.2f} {st['TOTAL']:>9} {row['static_frac']:>8.1%} "
                  f"{row['encl']:>7.1%} {row['at_goal']:>7.1%} {row['topple']:>7.1%} "
                  f"{row['success']:>8.1%}")
        restore()

    print("\n" + "=" * 100)
    print("SUMMARY -- pooled over batches")
    print("=" * 100)

    def pooled(cells):
        S = torch.tensor([x for c in cells for x in c["succ_mask"]], dtype=torch.float32)
        E = [c["encl"] for c in cells]
        return float(S.mean()), len(S), min(E), max(E)

    print(f"\n   ARM 1  close duration       [prediction: flat, then a cliff]")
    print(f"      {'close':>8} {'env':>5} {'demo':>6} {'SUCCESS':>9} {'n':>5} {'encl range':>14}")
    base = None
    arm1_curve = []
    for c in closes:
        m, k, e0, e1 = pooled(results["arm1"][c])
        if base is None:
            base = m
        arm1_curve.append((c, m))
        print(f"      {c:>8} {c // 8:>5} {results['arm1'][c][0]['demo_len']:>6} "
              f"{m:>8.1%} {k:>5} {e0:>6.1%}..{e1:<6.1%}  vs 560 {m - base:+.1%}")

    print(f"\n   ARM 2  the other holds")
    print(f"      {'scale':>8} {'demo':>6} {'static':>8} {'SUCCESS':>9} {'n':>5}")
    for s in scales:
        m, k, _, _ = pooled(results["arm2"][s])
        print(f"      {s:>8.2f} {results['arm2'][s][0]['demo_len']:>6} "
              f"{results['arm2'][s][0]['static_frac']:>7.1%} {m:>8.1%} {k:>5}")

    # verdict: plateau-then-cliff, or gradual?
    ok = [c for c, m in arm1_curve if m >= base - 0.05]
    shortest = min(ok) if ok else closes[0]
    drops = [abs(arm1_curve[i][1] - arm1_curve[i + 1][1]) for i in range(len(arm1_curve) - 1)]
    cliff = max(drops) if drops else 0.0
    gradual = cliff < 0.15 and (base - arm1_curve[-1][1]) > 0.15
    print("\n   " + "=" * 90)
    print(f"   shortest close within 5 points of the 560 baseline: {shortest} physics steps "
          f"({shortest // 8} env steps)")
    print(f"   largest single-step drop in the curve: {cliff:.1%}")
    if gradual:
        print("   VERDICT: GRADUAL decline -- prediction FALSIFIED. The close is doing something")
        print("   other than closing the gripper (most likely letting the row settle). Shortening")
        print("   it trades expert success against BC learnability -- report the trade, do not")
        print("   resolve it here.")
    else:
        print("   VERDICT: PLATEAU then CLIFF -- prediction holds. Adopt the shortest safe close;")
        print(f"   the demo's static fraction falls from 47.1 % to about "
              f"{results['arm1'][shortest][0]['static_frac']:.1%}.")
    print("   " + "=" * 90)

    out = {"num_envs": n, "grip_z": args_cli.grip_z, "screen": args_cli.screen,
           "reps": args_cli.reps, "closes": closes, "scales": scales,
           "shortest_safe_close": shortest, "max_drop": cliff, "gradual": bool(gradual),
           "arm1": {str(k): v for k, v in results["arm1"].items()},
           "arm2": {str(k): v for k, v in results["arm2"].items()}}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[p27] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
