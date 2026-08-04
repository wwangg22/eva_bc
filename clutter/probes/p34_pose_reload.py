# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P34 -- Does the frozen pose load back as the same pose? And what is it worth, really?

Why this exists
---------------
P33 selected a grasp pose by measurement and verified it at 75.4 % on 512 fresh episodes.
Stage 2 will generate every demo from that one chain, so the pose now has to survive a round
trip through `expert/pose_p33.json` and `ClutterExpert(pose_q=...)` -- a code path that runs
`_score_q` (forward kinematics against the live scene) instead of the CEM.

A load path that silently differs from the solve path would put a quiet bias into every demo
in the dataset, and it would be invisible: the expert would still print a plausible `o_align`
and still succeed most of the time. So this is a positive control, not a formality.

What run 1 found (2026-08-03) -- the reason this probe now has three arms
------------------------------------------------------------------------
Run 1 loaded the pose and replayed P33's own verification seeds. The grasp pose came back
exact (o_align identical to 12 digits, penetration 0.0, wrist to 1e-3 mm) and the score did
NOT: **74.2 % against P33's 75.4 %**, on episodes that should have been bit-identical.

The cause is not the load path. It is that `plan()` builds the other 22 waypoints with
`_dense`, which calls `_solve(..., restarts=1, std0=0.08)` -- **a CEM**. `ArmKin.cem` draws
`torch.randn` on every call, so the chain is DRAWN, not derived:

    freezing the grasp pose does NOT freeze the manoeuvre.

For an expert whose whole job is to generate a consistent dataset, that is a defect, and it
was invisible to every experiment run so far because none of them ever replayed a batch.
`ClutterExpert` gained `chain_seed=` (with the global RNG saved and restored around the
block); this probe now MEASURES whether that actually made it deterministic instead of
assuming it.

Three arms
----------
    REPEAT      the same 4 seeds, run TWICE with the same chain_seed, in one process.
                The two passes must be identical episode-for-episode. This is the direct
                test of the fix and it does not depend on P33 at all.

    REPLAY      the 4 verification batches P33 used (seeds 91000-91003).
                Compared against P33's 75.4 % -- but see prediction 1: an exact match is NOT
                expected any more, because P33's chain came from an unseeded draw that cannot
                be recovered. What is expected is that the gap is small.

    FRESH       4 batches never used for anything (seeds 77000-77003).
                An independent estimate. P33's verification was itself the second look at
                this pose, so a third, on untouched spawns, is what should be quoted.

Registered predictions
----------------------
1. **REPEAT is bit-identical between the two passes.** *Falsifier: any difference*, which
   would mean `chain_seed` does not control all the randomness in `plan()` and the expert
   still cannot produce a reproducible dataset.
   (Note the retraction: run 1's prediction that REPLAY would match P33 exactly was wrong in
   premise -- P33's chain was unseeded and is not recoverable. Chain-to-chain variation of
   ~1 point is now an expected quantity, not a bug, and REPEAT is what tests the fix.)
2. FRESH lands within +/- 5 points of 75.4 % (the binomial se at 512 episodes is ~1.9 pts,
   plus batch-difficulty spread which P33 showed to be ~4 pts peak-to-peak).
3. |REPLAY - 75.4| <= 3 points: chain-draw variation is small next to the 12-point pose sd.

Usage
-----
    python eva_bc/clutter/probes/p34_pose_reload.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Frozen-pose reload control.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--pose", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/expert/pose_p33.json")
parser.add_argument("--replay-seeds", type=str, default="91000,91001,91002,91003")
parser.add_argument("--fresh-seeds", type=str, default="77000,77001,77002,77003")
parser.add_argument("--chain-seed", type=int, default=1234)
parser.add_argument("--dump-chain", action="store_true",
                    help="freeze the dense chain into the pose file if it has none yet")
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p34_reload.json")
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
from _kin import _t  # noqa: E402,F401
from clutter_expert import ClutterExpert  # noqa: E402


def settle(e, k=30):
    for _ in range(k):
        e.sim.step()
        e.scene.update(e.physics_dt)


def main() -> None:
    spec = json.load(open(args_cli.pose))
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    settle(e)

    print("\n" + "=" * 100)
    print("P34 -- FROZEN-POSE RELOAD CONTROL")
    print("=" * 100)
    print(f"   pose: {spec['name']} from {spec['source']}")
    ref = spec["measured"]["verified_score"]
    print(f"   P33 verified it at {ref:.1%} over "
          f"{spec['measured']['verified_episodes']} episodes")

    if args_cli.dump_chain and "chain" not in spec:
        seed_ex = ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"],
                                chain_seed=args_cli.chain_seed, verbose=False)
        spec["chain"] = {
            "pts": [[float(x) for x in p] for p in seed_ex.pts],
            "qs": [[float(x) for x in q] for q in seed_ex.qs],
            "i_grip": int(seed_ex.i_grip),
            "note": ("Frozen because the CEM that derives it is not bit-reproducible even "
                     "under a fixed seed -- P34 measured max|dq| 0.109 rad between two "
                     "identically seeded builds. Dumped from one build; the REPLAY/FRESH "
                     "arms of P34 are what certify it."),
        }
        with open(args_cli.pose, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"\n   DUMPED chain ({len(spec['chain']['qs'])} waypoints) -> {args_cli.pose}")

    chain = spec.get("chain")
    ex = ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"],
                       chain_seed=args_cli.chain_seed, chain=chain, verbose=True)
    st = spec["pose_stats"]
    print(f"\n   RELOADED pose statistics vs the file:")
    for k, got, want in (("o_align", ex.pose["o_align"], st["o_align"]),
                         ("pen_mm", ex.pose["pen"] * 1000, st["pen_mm"]),
                         ("wrist_y_mm", float(ex.pose["wrist"][1]) * 1000, st["wrist_y_mm"]),
                         ("wrist_z_mm", float(ex.pose["wrist"][2]) * 1000, st["wrist_z_mm"])):
        d = abs(got - want)
        print(f"      {k:12} {got:12.6f}   file {want:12.6f}   delta {d:.2e}"
              f"   {'OK' if d < 1e-3 else '*** MISMATCH ***'}")

    def sweep(arm, ss, expert):
        print(f"\n   {arm} -- seeds {ss}")
        cells = []
        for s in ss:
            env.reset(seed=s)
            settle(e)
            r = expert.run_physics(expert.adapt())
            c = {"seed": s, "encl": float(r["held"].float().mean()),
                 "at_goal": float(r["at_goal"].float().mean()),
                 "topple": float(r["topple"].float().mean()),
                 "success": float(r["success"].float().mean()),
                 "succ_mask": r["success"].tolist()}
            cells.append(c)
            print(f"      seed {s}: encl {c['encl']:6.1%} | goal {c['at_goal']:6.1%} | "
                  f"topple {c['topple']:6.1%} | SUCCESS {c['success']:6.1%}")
        m = sum(c["success"] for c in cells) / len(cells)
        print(f"      {arm} MEAN {m:.1%} over {len(ss) * args_cli.num_envs} episodes")
        return {"seeds": ss, "cells": cells, "mean": m, "n": len(ss) * args_cli.num_envs}

    out = {}
    rep_seeds = [int(x) for x in args_cli.replay_seeds.split(",")]
    out["REPLAY"] = sweep("REPLAY", rep_seeds, ex)

    # REPEAT: a SECOND expert built from the same file and the same chain_seed. If the seed
    # controls everything, this is the same manoeuvre and the episode masks match exactly.
    ex2 = ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"],
                        chain_seed=args_cli.chain_seed, chain=chain, verbose=False)
    dq = max(float((a - b).abs().max()) for a, b in zip(ex.qs, ex2.qs))
    print(f"\n   REPEAT -- rebuilt expert, chain_seed={args_cli.chain_seed}")
    print(f"      max |dq| over all {len(ex.qs)} chain waypoints = {dq:.3e} rad")
    out["REPEAT"] = sweep("REPEAT", rep_seeds, ex2)
    flips = sum(int(a != b) for c1, c2 in zip(out["REPLAY"]["cells"], out["REPEAT"]["cells"])
                for a, b in zip(c1["succ_mask"], c2["succ_mask"]))
    out["REPEAT"]["chain_max_dq"] = dq
    out["REPEAT"]["flips_vs_replay"] = flips

    out["FRESH"] = sweep("FRESH", [int(x) for x in args_cli.fresh_seeds.split(",")], ex)

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"   REPEAT  {out['REPEAT']['mean']:6.1%}  vs REPLAY {out['REPLAY']['mean']:6.1%}"
          f"   chain max|dq| {out['REPEAT']['chain_max_dq']:.3e} rad"
          f"   episode flips {out['REPEAT']['flips_vs_replay']}")
    det = out["REPEAT"]["flips_vs_replay"] == 0 and out["REPEAT"]["chain_max_dq"] == 0.0
    print(f"   PREDICTION 1 (REPEAT bit-identical): {'HELD' if det else 'NOT HELD'}")
    if not det:
        print("      *** chain_seed does NOT make plan() reproducible. The expert cannot")
        print("      *** generate a consistent dataset until this is explained.")

    d1 = (out["REPLAY"]["mean"] - ref) * 100
    print(f"\n   REPLAY {out['REPLAY']['mean']:6.1%}  vs P33's {ref:6.1%}   delta {d1:+.2f} pts")
    print(f"   PREDICTION 3 (|delta| <= 3): {'HELD' if abs(d1) <= 3 else 'NOT HELD'}"
          f"   (P33's chain was unseeded and is not recoverable; this is chain-draw spread)")
    d2 = (out["FRESH"]["mean"] - ref) * 100
    print(f"\n   FRESH  {out['FRESH']['mean']:6.1%}  vs P33's {ref:6.1%}   delta {d2:+.2f} pts")
    print(f"   PREDICTION 2 (within +/-5): {'HELD' if abs(d2) <= 5 else 'NOT HELD'}")
    pooled = (out["REPLAY"]["mean"] * out["REPLAY"]["n"]
              + out["FRESH"]["mean"] * out["FRESH"]["n"]) / (out["REPLAY"]["n"] + out["FRESH"]["n"])
    print(f"\n   BEST ESTIMATE for this pose, quoting FRESH only (never-seen spawns): "
          f"{out['FRESH']['mean']:.1%}")
    print(f"   (pooled over replay+fresh, which double-counts P33's batches: {pooled:.1%})")

    with open(args_cli.out, "w") as f:
        json.dump({"args": vars(args_cli), "pose": spec, "arms": out}, f)
    print(f"\n   wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
