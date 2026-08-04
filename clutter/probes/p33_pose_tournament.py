# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P33 -- Stop trying to explain the pose variance. Select against it, and check the check.

The situation after P26-v4 and its meta-analysis
------------------------------------------------
Four attempts to find a forward statistic that predicts whether a grasp pose works have now
failed, and the last one failed hardest:

    o_align            P18   necessary, nowhere near sufficient      r2 = 0.004  (n=15)
    wrist height       P25   actively ANTI-correlated, -56.2 pts
    wrist side         P26 meta   +13.0 pts observational... but P32 arm +1 came back
                               57.0 % with sd 13.0 -- the within-side spread is as big
                               as everything else
    simulator screen   P26   FOUR extra CEM solves and four physical closes per selection,
                               credited with +10 points in Stage 1:     r2 = 0.016  (n=15)

Selection-level sd is **10-13 points** and nothing measured accounts for it. Meanwhile every
P26 arm used 3 selections, giving a 95 % CI of about +/-12 -- so every verdict that family
produced sat inside its own noise, including two I designed.

The change of approach
----------------------
Give up on explaining it. **Draw many poses, measure each one on held-out episodes, keep the
best.** That needs no theory of what makes a pose good, and unlike the P26 screen it scores
candidates on *the actual success condition over the actual spawn distribution* rather than on
a one-batch proxy.

The reason this is worth doing rather than merely tolerable: **Stage 2 needs ONE pose, not a
good average.** Demos are generated from a frozen chain. A 57 %-success expert that yields
clean successful episodes is as usable a data source as a 77 % one -- it just costs more
episodes to fill the dataset. What is *not* recoverable downstream is a multimodal demo
distribution: P28 found six pose clusters from eight draws, and a chunk policy that samples
across modes at every refill is the exact "mode error" failure EXP07 had to spend an RL stage
undoing. Freezing one measured-good pose is worth more to Stage 2 than three points of expert
average.

The design
----------
    K = 8 candidates, each solved on its own independent spawn draw
    SELECTION   all K evaluated on the SAME 2 seeded batches   (paired -- P26 never was)
    VERIFY      the top 2 AND the worst 1, on 4 FRESH seeded batches

Candidates are compared by restoring an object-state snapshot between runs, so within a batch
every candidate faces bit-identical spawns. This is P30's pairing discipline applied to pose
selection, which is what P26 was missing.

The verification stage is the point. Picking the best of K on a finite sample is optimistic by
construction; the only way to know by how much is to re-measure the winner on episodes that had
no vote. **Both numbers are reported, and their difference IS the selection optimism.**
Verifying the WORST candidate too turns the whole thing into a falsifiable test of the ranking
rather than a one-sided success story.

Why this will work: the variance is in the POSE, and it is stable
-----------------------------------------------------------------
P32 ran 18 poses over batch-paired spawns (pairing verified: all 12 slots bit-identical
across arms), which for the first time allows the variance to be decomposed:

    batch-0 score vs batch-1 score, across 18 poses       r = +0.812   r2 = 0.659
    within-pose sd (batch + binomial)                       5.7 pts
        of which the binomial floor at 128 envs is          4.4 pts
    TRUE pose sd, after removing within-pose noise         12.0 pts
    intraclass correlation (share of variance that is pose)  0.82

**82 % of the success variance is a property of the pose itself**, and a pose's score
reproduces on an independent spawn batch to within ~6 points -- most of which is just the
binomial floor. So there is a large, stable, and *measurable* quantity to select on, even
though nothing forward-computable predicts it. That is precisely the situation a tournament
is for.

From those numbers the expected yield, with per-candidate selection noise of ~4 points against
a true spread of 12 (shrinkage factor 0.90):

    K= 4   selection 68.3 %   verified 66.3 %   gain +11.3
    K= 8   selection 73.1 %   verified 70.4 %   gain +15.4
    K=12   selection 75.7 %   verified 72.6 %   gain +17.6

Registered predictions
----------------------
0. **`|wrist_y|` -- a lead from P32, registered here BEFORE it is tested.** Across P32's 18
   paired poses, the wrist's *lateral offset magnitude* correlates with the pose mean at
   **r = -0.804, r2 = 0.646** -- smaller |wrist_y| is better, over a range of only 20.1 to
   22.0 mm. It is the strongest predictor found in this effort by a wide margin (`o_align`
   0.059, wrist-side sign 0.048 on the same 18 poses), and it has a mechanism: with the grasp
   axis fixed at o_hat = x_hat, a larger |wrist_y| leans the forearm further into the
   neighbouring column, and the forearm is a much larger body than the finger blades P22
   measured.
   **It is also exactly the kind of finding that just died.** The wrist-side effect looked
   just as convincing observationally (+13.0 pts, Fisher p = 0.0056) and P32 refuted it at
   +2.8 +/- 8.3. So: predicted here, tested on 8 fresh candidates that had no part in
   generating it. **Prediction: r <= -0.5 on P33's candidates. Falsifier: |r| < 0.3.**
1. The winner's VERIFIED score beats the candidate mean by **>= 10 points**.
   *Falsifier: < 5 points.* That failure would be the important result -- it would mean pose
   quality is not a stable property of a pose but a pose-x-batch interaction, i.e. no frozen
   chain is good across the spawn distribution and the expert needs per-env pose adaptation.
2. Selection optimism (winner's selection score minus its verified score) is **0-8 points**.
   Expected shrinkage is small: per-candidate measurement noise at 256 episodes is ~3.1 pts
   against a true spread of ~13, so the ranking should be mostly signal.
3. The WORST candidate verifies **below** the candidate mean. *Falsifier: it verifies at or
   above the mean*, which refutes the ranking exactly as prediction 1's falsifier does.

Everything measurable about each candidate is recorded -- o_align, wrist y and z, j6, roll,
penetration, joint vector -- so these K paired, well-powered points replace the pooled P26
mess as the dataset for any future "what predicts success" question.

Usage
-----
    python eva_bc/clutter/probes/p33_pose_tournament.py --num_envs 128
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Held-out pose tournament with verification.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--cands", type=int, default=8)
parser.add_argument("--sel-batches", type=int, default=2)
parser.add_argument("--ver-batches", type=int, default=4)
parser.add_argument("--wrist-side", type=int, default=0)
parser.add_argument("--screen", type=int, default=0)
parser.add_argument("--solve-seed0", type=int, default=51000)
parser.add_argument("--sel-seed0", type=int, default=31000)
parser.add_argument("--ver-seed0", type=int, default=91000)
parser.add_argument("--out", type=str,
                    default="/home/eva/Desktop/isaacLab/eva_bc/clutter/runs/p33_tournament.json")
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
from clutter_expert import ClutterExpert, DIST  # noqa: E402


def settle(e, k=30):
    for _ in range(k):
        e.sim.step()
        e.scene.update(e.physics_dt)


def snapshot(e):
    return {k: _t(e.scene[k].data.root_state_w).clone() for k in ("target",) + DIST}


def restore(e, snap):
    for k, v in snap.items():
        e.scene[k].write_root_state_to_sim(v.clone())
    e.sim.forward()
    e.scene.update(e.physics_dt)


def fingerprint(e):
    org = e.scene.env_origins
    p = [(_t(e.scene[k].data.root_pos_w) - org)[:, :2] for k in ("target",) + DIST]
    return round(float(torch.stack(p).mul(1e4).round().sum()), 1)


def run_on_batches(e, cands, seeds, env, label):
    """Every candidate on every batch, spawn-identical within a batch. Returns per-cand cells."""
    cells = {i: [] for i, _ in cands}   # keyed by CANDIDATE ID, not position
    for b, seed in enumerate(seeds):
        env.reset(seed=seed)
        settle(e)
        snap, fp = snapshot(e), fingerprint(e)
        print(f"      {label} batch {b} (seed {seed}, fp {fp:.1f})")
        for i, ex in cands:
            restore(e, snap)
            r = ex.run_physics(ex.adapt())
            cell = {"batch": b, "seed": seed, "fp": fp,
                    "encl": float(r["held"].float().mean()),
                    "at_goal": float(r["at_goal"].float().mean()),
                    "topple": float(r["topple"].float().mean()),
                    "success": float(r["success"].float().mean()),
                    "succ_mask": r["success"].tolist()}
            cells[i].append(cell)
            print(f"         cand {i}: encl {cell['encl']:6.1%} | goal {cell['at_goal']:6.1%}"
                  f" | topple {cell['topple']:6.1%} | SUCCESS {cell['success']:6.1%}")
    return cells


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()
    n, K = e.num_envs, args_cli.cands

    print("\n" + "=" * 100)
    print("P33 -- HELD-OUT POSE TOURNAMENT, WITH THE SELECTION OPTIMISM MEASURED")
    print("=" * 100)
    print(f"   {K} candidates | select on {args_cli.sel_batches} paired batches "
          f"| verify top-2 + worst on {args_cli.ver_batches} FRESH batches | {n} envs")

    # ---- solve K candidates, each on its own independent spawn draw -------------------
    print(f"\n   SOLVING {K} CANDIDATES")
    cands, meta = [], []
    for i in range(K):
        # Seeded so the whole tournament is reproducible: `reset(seed=)` sets the global
        # torch seed, so the CEM draws that follow are deterministic too.
        env.reset(seed=args_cli.solve_seed0 + i)
        settle(e)
        ex = ClutterExpert(env, grip_z=args_cli.grip_z, screen=args_cli.screen,
                           wrist_side=args_cli.wrist_side, verbose=False)
        p = ex.pose
        m = {"cand": i, "o_align": float(p["o_align"]), "pen": float(p["pen"]),
             "wrist_y_mm": float(p["wrist"][1]) * 1000.0,
             "wrist_z_mm": float(p["wrist"][2]) * 1000.0,
             "j6": float(p["q"][5]), "roll": int(p.get("roll", 0)),
             "q": [float(x) for x in p["q"]]}
        meta.append(m)
        cands.append((i, ex))
        print(f"      cand {i}: o_align {m['o_align']:.4f} | wrist ({m['wrist_y_mm']:+6.1f},"
              f"{m['wrist_z_mm']:5.1f}) mm | j6 {m['j6']:+.3f} | roll {m['roll']:+d}"
              + ("" if m["o_align"] >= 0.99 else "   <-- BELOW THE 0.99 GATE"))

    # ---- selection --------------------------------------------------------------------
    print(f"\n   SELECTION -- all {K} candidates on the same batches")
    sel_seeds = [args_cli.sel_seed0 + b for b in range(args_cli.sel_batches)]
    sel = run_on_batches(e, cands, sel_seeds, env, "sel")
    sel_score = {i: sum(c["success"] for c in v) / len(v) for i, v in sel.items()}
    order = sorted(range(K), key=lambda i: -sel_score[i])
    cand_mean = sum(sel_score.values()) / K

    print(f"\n   SELECTION RANKING (candidate mean {cand_mean:.1%})")
    for r, i in enumerate(order):
        print(f"      {r + 1}. cand {i}: {sel_score[i]:6.1%}   o_align {meta[i]['o_align']:.4f}"
              f"   wrist_y {meta[i]['wrist_y_mm']:+6.1f} mm   j6 {meta[i]['j6']:+.3f}")

    # ---- verification: top 2 and the worst, on fresh batches --------------------------
    ver_ids = list(dict.fromkeys([order[0], order[1], order[-1]]))
    print(f"\n   VERIFICATION -- cands {ver_ids} on {args_cli.ver_batches} FRESH batches")
    ver_seeds = [args_cli.ver_seed0 + b for b in range(args_cli.ver_batches)]
    ver = run_on_batches(e, [(i, dict(cands)[i]) for i in ver_ids], ver_seeds, env, "ver")
    ver_score = {i: sum(c["success"] for c in v) / len(v) for i, v in ver.items()}

    # ---- verdict ----------------------------------------------------------------------
    w, second, worst = order[0], order[1], order[-1]
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"   {'cand':>6} {'role':>10} {'selection':>10} {'verified':>10} {'optimism':>10}")
    for i, role in ((w, "winner"), (second, "runner-up"), (worst, "worst")):
        print(f"   {i:>6} {role:>10} {sel_score[i]:10.1%} {ver_score[i]:10.1%} "
              f"{(sel_score[i] - ver_score[i]) * 100:+9.1f}")
    print(f"\n   candidate mean (selection, no selection applied)   {cand_mean:6.1%}")
    gain = (ver_score[w] - cand_mean) * 100
    opt = (sel_score[w] - ver_score[w]) * 100
    print(f"   winner VERIFIED minus candidate mean               {gain:+6.1f} pts")
    print(f"   selection optimism on the winner                   {opt:+6.1f} pts")
    print(f"\n   PREDICTION 1 (gain >= 10):  {'HELD' if gain >= 10 else 'NOT HELD'}"
          f"   falsifier (< 5): {'TRIGGERED' if gain < 5 else 'not triggered'}")
    print(f"   PREDICTION 2 (optimism 0-8): {'HELD' if 0 <= opt <= 8 else 'NOT HELD'}")
    print(f"   PREDICTION 3 (worst < mean): "
          f"{'HELD' if ver_score[worst] < cand_mean else 'NOT HELD'}"
          f"   (worst verified {ver_score[worst]:.1%} vs mean {cand_mean:.1%})")
    if gain < 5:
        print("\n   *** PREDICTION 1 FALSIFIED. Pose quality is not a stable property of a")
        print("   *** pose -- the variance is pose-x-batch interaction, and NO frozen chain")
        print("   *** generalises. Stage 2 must adapt the pose per env, not freeze one.")

    # ---- prediction 0: does |wrist_y| replicate on candidates it did not generate? -----
    xs = [abs(meta[i]["wrist_y_mm"]) for i in range(K)]
    ys = [sel_score[i] * 100 for i in range(K)]
    mx, my = sum(xs) / K, sum(ys) / K
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    rw = sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")
    print(f"\n   PREDICTION 0  |wrist_y| vs selection score:  r = {rw:+.3f}  r2 = {rw*rw:.3f}"
          f"   (n = {K}, P32 gave -0.804)")
    print(f"      range of |wrist_y| here: {min(xs):.1f} - {max(xs):.1f} mm")
    print(f"      {'HELD' if rw <= -0.5 else 'NOT HELD'}"
          f"   falsifier (|r| < 0.3): {'TRIGGERED' if abs(rw) < 0.3 else 'not triggered'}")

    with open(args_cli.out, "w") as f:
        json.dump({"args": vars(args_cli), "candidates": meta,
                   "sel_seeds": sel_seeds, "ver_seeds": ver_seeds,
                   "sel_score": {str(k): v for k, v in sel_score.items()},
                   "ver_score": {str(k): v for k, v in ver_score.items()},
                   "cand_mean": cand_mean, "order": order,
                   "sel_cells": {str(k): v for k, v in sel.items()},
                   "ver_cells": {str(k): v for k, v in ver.items()}}, f)
    print(f"\n   wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
