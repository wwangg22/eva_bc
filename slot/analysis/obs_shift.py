# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Read back EXP_STEER §8d -- which observation channel stops looking like training data?

A perturbation costing 83 points has to reach the policy somehow. `--obs-noise 0.05` is injected
straight into the observation and costs 4.2 points (null); `--action-noise 0.02` enters the
physics and costs 83. The difference must show up as a distribution shift in whatever the
simulator hands back, so `eval_act.py` now records, per channel, the runtime mean/std of the
observation and its mean |z| against the checkpoint's own training normaliser.

`mean_abs_z` is the statistic. For a channel whose runtime distribution matches training,
E|z| = 0.798 (a standard normal). Much above that means the policy is being shown values it was
never trained on -- and the hypothesis (belief 9) is that `joint_vel_rel` is the one that blows
up, because the action is an absolute joint-position target: the drive filters target jitter out
of position and differentiates it into velocity.

.. code-block:: bash

    python analysis/obs_shift.py --a runs/bc_armB_seed0/shift_clean.json \
                                --b runs/bc_armB_seed0/shift_act002.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# 34-D layout, from precision_slot_env_cfg.py ObservationsCfg.PolicyCfg
GROUPS = [
    ("joint_pos_rel", 0, 8),
    ("joint_vel_rel", 8, 16),
    ("block_pose", 16, 23),
    ("slot_frame", 23, 27),
    ("last_action", 27, 34),
]
EXP_ABS_Z = 0.7979  # E|z| for a standard normal -- what an in-distribution channel looks like


def load(p: Path) -> dict:
    r = json.loads(p.read_text())
    if "obs_dist" not in r:
        raise SystemExit(f"{p} predates the obs_dist diagnostic -- re-run this cell")
    d = r["obs_dist"]
    d["label"] = p.stem
    d["success"] = r.get("success_rate_later")
    d["action_noise"] = r.get("config", {}).get("action_noise")
    d["obs_noise"] = r.get("config", {}).get("obs_noise")
    return d


def gmean(d: dict, key: str, lo: int, hi: int) -> float:
    v = d[key][lo:hi]
    return sum(v) / len(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="reference cell (clean)")
    ap.add_argument("--b", required=True, help="perturbed cell")
    args = ap.parse_args()
    a, b = load(Path(args.a)), load(Path(args.b))

    print(f"\n  A = {a['label']}   success_later={a['success']}   "
          f"obs_noise={a['obs_noise']} action_noise={a['action_noise']}   n={a['n_samples']}")
    print(f"  B = {b['label']}   success_later={b['success']}   "
          f"obs_noise={b['obs_noise']} action_noise={b['action_noise']}   n={b['n_samples']}")
    print(f"\n  mean |z| against the checkpoint's training normaliser "
          f"(in-distribution reference = {EXP_ABS_Z:.3f})\n")
    print(f"  {'group':<15} {'A':>8} {'B':>8} {'B - A':>9} {'B / A':>8}   "
          f"{'runtime std B / train std':>26}")
    print("  " + "-" * 76)
    rows = []
    for name, lo, hi in GROUPS:
        za, zb = gmean(a, "mean_abs_z", lo, hi), gmean(b, "mean_abs_z", lo, hi)
        # how much wider the channel's runtime spread is than the training spread
        widen = sum(b["runtime_std"][i] / t for i, t in enumerate(b["train_std"])
                    if lo <= i < hi and t > 0) / (hi - lo)
        rows.append((name, za, zb, zb - za, zb / za if za else float("inf"), widen))
        print(f"  {name:<15} {za:8.3f} {zb:8.3f} {zb - za:+9.3f} {rows[-1][4]:8.2f}"
              f" {widen:26.2f}")

    worst = max(rows, key=lambda r: r[3])
    runner = sorted(rows, key=lambda r: r[3])[-2]
    print(f"\n  largest shift: {worst[0]}  (+{worst[3]:.3f})   "
          f"runner-up {runner[0]} (+{runner[3]:.3f})   ratio {worst[3] / max(runner[3], 1e-9):.2f}x")
    print(f"  belief 9 (joint_vel is worst, by >2x the next group): "
          f"{'HELD' if worst[0] == 'joint_vel_rel' and worst[3] > 2 * runner[3] else 'FALSIFIED'}")
    jp = next(r for r in rows if r[0] == "joint_pos_rel")
    jv = next(r for r in rows if r[0] == "joint_vel_rel")
    print(f"  belief 10 (joint_pos stays under 1.5 and below joint_vel): "
          f"{'HELD' if jp[2] < 1.5 and jp[2] < jv[2] else 'FALSIFIED'}"
          f"  (joint_pos |z|={jp[2]:.3f}, joint_vel |z|={jv[2]:.3f})")

    print("\n  worst individual channels (B - A)")
    per = sorted(range(len(a["mean_abs_z"])),
                 key=lambda i: b["mean_abs_z"][i] - a["mean_abs_z"][i], reverse=True)[:6]
    for i in per:
        g = next(nm for nm, lo, hi in GROUPS if lo <= i < hi)
        print(f"    obs[{i:2d}] {g:<15} |z| {a['mean_abs_z'][i]:6.3f} -> {b['mean_abs_z'][i]:6.3f}"
              f"   runtime std {a['runtime_std'][i]:.4g} -> {b['runtime_std'][i]:.4g}"
              f"   (train {b['train_std'][i]:.4g})")
    print()


if __name__ == "__main__":
    main()
