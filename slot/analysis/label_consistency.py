# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""How well-conditioned is the action as a function of the observation? (CPU, no simulator.)

The question
------------
Behaviour cloning fits ``a = f(obs)``. If two frames drawn from *different* demos have nearly
identical 34-D observations but materially different recorded actions, no deterministic policy
can fit both, and the achievable regression error is bounded below by that disagreement. That
lower bound is a property of the **data**, not of the model, and it is worth knowing before
reading a training result — otherwise a policy that has already hit the floor looks like a
policy that has failed to converge.

There is a specific reason to suspect it here. The arm's null space makes the demos multimodal:
the same task-space pose is reachable at joint configurations differing by up to 133 mrad, and
the expert's IK picks a branch by warm start rather than by any property of the observation. If
that branch choice is *not* predictable from the 34-D observation, it is irreducible label noise.

But the observation carries ``joint_pos_rel`` (dims 0:8) — the arm's own configuration — so the
branch is in fact observable, and the policy *should* be able to condition on it. The measurement
below separates those two cases by running the same analysis twice: once on the full
observation, and once on the environment-state half only (dims 16:34 — block pose, slot error,
last action), which is what the policy would see if proprioception carried no branch information.

Method
------
1. Subsample frames uniformly across all demos.
2. Normalize observations by the per-dimension std used in training (so the neighbour metric is
   the one the network effectively sees).
3. For each query frame, find its nearest neighbour **from a different demo** — same-demo
   neighbours are trivially close in both obs and action and would swamp the statistic.
4. Report the action disagreement as a function of observation distance. The intercept as
   obs-distance goes to 0 is the label-noise floor.

Also reports the disagreement in **millimetres of commanded TCP motion** where possible, since
"0.05 action units" is not a quantity anyone can reason about. Arm actions are joint position
targets with ``scale=0.5, use_default_offset=True``, so an action delta of d maps to a joint
target delta of 0.5*d radians.

.. code-block:: bash

    python slot/analysis/label_consistency.py slot/data/v2/nominal_s*.hdf5 --samples 40000
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slot_act.dataset import ACTION_DIM, ENV_STATE_SLICE, OBS_DIM  # noqa: E402

ARM_ACTION_SCALE = 0.5


def load_frames(paths: list[Path], samples: int, rng: np.random.Generator, successful_only=True):
    """-> obs (N, 34), act (N, 7), demo_id (N,), t (N,)."""
    obs_l, act_l, did_l, t_l = [], [], [], []
    d = 0
    for p in paths:
        with h5py.File(str(p), "r") as f:
            g = f["data"]
            for k in sorted(g.keys(), key=lambda s: int(s.split("_")[-1])):
                if successful_only and not bool(g[k].attrs["success"]):
                    continue
                obs_l.append(np.asarray(g[k]["obs/policy"], dtype=np.float32))
                act_l.append(np.asarray(g[k]["actions"], dtype=np.float32))
                T = obs_l[-1].shape[0]
                did_l.append(np.full(T, d, np.int32))
                t_l.append(np.arange(T, dtype=np.int32))
                d += 1
    obs = np.concatenate(obs_l)
    act = np.concatenate(act_l)
    did = np.concatenate(did_l)
    t = np.concatenate(t_l)
    print(f"  {d} successful demos, {len(obs):,} frames")
    if samples < len(obs):
        idx = rng.choice(len(obs), samples, replace=False)
        obs, act, did, t = obs[idx], act[idx], did[idx], t[idx]
        print(f"  subsampled to {len(obs):,} frames")
    return obs, act, did, t


def analyse(tag: str, feat: np.ndarray, act: np.ndarray, did: np.ndarray, t: np.ndarray) -> None:
    """Nearest cross-demo neighbour in `feat`; report action disagreement vs feature distance."""
    from scipy.spatial import cKDTree

    std = feat.std(axis=0)
    std[std < 1e-6] = 1.0
    z = feat / std
    tree = cKDTree(z)

    # k neighbours, then take the first that comes from a different demo. 24 is comfortably
    # more than the number of same-demo frames that can be nearest at this subsampling rate.
    k = 24
    dist, nn = tree.query(z, k=k, workers=-1)
    same = did[nn] == did[:, None]
    dist = np.where(same, np.inf, dist)
    j = dist.argmin(axis=1)
    rows = np.arange(len(z))
    d_obs = dist[rows, j]
    partner = nn[rows, j]
    ok = np.isfinite(d_obs)
    d_obs, partner = d_obs[ok], partner[ok]
    src = rows[ok]

    d_arm = np.abs(act[src, :6] - act[partner, :6]).max(axis=1)
    d_grip = act[src, 6] != act[partner, 6]
    dt = np.abs(t[src] - t[partner])

    print(f"\n  --- {tag} ---")
    print(f"  {len(src):,} query frames with a cross-demo neighbour")
    print(f"  neighbour distance (normalized, per-dim std units): "
          f"p10 {np.percentile(d_obs, 10):.3f}  median {np.percentile(d_obs, 50):.3f}  "
          f"p90 {np.percentile(d_obs, 90):.3f}")
    print(f"  |t_query - t_neighbour|: median {np.median(dt):.0f} steps "
          f"(large values would mean the metric is matching across phases)")

    # Bin by observation distance. The lowest bin is the label-noise floor: frames the network
    # genuinely cannot tell apart.
    qs = np.percentile(d_obs, [0, 5, 20, 50, 80, 100])
    print(f"\n  {'obs-dist bin':>22} {'n':>7} {'|dArm| p50':>11} {'p90':>8} {'max':>8} "
          f"{'-> p50 mrad':>12} {'grip flip':>10}")
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (d_obs >= lo) & (d_obs <= hi)
        if not m.any():
            continue
        p50 = np.percentile(d_arm[m], 50)
        print(f"  [{lo:>8.3f},{hi:>8.3f}] {m.sum():>7,} {p50:>11.4f} "
              f"{np.percentile(d_arm[m], 90):>8.4f} {d_arm[m].max():>8.4f} "
              f"{1000 * ARM_ACTION_SCALE * p50:>12.1f} {100 * d_grip[m].mean():>9.2f}%")

    floor = np.percentile(d_arm[d_obs <= qs[1]], 50)
    print(f"\n  LABEL-NOISE FLOOR (median |dAction| among the 5 % most-similar observation "
          f"pairs):\n    {floor:.4f} action units = {1000 * ARM_ACTION_SCALE * floor:.1f} mrad "
          f"of commanded joint target")
    print(f"  for scale: the ACTION std over this data is "
          f"{act[:, :6].std(axis=0).mean():.4f} units, so the floor is "
          f"{100 * floor / act[:, :6].std(axis=0).mean():.1f} % of the signal")


def main() -> int:
    ap = argparse.ArgumentParser(description="Label-consistency analysis of the demo pools.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--samples", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    paths: list[Path] = []
    for p in a.paths:
        paths.extend(Path(x) for x in (glob.glob(p) if any(c in p for c in "*?[") else [p]))
    paths = sorted(set(paths))
    print(f"\n{len(paths)} pool(s)")

    rng = np.random.default_rng(a.seed)
    obs, act, did, t = load_frames(paths, a.samples, rng)
    assert obs.shape[1] == OBS_DIM and act.shape[1] == ACTION_DIM

    # Full observation: the policy sees proprioception, so a null-space branch IS observable.
    analyse("FULL 34-D observation (what the policy actually sees)", obs, act, did, t)
    # Environment state only: what would be left if proprioception carried no branch info.
    analyse("ENV-STATE only, dims 16:34 (block pose + slot error + last action)",
            obs[:, ENV_STATE_SLICE], act, did, t)

    print("\n  Reading this: if the FULL-observation floor is much lower than the ENV-STATE-only\n"
          "  floor, the null-space branch is recoverable from proprioception and BC can fit it.\n"
          "  If the two are similar and both large, the labels are genuinely ambiguous and a\n"
          "  deterministic regressor cannot beat that floor -- which is the case flow matching\n"
          "  exists to handle, since x0 lets one observation map to a distribution of chunks.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
