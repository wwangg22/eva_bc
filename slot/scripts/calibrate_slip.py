# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrate the in-hand-slip loss censor against demos that actually failed, then apply it.

Why this is a separate, offline step
------------------------------------
The censor's job is to zero the loss on frames where the expert's recorded action is *wrong*
for the recorded observation -- which happens as soon as the block stops being where the
open-loop plan assumes. The obvious implementation is a fixed millimetre threshold on
``|block_pos - TCP|`` measured from the grasp. Guessing that threshold at collection time went
badly twice:

* 3.0 mm censored **100 % of demos, successes included**, because the *nominal* drift already
  reaches 4.7 mm by the end of `push` -- the block genuinely slides a few mm in the pads while
  being dragged, and swings transiently because it hangs 33 mm below the grip point;
* an earlier version measured past the release, where there is nothing in the hand, and read
  the gripper retreating 92 mm as a 92 mm "slip".

So ``collect_demos.py`` now stores the raw per-step signal as a ``slip_mm`` dataset and leaves
the censor **off**. The threshold is derived here, from data containing real failures, and can
be re-derived without ever re-running the simulator.

The statistic
-------------
Not raw slip -- **excess** slip. Nominal slip grows monotonically through the trajectory
(2.2 -> 1.8 -> 1.8 -> 3.6 -> 4.8 mm at the ends of lift/back/spin/turn/push), so any fixed
threshold on the raw value is really a threshold on *time*. Subtracting the per-timestep median
over successful demos removes that trend and leaves the part that is specific to the episode:

    excess(t) = slip(t) - median_over_successful_demos(slip(t))

The threshold is then chosen at a **low false-positive rate on successful demos**, the same
discipline eva_bc used for the grasp bit (which held a 0 % FPR gate). Censoring good data is
the expensive mistake here: the pool is the product.

This script can also conclude that slip does **not** separate outcomes, in which case the
censor should not be used at all. That verdict is a real possible outcome, not a failure.

.. code-block:: bash

    python slot/scripts/calibrate_slip.py slot/data/v2/*.hdf5
    python slot/scripts/calibrate_slip.py slot/data/v2/*.hdf5 --apply --fpr 0.02
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def load(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, list[tuple[Path, str]], dict]:
    """-> slip (N, T), success (N,), the (file, key) of each demo, and the phase boundaries."""
    import json

    slips, succ, where, segs = [], [], [], None
    for p in paths:
        with h5py.File(str(p), "r") as f:
            g = f["data"]
            for k in sorted(g.keys(), key=lambda s: int(s.split("_")[-1])):
                if "slip_mm" not in g[k]:
                    raise SystemExit(f"{p}:{k} has no slip_mm dataset -- collected before the "
                                     f"signal was stored. Re-collect or drop this file.")
                slips.append(np.asarray(g[k]["slip_mm"], dtype=np.float32))
                succ.append(bool(g[k].attrs["success"]))
                where.append((p, k))
                if segs is None:
                    segs = {s["seg"]: s["t"] for s in json.loads(g[k].attrs["segments"])}
    return np.stack(slips), np.asarray(succ), where, segs


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random failure scores higher than a random success). 0.5 = no signal."""
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate and apply the in-hand-slip censor.")
    ap.add_argument("paths", nargs="+", type=str)
    ap.add_argument("--fpr", type=float, default=0.02,
                    help="tolerated fraction of SUCCESSFUL demos that may be censored")
    ap.add_argument("--min_auc", type=float, default=0.65,
                    help="below this the signal is judged uninformative and nothing is applied")
    ap.add_argument("--apply", action="store_true", help="write train_mask back into the files")
    a = ap.parse_args()

    paths = [Path(p) for p in a.paths]
    slip, succ, where, segs = load(paths)
    n, T = slip.shape
    print(f"\n  {n} demos from {len(paths)} file(s), T = {T}   "
          f"successful {int(succ.sum())} ({100 * succ.mean():.1f} %)")
    if succ.all():
        print("\n  Every demo succeeded -- there is nothing to calibrate against. Include a pool\n"
              "  collected with noise (the DART pools contain real failures).\n")
        return 1

    # nominal profile from successful demos only, so failures cannot drag the baseline
    nominal = np.median(slip[succ], axis=0)
    excess = slip - nominal[None, :]
    print(f"\n  nominal slip profile [mm]  t=100 {nominal[100]:.2f}  t=250 {nominal[250]:.2f}  "
          f"t=400 {nominal[400]:.2f}  t=460 {nominal[460]:.2f}")

    # Candidate windows. The window matters more than the statistic, because the two halves of
    # the gripped span mean opposite things:
    #   * CARRY (grasp -> push): the block should ride rigidly in the pads. Any drift here is
    #     genuine expert error -- the open-loop plan was built for the original grasp.
    #   * PUSH (push -> release): the expert deliberately drives the block into the BACK STOP.
    #     Once it bottoms out the gripper keeps executing its remaining waypoints while the
    #     block cannot move, so the offset grows by several mm. That is the signature of a
    #     *fully seated* insert, i.e. a SUCCESS indicator.
    # Measured over the whole span the second effect dominates and inverts the statistic:
    # failed demos showed p90 6.99 mm against seated 13.33 mm. Censoring on that would have
    # deleted precisely the best demos.
    push_t = segs["push"]
    windows = {
        "whole gripped span": (0, T),
        "CARRY only (grasp -> push)": (0, push_t),
        "PUSH only (push -> release)": (push_t, T),
    }
    print(f"\n  push phase starts at t = {push_t} of {T}")
    print(f"\n  {'window':>28} {'AUC raw':>9} {'AUC excess':>11}   "
          f"{'excess p50 seated/failed':>26}")
    results = {}
    for name, (lo, hi) in windows.items():
        raw_s = slip[:, lo:hi].max(axis=1)
        exc_s = excess[:, lo:hi].max(axis=1)
        ar, ae = auc(raw_s[~succ], raw_s[succ]), auc(exc_s[~succ], exc_s[succ])
        results[name] = (ae, exc_s)
        print(f"  {name:>28} {ar:>9.3f} {ae:>11.3f}   "
              f"{np.percentile(exc_s[succ], 50):>12.2f} /{np.percentile(exc_s[~succ], 50):>10.2f}")
    print("\n  AUC is P(a random FAILURE scores higher than a random success). 0.5 = no signal;"
          "\n  below 0.5 means the statistic is a SUCCESS indicator and must not be used to censor.")

    best = max(results.items(), key=lambda kv: kv[1][0])
    name, (a_exc, score) = best
    print(f"\n  best window: {name}  (excess AUC {a_exc:.3f})")

    if not np.isfinite(a_exc) or a_exc < a.min_auc:
        print(f"\n  VERDICT: no window separates outcomes (best excess AUC {a_exc:.3f} < "
              f"{a.min_auc}).\n           Do not censor on slip -- a censor with no signal only "
              f"deletes good data.\n           This is a real answer: it says the failures are "
              f"not caused by anything\n           visible in the in-hand offset. Given that "
              f"identical actions from an\n           identical state already flip 23-25 % of "
              f"outcomes on this task, most\n           failures are simulator chaos rather than "
              f"expert error -- and chaos is\n           not maskable.\n")
        return 0

    thr = float(np.quantile(score[succ], 1.0 - a.fpr))
    caught = float((score[~succ] > thr).mean())
    print(f"\n  threshold at {100 * a.fpr:.0f} % FPR on successful demos: "
          f"excess > {thr:.2f} mm")
    print(f"    catches {100 * caught:.1f} % of failed demos, "
          f"censors {100 * (score[succ] > thr).mean():.1f} % of successful ones")

    lo, hi = windows[name]
    win = np.zeros(T, dtype=bool)
    win[lo:hi] = True
    cut = np.where((excess > thr) & win[None, :], np.arange(T)[None, :], T).min(axis=1)
    fire = cut < T
    if fire.any():
        print(f"    censor fires at t = p10 {np.percentile(cut[fire], 10):.0f}, "
              f"median {np.percentile(cut[fire], 50):.0f}, p90 {np.percentile(cut[fire], 90):.0f} "
              f"of {T}   -> keeps {100 * cut[fire].mean() / T:.0f} % of those episodes' frames")

    if not a.apply:
        print("\n  (dry run -- pass --apply to write these masks back)\n")
        return 0

    per_file: dict[Path, int] = {}
    for i, (p, k) in enumerate(where):
        if not fire[i]:
            continue
        with h5py.File(str(p), "r+") as f:
            m = np.asarray(f["data"][k]["train_mask"])
            m[cut[i]:] = 0
            f["data"][k]["train_mask"][...] = m
            f["data"][k].attrs["slip_censor_t"] = int(cut[i])
            f["data"][k].attrs["slip_excess_thr_mm"] = thr
        per_file[p] = per_file.get(p, 0) + 1
    print("\n  applied:")
    for p, c in sorted(per_file.items()):
        print(f"    {p.name:>28}  {c} demos censored")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
