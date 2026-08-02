#!/usr/bin/env python
"""EXP03 replica analysis: buckets, arm stats, pairwise episode churn."""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from taxonomy import classify

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
EVALS = {
    "v1": RUNS / "flow_nominal_v1/eval_gate2_64ep_diag.json",
    "v3": RUNS / "flow_dagger_v3/eval_gate2_64ep.json",
    **{n: RUNS / f"exp03_{n}/eval_64ep.json" for n in ["N1", "N2", "N3", "D1", "D2", "D3"]},
}

MISS_FREEZE = ("never_lifted", "placed1_stuck_low")
MID_CARRY = ("lifted_never_placed", "placed1_stuck_lift")


def main():
    buckets, succ = {}, {}
    for name, path in EVALS.items():
        eps = json.loads(path.read_text())["per_episode"]
        buckets[name] = [classify(e) for e in eps]
        succ[name] = sum(b == "success" for b in buckets[name])

    print(f"{'run':4s} {'succ':>7s} {'miss/freeze':>12s} {'mid-carry':>10s}  buckets")
    for name, bs in buckets.items():
        c = Counter(bs)
        mf = sum(c[b] for b in MISS_FREEZE)
        mc = sum(c[b] for b in MID_CARRY)
        print(f"{name:4s} {succ[name]:3d}/64 {mf:12d} {mc:10d}  {dict(c)}")

    print("\npairwise success-outcome churn (episodes whose success/fail flips):")
    names = list(EVALS)
    print("     " + "".join(f"{n:>5s}" for n in names))
    for a in names:
        row = ""
        for b in names:
            churn = sum((x == "success") != (y == "success")
                        for x, y in zip(buckets[a], buckets[b]))
            row += f"{churn:5d}"
        print(f"{a:4s} {row}")

    within_n = [sum((x == 'success') != (y == 'success')
                    for x, y in zip(buckets[a], buckets[b]))
                for a, b in combinations(["N1", "N2", "N3"], 2)]
    within_d = [sum((x == 'success') != (y == 'success')
                    for x, y in zip(buckets[a], buckets[b]))
                for a, b in combinations(["D1", "D2", "D3"], 2)]
    print(f"\nwithin-nominal churn: {within_n}  within-dagger churn: {within_d}")
    print(f"v1<->v3 churn (the '17 fixed/17 broken'): "
          f"{sum((x == 'success') != (y == 'success') for x, y in zip(buckets['v1'], buckets['v3']))}")


if __name__ == "__main__":
    main()
