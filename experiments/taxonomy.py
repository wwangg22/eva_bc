#!/usr/bin/env python
"""Failure-taxonomy classifier over eval_act.py per-episode JSON records.

Buckets (thresholds per POSTMORTEM §3: "lifted" = max z > 0.06 m; table rest ~0.02):
  success              placed_final >= 2
  placed1_stuck_low    placed_final == 1, unplaced can never lifted (miss->freeze on can 2)
  placed1_stuck_lift   placed_final == 1, unplaced can reached carry height, never landed
  never_lifted         placed_final == 0, no can lifted (missed from the start)
  lifted_never_placed  placed_final == 0, >=1 can lifted, zero delivered (mid-carry loss)
  dropped_after_place  placed_max > placed_final (post-placement drop)

Usage: python taxonomy.py eval1.json [eval2.json ...]
       python taxonomy.py --diff base.json other.json   (adds per-episode transition diff)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

LIFT_Z = 0.06


def classify(ep: dict) -> str:
    pf, pm = ep["placed_final"], ep["placed_max"]
    zs = ep["max_can_z"]
    if pm > pf:
        return "dropped_after_place"
    if pf >= 2:
        return "success"
    lifted = sum(z > LIFT_Z for z in zs)
    if pf == 1:
        # both cans exceed LIFT_Z iff the unplaced one was also lifted
        return "placed1_stuck_lift" if lifted >= 2 else "placed1_stuck_low"
    return "lifted_never_placed" if lifted >= 1 else "never_lifted"


def report(path: str) -> dict[int, str]:
    data = json.loads(open(path).read())
    eps = data["per_episode"]
    out = {}
    buckets = defaultdict(list)
    for i, ep in enumerate(eps):
        b = classify(ep)
        out[i] = b
        buckets[b].append(i)
    n = len(eps)
    succ = len(buckets["success"])
    print(f"\n{path}: {succ}/{n} = {succ / n:.1%}  "
          f"(n_action_steps={data.get('config', {}).get('n_action_steps_effective', '?')})")
    for b in ["placed1_stuck_low", "placed1_stuck_lift", "never_lifted",
              "lifted_never_placed", "dropped_after_place"]:
        if buckets[b]:
            print(f"  {b:22s} {len(buckets[b]):2d}  eps={buckets[b]}")
    return out


def main():
    args = [a for a in sys.argv[1:] if a != "--diff"]
    diff = "--diff" in sys.argv
    maps = [report(p) for p in args]
    if diff and len(maps) == 2:
        a, b = maps
        fixed = [i for i in a if a[i] != "success" and b[i] == "success"]
        broken = [i for i in a if a[i] == "success" and b[i] != "success"]
        print(f"\ntransitions {args[0]} -> {args[1]}:")
        print(f"  fixed  {len(fixed):2d}  {fixed}")
        print(f"  broken {len(broken):2d}  {broken}")
        trans = defaultdict(list)
        for i in a:
            if a[i] != b[i]:
                trans[(a[i], b[i])].append(i)
        for (x, y), eps in sorted(trans.items()):
            print(f"  {x:>22s} -> {y:22s} {eps}")


if __name__ == "__main__":
    main()
