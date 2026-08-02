#!/usr/bin/env python
"""EXP07 gate 1: assert a z=0 eval_steer JSON matches an x0-zeros base JSON
episode-for-episode (success, length, placed counts, max can heights).
Exit 0 = bit-exact-equivalent; exit 1 = mismatch (chain must abort)."""
import json
import sys

FIELDS = ["success", "length", "placed_final", "placed_max", "max_can_z"]


def main():
    steer_f, base_f = sys.argv[1], sys.argv[2]
    steer = {r["episode"]: r for r in json.load(open(steer_f))["per_episode"]}
    base = {r["episode"]: r for r in json.load(open(base_f))["per_episode"]}
    if set(steer) != set(base):
        print(f"MISMATCH: episode sets differ ({len(steer)} vs {len(base)})")
        sys.exit(1)
    bad = 0
    for e in sorted(steer):
        diffs = [f for f in FIELDS if steer[e][f] != base[e][f]]
        if diffs:
            bad += 1
            print(f"ep {e}: differs in {diffs}: "
                  f"{[(steer[e][f], base[e][f]) for f in diffs]}")
    if bad:
        print(f"GATE 1 FAIL: {bad}/{len(steer)} episodes differ ({steer_f} vs {base_f})")
        sys.exit(1)
    sr = sum(r["success"] for r in steer.values()) / len(steer)
    print(f"GATE 1 PASS: {len(steer)} episodes identical, success {sr:.3f} ({steer_f})")


if __name__ == "__main__":
    main()
