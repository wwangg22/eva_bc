#!/usr/bin/env python
"""EXP07 s1 failure analysis: trained x0-steering vs fixed-x0-zeros base, per episode.

Both sides are deterministic on identical spawn suites (base: x0 zeros; steer:
z = clamp(mu), x0 = tanh(z)), so per-episode diffs are causal, not churn.
Reports per suite: success, base->steer transition matrix (fixed / broken lists
per bucket), steer-only failure buckets, and z-magnitude stats for
success-vs-failure episodes (belief 3: state dependence)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from taxonomy import classify

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"

PAIRS = {
    "seed42": (RUNS / "exp06_residual" / "x0sweep_s-1_seed42.json",
               RUNS / "exp07_steer" / "s1_best_seed42.json"),
    "seed123": (RUNS / "exp06_residual" / "x0sweep_s-1_seed123.json",
                RUNS / "exp07_steer" / "s1_best_seed123.json"),
}


def load(path):
    d = json.load(open(path))
    return d, {r["episode"]: r for r in d["per_episode"]}


def main():
    tot_b = tot_s = tot_n = 0
    for suite, (base_f, steer_f) in PAIRS.items():
        db, base = load(base_f)
        ds, steer = load(steer_f)
        n = len(base)
        tot_n += n
        sb = sum(r["success"] for r in base.values())
        ss = sum(r["success"] for r in steer.values())
        tot_b += sb
        tot_s += ss
        print(f"\n=== {suite}: base {sb}/{n} ({sb/n:.1%})  ->  steer {ss}/{n} ({ss/n:.1%}) ===")

        fixed = [e for e in base if not base[e]["success"] and steer[e]["success"]]
        broken = [e for e in base if base[e]["success"] and not steer[e]["success"]]
        print(f"fixed {len(fixed)}: {fixed}")
        print(f"broken {len(broken)}: {broken}")

        print("base-bucket -> steer outcome for every base failure:")
        trans = defaultdict(Counter)
        for e in base:
            if base[e]["success"]:
                continue
            bb = classify(base[e])
            rb = "SUCCESS" if steer[e]["success"] else classify(steer[e])
            trans[bb][rb] += 1
        for bb, c in sorted(trans.items()):
            print(f"  {bb:22s} -> {dict(c)}")
        print("steer-only failure buckets (incl. newly broken):")
        sc = Counter(classify(steer[e]) for e in steer if not steer[e]["success"])
        print(f"  {dict(sc)}")

        ms = [r["mean_z_mag"] for r in steer.values() if r["success"]]
        mf = [r["mean_z_mag"] for r in steer.values() if not r["success"]]
        if ms and mf:
            print(f"mean |z|: success eps {sum(ms)/len(ms):.4f}  "
                  f"fail eps {sum(mf)/len(mf):.4f}")
        vs = [r["z_std"] for r in steer.values() if r["success"]]
        vf = [r["z_std"] for r in steer.values() if not r["success"]]
        if vs and vf:
            print(f"z_std (within-episode): success eps {sum(vs)/len(vs):.4f}  "
                  f"fail eps {sum(vf)/len(vf):.4f}")

    print(f"\n=== POOLED: base {tot_b}/{tot_n} ({tot_b/tot_n:.1%})  ->  "
          f"steer {tot_s}/{tot_n} ({tot_s/tot_n:.1%}) ===")
    print("references: fixed-x0-zeros base 55.5% | +5-pt rule 60.5% | "
          "stochastic base 64.1% | Gate 6 target 90%")


if __name__ == "__main__":
    main()
