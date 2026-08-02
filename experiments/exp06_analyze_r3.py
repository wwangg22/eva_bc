#!/usr/bin/env python
"""EXP06 r3 failure analysis: residual vs fixed-x0-zeros base, per episode.

Both sides are deterministic on identical spawn suites (fixed x0 zeros), so
per-episode diffs are causal, not churn. Reports per suite: success, bucket
counts, base->residual transition matrix (fixed / broken lists per bucket), and
residual-magnitude stats for success-vs-failure episodes.
"""
from __future__ import annotations

import json
from pathlib import Path

from taxonomy import classify

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "exp06_residual"

PAIRS = {
    "seed42": ("x0sweep_s-1_seed42.json", "r3_best_seed42.json"),
    "seed123": ("x0sweep_s-1_seed123.json", "r3_best_seed123.json"),
}


def load(name):
    d = json.load(open(RUNS / name))
    return d, {r["episode"]: r for r in d["per_episode"]}


def main():
    tot_b = tot_r = tot_n = 0
    for suite, (base_f, res_f) in PAIRS.items():
        db, base = load(base_f)
        dr, res = load(res_f)
        n = len(base)
        tot_n += n
        sb = sum(r["success"] for r in base.values())
        sr = sum(r["success"] for r in res.values())
        tot_b += sb
        tot_r += sr
        print(f"\n=== {suite}: base {sb}/{n} ({sb/n:.1%})  ->  residual {sr}/{n} ({sr/n:.1%}) ===")

        fixed = [e for e in base if not base[e]["success"] and res[e]["success"]]
        broken = [e for e in base if base[e]["success"] and not res[e]["success"]]
        print(f"fixed {len(fixed)}: {fixed}")
        print(f"broken {len(broken)}: {broken}")

        print("base-bucket -> residual outcome for every base failure:")
        from collections import Counter, defaultdict
        trans = defaultdict(Counter)
        for e in base:
            if base[e]["success"]:
                continue
            bb = classify(base[e])
            rb = "SUCCESS" if res[e]["success"] else classify(res[e])
            trans[bb][rb] += 1
        for bb, c in sorted(trans.items()):
            print(f"  {bb:22s} -> {dict(c)}")
        print("residual-only failure buckets (incl. newly broken):")
        rc = Counter(classify(res[e]) for e in res if not res[e]["success"])
        print(f"  {dict(rc)}")

        ms = [r["mean_res_mag"] for r in res.values() if r["success"]]
        mf = [r["mean_res_mag"] for r in res.values() if not r["success"]]
        if ms and mf:
            print(f"mean |residual|/joint: success eps {sum(ms)/len(ms):.4f}  "
                  f"fail eps {sum(mf)/len(mf):.4f}  (bound 0.1)")

    print(f"\n=== POOLED: base {tot_b}/{tot_n} ({tot_b/tot_n:.1%})  ->  "
          f"residual {tot_r}/{tot_n} ({tot_r/tot_n:.1%}) ===")
    print("references: fixed-x0-zeros base 55.5% | +5-pt rule threshold 60.5% | "
          "stochastic base 64.1% | Gate 6 target 90%")


if __name__ == "__main__":
    main()
