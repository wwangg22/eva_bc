#!/usr/bin/env python
"""Read the gate-B theta sweep and report where the arm stops being able to do the task.

theta_max is defined as the largest |angle| whose cell holds >= 0.95 seated AND whose
neighbours below it also hold -- a single cell passing past a failing one is a fluke, not a
range. The failure *kind* is printed beside the rate because the two kinds call for different
fixes: `grip` means the wrist ran out of travel and dropped the block, `depth`/`seat` mean the
traverse or the push missed.

    python slot/analysis/theta_sweep.py logs/theta_sweep
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_report import wilson  # noqa: E402

BAR = 0.95


def main(root: str) -> None:
    rows = []
    for f in sorted(Path(root).glob("theta_*.json")):
        d = json.loads(f.read_text())
        r = d["stats"]["result"]
        rows.append((float(d["args"]["slot_yaw"][0]), d["n"], r))
    if not rows:
        print(f"no theta_*.json under {root}")
        return
    rows.sort(key=lambda x: x[0])

    print(f"  {'theta':>8} {'deg':>7} {'seated':>10} {'rate':>7} {'Wilson':>16} {'steps':>9} "
          f"{'depth':>7} {'lat':>7}   failures")
    invalid = []
    for th, n, r in rows:
        lo, hi = wilson(r["seated"], n)
        # An end-of-episode metric is meaningless in an env that reset: the block has been
        # re-randomised. theta = -0.7 spent 603 of 600 steps, reset all 128 envs, and reported
        # "yawed 128/128" -- which reads exactly like a wrist-travel limit and was nothing of
        # the sort. So validity is checked BEFORE the failure attribution is believed.
        over = r["steps_used"] >= r["step_budget"]
        bad_cell = over or r["resets"] > 0
        fails = ", ".join(f"{k} {r['fail_' + k]}" for k in ("depth", "yaw", "seat", "grip")
                          if r["fail_" + k])
        note = ""
        if over:
            note = f"  <-- INVALID: {r['steps_used']}/{r['step_budget']} steps, all envs timed out"
            invalid.append(th)
        elif r["resets"] > 0:
            note = f"  <-- {r['resets']}/{n} envs reset mid-episode; their metrics are of a new block"
        print(f"  {th:+8.3f} {math.degrees(th):+7.1f} {r['seated']:>6}/{n:<3} {r['rate']:7.3f} "
              f"  [{lo:.3f}, {hi:.3f}] {r['steps_used']:5d}/{r['step_budget']:<3d} "
              f"{r['depth_mm']:6.1f}m {r['lat_mm']:6.2f}m   {fails or '-'}{note}")

    # A cell whose episodes timed out measured the horizon, not the arm, so it cannot count
    # either for or against theta_max.
    live = [(th, n, r) for th, n, r in rows if r["steps_used"] < r["step_budget"]]
    # |theta| collapses the two signs, and this task is NOT symmetric: the block spawns at one
    # y, so the traverse to the slot is 55 % longer for one sign of theta. A usable range is
    # symmetric, so it is bounded by the WORSE side -- taking the max over |theta| would report
    # a range that fails half the time.
    ok = {abs(th) for th, n, r in live if r["rate"] >= BAR}
    bad = {abs(th) for th, n, r in live if r["rate"] < BAR}
    ok -= bad
    # largest magnitude that passes with nothing smaller failing
    cands = [a for a in sorted(ok) if not any(b < a for b in bad)]
    theta_max = max(cands) if cands else 0.0
    print(f"\n  bar {BAR:.2f}: passing |theta| {sorted(ok)}")
    if bad:
        print(f"  failing |theta| {sorted(bad)}")
    if invalid:
        print(f"  EXCLUDED (horizon overflow, measured the budget not the arm): {sorted(invalid)}")
    print(f"  ==> theta_max = {theta_max:.3f} rad ({math.degrees(theta_max):.1f} deg)")
    if bad and min(bad) <= min(ok, default=1e9):
        print("  WARNING: a small angle failed -- this is not an envelope limit, look for a bug")
    # theta_max is defined on |theta|, which silently averages the two signs. This task is
    # asymmetric -- the block spawns at one y, so the traverse to the slot is 55 % longer for
    # one sign -- and a symmetric summary hides that entirely.
    for sgn, name in ((+1, "positive"), (-1, "negative")):
        side = [(th, r["rate"]) for th, n, r in live if th * sgn > 0]
        good = [th for th, rate in side if rate >= BAR]
        worst = min((abs(th) for th, rate in side if rate < BAR), default=None)
        lim = max((abs(t) for t in good if worst is None or abs(t) < worst), default=0.0)
        print(f"  {name:>8} side: holds to {lim:.3f} rad ({math.degrees(lim):+.1f} deg)"
              + (f", first failure at {worst:.3f}" if worst else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/theta_sweep")
