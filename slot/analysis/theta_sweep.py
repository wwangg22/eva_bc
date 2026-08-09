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

    print(f"  {'theta':>8} {'deg':>7} {'seated':>10} {'rate':>7} {'Wilson':>16} "
          f"{'depth':>7} {'lat':>7} {'yawerr':>7}   failures")
    for th, n, r in rows:
        lo, hi = wilson(r["seated"], n)
        fails = ", ".join(f"{k[5:]} {r['fail_' + k[5:]]}" for k in
                          ("fail_depth", "fail_yaw", "fail_seat", "fail_grip")
                          if r["fail_" + k[5:]])
        print(f"  {th:+8.3f} {math.degrees(th):+7.1f} {r['seated']:>6}/{n:<3} {r['rate']:7.3f} "
              f"  [{lo:.3f}, {hi:.3f}] {r['depth_mm']:6.1f}m {r['lat_mm']:6.2f}m "
              f"{r['yaw']:7.4f}   {fails or '-'}")

    ok = {abs(th) for th, n, r in rows if r["rate"] >= BAR}
    bad = {abs(th) for th, n, r in rows if r["rate"] < BAR}
    # largest magnitude that passes with nothing smaller failing
    cands = [a for a in sorted(ok) if not any(b < a for b in bad)]
    theta_max = max(cands) if cands else 0.0
    print(f"\n  bar {BAR:.2f}: passing |theta| {sorted(ok)}")
    if bad:
        print(f"  failing |theta| {sorted(bad)}")
    print(f"  ==> theta_max = {theta_max:.3f} rad ({math.degrees(theta_max):.1f} deg)")
    if bad and min(bad) <= min(ok, default=1e9):
        print("  WARNING: a small angle failed -- this is not an envelope limit, look for a bug")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/theta_sweep")
