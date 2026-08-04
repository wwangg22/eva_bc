# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Does lateral misalignment explain the depth failures? Pool |lateral| by outcome and clearance.

`failure_taxonomy.py` answers "which bucket did the failures land in". This answers the follow-up
that decides what to do about them: **within a bucket, do failures differ from successes?**

It exists because a champion-only reading of the same data is misleading. At `bc_armB_seed0` on
`-v0` the two `stalled_in_mouth` failures sit at |lateral| 1.15 mm against successes' 0.45 mm,
which looks like a jam — the lateral error eating the clearance. That is **n = 2**. Pooled over
all six runs and both spawn seeds the gap shrinks at 1.5 mm and **vanishes entirely at 0.5 mm**,
where stalled failures and successes have the same lateral distribution to two decimals. So the
stall is not a lateral-precision failure, at least not on the rung where lateral precision is
most expensive.

Read the three failure buckets as three mechanisms:

    stalled_in_mouth   aligned as well as a success, stopped short   -> a PUSH/DEPTH failure
    never_entered      p90 lateral 4-7 mm, far beyond any clearance  -> an AIMING failure
    gross_miss         40-130 mm off axis                            -> a TRANSPORT failure

**Read the numbers with the censoring in mind** (gotcha 30): once the block is inside the channel
its |lateral| cannot exceed the clearance, so the success and stalled rows are both truncated at
the rung's own value and their *spread* is not comparable across rungs. What is comparable is
success vs stalled **within** a rung, which is the comparison this makes.

.. code-block:: bash

    python analysis/lateral_by_bucket.py 'runs/*/eval_ckpt_final_*.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_taxonomy import bucket, clearance_of  # noqa: E402

ORDER = ["success", "stalled_in_mouth", "never_entered", "gross_miss", "never_lifted",
         "seat_reject", "yaw_reject", "other"]


def main() -> int:
    ap = argparse.ArgumentParser(description="|lateral| by outcome bucket and clearance.")
    ap.add_argument("paths", nargs="*", default=["runs/*/eval_ckpt_final_*.json"])
    a = ap.parse_args()

    files = sorted({f for p in a.paths for f in glob.glob(p)})
    if not files:
        print("no eval JSONs matched")
        return 1

    rows: dict[tuple[float, str], list[float]] = {}
    for f in files:
        r = json.loads(Path(f).read_text())
        clear = clearance_of(r["task"])
        for e in (x for x in r["per_episode"] if x["episode_index_in_env"] > 0):
            key = "success" if e["success"] else bucket(e)
            rows.setdefault((clear, key), []).append(abs(e["lateral_mm"]))

    print(f"\npooled over {len(files)} eval files, later cohort only "
          f"(episode_index_in_env > 0)")
    print(f"{'clear':>6}  {'cohort':<18}{'n':>6}{'p25':>9}{'p50':>8}{'p75':>8}{'p90':>8}"
          f"   |lateral| mm")
    for clear in sorted({k[0] for k in rows}, reverse=True):
        print(f"  {'-' * 70}")
        for b in ORDER:
            v = sorted(rows.get((clear, b), []))
            if not v:
                continue
            q = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
            flag = ""
            if b == "stalled_in_mouth":
                s = sorted(rows.get((clear, "success"), []))
                if s:
                    sm = s[min(len(s) - 1, len(s) // 2)]
                    flag = ("   <- SAME as success" if abs(q(.5) - sm) < 0.10
                            else f"   <- {q(.5) / sm:.2f}x the success median")
            print(f"{clear:>6}  {b:<18}{len(v):>6}{q(.25):>9.2f}{q(.5):>8.2f}{q(.75):>8.2f}"
                  f"{q(.9):>8.2f}{flag}")
    print("\n  CENSORED (gotcha 30): inside the channel |lateral| cannot exceed the clearance, so"
          "\n  the success and stalled rows are truncated at each rung's own value. Compare them"
          "\n  to each other WITHIN a rung; do not compare spreads ACROSS rungs.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
