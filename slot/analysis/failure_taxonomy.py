# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Classify the failures in one or more eval JSONs into mechanism buckets.

Ported in spirit from the pick-place taxonomy, but the buckets are **this task's**, chosen from
what the data actually showed rather than by renaming pick-place's. The ordering matters: a
block that was never lifted cannot also be "stalled in the mouth", so the buckets are applied as
a decision list and every failure lands in exactly one.

    never_lifted        max_obj_z never cleared the carry height -- the pick-place headline
                        bucket. Expected to be EMPTY here: the expert grasps 2038/2038 and all
                        61 of its failures are `release=unseated`. If it is ever non-empty for a
                        trained policy, that is a failure mode the demonstrator never had.
    gross_miss          the block ended more than one BLOCK HALF-WIDTH (15 mm) off the slot
                        axis. Split out of never_entered because conflating the two hides the
                        story: at ckpt_0030000 the never-entered failures sat at 1.4-1.9 mm
                        (marginal jams on the lip), while at ckpt_final they are 23-73 mm --
                        the block is nowhere near the slot. Those are TRANSPORT failures and
                        share no mechanism with a precision failure. The 15 mm threshold is the
                        block's own half-width (mdp.BLOCK_HALF[1] = 0.0150), not a tuned number.
    never_entered       insertion depth < 0 with the block still on the slot axis: it reached
                        the mouth and did not cross it. On -v0 at 30k these came with |lateral|
                        1.4-1.9 mm against a 1.5 mm clearance -- a jam on the lip, where the
                        lateral error consumed the whole budget.
    stalled_in_mouth    0 <= depth < SUCCESS_DEPTH: it entered and stopped short. Distinguish
                        these from never_entered: one is an AIMING failure, the other a PUSH
                        failure, and they call for different interventions.
    yaw_reject          depth >= SUCCESS_DEPTH but |yaw| > SUCCESS_YAW. Expected rare -- every
                        -v0 failure measured |yaw| <= 0.030 rad against a 0.12 tolerance.
    seat_reject         depth >= SUCCESS_DEPTH, yaw fine, and the env's own `is_inserted` says
                        TRUE -- rejected only by slot_mdp's seated-height guard. These are the
                        episodes that would inflate the headline if the guard were removed (a
                        Stage A probe cell scored 93.8 % with 13.28 mm mean lateral error).
    other               anything the list above does not catch. Should be empty; if it is not,
                        the list is wrong and needs a new bucket rather than a shrug.

Reports per-bucket counts plus the |lateral| and depth distributions that motivate them, and --
because the clearance censors lateral error (gotcha 30) -- prints the clearance alongside so the
two are never read apart.

.. code-block:: bash

    python slot/analysis/failure_taxonomy.py runs/bc_armA_seed0/eval_ckpt_final_*.json
    python slot/analysis/failure_taxonomy.py 'runs/bc_arm*/eval_ckpt_final_*Tight*.json' --pool
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

SUCCESS_DEPTH_MM = 40.0   # mdp.common.SUCCESS_DEPTH = 0.040
SUCCESS_YAW = 0.12        # mdp.common.SUCCESS_YAW
CARRY_Z_MIN = 0.050       # block rides at ~0.061 carried, 0.055 seated, 0.032-0.035 on the table
GROSS_LATERAL_MM = 15.0   # mdp.BLOCK_HALF[1] = 0.0150 -- one block half-width off the axis
CLEARANCE_MM = {"Loose": 3.0, "Tight": 0.5}   # anything else is the 1.5 mm default


def clearance_of(task: str) -> float:
    for key, mm in CLEARANCE_MM.items():
        if key in task:
            return mm
    return 1.5


def bucket(e: dict) -> str:
    """Decision list -- first match wins, so every failure lands in exactly one bucket."""
    mz = e["max_obj_z"]
    mz = max(mz) if isinstance(mz, list) else mz
    if mz < CARRY_Z_MIN:
        return "never_lifted"
    if abs(e["lateral_mm"]) > GROSS_LATERAL_MM:
        return "gross_miss"
    if e["depth_mm"] < 0:
        return "never_entered"
    if e["depth_mm"] < SUCCESS_DEPTH_MM:
        return "stalled_in_mouth"
    if abs(e["yaw_rad"]) > SUCCESS_YAW:
        return "yaw_reject"
    if e.get("inserted_raw"):
        return "seat_reject"
    return "other"


ORDER = ["never_lifted", "gross_miss", "never_entered", "stalled_in_mouth", "yaw_reject",
         "seat_reject", "other"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Failure taxonomy for slot eval JSONs.")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pool", action="store_true",
                    help="pool every input into one taxonomy instead of one per file")
    a = ap.parse_args()

    files: list[Path] = []
    for p in a.paths:
        files.extend(Path(x) for x in (glob.glob(p) if any(c in p for c in "*?[") else [p]))
    files = sorted(set(files))
    if not files:
        print("no eval JSONs matched")
        return 1

    groups: dict[str, list] = {}
    tasks: dict[str, set] = {}
    for f in files:
        r = json.loads(f.read_text())
        key = "POOLED" if a.pool else f"{f.parent.name}/{f.name}"
        # later cohort only: the first episode each env runs carries the PhysX first-episode bias
        groups.setdefault(key, []).extend(
            e for e in r["per_episode"] if e["episode_index_in_env"] > 0)
        tasks.setdefault(key, set()).add(r["task"])

    for key, eps in groups.items():
        succ = [e for e in eps if e["success"]]
        fails = [e for e in eps if not e["success"]]
        tset = tasks[key]
        clear = {clearance_of(t) for t in tset}
        cstr = "/".join(f"{c:g}" for c in sorted(clear))

        print(f"\n{'=' * 92}\n  {key}")
        print(f"  tasks: {', '.join(sorted(tset))}   per-side clearance {cstr} mm")
        print(f"{'=' * 92}")
        print(f"  later-cohort success {len(succ)}/{len(eps)} = {len(succ) / len(eps):.3f}"
              f"   ({len(fails)} failures)")

        if fails:
            c = Counter(bucket(e) for e in fails)
            print(f"\n  {'bucket':<20}{'n':>5}{'% of fails':>12}   {'depth mm (med)':>15}"
                  f"{'|lat| mm (med)':>16}")
            for b in ORDER:
                if not c[b]:
                    continue
                sel = [e for e in fails if bucket(e) == b]
                print(f"  {b:<20}{c[b]:>5}{100 * c[b] / len(fails):>11.1f}%   "
                      f"{st.median(e['depth_mm'] for e in sel):>15.1f}"
                      f"{st.median(abs(e['lateral_mm']) for e in sel):>16.2f}")
            empty = [b for b in ORDER if not c[b]]
            if empty:
                print(f"  empty buckets: {', '.join(empty)}")
            if c["other"]:
                print("  !! 'other' is non-empty -- the decision list is incomplete, add a bucket")

        if succ:
            lat = sorted(abs(e["lateral_mm"]) for e in succ)
            q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]  # noqa: E731
            print(f"\n  successes |lateral| mm: p25 {q(.25):.2f}  p50 {q(.50):.2f}  "
                  f"p75 {q(.75):.2f}  p90 {q(.90):.2f}  max {lat[-1]:.2f}")
            if len(clear) == 1:
                c0 = next(iter(clear))
                print(f"  -> fills {100 * lat[-1] / c0:.0f}% of the {c0:g} mm clearance. "
                      f"REMEMBER (gotcha 30): this quantity is CENSORED by the channel -- its "
                      f"max is the clearance by construction, so it cannot be extrapolated to a "
                      f"different rung.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
