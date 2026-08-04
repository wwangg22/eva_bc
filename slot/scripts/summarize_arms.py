# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate the Stage C eval JSONs into the arm comparison, and apply the pre-registered rule.

The rule was fixed in ``docs/slot/EXP_BC_ARMS.md`` before any training ran:

* B > A on pooled ``success_rate_later`` for **both** tasks, by more than the within-arm spread
  across training seeds  ->  DART data helps.
* the arms within each other's seed spread  ->  no measurable difference at this volume. Do not
  claim it helped.
* either arm misses the Stage C bar  ->  the bottleneck is not data composition.

Applying it in code rather than by eye is deliberate. eva_bc's history contains several
single-run comparisons that read as decisive and were void; the same data and recipe spanned
32.8-59.4 % across *training seeds alone* there. The margin has to clear that spread or it is
not a result.

Everything is computed from ``success_rate_later`` and from the per-episode records, never from
the headline ``success_rate``, which carries the first-episode bias and moves with --num-envs.

.. code-block:: bash

    python slot/scripts/summarize_arms.py runs/bc_arm*/eval_ckpt_final_*.json
    python slot/scripts/summarize_arms.py 'runs/bc_arm*/eval_*.json' --by-checkpoint
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

NAME_RE = re.compile(r"bc_arm(?P<arm>[AB])_seed(?P<seed>\d+)")
EVAL_RE = re.compile(r"eval_(?P<ckpt>ckpt_[^_]+)_(?P<task>[^_]+?)_s(?P<spawn>\d+)\.json$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- correct at the small n and near-boundary rates seen here,
    where the normal approximation puts the bound outside [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage C arm comparison.")
    ap.add_argument("paths", nargs="+", help="eval JSONs (globs are expanded here too)")
    ap.add_argument("--bar-v0", type=float, default=0.40, help="pre-registered Stage C bar, -v0")
    ap.add_argument("--bar-loose", type=float, default=0.55, help="pre-registered bar, -Loose-v0")
    a = ap.parse_args()

    # -Tight-v0 has NO pre-registered bar and was NOT part of the pre-registered arm comparison
    # (run_eval_sweep.sh evaluates it because it is in the deliverable and the marginal cost is
    # one more pass). Two bugs were latent here and are fixed by keeping that distinction:
    #   1. `t.endswith("Slot-v0")` is False for "Rebot-PrecisionSlot-Tight-v0", so Tight was
    #      being judged against the LOOSE bar (0.55) -- and a miss there fires the "the
    #      bottleneck is not data composition" verdict, which would have mis-framed the whole
    #      read of the sweep.
    #   2. The rule says "B > A on BOTH tasks". Silently letting a third task into that
    #      conjunction changes a pre-registered rule after seeing the data.
    # Tight is therefore REPORTED in full and EXCLUDED from both the bar check and the verdict.
    PREREG = ("Rebot-PrecisionSlot-Loose-v0", "Rebot-PrecisionSlot-v0")

    files: list[Path] = []
    for p in a.paths:
        files.extend(Path(x) for x in (glob.glob(p) if any(c in p for c in "*?[") else [p]))
    files = sorted(set(files))
    if not files:
        print("no eval JSONs matched")
        return 1

    # (arm, seed, task) -> list of (spawn, k_later, n_later, k_all, n_all)
    cells: dict[tuple[str, int, str], list] = defaultdict(list)
    ckpts: set[str] = set()
    for f in files:
        m, e = NAME_RE.search(str(f)), EVAL_RE.search(f.name)
        if not m or not e:
            print(f"  (skipping unparseable name: {f})")
            continue
        r = json.loads(f.read_text())
        later = [x for x in r["per_episode"] if x["episode_index_in_env"] > 0]
        cells[(m["arm"], int(m["seed"]), e["task"])].append(
            (int(e["spawn"]), sum(bool(x["success"]) for x in later), len(later),
             sum(bool(x["success"]) for x in r["per_episode"]), len(r["per_episode"])))
        ckpts.add(e["ckpt"])

    if len(ckpts) > 1:
        print(f"\n  WARNING: mixing checkpoints {sorted(ckpts)} in one comparison.\n")
    tasks = sorted({k[2] for k in cells})

    print("\n" + "=" * 92)
    print(f"  STAGE C ARM COMPARISON   checkpoint(s): {', '.join(sorted(ckpts))}")
    print("=" * 92)

    pooled: dict[tuple[str, str], tuple[int, int]] = {}
    per_seed: dict[tuple[str, str], list[float]] = defaultdict(list)
    for task in tasks:
        print(f"\n  {task}")
        print(f"    {'arm':>4} {'train':>6} {'spawn':>6} {'later k/n':>12} {'later rate':>11} "
              f"{'all rate':>9}   {'first-ep bias':>14}")
        for arm in ("A", "B"):
            tk = tn = 0
            for seed in sorted({k[1] for k in cells if k[0] == arm}):
                rows = sorted(cells.get((arm, seed, task), []))
                if not rows:
                    continue
                sk = sn = 0
                for spawn, k, n, ka, na in rows:
                    first_k, first_n = ka - k, na - n
                    bias = (first_k / first_n - k / n) if first_n and n else float("nan")
                    print(f"    {arm:>4} {seed:>6} {spawn:>6} {f'{k}/{n}':>12} "
                          f"{k / n:>11.3f} {ka / na:>9.3f}   {bias:>+13.3f}")
                    sk += k
                    sn += n
                    tk += k
                    tn += n
                if sn:
                    per_seed[(arm, task)].append(sk / sn)
            if tn:
                pooled[(arm, task)] = (tk, tn)
                lo, hi = wilson(tk, tn)
                print(f"    {arm:>4} {'POOLED':>6} {'':>6} {f'{tk}/{tn}':>12} {tk / tn:>11.3f} "
                      f"{'':>9}   95% CI [{lo:.3f}, {hi:.3f}]")

    # ---- pairing: are the two arms facing the same spawns?
    # The env seed fixes the reset draws, so two runs at the same seed should see identical
    # spawns -- but the flow policy calls torch.randn for x0 on the same global generator the
    # reset events draw from, so this only holds while every env refills in lockstep and none
    # flushes. Verified rather than assumed; if it holds the comparison is PAIRED and McNemar
    # applies, which is materially more powerful than comparing two independent proportions.
    paired_ok = None
    if True:
        by_key: dict[tuple, dict[str, dict[tuple, tuple]]] = {}
        for f in files:
            m, e = NAME_RE.search(str(f)), EVAL_RE.search(f.name)
            if not m or not e:
                continue
            r = json.loads(f.read_text())
            if not all("spawn_pos" in x for x in r["per_episode"]):
                continue
            k = (e["task"], e["spawn"], int(m["seed"]))
            by_key.setdefault(k, {})[m["arm"]] = {
                (x["env"], x["episode_index_in_env"]): (tuple(x["spawn_pos"]), bool(x["success"]))
                for x in r["per_episode"]}
        mism = same = 0
        b_only = a_only = 0
        for k, arms in by_key.items():
            if set(arms) != {"A", "B"}:
                continue
            common = set(arms["A"]) & set(arms["B"])
            for c in common:
                sa, ra = arms["A"][c]
                sb, rb = arms["B"][c]
                if max(abs(x - y) for x, y in zip(sa, sb)) > 1e-4:
                    mism += 1
                    continue
                same += 1
                a_only += ra and not rb
                b_only += rb and not ra
        if same or mism:
            print("\n" + "=" * 92)
            print("  PAIRING CHECK (same env seed -> same spawns?)")
            print("=" * 92)
            paired_ok = mism == 0 and same > 0
            print(f"    {same} episode slots with identical spawns, {mism} mismatched")
            if paired_ok:
                n_disc = a_only + b_only
                print(f"    -> the arms ARE paired. Discordant pairs: A-only {a_only}, "
                      f"B-only {b_only}, total {n_disc}")
                if n_disc:
                    # McNemar, normal approximation with continuity correction
                    chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
                    print(f"    McNemar chi2 = {chi2:.3f} on 1 df   "
                          f"({'p < 0.05' if chi2 > 3.841 else 'p > 0.05'}, "
                          f"{'significant' if chi2 > 3.841 else 'NOT significant'})")
                    print("    (this counts only episodes where the arms disagreed; concordant "
                          "ones carry no information about the difference)")
            else:
                print("    -> spawns DIFFER between arms. The comparison is unpaired; the "
                      "policy's own RNG draws desynchronised the reset stream. Use the pooled "
                      "rates below and ignore any paired reasoning.")

    # ---- the pre-registered decision
    print("\n" + "=" * 92)
    print("  PRE-REGISTERED DECISION RULE (EXP_BC_ARMS.md)")
    print("=" * 92)
    verdicts = []
    extra = [t for t in tasks if t not in PREREG]
    if extra:
        print(f"\n  NOT part of the pre-registered comparison, reported separately below: "
              f"{', '.join(extra)}")
    for task in [t for t in tasks if t in PREREG]:
        if ("A", task) not in pooled or ("B", task) not in pooled:
            print(f"\n  {task}: incomplete -- need both arms")
            continue
        ak, an = pooled[("A", task)]
        bk, bn = pooled[("B", task)]
        ra, rb = ak / an, bk / bn
        spreads = []
        for arm in ("A", "B"):
            v = per_seed[(arm, task)]
            s = (max(v) - min(v)) if len(v) > 1 else float("nan")
            spreads.append(s)
            print(f"\n  {task}  arm {arm}: per-training-seed later-rates "
                  f"{[f'{x:.3f}' for x in v]}  spread {s:.3f}")
        gap = rb - ra
        spread = max([s for s in spreads if s == s], default=float("nan"))
        clears = gap > spread
        print(f"  {task}  B - A = {gap:+.3f}   max within-arm seed spread = {spread:.3f}"
              f"   -> gap {'EXCEEDS' if clears else 'does NOT exceed'} the spread")
        verdicts.append((task, gap, spread, clears, ra, rb))

    bar = {"Rebot-PrecisionSlot-v0": a.bar_v0, "Rebot-PrecisionSlot-Loose-v0": a.bar_loose}
    print("\n  Stage C bar (pre-registered tasks only):")
    missed = False
    for task in [t for t in tasks if t in bar]:
        for arm in ("A", "B"):
            if (arm, task) not in pooled:
                continue
            k, n = pooled[(arm, task)]
            ok = k / n >= bar[task]
            missed |= not ok
            print(f"    arm {arm} on {task}: {k / n:.3f} vs bar {bar[task]:.2f}   "
                  f"{'CLEARS' if ok else 'MISSES'}")
    for task in extra:
        print(f"\n  {task} — no pre-registered bar; descriptive only:")
        for arm in ("A", "B"):
            if (arm, task) not in pooled:
                continue
            k, n = pooled[(arm, task)]
            lo, hi = wilson(k, n)
            print(f"    arm {arm}: {k}/{n} = {k / n:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
        print("    -> feeds docs/slot/EXP_TIGHT.md, not the arm decision.")

    print("\n  VERDICT  (over the pre-registered tasks only: "
          f"{', '.join(t for t in tasks if t in PREREG) or 'NONE PRESENT'})")
    if not verdicts:
        print("    No pre-registered task has both arms present -- no verdict is available.")
    if missed:
        print("    At least one arm missed the Stage C bar. Per the rule, the bottleneck is not")
        print("    data composition -- diagnose the policy/controller before reading the arm gap.")
    if verdicts and all(v[3] and v[1] > 0 for v in verdicts):
        print("    B beats A on BOTH tasks by more than the within-arm seed spread.")
        print("    -> DART data helps. Collect more; take the arm-B champion into Stage D.")
    elif verdicts and all(not v[3] for v in verdicts):
        print("    The arms are within each other's seed spread on every task.")
        print("    -> DART made no measurable difference at this volume. Do NOT claim it helped.")
        print("       Proceed on the pooled best checkpoint; put the next effort into HG-DAgger,")
        print("       which supplies corrective data the expert genuinely cannot.")
    elif verdicts:
        print("    Mixed: the gap clears the seed spread on some tasks and not others. Report the")
        print("    per-task numbers; do not summarise this as a single win or loss.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
