# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Paired (McNemar) comparison of any set of eval JSONs that share a spawn seed.

Two eval runs at the same ``--seed`` face the **same spawns in the same episode slots** on this
task -- verified, not assumed, by comparing the recorded ``spawn_pos`` slot for slot. When that
holds the comparison is *paired*, and comparing two independent proportions throws away most of
the information: episodes where both checkpoints agree carry nothing about the difference
between them, and they are the large majority.

The difference this makes is not cosmetic. Over the arm-A seed-0 learning curve the Wilson
intervals for 30k (0.896), 50k (0.854) and 100k (0.927) all overlap, which on its own supports
no conclusion at all. Paired, the same 96 episodes say clearly that 10k -> 30k is real
(chi2 = 17.3) and that **nothing after 30k is** (chi2 <= 2.1): 30k and 100k agree on 81 of 96
episodes and disagree 6 one way, 9 the other.

Reports, for every pair of inputs: whether the spawns actually match, the discordant counts,
and McNemar's chi-squared with the continuity correction (1 df, critical value 3.841 at
p = 0.05). Only the ``episode_index_in_env > 0`` cohort is used -- the first episode each env
runs carries the PhysX first-episode bias.

.. code-block:: bash

    python slot/analysis/paired_evals.py 'slot/runs/bc_armA_seed0/eval_ckpt_*.json'
    python slot/analysis/paired_evals.py a.json b.json --label-from-path
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import re
import sys
from pathlib import Path

CKPT_RE = re.compile(r"ckpt_(\d+|final)")


def label_for(path: Path, from_path: bool) -> str:
    if from_path:
        return str(path.parent.name) + "/" + path.stem
    m = CKPT_RE.search(path.name)
    return m.group(1) if m else path.stem


def load(path: Path) -> dict:
    r = json.loads(path.read_text())
    return {
        (e["env"], e["episode_index_in_env"]): (tuple(e.get("spawn_pos", ())), bool(e["success"]))
        for e in r["per_episode"]
        if e["episode_index_in_env"] > 0
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired McNemar comparison of eval JSONs.")
    ap.add_argument("paths", nargs="+", help="eval JSONs (globs expanded here too)")
    ap.add_argument("--label-from-path", action="store_true",
                    help="label by run dir instead of checkpoint number")
    a = ap.parse_args()

    files: list[Path] = []
    for p in a.paths:
        files.extend(Path(x) for x in (glob.glob(p) if any(c in p for c in "*?[") else [p]))
    files = sorted(set(files))
    if len(files) < 2:
        print("need at least two eval JSONs")
        return 1

    cells = {label_for(f, a.label_from_path): load(f) for f in files}
    # numeric checkpoint labels sort numerically, everything else alphabetically
    def key(k):
        return (0, int(k)) if k.isdigit() else ((0, 10**9) if k == "final" else (1, 0))

    labels = sorted(cells, key=key)

    print(f"\npaired comparison over {len(files)} eval JSON(s), later-episode cohort only\n")
    for lab in labels:
        c = cells[lab]
        k = sum(v[1] for v in c.values())
        print(f"  {lab:>28}  {k}/{len(c)} = {k / len(c):.3f}")
    print()

    any_unpaired = False
    for x, y in itertools.combinations(labels, 2):
        A, B = cells[x], cells[y]
        common = set(A) & set(B)
        if not common:
            print(f"  {x} vs {y}: no shared episode slots")
            continue
        # spawn_pos is absent in JSONs written before the field existed; treat as unverifiable
        # rather than as a mismatch, and say so.
        checkable = [c for c in common if A[c][0] and B[c][0]]
        mism = sum(
            max(abs(p - q) for p, q in zip(A[c][0], B[c][0])) > 1e-4 for c in checkable
        )
        paired = mism == 0 and bool(checkable)
        any_unpaired |= not paired

        x_only = sum(A[c][1] and not B[c][1] for c in common)
        y_only = sum(B[c][1] and not A[c][1] for c in common)
        n = x_only + y_only
        chi2 = (abs(x_only - y_only) - 1) ** 2 / n if n else 0.0
        sig = "p < 0.05  REAL" if chi2 > 3.841 else "p > 0.05  noise"

        if not checkable:
            note = "spawns UNVERIFIABLE (no spawn_pos) -- treat as unpaired"
        elif not paired:
            note = f"spawns DIFFER in {mism}/{len(checkable)} slots -- NOT paired, chi2 invalid"
        else:
            note = sig

        print(f"  {x:>12} vs {y:<12} n={len(common):3d}  agree={len(common) - n:3d}  "
              f"discordant {x_only:3d}/{y_only:<3d}  chi2={chi2:7.3f}   {note}")

    if any_unpaired:
        print("\n  WARNING: at least one pair is not verifiably paired. McNemar assumes the same\n"
              "  episode faced the same spawn; where that fails, use the pooled rates and Wilson\n"
              "  intervals instead and ignore the chi-squared above.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
