# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Can a good x0 be recognised without running it?  (EXP_STEER §13)

The constant-x0 probe found a 79-point spread across five latents under 2 % action noise. If
some cheap statistic of the (50, 7) matrix predicted the score, latent search would be free:
draw a thousand, keep the best-looking one. This script tests the obvious candidates against the
measured cells and reports the rank correlation.

Candidates, and why each is plausible:
  ||x0||        total norm -- the typical-set argument of §10b. Expected to be useless *within*
                the structured family (every draw sits on the shell by construction), and it is
                kept only so that expectation is on the record rather than assumed.
  ||DC||        norm of the mean across the 50 chunk positions. The broadcast cells are pure DC
                and score 0.000, so a large DC component might be the poison.
  AC std        spread around that mean -- the temporal variation broadcasting destroys.
  grip DC       the gripper column's mean. The failure is a refusal to push, and the gripper
                channel is what holds the block during it.

.. code-block:: bash

    python analysis/latent_stats.py --run runs/bc_armB_seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CHUNK, ACTION_DIM = 50, 7


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation; n is 4-8 here, so this is descriptive, not a test."""
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/bc_armB_seed0")
    a = ap.parse_args()
    run = Path(a.run)

    rows = []
    for s in range(1, 9):
        p = run / f"x0probe_act002_s{s}.json"
        if not p.exists():
            continue
        score = json.loads(p.read_text())["success_rate_later"]
        g = torch.Generator().manual_seed(s)
        X = torch.randn((CHUNK, ACTION_DIM), generator=g)
        dc = X.mean(0)
        rows.append({
            "seed": s, "score": score,
            "norm": float(X.norm()), "dc_norm": float(dc.norm()),
            "ac_std": float((X - dc).std()), "grip_dc": float(dc[6]),
        })

    if not rows:
        raise SystemExit("no x0probe_act002_s*.json cells found")

    keys = ["norm", "dc_norm", "ac_std", "grip_dc"]
    print(f"\n  {'seed':>4} {'score':>6} " + " ".join(f"{k:>8}" for k in keys))
    for r in sorted(rows, key=lambda r: -r["score"]):
        print(f"  {r['seed']:>4} {r['score']:6.3f} " + " ".join(f"{r[k]:8.3f}" for k in keys))

    print(f"\n  Spearman rank correlation with success (n = {len(rows)}, descriptive only)")
    for k in keys:
        print(f"    {k:<10} {spearman([r[k] for r in rows], [r['score'] for r in rows]):+.3f}")
    print("\n  A |rho| near 1 would mean latent search can be done offline. Anything else means\n"
          "  the only way to find a good x0 is to run it -- which is what §13 concludes.\n")


if __name__ == "__main__":
    main()
