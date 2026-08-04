# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Read back EXP_ROBUSTNESS (docs/slot/EXP_ROBUSTNESS.md) against its pre-registered beliefs.

Every cell is the same checkpoint at the same spawn seed with ONE knob moved, so the whole
report is a set of two-sample comparisons against `robust_gate_nominal`.

**Which test, and why it is decided by the data rather than by me.** The pre-registration said
"unpaired, always", because `--fixed-x0` had already been caught changing the reset draw
(`EXP_TIGHT.md` §7c) and I did not want to assume otherwise a second time. But shifting
`SLOT_CENTER` consumes no RNG, so the spawn stream *may* be untouched -- in which case McNemar on
matched spawns is valid and far more powerful at n=96. Rather than guess, this script does what
`paired_evals.py` does: compare the recorded `spawn_pos` episode-for-episode and let the answer
decide the test. It prints which test it used and why, so the choice is auditable. Cells whose
spawns provably differ (`--spawn-scale`, which changes the sampling range itself) fall back to
the unpaired z-test, as pre-registered.

Reported per cell: later-cohort success (the first episode each env runs carries the PhysX
warm-start bias, measured on this project between -2.1 and +18.7 points), a Wilson 95 % interval,
the delta against the gate with its test, and the failure-bucket mix -- because "success fell
20 points" and "success fell 20 points and every new failure is a gross_miss" are different
findings.

.. code-block:: bash

    python analysis/robustness_report.py                      # defaults to the champion run
    python analysis/robustness_report.py --run runs/bc_armA_seed0
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_taxonomy import ORDER, bucket, clearance_of  # noqa: E402

GATE = "gate_nominal"
Z95 = 1.959963985


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval. Used instead of normal-approx because cells run to 0.98."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Pooled two-proportion z-test, two-sided."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (k1 / n1 - k2 / n2) / se
    return (z, 2 * norm_sf(abs(z)))


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, float]:
    """Exact binomial McNemar on the discordant pairs (n is small; no chi-square approx)."""
    b = sum(1 for a, c in pairs if a and not c)   # gate win
    c_ = sum(1 for a, c in pairs if c and not a)  # cell win
    n = b + c_
    if n == 0:
        return (b, c_, 1.0)
    lo = min(b, c_)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return (b, c_, min(1.0, 2 * tail))


def load(path: Path, name: str | None = None) -> dict:
    r = json.loads(path.read_text())
    later = [e for e in r["per_episode"] if e["episode_index_in_env"] > 0]
    return {
        "name": name if name is not None else path.name.split("robust_")[1][:-5],
        "task": r["task"],
        "cfg": r["config"],
        "later": later,
        "k": sum(e["success"] for e in later),
        "n": len(later),
        # keyed by (env, episode_index_in_env) so pairing survives any reordering
        "spawn": {(e["env"], e["episode_index_in_env"]): tuple(e.get("spawn_pos") or ())
                  for e in later},
        "succ": {(e["env"], e["episode_index_in_env"]): bool(e["success"]) for e in later},
    }


def spawns_match(a: dict, b: dict) -> tuple[bool, str]:
    """True only if every shared episode's spawn is recorded AND identical.

    A missing `spawn_pos` is 'unknown', never 'equal' -- the distinction that caught a wrong
    pairing claim in session 5 (an eval predating the field compared empty tuples as equal).
    """
    common = sorted(set(a["spawn"]) & set(b["spawn"]))
    if not common:
        return (False, "no shared episodes")
    checkable = [c for c in common if a["spawn"][c] and b["spawn"][c]]
    if len(checkable) < len(common):
        return (False, f"spawn_pos missing on {len(common) - len(checkable)}/{len(common)}")
    mism = sum(max(abs(p - q) for p, q in zip(a["spawn"][c], b["spawn"][c])) > 1e-4
               for c in checkable)
    if mism:
        return (False, f"{mism}/{len(checkable)} spawns differ")
    return (True, f"all {len(checkable)} spawns identical")


def main() -> int:
    ap = argparse.ArgumentParser(description="EXP_ROBUSTNESS read-back.")
    ap.add_argument("--run", default="runs/bc_armB_seed0")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.run}/robust_*.json"))
    if not files:
        print(f"no robust_*.json under {a.run}")
        return 1
    cells = {c["name"]: c for c in (load(Path(f)) for f in files)}
    if GATE not in cells:
        print(f"!! {GATE} missing -- belief 1 is untested, nothing below is interpretable")
        return 1

    # A cell must be compared to a baseline on ITS OWN TASK. The dy ladder crosses clearances
    # (loose_dy_p002 is on -Loose-v0), and scoring those against the -v0 gate would charge them
    # for the task difference as well as the perturbation -- turning loose_dy_p003's honest
    # -0.167 vs its own task into a -0.219 that means nothing. Prefer a robust_ cell with dy=0
    # on that task; otherwise fall back to the sweep's own cell for it.
    refs: dict[str, dict] = {cells[GATE]["task"]: cells[GATE]}
    for c in cells.values():
        cf = c["cfg"]
        zeroed = not (cf["slot_dx"] or cf["slot_dy"]) and cf["spawn_scale"] == 1.0 \
            and not cf.get("arm_jitter") and not cf.get("obs_noise")
        if zeroed and c["task"] not in refs:
            refs[c["task"]] = c
    for c in list(cells.values()):
        if c["task"] in refs:
            continue
        sweep = Path(a.run) / f"eval_ckpt_final_{c['task']}_s777.json"
        if sweep.exists():
            refs[c["task"]] = load(sweep, name=f"sweep:{c['task']}")

    g = cells[GATE]
    print(f"\n{'=' * 108}")
    print(f"  EXP_ROBUSTNESS -- {a.run}/ckpt_final.pt, spawn seed 777, later cohort only")
    print(f"  baselines (each cell is scored against the one for ITS OWN task):")
    for t, r in sorted(refs.items()):
        print(f"    {t:<32} {r['k']:>3}/{r['n']:<4} = {r['k'] / r['n']:.3f}   [{r['name']}]")
    print(f"{'=' * 108}")
    print(f"  {'cell':<14}{'dx mm':>7}{'dy mm':>7}{'spawn':>7}{'armjit':>7}{'noise':>7}"
          f"{'ep s':>6}{'clr':>5}   {'succ':>10}{'rate':>7}  {'95% CI':<16}{'delta':>8}  test")
    print(f"  {'-' * 122}")

    order = [GATE] + [n for n in ("dx_m010", "dx_p005", "dx_p010", "dx_p020",
                                  "dy_m010", "dy_p005", "dy_p010", "dy_p020",
                                  "spawn15", "spawn20",
                                  "arm005", "arm010", "noise05", "noise20",
                                  "horizon20", "combo") if n in cells]
    order += [n for n in cells if n not in order]

    rows = []
    for name in order:
        c = cells[name]
        cfg = c["cfg"]
        lo, hi = wilson(c["k"], c["n"])
        rate = c["k"] / c["n"] if c["n"] else float("nan")
        ref = refs.get(c["task"])
        if ref is None or ref["name"] == name:
            test = "(reference)" if ref is not None else "!! no baseline for this task"
            delta = ""
        else:
            paired, why = spawns_match(ref, c)
            if paired:
                common = sorted(set(ref["succ"]) & set(c["succ"]))
                b, cw, p = mcnemar([(ref["succ"][k], c["succ"][k]) for k in common])
                test = f"McNemar exact b={b} c={cw} p={p:.4g}  [{why}]"
            else:
                z, p = two_prop_z(c["k"], c["n"], ref["k"], ref["n"])
                test = f"2-prop z={z:+.2f} p={p:.4g}  [unpaired: {why}]"
            delta = f"{rate - ref['k'] / ref['n']:+.3f}"
            if ref["name"] != GATE:
                test += f"  vs {ref['name']}"
        print(f"  {name:<14}{1000 * cfg['slot_dx']:>7.1f}{1000 * cfg['slot_dy']:>7.1f}"
              f"{cfg['spawn_scale']:>7.2f}{cfg.get('arm_jitter', 0.0):>7.2f}"
              f"{cfg.get('obs_noise', 0.0):>7.2f}{cfg.get('episode_length_s') or 12.0:>6.0f}"
              f"{clearance_of(c['task']):>5.1f}   {c['k']:>4}/{c['n']:<5}{rate:>7.3f}"
              f"  [{lo:.3f},{hi:.3f}]  {delta:>8}  {test}")
        rows.append((name, c))

    print(f"\n  {'cell':<14}{'fails':>6}   failure mix (later cohort)"
          f"{'':>14}{'fail depth mm':>16}{'fail |lat| mm':>15}")
    print(f"  {'-' * 104}")
    for name, c in rows:
        fails = [e for e in c["later"] if not e["success"]]
        if not fails:
            print(f"  {name:<14}{0:>6}   (none)")
            continue
        cnt = Counter(bucket(e) for e in fails)
        mix = " ".join(f"{b}={cnt[b]}" for b in ORDER if cnt[b])
        print(f"  {name:<14}{len(fails):>6}   {mix:<40}"
              f"{st.median(e['depth_mm'] for e in fails):>16.1f}"
              f"{st.median(abs(e['lateral_mm']) for e in fails):>15.2f}")

    print(f"\n  n={g['n']} per cell: a 10-point delta is roughly the resolution limit here "
          f"(Wilson half-width ~0.03-0.09 depending on rate).")
    print("  Single-seed caution (PLAN 5.28): one checkpoint at one spawn seed. Replicate any "
          "claimed mechanism.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
