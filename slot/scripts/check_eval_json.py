# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Audit an ``eval_act.py`` results JSON for accounting errors. No GPU, no simulator.

``test_pipeline_cpu.py`` validates everything in the training path except the Isaac Sim rollout
loop. This closes that gap from the other side: run the eval once on a *throwaway* checkpoint
and check that the numbers it reports are arithmetically consistent with the episodes it claims
to have run. An eval harness that miscounts is worse than no eval, and every property below is
one that has already gone wrong once on this project or in the pick-place work it came from:

* **the first-episode cohort split** — every A/B decision rests on ``success_rate_later``. If
  the split is wrong (e.g. ``ep_index`` incremented before the record was written), the two
  cohorts are mislabelled and the bias is *still* in the number, invisibly.
* **the episode horizon** — every episode must be exactly 599 steps. A shorter one means an
  early termination is live (``block_dropped``/``block_toppled`` were meant to be nulled), which
  would silently re-randomise the scene and give the policy a second attempt; a longer one means
  ``--episode-length-s`` did not take.
* **rates match the per-episode records** — the headline is recomputed from ``per_episode`` and
  compared, so a rate can never disagree with the episodes behind it.
* **success implies the raw predicate** — ``success`` is ``is_inserted AND seated``; it cannot
  be true where ``inserted_raw`` is false. If it ever is, the two predicates have drifted apart.

.. code-block:: bash

    python slot/scripts/check_eval_json.py runs/instrument_test/eval.json --expect-len 599
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit an eval_act.py results JSON.")
    ap.add_argument("path")
    ap.add_argument("--expect-len", type=int, default=600,
                    help="every episode must be exactly this many control steps. 600 = "
                         "episode_length_s 12.0 / (decimation 8 * sim.dt 1/400). Note the "
                         "demos are T=599: the collector stores one (obs, action) pair per "
                         "step and the final observation has no action after it, so the demo "
                         "arrays are one shorter than the episode. Same horizon, different "
                         "bookkeeping -- do not 'fix' either to match the other.")
    ap.add_argument("--expect-envs", type=int, default=None,
                    help="if given, n_first_episode must equal this (one per env)")
    a = ap.parse_args()

    r = json.loads(Path(a.path).read_text())
    eps = r["per_episode"]
    n = len(eps)
    print(f"\n{Path(a.path).name}: {n} episodes, task {r['task']}, seed {r['seed']}, "
          f"num_envs {r['config']['num_envs']}\n")

    check("episodes count matches per_episode length", r["episodes"] == n, f"{r['episodes']} vs {n}")

    # --- the cohort split, which every A/B decision depends on
    first = [e for e in eps if e["episode_index_in_env"] == 0]
    later = [e for e in eps if e["episode_index_in_env"] > 0]
    check("n_first_episode matches the records", r["n_first_episode"] == len(first),
          f"{r['n_first_episode']} vs {len(first)}")
    check("n_later matches the records", r["n_later"] == len(later),
          f"{r['n_later']} vs {len(later)}")
    check("the two cohorts partition the episodes", len(first) + len(later) == n)
    envs = r["config"]["num_envs"]
    expect_first = a.expect_envs if a.expect_envs is not None else min(envs, n)
    check("exactly one first-episode per env (no env ran twice before others ran once)",
          len(first) == expect_first, f"{len(first)} first-episodes over {envs} envs")
    check("every env contributed at most one episode_index 0",
          len({e["env"] for e in first}) == len(first),
          f"{len({e['env'] for e in first})} distinct envs among {len(first)} first-episodes")

    # --- rates recomputed from the records themselves
    def rate(rs):
        return (sum(bool(e["success"]) for e in rs) / len(rs)) if rs else None

    for key, rs in (("success_rate", eps), ("success_rate_first_episode", first),
                    ("success_rate_later", later)):
        got, exp = r[key], rate(rs)
        ok = (got is None and exp is None) or (got is not None and exp is not None
                                               and abs(got - exp) < 1e-9)
        check(f"{key} equals the rate over its own records", ok, f"reported {got} vs {exp}")

    # --- horizon: an episode shorter than the horizon means an early termination is LIVE
    lens = Counter(e["length"] for e in eps)
    check(f"every episode ran exactly {a.expect_len} steps", set(lens) == {a.expect_len},
          f"lengths {dict(lens)}")
    check("mean_ep_len matches the records",
          abs(r["mean_ep_len"] - sum(e["length"] for e in eps) / n) < 1e-6)

    # --- predicate consistency: success = inserted_raw AND seated, so it implies inserted_raw
    bad = [e["episode"] for e in eps if e["success"] and not e["inserted_raw"]]
    check("success never true where inserted_raw is false", not bad, f"episodes {bad[:8]}")
    gap = sum(e["inserted_raw"] and not e["success"] for e in eps)
    print(f"    seat guard rejected {gap} episode(s) that the env's bare is_inserted accepted")

    # --- schema completeness (a missing field reads as 0/None downstream)
    need = {"episode", "env", "episode_index_in_env", "length", "success", "flushes",
            "placed_final", "placed_max", "max_obj_z", "final_obj_pos", "inserted_raw",
            "depth_mm", "lateral_mm", "yaw_rad"}
    missing = {k for k in need if any(k not in e for e in eps)}
    check("every per-episode record carries the full schema", not missing, f"missing {missing}")
    # spawn_pos was added mid-session; JSONs written before that lack it and are still valid.
    if all("spawn_pos" in e for e in eps):
        print(f"    spawn_pos present -- this run can be checked for pairing against another")
    else:
        print("    spawn_pos absent (written before the field existed); pairing cannot be "
              "verified for this run")

    d = [e["depth_mm"] for e in eps]
    lat = [abs(e["lateral_mm"]) for e in eps]
    print(f"\n    success {r['success_rate']:.3f}   first {r['success_rate_first_episode']}   "
          f"later {r['success_rate_later']}")
    print(f"    depth mm  min {min(d):.1f}  median {sorted(d)[n // 2]:.1f}  max {max(d):.1f}   "
          f"(success needs >= 40.0)")
    print(f"    |lateral| mm  median {sorted(lat)[n // 2]:.2f}  max {max(lat):.2f}")
    print(f"    flushes {r['flush_count']} over {n} episodes (enabled: {r['flush_enabled']})")

    print()
    if FAILURES:
        print(f"  {len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"    - {f}")
        print()
        return 1
    print("  EVAL JSON IS SELF-CONSISTENT\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
