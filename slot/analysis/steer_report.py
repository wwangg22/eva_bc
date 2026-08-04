# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Read back EXP_STEER §8 -- the arm-C gate and the constant-x0 probe.

Two independent questions, both answered here because they share a protocol:

**The gate.** `eval_steer.py`'s noise path must be the same perturbation `eval_act.py`'s is.
With `--fixed-x0 zeros` neither harness draws a randn for x0 and both draw one
`randn_like(action)` per env step, so the two should agree EPISODE-FOR-EPISODE. This script
checks that first and only falls back to rate comparison if the streams diverge -- and says
which it used. A rate match with a per-episode mismatch is reported as its own finding, because
it would mean the paths consume randomness differently and no future paired comparison between
them is valid.

**The probe.** Belief 2's mechanism is mode collapse at staging: if some x0 selects the push,
holding it constant should beat the stochastic policy that redraws every 15 steps. Five constant
draws under 2 % action noise bound what x0 selection alone can buy. The spread is the load-
bearing statistic -- a small spread means the chunk barely depends on x0 under noise, which
kills belief 2 outright and stops the PPO arms from launching.

Every rate is the LATER cohort (`episode_index_in_env > 0`): the first episode a process runs
carries a warm-start bias measured on this project between -2.1 and +18.7 points.

.. code-block:: bash

    python analysis/steer_report.py
    python analysis/steer_report.py --run runs/bc_armA_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_taxonomy import bucket  # noqa: E402
from robustness_report import two_prop_z, wilson  # noqa: E402

PROBE = [("zeros", "x0probe_act002_zeros.json")] + [
    (f"seed{s}", f"x0probe_act002_s{s}.json") for s in (1, 2, 3, 4)
]
# §10: the same seeds with row 0 broadcast across all 50 chunk positions -- the ONLY family
# x0-steering can express (steer_core.set_steer broadcasts one 7-vector). b<s> vs s isolates
# per-chunk-position structure, so this pair decides whether the steering action space is
# even the right shape.
BCAST = [(f"b{s}", f"x0probe_act002_b{s}.json") for s in (1, 2, 3, 4)]
EXTRA = [(f"seed{s}", f"x0probe_act002_s{s}.json") for s in (5, 6, 7, 8)]


def load(path: Path) -> dict | None:
    """Later-cohort view of any eval JSON (eval_act.py or eval_steer.py -- same fields)."""
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    later = [e for e in r["per_episode"] if e["episode_index_in_env"] > 0]
    return {
        "path": path,
        "cfg": r.get("config", {}),
        "k": sum(e["success"] for e in later),
        "n": len(later),
        # keyed so the comparison survives any reordering of the records list
        "succ": {(e["env"], e["episode_index_in_env"]): bool(e["success"]) for e in later},
        "len": {(e["env"], e["episode_index_in_env"]): int(e["length"]) for e in later},
        # eval_steer.py records fewer per-episode fields than eval_act.py, so the taxonomy is
        # best-effort: absent the geometry fields there is no bucket to assign.
        "fails": Counter(bucket(e) for e in later
                         if not e["success"] and "lateral_mm" in e),
        # Where the block ENDED on a failure. `never_entered` lumps two very different states --
        # frozen at the staging waypoint (x ~ 0.166, carried height z ~ 0.062) and jammed on the
        # lip (x ~ 0.23, block height) -- so the bucket alone cannot say whether a better latent
        # fails *closer to the slot* or just fails less often.
        "fail_xz": [(e["final_obj_pos"][0][0], e["final_obj_pos"][0][2]) for e in later
                    if not e["success"] and e.get("final_obj_pos")],
        "spawn": {(e["env"], e["episode_index_in_env"]): tuple(e.get("spawn_pos") or ())
                  for e in later},
    }


def pct(c: dict) -> str:
    lo, hi = wilson(c["k"], c["n"])
    return f"{c['k'] / c['n']:.3f} [{lo:.3f}, {hi:.3f}]  n={c['n']}"


def gate(run: Path) -> None:
    print("\n" + "=" * 78)
    print("GATE -- does eval_steer.py's noise path equal eval_act.py's?  (EXP_STEER 8a)")
    print("=" * 78)
    ref = load(run / "x0probe_act002_zeros.json")     # C1: eval_act, fixed_x0=zeros
    sub = load(run / "steergate_steerzero_act002.json")  # C2: eval_steer, z=0
    if ref is None or sub is None:
        print("  cells missing -- run scripts/run_steer_gate.sh")
        return

    for nm, c in (("C1 eval_act  x0=zeros", ref), ("C2 eval_steer z=0    ", sub)):
        an = c["cfg"].get("action_noise")
        print(f"  {nm}  {pct(c)}   action_noise={an}")
    assert ref["cfg"].get("action_noise") == sub["cfg"].get("action_noise") == 0.02, \
        "gate cells must both be at 2 % action noise; check provenance before reading anything"

    common = sorted(set(ref["succ"]) & set(sub["succ"]))
    disagree = [c for c in common if ref["succ"][c] != sub["succ"][c]]
    len_diff = [c for c in common if ref["len"][c] != sub["len"][c]]
    print(f"\n  shared episodes: {len(common)}")
    print(f"  outcome mismatches: {len(disagree)}"
          + (f"   e.g. {disagree[:5]}" if disagree else "   <- EPISODE-FOR-EPISODE IDENTICAL"))
    print(f"  length  mismatches: {len(len_diff)}")

    z, p = two_prop_z(sub["k"], sub["n"], ref["k"], ref["n"])
    d = sub["k"] / sub["n"] - ref["k"] / ref["n"]
    print(f"  rate delta (C2 - C1): {d:+.3f}   z={z:+.2f}  p={p:.4f}")

    if not disagree and not len_diff:
        print("\n  VERDICT: GATE PASSES in its strong form. The steering path reproduces the "
              "\n           eval_act path episode-for-episode; steered numbers are comparable.")
    elif p > 0.05:
        print("\n  VERDICT: gate passes on RATE but not per-episode. The two harnesses consume "
              "\n           randomness differently -- treat every steer-vs-eval_act comparison as "
              "\n           UNPAIRED, and find the divergence before claiming a paired result.")
    else:
        print("\n  VERDICT: GATE FAILS. The training/eval environment is not the environment the "
              "\n           83-point deficit was measured in. Stop; nothing downstream is readable.")


def probe(run: Path) -> None:
    print("\n" + "=" * 78)
    print("PROBE -- does x0 move anything under 2 % action noise?  (EXP_STEER 8b)")
    print("=" * 78)
    cells = [(nm, load(run / f)) for nm, f in PROBE]
    have = [(nm, c) for nm, c in cells if c is not None]
    if not have:
        print("  no probe cells -- run scripts/run_steer_gate.sh")
        return

    # references, both stochastic-x0, from the existing robustness sweep
    stoch_noise = load(run / "robust_act002.json")
    stoch_clean = load(run / "eval_ckpt_final_Rebot-PrecisionSlot-v0_s777.json")
    zeros_clean = load(run / "eval_x0zeros_Rebot-PrecisionSlot-v0_s777.json")

    print("\n  reference cells (stochastic x0 = fresh draw every 15-step refill)")
    for nm, c in (("clean", stoch_clean), ("2% action noise", stoch_noise)):
        if c:
            print(f"    {nm:<18} {pct(c)}")
    if zeros_clean:
        print(f"    {'x0=zeros, clean':<18} {pct(zeros_clean)}"
              "   <- is `zeros` a good draw absent noise?")

    print("\n  constant-x0 cells under 2 % action noise")
    for nm, c in have:
        # the bucket mix matters as much as the rate: 'success fell 60 points' and 'success fell
        # 60 points and every new failure is a gross_miss' are different findings
        mix = "  ".join(f"{b}={k}" for b, k in c["fails"].most_common(3))
        print(f"    {nm:<18} {pct(c)}   {mix}")
        if c["fail_xz"]:
            xs = sorted(p[0] for p in c["fail_xz"])
            zs = sorted(p[1] for p in c["fail_xz"])
            med = lambda v: v[len(v) // 2]  # noqa: E731
            frozen = sum(1 for x, _ in c["fail_xz"] if x < 0.18)
            print(f"    {'':<18} failed block: median x={med(xs):.3f} z={med(zs):.3f}"
                  f"   at staging (x<0.18): {frozen}/{len(c['fail_xz'])}"
                  f"   [stage_x=0.165, slot=0.245]")
    if len(have) < len(PROBE):
        print(f"    ({len(PROBE) - len(have)} cell(s) still running)")

    rates = {nm: c["k"] / c["n"] for nm, c in have}
    best, worst = max(rates, key=rates.get), min(rates, key=rates.get)
    spread = rates[best] - rates[worst]
    print(f"\n  spread: best {best}={rates[best]:.3f}  worst {worst}={rates[worst]:.3f}"
          f"  -> {spread * 100:.1f} pts")
    if len(have) >= 2:
        bc, wc = dict(have)[best], dict(have)[worst]
        z, p = two_prop_z(bc["k"], bc["n"], wc["k"], wc["n"])
        print(f"  best vs worst: z={z:+.2f}  p={p:.4g}"
              "   (post-hoc extremes of 5 -- treat p as descriptive)")
    if stoch_noise:
        bc = dict(have)[best]
        z, p = two_prop_z(bc["k"], bc["n"], stoch_noise["k"], stoch_noise["n"])
        d = rates[best] - stoch_noise["k"] / stoch_noise["n"]
        print(f"  best constant vs stochastic: {d:+.3f}  z={z:+.2f}  p={p:.4g}")

    print("\n  pre-registered beliefs (EXP_STEER 8b)")
    print(f"    6. spread > 15 pts          -> {'HELD' if spread > 0.15 else 'FALSIFIED'}"
          f"  ({spread * 100:.1f} pts)")
    print(f"    7. best > 0.30              -> {'HELD' if rates[best] > 0.30 else 'FALSIFIED'}"
          f"  ({rates[best]:.3f})")
    print(f"    8. zeros is not the best    -> "
          f"{'HELD' if best != 'zeros' else 'FALSIFIED'}  (best = {best})")

    # --- §10: does the good latent need per-chunk-position structure? ---
    bc = [(nm, load(run / f)) for nm, f in BCAST]
    bc = [(nm, c) for nm, c in bc if c is not None]
    if bc:
        print("\n  broadcast form (row 0 repeated across all 50 chunk positions --"
              " the family steering CAN express)")
        for nm, c in bc:
            full = dict(have).get(f"seed{nm[1:]}")
            tail = ""
            if full:
                z, p = two_prop_z(c["k"], c["n"], full["k"], full["n"])
                tail = (f"   vs full seed{nm[1:]} {full['k'] / full['n']:.3f}: "
                        f"{c['k'] / c['n'] - full['k'] / full['n']:+.3f}  p={p:.4g}")
            print(f"    {nm:<18} {pct(c)}{tail}")
        print("\n    If the broadcast cells track their full counterparts, x0-steering's action")
        print("    space is the right shape. If they collapse to ~0.15, the 50 chunk positions")
        print("    need DIFFERENT x0 and SteerCore's one-vector broadcast cannot represent the")
        print("    behaviour -- no amount of PPO would find it.")

    ex = [(nm, load(run / f)) for nm, f in EXTRA]
    ex = [(nm, c) for nm, c in ex if c is not None]
    if ex:
        allf = have + ex
        rr = sorted(c["k"] / c["n"] for _, c in allf)
        good = sum(1 for r in rr if r > 0.50)
        print(f"\n  full-draw population (n={len(allf)}): "
              + " ".join(f"{r:.3f}" for r in rr))
        print(f"    above 0.50: {good}/{len(allf)}   mean {sum(rr) / len(rr):.3f}"
              "   <- expected score of COMMITTING to a random draw")

    # seed 1 outside the 2 % condition: is it a NOISE-ROBUST latent or just a better one?
    ctl = [("seed1 clean", load(run / "x0probe_clean_s1.json")),
           ("seed1 @ 5% act noise", load(run / "x0probe_act005_s1.json"))]
    ctl = [(nm, c) for nm, c in ctl if c is not None]
    if ctl:
        print("\n  seed-1 controls (stochastic reference: clean 0.979, 5 % noise 0.000)")
        for nm, c in ctl:
            print(f"    {nm:<22} {pct(c)}")

    # --- oracle ceiling for EPISODE-LEVEL latent selection -------------------------------
    # Every --fixed-x0 cell draws exactly the same number of randn calls (none for x0, one
    # randn_like(action) per step), so the reset stream is untouched and the cells are paired on
    # BOTH the spawn and the noise realisation -- verified below, not assumed. That makes an
    # oracle meaningful: how well could a perfect per-episode chooser do, given these latents?
    # It bounds what a state-conditioned steerer can win over simply always using the best one.
    keys = sorted(set.intersection(*(set(c["succ"]) for _, c in have))) if len(have) > 1 else []
    if keys:
        ref = have[0][1]
        bad = [nm for nm, c in have[1:] for k in keys
               if max(abs(a - b) for a, b in zip(ref["spawn"][k], c["spawn"][k])) > 1e-4]
        print(f"\n  pairing check: {'IDENTICAL spawns across all cells' if not bad else 'DIFFER in ' + ','.join(sorted(set(bad)))}")
        orc = sum(any(c["succ"][k] for _, c in have) for k in keys) / len(keys)
        best_c = dict(have)[best]
        bk = sum(best_c["succ"][k] for k in keys) / len(keys)
        unsolved = sum(1 for k in keys if not any(c["succ"][k] for _, c in have))
        print(f"\n  oracle over the {len(have)} latents (>=1 succeeds): {orc:.3f}"
              f"   best single: {bk:.3f}   headroom for per-episode selection: "
              f"{(orc - bk) * 100:+.1f} pts")
        print(f"    episodes no latent solves: {unsolved}/{len(keys)}"
              "   <- these need something other than a better x0")

    print("\n  DECISION (EXP_STEER 8b):", end=" ")
    if len(have) < len(PROBE):
        print("INCOMPLETE -- wait for all five cells.")
    elif spread < 0.05:
        print("DO NOT LAUNCH PPO. Under noise the chunk is essentially independent of\n"
              "    x0: there is no mode to select, so the deficit is a capability failure, not a\n"
              "    sampling failure. Write it up; the fix is noise-augmented data, not steering.")
    elif rates[best] > 0.30:
        print("LAUNCH. Mechanism holds and a constant draw already recovers ground;\n"
              "    a state-conditioned steerer has strictly more to work with.")
    else:
        print("LAUNCH. x0 matters but no constant draw suffices -- which is exactly the\n"
              "    case a conditional steerer exists for, and the most interesting outcome.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/bc_armB_seed0")
    a = ap.parse_args()
    run = Path(a.run)
    gate(run)
    probe(run)
    print()


if __name__ == "__main__":
    main()
