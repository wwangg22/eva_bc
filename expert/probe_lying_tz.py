"""Offline cuRobo probe: lying-can grasp-plan feasibility vs close-height tz.

Extracts the exact failing states from demos_smoke_32ep.h5 (recorded during the
v6_lyingtz eval), rebuilds the planner world sim-free (replay_place_fail.Expert +
build_scene), and sweeps the grasp close-height tz directly (bypassing the
runner's `tz = clip(can_z + off, ...)` computation) to map which tz values plan
successfully for each failing state.

Obs decode (41-d policy obs, pick_place_v1):
  [0:8]   joint_pos_rel  (abs q = rel + default; arm default (0,-1.35,-0.3,-0.85,0,0))
  [8:16]  joint_vel_rel
  [16:32] objects_canonical: target-first pose7 (pos3 + quat4 XYZW, robot-root
          frame == env-local: robot base at (0,0,0) identity) x2 + 2 placed flags
  [32:34] basket_center_xy
  [34:41] last_action

Pure cuRobo + torch + numpy + h5py (no isaaclab). GPU required for the sweep.
Run (env_isaaclab6, ONLY when the GPU is free):
  python probe_lying_tz.py --decode-only          # verify state extraction (no GPU)
  python probe_lying_tz.py --cases 0,1            # sweep a chunk, append raw json
  python probe_lying_tz.py --report               # aggregate -> lying_tz_probe_results.md
"""

import argparse
import json
import math
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

H5 = os.path.join(HERE, "demos_smoke_32ep.h5")
RAW = os.path.join(HERE, "lying_tz_probe_raw.json")
MD = os.path.join(HERE, "lying_tz_probe_results.md")
Q_DEFAULT = np.array([0.0, -1.35, -0.3, -0.85, 0.0, 0.0])
# 0.012 = what v6 actually used for lying cans (clip floor: root z 0.008 + 0.002)
TZ_SWEEP = (0.012, 0.014, 0.016, 0.018, 0.020, 0.022, 0.024, 0.026, 0.028)
TCP_OFF = 0.075

# (label, demo, can, t_rule, expected_xy_from_log)
# t_rule: ("settle_end", None)      -> t=10 (settle mark t=0 + 10 hold steps)
#         ("mark", seg)             -> t of that approach mark (plan-time state)
#         ("reopen_end", seg)       -> t of that reopen mark + 10 hold steps
#                                      (= state of the next, PLAN-FAILing attempt)
CASES = [
    ("demo_6/object_a", "demo_6", "object_a", ("reopen_end", "object_a:f0:r0"), (0.242, -0.068)),
    ("demo_7/object_a", "demo_7", "object_a", ("reopen_end", "object_a:f0:r1"), (0.239, 0.153)),
    ("demo_19/object_a", "demo_19", "object_a", ("settle_end", None), (0.258, -0.105)),
    ("demo_19/object_b", "demo_19", "object_b", ("mark", "object_b:f0:g0"), None),
    ("demo_28/object_b", "demo_28", "object_b", ("mark", "object_b:f0:g0"), None),
    ("demo_24/object_b", "demo_24", "object_b", ("settle_end", None), (0.167, -0.142)),
]


def axis_from_quat_xyzw(q):
    """World cylinder axis = body Y of the Y-up can. Verbatim run_expert_v1.can_axis."""
    x, y, z, w = [float(v) for v in q]
    return np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])


def case_t(segments, rule):
    kind, seg = rule
    if kind == "settle_end":
        return 10
    for s in segments:
        if s["seg"] == seg and s["phase"] == ("approach" if kind == "mark" else "reopen"):
            return s["t"] + (10 if kind == "reopen_end" else 0)
    raise KeyError(f"segment {seg!r} not found for rule {rule}")


def extract_cases():
    out = []
    with h5py.File(H5, "r") as f:
        for label, demo, can, rule, exp_xy in CASES:
            g = f["data"][demo]
            segs = json.loads(g.attrs["segments"])
            t = case_t(segs, rule)
            obs = g["obs"]["policy"][t]
            q6 = (obs[0:6].astype(np.float64) + Q_DEFAULT).tolist()
            slots = []
            for base in (16, 23):
                pos = obs[base:base + 3].astype(np.float64)
                quat = obs[base + 3:base + 7].astype(np.float64)
                slots.append({"pos": pos, "quat": quat,
                              "axis": axis_from_quat_xyzw(quat),
                              "r": float(np.hypot(pos[0], pos[1]))})
            basket_xy = obs[32:34].astype(np.float64).tolist()
            # slot -> can identity: log xy match if we have one; else the runner's
            # fetch order (sorted by |xy| radius) pins which can this fetch targeted:
            # demo_19/object_b was the SECOND fetch (larger r), demo_28/object_b the
            # first (smaller r).
            if exp_xy is not None:
                errs = [float(np.hypot(s["pos"][0] - exp_xy[0], s["pos"][1] - exp_xy[1]))
                        for s in slots]
                si = int(np.argmin(errs))
                xy_err = errs[si]
            else:
                rs = [s["r"] for s in slots]
                si = int(np.argmax(rs)) if label == "demo_19/object_b" else int(np.argmin(rs))
                xy_err = None
            s = slots[si]
            lying = abs(float(s["axis"][2])) < 0.5
            out.append({
                "label": label, "t": int(t), "q6": q6, "basket_xy": basket_xy,
                "slot": si, "xy_err_vs_log": xy_err,
                "pos": s["pos"].tolist(), "r": s["r"],
                "quat_xyzw": s["quat"].tolist(),
                "axis": s["axis"].tolist(), "lying": lying,
                "other_pos": slots[1 - si]["pos"].tolist(),
                "flags": obs[30:32].tolist(),
            })
    return out


def print_cases(cases):
    for c in cases:
        p, ax = c["pos"], c["axis"]
        err = "n/a" if c["xy_err_vs_log"] is None else f"{c['xy_err_vs_log'] * 1000:.1f}mm"
        print(f"{c['label']:>18} t={c['t']:<4} slot={c['slot']} log-err={err:>6} "
              f"xy=({p[0]:+.3f},{p[1]:+.3f}) z={p[2]:.3f} r={c['r']:.3f} "
              f"lying={c['lying']} axis=({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})")
        print(f"{'':>18} q6={np.round(c['q6'], 3).tolist()} "
              f"basket=({c['basket_xy'][0]:.3f},{c['basket_xy'][1]:.3f}) flags={c['flags']}")
        if c["xy_err_vs_log"] is not None and c["xy_err_vs_log"] > 0.005:
            print(f"{'':>18} *** XY CROSS-CHECK FAIL (> 5 mm) — decode suspect ***")


def plan_grasp_tz(ex, q6, can_xy, tz, align_axis_xy):
    """run_expert_v1.Expert.plan_grasp with tz passed directly (no can_z+off)."""
    from spike_plan_grasp import table_candidates

    cands = table_candidates(tuple(can_xy), K=ex.K, target_z=float(tz),
                             align_axis_xy=align_axis_xy)
    if not cands:
        return "no-candidates", 0, None, None
    uniq = len({tuple(np.round(p, 5)) for p, _ in cands})
    res = ex.planner.plan_grasp(
        grasp_poses=ex.goalset(cands), current_state=ex.js(q6),
        grasp_approach_axis="z", grasp_approach_offset=0.10, grasp_approach_in_tool_frame=False,
        grasp_lift_axis="z", grasp_lift_offset=0.08, grasp_lift_in_tool_frame=False,
    )
    if not bool(res.success.any().item()):
        return "trajopt-fail", uniq, None, None
    return "success", uniq, int(res.goalset_index.view(-1)[0].item()), res


# NOTE: FK fingertip-z at the grasp config was attempted and dropped: the grasp
# table's pocket->tool offset is NOT the place-goal TCP_OFF relation (inverse of
# _pose_from gives pocket errors of ~3-5 cm vs the constructed (can_xy, tz)
# pocket), and the grasp_interpolated_trajectory endpoint sits at the 0.10 m
# approach standoff, not the grasp pose. Not trivially available offline.


def run_sweep(cases, case_idx, tzs):
    from replay_place_fail import Expert, build_scene

    ex = Expert()
    raw = json.load(open(RAW)) if os.path.exists(RAW) else {}
    for ci in case_idx:
        c = cases[ci]
        names = ("object_a", "object_b")
        can = c["label"].split("/")[1]
        other = names[1 - names.index(can)]
        world = build_scene(c["basket_xy"],
                            {can: (c["pos"], False), other: (c["other_pos"], False)})
        ex.planner.update_world(world)
        ax = c["axis"]
        align = (float(ax[0]), float(ax[1])) if c["lying"] else None
        for tz in tzs:
            status, n_cands, gi, res = plan_grasp_tz(
                ex, c["q6"], (c["pos"][0], c["pos"][1]), tz, align)
            row = {"case": c["label"], "tz": tz, "n_cands": n_cands,
                   "result": status, "gi": gi}
            raw[f"{c['label']}@{tz:.3f}"] = row
            print(f"  {c['label']:>18} tz={tz:.3f} cands={n_cands:>2} {status}"
                  + (f" gi={gi}" if gi is not None else ""))
            json.dump(raw, open(RAW, "w"), indent=1)
    print(f"raw -> {RAW}")


def write_report(cases):
    raw = json.load(open(RAW))
    sym = {"success": "OK", "trajopt-fail": "TF", "no-candidates": "NC"}
    lines = ["# lying-can tz feasibility probe (v6 failing states, offline cuRobo)",
             "",
             "Grasp-plan result per (failing state, close-height tz), tz passed directly to",
             "`table_candidates`/`plan_grasp` (runner formula bypassed). World = both cans on",
             "the table, none held; start q6 = recorded joints at the failing attempt.",
             "",
             "OK = plan success (goalset idx in parens), TF = trajopt-fail, NC = no-candidates.",
             "",
             "| case | " + " | ".join(f"tz={t:.3f}" for t in TZ_SWEEP) + " |",
             "|---|" + "---|" * len(TZ_SWEEP)]
    for c in cases:
        cells = []
        for tz in TZ_SWEEP:
            r = raw.get(f"{c['label']}@{tz:.3f}")
            if r is None:
                cells.append("-")
                continue
            cell = sym[r["result"]] + (f" ({r['gi']})" if r["gi"] is not None else "")
            cells.append(cell)
        lines.append(f"| {c['label']} | " + " | ".join(cells) + " |")
    lines += ["", "Candidate counts (unique poses in the K=16 goalset):", ""]
    for c in cases:
        counts = [str(raw[f"{c['label']}@{tz:.3f}"]["n_cands"])
                  for tz in TZ_SWEEP if f"{c['label']}@{tz:.3f}" in raw]
        lines.append(f"- {c['label']}: " + ", ".join(counts))
    lines += ["", "## Decoded failing states (from demos_smoke_32ep.h5)", "",
              "| case | t | xy | r | z | axis (world) | log xy err | q6 |",
              "|---|---|---|---|---|---|---|---|"]
    for c in cases:
        p, ax = c["pos"], c["axis"]
        err = "-" if c["xy_err_vs_log"] is None else f"{c['xy_err_vs_log'] * 1000:.1f} mm"
        lines.append(
            f"| {c['label']} | {c['t']} | ({p[0]:+.3f},{p[1]:+.3f}) | {c['r']:.3f} | "
            f"{p[2]:.3f} | ({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}) | {err} | "
            f"{np.round(c['q6'], 3).tolist()} |")
    lines += ["", f"basket centers: " + "; ".join(
        f"{c['label']} ({c['basket_xy'][0]:.3f},{c['basket_xy'][1]:.3f})" for c in cases)]
    lines += ["", "## Findings", "",
              "1. **v6's actual lying tz was 0.012, not 0.014**: lying cans report root z = 0.008",
              "   in the obs/sim (can origin is ~4 mm below the geometric center; upright rest",
              "   z = 0.015, not 0.018), so `clip(0.008 + 0.002, 0.012, 0.045)` hit the clip FLOOR.",
              "2. **The live trajopt PLAN-FAILs do not reproduce from a fresh planner**: every",
              "   lying case (incl. demo_19/object_a from the untouched settle state, and demo_6's",
              "   exact live attempt-1 call with tz=0.012 + exclude=(14,)) plans successfully at",
              "   EVERY tz 0.012-0.028. The live failures at r 0.24-0.29 are long-running-process",
              "   solver-state/stochastic effects (consistent with the runner's note that",
              "   reset_seed() shifted plan_fail 4 -> 17), not tz geometry.",
              "3. **NO-CANDIDATES (demo_24, r=0.219) is tz-sensitive**: empty at tz <= 0.014,",
              "   1-2 candidates at 0.016-0.018, full 16 at tz >= 0.020 (higher target_z lets the",
              "   mid-band (r >= 0.223) table entries pass the score gate). Raising lying tz to",
              "   ~0.020 buys goalset coverage below the lying band's r >= 0.270 floor.",
              "4. **demo_28/object_b was UPRIGHT at its first attempt** (axis z = +1.00,",
              "   z = 0.015), not lying: its 3 air-closes at tz = 0.031 are an upright-family",
              "   failure. A can in that episode is knocked over lying only later (t~300).",
              "5. Obs decode confirmed: [0:8] joint_pos_rel vs default (0,-1.35,-0.3,-0.85,0,0)",
              "   + gripper (0.04,0.04); [16:32] objects_canonical target-first, pos3 + quat4",
              "   XYZW in robot-root frame == env-local (base at origin, identity); [32:34]",
              "   basket_center_xy. Log xy cross-checks all within 0.6 mm."]
    with open(MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report -> {MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--cases", default=None, help="comma case indices (default all)")
    ap.add_argument("--tz", default=None, help="comma tz values (default full sweep)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    cases = extract_cases()
    print_cases(cases)
    if args.decode_only:
        return
    if args.report:
        write_report(cases)
        return
    idx = list(range(len(cases))) if args.cases is None else [int(i) for i in args.cases.split(",")]
    tzs = list(TZ_SWEEP) if args.tz is None else [float(t) for t in args.tz.split(",")]
    run_sweep(cases, idx, tzs)


if __name__ == "__main__":
    main()
