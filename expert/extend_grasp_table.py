"""Extend the proven grasp table into its measured coverage gaps (the "annulus" blocker).

The 12953-entry table (q/pocket/fwd) has NO entries below r~0.221 (any z) and its
lying band (z<=0.020) only covers r 0.270-0.326; dropped/rolled cans land at
r 0.16-0.27 and up to ~0.36 lying, where table_candidates honestly returns [] and
the expert plan-fails (Gate-1 perturb capped at 71.9%, DAgger takeovers at
r=0.192/0.201/0.207 unservable).

Two subcommands:

  synthesize (CPU-only, no cuRobo/CUDA):
      Build CANDIDATE entries for the gap grid and save extension_candidates.pt.
      Sources: (a) "analytic" — the proven grasp-family sampler grasp_candidates()
      swept over wider elevations, vertically shifted to the target pocket z;
      (b) "edge" — nearest existing table entries by (r, z), stored by REFERENCE
      (seed row + azimuth rotation + residual shift, the exact construction
      table_candidates uses at serve time; poses are FK'd from the seed q in the
      GPU stage — kinematics is CUDA-only). No pose is ever rigidly translated
      laterally beyond the serve-proven residual envelope.

  verify (GPU — do NOT run while the GPU is busy):
      Offline planner (Expert cfg: max_goalset=16, cuboid cache 16), per-target
      scene = table + target can (run_expert_v1.build_scene conventions), the
      EXACT Expert.plan_grasp call shape from Q_HOME, goalsets padded to K=16
      (fixed shapes; the first planner call in the process is a pose-goalset
      plan_grasp). A candidate is verified iff it is the goalset WINNER
      (res.goalset_index) of a fully successful solve; its table q is the final
      state of res.grasp_trajectory (== the planner's own lift-start config),
      pocket/fwd re-derived by FK for exact serve-time consistency.
      Writes grasp_table_extended.pt (original rows byte-identical, verified rows
      appended) + extend_report.json.

All quats WXYZ (cuRobo convention) end to end — nothing is converted.

Run:
  conda activate env_isaaclab6 && python extend_grasp_table.py synthesize
  # GPU (later): python extend_grasp_table.py verify --limit 6 --rounds 1   (smoke)
  #              python extend_grasp_table.py verify                        (full)
Serve extended table: REBOT_GRASP_TABLE=$PWD/grasp_table_extended.pt (see
spike_plan_grasp.GRASP_TABLE_PT).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from spike_plan_grasp import (  # noqa: E402
    CAN_CENTER_Z, CAN_DIMS, GRASP_TABLE_PT, ROBOT_YML, TCP_OFFSET, grasp_candidates,
)

Q_HOME = [0.0, -1.35, -0.3, -0.85, 0.0, 0.0]
K = 16

# The table's pocket convention, measured by FK over all 12953 entries (GPU probe,
# 2026-08-01): pocket = gripper_end_pos + 0.0480 * fwd, std 0.0000, perp residual
# <= 0.4 mm. I.e. the proven pinch point sits 48 mm along the finger axis — 27 mm
# SHORT of the 75 mm TCP that grasp_candidates() targets. Analytic candidates are
# therefore shifted +27 mm along fwd so the proven pocket point (not the TCP) lands
# on the can, and verified entries store pocket = pos + POCKET_ALONG * fwd so
# table_candidates serves them exactly like original entries.
POCKET_ALONG = 0.048

CAND_PT = os.path.join(HERE, "extension_candidates.pt")
EXT_PT = os.path.join(HERE, "grasp_table_extended.pt")
REPORT_JSON = os.path.join(HERE, "extend_report.json")

# ---- target grid (the measured gaps) ------------------------------------------------
# lying cans: pocket z 0.012-0.020 (Expert targets tz = can_z + 0.012, can_z ~ 0.012)
LYING_TZ = (0.012, 0.016, 0.020)
LYING_R = tuple(round(r, 2) for r in np.arange(0.15, 0.3701, 0.02))   # <=0.02 spacing
# upright pocket band below the existing r floor (0.221)
POCKET_TZ = (0.024, 0.031, 0.038, 0.045)
POCKET_R = (0.15, 0.17, 0.19, 0.21, 0.23)
# table_candidates rotates entries by an ARBITRARY az_t - az_n about base z, so any
# azimuth is served from any entry; a few azimuths only add verification diversity
# across the spawn sector (az +/-45 deg).
AZ_DEG = (-40.0, 0.0, 40.0)

# analytic family sweep: proven dazs, elevations widened past the table's 4-33 deg
ANALYTIC_DAZS = (-175.0, -159.0, -145.0)
ANALYTIC_ELEVS = (8.0, 16.0, 24.0, 32.0, 40.0)
# edge source: same score metric as table_candidates, relaxed past its 0.035 serve
# cutoff (these are candidates for verification, not serves)
EDGE_SCORE_MAX = 0.06
EDGE_MAX_DZ = 0.030
EDGE_K = 10

SRC_ANALYTIC, SRC_EDGE = 0, 1
SRC_NAME = {SRC_ANALYTIC: "analytic", SRC_EDGE: "edge"}


def rotz_apply(th, vec3):
    c, s = math.cos(th), math.sin(th)
    x, y, z = vec3
    return np.array([c * x - s * y, s * x + c * y, z])


def rotz_quat_wxyz(th, qwxyz):
    """Premultiply wxyz quat by a rotation about base z (table_candidates math)."""
    qw, qz = math.cos(th / 2), math.sin(th / 2)
    w, x, y, z = qwxyz
    return np.array([qw * w - qz * z, qw * x - qz * y, qw * y + qz * x, qw * z + qz * w])


def quat_wxyz_xhat(qwxyz):
    w, x, y, z = qwxyz
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])


def build_targets():
    """(band, r, az_deg, tz) grid -> list of scene dicts with the can pose the verify
    scene will use (build_scene convention: z clamped to >=0.02; Expert tz offsets
    inverted: lying can_z = tz - 0.012, upright can_z = tz - 0.016)."""
    targets = []
    for band, rs, tzs, off in (("lying", LYING_R, LYING_TZ, 0.012),
                               ("pocket", POCKET_R, POCKET_TZ, 0.016)):
        for tz in tzs:
            for r in rs:
                for az in AZ_DEG:
                    a = math.radians(az)
                    targets.append({
                        "scene_id": len(targets), "band": band,
                        "r": float(r), "az_deg": float(az), "tz": float(tz),
                        "xy": [r * math.cos(a), r * math.sin(a)],
                        "can_z": max(tz - off, 0.02),
                    })
    return targets


def r_bin(r):
    return round(math.floor(r / 0.02) * 0.02, 2)


# ---------------------------------- synthesize --------------------------------------

def cmd_synthesize(args):
    d = torch.load(GRASP_TABLE_PT, map_location="cpu", weights_only=False)
    pk = d["pocket"].numpy()
    fwd_tab = d["fwd"].numpy()
    n_tab = len(pk)
    r_tab = np.linalg.norm(pk[:, :2], axis=1)
    az_tab = np.arctan2(pk[:, 1], pk[:, 0])

    targets = build_targets()
    rows = {k: [] for k in ("pos", "quat", "pocket", "fwd", "source", "seed_idx",
                            "seed_rot", "target_r", "target_z", "target_az", "scene_id")}

    def add(pos, quat, pocket, fwd, source, seed_idx, seed_rot, t):
        rows["pos"].append(pos)
        rows["quat"].append(quat)
        rows["pocket"].append(pocket)
        rows["fwd"].append(fwd)
        rows["source"].append(source)
        rows["seed_idx"].append(seed_idx)
        rows["seed_rot"].append(seed_rot)
        rows["target_r"].append(t["r"])
        rows["target_z"].append(t["tz"])
        rows["target_az"].append(math.radians(t["az_deg"]))
        rows["scene_id"].append(t["scene_id"])

    nan3, nan4 = np.full(3, np.nan), np.full(4, np.nan)
    for t in targets:
        xy, tz = t["xy"], t["tz"]
        pocket_t = np.array([xy[0], xy[1], tz])
        # (a) analytic: proven family at the target xy, shifted vertically from
        # CAN_CENTER_Z to the target pocket z, then +27 mm along fwd so the proven
        # 48 mm pocket point (POCKET_ALONG) lands on the can instead of the TCP.
        for p_, q_ in grasp_candidates(tuple(xy), dazs=ANALYTIC_DAZS, elevs=ANALYTIC_ELEVS):
            quat = np.array(q_)
            fwd = -quat_wxyz_xhat(quat)
            fwd /= np.linalg.norm(fwd)
            pos = (np.array(p_) + [0.0, 0.0, tz - CAN_CENTER_Z]
                   + (TCP_OFFSET - POCKET_ALONG) * fwd)
            add(pos, quat, pocket_t, fwd, SRC_ANALYTIC, -1, 0.0, t)
        # (b) edge: nearest proven entries by (r, z); pose deferred to GPU FK
        # (kinematics is CUDA-only) — stored as seed row + azimuth rotation; the
        # residual shift to pocket_t is re-derived in verify exactly as
        # table_candidates does at serve time.
        score = np.abs(r_tab - t["r"]) + 2.0 * np.abs(pk[:, 2] - tz)
        ok = (score < EDGE_SCORE_MAX) & (np.abs(pk[:, 2] - tz) <= EDGE_MAX_DZ)
        sel = np.argsort(score)
        sel = sel[ok[sel]]
        if len(sel) > EDGE_K:  # spread like table_candidates, not just the closest
            sel = sel[np.linspace(0, len(sel) - 1, EDGE_K).astype(int)]
        az_t = math.radians(t["az_deg"])
        for i in sel:
            th = az_t - float(az_tab[i])
            add(nan3, nan4, pocket_t, rotz_apply(th, fwd_tab[i]),
                SRC_EDGE, int(i), th, t)

    out = {
        "pos": torch.tensor(np.array(rows["pos"]), dtype=torch.float64),
        "quat": torch.tensor(np.array(rows["quat"]), dtype=torch.float64),
        "pocket": torch.tensor(np.array(rows["pocket"]), dtype=torch.float32),
        "fwd": torch.tensor(np.array(rows["fwd"]), dtype=torch.float32),
        "source": torch.tensor(rows["source"], dtype=torch.int8),
        "seed_idx": torch.tensor(rows["seed_idx"], dtype=torch.int64),
        "seed_rot": torch.tensor(rows["seed_rot"], dtype=torch.float64),
        "target_r": torch.tensor(rows["target_r"], dtype=torch.float32),
        "target_z": torch.tensor(rows["target_z"], dtype=torch.float32),
        "target_az": torch.tensor(rows["target_az"], dtype=torch.float32),
        "scene_id": torch.tensor(rows["scene_id"], dtype=torch.int32),
        "needs_verify": True,
        "targets": targets,
        "meta": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_table": GRASP_TABLE_PT, "source_table_n": n_tab,
            "quat_convention": "wxyz",
            "analytic_dazs": ANALYTIC_DAZS, "analytic_elevs": ANALYTIC_ELEVS,
            "edge_score_max": EDGE_SCORE_MAX, "edge_k": EDGE_K,
        },
    }
    torch.save(out, args.out)
    n = len(rows["source"])
    print(f"saved {n} candidates ({int((out['source']==SRC_ANALYTIC).sum())} analytic, "
          f"{int((out['source']==SRC_EDGE).sum())} edge) over {len(targets)} scenes "
          f"-> {args.out}\n")

    # coverage summary: r-bin x z-band, existing table vs new candidates
    tr = np.array(rows["target_r"])
    tzs = np.array(rows["target_z"])
    src = np.array(rows["source"])
    lying_t, lying_e = tzs <= 0.020, pk[:, 2] <= 0.020
    print("coverage (r-bin x z-band): existing table entries | new candidates (analytic+edge)")
    print(f"{'r_bin':>6} | {'lying_tab':>9} {'lying_new':>9} | {'pocket_tab':>10} {'pocket_new':>10}")
    for rb in sorted({r_bin(r) for r in np.concatenate([tr, r_tab])}):
        in_t = (tr >= rb) & (tr < rb + 0.02)
        in_e = (r_tab >= rb) & (r_tab < rb + 0.02)
        ln, pn = int((in_t & lying_t).sum()), int((in_t & ~lying_t).sum())
        la = int((in_t & lying_t & (src == SRC_ANALYTIC)).sum())
        pa = int((in_t & ~lying_t & (src == SRC_ANALYTIC)).sum())
        print(f"{rb:6.2f} | {int((in_e & lying_e).sum()):9d} "
              f"{f'{ln} ({la}a+{ln-la}e)' if ln else '0':>9} | "
              f"{int((in_e & ~lying_e).sum()):10d} "
              f"{f'{pn} ({pa}a+{pn-pa}e)' if pn else '0':>10}")


# ------------------------------------ verify ----------------------------------------

def cmd_verify(args):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Cuboid, Scene
    from curobo.types import GoalToolPose, JointState

    cand = torch.load(args.candidates, map_location="cpu", weights_only=False)
    orig = torch.load(GRASP_TABLE_PT, map_location="cpu", weights_only=False)
    orig_pk = orig["pocket"].numpy()

    t0 = time.time()
    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YML, scene_model=None, collision_cache={"cuboid": 16}, max_goalset=K,
    )
    planner = MotionPlanner(cfg)
    # REQUIRED (measured 2026-08-01): without warmup, lazy cuda-graph capture leaves
    # the planner corrupted after ~2 plan_grasp calls — every later approach plan
    # fails ("Start or End state in collision"). With warmup: 12/12 across repeats.
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print(f"planner init+warmup: {time.time()-t0:.1f}s; tool_frames={planner.tool_frames}")

    def fk(q):  # (M,6) cpu float -> pos (M,3), quat wxyz (M,4) numpy
        st = planner.kinematics.compute_kinematics(JointState.from_position(
            q.cuda().float(), joint_names=planner.joint_names))
        return (st.tool_poses.position.reshape(-1, 3).cpu().numpy(),
                st.tool_poses.quaternion.reshape(-1, 4).cpu().numpy())

    # materialize edge poses: FK the seed rows once, then rotate + residual-shift
    # (identical to table_candidates's serve-time construction)
    seed_rows = sorted({int(i) for i in cand["seed_idx"].tolist() if i >= 0})
    seed_pose = {}
    if seed_rows:
        ps, qs = fk(orig["q"][seed_rows])
        seed_pose = {row: (ps[j], qs[j]) for j, row in enumerate(seed_rows)}

    n = len(cand["source"])
    pos_all, quat_all = cand["pos"].numpy(), cand["quat"].numpy()
    pocket_all = cand["pocket"].numpy()

    def cand_pose(i):
        if int(cand["source"][i]) == SRC_ANALYTIC:
            return pos_all[i], quat_all[i]
        row, th = int(cand["seed_idx"][i]), float(cand["seed_rot"][i])
        p_seed, q_seed = seed_pose[row]
        p_rot = rotz_apply(th, p_seed)
        pk_rot = rotz_apply(th, orig_pk[row])
        return p_rot + (pocket_all[i] - pk_rot), rotz_quat_wxyz(th, q_seed)

    def goalset(idx_list):  # pad cyclically to fixed K; returns (goal, idx_map)
        idx_map = list(idx_list)
        while len(idx_map) < K:
            idx_map.append(idx_map[len(idx_map) % len(idx_list)])
        pos = torch.zeros(1, 1, 1, K, 3, device="cuda")
        quat = torch.zeros(1, 1, 1, K, 4, device="cuda")
        for j, ci in enumerate(idx_map):
            p_, q_ = cand_pose(ci)
            pos[0, 0, 0, j] = torch.tensor(np.asarray(p_, dtype=np.float32))
            quat[0, 0, 0, j] = torch.tensor(np.asarray(q_, dtype=np.float32))
        return GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat), idx_map

    start = JointState.from_position(
        torch.tensor([Q_HOME], device="cuda", dtype=torch.float32),
        joint_names=planner.joint_names)

    scenes = [t for t in cand["targets"]
              if (args.band == "all" or t["band"] == args.band)
              and args.r_min <= t["r"] <= args.r_max]
    if args.limit:
        scenes = scenes[: args.limit]
    by_scene = {}
    for i in range(n):
        by_scene.setdefault(int(cand["scene_id"][i]), []).append(i)

    verified = []  # dicts: cand idx, q, pocket, fwd, pocket_err
    n_calls = 0
    fail_stats = {}  # planner status string -> count (per-phase failure anatomy)
    for t in scenes:
        ids = by_scene.get(t["scene_id"], [])
        if not ids:
            continue
        scene_fails = []
        xy = t["xy"]
        sc = Scene(cuboid=[
            Cuboid(name="table", dims=[1.2, 1.2, 0.05], pose=[0.25, 0, -0.027, 1, 0, 0, 0]),
            Cuboid(name="target_can", dims=list(CAN_DIMS),
                   pose=[xy[0], xy[1], t["can_z"], 1, 0, 0, 0]),
        ])
        # source-pure chunks -> clean analytic-vs-edge attribution (a mixed goalset
        # would let the cheaper family shadow the other in winner selection)
        chunks = []
        for src in (SRC_ANALYTIC, SRC_EDGE):
            grp = sorted(i for i in ids if int(cand["source"][i]) == src)
            chunks += [grp[j:j + K] for j in range(0, len(grp), K)]
        got = []
        for chunk in chunks:
            pool = list(chunk)
            harvested = 0
            for _ in range(args.attempts):
                if not pool:
                    break
                planner.update_world(sc)  # production pattern: refresh before every plan
                goal, idx_map = goalset(pool)
                res = planner.plan_grasp(
                    grasp_poses=goal, current_state=start,
                    grasp_approach_axis="z", grasp_approach_offset=0.10,
                    grasp_approach_in_tool_frame=False,
                    grasp_lift_axis="z", grasp_lift_offset=0.08,
                    grasp_lift_in_tool_frame=False,
                )
                n_calls += 1
                ok = bool(res.success.any().item())
                if not ok:
                    stat = res.status or "?"
                    fail_stats[stat] = fail_stats.get(stat, 0) + 1
                    if res.goalset_index is None:
                        # no goal in the set was even reachable: chunk is dead
                        scene_fails.append(
                            f"{SRC_NAME[int(cand['source'][pool[0]])]}: {stat}")
                        break
                    # goalset found a winner but approach/grasp/lift failed:
                    # drop that winner and retry the rest (Expert's exclude-retry)
                    bad = idx_map[int(res.goalset_index.view(-1)[0].item())]
                    scene_fails.append(f"{SRC_NAME[int(cand['source'][bad])]}: {stat}")
                    pool.remove(bad)
                    continue
                ci = idx_map[int(res.goalset_index.view(-1)[0].item())]
                # grasp config = last VALID row of the trimmed interpolated grasp
                # trajectory (bug #692: raw js_solution tail rows can hold stale
                # states — measured 0.100 m off, exactly the approach pose). The
                # trajectory carries ALL model joints (arm + fingers): select the
                # arm joints by name. FK acceptance check guards the entry.
                traj = res.grasp_interpolated_trajectory
                names = getattr(traj, "joint_names", None)
                jidx = ([names.index(j) for j in planner.joint_names] if names
                        else list(range(len(planner.joint_names))))
                flat = traj.position.reshape(-1, traj.position.shape[-1])
                lt = res.grasp_interpolated_last_tstep
                nlast = int(torch.as_tensor(lt).view(-1)[0].item()) if lt is not None \
                    else flat.shape[0]
                qg = flat[max(nlast - 1, 0)][jidx].detach().cpu().clone()
                p_fk, q_fk = fk(qg.view(1, -1))
                wp = np.asarray(cand_pose(ci)[0], dtype=float)
                if np.linalg.norm(p_fk[0] - wp) > 0.005:
                    fail_stats["extract-reject"] = fail_stats.get("extract-reject", 0) + 1
                    scene_fails.append(f"{SRC_NAME[int(cand['source'][ci])]}: extract-reject")
                    pool.remove(ci)
                    continue
                fwd = -quat_wxyz_xhat(q_fk[0])
                fwd /= np.linalg.norm(fwd)
                pocket = p_fk[0] + POCKET_ALONG * fwd  # table pocket convention
                verified.append({
                    "cand": ci, "q": qg, "pocket": pocket, "fwd": fwd,
                    "pocket_err": float(np.linalg.norm(pocket - pocket_all[ci])),
                })
                got.append(ci)
                pool.remove(ci)
                harvested += 1
                if harvested >= args.rounds:
                    break
        print(f"{t['band']:>6} r={t['r']:.2f} az={t['az_deg']:+.0f} tz={t['tz']:.3f}: "
              f"{len(got)}/{len(ids)} verified "
              f"({sum(1 for c in got if int(cand['source'][c]) == SRC_ANALYTIC)}a"
              f"+{sum(1 for c in got if int(cand['source'][c]) == SRC_EDGE)}e)"
              + (f"  fails: {'; '.join(scene_fails)}" if scene_fails else ""),
              flush=True)

    # ---- outputs
    report_bins = {}
    scene_ids = {t["scene_id"] for t in scenes}
    in_run = [i for i in range(n) if int(cand["scene_id"][i]) in scene_ids]
    got_set = {v["cand"] for v in verified}
    for i in in_run:
        key = (("lying" if float(cand["target_z"][i]) <= 0.020 else "pocket"),
               r_bin(float(cand["target_r"][i])), SRC_NAME[int(cand["source"][i])])
        b = report_bins.setdefault(key, {"candidates": 0, "verified": 0})
        b["candidates"] += 1
        b["verified"] += int(i in got_set)
    report = {
        "summary": {
            "n_scenes": len(scenes), "n_candidates": len(in_run),
            "n_verified": len(verified), "n_plan_calls": n_calls,
            "fail_statuses": fail_stats,
            "rounds": args.rounds, "limit": args.limit,
            "band": args.band, "r_min": args.r_min, "r_max": args.r_max,
            "mean_pocket_err": round(float(np.mean([v["pocket_err"] for v in verified])), 5)
            if verified else None,
        },
        "bins": [{"band": k[0], "r_bin": k[1], "source": k[2], **b,
                  "pass_rate": round(b["verified"] / b["candidates"], 3)}
                 for k, b in sorted(report_bins.items())],
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print("\nSUMMARY:", json.dumps(report["summary"], indent=2))
    print("report ->", args.report)

    if verified:
        ext = {
            "q": torch.cat([orig["q"], torch.stack([v["q"] for v in verified]).to(orig["q"].dtype)]),
            "pocket": torch.cat([orig["pocket"], torch.tensor(
                np.array([v["pocket"] for v in verified]), dtype=orig["pocket"].dtype)]),
            "fwd": torch.cat([orig["fwd"], torch.tensor(
                np.array([v["fwd"] for v in verified]), dtype=orig["fwd"].dtype)]),
            "extension_meta": {
                "n_original": len(orig["q"]), "n_extension": len(verified),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "candidates_file": args.candidates,
                "source": [SRC_NAME[int(cand["source"][v["cand"]])] for v in verified],
                "target_r": [float(cand["target_r"][v["cand"]]) for v in verified],
                "target_z": [float(cand["target_z"][v["cand"]]) for v in verified],
                "pocket_err": [v["pocket_err"] for v in verified],
            },
        }
        torch.save(ext, args.out)
        print(f"extended table: {len(orig['q'])} original + {len(verified)} verified "
              f"-> {args.out}")
        print(f"serve it: REBOT_GRASP_TABLE={args.out}")
    else:
        print("no candidates verified; extended table NOT written")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthesize", help="CPU: generate gap candidates")
    s.add_argument("--out", default=CAND_PT)
    v = sub.add_parser("verify", help="GPU: plan-verify candidates, write extended table")
    v.add_argument("--candidates", default=CAND_PT)
    v.add_argument("--out", default=EXT_PT)
    v.add_argument("--report", default=REPORT_JSON)
    v.add_argument("--limit", type=int, default=None, help="verify only the first N scenes")
    v.add_argument("--band", choices=("lying", "pocket", "all"), default="all",
                   help="restrict scenes to one z-band (filtered before --limit)")
    v.add_argument("--r-min", type=float, default=0.0)
    v.add_argument("--r-max", type=float, default=1.0)
    v.add_argument("--rounds", type=int, default=2,
                   help="max winners harvested per 16-goalset chunk")
    v.add_argument("--attempts", type=int, default=4,
                   help="max plan calls per chunk (failed winners are excluded and retried)")
    args = ap.parse_args()
    if args.cmd == "synthesize":
        cmd_synthesize(args)
    else:
        cmd_verify(args)


if __name__ == "__main__":
    main()
