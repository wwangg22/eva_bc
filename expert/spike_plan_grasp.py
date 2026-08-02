"""SPIKE (PLAN.md ledger item 3, GO/NO-GO): K~20 side-grasp plan_grasp on the real scene.

Measures, over a grid of can positions spanning the upgraded spawn region:
  - full plan_grasp success rate (goalset -> approach -> linear grasp -> lift)
  - per-stage failure anatomy
  - wall time per call
  - VRAM

Scene: table plane, 5-cuboid basket at v0 position, second can as obstacle,
target can as obstacle (finger collisions are disabled by plan_grasp during
goalset/grasp/lift, so the can protects the arm without blocking the fingers).

Frames (validated against Isaac Lab to 0.1 mm, 2026-08-01):
  - cuRobo poses are WXYZ; gripper_end tool frame: fingers/TCP along -X
    (TCP offset -0.075), fingers separate along +/-Y.
  - grasp pose: tool x_hat = -approach_dir, y_hat horizontal, z_hat = x cross y.

Run: conda activate env_isaaclab6 && python spike_plan_grasp.py
"""

import json
import math
import os
import subprocess
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")
OUT_JSON = os.path.join(HERE, "spike_results.json")

# Grasp-table source. Default = the proven 12953-entry table; set REBOT_GRASP_TABLE
# (e.g. to expert/grasp_table_extended.pt from extend_grasp_table.py) to serve the
# gap-filling extension entries with zero code change.
GRASP_TABLE_PT = os.environ.get(
    "REBOT_GRASP_TABLE",
    "/home/william/Desktop/isaacLab/reBot/reBot_RL/data/pick_place_demos/grasp_table.pt",
)

TCP_OFFSET = 0.075          # TCP is -0.075 x in gripper_end frame
BASKET_CENTER = (0.22, -0.12)
B_IN, B_T, B_H, B_F = 0.055, 0.006, 0.04, 0.004
B_OUT = 2 * (B_IN + B_T)
B_WALL_C = B_IN + B_T / 2
CAN_DIMS = (0.024, 0.024, 0.036)
CAN_CENTER_Z = 0.031
OTHER_CAN = (0.29, 0.10)

# the env's true operating start (_START_POSE): arm raised, gripper clear of the table.
# (The earlier "settled" probe pose had the gripper 3 mm off the table — its spheres sat
# inside the table obstacle, which invalidated the start state and every trajopt.)
Q_START = [0.0, -1.35, -0.3, -0.85, 0.0, 0.0]


def scene_cuboids(target_xy):
    from curobo.scene import Cuboid, Scene

    cx, cy = BASKET_CENTER
    cubs = [
        Cuboid(name="table", dims=[1.2, 1.2, 0.05], pose=[0.25, 0, -0.027, 1, 0, 0, 0]),
        Cuboid(name="b_floor", dims=[B_OUT, B_OUT, B_F], pose=[cx, cy, B_F / 2, 1, 0, 0, 0]),
        Cuboid(name="b_px", dims=[B_T, B_OUT, B_H], pose=[cx + B_WALL_C, cy, B_H / 2, 1, 0, 0, 0]),
        Cuboid(name="b_nx", dims=[B_T, B_OUT, B_H], pose=[cx - B_WALL_C, cy, B_H / 2, 1, 0, 0, 0]),
        Cuboid(name="b_py", dims=[B_OUT, B_T, B_H], pose=[cx, cy + B_WALL_C, B_H / 2, 1, 0, 0, 0]),
        Cuboid(name="b_ny", dims=[B_OUT, B_T, B_H], pose=[cx, cy - B_WALL_C, B_H / 2, 1, 0, 0, 0]),
        Cuboid(name="other_can", dims=list(CAN_DIMS),
               pose=[OTHER_CAN[0], OTHER_CAN[1], CAN_CENTER_Z, 1, 0, 0, 0]),
        Cuboid(name="target_can", dims=list(CAN_DIMS),
               pose=[target_xy[0], target_xy[1], CAN_CENTER_Z, 1, 0, 0, 0]),
    ]
    return Scene(cuboid=cubs)


def rot_to_wxyz(R):
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return [q[3], q[0], q[1], q[2]]


def grasp_candidates(can_xy, dazs=(-175.0, -159.0, -145.0), elevs=(8.0, 18.0, 26.0)):
    """Empirical RS-rebot side-grasp family for a vertical cylinder at can_xy.

    Derived from FK over grasp_table.pt (12953 proven grasps, 2026-08-01):
      finger dir fwd = -x_hat exactly; fwd azimuth = pocket azimuth + daz,
      daz in [-178, -138] deg (fingers point BACK toward the robot — the arm
      reaches past the can); fwd elevation in [4, 33] deg (fingers tilt up);
      finger-separation axis y_hat roughly horizontal; TCP at can mid-height.
    """
    cx, cy = can_xy
    paz = math.atan2(cy, cx)
    tcp = np.array([cx, cy, CAN_CENTER_Z])
    poses = []
    for daz in dazs:
        phi = paz + math.radians(daz)
        for elev in elevs:
            e = math.radians(elev)
            fwd = np.array([math.cos(phi) * math.cos(e), math.sin(phi) * math.cos(e), math.sin(e)])
            x_hat = -fwd
            y_hat = np.cross([0, 0, 1.0], fwd)
            y_hat /= np.linalg.norm(y_hat)
            z_hat = np.cross(x_hat, y_hat)
            for flip in (1.0, -1.0):
                R = np.column_stack([x_hat, flip * y_hat, flip * z_hat])
                gpos = tcp + TCP_OFFSET * x_hat  # gripper_end origin sits past the can
                poses.append((gpos.tolist(), rot_to_wxyz(R)))
    return poses


_TABLE = None
# Instrumentation: table row index behind each returned candidate (parallel to the
# returned poses list, post-padding). Rows >= 12953 come from grasp_table_extended.pt.
LAST_CAND_ROWS = []
# Parallel align score (max |cos| of finger-sep/finger-fwd axis vs can axis) for the
# lying-can branch; None entries when align_axis_xy was not given.
LAST_CAND_ALIGN = []


def table_candidates(can_xy, K=16, max_xy_dist=0.035, target_z=None,
                     align_axis_xy=None, exclude=(), exclude_rows=()):
    """align_axis_xy: for LYING cans — keep candidates whose finger-separation axis
    (tool y in world) aligns with the can's cylinder axis (end-cap pinch) or whose
    finger direction aligns with it (axial grip). exclude: candidate list indices
    (from a previous call with identical args) to drop, for retry diversity."""
    """Goalset from the proven grasp table: K nearest entries by pocket xy, FK'd.

    The synthetic ring family is mostly IK-infeasible (2-6/18) because feasible
    (daz, elev) vary by region; the 12953-entry table IS the region-conditioned
    grasp source (24/24 of its poses IK-solve with fingers disabled).
    """
    global _TABLE
    if _TABLE is None:
        import torch as _t
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState as _JS

        d = _t.load(GRASP_TABLE_PT, map_location="cpu", weights_only=False)
        kin = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT_YML))
        q = d["q"].cuda().float()
        st = kin.compute_kinematics(_JS.from_position(q, joint_names=kin.joint_names))
        _TABLE = {
            "pocket": d["pocket"].numpy(),
            "pos": st.tool_poses.position.reshape(-1, 3).cpu().numpy(),
            "quat": st.tool_poses.quaternion.reshape(-1, 4).cpu().numpy(),
        }
    # Match by RADIUS, then ROTATE the whole proven pose about the base z-axis to the
    # target azimuth (a pure joint1 offset — reachability-preserving by construction).
    # A rigid xy TRANSLATION of 2-3 cm empirically exits this wrist's feasibility
    # manifold (0/16 IK even in empty space), so never translate laterally.
    if target_z is None:
        target_z = CAN_CENTER_Z
    pk = _TABLE["pocket"]
    r_n = np.linalg.norm(pk[:, :2], axis=1)
    r_t = float(np.hypot(*can_xy))
    az_t = math.atan2(can_xy[1], can_xy[0])
    score = np.abs(r_n - r_t) + 2.0 * np.abs(pk[:, 2] - target_z)
    order = np.argsort(score)
    sel = order[: K * 8][score[order[: K * 8]] < max_xy_dist]
    if len(sel) == 0:
        # do NOT fall back to unfiltered nearest-K: distant-family candidates produce
        # infeasible goalsets that burn every retry (32-ep A/B: plan_fail 4 -> 13+,
        # success 87.5% -> 72-81%). An honest empty return lets the caller skip.
        return []
    take = K * 3 if align_axis_xy is not None else K
    sel = sel[np.linspace(0, len(sel) - 1, min(take, len(sel))).astype(int)]  # spread
    poses = []
    rows = []
    aligns = []
    for i in sel:
        az_n = math.atan2(pk[i, 1], pk[i, 0])
        th = az_t - az_n
        c, s = math.cos(th), math.sin(th)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        p_rot = Rz @ _TABLE["pos"][i]
        pk_rot = Rz @ pk[i]
        # wxyz quat premultiply by rotation about z
        qw, qx, qy, qz = math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)
        w, x, y, z = _TABLE["quat"][i]
        q_new = [qw * w - qz * z, qw * x - qz * y, qw * y + qz * x, qw * z + qz * w]
        shift = np.array([can_xy[0], can_xy[1], target_z]) - pk_rot  # small radial+z residual
        poses.append(((p_rot + shift).tolist(), q_new))
        rows.append(int(i))
        aligns.append(None)
    if align_axis_xy is not None and poses:
        ax = np.array(align_axis_xy, dtype=float)
        n = np.linalg.norm(ax)
        if n > 1e-6:
            ax /= n
            scored = []
            for (p_, q_), rw in zip(poses, rows):
                w, x, y, z = q_
                yhat = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z)])  # tool-y xy
                fwd = -np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)])  # -tool-x xy
                a = max(abs(float(np.dot(yhat, ax))) / (np.linalg.norm(yhat) + 1e-9),
                        abs(float(np.dot(fwd, ax))) / (np.linalg.norm(fwd) + 1e-9))
                scored.append((a, (p_, q_), rw))
            scored.sort(key=lambda t: -t[0])
            poses = [pq for _, pq, _ in scored[:K]]
            rows = [rw for _, _, rw in scored[:K]]
            aligns = [a for a, _, _ in scored[:K]]
    keep = [j for j in range(len(poses))
            if j not in set(exclude) and rows[j] not in set(exclude_rows)]
    poses = [poses[j] for j in keep]
    rows = [rows[j] for j in keep]
    aligns = [aligns[j] for j in keep]
    while 0 < len(poses) < K:  # pad to fixed K: cuda-graph capture requires a stable goalset shape
        poses.append(poses[len(poses) % max(len(poses), 1)])
        rows.append(rows[len(rows) % max(len(rows), 1)])
        aligns.append(aligns[len(aligns) % max(len(aligns), 1)])
    global LAST_CAND_ROWS, LAST_CAND_ALIGN
    LAST_CAND_ROWS = list(rows[:K])
    LAST_CAND_ALIGN = list(aligns[:K])
    return poses[:K]


def main():
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState

    t0 = time.time()
    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YML,
        scene_model=None,
        collision_cache={"cuboid": 16},
        max_goalset=24,
    )
    planner = MotionPlanner(cfg)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    print(f"planner init+warmup: {time.time()-t0:.1f}s; tool_frames={planner.tool_frames}")
    print("joint order:", planner.joint_names)

    q = torch.tensor([Q_START], device="cuda", dtype=torch.float32)
    start = JointState.from_position(q, joint_names=planner.joint_names)

    results = []
    radii = [0.20, 0.24, 0.28, 0.32]
    azimuths = [-40, -20, 0, 20, 40]
    torch.cuda.reset_peak_memory_stats()

    for r in radii:
        for az in azimuths:
            xy = (r * math.cos(math.radians(az)), r * math.sin(math.radians(az)))
            d_basket = math.hypot(xy[0] - BASKET_CENTER[0], xy[1] - BASKET_CENTER[1])
            d_other = math.hypot(xy[0] - OTHER_CAN[0], xy[1] - OTHER_CAN[1])
            tag = f"r={r:.2f} az={az:+d} xy=({xy[0]:.3f},{xy[1]:.3f})"
            if d_basket < 0.115 or d_other < 0.05:
                results.append({"r": r, "az": az, "skipped": True,
                                "why": f"d_basket={d_basket:.3f} d_other={d_other:.3f}"})
                print(f"SKIP  {tag} (too close: basket {d_basket:.3f} / can {d_other:.3f})")
                continue

            planner.update_world(scene_cuboids(xy))
            cands = table_candidates(xy)
            if not cands:
                results.append({"r": r, "az": az, "skipped": True, "why": "no table entries near"})
                print(f"SKIP  {tag} (no grasp-table entries within 3.5cm)")
                continue
            K = len(cands)
            pos = torch.zeros(1, 1, 1, K, 3, device="cuda")
            quat = torch.zeros(1, 1, 1, K, 4, device="cuda")
            for i, (p_, q_) in enumerate(cands):
                pos[0, 0, 0, i] = torch.tensor(p_)
                quat[0, 0, 0, i] = torch.tensor(q_)
            goal = GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)

            t1 = time.time()
            # approach: vertical descent from 10 cm above (the empirical family's
            # only collision-free linear approach — matches the old expert's DOWN
            # phase); lift: straight up in world frame.
            res = planner.plan_grasp(
                grasp_poses=goal,
                current_state=start,
                grasp_approach_axis="z",
                grasp_approach_offset=0.10,
                grasp_approach_in_tool_frame=False,
                grasp_lift_axis="z",
                grasp_lift_offset=0.08,
                grasp_lift_in_tool_frame=False,
            )
            dt = time.time() - t1
            row = {
                "r": r, "az": az, "xy": [round(v, 4) for v in xy], "skipped": False,
                "K": K, "time_s": round(dt, 3),
                "success": bool(res.success.any().item()),
                "approach": bool(res.approach_success.any().item()),
                "grasp": bool(res.grasp_success.any().item()),
                "lift": bool(res.lift_success.any().item()),
                "goalset_index": int(res.goalset_index.view(-1)[0].item()) if res.goalset_index is not None else None,
                "status": res.status,
            }
            results.append(row)
            print(f"{'PASS' if row['success'] else 'FAIL'}  {tag} t={dt:.2f}s "
                  f"appr={row['approach']} grasp={row['grasp']} lift={row['lift']} "
                  f"gi={row['goalset_index']} [{res.status}]")

    ran = [x for x in results if not x["skipped"]]
    ok = [x for x in ran if x["success"]]
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    summary = {
        "feasibility_rate": round(len(ok) / max(len(ran), 1), 3),
        "n_ran": len(ran), "n_pass": len(ok),
        "n_skipped": len(results) - len(ran),
        "mean_time_s": round(float(np.mean([x["time_s"] for x in ran])), 3) if ran else None,
        "torch_peak_alloc_MiB": round(peak_alloc, 1),
        "nvidia_smi_used": smi,
    }
    print("\nSUMMARY:", json.dumps(summary, indent=2))
    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary, "scenarios": results}, f, indent=2)
    print("saved ->", OUT_JSON)


if __name__ == "__main__":
    main()
