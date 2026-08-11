#!/usr/bin/env python
"""Which part of the workstation scene kills plan_grasp? Pure cuRobo, no Isaac.

run_expert_ws's first smoke run failed 6/6 with "Goalset planning returned None" at
spawns the upstream spike measured 10/10 on — so something in MY scene or candidate
construction differs from the spike protocol. This probe rebuilds the call at the two
failing spawns and sweeps one difference at a time:

    scene:  none -> table(flush) -> table(recessed 2mm) -> +cube -> +box+clutter
    align:  cube face axis vs None
    tz:     the cube formula (0.044) vs the spike's can default

Run: conda activate env_isaaclab6 && python -u re3sim/expert/probe_ws_plan.py
"""

import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_EXPERT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "expert"))
sys.path.insert(0, _EXPERT_DIR)
sys.path.insert(0, HERE)

from spike_plan_grasp import table_candidates  # noqa: E402

# run_expert_ws launches Isaac at import, so its scene builder cannot be imported here;
# the constants below mirror re3sim mdp/common.py (BOX_*) and run_expert_ws.build_scene.
Q_HOME = [0.0, -1.35, -0.3, -0.85, 0.0, 0.0]
BOX_OX, BOX_OY, BOX_H, BOX_W, BOX_FT = 0.218, 0.150, 0.093, 0.003, 0.006


def build_scene(box_xy, box_yaw, objs):
    from curobo.scene import Cuboid, Scene

    bx, by = float(box_xy[0]), float(box_xy[1])
    yq = [math.cos(box_yaw / 2), 0.0, 0.0, math.sin(box_yaw / 2)]
    c, s = math.cos(box_yaw), math.sin(box_yaw)
    at = lambda dx, dy, z: [bx + c * dx - s * dy, by + s * dx + c * dy, z] + yq  # noqa: E731
    dims = {"cube": [0.056] * 3, "rolloftape": [0.091, 0.091, 0.024],
            "tapemeasure": [0.0715, 0.064, 0.036]}
    cubs = [
        Cuboid(name="table", dims=[1.60, 1.00, 0.05], pose=[0.4, 0.0, -0.025, 1, 0, 0, 0]),
        Cuboid(name="b_floor", dims=[BOX_OX, BOX_OY, BOX_FT], pose=at(0, 0, BOX_FT / 2)),
        Cuboid(name="b_px", dims=[BOX_W, BOX_OY, BOX_H], pose=at(+(BOX_OX - BOX_W) / 2, 0, BOX_H / 2)),
        Cuboid(name="b_nx", dims=[BOX_W, BOX_OY, BOX_H], pose=at(-(BOX_OX - BOX_W) / 2, 0, BOX_H / 2)),
        Cuboid(name="b_py", dims=[BOX_OX, BOX_W, BOX_H], pose=at(0, +(BOX_OY - BOX_W) / 2, BOX_H / 2)),
        Cuboid(name="b_ny", dims=[BOX_OX, BOX_W, BOX_H], pose=at(0, -(BOX_OY - BOX_W) / 2, BOX_H / 2)),
    ]
    for name, (xyz, yaw, in_hand) in objs.items():
        if in_hand:
            continue
        cubs.append(Cuboid(name=name, dims=dims[name],
                           pose=[float(xyz[0]), float(xyz[1]), max(float(xyz[2]), 0.012),
                                 math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
    return Scene(cuboid=cubs)

# the two spawns the smoke run failed on (seed 301), plus a mid-band control
CASES = [
    ((0.178, 0.123), 0.028, 0.9),    # (xy), cube z, cube yaw
    ((0.268, 0.012), 0.028, 2.1),
    ((0.240, -0.060), 0.028, 0.3),
]
BOX = ((0.30, 0.10), 1.57)
CLUTTER = {"rolloftape": ((0.22, -0.18), 0.012, 0.0),
           "tapemeasure": ((0.16, 0.20), 0.018, 0.5)}


def scene_variant(kind, cube_xy, cube_z, cube_yaw):
    from curobo.scene import Cuboid, Scene

    table = lambda z: Cuboid(name="table", dims=[1.60, 1.00, 0.05],  # noqa: E731
                             pose=[0.4, 0.0, z, 1, 0, 0, 0])
    cube = Cuboid(name="cube", dims=[0.056] * 3,
                  pose=[cube_xy[0], cube_xy[1], cube_z,
                        math.cos(cube_yaw / 2), 0, 0, math.sin(cube_yaw / 2)])
    if kind == "none":
        return Scene(cuboid=[Cuboid(name="dummy", dims=[0.01] * 3,
                                    pose=[2, 2, 2, 1, 0, 0, 0])])
    if kind == "table_flush":
        return Scene(cuboid=[table(-0.025)])
    if kind == "table_recessed":
        return Scene(cuboid=[table(-0.027)])
    if kind == "rec_tbl+cube":
        return Scene(cuboid=[table(-0.027), cube])
    # "full_ws": exactly what run_expert_ws.build_scene produces
    objs = {"cube": (np.array([*cube_xy, cube_z]), cube_yaw, False)}
    for n, (xy, z, yw) in CLUTTER.items():
        objs[n] = (np.array([*xy, z]), yw, False)
    return build_scene(np.array(BOX[0]), BOX[1], objs)


def main():
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState

    cfg = MotionPlannerCfg.create(
        robot=os.path.join(_EXPERT_DIR, "rs_rebot.yml"),
        scene_model=None, collision_cache={"cuboid": 16}, max_goalset=16)
    planner = MotionPlanner(cfg)

    def js(q6):
        return JointState.from_position(
            torch.as_tensor(q6, device="cuda", dtype=torch.float32).view(1, 6),
            joint_names=planner.joint_names)

    def goalset(poses):
        K = len(poses)
        pos = torch.zeros(1, 1, 1, K, 3, device="cuda")
        quat = torch.zeros(1, 1, 1, K, 4, device="cuda")
        for i, (p_, q_) in enumerate(poses):
            pos[0, 0, 0, i] = torch.tensor(p_)
            quat[0, 0, 0, i] = torch.tensor(q_)
        return GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)

    for (xy, cz, cyaw) in CASES:
        tz = float(np.clip(cz + 0.016, 0.012, 0.045))
        for align in (None, (math.cos(cyaw), math.sin(cyaw))):
            cands = table_candidates(tuple(xy), K=16, target_z=tz, align_axis_xy=align)
            if not cands:
                print(f"xy={xy} align={'face' if align else 'none'}: NO-CANDIDATES")
                continue
            for kind in ("none", "table_flush", "table_recessed", "rec_tbl+cube", "full_ws"):
                planner.update_world(scene_variant(kind, xy, cz, cyaw))
                res = planner.plan_grasp(
                    grasp_poses=goalset(cands), current_state=js(Q_HOME),
                    grasp_approach_axis="z", grasp_approach_offset=0.10,
                    grasp_approach_in_tool_frame=False,
                    grasp_lift_axis="z", grasp_lift_offset=0.08,
                    grasp_lift_in_tool_frame=False)
                ok = bool(res.success.any().item()) if res is not None else False
                status = "" if ok else f"   status={getattr(res, 'status', None)}"
                print(f"xy={xy} align={'face' if align else 'none'} scene={kind:>14}: "
                      f"{'OK' if ok else 'FAIL'}{status}")


if __name__ == "__main__":
    main()
