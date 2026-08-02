"""Round 2: proven vs synthetic candidates, fingers-disabled IK, table at surface."""
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from spike_plan_grasp import grasp_candidates, CAN_CENTER_Z

GRIPPER_LINKS = ["gripper_end", "gripper_left", "gripper_right", "attached_object"]


def ik_many(planner, poses, tag, disable=True):
    if disable:
        planner.disable_link_collision(GRIPPER_LINKS)
    ok = 0
    for pp, qq in poses:
        pos = torch.zeros(1, 1, 1, 1, 3, device="cuda")
        quat = torch.zeros(1, 1, 1, 1, 4, device="cuda")
        pos[0, 0, 0, 0] = torch.tensor(pp)
        quat[0, 0, 0, 0] = torch.tensor(qq)
        goal = GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)
        res = planner.ik_solver.solve_pose(goal)
        ok += bool(res.success.any().item())
    if disable:
        planner.enable_link_collision(GRIPPER_LINKS)
    print(f"{tag}: {ok}/{len(poses)}")
    return ok


def main():
    act = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
    d = torch.load(
        "/home/william/Desktop/isaacLab/reBot/reBot_RL/data/pick_place_demos/grasp_table.pt",
        map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    idx = torch.randperm(d["q"].shape[0])[:24]
    q_proven = d["q"][idx].cuda().float()
    pockets = d["pocket"][idx].numpy()

    from curobo.kinematics import Kinematics, KinematicsCfg
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT_YML))
    st = kin.compute_kinematics(JointState.from_position(q_proven, joint_names=kin.joint_names))
    pos = st.tool_poses.position.reshape(-1, 3).cpu().numpy()
    quat = st.tool_poses.quaternion.reshape(-1, 4).cpu().numpy()
    proven = [(pos[i].tolist(), quat[i].tolist()) for i in range(len(pos))]

    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YML,
        scene_model={"cuboid": {"table": {"dims": [1.2, 1.2, 0.05],
                                          "pose": [0.25, 0, -0.027, 1, 0, 0, 0]}}},
        collision_cache={"cuboid": 16},
        max_goalset=8,
        optimizer_collision_activation_distance=act,
    )
    planner = MotionPlanner(cfg)
    print(f"activation_distance={act}")
    ik_many(planner, proven, "proven poses, table, fingers ENABLED", disable=False)
    ik_many(planner, proven, "proven poses, table, fingers disabled")

    for r, az in [(0.24, 0), (0.28, 40)]:
        xy = (r * math.cos(math.radians(az)), r * math.sin(math.radians(az)))
        cands = grasp_candidates(xy)
        ik_many(planner, cands, f"synthetic r={r} az={az} K={len(cands)}, fingers disabled")

    # mean pocket height sanity
    print("pocket z mean of sample:", pockets[:, 2].mean().round(4), "CAN_CENTER_Z:", CAN_CENTER_Z)


if __name__ == "__main__":
    main()
