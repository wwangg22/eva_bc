"""Isolate why goalset IK returns zero solutions.

Uses proven-reachable poses: FK of grasp_table.pt configs. If IK can't solve
those, the problem is the solver/model (self-collision spheres, world), not
the grasp family.
"""
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


def build(scene, tag):
    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YML, scene_model=scene, collision_cache={"cuboid": 16}, max_goalset=8,
    )
    p = MotionPlanner(cfg)
    print(f"--- planner[{tag}] ready")
    return p


def ik_batch(planner, poses_wxyz, tag):
    """poses: list of (pos, quat_wxyz). Solve each as single-goal IK."""
    ok = 0
    fails = []
    for i, (pp, qq) in enumerate(poses_wxyz):
        pos = torch.zeros(1, 1, 1, 1, 3, device="cuda")
        quat = torch.zeros(1, 1, 1, 1, 4, device="cuda")
        pos[0, 0, 0, 0] = torch.tensor(pp)
        quat[0, 0, 0, 0] = torch.tensor(qq)
        goal = GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)
        res = planner.ik_solver.solve_pose(goal)
        s = bool(res.success.any().item())
        ok += s
        if not s and len(fails) < 3:
            fails.append(i)
    print(f"{tag}: IK success {ok}/{len(poses_wxyz)} (first fails: {fails})")
    return ok


def main():
    d = torch.load(
        "/home/william/Desktop/isaacLab/reBot/reBot_RL/data/pick_place_demos/grasp_table.pt",
        map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    idx = torch.randperm(d["q"].shape[0])[:24]
    q_proven = d["q"][idx].cuda().float()

    # FK those configs with a bare kinematics model
    from curobo.kinematics import Kinematics, KinematicsCfg
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(ROBOT_YML))
    jn = kin.joint_names
    st = kin.compute_kinematics(JointState.from_position(q_proven, joint_names=jn))
    pos = st.tool_poses.position.reshape(-1, 3).cpu().numpy()
    quat = st.tool_poses.quaternion.reshape(-1, 4).cpu().numpy()
    poses = [(pos[i].tolist(), quat[i].tolist()) for i in range(len(pos))]

    # sphere sanity at proven configs: lowest sphere z per link group
    sph = st.get_link_spheres().cpu().numpy()
    print("sphere tensor shape:", sph.shape)

    # 1) empty world
    p_empty = build(None, "empty world")
    ik_batch(p_empty, poses, "proven poses, empty world")

    # 2) table-only world (config-time scene)
    p_table = build(
        {"cuboid": {"table": {"dims": [1.2, 1.2, 0.05], "pose": [0.25, 0, -0.027, 1, 0, 0, 0]}}},
        "table world")
    ik_batch(p_table, poses, "proven poses, table world")

    # 3) table world loaded via update_world(Scene(...)) — the spike's code path
    from curobo.scene import Cuboid, Scene
    p_upd = build(None, "empty -> update_world")
    p_upd.update_world(Scene(cuboid=[
        Cuboid(name="table", dims=[1.2, 1.2, 0.05], pose=[0.25, 0, -0.027, 1, 0, 0, 0])]))
    ik_batch(p_upd, poses, "proven poses, update_world table")

    # 4) lower table by 2 cm — is it marginal or categorical?
    p_low = build(
        {"cuboid": {"table": {"dims": [1.2, 1.2, 0.05], "pose": [0.25, 0, -0.047, 1, 0, 0, 0]}}},
        "low table world")
    ik_batch(p_low, poses, "proven poses, table 2cm lower")


if __name__ == "__main__":
    main()
