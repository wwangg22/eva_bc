"""Round 3: isolate the goalset-path failure (poses individually IK-solve)."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from spike_plan_grasp import table_candidates, scene_cuboids, Q_START

GRIPPER_LINKS = ["gripper_end", "gripper_left", "gripper_right", "attached_object"]
XY = (0.28, 0.0)  # a grid point with plenty of table entries


def goal_of(cands):
    K = len(cands)
    pos = torch.zeros(1, 1, 1, K, 3, device="cuda")
    quat = torch.zeros(1, 1, 1, K, 4, device="cuda")
    for i, (p_, q_) in enumerate(cands):
        pos[0, 0, 0, i] = torch.tensor(p_)
        quat[0, 0, 0, i] = torch.tensor(q_)
    return GoalToolPose(tool_frames=PLANNER.tool_frames, position=pos, quaternion=quat)


def report(tag, res):
    if res is None:
        print(f"{tag}: None")
        return
    s = res.success
    print(f"{tag}: success={bool(s.any().item())} (tensor {s.view(-1).tolist()})")


cfg = MotionPlannerCfg.create(
    robot=ROBOT_YML,
    scene_model=None,
    collision_cache={"cuboid": 16},
    max_goalset=24,
)
PLANNER = MotionPlanner(cfg)
PLANNER.update_world(scene_cuboids(XY))

cands = table_candidates(XY)
print(f"candidates: {len(cands)}")
q = torch.tensor([Q_START], device="cuda", dtype=torch.float32)
start = JointState.from_position(q, joint_names=PLANNER.joint_names)

PLANNER.disable_link_collision(GRIPPER_LINKS)

# (a) goalset, bare solve_pose
res = PLANNER.ik_solver.solve_pose(goal_of(cands))
report("a) goalset-16 bare", res)

# (b) single goal with plan_pose-style args
res = PLANNER.ik_solver.solve_pose(
    goal_of(cands[:1]),
    return_seeds=PLANNER.trajopt_solver.config.num_seeds,
    current_state=start,
)
report("b) single + seeds + current_state", res)

# (c) goalset with plan_pose-style args
res = PLANNER.ik_solver.solve_pose(
    goal_of(cands),
    return_seeds=PLANNER.trajopt_solver.config.num_seeds,
    current_state=start,
)
report("c) goalset + seeds + current_state", res)

PLANNER.enable_link_collision(GRIPPER_LINKS)

# (d) full plan_pose (IK + trajopt) on the goalset, fingers auto-disabled like plan_grasp
PLANNER.disable_link_collision(GRIPPER_LINKS)
res = PLANNER.plan_pose(goal_of(cands), start)
PLANNER.enable_link_collision(GRIPPER_LINKS)
report("d) plan_pose goalset (trajopt too)", res)
if res is not None:
    print("   goalset_index:", None if res.goalset_index is None else res.goalset_index.view(-1).tolist())
