"""Round 5: is the start state self-collision-flagged, killing all trajopt?"""
import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")
PATCHED_YML = os.path.join(HERE, "rs_rebot_scpatch.yml")

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from spike_plan_grasp import table_candidates, scene_cuboids, Q_START

XY = (0.28, 0.0)

# variant yml with the 3 false-positive pairs ignored
d = yaml.safe_load(open(ROBOT_YML))
kin = d.get("robot_cfg", d)["kinematics"]
sci = kin["self_collision_ignore"]
for a, b in [("base_link", "gripper_end"), ("link1", "link5"), ("link2", "link5")]:
    sci.setdefault(a, [])
    if b not in sci[a]:
        sci[a].append(b)
with open(PATCHED_YML, "w") as f:
    yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False)


def trial(tag, yml, start_q):
    cfg = MotionPlannerCfg.create(robot=yml, scene_model=None,
                                  collision_cache={"cuboid": 16}, max_goalset=24)
    p = MotionPlanner(cfg)
    p.update_world(scene_cuboids(XY))
    cands = table_candidates(XY)
    K = len(cands)
    pos = torch.zeros(1, 1, 1, K, 3, device="cuda")
    quat = torch.zeros(1, 1, 1, K, 4, device="cuda")
    for i, (p_, q_) in enumerate(cands):
        pos[0, 0, 0, i] = torch.tensor(p_)
        quat[0, 0, 0, i] = torch.tensor(q_)
    goal = GoalToolPose(tool_frames=p.tool_frames, position=pos, quaternion=quat)
    start = JointState.from_position(
        torch.tensor([start_q], device="cuda", dtype=torch.float32), joint_names=p.joint_names)
    res = p.plan_grasp(grasp_poses=goal, current_state=start,
                       grasp_approach_axis="z", grasp_approach_offset=0.10,
                       grasp_approach_in_tool_frame=False,
                       grasp_lift_axis="z", grasp_lift_offset=0.08,
                       grasp_lift_in_tool_frame=False)
    print(f"{tag}: success={bool(res.success.any().item())} appr={bool(res.approach_success.any().item())} "
          f"grasp={bool(res.grasp_success.any().item())} lift={bool(res.lift_success.any().item())} [{res.status}]")
    del p
    torch.cuda.empty_cache()


# proven config as alternative start
dtab = torch.load("/home/william/Desktop/isaacLab/reBot/reBot_RL/data/pick_place_demos/grasp_table.pt",
                  map_location="cpu", weights_only=False)
q_proven = dtab["q"][5].tolist()

trial("A original yml, Q_START     ", ROBOT_YML, Q_START)
trial("B original yml, proven start", ROBOT_YML, q_proven)
trial("C sc-ignore yml, Q_START    ", PATCHED_YML, Q_START)
trial("D sc-ignore yml, proven start", PATCHED_YML, q_proven)
