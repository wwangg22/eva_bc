"""Round 4: pin down cuda-graph interaction; test padded goalsets and no-graph mode."""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState
from spike_plan_grasp import table_candidates, scene_cuboids, Q_START

XY = (0.28, 0.0)
MAXG = 24


def goal_of(planner, cands, pad_to=None):
    K = len(cands)
    n = pad_to or K
    pos = torch.zeros(1, 1, 1, n, 3, device="cuda")
    quat = torch.zeros(1, 1, 1, n, 4, device="cuda")
    for i in range(n):
        p_, q_ = cands[min(i, K - 1)]
        pos[0, 0, 0, i] = torch.tensor(p_)
        quat[0, 0, 0, i] = torch.tensor(q_)
    return GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)


def run(tag, use_graph, warmup, pad):
    cfg = MotionPlannerCfg.create(
        robot=ROBOT_YML, scene_model=None, collision_cache={"cuboid": 16},
        max_goalset=MAXG, use_cuda_graph=use_graph,
    )
    p = MotionPlanner(cfg)
    if warmup:
        p.warmup(enable_graph=True, num_warmup_iterations=2)
    p.update_world(scene_cuboids(XY))
    cands = table_candidates(XY)
    q = torch.tensor([Q_START], device="cuda", dtype=torch.float32)
    start = JointState.from_position(q, joint_names=p.joint_names)
    t0 = time.time()
    try:
        res = p.plan_grasp(
            grasp_poses=goal_of(p, cands, pad_to=pad),
            current_state=start,
            grasp_approach_axis="z", grasp_approach_offset=0.10,
            grasp_approach_in_tool_frame=False,
            grasp_lift_axis="z", grasp_lift_offset=0.08, grasp_lift_in_tool_frame=False,
        )
        dt = time.time() - t0
        print(f"{tag}: success={bool(res.success.any().item())} "
              f"appr={bool(res.approach_success.any().item())} "
              f"grasp={bool(res.grasp_success.any().item())} "
              f"lift={bool(res.lift_success.any().item())} t={dt:.2f}s [{res.status}]")
    except Exception as e:
        print(f"{tag}: EXCEPTION {type(e).__name__}: {e}")
    del p
    torch.cuda.empty_cache()


run("A graph+warmup+pad24", True, True, MAXG)
run("B graph+NOwarmup+pad24", True, False, MAXG)
run("C nograph+warmup+K16", False, True, None)
run("D nograph+NOwarmup+K16", False, False, None)
