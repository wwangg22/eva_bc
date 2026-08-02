"""Root-cause replay for cuRoboV2 pose-place refusals (Gate 1 gap (a)).

Loads place_fail_cases.pt (dumped by run_expert_v1.py when both plan_place rungs
were refused), rebuilds the planner world for each case exactly as the runner
does, and runs a hypothesis ladder per case:
  H0a/H0b baseline: attached, full world, heights (0.10,0.13) / (0.13,0.16)
                    - should reproduce the refusal
  H1  no-attach:    detached, held can absent - attached-sphere fit the blocker?
  H2  no-walls:     attached, basket walls removed - wall activation the blocker?
  H3  higher-goals: attached, full world, heights (0.18,0.22) - goals vs path?
  H4  start-valid:  attached, full world, plan_cspace to Q_HOME - if even this
                    fails, the START state (lift pose + attached spheres) is bad.

Pure cuRobo + torch + numpy (no isaaclab/gymnasium). GPU required.
Run (env_isaaclab6, ONLY when the GPU is free):
  conda activate env_isaaclab6 && python replay_place_fail.py [--cases PATH]
Writes tally to place_fail_analysis_raw.md in this dir.
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")
sys.path.insert(0, HERE)
from spike_plan_grasp import (  # noqa: E402,F401
    CAN_CENTER_Z, B_IN, B_T, B_H, B_F, B_OUT, B_WALL_C, CAN_DIMS,
)

Q_HOME = [0.0, -1.35, -0.3, -0.85, 0.0, 0.0]
TCP_OFF = 0.075
CARRY_PT = "/home/william/Desktop/isaacLab/reBot/reBot_RL/data/pick_place_demos/carry_waypoints.pt"
HYPS = ["H0a", "H0b", "H1", "H2", "H3", "H4"]


def build_scene(basket_xy, can_states, walls=True):
    """Verbatim from run_expert_v1.build_scene, plus walls=False for H2."""
    from curobo.scene import Cuboid, Scene

    bx, by = float(basket_xy[0]), float(basket_xy[1])
    cubs = [
        Cuboid(name="table", dims=[1.2, 1.2, 0.05], pose=[0.25, 0, -0.027, 1, 0, 0, 0]),
        Cuboid(name="b_floor", dims=[B_OUT, B_OUT, B_F], pose=[bx, by, B_F / 2, 1, 0, 0, 0]),
    ]
    if walls:
        cubs += [
            Cuboid(name="b_px", dims=[B_T, B_OUT, B_H], pose=[bx + B_WALL_C, by, B_H / 2, 1, 0, 0, 0]),
            Cuboid(name="b_nx", dims=[B_T, B_OUT, B_H], pose=[bx - B_WALL_C, by, B_H / 2, 1, 0, 0, 0]),
            Cuboid(name="b_py", dims=[B_OUT, B_T, B_H], pose=[bx, by + B_WALL_C, B_H / 2, 1, 0, 0, 0]),
            Cuboid(name="b_ny", dims=[B_OUT, B_T, B_H], pose=[bx, by - B_WALL_C, B_H / 2, 1, 0, 0, 0]),
        ]
    for name, (xyz, in_hand) in can_states.items():
        if in_hand:
            continue  # attached: lives on the robot, not in the world
        cubs.append(Cuboid(name=name, dims=list(CAN_DIMS),
                           pose=[float(xyz[0]), float(xyz[1]), max(float(xyz[2]), 0.02), 1, 0, 0, 0]))
    return Scene(cuboid=cubs)


class Expert:
    """Planner wrapper — verbatim copies of the runner's planning methods."""

    def __init__(self):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        cfg = MotionPlannerCfg.create(
            robot=ROBOT_YML, scene_model=None, collision_cache={"cuboid": 16}, max_goalset=16,
        )
        self.planner = MotionPlanner(cfg)
        self.K = 16

    def js(self, q6):
        from curobo.types import JointState

        t = torch.as_tensor(q6, device="cuda", dtype=torch.float32).view(1, 6)
        return JointState.from_position(t, joint_names=self.planner.joint_names)

    def goalset(self, poses):
        from curobo.types import GoalToolPose

        K = len(poses)
        pos = torch.zeros(1, 1, 1, K, 3, device="cuda")
        quat = torch.zeros(1, 1, 1, K, 4, device="cuda")
        for i, (p_, q_) in enumerate(poses):
            pos[0, 0, 0, i] = torch.tensor(p_)
            quat[0, 0, 0, i] = torch.tensor(q_)
        return GoalToolPose(tool_frames=self.planner.tool_frames, position=pos, quaternion=quat)

    def _carry_orients(self):
        """Tool orientations of the proven carry configs (FK'd once, wxyz)."""
        if not hasattr(self, "_carry_q"):
            from curobo.types import JointState

            d = torch.load(CARRY_PT, map_location="cpu", weights_only=False)
            st = self.planner.kinematics.compute_kinematics(
                JointState.from_position(
                    d["q_over"].cuda().float(), joint_names=self.planner.joint_names))
            self._carry_q = st.tool_poses.quaternion.reshape(-1, 4).cpu().numpy()
            self._carry_az = [math.atan2(float(p[1]), float(p[0])) for p in d["pocket_over"]]
        return self._carry_q, self._carry_az

    @staticmethod
    def _rotz_quat(qwxyz, th):
        c, s = math.cos(th / 2), math.sin(th / 2)
        w, x, y, z = qwxyz
        return [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]

    @staticmethod
    def _pose_from(qn, basket_xy, tz):
        xx = 1 - 2 * (qn[2] ** 2 + qn[3] ** 2)
        xy = 2 * (qn[1] * qn[2] + qn[0] * qn[3])
        xz = 2 * (qn[1] * qn[3] - qn[0] * qn[2])
        tcp = np.array([basket_xy[0], basket_xy[1], tz])
        return (tcp + TCP_OFF * np.array([xx, xy, xz])).tolist(), list(qn)

    def place_goalset(self, q6, basket_xy, heights=(0.10, 0.13)):
        """Candidate orientations: current grasp orientation + the two proven carry-band
        orientations, all azimuth-rotated over the basket."""
        st = self.planner.kinematics.compute_kinematics(self.js(q6))
        gpos = st.tool_poses.position.reshape(3).cpu().numpy()
        gq = st.tool_poses.quaternion.reshape(4).cpu().numpy()
        az_b = math.atan2(basket_xy[1], basket_xy[0])
        poses = []
        for daz in (-0.26, 0.0, 0.26):
            qn = self._rotz_quat(gq, az_b - math.atan2(gpos[1], gpos[0]) + daz)
            for tz in heights:
                poses.append(self._pose_from(qn, basket_xy, tz))
        cq, caz = self._carry_orients()
        for i in range(len(cq)):
            for daz in (0.0, -0.26, 0.26):
                qn = self._rotz_quat(cq[i], az_b - caz[i] + daz)
                poses.append(self._pose_from(qn, basket_xy, heights[0]))
        n = len(poses)
        while len(poses) < self.K:
            poses.append(poses[len(poses) % n])
        return poses[: self.K]

    def plan_place(self, q6, basket_xy, heights=(0.10, 0.13)):
        poses = self.place_goalset(q6, basket_xy, heights)
        return self.planner.plan_pose(self.goalset(poses), self.js(q6))

    def plan_to_config(self, q6, q_goal):
        from curobo.types import JointState

        goal = JointState.from_position(
            torch.as_tensor([list(q_goal)], device="cuda", dtype=torch.float32),
            joint_names=self.planner.joint_names)
        return self.planner.plan_cspace(goal, self.js(q6))


def check(res):
    ok = res is not None and bool(res.success.any().item())
    return ok, ("none" if res is None else str(getattr(res, "status", "?")))


def run_case(ex, case):
    q6, tgt, bxy, held = case["q6"], case["tgt"], case["basket_xy"], case["held"]
    tgt = (float(tgt[0]), float(tgt[1]))
    full = {n: (xyz, False) for n, xyz in case["cans"].items()}       # held IN scene (for attach)
    carried = {n: (xyz, n == held) for n, xyz in case["cans"].items()}  # held on the robot
    out = {}

    def attach():
        # attach_from_scene disables the held obstacle in the CURRENT world; the same
        # world must stay loaded until detach() (which re-enables it) — never
        # update_world while attached.
        try:
            from curobo.types import Pose
            # identity world offset: fit returns WORLD-frame centers (upstream
            # docstring bug) — this makes update() convert them to link-local
            ex.planner.attachment_manager.attach_from_scene(
                ex.js(q6), [held],
                world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0]))
            return True, "ok"
        except Exception as e:  # noqa: BLE001 — record and continue the ladder
            return False, repr(e)

    # phase A: attached, full world
    ex.planner.update_world(build_scene(bxy, full))
    out["attach"] = attach()
    out["H0a"] = check(ex.plan_place(q6, tgt))                          # rung 1 baseline
    out["H0b"] = check(ex.plan_place(q6, tgt, heights=(0.13, 0.16)))    # rung 2 baseline
    out["H3"] = check(ex.plan_place(q6, tgt, heights=(0.18, 0.22)))     # goals above walls
    out["H4"] = check(ex.plan_to_config(q6, Q_HOME))                    # start-state validity
    if out["attach"][0]:
        ex.planner.attachment_manager.detach()
    # phase B: attached, walls removed
    ex.planner.update_world(build_scene(bxy, full, walls=False))
    a2 = attach()
    out["H2"] = check(ex.plan_place(q6, tgt))                           # no basket walls
    if a2[0]:
        ex.planner.attachment_manager.detach()
    # phase C: no attach, held can absent (it's in the gripper, not the world)
    ex.planner.update_world(build_scene(bxy, carried))
    out["H1"] = check(ex.plan_place(q6, tgt))                           # no attached spheres
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join(HERE, "place_fail_cases.pt"))
    args = ap.parse_args()

    cases = torch.load(args.cases, map_location="cpu", weights_only=False)
    print(f"{len(cases)} refused place cases from {args.cases}")
    ex = Expert()

    tally = {h: 0 for h in HYPS}
    rows = []
    print(f"\n{'idx':>3} {'held':>8} {'r_bask':>6}  " + "  ".join(f"{h:>4}" for h in HYPS))
    for i, case in enumerate(cases):
        out = run_case(ex, case)
        r_b = math.hypot(float(case["basket_xy"][0]), float(case["basket_xy"][1]))
        for h in HYPS:
            tally[h] += out[h][0]
        marks = "  ".join(f"{'PASS' if out[h][0] else 'FAIL':>4}" for h in HYPS)
        print(f"{i:>3} {case['held']:>8} {r_b:6.3f}  {marks}")
        fails = [f"{h}={out[h][1]}" for h in HYPS if not out[h][0]]
        if not out["attach"][0]:
            fails.insert(0, f"attach={out['attach'][1]}")
        if fails:
            print(f"      status: {'; '.join(fails)}")
        rows.append((i, case["held"], r_b, out))

    n = len(cases)
    lines = ["# place_fail replay — raw tally", "",
             f"cases: {n} (from `{os.path.basename(args.cases)}`)", "",
             "| hypothesis | pass | fail |", "|---|---|---|"]
    print(f"\nTALLY over {n} cases:")
    for h in HYPS:
        print(f"  {h}: {tally[h]}/{n} pass")
        lines.append(f"| {h} | {tally[h]} | {n - tally[h]} |")
    lines += ["", "H0a/H0b = baseline rung1/rung2 (expect FAIL = refusal reproduced); "
              "H1 = no-attach; H2 = no basket walls; H3 = heights (0.18,0.22); "
              "H4 = plan_cspace to Q_HOME (start validity)."]
    md_path = os.path.join(HERE, "place_fail_analysis_raw.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved -> {md_path}")


if __name__ == "__main__":
    main()
