"""cuRoboV2 planning expert driving Rebot-PickPlace-Play-v1 (Stage 1 bring-up).

Per episode: for each can (nearest first): plan_grasp (goalset from proven-table
candidates, azimuth-rotation construction) -> execute approach/descent at 50 Hz ->
close + clean-grasp check -> lift + hold check (regrasp once on failure) ->
attach -> plan transport to basket goalset -> release -> verify placed -> home.
Memoryless at task level: every plan reads live privileged state.

Run (env_isaaclab6):
  python run_expert_v1.py --episodes 8 --video
Outputs: expert_v1_results.json + expert_v1_env0.mp4 (for Big Will) in this dir.

Labeling (for Stage 2 demo gen / BC loss mask): each episode's metrics carry
`segments` ([{t, phase, seg}] step-indexed phase transitions) and `outcomes`
(seg id -> {outcome, ...}). Mask rule: train_mask=0 over segments whose seg
outcome is "missed" (failed grasp attempt: approach/descend/close/lift up to
failure DETECTION) or "lost" (transport that dropped the can). Post-detection
segments (reopen, re-approach, refetch) are recovery — trainable. Transport
outcomes record `via` (pose / pose-high / carry / carry-direct / +hop) so
unplanned carry-direct interps can be excluded or ablated from the nominal pool.

Perturb mode (--perturb, Stage 3.1): ONE scripted event per episode, sampled from
{nudge live can @approach, nudge other can @lift, forced slip @transport, nudge other
can @transport}; a fired event is step-logged in `perturb_steps` and flips
`episode_kind` to "recovery_scripted" (unfired/off -> "nominal", empty list).
Stage-4 censoring masks action_is_pad from the event step within straddling chunks;
the missed/lost train_mask above stays unconditional.

Diversify mode (--diversify, PLAN §2.2): preserve multi-modality for Stage-2 demo gen —
fetch order reversed with p=0.5 (executed order always recorded in m["order"]), and
FIRST grasp attempts drop each un-tried candidate index with p=0.3 (never below 8
survivors) so cuRobo's cost-based winner varies across episodes (approx. top-p
sampling of the feasible set); retries keep the full set minus tried.
"""

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video-ep", type=int, default=0, help="episode index to record with --video")
parser.add_argument("--record-h5", type=str, default=None)
parser.add_argument("--perturb", action="store_true")
parser.add_argument("--diversify", action="store_true")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import gymnasium as gym
import numpy as np
import torch

import reBot_RL.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_YML = os.path.join(HERE, "rs_rebot.yml")

import sys
sys.path.insert(0, HERE)
from spike_plan_grasp import (  # noqa: E402
    table_candidates, CAN_CENTER_Z, B_IN, B_T, B_H, B_F, B_OUT, B_WALL_C, CAN_DIMS,
)
import spike_plan_grasp as _spg  # noqa: E402  (LAST_CAND_ROWS instrumentation)
NEW_ROW_MIN = 12953  # rows >= this in grasp_table_extended.pt are the appended entries
# Finger-BODY z minus gripped can-center z AT CLOSE. v14 logged fingers POST-LIFT
# (body 0.092 gripping a can lifted to ~0.088); at close: 0.092 - 0.080 lift = 0.012
# body z on a 0.008 can -> 0.004.
PAD_BODY_OFF = 0.004

Q_HOME = [0.0, -1.35, -0.3, -0.85, 0.0, 0.0]
TCP_OFF = 0.075
CTRL_DT = 0.02
PLACE_MARGIN = 0.045  # placed check: |can - basket|_xy


def build_scene(basket_xy, can_states):
    from curobo.scene import Cuboid, Scene

    bx, by = float(basket_xy[0]), float(basket_xy[1])
    cubs = [
        Cuboid(name="table", dims=[1.2, 1.2, 0.05], pose=[0.25, 0, -0.027, 1, 0, 0, 0]),
        Cuboid(name="b_floor", dims=[B_OUT, B_OUT, B_F], pose=[bx, by, B_F / 2, 1, 0, 0, 0]),
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

    def plan_grasp(self, q6, can_xy, can_z=None, align_axis_xy=None, exclude=(), tz_extra=0.0,
                   exclude_rows=()):
        # target the LIVE can height. Upright: pocket ~16 mm above center. Lying
        # (center z~0.012, top at 0.024): fingers must straddle the cylinder at its
        # CENTERLINE — the old +0.016 put the pocket at 0.028, 4 mm above the can's
        # top surface (air-closes with close_disp=0.0, ep0-style misses).
        if can_z is None:
            tz = None
        else:
            # Lying off calibrated from v6 h5 FK audit: commanded TCP tracks tz-0.023,
            # and execution stalls ~10 mm high on table contact, so tz=0.012 (v6) put
            # fingertips scraping the table (stall lottery: 8-50 mm -> 56% success).
            # off=+0.012 puts EXECUTED fingertips at the lying can's centerline; as a
            # bonus tz>=0.020 lets mid-band table entries serve r<0.221 targets
            # (offline probe: full 16-goalsets where v6 got NO-CANDIDATES).
            off = 0.012 if align_axis_xy is not None else 0.016
            tz = float(np.clip(can_z + off + tz_extra, 0.012, 0.045))
        cands = table_candidates(tuple(can_xy), K=self.K, target_z=tz,
                                 align_axis_xy=align_axis_xy, exclude=exclude,
                                 exclude_rows=exclude_rows)
        if not cands:
            return "no-candidates"
        res = self.planner.plan_grasp(
            grasp_poses=self.goalset(cands), current_state=self.js(q6),
            grasp_approach_axis="z", grasp_approach_offset=0.10, grasp_approach_in_tool_frame=False,
            grasp_lift_axis="z", grasp_lift_offset=0.08, grasp_lift_in_tool_frame=False,
        )
        if not bool(res.success.any().item()):
            self.last_fail_status = str(getattr(res, "status", "?"))
            return None
        return res

    def _carry_orients(self):
        """Tool orientations of the proven carry configs (FK'd once, wxyz)."""
        if not hasattr(self, "_carry_q"):
            from curobo.types import JointState

            d = torch.load(
                "/home/william/Desktop/isaacLab/reBot/reBot_RL/data/pick_place_demos/carry_waypoints.pt",
                map_location="cpu", weights_only=False)
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

    def plan_home(self, q6):
        return self.plan_to_config(q6, Q_HOME)

    def plan_to_config(self, q6, q_goal):
        from curobo.types import JointState

        goal = JointState.from_position(
            torch.as_tensor([list(q_goal)], device="cuda", dtype=torch.float32),
            joint_names=self.planner.joint_names)
        return self.planner.plan_cspace(goal, self.js(q6))

    def carry_configs(self, basket_xy):
        """Azimuth-adjusted proven carry configs (over, lower) for this basket."""
        d = torch.load(
            "/home/william/Desktop/isaacLab/reBot/reBot_RL/data/pick_place_demos/carry_waypoints.pt",
            map_location="cpu", weights_only=False)
        r_b = float(np.hypot(*basket_xy))
        band = 0 if r_b < float(d["r_split"]) else 1
        az_b = math.atan2(basket_xy[1], basket_xy[0])
        az_w = math.atan2(float(d["pocket_over"][band][1]), float(d["pocket_over"][band][0]))
        daz = az_b - az_w
        q_over = d["q_over"][band].numpy().copy(); q_over[0] += daz
        q_lower = d["q_lower"][band].numpy().copy(); q_lower[0] += daz
        # radial landing error of the rotated pocket vs this basket
        r_pocket = float(np.linalg.norm(d["pocket_lower"][band][:2]))
        return q_over.tolist(), q_lower.tolist(), abs(r_pocket - r_b)


def traj_to_qs(result, which=None):
    """Trimmed, 20ms-resampled joint trajectory from a plan result (bug #692 trim)."""
    if which:
        traj = getattr(result, f"{which}_interpolated_trajectory")
        last = getattr(result, f"{which}_interpolated_last_tstep")
    else:
        traj = result.interpolated_trajectory
        last = result.interpolated_last_tstep
    pos = traj.position
    pos = pos.reshape(-1, pos.shape[-1]).cpu().numpy()
    if last is not None:
        n = int(torch.as_tensor(last).view(-1)[0].item())
        pos = pos[: max(n, 2)]
    t_src = np.arange(len(pos)) * 0.025
    t_dst = np.arange(0, t_src[-1] + 1e-9, CTRL_DT)
    out = np.stack([np.interp(t_dst, t_src, pos[:, j]) for j in range(pos.shape[1])], axis=1)
    return out


def build_train_mask(T, segments, outcomes):
    """BC loss mask: 0 over failed-grasp/lost-transport segments (train recovery, never the miss)."""
    mask = np.ones(T, dtype=np.uint8)
    for k, s in enumerate(segments):
        seg = s["seg"]
        if seg is None or seg not in outcomes:
            continue
        if outcomes[seg]["outcome"] in ("missed", "lost"):
            end = segments[k + 1]["t"] if k + 1 < len(segments) else T
            mask[s["t"]:end] = 0
    return mask


class Driver:
    def __init__(self):
        env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
        env_cfg.terminations.time_out = None
        # no auto-resets mid-episode: a dropped can must not silently re-randomize the
        # scene; drops are detected manually and fail the episode. The drop-penalty
        # reward reads this termination term, so it goes too (rewards are unused here).
        env_cfg.terminations.object_dropping = None
        env_cfg.rewards.dropping_penalty = None
        if args_cli.video:
            from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG
            env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(width=640, height=360)
        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed
        self.env = gym.make(args_cli.task, cfg=env_cfg)
        self.u = self.env.unwrapped
        self.robot = self.u.scene["robot"]
        self.expert = Expert()
        self.frames = []
        self.record = False
        self.q_default = None
        # per-step labeling: segments + outcomes let demo-gen build a BC loss mask
        # (mask=0 over segments whose outcome is missed/lost — train recovery, not the miss)
        self.step_idx = 0
        self.segments = []
        self.outcomes = {}
        self.place_fail_dumps = []
        self.h5_path = args_cli.record_h5
        self.last_obs = None
        self.ep_obs = []
        self.ep_act = []
        self.demos = []
        # scripted perturbations (Stage 3.1): one event/episode, armed at mark(), fired in step_action
        self.perturb = args_cli.perturb
        # diversity preservation (Stage 2.2): order shuffle + grasp-candidate dropout
        self.diversify = args_cli.diversify
        self.rng = np.random.default_rng(args_cli.seed)
        self.pending_event = None
        self.grip_override = 0
        self.perturb_log = []
        print(f"    [dbg] body_names={list(self.robot.data.body_names)}")

    # -- state --
    def q6(self):
        return self.robot.data.joint_pos[0, :6].cpu().numpy().tolist()

    def _fids(self):
        if not hasattr(self, "_finger_ids"):
            names = list(self.robot.data.body_names)
            self._finger_ids = [i for i, n in enumerate(names)
                                if any(k in n.lower() for k in ("finger", "left", "right"))]
        return self._finger_ids

    def _finger_world(self):
        """Local-frame positions of the finger bodies (debug instrumentation)."""
        org = self.u.scene.env_origins[0].cpu().numpy()
        fw = self.robot.data.body_pos_w[0, self._fids()].cpu().numpy() - org
        return "[" + ";".join(f"({r[0]:.4f},{r[1]:.4f},{r[2]:.4f})" for r in fw) + "]"

    def _jac_z_nudge(self, q6, dz):
        """Small shoulder/elbow correction producing tool dz with minimal xy motion
        (numerical jacobian on joints 1,2 via planner FK; no cuRobo planning)."""
        kin = self.expert.planner.kinematics

        def fk(q):
            st = kin.compute_kinematics(self.expert.js(list(q)))
            return st.tool_poses.position.reshape(3).cpu().numpy()

        q0 = np.asarray(q6, dtype=float)
        p0 = fk(q0)
        eps = 0.01
        J = np.zeros((3, 2))
        for k, j in enumerate((1, 2)):
            qp = q0.copy()
            qp[j] += eps
            J[:, k] = (fk(qp) - p0) / eps
        A = np.vstack([5.0 * J[2:3], J[0:1], J[1:2]])
        b = np.array([5.0 * dz, 0.0, 0.0])
        dq, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.abs(dq).max() > 0.15:
            return None
        q1 = q0.copy()
        q1[1] += dq[0]
        q1[2] += dq[1]
        return q1.tolist()

    def _grasp_z_correct(self, name):
        """Closed-loop lying-grasp height check (v18). Returns True = abort this row.
        v17 data: bad rows execute pads 3-5cm high and DON'T respond to joint nudges
        (clamped by limits/contact) — only detection + row swap helps. Small residuals
        (4-10mm) still get one jacobian nudge."""
        org_z = float(self.u.scene.env_origins[0, 2].item())
        can_z = float(self.can_pos(name)[2])
        for it in range(2):
            fz = float(self.robot.data.body_pos_w[0, self._fids(), 2].mean().item()) - org_z
            dz = fz - (can_z + PAD_BODY_OFF)
            if dz <= 0.004:
                return False
            if dz > 0.010 or it == 1:
                print(f"    [zfix-abort] {name} dz={dz:.4f} fz={fz:.4f} can_z={can_z:.4f}")
                return True
            q_new = self._jac_z_nudge(self.q6(), -dz)
            if q_new is None:
                return True
            print(f"    [zfix] {name} dz={dz:.4f} fz={fz:.4f} can_z={can_z:.4f}")
            self.hold(12, +1, q=q_new)
        return False

    def can_pos(self, name):
        obj = self.u.scene[name]
        return (obj.data.root_pos_w[0] - self.u.scene.env_origins[0]).cpu().numpy()

    def can_axis(self, name):
        """World cylinder axis (body Y of the Y-up YCB can). Quat is XYZW."""
        x, y, z, w = self.u.scene[name].data.root_quat_w[0].cpu().numpy()
        return np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])

    def basket_xy(self):
        import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
        return mdp.basket_centers_local(self.u)[0].cpu().numpy()

    def placed(self, name):
        p = self.can_pos(name)
        b = self.basket_xy()
        return abs(p[0] - b[0]) < PLACE_MARGIN and abs(p[1] - b[1]) < PLACE_MARGIN and p[2] < 0.05

    # -- perturbation primitives (Stage 3.1) --
    def nudge_can(self, name, min_d=0.02, max_d=0.05):
        """Teleport a can laterally (mirrors mdp.events.nudge_objects pose/vel writes)."""
        obj = self.u.scene[name]
        env_ids = torch.zeros(1, dtype=torch.long, device=self.u.device)
        pose = obj.data.root_pose_w.torch[env_ids].clone()
        d = float(self.rng.uniform(min_d, max_d))
        ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
        local_xy = pose[:, :2] - self.u.scene.env_origins[env_ids, :2]
        local_xy[0, 0] += d * math.cos(ang)
        local_xy[0, 1] += d * math.sin(ang)
        r = float(torch.linalg.norm(local_xy[0]))
        if r > 1e-6:
            local_xy *= min(max(r, 0.19), 0.32) / r  # keep it graspable, not in the robot's lap
        pose[:, :2] = self.u.scene.env_origins[env_ids, :2] + local_xy
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.u.device), env_ids=env_ids)

    def force_slip(self):
        self.grip_override = int(self.rng.integers(3, 6))  # command OPEN for 3-5 steps

    # -- control --
    def step_action(self, q_target6, grip):
        a = torch.zeros(1, 7, device="cuda")
        dq = torch.as_tensor(q_target6, device="cuda", dtype=torch.float32)
        a[0, :6] = (dq - self.q_default) / 0.5
        a[0, 6] = grip
        if self.grip_override > 0:
            a[0, 6] = +1.0
            self.grip_override -= 1
        if self.h5_path is not None:
            self.ep_obs.append(self.last_obs)
            self.ep_act.append(a[0].cpu().numpy().astype("float32"))
        ev = self.pending_event
        if ev is not None and ev["fire_at"] is not None and ev["fire_at"] <= self.step_idx:
            if ev["type"] == "nudge":
                self.nudge_can(ev["target"])
            else:
                self.force_slip()
            self.perturb_log.append({"t": self.step_idx, "type": ev["type"],
                                     "phase": ev["phase"], "target": ev["target"]})
            print(f"    [perturb] type={ev['type']} t={self.step_idx} target={ev['target']}")
            self.pending_event = None
        obs = self.env.step(a)[0]
        if self.h5_path is not None:
            self.last_obs = obs["policy"][0].cpu().numpy().astype("float32")
        self.step_idx += 1
        if self.record:
            cam = self.u.scene["workspace_cam"]
            self.frames.append(cam.data.output["rgb"][0].cpu().numpy())

    def mark(self, phase, seg=None):
        self.segments.append({"t": self.step_idx, "phase": phase, "seg": seg})
        ev = self.pending_event
        if self.perturb and ev is not None and ev["fire_at"] is None and ev["phase"] == phase:
            # arm on the FIRST matching mark only (fire_at set = armed; None'd once fired)
            live = seg.split(":")[0] if seg else None
            if ev["target"] == "live":
                ev["target"] = live
            elif ev["target"] == "other":
                ev["target"] = "object_b" if live == "object_a" else "object_a"
            ev["fire_at"] = self.step_idx + int(self.rng.integers(5, 40))

    def run_traj(self, qs, grip):
        for q in qs:
            self.step_action(q[:6], grip)

    def hold(self, n, grip, q=None):
        qh = self.q6() if q is None else q
        for _ in range(n):
            self.step_action(qh, grip)

    def sync_world(self, held=None):
        cans = {}
        for name in ("object_a", "object_b"):
            cans[name] = (self.can_pos(name), name == held)
        self.expert.planner.update_world(build_scene(self.basket_xy(), cans))

    # -- episode --
    def fetch_one(self, name, metrics, tag="f0"):
        """Grasp + place one can. Returns True if it ends placed."""
        tried = []
        tried_rows = []  # table rows that mis-executed or missed (lying path) — never re-serve
        sid = None
        for attempt in range(3):
            sid = f"{name}:{tag}:g{attempt}"
            self.sync_world()
            p = self.can_pos(name)
            axis = self.can_axis(name)
            lying = abs(float(axis[2])) < 0.5
            exclude = tuple(tried)
            if self.diversify and attempt == 0:
                # retries (attempt > 0) skip the random dropout on purpose: recovery
                # uses the full candidate set minus tried (reliability beats diversity).
                avail = [j for j in range(self.expert.K) if j not in tried]
                dropped = [j for j in avail if self.rng.random() < 0.3]
                if len(avail) - len(dropped) < 8:
                    keep = set(self.rng.choice(avail, size=8, replace=False).tolist())
                    dropped = [j for j in avail if j not in keep]
                extra = sorted(set(dropped))
                exclude = tuple(set(tried) | set(extra))
            res = self.expert.plan_grasp(
                self.q6(), (p[0], p[1]), can_z=float(p[2]),
                align_axis_xy=(float(axis[0]), float(axis[1])) if lying else None,
                exclude=exclude, exclude_rows=tuple(tried_rows) if lying else ())
            metrics["plans"] += 1
            if res == "no-candidates" or res is None:
                metrics["plan_fail"] += 1
                print(f"    [{name} attempt {attempt}] plan_grasp {'NO-CANDIDATES' if res == 'no-candidates' else 'PLAN-FAIL'} "
                      f"(lying={lying}, xy=({p[0]:.3f},{p[1]:.3f}), "
                      f"status={getattr(self.expert, 'last_fail_status', '?') if res is None else 'n/a'})")
                if res is None:
                    # (reset_seed() here regressed plan_fail 4->17 across the whole
                    # run — global solver side effects; do NOT reintroduce)
                    continue
                return False
            tried.append(int(res.goalset_index.view(-1)[0].item()))
            self.mark("approach", sid)
            self.run_traj(traj_to_qs(res, "approach"), +1)
            grasp_qs = traj_to_qs(res, "grasp")
            self.mark("descend", sid)
            self.run_traj(grasp_qs, +1)
            self.mark("close", sid)
            self.hold(6, +1, q=grasp_qs[-1][:6])  # let PD tracking settle before closing
            if lying and self._grasp_z_correct(name):
                # Row mis-executed (pads clamped cm-high; v17 data: nudging can't recover).
                # Exclude the ROW and retry — reverse out open first (planning from the
                # descended pose risks start-state collision).
                gi = int(res.goalset_index.view(-1)[0].item())
                row = _spg.LAST_CAND_ROWS[gi] if gi < len(_spg.LAST_CAND_ROWS) else -1
                if row >= 0:
                    tried_rows.append(row)
                metrics["misexec"] = metrics.get("misexec", 0) + 1
                self.outcomes[sid] = {"outcome": "misexec", "gi": tried[-1], "row": row}
                print(f"    [{name} attempt {attempt}] MISEXEC-ABORT row={row} (pads clamped high; row excluded)")
                self.run_traj(traj_to_qs(res, "lift"), +1)
                continue
            before = self.can_pos(name)
            self.hold(15, -1)
            after_close = self.can_pos(name)
            disp = float(np.linalg.norm(after_close[:2] - before[:2]))
            metrics["close_disp"].append(round(disp, 4))
            self.mark("lift", sid)
            self.run_traj(traj_to_qs(res, "lift"), -1)
            can_z = self.can_pos(name)[2]
            gi = int(res.goalset_index.view(-1)[0].item())
            row = _spg.LAST_CAND_ROWS[gi] if gi < len(_spg.LAST_CAND_ROWS) else -1
            align = _spg.LAST_CAND_ALIGN[gi] if gi < len(_spg.LAST_CAND_ALIGN) else None
            # Executed pocket = FK tool pos + TCP_OFF along tool x-hat (same frame math
            # as Expert._pose_from), vs the can at the moment of close.
            st_dbg = self.expert.planner.kinematics.compute_kinematics(self.expert.js(self.q6()))
            tq = st_dbg.tool_poses.quaternion.reshape(4).cpu().numpy()
            xhat = np.array([1 - 2 * (tq[2] ** 2 + tq[3] ** 2),
                             2 * (tq[1] * tq[2] + tq[0] * tq[3]),
                             2 * (tq[1] * tq[3] - tq[0] * tq[2])])
            # Calibrated against the grasp table ({q,pocket} pairs, std=0 over 200 rows):
            # physical finger pocket = tool_origin - 0.048*tool_xhat. (TCP_OFF=0.075 is
            # the place-goalset hover convention, NOT the grasp pocket offset.)
            tool_p = st_dbg.tool_poses.position.reshape(3).cpu().numpy()
            pocket = tool_p - 0.048 * xhat
            dp = pocket - after_close
            axc = self.can_axis(name)
            print(f"    [{name} attempt {attempt}] gi={gi} "
                  f"close_disp={disp:.4f} can_z_after_lift={can_z:.3f} "
                  f"xy=({p[0]:.3f},{p[1]:.3f}) r={float(np.hypot(p[0], p[1])):.3f} "
                  f"lying={lying} row={row} new={row >= NEW_ROW_MIN} "
                  f"plan_to_close_drift={float(np.linalg.norm(before[:2] - p[:2])):.4f} "
                  f"align={-1.0 if align is None else align:.3f} "
                  f"pocket_minus_can=({dp[0]:.4f},{dp[1]:.4f},{dp[2]:.4f}) "
                  f"can_axis=({float(axc[0]):.2f},{float(axc[1]):.2f},{float(axc[2]):.2f}) "
                  f"tool_p=({tool_p[0]:.4f},{tool_p[1]:.4f},{tool_p[2]:.4f}) "
                  f"xhat=({xhat[0]:.3f},{xhat[1]:.3f},{xhat[2]:.3f}) "
                  f"fingers=({float(self.robot.data.joint_pos[0, 6]):.4f},{float(self.robot.data.joint_pos[0, 7]):.4f}) "
                  f"fpos={self._finger_world()}")
            if can_z > 0.05:
                metrics["clean_grasps"] += 1 if disp < 0.005 else 0
                metrics["grasps"] += 1
                self.outcomes[sid] = {"outcome": "grasped", "close_disp": round(disp, 4),
                                      "clean": bool(disp < 0.005), "gi": tried[-1],
                                      "diversified": bool(self.diversify and attempt == 0)}
                break
            metrics["failed_grasps"] += 1
            if lying and row >= 0:
                tried_rows.append(row)  # closed-and-missed rows never re-serve either
            self.outcomes[sid] = {"outcome": "missed", "close_disp": round(disp, 4),
                                  "clean": False, "gi": tried[-1],
                                  "diversified": bool(self.diversify and attempt == 0)}
            # reopen = first act of recovery (post-detection): trainable, own segment
            self.mark("reopen", f"{name}:{tag}:r{attempt}")
            self.outcomes[f"{name}:{tag}:r{attempt}"] = {"outcome": "recovery"}
            self.hold(10, +1)  # release whatever we pinched and retry
        else:
            return False

        # attach + transport (target can must be in the scene for attach_from_scene)
        self.sync_world(held=None)
        try:
            from curobo.types import Pose
            # identity world_objects_pose_offset is REQUIRED: the sphere fit returns
            # WORLD-frame centers (docstring claims obstacle frame — upstream bug);
            # without it they're written link-local verbatim, putting phantom spheres
            # ~30 cm from the can (sometimes inside the table -> "Start state in
            # collision" = the root cause of the rung-1/2 place refusals).
            self.expert.planner.attachment_manager.attach_from_scene(
                self.expert.js(self.q6()), [name],
                world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0]))
            attached = True
        except Exception as e:
            print("[attach] failed:", e)
            attached = False
        b = self.basket_xy()
        # aim at the emptier half if the other can is already in the basket
        other = "object_b" if name == "object_a" else "object_a"
        tgt = np.array([float(b[0]), float(b[1])])
        if self.placed(other):
            away = tgt - self.can_pos(other)[:2]
            n = np.linalg.norm(away)
            tgt = tgt + (away / n * 0.022 if n > 1e-4 else np.array([0.022, 0.0]))
        cid = f"{name}:{tag}:carry"
        self.mark("transport", cid)
        placed_via = None
        pres = self.expert.plan_place(self.q6(), (float(tgt[0]), float(tgt[1])))
        if pres is not None and bool(pres.success.any().item()):
            placed_via = "pose"
        else:
            print(f"    [{name}] place rung1 {'IK-none' if pres is None else 'trajopt-fail'}; retry high")
            pres = self.expert.plan_place(self.q6(), (float(tgt[0]), float(tgt[1])),
                                          heights=(0.13, 0.16))
            if pres is not None and bool(pres.success.any().item()):
                placed_via = "pose-high"
        if placed_via:
            self.run_traj(traj_to_qs(pres), -1)
            dres = self.expert.plan_place(self.q6(), (float(tgt[0]), float(tgt[1])), heights=(0.07,))
            if dres is not None and bool(dres.success.any().item()):
                self.run_traj(traj_to_qs(dres), -1)
        else:
            # dump the refused case for offline root-cause replay (Gate 1 gap (a))
            self.place_fail_dumps.append({
                "q6": self.q6(), "tgt": [float(tgt[0]), float(tgt[1])],
                "basket_xy": [float(b[0]), float(b[1])], "held": name,
                "cans": {n: self.can_pos(n).tolist() for n in ("object_a", "object_b")},
            })
            # rung 3: proven joint-space carry via plan_cspace, azimuth-adjusted
            q_over, q_lower, r_err = self.expert.carry_configs((float(b[0]), float(b[1])))
            cres = self.expert.plan_to_config(self.q6(), q_over)
            if cres is not None and bool(cres.success.any().item()):
                placed_via = f"carry(r_err={r_err:.3f})"
                self.run_traj(traj_to_qs(cres), -1)
            else:
                # rung 4: direct joint interp to the carry config (the old expert's
                # standard move — both endpoints are high-airspace FK-verified configs)
                metrics["place_plan_fail"] += 1  # keep counting planner refusals
                placed_via = f"carry-direct(r_err={r_err:.3f})"
                q_now = np.array(self.q6())
                for al in np.linspace(0, 1, 60):
                    self.step_action(((1 - al) * q_now + al * np.array(q_over)).tolist(), -1)
            # precision hop: from over-basket, a short pose move to the exact center
            # often succeeds where the long transport plan was refused
            hop = self.expert.plan_place(self.q6(), (float(tgt[0]), float(tgt[1])), heights=(0.07,))
            if hop is not None and bool(hop.success.any().item()):
                placed_via += "+hop"
                self.run_traj(traj_to_qs(hop), -1)
            else:
                q_now = np.array(self.q6())
                for al in np.linspace(0, 1, 30):  # linear descend to q_lower (old DOWNK)
                    self.step_action(((1 - al) * q_now + al * np.array(q_lower)).tolist(), -1)
        # still holding? if the can slipped, don't "place" empty air — report lost
        if self.can_pos(name)[2] < 0.04:
            print(f"    [{name}] LOST GRIP before release (can z={self.can_pos(name)[2]:.3f})")
            self.outcomes[cid] = {"outcome": "lost", "via": placed_via}
            if attached:
                self.expert.planner.attachment_manager.detach()
            return "lost"
        self.outcomes[cid] = {"outcome": "delivered", "via": placed_via}
        held_pos = self.can_pos(name)
        self.mark("release", cid)
        self.hold(12, +1)  # release
        if attached:
            self.expert.planner.attachment_manager.detach()
        self.hold(8, +1)
        print(f"    [{name}] placed_via={placed_via}")
        final = self.can_pos(name)
        print(f"    [{name}] pre-release={held_pos.round(3).tolist()} "
              f"post-release={final.round(3).tolist()} "
              f"basket={b.round(3).tolist()} placed={self.placed(name)}")
        # retreat home
        self.mark("retreat", f"{name}:{tag}")
        self.sync_world()
        hres = self.expert.plan_home(self.q6())
        if hres is not None and bool(hres.success.any().item()):
            self.run_traj(traj_to_qs(hres), +1)
        return self.placed(name)

    def episode(self, i):
        obs = self.env.reset()[0]
        if self.h5_path:
            self.last_obs = obs["policy"][0].cpu().numpy().astype("float32")
            self.ep_obs = []
            self.ep_act = []
        self.q_default = self.robot.data.default_joint_pos[0, :6].clone()
        self.step_idx = 0
        self.segments = []
        self.outcomes = {}
        self.perturb_log = []
        self.pending_event = None
        self.grip_override = 0
        if self.perturb:
            # one event per episode; "other" = the not-currently-fetched can, resolved at arm
            # time (both cans start un-placed, so an "other" always exists when lift/transport
            # first fire)
            choices = [("nudge", "approach", "live"), ("nudge", "lift", "other"),
                       ("slip", "transport", "live"), ("nudge", "transport", "other")]
            typ, phase, target = choices[int(self.rng.integers(len(choices)))]
            self.pending_event = {"type": typ, "phase": phase, "target": target, "fire_at": None}
        self.mark("settle")
        self.hold(10, +1)  # settle
        m = {"plans": 0, "plan_fail": 0, "grasps": 0, "failed_grasps": 0,
             "clean_grasps": 0, "close_disp": [], "place_plan_fail": 0}
        order = sorted(("object_a", "object_b"),
                       key=lambda n: float(np.linalg.norm(self.can_pos(n)[:2])))
        if self.diversify and self.rng.random() < 0.5:
            order = order[::-1]
        m["order"] = list(order)
        for name in order:
            r = self.fetch_one(name, m, tag="f0")
            if r == "lost":  # dropped mid-transport: recover by refetching from live state
                self.mark("retreat", f"{name}:recover")
                hres = self.expert.plan_home(self.q6())
                if hres is not None and bool(hres.success.any().item()):
                    self.run_traj(traj_to_qs(hres), +1)
                self.fetch_one(name, m, tag="f1")
        m["success"] = all(self.placed(n) for n in ("object_a", "object_b"))
        m["placed"] = {n: bool(self.placed(n)) for n in ("object_a", "object_b")}
        m["steps"] = self.step_idx
        m["segments"] = self.segments
        m["outcomes"] = self.outcomes
        m["perturb_steps"] = self.perturb_log
        m["episode_kind"] = "recovery_scripted" if self.perturb_log else "nominal"
        print(f"[ep {i}] success={m['success']} placed={m['placed']} grasps={m['grasps']} "
              f"failed={m['failed_grasps']} clean={m['clean_grasps']} close_disp={m['close_disp']}")
        if self.h5_path:
            self.demos.append({"obs": np.stack(self.ep_obs), "act": np.stack(self.ep_act), "meta": m})
        return m


def main():
    drv = Driver()
    all_m = []
    for i in range(args_cli.episodes):
        drv.record = args_cli.video and i == args_cli.video_ep
        all_m.append(drv.episode(i))
        if drv.record and drv.frames:
            import imageio
            path = os.path.join(HERE, f"expert_v1_ep{args_cli.video_ep}.mp4")
            w = imageio.get_writer(path, fps=50, macro_block_size=1)
            for f in drv.frames:
                w.append_data(f)
            w.close()
            print(f"[video] {path} ({len(drv.frames)} frames) — for Big Will")
            drv.frames = []
    n = len(all_m)
    summary = {
        "episodes": n,
        "success_rate": sum(m["success"] for m in all_m) / n,
        "failed_grasps_mean": sum(m["failed_grasps"] for m in all_m) / n,
        "clean_grasp_frac": (sum(m["clean_grasps"] for m in all_m) /
                             max(1, sum(m["grasps"] for m in all_m))),
        "plan_fail": sum(m["plan_fail"] for m in all_m),
        "place_plan_fail": sum(m["place_plan_fail"] for m in all_m),
    }
    if args_cli.perturb:
        pert = [m for m in all_m if m.get("perturb_steps")]
        rate = sum(m["success"] for m in pert) / len(pert) if pert else None
        summary["perturbed_eps"] = len(pert)
        summary["perturbed_success_rate"] = rate
        summary["recovery_success_rate"] = rate  # Gate 1 wording alias
    print("SUMMARY:", json.dumps(summary, indent=2))
    with open(os.path.join(HERE, "expert_v1_results.json"), "w") as f:
        json.dump({"summary": summary, "episodes": all_m}, f, indent=2)
    if drv.h5_path:
        import h5py
        with h5py.File(os.path.join(HERE, drv.h5_path), "w") as f:
            grp = f.create_group("data")
            for i, d in enumerate(drv.demos):
                g = grp.create_group(f"demo_{i}")
                g.create_dataset("obs/policy", data=d["obs"], compression="gzip")
                g.create_dataset("actions", data=d["act"], compression="gzip")
                T = len(d["act"])
                g.create_dataset("train_mask", data=build_train_mask(T, d["meta"]["segments"], d["meta"]["outcomes"]))
                g.attrs["success"] = bool(d["meta"]["success"])
                g.attrs["num_samples"] = T
                g.attrs["episode_kind"] = d["meta"].get("episode_kind", "nominal")
                g.attrs["perturb_steps"] = json.dumps(d["meta"].get("perturb_steps", []))
                g.attrs["segments"] = json.dumps(d["meta"]["segments"])
                g.attrs["outcomes"] = json.dumps(d["meta"]["outcomes"])
            f["data"].attrs["total"] = len(drv.demos)
        print(f"[h5] wrote {len(drv.demos)} demos -> {drv.h5_path}")
    if drv.place_fail_dumps:
        torch.save(drv.place_fail_dumps, os.path.join(HERE, "place_fail_cases.pt"))
        print(f"[dump] {len(drv.place_fail_dumps)} refused place cases -> place_fail_cases.pt")
    drv.env.close()
    app.close()


if __name__ == "__main__":
    main()
