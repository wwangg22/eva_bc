"""cuRoboV2 planning expert driving Rebot-Workstation-PickPlace1-Play-v0.

**The pick-and-place expert (`expert/run_expert_v1.py`), pointed at the workstation task.**
Same pipeline per episode: plan_grasp (goalset from the proven-table candidates,
azimuth-rotation construction) -> execute approach/descent at 50 Hz -> close + clean-grasp
check -> lift + hold check (regrasp on failure, 3 attempts) -> attach -> plan transport to
the box goalset -> release -> verify placed -> home. Memoryless at task level: every plan
reads live privileged state. The planning machinery is IMPORTED from the pick-and-place
expert (`spike_plan_grasp.table_candidates`, the shared 12953-entry
`data/pick_place_demos/grasp_table.pt`, the carry-config fallback rungs) rather than
copied — one grasp source, kept in sync by construction.

What is workstation-specific, and nothing else is:

* **one grasp target** — the 56 mm authored Rubix cube (`mdp.TARGET_NAME`); the roll of
  tape and the tape measure are obstacles in the collision scene, never grasped.
* **the receptacle is the authored open box** (218 x 150 x 93 mm outer, 3 mm walls), which
  RANDOMISES in position and yaw — the collision scene builds its floor + 4 walls at the
  live pose, and `placed` is the env's own `mdp.placed_mask`, not a copied margin check.
* **cube yaw alignment.** A cube is not azimuthally symmetric like an upright can: the
  candidate filter gets the cube's face axis (`align_axis_xy`), whose max-of-two scoring
  (tool-y ∥ axis == fingers on that face pair, tool-x ∥ axis == fingers on the other pair)
  covers both valid grasp families in one call. The grasp-table z band (pocket
  0.012–0.045 m) meets the cube at `cube_z + 0.016 = 0.044` — pads on the upper-middle of
  the 56 mm cube, the same region the verified ArmKin expert's swept `GRIP_Z = 0.056`
  (TCP at the top face) lands the fingers.
* **place heights clear a 93 mm wall**, not pick-place's 40 mm basket: approach hovers
  (0.17, 0.20), precision drop 0.14 — bracketing the ArmKin expert's swept
  `CARRY_Z = 0.175`. The cube's bottom clears the wall by >20 mm throughout.
* **no lying-can machinery.** The cube cannot lie down, so the v17/v18 lying-row
  mis-execution detection (`_grasp_z_correct`, `PAD_BODY_OFF`) has nothing to detect and
  is deliberately absent rather than ported dead.

The env's spawn annulus was narrowed to 0.20–0.28 m for exactly this expert
(reBot_RL cb7148b): the grasp table spans tool radii 0.221–0.337 m and refuses distant
substitutions, and at the old 0.15 m minimum only 57 % of spawns had a goalset (measured);
with the new band, 200/200.

Run (env_isaaclab6):
  python -u re3sim/expert/run_expert_ws.py --episodes 32
  python -u re3sim/expert/run_expert_ws.py --episodes 4 --video   # + station/wrist mp4s

Outputs: expert_ws_results.json (+ expert_ws_ep0_{station,wrist}.mp4 with --video,
optional --record-h5 demos with the same segments/outcomes/train_mask labelling scheme as
the pick-and-place expert — mask=0 over "missed"/"lost" segments, recovery trainable).
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
parser.add_argument("--video-all", action="store_true",
                    help="film EVERY episode (station + wrist mp4s, outcome in the filename) "
                         "— for review runs where the successes are wanted on film")
parser.add_argument("--record-h5", type=str, default=None)
parser.add_argument("--perturb", action="store_true")
parser.add_argument("--diversify", action="store_true")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--task", default="Rebot-Workstation-PickPlace1-Play-v0")
parser.add_argument("--shards", type=str, default=None, metavar="DIR",
                    help="Write per-episode VISION SHARDS (act/dataset_vision.py layout: "
                         "wrist_rgb / workspace_rgb uint8, proprio 23, obs41, actions) for "
                         "the student. Successful episodes only unless "
                         "--shard-include-failures. Needs a -Vision*/-VisionDR task (the "
                         "scene owns the student cameras) and --seed.")
parser.add_argument("--shard-include-failures", action="store_true")
parser.add_argument("--dagger-ckpt", type=str, default=None, metavar="CKPT",
                    help="DAgger takeover (collect_demos.py's design, on THIS expert per "
                         "directive): the vision student drives dagger-min..max steps after "
                         "warmup, then the drifted state becomes the episode's initial "
                         "condition — the cuRobo expert plans from it and the recorded "
                         "shard starts there. Requires --shards.")
parser.add_argument("--dagger-min", type=int, default=60)
parser.add_argument("--dagger-max", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.shards:
    if args_cli.video or args_cli.video_all:
        parser.error("--shards and --video record at different resolutions; separate runs")
    if "Vision" not in args_cli.task:
        parser.error("--shards needs a -Vision*/-VisionDR-* task: the scene owns the "
                     "student cameras there (plain tasks have no cameras to record)")
    if args_cli.seed is None:
        parser.error("--shards requires --seed (shard filenames + reproducibility)")
if args_cli.dagger_ckpt and not args_cli.shards:
    parser.error("--dagger-ckpt only makes sense with --shards: the point is to write "
                 "vision shards from the states the student actually reaches")
args_cli.headless = True
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks.manager_based.re3sim.mdp as mdp  # noqa: E402
from reBot_RL.tasks.manager_based.re3sim.workstation_env_cfg import (  # noqa: E402
    STATION_CAM_EYE, STATION_CAM_TARGET,
)

HERE = os.path.dirname(os.path.abspath(__file__))
#: the pick-and-place expert's home — planning machinery is imported from there
_EXPERT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "expert"))

import sys  # noqa: E402

sys.path.insert(0, _EXPERT_DIR)
from spike_plan_grasp import table_candidates  # noqa: E402
import spike_plan_grasp as _spg  # noqa: E402  (LAST_CAND_ROWS instrumentation)

Q_HOME = [0.0, -1.35, -0.3, -0.85, 0.0, 0.0]   # == _START_POSE arm joints, same as pick-place
TCP_OFF = 0.075          # place-goalset hover convention (NOT the grasp pocket offset)
CTRL_DT = 0.02
#: pocket-height offset above the cube CENTER for the grasp-table z target: 0.028 + 0.016
#: = 0.044, inside the table's 0.012–0.045 band, pads on the cube's upper-middle.
CUBE_TZ_OFF = 0.016
#: TCP hover heights over the box for the transport goalset, and the precision drop. The
#: box wall is 0.093; the verified ArmKin expert releases at CARRY_Z = 0.175.
PLACE_HEIGHTS = (0.17, 0.20)
PLACE_HEIGHTS_HIGH = (0.20, 0.23)
DROP_HEIGHT = 0.14
#: "the grasp took": cube centre above spawn height (0.028) by more than settle noise.
#: Same predicate family as the ArmKin expert's HOLD_RISE = 0.030.
LIFT_OK_Z = 0.060

CUBE = mdp.TARGET_NAME
CLUTTER = list(mdp.CLUTTER_NAMES)


def _yaw_quat_wxyz(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def build_scene(box_xy, box_yaw, obj_states):
    """cuRobo collision scene: desk slab, the box as floor + 4 yawed walls, the objects.

    Same role as run_expert_v1.build_scene; the box replaces the basket and carries a YAW
    (reset_box randomises it), so every box cuboid is posed with the yaw quaternion.
    """
    from curobo.scene import Cuboid, Scene

    bx, by = float(box_xy[0]), float(box_xy[1])
    yq = _yaw_quat_wxyz(float(box_yaw))
    c, s = math.cos(float(box_yaw)), math.sin(float(box_yaw))

    def at(dx, dy, z):  # box-local xy offset -> world pose at height z
        return [bx + c * dx - s * dy, by + s * dx + c * dy, z] + yq

    ox, oy = mdp.BOX_OUTER_X, mdp.BOX_OUTER_Y
    wt, wh, ft = mdp.BOX_WALL, mdp.BOX_HEIGHT, mdp.BOX_FLOOR_THICKNESS
    cubs = [
        # ⭐ the desk, recessed 2 mm (top at z = -0.002, not 0.0). A FLUSH slab kills EVERY
        # plan_grasp with "Goalset planning returned None": the grasp poses put finger
        # collision spheres at table level and a top face at exactly 0 collides with them.
        # Measured by probe_ws_plan.py (30/30 rows: flush=FAIL, recessed=OK, cube and
        # clutter obstacles innocent) — the same lesson spike_plan_grasp.py's z=-0.027
        # already encoded for the pick-place scene, re-learned here the hard way.
        Cuboid(name="table", dims=[1.60, 1.00, 0.05], pose=[0.4, 0.0, -0.027, 1, 0, 0, 0]),
        Cuboid(name="b_floor", dims=[ox, oy, ft], pose=at(0.0, 0.0, ft / 2)),
        Cuboid(name="b_px", dims=[wt, oy, wh], pose=at(+(ox - wt) / 2, 0.0, wh / 2)),
        Cuboid(name="b_nx", dims=[wt, oy, wh], pose=at(-(ox - wt) / 2, 0.0, wh / 2)),
        Cuboid(name="b_py", dims=[ox, wt, wh], pose=at(0.0, +(oy - wt) / 2, wh / 2)),
        Cuboid(name="b_ny", dims=[ox, wt, wh], pose=at(0.0, -(oy - wt) / 2, wh / 2)),
    ]
    dims = {
        CUBE: [mdp.CUBE_SIZE] * 3,
        "rolloftape": [mdp.TAPE_DIAMETER, mdp.TAPE_DIAMETER, mdp.TAPE_HEIGHT],
        "tapemeasure": [mdp.TAPEMEASURE_LONG, mdp.TAPEMEASURE_ACROSS, mdp.TAPEMEASURE_HEIGHT],
    }
    for name, (xyz, yaw, in_hand) in obj_states.items():
        if in_hand:
            continue  # attached: lives on the robot, not in the world
        cubs.append(Cuboid(
            name=name, dims=list(dims[name]),
            pose=[float(xyz[0]), float(xyz[1]), max(float(xyz[2]), 0.012)]
                 + _yaw_quat_wxyz(float(yaw))))
    return Scene(cuboid=cubs)


class Expert:
    """run_expert_v1.Expert with the cube's z-target and the box goalset; nothing else."""

    def __init__(self):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        cfg = MotionPlannerCfg.create(
            robot=os.path.join(_EXPERT_DIR, "rs_rebot.yml"),
            scene_model=None, collision_cache={"cuboid": 16}, max_goalset=16,
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

    def plan_grasp(self, q6, cube_xy, cube_z, face_axis_xy, exclude=(), exclude_rows=()):
        """Goalset grasp on the cube. Upright-object z formula only — a cube cannot lie."""
        tz = float(np.clip(cube_z + CUBE_TZ_OFF, 0.012, 0.045))
        cands = table_candidates(tuple(cube_xy), K=self.K, target_z=tz,
                                 align_axis_xy=face_axis_xy, exclude=exclude,
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

    # --- transport: identical machinery to run_expert_v1, box for basket -----------------

    def _carry_orients(self):
        if not hasattr(self, "_carry_q"):
            from curobo.types import JointState

            d = torch.load(_CARRY_PT, map_location="cpu", weights_only=False)
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
    def _pose_from(qn, box_xy, tz):
        xx = 1 - 2 * (qn[2] ** 2 + qn[3] ** 2)
        xy = 2 * (qn[1] * qn[2] + qn[0] * qn[3])
        xz = 2 * (qn[1] * qn[3] - qn[0] * qn[2])
        tcp = np.array([box_xy[0], box_xy[1], tz])
        return (tcp + TCP_OFF * np.array([xx, xy, xz])).tolist(), list(qn)

    def place_goalset(self, q6, box_xy, heights=PLACE_HEIGHTS):
        st = self.planner.kinematics.compute_kinematics(self.js(q6))
        gpos = st.tool_poses.position.reshape(3).cpu().numpy()
        gq = st.tool_poses.quaternion.reshape(4).cpu().numpy()
        az_b = math.atan2(box_xy[1], box_xy[0])
        poses = []
        for daz in (-0.26, 0.0, 0.26):
            qn = self._rotz_quat(gq, az_b - math.atan2(gpos[1], gpos[0]) + daz)
            for tz in heights:
                poses.append(self._pose_from(qn, box_xy, tz))
        cq, caz = self._carry_orients()
        for i in range(len(cq)):
            for daz in (0.0, -0.26, 0.26):
                qn = self._rotz_quat(cq[i], az_b - caz[i] + daz)
                poses.append(self._pose_from(qn, box_xy, heights[0]))
        n = len(poses)
        while len(poses) < self.K:
            poses.append(poses[len(poses) % n])
        return poses[: self.K]

    def plan_place(self, q6, box_xy, heights=PLACE_HEIGHTS):
        return self.planner.plan_pose(self.goalset(self.place_goalset(q6, box_xy, heights)),
                                      self.js(q6))

    def plan_home(self, q6):
        return self.plan_to_config(q6, Q_HOME)

    def plan_to_config(self, q6, q_goal):
        from curobo.types import JointState

        goal = JointState.from_position(
            torch.as_tensor([list(q_goal)], device="cuda", dtype=torch.float32),
            joint_names=self.planner.joint_names)
        return self.planner.plan_cspace(goal, self.js(q6))

    def carry_configs(self, box_xy):
        d = torch.load(_CARRY_PT, map_location="cpu", weights_only=False)
        r_b = float(np.hypot(*box_xy))
        band = 0 if r_b < float(d["r_split"]) else 1
        az_b = math.atan2(box_xy[1], box_xy[0])
        az_w = math.atan2(float(d["pocket_over"][band][1]), float(d["pocket_over"][band][0]))
        daz = az_b - az_w
        q_over = d["q_over"][band].numpy().copy(); q_over[0] += daz
        q_lower = d["q_lower"][band].numpy().copy(); q_lower[0] += daz
        r_pocket = float(np.linalg.norm(d["pocket_lower"][band][:2]))
        return q_over.tolist(), q_lower.tolist(), abs(r_pocket - r_b)


#: proven carry configs, shared with the pick-and-place expert (reBot/ is two up from
#: reBot_ACT/expert)
_CARRY_PT = os.path.join(
    os.path.dirname(os.path.dirname(_EXPERT_DIR)),
    "reBot_RL", "data", "pick_place_demos", "carry_waypoints.pt")


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
    return np.stack([np.interp(t_dst, t_src, pos[:, j]) for j in range(pos.shape[1])], axis=1)


def build_train_mask(T, segments, outcomes):
    """BC loss mask: 0 over failed-grasp/lost-transport segments — train recovery, never the miss."""
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
        # no auto-resets mid-episode: a dropped cube must not silently re-randomize the
        # scene (and, on the DR task, re-randomize the CAMERAS mid-recording); drops are
        # detected manually and fail the episode.
        env_cfg.terminations.target_dropped = None
        if args_cli.video or args_cli.video_all:
            import isaaclab.sim as sim_utils
            from isaaclab.sensors import CameraCfg
            from reBot_RL.tasks.manager_based.lift.camera_cfg import WRIST_CAM_CFG

            env_cfg.scene.wrist_cam = WRIST_CAM_CFG.replace(
                width=640, height=480, data_types=["rgb"], update_period=0.0)
            env_cfg.scene.station_cam = CameraCfg(
                prim_path="{ENV_REGEX_NS}/StationCam", update_period=0.0,
                width=640, height=480, data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=17.0, horizontal_aperture=20.955,
                                                 clipping_range=(0.02, 20.0)))
        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed
        self.env = gym.make(args_cli.task, cfg=env_cfg)
        self.u = self.env.unwrapped
        self.robot = self.u.scene["robot"]
        self.expert = Expert()
        self.frames = {"station": [], "wrist": []}
        self.record = False
        self.q_default = None
        self.step_idx = 0
        self.segments = []
        self.outcomes = {}
        self.place_fail_dumps = []
        self.h5_path = args_cli.record_h5
        self.last_obs = None
        self.ep_obs = []
        self.ep_act = []
        self.demos = []
        self.perturb = args_cli.perturb
        self.diversify = args_cli.diversify
        self.rng = np.random.default_rng(args_cli.seed)
        self.pending_event = None
        self.grip_override = 0
        self.perturb_log = []
        # -- vision-shard recording (see --shards) --
        self.shards = args_cli.shards
        self.shard_rec = False       # true only between post-warmup and episode end
        self.sh = None               # per-episode buffers
        self.last_pol = None         # obs["policy"][0] BEFORE the pending action
        # -- DAgger takeover (see --dagger-ckpt) --
        self.dagger = None
        if args_cli.dagger_ckpt:
            sys.path.insert(0, os.path.join(HERE, "..", "act"))
            from policy_runner_vision import (  # noqa: PLC0415
                VisionController, build_student_batch, load_vision_checkpoint,
            )
            pol, nrm, cfg, cam_shape = load_vision_checkpoint(args_cli.dagger_ckpt, "cuda:0")
            got = tuple(self.u.scene["wrist_cam"].image_shape)
            if tuple(cam_shape[1:]) != got:
                raise SystemExit(
                    f"student was trained at {tuple(cam_shape[1:])} but the task renders "
                    f"{got}; a student driven at a resolution it never saw is not the "
                    f"student whose states we want to label")
            self.dagger = {"ctrl": VisionController(pol, nrm, cfg, "cuda:0"),
                           "batch": build_student_batch}
            print(f"[dagger] student {args_cli.dagger_ckpt} drives "
                  f"{args_cli.dagger_min}-{args_cli.dagger_max} steps before the cuRobo "
                  f"expert takes over", flush=True)

    # -- state --
    def q6(self):
        return self.robot.data.joint_pos[0, :6].cpu().numpy().tolist()

    def obj_pos(self, name):
        obj = self.u.scene[name]
        return (obj.data.root_pos_w[0] - self.u.scene.env_origins[0]).cpu().numpy()

    def obj_yaw(self, name):
        """World yaw of a (flat) object from its root quaternion.

        ⭐ `root_quat_w` is **XYZW** in this build (the og expert's `can_axis` says so, and
        `events.reset_objects` writes sin/cos into pose columns 5/6 — z and w of an xyzw
        layout). Unpacking it wxyz made every cube read yaw = π exactly, so the grasp
        alignment axis was a CONSTANT wrong direction: 12-ep run, every air-close at
        "yaw=3.14", every success at a genuinely-read yaw. The exact bug class the smoke
        instrumentation exists to catch.
        """
        x, y, z, w = self.u.scene[name].data.root_quat_w[0].cpu().numpy()
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def cube_face_axis(self):
        """World xy direction of one cube face normal — `table_candidates`' max-of-two
        alignment scoring makes one axis cover both valid grasp families."""
        yaw = self.obj_yaw(CUBE)
        return (math.cos(yaw), math.sin(yaw))

    def box_xy(self):
        return mdp.box_centers_local(self.u)[0].cpu().numpy()

    def box_yaw(self):
        return float(mdp.box_yaws(self.u)[0].item())

    def placed(self):
        return bool(mdp.placed_mask(self.u)[0].item())

    def pocket_now(self):
        """Executed finger pocket, world frame: FK tool origin − 0.048 · tool-x̂ (the og
        expert's calibration, std = 0 over 200 grasp-table rows)."""
        st = self.expert.planner.kinematics.compute_kinematics(self.expert.js(self.q6()))
        tq = st.tool_poses.quaternion.reshape(4).cpu().numpy()
        xhat = np.array([1 - 2 * (tq[2] ** 2 + tq[3] ** 2),
                         2 * (tq[1] * tq[2] + tq[0] * tq[3]),
                         2 * (tq[1] * tq[3] - tq[0] * tq[2])])
        return st.tool_poses.position.reshape(3).cpu().numpy() - 0.048 * xhat

    # -- perturbation primitives --
    def nudge_obj(self, name, min_d=0.02, max_d=0.05):
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
            # keep it inside the band the grasp table serves (see module docstring)
            local_xy *= min(max(r, 0.20), 0.28) / r
        pose[:, :2] = self.u.scene.env_origins[env_ids, :2] + local_xy
        obj.write_root_pose_to_sim(pose, env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.u.device), env_ids=env_ids)

    def force_slip(self):
        self.grip_override = int(self.rng.integers(3, 6))

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
        if self.shard_rec:
            # frame-precedes-action: the student learns (image_t, proprio_t) -> a_t, so
            # grab the render + obs of the CURRENT state before this action steps the sim
            for key, sensor in (("wrist", "wrist_cam"), ("station", "station_cam")):
                cam = self.u.scene[sensor]
                cam.update(dt=0.0)
                fr = cam.data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().clone()
                self.sh[key].append(fr)
            self.sh["obs"].append(self.last_pol.clone())
            self.sh["act"].append(a[0].detach().cpu().clone())
        ev = self.pending_event
        if ev is not None and ev["fire_at"] is not None and ev["fire_at"] <= self.step_idx:
            if ev["type"] == "nudge":
                self.nudge_obj(ev["target"])
            else:
                self.force_slip()
            self.perturb_log.append({"t": self.step_idx, "type": ev["type"],
                                     "phase": ev["phase"], "target": ev["target"]})
            print(f"    [perturb] type={ev['type']} t={self.step_idx} target={ev['target']}")
            self.pending_event = None
        obs = self.env.step(a)[0]
        if self.h5_path is not None:
            self.last_obs = obs["policy"][0].cpu().numpy().astype("float32")
        if self.shards is not None:
            self.last_pol = obs["policy"][0].detach().cpu()
        self.step_idx += 1
        if self.record:
            for key, sensor in (("station", "station_cam"), ("wrist", "wrist_cam")):
                cam = self.u.scene[sensor]
                cam.update(dt=0.0)
                self.frames[key].append(cam.data.output["rgb"][0, ..., :3].cpu().numpy())

    def mark(self, phase, seg=None):
        self.segments.append({"t": self.step_idx, "phase": phase, "seg": seg})
        ev = self.pending_event
        if self.perturb and ev is not None and ev["fire_at"] is None and ev["phase"] == phase:
            ev["fire_at"] = self.step_idx + int(self.rng.integers(5, 40))

    def run_traj(self, qs, grip):
        for q in qs:
            self.step_action(q[:6], grip)

    def hold(self, n, grip, q=None):
        qh = self.q6() if q is None else q
        for _ in range(n):
            self.step_action(qh, grip)

    def aim_station(self):
        """One env: the single-pose set_world_poses_from_view is safe here (and the
        -VisionDR task owns the pose via its reset event, so it is skipped there)."""
        if not (args_cli.video or args_cli.video_all):
            return
        if getattr(self.u.cfg.events, "aim_station_cam", None) is not None:
            return
        org = self.u.scene.env_origins
        self.u.scene["station_cam"].set_world_poses_from_view(
            org + torch.tensor(STATION_CAM_EYE, device=self.u.device),
            org + torch.tensor(STATION_CAM_TARGET, device=self.u.device))

    def dagger_drive(self):
        """Let the student drive, then hand the scene to the expert wherever it ends up.

        Mirrors collect_demos.py's takeover design: runs after the splat warmup (a student
        driven on frames with no desk is not the student being evaluated) and BEFORE the
        shard buffers open, so the drifted state is simply the episode's initial condition —
        the expert plans from it via the live-state reads it already does, and no other code
        knows DAgger happened. Frame ordering matches step_action's frame-precedes-action:
        cameras are update()d for the CURRENT state before the student acts on it.

        The freeze at the end is UNRECORDED and mirrors collect_demos.py's frz hold: the
        gripper opens and the world settles BEFORE the shard opens. Round 1 (dagger_vdr_s41)
        omitted it, so 4.6% of shards opened on a held/falling cube with 'open the gripper'
        as the first labels — supervision that directly contradicts the carry."""
        k = int(self.rng.integers(args_cli.dagger_min, args_cli.dagger_max + 1))
        ctrl = self.dagger["ctrl"]
        ctrl.reset()
        obs41 = self.u.observation_manager.compute()["policy"]
        for _ in range(k):
            for name in ("wrist_cam", "station_cam"):
                self.u.scene[name].update(dt=0.0)
            batch = self.dagger["batch"](
                obs41, {"wrist": self.u.scene["wrist_cam"],
                        "workspace": self.u.scene["station_cam"]}, "cuda:0")
            a = ctrl.act(batch).to("cuda:0")
            obs41 = self.env.step(a)[0]["policy"]
        frz = torch.zeros(1, 7, device="cuda")
        frz[0, :6] = (self.robot.data.joint_pos[0, :6] - self.q_default) / 0.5
        frz[0, 6] = 1.0
        for _ in range(40):
            obs41 = self.env.step(frz)[0]["policy"]
        self.last_pol = obs41[0].detach().cpu()
        print(f"    [dagger] student drove {k} steps before takeover", flush=True)

    def sync_world(self, held=None):
        objs = {}
        for name in (CUBE, *CLUTTER):
            objs[name] = (self.obj_pos(name), self.obj_yaw(name), name == held)
        self.expert.planner.update_world(build_scene(self.box_xy(), self.box_yaw(), objs))

    # -- episode --
    def fetch_cube(self, metrics, tag="f0"):
        """Grasp + place the cube. Returns True if it ends placed, or "lost"."""
        tried = []
        tried_rows = []
        sid = None
        for attempt in range(3):
            sid = f"{CUBE}:{tag}:g{attempt}"
            self.sync_world()
            p = self.obj_pos(CUBE)
            exclude = tuple(tried)
            if self.diversify and attempt == 0:
                avail = [j for j in range(self.expert.K) if j not in tried]
                dropped = [j for j in avail if self.rng.random() < 0.3]
                if len(avail) - len(dropped) < 8:
                    keep = set(self.rng.choice(avail, size=8, replace=False).tolist())
                    dropped = [j for j in avail if j not in keep]
                exclude = tuple(set(tried) | set(sorted(set(dropped))))
            res = self.expert.plan_grasp(
                self.q6(), (p[0], p[1]), float(p[2]), self.cube_face_axis(),
                exclude=exclude, exclude_rows=tuple(tried_rows))
            metrics["plans"] += 1
            if res == "no-candidates" or res is None:
                # fall back to the UNALIGNED candidate family before burning the attempt:
                # alignment is a preference (fingers meet a face pair square), not a
                # requirement — the 89 mm opening clears the cube's 79 mm diagonal, so a
                # corner-first close usually still seats, and the ArmKin expert's lift
                # screen never required face alignment at all.
                res = self.expert.plan_grasp(
                    self.q6(), (p[0], p[1]), float(p[2]), None,
                    exclude=exclude, exclude_rows=tuple(tried_rows))
                if res not in ("no-candidates", None):
                    metrics["align_fallback"] = metrics.get("align_fallback", 0) + 1
                    print(f"    [{CUBE} attempt {attempt}] aligned goalset failed; "
                          f"UNALIGNED fallback planned")
            if res == "no-candidates" or res is None:
                metrics["plan_fail"] += 1
                print(f"    [{CUBE} attempt {attempt}] plan_grasp "
                      f"{'NO-CANDIDATES' if res == 'no-candidates' else 'PLAN-FAIL'} "
                      f"(xy=({p[0]:.3f},{p[1]:.3f}), r={float(np.hypot(p[0], p[1])):.3f}, "
                      f"status={getattr(self.expert, 'last_fail_status', '?') if res is None else 'n/a'})")
                if res is None:
                    continue
                return False
            tried.append(int(res.goalset_index.view(-1)[0].item()))
            self.mark("approach", sid)
            self.run_traj(traj_to_qs(res, "approach"), +1)
            grasp_qs = traj_to_qs(res, "grasp")
            self.mark("descend", sid)
            self.run_traj(grasp_qs, +1)
            self.mark("close", sid)
            self.hold(6, +1, q=grasp_qs[-1][:6])   # let PD tracking settle before closing
            # ⭐ Pre-close mis-execution gate — the og expert's `_grasp_z_correct`,
            # generalised: some rows EXECUTE centimetres from where they plan (v17's
            # lesson: they don't respond to nudges, only detection + row swap helps).
            # Measured here before the gate existed (64 eps, seed 303): 5 episodes lost
            # to closes whose executed pocket sat 3–10 cm from the cube, shoving it off
            # the desk lane. Fingers closing on nothing teach nothing; abort the row.
            before = self.obj_pos(CUBE)
            pk = self.pocket_now()
            mis_xy = float(np.linalg.norm(pk[:2] - before[:2]))
            # XY only. A z-gate on the FK pocket was tried and REGRESSED 76.6 % -> 62.5 %:
            # the -0.048·x̂ correction is XY-calibrated (the og's z check read finger BODY
            # heights from the sim, a different signal), so FK-pocket z read 0.07-0.09 on
            # healthy rows and the gate burned every attempt of otherwise-good episodes.
            if mis_xy > 0.022:
                gi_bad = tried[-1]
                row_bad = _spg.LAST_CAND_ROWS[gi_bad] if gi_bad < len(_spg.LAST_CAND_ROWS) else -1
                if row_bad >= 0:
                    tried_rows.append(row_bad)
                metrics["misexec"] = metrics.get("misexec", 0) + 1
                self.outcomes[sid] = {"outcome": "misexec", "gi": gi_bad, "row": row_bad,
                                      "mis_xy": round(mis_xy, 4), "pocket_z": round(float(pk[2]), 4)}
                print(f"    [{CUBE} attempt {attempt}] MISEXEC-ABORT row={row_bad} "
                      f"mis_xy={mis_xy:.4f} pocket_z={pk[2]:.4f} (row excluded)")
                self.run_traj(traj_to_qs(res, "lift"), +1)   # reverse out, fingers open
                self.sync_world()
                hres = self.expert.plan_home(self.q6())
                if hres is not None and bool(hres.success.any().item()):
                    self.run_traj(traj_to_qs(hres), +1)
                continue
            self.hold(15, -1)
            after_close = self.obj_pos(CUBE)
            disp = float(np.linalg.norm(after_close[:2] - before[:2]))
            metrics["close_disp"].append(round(disp, 4))
            self.mark("lift", sid)
            self.run_traj(traj_to_qs(res, "lift"), -1)
            cube_z = self.obj_pos(CUBE)[2]
            gi = tried[-1]
            row = _spg.LAST_CAND_ROWS[gi] if gi < len(_spg.LAST_CAND_ROWS) else -1
            align = _spg.LAST_CAND_ALIGN[gi] if gi < len(_spg.LAST_CAND_ALIGN) else None
            # Executed pocket vs the cube at close — the og expert's calibrated
            # diagnostic (pocket = tool_origin - 0.048 * tool_xhat, std=0 over 200 rows).
            st_dbg = self.expert.planner.kinematics.compute_kinematics(
                self.expert.js(self.q6()))
            tq = st_dbg.tool_poses.quaternion.reshape(4).cpu().numpy()
            xhat = np.array([1 - 2 * (tq[2] ** 2 + tq[3] ** 2),
                             2 * (tq[1] * tq[2] + tq[0] * tq[3]),
                             2 * (tq[1] * tq[3] - tq[0] * tq[2])])
            tool_p = st_dbg.tool_poses.position.reshape(3).cpu().numpy()
            dp = (tool_p - 0.048 * xhat) - after_close
            print(f"    [{CUBE} attempt {attempt}] gi={gi} row={row} "
                  f"align={-1.0 if align is None else align:.3f} "
                  f"close_disp={disp:.4f} cube_z_after_lift={cube_z:.3f} "
                  f"xy=({p[0]:.3f},{p[1]:.3f}) r={float(np.hypot(p[0], p[1])):.3f} "
                  f"yaw={self.obj_yaw(CUBE):.2f} "
                  f"pocket_minus_cube=({dp[0]:.4f},{dp[1]:.4f},{dp[2]:.4f}) "
                  f"fingers=({float(self.robot.data.joint_pos[0, 6]):.4f},"
                  f"{float(self.robot.data.joint_pos[0, 7]):.4f})")
            if cube_z > LIFT_OK_Z:
                metrics["clean_grasps"] += 1 if disp < 0.005 else 0
                metrics["grasps"] += 1
                self.outcomes[sid] = {"outcome": "grasped", "close_disp": round(disp, 4),
                                      "clean": bool(disp < 0.005), "gi": gi,
                                      "diversified": bool(self.diversify and attempt == 0)}
                break
            metrics["failed_grasps"] += 1
            if row >= 0:
                tried_rows.append(row)   # closed-and-missed rows never re-serve
            self.outcomes[sid] = {"outcome": "missed", "close_disp": round(disp, 4),
                                  "clean": False, "gi": gi,
                                  "diversified": bool(self.diversify and attempt == 0)}
            self.mark("reopen", f"{CUBE}:{tag}:r{attempt}")
            self.outcomes[f"{CUBE}:{tag}:r{attempt}"] = {"outcome": "recovery"}
            self.hold(10, +1)
            # Retreat home before the retry: planning from the post-lift hover over the
            # cube intermittently refuses with "Start state in collision" (measured: ep7
            # of the 12-ep A/B burned both remaining attempts on it). Same move the
            # lost-transport recovery already makes.
            self.sync_world()
            hres = self.expert.plan_home(self.q6())
            if hres is not None and bool(hres.success.any().item()):
                self.run_traj(traj_to_qs(hres), +1)
        else:
            return False

        # attach + transport (the cube must be in the scene for attach_from_scene)
        self.sync_world(held=None)
        try:
            from curobo.types import Pose

            # identity world_objects_pose_offset is REQUIRED — see run_expert_v1 (sphere
            # fit returns WORLD-frame centers; without it phantom spheres land ~30 cm off)
            self.expert.planner.attachment_manager.attach_from_scene(
                self.expert.js(self.q6()), [CUBE],
                world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0]))
            attached = True
        except Exception as e:  # noqa: BLE001
            print("[attach] failed:", e)
            attached = False
        b = self.box_xy()
        tgt = (float(b[0]), float(b[1]))
        cid = f"{CUBE}:{tag}:carry"
        self.mark("transport", cid)
        placed_via = None
        pres = self.expert.plan_place(self.q6(), tgt)
        if pres is not None and bool(pres.success.any().item()):
            placed_via = "pose"
        else:
            print(f"    [{CUBE}] place rung1 {'IK-none' if pres is None else 'trajopt-fail'}; retry high")
            pres = self.expert.plan_place(self.q6(), tgt, heights=PLACE_HEIGHTS_HIGH)
            if pres is not None and bool(pres.success.any().item()):
                placed_via = "pose-high"
        if placed_via:
            self.run_traj(traj_to_qs(pres), -1)
            dres = self.expert.plan_place(self.q6(), tgt, heights=(DROP_HEIGHT,))
            if dres is not None and bool(dres.success.any().item()):
                self.run_traj(traj_to_qs(dres), -1)
        else:
            self.place_fail_dumps.append({
                "q6": self.q6(), "tgt": list(tgt), "box_yaw": self.box_yaw(),
                "objs": {n: self.obj_pos(n).tolist() for n in (CUBE, *CLUTTER)},
            })
            q_over, q_lower, r_err = self.expert.carry_configs(tgt)
            cres = self.expert.plan_to_config(self.q6(), q_over)
            if cres is not None and bool(cres.success.any().item()):
                placed_via = f"carry(r_err={r_err:.3f})"
                self.run_traj(traj_to_qs(cres), -1)
            else:
                metrics["place_plan_fail"] += 1
                placed_via = f"carry-direct(r_err={r_err:.3f})"
                q_now = np.array(self.q6())
                for al in np.linspace(0, 1, 60):
                    self.step_action(((1 - al) * q_now + al * np.array(q_over)).tolist(), -1)
            hop = self.expert.plan_place(self.q6(), tgt, heights=(DROP_HEIGHT,))
            if hop is not None and bool(hop.success.any().item()):
                placed_via += "+hop"
                self.run_traj(traj_to_qs(hop), -1)
            else:
                q_now = np.array(self.q6())
                for al in np.linspace(0, 1, 30):
                    self.step_action(((1 - al) * q_now + al * np.array(q_lower)).tolist(), -1)
        if self.obj_pos(CUBE)[2] < 0.04 and not self.placed():
            print(f"    [{CUBE}] LOST GRIP before release (cube z={self.obj_pos(CUBE)[2]:.3f})")
            self.outcomes[cid] = {"outcome": "lost", "via": placed_via}
            if attached:
                self.expert.planner.attachment_manager.detach()
            return "lost"
        self.outcomes[cid] = {"outcome": "delivered", "via": placed_via}
        held_pos = self.obj_pos(CUBE)
        self.mark("release", cid)
        self.hold(12, +1)
        if attached:
            self.expert.planner.attachment_manager.detach()
        self.hold(8, +1)
        final = self.obj_pos(CUBE)
        print(f"    [{CUBE}] placed_via={placed_via} pre-release={held_pos.round(3).tolist()} "
              f"post-release={final.round(3).tolist()} box={b.round(3).tolist()} "
              f"yaw={self.box_yaw():.2f} placed={self.placed()}")
        self.mark("retreat", f"{CUBE}:{tag}")
        self.sync_world()
        hres = self.expert.plan_home(self.q6())
        if hres is not None and bool(hres.success.any().item()):
            self.run_traj(traj_to_qs(hres), +1)
        return self.placed()

    def episode(self, i):
        obs = self.env.reset()[0]
        self.aim_station()
        if self.h5_path:
            self.last_obs = obs["policy"][0].cpu().numpy().astype("float32")
            self.ep_obs = []
            self.ep_act = []
        if self.shards is not None:
            self.last_pol = obs["policy"][0].detach().cpu()
        self.q_default = self.robot.data.default_joint_pos[0, :6].clone()
        self.step_idx = 0
        self.segments = []
        self.outcomes = {}
        self.perturb_log = []
        self.pending_event = None
        self.grip_override = 0
        if self.record or self.shards is not None:
            # ⭐ the gaussian desk is not resident until real STEPS have run (HANDOFF
            # §4.9) — pre-roll at home with the frames discarded, or the film opens on
            # the arm floating over a bare grid
            rec, self.record = self.record, False
            self.hold(120, +1)
            self.record = rec
        if self.dagger is not None:
            self.dagger_drive()
        if self.shards is not None:
            self.sh = {"wrist": [], "station": [], "obs": [], "act": []}
            self.shard_rec = True
        if self.perturb:
            choices = [("nudge", "approach", CUBE), ("nudge", "lift", CLUTTER[0]),
                       ("slip", "transport", CUBE), ("nudge", "transport", CLUTTER[0])]
            typ, phase, target = choices[int(self.rng.integers(len(choices)))]
            self.pending_event = {"type": typ, "phase": phase, "target": target, "fire_at": None}
        self.mark("settle")
        self.hold(10, +1)
        m = {"plans": 0, "plan_fail": 0, "grasps": 0, "failed_grasps": 0,
             "clean_grasps": 0, "close_disp": [], "place_plan_fail": 0}
        r = self.fetch_cube(m, tag="f0")
        if r == "lost":   # dropped mid-transport: recover by refetching from live state
            self.mark("retreat", f"{CUBE}:recover")
            hres = self.expert.plan_home(self.q6())
            if hres is not None and bool(hres.success.any().item()):
                self.run_traj(traj_to_qs(hres), +1)
            self.fetch_cube(m, tag="f1")
        m["success"] = self.placed()
        m["steps"] = self.step_idx
        m["segments"] = self.segments
        m["outcomes"] = self.outcomes
        m["perturb_steps"] = self.perturb_log
        m["episode_kind"] = "recovery_scripted" if self.perturb_log else "nominal"
        print(f"[ep {i}] success={m['success']} grasps={m['grasps']} "
              f"failed={m['failed_grasps']} clean={m['clean_grasps']} "
              f"plan_fail={m['plan_fail']} close_disp={m['close_disp']}")
        if self.h5_path:
            self.demos.append({"obs": np.stack(self.ep_obs), "act": np.stack(self.ep_act), "meta": m})
        if self.shards is not None:
            self.shard_rec = False
            if m["success"] or args_cli.shard_include_failures:
                os.makedirs(self.shards, exist_ok=True)
                obs41 = torch.stack(self.sh["obs"])
                # proprio is BYTE-DERIVED from the observation slices the student may see;
                # obs41[16:34] (cube/box/clutter/placed) stays TEACHER-ONLY (same contract
                # as collect_demos.py's shard writer)
                proprio = torch.cat([obs41[:, 0:16], obs41[:, 34:41]], dim=1)
                assert proprio.shape[1] == 23, proprio.shape
                wrist = torch.stack(self.sh["wrist"])
                station = torch.stack(self.sh["station"])
                assert wrist.shape[1:] == station.shape[1:] == (120, 160, 3), \
                    (wrist.shape, station.shape)
                torch.save({
                    "wrist_rgb": wrist,
                    "workspace_rgb": station,
                    "proprio": proprio,
                    "obs41": obs41,                    # TEACHER-ONLY
                    "actions": torch.stack(self.sh["act"]),
                    "success": bool(m["success"]),
                    "attempts": int(m["grasps"] + m["failed_grasps"]),
                    "seed": int(args_cli.seed),
                    "env_index": int(i),
                }, os.path.join(self.shards, f"ep_{args_cli.seed}_{i:03d}.pt"))
                print(f"[shards] ep {i}: wrote {obs41.shape[0]} steps "
                      f"(success={m['success']}) -> {self.shards}", flush=True)
            else:
                print(f"[shards] ep {i}: FAILED episode, shard skipped", flush=True)
            self.sh = None
        return m


def main():
    drv = Driver()
    all_m = []
    for i in range(args_cli.episodes):
        drv.record = args_cli.video_all or (args_cli.video and i == args_cli.video_ep)
        all_m.append(drv.episode(i))
        if drv.record and any(drv.frames.values()):
            import imageio

            tag = f"ep{i}_{'ok' if all_m[-1]['success'] else 'fail'}" if args_cli.video_all \
                else f"ep{args_cli.video_ep}"
            for key, frames in drv.frames.items():
                if not frames:
                    continue
                path = os.path.join(HERE, f"expert_ws_{tag}_{key}.mp4")
                w = imageio.get_writer(path, fps=50, macro_block_size=1)
                for f in frames:
                    w.append_data(f.astype(np.uint8))
                w.close()
                print(f"[video] {path} ({len(frames)} frames) — for Big Will")
            drv.frames = {"station": [], "wrist": []}
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
    print("SUMMARY:", json.dumps(summary, indent=2))
    with open(os.path.join(HERE, "expert_ws_results.json"), "w") as f:
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
                g.create_dataset("train_mask",
                                 data=build_train_mask(T, d["meta"]["segments"], d["meta"]["outcomes"]))
                g.attrs["success"] = bool(d["meta"]["success"])
                g.attrs["num_samples"] = T
                g.attrs["episode_kind"] = d["meta"].get("episode_kind", "nominal")
                g.attrs["segments"] = json.dumps(d["meta"]["segments"])
                g.attrs["outcomes"] = json.dumps(d["meta"]["outcomes"])
            f["data"].attrs["total"] = len(drv.demos)
        print(f"[h5] wrote {len(drv.demos)} demos -> {drv.h5_path}")
    if drv.place_fail_dumps:
        torch.save(drv.place_fail_dumps, os.path.join(HERE, "place_fail_cases_ws.pt"))
        print(f"[dump] {len(drv.place_fail_dumps)} refused place cases -> place_fail_cases_ws.pt")
    drv.env.close()
    app.close()


if __name__ == "__main__":
    main()
