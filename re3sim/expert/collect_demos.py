# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Generate demonstrations for ``Rebot-Workstation-PickPlace1-v0`` with the scripted expert.

Design note: **plan serially, execute in parallel.**

``ArmKin`` uses the env count as its CEM population -- one batched
``write_joint_state_to_sim`` + ``sim.forward()`` evaluates every candidate at once. The
obvious way to use it is to plan for env 0 and drive every env with that same trajectory,
which throws away 127 of 128 envs because each env has a *different* object layout.

Instead: solve one plan per env (serial, the expensive part), stack the per-env waypoints into
``(n, 6)`` tensors, and execute all of them simultaneously -- ``ArmKin.act`` already accepts a
per-env joint target. That yields ``num_envs`` demonstrations per batch instead of one, and
the planning cost is paid once.

Plans differ in length between envs (the waypoint chains are densified to a fixed Cartesian
spacing, so a longer carry means more waypoints). Each segment is padded to the batch maximum
by repeating its final waypoint, which is a *hold* at the intended pose -- not a fabricated
motion.

Recording is causal: the observation is captured **before** the step that the action was
chosen from, never the one that resulted.

.. code-block:: bash

    python -u re3sim/expert/collect_demos.py --headless --num_envs 128 --batches 4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted-expert demo generation.")
parser.add_argument("--task", type=str, default="Rebot-Workstation-PickPlace1-v0")
parser.add_argument("--num_envs", type=int, default=128, help="demos per batch AND CEM population")
parser.add_argument("--batches", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="re3sim/expert/data/demos.hdf5")
parser.add_argument("--steps_per_wp", type=int, default=10)
parser.add_argument("--chatty", action="store_true", help="per-plan progress (--verbose is AppLauncher's)")
parser.add_argument("--teleport-pregrasp", action="store_true",
                    help="DIAGNOSTIC ONLY. Skip the action-driven home segment and teleport "
                         "straight to the pre-grasp. Demonstrations recorded this way begin "
                         "in a state `env.reset()` never produces and MUST NOT be trained on "
                         "-- this exists to isolate whether a failure is in the approach or "
                         "in everything after it.")
parser.add_argument("--record-video", type=str, default=None, metavar="DIR",
                    help="Film env 0 from the wrist camera and a workstation camera while the "
                         "expert executes. Recorded through THIS executor rather than a "
                         "re-implementation, so the film is of the manoeuvre that actually "
                         "produces the demonstrations.")
parser.add_argument("--record-stride", type=int, default=2, help="record every Nth env step")
parser.add_argument("--record-fps", type=int, default=25)
parser.add_argument("--record-width", type=int, default=960)
parser.add_argument("--record-height", type=int, default=540)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.record_video:
    # must be set BEFORE AppLauncher starts Kit, or the render products never exist
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
import time

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.re3sim import mdp

if args_cli.record_video:
    import imageio.v2 as imageio  # noqa: E402
    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.sensors import CameraCfg  # noqa: E402
    # The user's validated wrist mount: D405 at 84 deg HFOV, `tilt_x_m30`, optical axis 1.8
    # deg off the TCP. Re-deriving one here would film a camera that is not on the rig.
    from reBot_RL.tasks.manager_based.lift.camera_cfg import WRIST_CAM_CFG  # noqa: E402
    # In FRONT of the arm, per the 2026-08-09 directive. Imported from the env cfg rather than
    # written here: `scripts/render_workstation.py` inspects the same view, and the two used
    # to carry their own copies of the pose.
    from reBot_RL.tasks.manager_based.re3sim.workstation_env_cfg import (  # noqa: E402
        STATION_CAM_EYE, STATION_CAM_TARGET,
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "probes"))
sys.path.insert(0, _HERE)
from _kin import ArmKin, Q_CLOSE, Q_OPEN  # noqa: E402
from workstation_expert import (  # noqa: E402
    CARRY_Z, grasp_choices, plan_episode,
)

#: Integral gain and clamp for the steady-state bias compensator. 0.15 is slow enough that
#: ordinary tracking lag does not wind it up over the ~10 steps a waypoint gets, and 0.25 rad
#: is well inside the +/-0.5 rad the action encoding spans before |a| exceeds 1.
BIAS_GAIN = float(os.environ.get('BIAS_GAIN', '0.15'))
BIAS_MAX = float(os.environ.get('BIAS_MAX', '0.0'))
#: env steps held at the grasp pose, fingers still OPEN, before the close. Swept like
#: `GRIP_Z`; see the settle block in `main` for why it exists at all.
GRASP_SETTLE = int(os.environ.get('GRASP_SETTLE', '40'))
#: In-sim close screen -- see `screen` in `main`. 0 disables it entirely and reproduces the
#: shipped planner exactly, which is what every number before 2026-08-06 was measured with.
#: Each extra round re-plans only the envs whose grasp did not lift, so the cost is one close
#: (batched, ~cheap) plus a CEM solve for the shrinking set of failures.
SCREEN_ROUNDS = int(os.environ.get('SCREEN_ROUNDS', '4'))
#: physics steps: settle open at the grasp pose, then close. 560 close steps is what the
#: clutter expert uses and matches this expert's own 70 env steps x decimation 8.
SCREEN_SETTLE = int(os.environ.get('SCREEN_SETTLE', '160'))
SCREEN_CLOSE = int(os.environ.get('SCREEN_CLOSE', '560'))
#: control steps of lift, and the rise that counts as "it actually picked it up" [m]. Scoring
#: on the LIFT rather than on gap-vs-width needs no object-width constant and is the predicate
#: that actually matters. clutter's `_screen` scored `held & ~toppled` for its own reasons.
SCREEN_LIFT = int(os.environ.get('SCREEN_LIFT', '25'))
#: ⭐ Screen through the REAL DESCENT rather than by teleporting onto the grasp pose.
#:
#: The teleport screen answers "would this grasp hold if the arm were already there", and the
#: run never is. MEASURED 2026-08-09, 32 envs: the teleport screen certified 30/32 candidates
#: and only 20/32 gripped in the episode, with **10 of the 12 failures closing on AIR** -- a
#: finger gap under 20 mm, i.e. nothing between the jaws at all. That is not a grasp-quality
#: failure, it is an arrival failure, and a screen that starts at the destination is blind to
#: it by construction.
#:
#: So start at the transit pose and replay the descent at the executor's own waypoint spacing
#: and step count before closing. Roughly doubles the cost of a screen slot and makes it a
#: measurement of the manoeuvre that actually runs.
SCREEN_DESCEND = os.environ.get('SCREEN_DESCEND', '1') == '1'
SCREEN_RISE = float(os.environ.get('SCREEN_RISE', '0.030'))
#: ⭐ How many members of the grasp GOALSET to solve per env, and how many times the executor
#: may fall back to the next one after a grasp fails.
#:
#: This is `run_expert_v1`'s architecture, which is what the 2026-08-09 directive asks for:
#: plan a *set* of candidate grasps, then check after the close whether the object actually
#: came up and RE-GRASP with a different member if it did not. The old executor ran its single
#: plan blind from home to release and checked nothing, which is why its entire loss sat in
#: one bucket -- MEASURED at 64 envs: `never-got-there 17 | lifted-but-lost 0 |
#: over-box-not-inside 0`. Nothing after the grasp has ever failed in any batch, so a retry at
#: the grasp is the only recovery worth building.
#:
#: ⭐ THE DEFAULTS ARE THE VERIFIED CONFIGURATION. `--num_envs 128` with no environment
#: variables at all reproduces the measured 94-96 % expert; a default that differs from the
#: configuration a result was measured under is a trap, and this file has been bitten by
#: exactly that before (`SCREEN_ROUNDS` used to default to 0, i.e. no screen, while every
#: quoted number was measured with one).
#:
#: `GOALSET=1 RETRIES=0 SCREEN_ROUNDS=0 BIAS_MAX=0.25` reproduces the pre-2026-08-09 executor.
GOALSET = int(os.environ.get('GOALSET', '4'))
RETRIES = int(os.environ.get('RETRIES', '2'))
#: Cube rise that counts as "the grasp worked", measured from just before the close [m]. Same
#: predicate as the screen's, and deliberately NOT finger gap: a gap the width of the cube is
#: also what a finger resting against it without gripping reads.
HOLD_RISE = float(os.environ.get('HOLD_RISE', '0.030'))
#: Env steps held at the transit pose before the descent begins. The screen and the executor
#: MUST agree on this or the screen is flying a different manoeuvre; they used to carry
#: separate literals, which is the shape of every drift bug in this file's history.
TRANSIT_SETTLE = int(os.environ.get('TRANSIT_SETTLE', '40'))

#: phase labels, recorded per step so a later analysis can slice by segment
PH_APPROACH, PH_CLOSE, PH_LIFT, PH_CARRY, PH_RELEASE, PH_RETREAT = range(6)


def pad(segs: list[list[torch.Tensor]], dev) -> torch.Tensor:
    """Stack per-env waypoint lists into (T, n, 6), padding short ones by holding the last."""
    n = len(segs)
    T = max(len(s) for s in segs)
    out = torch.zeros(T, n, 6, device=dev)
    for i, s in enumerate(segs):
        for t in range(T):
            out[t, i] = s[min(t, len(s) - 1)]
    return out


def main() -> None:
    torch.manual_seed(args_cli.seed)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.episode_length_s = 1.0e5          # the expert owns episode boundaries, not the MDP
    env_cfg.seed = args_cli.seed
    if args_cli.record_video:
        env_cfg.scene.wrist_cam = WRIST_CAM_CFG.replace(
            width=args_cli.record_width, height=args_cli.record_height,
            data_types=["rgb"], update_period=0.0)
        env_cfg.scene.station_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/StationCam", update_period=0.0,
            width=args_cli.record_width, height=args_cli.record_height, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=17.0, horizontal_aperture=20.955,
                                             clipping_range=(0.02, 20.0)))
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev, n = e.device, e.num_envs
    kin = ArmKin(env)

    rec_frames = {"wrist": [], "station": []}
    if args_cli.record_video:
        wrist_cam, station_cam = e.scene["wrist_cam"], e.scene["station_cam"]
        origin0 = e.scene.env_origins[0]

        # The splats are spawned ONCE at `/World/Splats` (see workstation_env_cfg.py) on the
        # assumption that env 0 stands at the world origin. That holds only for num_envs == 1:
        # Isaac Lab's grid cloner CENTRES the env grid on the origin, so with 8 envs the
        # backdrop sits between envs and the filmed env has no desk under it at all -- the arm
        # and objects float on the bare ground plane. Move it onto whichever env we are
        # filming. Appearance-only, so this changes nothing the MDP can see.
        if float(origin0.abs().max()) > 1e-6:
            import omni.usd  # noqa: PLC0415
            from pxr import Gf, UsdGeom  # noqa: PLC0415
            prim = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Splats")
            if prim and prim.IsValid():
                # Re-use an existing translate op if the spawned USD already carries one;
                # AddTranslateOp on a prim that has a transform stack appends a SECOND op and
                # the two then compose instead of replacing.
                xf = UsdGeom.Xformable(prim)
                op = next((o for o in xf.GetOrderedXformOps()
                           if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
                (op or xf.AddTranslateOp()).Set(
                    Gf.Vec3d(*[float(v) for v in origin0.cpu().numpy()]))
                print(f"[record] moved /World/Splats onto env 0 at "
                      f"{np.round(origin0.cpu().numpy(), 3)}", flush=True)
            else:
                print("[record] WARNING: no /World/Splats prim -- filming without the desk",
                      flush=True)

        def aim_station():
            # re-aimed after every reset: `reset_scene_to_default` restores prim poses, and a
            # camera left at the default pose films the floor.
            station_cam.set_world_poses_from_view(
                (torch.tensor([list(STATION_CAM_EYE)], device=e.device) + origin0),
                (torch.tensor([list(STATION_CAM_TARGET)], device=e.device) + origin0))

        def grab():
            e.sim.render()
            wrist_cam.update(dt=0.0)
            station_cam.update(dt=0.0)
            rec_frames["wrist"].append(
                wrist_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            rec_frames["station"].append(
                station_cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))

    all_obs, all_act, all_ph, all_ok = [], [], [], []
    all_seed, all_planned, all_msk, all_att = [], [], [], []
    t_start = time.time()

    for b in range(args_cli.batches):
        # One spawn seed per batch, recorded per demo. Evaluation MUST use seeds outside this
        # range: a policy scored on the layouts it was trained on measures memorisation.
        batch_seed = args_cli.seed * 1000 + b
        obs_d, _ = env.reset(seed=batch_seed)
        cube = mdp.object_pos_local(e, mdp.TARGET_NAME).clone()
        cube_yaw = mdp.yaw_of(mdp.object_quat(e, mdp.TARGET_NAME)).clone()
        boxc = mdp.box_centers_local(e).clone()
        clut = {c: mdp.object_pos_local(e, c).clone() for c in mdp.CLUTTER_NAMES}
        # ⭐ The SPAWN orientations, captured here and restored by `restore_objects`. The cube's
        # yaw is what `plan_episode` solved the opening axis against, so it is part of the
        # problem statement, not part of the state a screen may leave behind.
        quat0 = {nm: e.scene[nm].data.root_state_w.torch[:, 3:7].clone()
                 for nm in mdp.OBJECT_NAMES}

        # ---------------------------------------------------------------- plan, serially
        def plan_i(i, choices=None):
            obstacles = [
                (clut["rolloftape"][i], mdp.TAPE_DIAMETER / 2, mdp.TAPE_HEIGHT),
                (clut["tapemeasure"][i], mdp.TAPEMEASURE_LONG / 2, mdp.TAPEMEASURE_HEIGHT),
            ]
            return plan_episode(kin, cube[i, :2], float(cube_yaw[i]), boxc[i], obstacles,
                                verbose=args_cli.chatty, choices=choices)

        def plan_goalset(i):
            """Up to ``GOALSET`` COMPLETE trajectories for env i, one per goalset member.

            Each is solved and audited independently -- own descent chain, own transit, own
            lift and carry -- because the executor may have to run any of them from the reset
            pose. Members come from ``grasp_choices()``: both opening axes at the measured
            grip height, then both at the runner-up height. A member whose gate fails is
            skipped rather than substituted, so a short goalset is an honest report that this
            layout has few reachable grasps.
            """
            out = []
            for c in grasp_choices():
                if len(out) >= GOALSET:
                    break
                p = plan_i(i, [c])
                if p is not None:
                    out.append(p)
            return out

        def restore_objects():
            """Put every body back where the plans were solved against."""
            for name, p0 in [(mdp.TARGET_NAME, cube)] + [(c, clut[c]) for c in mdp.CLUTTER_NAMES]:
                obj = e.scene[name]
                st = obj.data.root_state_w.torch.clone()
                st[:, :3] = p0 + e.scene.env_origins
                # ⭐ ORIENTATION TOO, from the SPAWN capture. This used to restore position and
                # velocity only, on the reasoning that the cube's random spawn yaw is what the
                # plan was solved against and must not be overwritten -- which had the logic
                # exactly backwards. `st[:, 3:7]` is the CURRENT quaternion, and a screen's
                # close rotates the cube; leaving it alone therefore preserved the ROTATED
                # cube, so every screen after the first, and the episode itself, ran against a
                # cube whose yaw no longer matched the opening axis every plan was solved for.
                # Restoring the captured spawn quaternion is what "the cube the plan was made
                # for" actually means.
                st[:, 3:7] = quat0[name]
                st[:, 7:] = 0.0
                obj.write_root_state_to_sim(st)
            e.sim.forward()
            e.scene.update(e.physics_dt)

        def screen(cands):
            """Fly each candidate's descent IN THE SIMULATOR, close, and lift. Returns the RISE.

            Ported from `clutter/expert/clutter_expert.py::_screen`, which took that task to
            94 %. Its finding, re-derived independently here (03_EXPERT_AND_BC.md 6.3): NO
            forward-kinematic statistic predicts whether a grasp works. `pos_err`, `o_align`
            and keep-out penetration are near-identical across a candidate that grips and one
            that closes on air, because `cem` scores poses through `write_joint_state_to_sim`
            -- kinematically, with no gravity and no contact. So stop ranking them and test
            them.

            Batched: every env screens ITS OWN candidate in the same physics steps, so a round
            costs one close regardless of `n`. Bypasses the MDP via `hold_phys`/`run_phys`, so
            no termination or reset event fires mid-screen.
            """
            ref = next((p for p in cands if p is not None), None)
            if ref is None:
                return [float("-inf")] * n
            sub = [p if p is not None else ref for p in cands]
            q_grip = torch.stack([p["grasp"][-1] for p in sub])
            q_top = torch.stack([p["lift"][-1] for p in sub])
            restore_objects()
            if SCREEN_DESCEND:
                # Start where the episode starts its descent -- the end of the transit -- and
                # fly the same waypoints, at the same 10 control steps each, that the executor
                # will. `run_phys`'s `substeps=8` mirrors the env's decimation, so speeds match
                # what a policy could command.
                q0 = torch.stack([p["home"][-1] for p in sub])
                kin.teleport_arm(q0, q_fing=Q_OPEN)
                kin.hold_phys(q0, 8 * TRANSIT_SETTLE, q_fing=Q_OPEN)
                Wd = pad([p["grasp"] for p in sub], dev)
                q_prev = q0
                for t in range(Wd.shape[0]):
                    kin.run_phys(q_prev, Wd[t], args_cli.steps_per_wp, q_fing=Q_OPEN)
                    q_prev = Wd[t]
                kin.hold_phys(q_grip, 8 * GRASP_SETTLE, q_fing=Q_OPEN)
            else:
                kin.teleport_arm(q_grip, q_fing=Q_OPEN)
                # settle OPEN first: the teleport places the arm kinematically, and the whole
                # point is to find out where the drive actually holds it once gravity is in
                # play.
                kin.hold_phys(q_grip, SCREEN_SETTLE, q_fing=Q_OPEN)
            kin.hold_phys(q_grip, SCREEN_CLOSE, q_fing=Q_CLOSE)
            z0 = mdp.object_pos_local(e, mdp.TARGET_NAME)[:, 2].clone()
            kin.run_phys(q_grip, q_top, SCREEN_LIFT, q_fing=Q_CLOSE)
            rose = mdp.object_pos_local(e, mdp.TARGET_NAME)[:, 2] - z0
            restore_objects()
            # ⭐ Return the RISE, not a pass/fail. The screen lifts to `LIFT_Z`, so a properly
            # gripped cube comes up ~120 mm and a marginal one -- pinched on a corner, or
            # dragged up by one finger -- comes up 30-50 mm. Thresholding that to a bool threw
            # away the only graded quality signal in the whole pipeline, and the grasps that
            # fail in the episode after passing the screen are exactly the marginal ones.
            # `-inf` for a slot this env does not have, so it sorts last.
            return [float(rose[i]) if cands[i] is not None else float("-inf")
                    for i in range(n)]

        # ------------------------------------------------------------ plan the GOALSET
        t0 = time.time()
        n_lift = [0] * n          # screened-lifting candidates per env
        best = [[] for _ in range(n)]     # each env's candidate rises, best first
        goals = [plan_goalset(i) for i in range(n)]
        n_ok = sum(1 for g in goals if g)
        print(f"[batch {b}] goalset: {n_ok}/{n} envs solved, "
              f"{sum(len(g) for g in goals) / max(1, n_ok):.2f} candidates each "
              f"({time.time() - t0:.0f}s)", flush=True)
        if n_ok == 0:
            print("[batch] nothing solved -- skipping")
            continue

        # ------------------------------------------------- order the goalset by the close
        # ORDERING, not filtering, and that is the change. The screen used to REPLACE a
        # candidate that did not lift; now the loser stays in the goalset behind the winner,
        # because the screen is a good ranking signal and a poor certificate: it teleports to
        # the grasp while the real run DESCENDS 140 mm onto it. MEASURED 2026-08-09 with the
        # authored cube -- 63/64 candidates certified, 47/64 episodes succeeded. Those 16
        # episodes are exactly what the executor's retry is for.
        if SCREEN_ROUNDS > 0:
            depth = max(len(g) for g in goals)
            lifts = []
            for slot in range(depth):
                lifts.append(screen([g[slot] if slot < len(g) else None for g in goals]))
                r = [v for v in lifts[slot] if v > float("-inf")]
                up = [v for v in r if v > SCREEN_RISE]
                print(f"[batch {b}] screen slot {slot}: {len(up)}/{n} lift"
                      + (f"   median rise {sorted(up)[len(up) // 2] * 1000:.0f} mm"
                         if up else ""), flush=True)
            live = [False] * n
            for i in range(n):
                rise = [lifts[s][i] for s in range(len(goals[i]))]
                # Best rise first. The executor takes members in order, so this is what makes
                # it attempt the most convincingly-gripped candidate rather than merely a
                # passing one.
                order = sorted(range(len(rise)), key=lambda k: -rise[k])
                goals[i] = [goals[i][k] for k in order]
                best[i] = [rise[k] for k in order]
                n_lift[i] = sum(1 for r in rise if r > SCREEN_RISE)
                live[i] = n_lift[i] > 0
            top = sorted(r[0] for r in best if r and r[0] > float("-inf"))
            print(f"[batch {b}] screened: {sum(live)}/{n} envs have >=1 lifting candidate; "
                  f"best-candidate rise median {top[len(top) // 2] * 1000:.0f} mm, "
                  f"p10 {top[len(top) // 10] * 1000:.0f} mm "
                  f"({time.time() - t0:.0f}s)", flush=True)

            # ⭐ RE-DRAW the envs no member of whose goalset lifts. This is the ceiling that
            # matters: with a 3-member goalset, MEASURED at 32 envs, 27/32 envs had a lifting
            # candidate and 26/32 episodes grasped -- i.e. the executor was already converting
            # nearly every env the PLANNER could serve, and the remainder was a planning
            # shortfall, not an execution one. Retrying an env with a candidate that has been screened and does not
            # lift is spending steps on a known loser.
            #
            # `_solve_grasp`'s CEM is stochastic, so re-planning the same member is a
            # genuinely different draw, and that variance is large: the clutter task measured
            # 32.8 % -> 74.2 % between two draws of the SAME configuration. A winner goes to
            # the FRONT of the goalset, since it is the only member known to work.
            for r in range(1, SCREEN_ROUNDS):
                todo_i = [i for i in range(n) if not live[i]]
                if not todo_i:
                    break
                cand = [None] * n
                for i in todo_i:
                    cand[i] = plan_i(i, [grasp_choices()[r % len(grasp_choices())]])
                h = screen(cand)
                for i in todo_i:
                    if cand[i] is not None and h[i] > SCREEN_RISE:
                        goals[i] = [cand[i]] + goals[i]
                        best[i] = [h[i]] + best[i]
                        live[i] = True
                        n_lift[i] += 1
                print(f"[batch {b}] re-draw {r}: {sum(live)}/{n} envs now have a lifting "
                      f"candidate (re-drew {len(todo_i)}, {time.time() - t0:.0f}s)", flush=True)

        planned = [bool(g) for g in goals]
        # Screened-lifting candidates sit at the FRONT of each goalset, so this count is also
        # the index at which an env runs out of members worth attempting.
        # With the screen OFF there is no information about which member lifts, so every
        # solved member is worth attempting; with it on, the lifting ones sit at the front and
        # this count is the index at which an env runs out of members worth trying.
        n_lift_t = torch.tensor(
            [max(1, n_lift[i]) if SCREEN_ROUNDS > 0 else len(goals[i]) for i in range(n)],
            device=dev)
        # An env with no solved candidate still has to be driven with something legal, so it
        # borrows another env's goalset and is excluded from the dataset by `train_mask`.
        # Substituting an arbitrary trajectory would drive the arm through THIS env's objects,
        # so it is not a free choice -- it is a hold at somebody else's poses.
        ref = next(g for g in goals if g)
        goals = [g if g else ref for g in goals]

        # ------------------------------------------------------------------- execute, batched
        rec_obs, rec_act, rec_ph, rec_msk = [], [], [], []
        last_obs = None

        # --- steady-state bias compensation -------------------------------------------------
        # MEASURED 2026-08-06: teleported exactly onto a CEM-solved pre-grasp, the arm settles
        # 19.8 mm away from it (p90 117 mm) -- while at carry height, 175 mm up, it holds the
        # same kind of pose to 1.5 mm. That asymmetry is the signature of a POSITION DRIVE
        # HOLDING AGAINST GRAVITY: torque is stiffness x error, so a pose that needs torque can
        # only be held WITH an error. A low, far reach over the table needs the most torque and
        # therefore sags the most.
        #
        # `cem` cannot see this at all. It evaluates candidates with `write_joint_state_to_sim`,
        # which places the arm kinematically -- no gravity, no contact. Every pose it returns is
        # reachable in that sense and some of them are simply not HOLDABLE.
        #
        # The fix is the standard one: command past the target by the steady-state error, so the
        # drive's own error lands the arm on the target. An integrator finds that offset without
        # anyone having to model the arm. Clamped for anti-windup, since `q - q_now` also
        # contains ordinary tracking lag while a waypoint is being traversed.
        #
        # The recorded action is the COMPENSATED one, which is correct: it is what was actually
        # submitted to `env.step`, and it is what a policy must emit to hold the same pose.
        bias = torch.zeros(n, 6, device=dev)
        _rec_t = [0]
        # Saturation is the difference between "the compensator did not help" and "the
        # compensator was never allowed to". Report it rather than infer it.
        _bias_peak = [0.0]

        _all = torch.ones(n, dtype=torch.bool, device=dev)

        def act_of(q, close):
            """The env's 7-D action, with a PER-ENV gripper command.

            ``ArmKin.act`` takes one bool for the whole batch, which was fine while every env
            ran the same script. It no longer is: while one env retries a grasp with its
            fingers OPEN, another is holding the cube it already picked up and must keep them
            SHUT. ``a[:, 6]`` was always per-env -- ``BinaryJointPositionActionCfg`` reads its
            sign row by row -- so this needs no change to the action space, and the recorded
            demonstrations stay exactly as emittable by a policy as before.
            """
            a = torch.zeros((n, 7), device=dev)
            a[:, :6] = (q - kin.q_arm0.unsqueeze(0)) / 0.5
            a[:, 6] = torch.where(close, -1.0, 1.0) if torch.is_tensor(close) else (
                -1.0 if close else 1.0)
            return a

        def step(q, close, phase, mask=None):
            nonlocal last_obs, bias
            a = act_of(q + bias, close)
            rec_obs.append(last_obs.clone())
            rec_act.append(a.clone())
            rec_ph.append(phase)
            # Per-STEP, per-env: an env that has already got the cube holds still while the
            # others retry, and those idle steps are masked out rather than taught as
            # "wait here". Without this the retry machinery would poison every successful
            # demonstration it shares a batch with.
            rec_msk.append((_all if mask is None else mask).clone())
            o, _, _, _, _ = env.step(a)
            last_obs = o["policy"]
            if args_cli.record_video:
                _rec_t[0] += 1
                if _rec_t[0] % args_cli.record_stride == 0:
                    grab()
            q_now = kin.robot.data.joint_pos.torch()[:, kin.arm_dof] \
                if callable(getattr(kin.robot.data.joint_pos, "torch", None)) \
                else kin.robot.data.joint_pos[:, kin.arm_dof]
            bias = (bias + BIAS_GAIN * (q - q_now)).clamp(-BIAS_MAX, BIAS_MAX)
            _bias_peak[0] = max(_bias_peak[0], float(bias.abs().max()))

        # Put the arm back at the RESET pose, not at the pre-grasp. Planning teleported it all
        # over the workspace through `cem`/`refine`, so it has to be restored -- but restoring
        # it to `seg_pre` would mean every recorded episode starts from a state `env.reset()`
        # never produces, and the policy would meet an unseen observation at step 0 of every
        # evaluation. The plan's `home` segment drives that approach with actions instead.
        q_home = kin.q_arm0.unsqueeze(0).repeat(n, 1)
        kin.teleport_arm(q_home, q_fing=Q_OPEN)
        # those same teleports swept the arm through the objects; PhysX would resolve the
        # overlap on the first real step, so restore the layout the plans were solved against
        for name, p0 in [(mdp.TARGET_NAME, cube)] + [(c, clut[c]) for c in mdp.CLUTTER_NAMES]:
            obj = e.scene[name]
            st = obj.data.root_state_w.torch.clone()
            st[:, :3] = p0 + e.scene.env_origins
            st[:, 3:7] = quat0[name]
            st[:, 7:] = 0.0
            obj.write_root_state_to_sim(st)
        e.sim.forward()
        e.scene.update(e.physics_dt)
        if args_cli.record_video:
            aim_station()
        last_obs = e.observation_manager.compute()["policy"]

        ok_plan = torch.tensor(planned, device=dev)

        _tgt = [None]

        def retarget(sel):
            """Cartesian targets for the diagnostics, for the candidate each env is running.

            Recomputed per attempt, because a retry runs a DIFFERENT member of the goalset and
            comparing the arm against the previous member's targets would report a huge error
            for a manoeuvre that is going exactly where it was sent. They cannot be derived
            here from the arm: `kin.fk` writes joint state to the sim to read geometry back,
            so calling it mid-execution would teleport the arm and destroy the run.
            """
            _tgt[0] = {k: torch.stack([goals[i][sel[i]]["tcp"][k] for i in range(n)])
                       for k in goals[0][0]["tcp"]}

        _live = [None]      # envs currently executing the leg being diagnosed

        def diag(tag, key):
            """Where is the arm actually, against where the plan said it should be?

            A failure taxonomy says which *leg* broke; this says whether the arm even got
            where the planner sent it. Without it, "never got there" is equally consistent
            with an unreachable plan, a saturating action and a tracking lag -- and those want
            three different fixes.
            """
            if not args_cli.chatty:
                return
            sel = ok_plan if _live[0] is None else (ok_plan & _live[0])
            if not bool(sel.any()):
                return
            err = (kin.tcp_now() - _tgt[0][key]).norm(dim=1)[sel] * 1000
            cz = mdp.object_pos_local(e, mdp.TARGET_NAME)[:, 2][sel] * 1000
            # Lowest arm-body origin. If the TCP is stuck far from its target AND this is at
            # the floor, the arm is pressed into the table rather than lagging behind a
            # command -- those look identical in a position-error number alone.
            bz = (kin.robot.data.body_pos_w[..., 2] - e.scene.env_origins[:, 2:3]).min(dim=1).values
            # SIGNED per-axis error, not just the norm. A norm says "35 mm off" and leaves the
            # fix ambiguous; the sign says whether the arm is sagging under the target (aim
            # high) or stopping short of it (aim far), and those are opposite corrections.
            d = (kin.tcp_now() - _tgt[0][key])[sel].median(dim=0).values * 1000
            print(f"    [bias peak {_bias_peak[0]:.3f} rad of {BIAS_MAX:.2f} max"
                  + ("  <<< SATURATED" if _bias_peak[0] >= 0.98 * BIAS_MAX else "") + "]")
            print(f"    [diag {tag:7s}] TCP err vs plan: median {err.median():7.1f} mm  "
                  f"p90 {err.quantile(0.9):7.1f} mm | signed dx/dy/dz "
                  f"{d[0]:+6.1f} {d[1]:+6.1f} {d[2]:+6.1f} mm | gap "
                  f"{float(kin.gap()[sel].median()) * 1000:5.1f} mm"
                  f" | lowest body {float(bz[sel].min()) * 1000:5.1f} mm"
                  f" | cube z {float(cz.median()):5.1f} mm", flush=True)

        # ---------------------------------------------------------- drive helpers, per env
        prev = q_home

        def pack(per_env, active):
            """(T, n, 6). Env i contributes its own waypoints if ``active[i]``, else holds.

            This is what keeps the batch in lockstep while the envs do different things: a
            finished env is given a one-waypoint "segment" at the pose it is already holding,
            which `pad` then repeats for as long as the busiest env needs.
            """
            return pad([per_env[i] if bool(active[i]) else [prev[i]] for i in range(n)], dev)

        def go1(q, close, phase, mask):
            nonlocal prev
            for t in range(args_cli.steps_per_wp):
                f = min(1.0, (t + 1) / max(1.0, args_cli.steps_per_wp * 0.8))
                step((1 - f) * prev + f * q, close, phase, mask)
            prev = q

        def go(W, close, phase, mask):
            for t in range(W.shape[0]):
                go1(W[t], close, phase, mask)

        def stay(steps, close, phase, mask):
            for _ in range(steps):
                step(prev, close, phase, mask)

        # ======================================================================= THE LOOP
        # ⭐ `run_expert_v1`'s per-object loop, ported without cuRobo:
        #
        #     approach -> descend -> close -> HOLD CHECK -> (regrasp with the next goalset
        #     member) -> carry -> release -> VERIFY PLACED
        #
        # What could NOT be ported is cuRobo's ability to plan a fresh collision-free path
        # from wherever the arm happens to be. `kin.cem` and `kin.fk` both teleport the arm to
        # read geometry back, so there is no mid-episode planning available at all. The retry
        # therefore RETRACES the exact path it came down -- lift, then descent, then transit,
        # each reversed -- back to the reset pose, and starts the next candidate from there.
        # Every leg of that is a path this env already audited for table clearance and
        # clutter, traversed backwards; nothing new is ever swept through the scene.
        cur = [0] * n          # goalset member each env is about to attempt
        ran = [0] * n          # goalset member the arm physically traversed last
        got = torch.zeros(n, dtype=torch.bool, device=dev)
        att = torch.zeros(n, dtype=torch.long, device=dev)

        for a_i in range(1 + RETRIES):
            # ⭐ Retry only into a candidate the screen says LIFTS. MEASURED at 32 envs: with
            # unrestricted retries, attempts 1 and 2 converted +1 and +0 while every env that
            # had already got the cube waited out two full retrace-and-descend cycles holding
            # it -- and 2 of 26 dropped it in that time. Attempting a member that has been
            # screened and does not lift is spending a known loser's steps on everybody.
            todo = ~got & (torch.tensor(cur, device=dev) < n_lift_t)
            if not bool(todo.any()):
                break
            att[todo] = a_i
            retarget(cur)
            _live[0] = todo
            if a_i > 0:
                back = [list(reversed(goals[i][ran[i]]["lift"]))
                        + list(reversed(goals[i][ran[i]]["grasp"]))
                        + list(reversed(goals[i][ran[i]]["home"])) for i in range(n)]
                # `close=got`: the envs that already have the cube keep holding it while the
                # others walk back with their fingers open.
                go(pack(back, todo), got, PH_RETREAT, todo)
                stay(10, got, PH_RETREAT, todo)

            if a_i == 0 and args_cli.teleport_pregrasp:
                pre = torch.stack([goals[i][0]["pre"] for i in range(n)])
                kin.teleport_arm(pre, q_fing=Q_OPEN)
                e.sim.forward()
                e.scene.update(e.physics_dt)
                last_obs = e.observation_manager.compute()["policy"]
                prev = pre
                stay(8, got, PH_APPROACH, todo)
            else:
                stay(8, got, PH_APPROACH, todo)
                go(pack([goals[i][cur[i]]["home"] for i in range(n)], todo),
                   got, PH_APPROACH, todo)
            # Settle at the transit pose before descending. The transit is the longest leg in
            # the plan and its waypoints get 10 steps each; if the arm is merely still moving
            # when the descent begins, every millimetre of that lag is carried into the grasp.
            stay(TRANSIT_SETTLE, got, PH_APPROACH, todo)
            diag("home", "high")
            diag("home", "high_fk")   # same instant, against the COMMANDED pose's own FK

            # Ramp the measured grasp bias in across the descent, reaching full at the grasp
            # pose -- and only on the envs that are actually descending, or it would shove a
            # holding env off the pose it is parked at.
            GB = torch.stack([goals[i][cur[i]]["bias"] for i in range(n)]) \
                * todo.unsqueeze(1).float()
            Wg = pack([goals[i][cur[i]]["grasp"] for i in range(n)], todo)
            for t in range(Wg.shape[0]):
                go1(Wg[t] + ((t + 1) / Wg.shape[0]) * GB, got, PH_APPROACH, todo)
            # Settle at the grasp pose before closing, for exactly the reason the transit
            # settles above -- and the omission here is why the two legs measured so
            # differently. MEASURED 2026-08-06, 64 envs: with a 40-step settle the transit
            # arrives 1.1 mm from its planned pose; the grasp descent, which had NO settle at
            # all, arrived 11.1 mm out (p90 60.0 mm) and the fingers then closed on a pose the
            # arm was still travelling toward.
            stay(GRASP_SETTLE, got, PH_APPROACH, todo)
            diag("grasp", "grip")

            z0 = mdp.object_pos_local(e, mdp.TARGET_NAME)[:, 2].clone()
            stay(70, _all, PH_CLOSE, todo)   # close. Every search already finished (A6)
            gap = kin.gap().clone()
            diag("closed", "grip")

            go(pack([goals[i][cur[i]]["lift"] for i in range(n)], todo), _all, PH_LIFT, todo)
            stay(10, _all, PH_LIFT, todo)

            # ⭐ THE HOLD CHECK. Did the cube actually come up? Scored on the object, not on
            # the fingers: a finger gap the width of the cube is also what a jaw resting
            # against it without gripping reads, and `_kin`'s own note says no forward
            # statistic predicts a grasp -- so measure the outcome instead of predicting it.
            rose = (mdp.object_pos_local(e, mdp.TARGET_NAME)[:, 2] - z0) > HOLD_RISE
            newly = rose & todo
            got = got | newly
            # Separate "closed on air" from "gripped it and lost it": the first says the arm
            # was in the wrong place, the second says the grip was too weak. They want
            # opposite fixes and a success rate cannot tell them apart.
            air = int(((~rose) & todo & (gap < 0.020)).sum())
            slip = int(((~rose) & todo & (gap >= 0.020)).sum())
            # Only for the envs that actually RAN this attempt. Advancing `ran` for an env
            # that sat this one out would claim the arm traversed a candidate it never
            # touched, and the carry -- which is solved from that candidate's own lift top --
            # would then be asked to start from a pose in a different IK branch.
            for i in range(n):
                if bool(todo[i]):
                    ran[i] = cur[i]
                    if not bool(got[i]):
                        cur[i] = min(cur[i] + 1, len(goals[i]) - 1)
            print(f"[batch {b}] attempt {a_i}: +{int(newly.sum())} held -> "
                  f"{int(got.sum())}/{n}   (failed: closed-on-air {air}, gripped-but-lost "
                  f"{slip})", flush=True)

        diag("lift", "top")
        # ------------------------------------------------ carry, release, verify, retreat
        # Each env carries with the candidate it actually ran: the carry chain is solved from
        # that candidate's own lift top, so using another member's would ask the arm to jump
        # branches while holding the cube.
        _live[0] = got
        retarget(ran)
        chosen = [goals[i][ran[i]] for i in range(n)]
        go(pad([p["carry"] for p in chosen], dev), _all, PH_CARRY, None)
        diag("carry", "over")
        stay(30, torch.zeros_like(got), PH_RELEASE, None)      # release
        go(pad([p["retreat"] for p in chosen], dev), torch.zeros_like(got), PH_RETREAT, None)
        stay(40, torch.zeros_like(got), PH_RETREAT, None)      # settle before judging


        # ⭐ VERIFY PLACED -- the last of `run_expert_v1`'s checks. It is a report, not a
        # recovery: `over-box-not-inside` and `lifted-but-lost` have measured EXACTLY ZERO in
        # every batch ever run on this task, so a re-place branch would be code that has never
        # had anything to do. It is reported per attempt instead, which is what says whether
        # the retry machinery is converting failures or merely spending steps.
        ok = mdp.placed_mask(e).clone()
        ok = ok & torch.tensor(planned, device=dev)
        cube_now = mdp.object_pos_local(e, mdp.TARGET_NAME)
        print(f"[batch {b}] SUCCESS {int(ok.sum())}/{n} = {float(ok.float().mean()):.1%}"
              f"   (planned {n_ok}/{n}, grasped {int(got.sum())}/{n})", flush=True)
        for a_i in range(1 + RETRIES):
            sel = (att == a_i) & torch.tensor(planned, device=dev)
            if bool(sel.any()):
                print(f"    attempt {a_i}: {int((ok & sel).sum()):3d}/{int(sel.sum()):3d} "
                      f"placed", flush=True)
        _taxonomy(e, ok, torch.tensor(planned, device=dev), cube_now)
        if args_cli.record_video:
            os.makedirs(args_cli.record_video, exist_ok=True)
            tag = "SUCCESS" if bool(ok[0]) else "FAILED"
            for name, frames in rec_frames.items():
                if not frames:
                    continue
                path = os.path.join(args_cli.record_video, f"expert_{name}.mp4")
                imageio.mimwrite(path, frames, fps=args_cli.record_fps, quality=8,
                                 macro_block_size=1)
                print(f"[record] {path}  ({len(frames)} frames, "
                      f"{len(frames) / args_cli.record_fps:.1f} s)  env 0 = {tag}", flush=True)
                frames.clear()

        all_obs.append(torch.stack(rec_obs).cpu().numpy())     # (T, n, obs)
        all_act.append(torch.stack(rec_act).cpu().numpy())     # (T, n, 7)
        all_ph.append(np.array(rec_ph, dtype=np.int8))
        all_msk.append(torch.stack(rec_msk).cpu().numpy())     # (T, n)
        all_att.append(att.cpu().numpy())
        all_ok.append(ok.cpu().numpy())
        all_seed.append(batch_seed)
        all_planned.append(np.asarray(planned, dtype=bool))

    _write(all_obs, all_act, all_ph, all_ok, all_seed, all_planned, all_msk, all_att,
           args_cli.out)
    print(f"[demos] total wall time {time.time() - t_start:.0f}s")
    env.close()


def _taxonomy(e, ok, planned, cube_now):
    """Where did the failures go? Aggregate success hides real change (LESSONS B2)."""
    over = mdp.over_box(e)
    lifted = cube_now[:, 2] > 0.06
    fail = (~ok) & planned
    never_moved = fail & (~over) & (~lifted)
    on_desk_elsewhere = fail & (~over) & lifted
    at_box_not_in = fail & over
    print(f"    taxonomy: plan-failed {int((~planned).sum()):3d} | "
          f"never-got-there {int(never_moved.sum()):3d} | "
          f"lifted-but-lost {int(on_desk_elsewhere.sum()):3d} | "
          f"over-box-not-inside {int(at_box_not_in.sum()):3d}", flush=True)


def _write(all_obs, all_act, all_ph, all_ok, all_seed, all_planned, all_msk, all_att, out):
    """Append-safe HDF5 in the layout `re3sim/act/dataset.py` reads.

    `obs/policy`, not `obs` -- the group nesting is what the clutter dataset uses and keeping
    it identical is what lets that whole training stack be a near-verbatim port.
    """
    if not all_obs:
        print("[demos] nothing to write")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_tot = sum(o.shape[1] for o in all_obs)
    n_good = int(sum(k.sum() for k in all_ok))
    with h5py.File(out, "w") as f:
        g = f.create_group("data")
        idx = 0
        for obs, act, ph, ok, seed, planned, msk, att in zip(
                all_obs, all_act, all_ph, all_ok, all_seed, all_planned, all_msk, all_att):
            for i in range(obs.shape[1]):
                d = g.create_group(f"demo_{idx}")
                d.create_group("obs").create_dataset("policy", data=obs[:, i],
                                                     compression="gzip")
                d.create_dataset("actions", data=act[:, i], compression="gzip")
                d.create_dataset("phase", data=ph)
                # train_mask censors the whole episode when the expert failed it: a failed
                # demonstration teaches the failure. Per-segment censoring is the refinement
                # to make once the failure taxonomy says which segment is at fault.
                d.attrs["success"] = bool(ok[i])
                d.attrs["planned"] = bool(planned[i])
                d.attrs["seed"] = int(seed)
                d.attrs["env_index"] = int(i)
                # How many grasp attempts this episode needed. 0 means it worked first time;
                # >0 means the trajectory contains a retrace to home and a second approach,
                # which is a real demonstration of recovery but a much harder one to imitate.
                # Recorded so the BC side can include or exclude them deliberately rather
                # than discovering them as unexplained variance.
                d.attrs["attempts"] = int(att[i])
                # Per-STEP now, not per-episode: an env that finished early holds still while
                # the others retry, and those idle steps are masked out. Still ANDed with
                # episode success -- a failed demonstration teaches the failure.
                d.create_dataset("train_mask", data=msk[:, i] & bool(ok[i]))
                idx += 1
        g.attrs["num_demos"] = idx
        g.attrs["num_success"] = n_good
        g.attrs["task"] = "re3sim_workstation"
        g.attrs["obs_dim"] = int(all_obs[0].shape[2])
    print(f"[demos] wrote {out}: {n_tot} episodes, {n_good} successful "
          f"({n_good / max(1, n_tot):.1%}) -- only the successful ones are unmasked")


if __name__ == "__main__":
    main()
    simulation_app.close()
