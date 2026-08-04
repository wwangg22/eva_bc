#!/usr/bin/env python
"""Camera attach + the no-privileged-information contract for the slot vision policy.

Two jobs, deliberately in one small module so there is exactly one place each can be got wrong:

1. **Attach the rig's two cameras to the slot env WITHOUT editing eva_rl.** The configs come
   from ``reBot_RL .../lift/camera_cfg.py`` (Big Will's ``e56e7df``): a D455 over the workspace
   and the D405 on the gripper, the latter tilted -30 deg about camera-local X so its optical
   axis lands 1.8 deg off the TCP at 0.171 m. They attach to the slot scene unchanged because
   ``WRIST_CAM_CFG.prim_path`` is built from ``_GRIPPER_END`` = ``<base>/link1/.../gripper_end``
   and ``precision_slot_env_cfg.py`` builds its ``ee_frame`` from the identical chain. Patching
   is post-``parse_env_cfg``, exactly as ``--slot-dx`` and ``--arm-jitter`` already do.

2. **Cut the 34-D privileged observation down to the 23-D student view, in one function.**
   ``student_proprio`` is the single source of truth; collection, training and eval all call it,
   and so does the blind control arm. A privileged dim can only leak by editing this file.

Inherited from eva_bc EXP08, which paid for each of these in debugging hours (docs/slot/
VISION_PLAN.md section 3):

* ``antialiasing_mode`` defaults to **DLSS**, whose input is far below its ~300 px minimum at
  160x90. EXP08 traced GPU-load-dependent frames to it and their DAgger gate is blocked on it as
  of this writing. We force it **Off** at attach time rather than inherit the default.
* **Never read** ``Camera.data.pos_w`` / ``quat_w_*`` on a link-mounted camera in Isaac Lab 3.0:
  the fabric ``worldMatrix`` is never recomposed, so the pose reads frozen at spawn while the
  RENDER correctly tracks the link. EXP08 retracted a batch of FOV numbers to this. Use
  ``wrist_camera_pose()`` below, which composes the live gripper body pose instead.
* With multiple ``Camera`` sensors, a second camera's buffer has been seen to **freeze after
  ~35 steps** while the arm provably crossed its view. ``frame_freshness`` exists so every
  collection run can assert frames actually update, on a moving robot, before it writes a shard.
"""

from __future__ import annotations

import torch

# 34-D privileged layout (precision_slot_env_cfg.py ObservationsCfg.PolicyCfg):
#   [0:8] joint_pos_rel | [8:16] joint_vel_rel | [16:23] block_pose_in_root
#   [23:27] slot_frame  | [27:34] last_action
# The student may see encoders and its own command, and nothing else. slot_frame is the worst
# of the privileged terms: its 4th element IS the success predicate.
STUDENT_PROPRIO_SLICES = (slice(0, 16), slice(27, 34))
STUDENT_STATE_DIM = 23
PRIVILEGED_SLICES = {"block_pose_in_root": slice(16, 23), "slot_frame": slice(23, 27)}

CAM_WIDTH, CAM_HEIGHT = 160, 90          # what the POLICY consumes
CAM_SHAPE = (3, CAM_HEIGHT, CAM_WIDTH)
CAMERA_NAMES = ("wrist_cam", "workspace_cam")

# Render at SUPERSAMPLE x the policy resolution and box-average down. Measured on this rig
# (scripts/run_render_sweep.sh, 150-step pinned pose, per-pixel noise proxy on the 160x90 frame
# the policy sees):
#
#   supersample   1      2      4      8      16     | DLSS at 1x
#   noise        27.5   14.2    7.9    4.6    3.2    |    2.3
#
# The 1/k fall-off is what independent per-pixel render noise does under box-averaging, so the
# grain is real sampling noise, not aliasing. Two knobs that look like they should help do
# NOTHING in this Isaac Lab build -- `samples_per_pixel` and `antialiasing_mode="FXAA"` both
# produced bit-identical frames to the defaults, so supersampling is the lever that exists.
# DLSS is the only cheaper route to low noise and it is rejected: it reconstructs from a lower
# internal resolution (its frames sit 22.6 away from the supersampled reference, vs 8.2 for 8x),
# and it is the prime suspect for the GPU-load-dependent frames blocking EXP08's Gate D.
# CHOSEN 4, not 8, and the reason is measured rather than aesthetic (scripts/vision_fps.py):
#
#   config              envs   ep-steps/s   256 episodes   temporal std
#   no cameras            16        369.0        0.12 h      -
#   supersample 2         16         79.6        0.54 h     ~18 (interpolated)
#   supersample 4         16         35.8        1.19 h      9.06
#   supersample 8          8         10.5        4.06 h      4.68
#   supersample 8         16      OUT OF MEMORY (11 GB card)  4.68
#
# 8x is memory-bound to 8 envs on this GPU and 3.4x slower per episode. 4x keeps 16 envs, cuts
# the raw 35.2 shimmer by 3.9x, and makes a DAgger round affordable -- and DAgger is where
# EXP08's points actually came from. Residual shimmer at 9.06 is on the record as the live
# suspect if the student's failures look like perception rather than precision.
SUPERSAMPLE = 4


def student_proprio(obs34: torch.Tensor) -> torch.Tensor:
    """(N, 34) privileged obs -> (N, 23) student proprioception. The ONLY way to build it."""
    assert obs34.shape[-1] == 34, obs34.shape
    out = torch.cat([obs34[..., s] for s in STUDENT_PROPRIO_SLICES], dim=-1)
    assert out.shape[-1] == STUDENT_STATE_DIM, out.shape
    return out


def attach_cameras(env_cfg, width: int | None = None, height: int | None = None,
                   env_spacing: float | None = 6.0, antialiasing: str = "Off",
                   supersample: int = SUPERSAMPLE) -> None:
    """Add both cameras to an already-parsed env cfg, in place. Import only AFTER AppLauncher.

    ``env_spacing`` follows lift_vision's 6.0: the workspace camera has a 90 deg horizontal
    FOV, so at the default spacing the neighbouring workstations are large in frame and the
    policy can learn to read the wrong one.
    """
    from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG, WRIST_CAM_CFG

    width = CAM_WIDTH * supersample if width is None else width
    height = CAM_HEIGHT * supersample if height is None else height
    # One render per policy step. Anything else silently gives the policy stale or duplicated
    # frames relative to the proprioception recorded alongside them.
    update_period = env_cfg.decimation * env_cfg.sim.dt
    kwargs = {"width": width, "height": height, "update_period": update_period}
    env_cfg.scene.wrist_cam = WRIST_CAM_CFG.replace(**kwargs)
    env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(**kwargs)

    # DLSS is a temporal upscaler; at 160x90 it is being asked to work far below its documented
    # ~300 px minimum input, and EXP08's frames turned out to depend on GPU load. Off.
    env_cfg.sim.render.antialiasing_mode = antialiasing
    if env_spacing is not None:
        env_cfg.scene.env_spacing = env_spacing

    assert env_cfg.scene.wrist_cam.height == height and env_cfg.scene.wrist_cam.width == width
    assert abs(env_cfg.scene.wrist_cam.update_period - update_period) < 1e-12
    print(f"[cameras] wrist + workspace @ {width}x{height}, update_period {update_period:g} s, "
          f"antialiasing {antialiasing}, env_spacing {env_cfg.scene.env_spacing}")


def audit_no_privileged(sample: dict) -> None:
    """Fail loudly if anything privileged reached a student sample (VISION_PLAN section 0)."""
    banned = ("block_pose", "slot_frame", "obs34", "insertion_depth", "lateral_error",
              "is_inserted", "object_pose")
    for key in sample:
        low = key.lower()
        assert not any(b in low for b in banned), f"privileged key in student sample: {key}"
    state = sample.get("observation.state")
    if state is not None:
        assert state.shape[-1] == STUDENT_STATE_DIM, (
            f"student state is {state.shape[-1]}-D, must be {STUDENT_STATE_DIM}")


def rgb(env, name: str) -> torch.Tensor:
    """(N, H, W, 3) uint8 RGB for one camera. Drops any alpha channel the sensor returns.

    **Cloned, and that is not optional.** The sensor hands back a view onto a buffer it
    overwrites in place, and ``.to(uint8)`` on already-uint8 data is a no-op that returns the
    same tensor -- so a caller holding "last frame" would be holding *this* frame. The first run
    of Gate G0 reported both cameras frozen 299/299 for exactly this reason: the instrument was
    subtracting a buffer from itself while the renderer was working fine.
    """
    data = env.scene[name].data.output["rgb"]
    return data[..., :3].to(torch.uint8).clone()


def rgb_native(env, name: str) -> torch.Tensor:
    """(N, 90, 160, 3) uint8 — what the POLICY consumes, box-averaged down from the render.

    Box-average, not bilinear resize: averaging k*k independent samples is the whole point, and
    it is what drops the per-pixel noise proxy from 27.5 to 4.6 at supersample 8.
    """
    img = rgb(env, name)
    if img.shape[1] == CAM_HEIGHT and img.shape[2] == CAM_WIDTH:
        return img
    x = img.permute(0, 3, 1, 2).float()
    x = torch.nn.functional.adaptive_avg_pool2d(x, (CAM_HEIGHT, CAM_WIDTH))
    return x.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)


def frame_freshness(prev: torch.Tensor | None, now: torch.Tensor) -> float:
    """Mean absolute inter-frame pixel change, env 0. Near-zero over many steps on a MOVING
    robot means the buffer has frozen -- see the module docstring."""
    if prev is None:
        return float("nan")
    return float((now[0].float() - prev[0].float()).abs().mean())


def wrist_camera_pose(env, body_name: str = "gripper_end") -> tuple[torch.Tensor, torch.Tensor]:
    """Live wrist-camera pose, composed from the gripper BODY physics pose.

    Isaac Lab 3.0 leaves ``Camera.data.pos_w``/``quat_w_*`` frozen at the spawn transform for a
    link-parented camera, so reading them yields a camera that appears to hover in mid-air while
    the arm moves. Returns (pos_w, quat_w_wxyz) of the body the camera rides; compose the cfg
    offset onto it for the true optical pose.
    """
    robot = env.scene["robot"]
    idx = robot.body_names.index(body_name)
    return robot.data.body_pos_w[:, idx], robot.data.body_quat_w[:, idx]
