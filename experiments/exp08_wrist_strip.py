#!/usr/bin/env python
"""EXP08 step 0: grasp-sequence strip through the NEW wrist cam (Big Will reviews).

Drives the champion stack (frozen flow base + trained steering head, deterministic)
in ONE env under the exact ladder eval protocol, with the wrist D405 grafted onto the
scene via its spawn-time OffsetCfg (the training-faithful path). Saves a wrist still
every --frame-every steps plus per-frame numerics (frame std, pixel diff, camera->can
distance, can off-optical-axis angle, placed count) so view coverage through
approach/grasp/carry can be checked without anyone eyeballing mid-run.

Run (env_isaaclab6, from act/.. root):
    python experiments/exp08_wrist_strip.py --ckpt runs/exp03_N3/ckpt_final.pt \
        --steer-ckpt runs/exp07_steer/s1_seed1/nn/exp07_steer.pth \
        --seed 123 --out-dir runs/exp08_vision/wrist_strip_seed123 --headless
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset import OBS_DIM
from act.eval_act import BatchedACTController, load_checkpoint
from act.eval_residual import load_residual_policy
from act.steer_core import STEER_ACTION_DIM, STEER_OBS_DIM, SteerCore


def quat_mul_xyzw(a, b):
    """Hamilton product for xyzw quaternion tensors of shape (4,)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conj_xyzw(q):
    return torch.stack([-q[0], -q[1], -q[2], q[3]])


def rotmat_xyzw(q):
    x, y, z, w = q
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steer-ckpt", required=True)
    parser.add_argument("--steer-cfg", default=str(Path(__file__).parent.parent / "act" / "steer_ppo_cfg.yaml"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frame-every", type=int, default=25)
    parser.add_argument("--geo-every", type=int, default=5,
                        help="log aim geometry every N steps (no still saved unless a frame step)")
    parser.add_argument("--twin", action="store_true",
                        help="mount-tracking test: spawn a STATIC env-root camera at the wrist cam's "
                             "reported (frozen) world pose and log per-sample frame diffs between the two. "
                             "Persistent diff at denoiser level (~5) = wrist render is static; "
                             "large diffs during motion = wrist render tracks the link.")
    parser.add_argument("--max-steps", type=int, default=0, help="cut episodes short after N env steps (0 = full)")
    parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episode-length-s", type=float, default=30.0)
    parser.add_argument("--out-dir", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np

    import reBot_RL.tasks  # noqa: F401
    import reBot_RL.tasks.manager_based.pick_place.mdp as mdp
    from isaaclab_tasks.utils import parse_env_cfg
    from reBot_RL.tasks.manager_based.lift.camera_cfg import WRIST_CAM_CFG
    from reBot_RL.tasks.manager_based.lift.rebot_lift_env_cfg import _GRIPPER_END
    from reBot_RL.tasks.manager_based.pick_place.mdp.common import OBJECT_NAMES

    device = "cuda:0"
    policy, stats, ckpt_cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(
        policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device
    )
    core = SteerCore(controller, mdp, Path(__file__).parent / "exp06_grasp_bit.pt", alpha_x0=1.0, flush=True)
    window = controller.n_action_steps
    steer_act = load_residual_policy(Path(args.steer_ckpt), Path(args.steer_cfg), device,
                                     obs_dim=STEER_OBS_DIM, act_dim=STEER_ACTION_DIM)

    # --- env: exact ladder eval protocol + wrist cam grafted onto the scene ---
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.terminations.object_dropping = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    env_cfg.scene.wrist_cam = WRIST_CAM_CFG.replace(
        width=640, height=360, data_types=["rgb", "distance_to_image_plane"]
    )
    if args.twin:
        # static env-root camera at the wrist cam's reported (frozen) world pose:
        # spawn parent is world-aligned, so world pose = gripper_end spawn pos + offset.
        from isaaclab.sensors import CameraCfg as _CameraCfg

        env_cfg.scene.twin_cam = WRIST_CAM_CFG.replace(
            prim_path="{ENV_REGEX_NS}/TwinCam",
            offset=_CameraCfg.OffsetCfg(pos=(0.210, 0.0, 0.260), rot=WRIST_CAM_CFG.offset.rot,
                                        convention="opengl"),
            width=640,
            height=360,
        )
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    assert u.max_episode_length % window == 0

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    def grab_frame(cam) -> np.ndarray:
        out = cam.data.output["rgb"]
        out = getattr(out, "torch", out)
        if callable(out):
            out = out()
        return out[0, ..., :3].detach().cpu().numpy().astype("uint8")

    with torch.inference_mode():
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]
        assert obs.shape == (1, OBS_DIM), obs.shape
        cam = u.scene["wrist_cam"]
        # cam.data poses are STALE (fabric worldMatrix never recomposed for link-child
        # prims; the render tracks the link, the telemetry does not -- true at t=0 only).
        # Derive the live camera pose from the gripper body's physics pose composed with
        # the fixed camera-in-body transform measured at reset. Correctness checks: at
        # step 1 the composed pose must reproduce the spawn telemetry, and cam->TCP must
        # stay rigid (~0.171 m) for the whole episode.
        robot = u.scene["robot"]
        gidx = robot.body_names.index(_GRIPPER_END.split("/")[-1])

        def body_pose():
            bp, bq = robot.data.body_pos_w, robot.data.body_quat_w
            bp = bp.torch() if callable(getattr(bp, "torch", None)) else bp
            bq = bq.torch() if callable(getattr(bq, "torch", None)) else bq
            return bp[0, gidx].to(torch.float32), bq[0, gidx].to(torch.float32)

        p_b0, q_b0 = body_pose()
        p_c0 = cam.data.pos_w[0].to(torch.float32)
        q_c0 = cam.data.quat_w_opengl[0].to(torch.float32)
        q_cib = quat_mul_xyzw(quat_conj_xyzw(q_b0), q_c0)
        p_cib = rotmat_xyzw(q_b0).T @ (p_c0 - p_b0)
        if args.twin:
            # copy the wrist cam's REPORTED world pose onto the env-root twin at runtime
            # (spawn-offset guesses proved wrong: the gripper_end parent prim is not
            # world-aligned). Also exercises runtime set_world_poses on an env-root camera.
            twin = u.scene["twin_cam"]
            wp_, wq_ = cam.data.pos_w.clone(), cam.data.quat_w_opengl.clone()
            twin.set_world_poses(wp_, wq_, convention="opengl")
            log(f"[twin] wrist reported pos {wp_[0].tolist()} quat_opengl(xyzw) {wq_[0].tolist()}")
            log(f"[twin] twin reported pos {twin.data.pos_w[0].tolist()} "
                f"quat_opengl(xyzw) {twin.data.quat_w_opengl[0].tolist()}")

        ep, ep_step, step_i = 0, 0, 0
        prev = None
        (out_root / f"ep{ep}").mkdir(exist_ok=True)
        while ep < args.episodes:
            if step_i % window == 0:
                core.set_steer(steer_act(core.build_obs(obs, u)))
            placed_now = mdp.placed_mask(u)
            last_placed = bool(placed_now.all())
            core.flush_check(u)
            action = core.controller.act(obs)
            obs_dict, _, terminated, truncated, _ = env.step(action.to(u.device))
            obs = obs_dict["policy"]
            ep_step += 1
            step_i += 1

            frame_step = (ep_step - 1) % args.frame_every == 0
            if frame_step or (ep_step - 1) % args.geo_every == 0:
                # aim geometry in the CAMERA frame (opengl: look -Z, up +Y, right +X):
                # v_cam = R(q)^T (p - cam_pos); h/v = signed angles off the optical axis,
                # in-FOV iff z_cam < 0 and |h| <= 42 deg (HFOV/2) and |v| <= 27 deg (VFOV/2 at 16:9).
                p_b, q_b = body_pose()
                q = quat_mul_xyzw(q_b, q_cib)  # live opengl-frame camera quat (xyzw)
                rot = rotmat_xyzw(q)
                cam_pos = p_b + rotmat_xyzw(q_b) @ p_cib
                placed = int(mdp.placed_mask(u).sum())
                tcp = u.scene["ee_frame"].data.target_pos_w[0, 0].to(torch.float32)

                def hv(p):
                    vc = rot.T @ (p - cam_pos)
                    return (float(torch.linalg.norm(vc)),
                            float(torch.rad2deg(torch.atan2(vc[0], -vc[2]))),
                            float(torch.rad2deg(torch.atan2(vc[1], -vc[2]))),
                            bool(vc[2] < 0) and abs(float(torch.rad2deg(torch.atan2(vc[0], -vc[2])))) <= 42.0
                            and abs(float(torch.rad2deg(torch.atan2(vc[1], -vc[2])))) <= 27.0)

                # cam->TCP is wrist-rigid: h/v must be constant (~1.8 deg total) every sample,
                # else the rotation math or the pose buffers are wrong.
                td, th, tv, _ = hv(tcp)
                # depth discriminator: wrist-mounted => gripper rigid at frame center =>
                # center depth ~constant 0.10-0.17 m; static camera => center depth tracks
                # whatever lies along the frozen ray (table ~0.3-0.5 m, dips on arm pass).
                dep = cam.data.output["distance_to_image_plane"]
                dep = getattr(dep, "torch", dep)
                if callable(dep):
                    dep = dep()
                dep = dep[0].detach().to(torch.float32)
                dep2 = dep.squeeze(-1) if dep.dim() == 3 else dep
                center = float(dep2[dep2.shape[0] // 2, dep2.shape[1] // 2])
                finite = dep2[torch.isfinite(dep2) & (dep2 > 0.01)]
                dmin = float(finite.min()) if finite.numel() else float("nan")
                geo = [f"cam({cam_pos[0]:+.3f},{cam_pos[1]:+.3f},{cam_pos[2]:+.3f})",
                       f"tcp {td:.3f} m h{th:+5.1f} v{tv:+5.1f}",
                       f"depth ctr {center:.3f} min {dmin:.3f}"]
                for name in OBJECT_NAMES:
                    p = u.scene[name].data.root_pos_w
                    p = (p.torch() if callable(getattr(p, "torch", None)) else p)[0].to(torch.float32)
                    d, h, v, in_fov = hv(p)
                    geo.append(f"{name}: w({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) {d:.3f} m "
                               f"h{h:+6.1f} v{v:+6.1f} tcp{float(torch.linalg.norm(p - tcp)):.3f} "
                               f"{'IN ' if in_fov else 'out'}")
                if frame_step:
                    frame = grab_frame(cam)
                    fname = f"step{ep_step:04d}_placed{placed}.png"
                    imageio.imwrite(out_root / f"ep{ep}" / fname, frame)
                    diff = float(np.abs(frame.astype(np.int16) - prev).mean()) if prev is not None else float("nan")
                    std = float(frame.std())
                    log(f"[ep{ep} step {ep_step:4d}] {fname}  std {std:6.1f}  diff {diff:6.1f}  "
                        f"placed {placed}  |  " + "  |  ".join(geo))
                    if std < 2.0:
                        log(f"[WARN] ep{ep} step {ep_step}: near-uniform frame (std {std:.2f})")
                    prev = frame.astype(np.int16)
                else:
                    log(f"[ep{ep} step {ep_step:4d}] geo                            "
                        f"placed {placed}  |  " + "  |  ".join(geo))
                if args.twin:
                    wf = grab_frame(cam).astype(np.int16)
                    tf = grab_frame(u.scene["twin_cam"]).astype(np.int16)
                    log(f"[ep{ep} step {ep_step:4d}] TWIN wrist-vs-static diff {np.abs(wf - tf).mean():6.2f}"
                        f"  (wrist std {wf.std():5.1f} / twin std {tf.std():5.1f})")
                    if frame_step:
                        imageio.imwrite(out_root / f"ep{ep}" / f"twin_step{ep_step:04d}.png",
                                        tf.astype("uint8"))

            if args.max_steps and ep_step >= args.max_steps:
                log(f"[ep{ep} CUT at {ep_step} steps (--max-steps)]")
                break

            if bool(terminated[0] | truncated[0]):
                log(f"[ep{ep} DONE] length {ep_step}  success={last_placed}")
                core.reset([0])
                ep += 1
                ep_step = 0
                prev = None
                if ep < args.episodes:
                    (out_root / f"ep{ep}").mkdir(exist_ok=True)

    (out_root / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"[done] strips + summary.txt -> {out_root}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
