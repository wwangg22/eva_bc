#!/usr/bin/env python
"""VISION_PLAN Gate G0 — do the cameras render, keep rendering, and see the task?

Cheapest gate in the plan and the one that protects everything after it: if the wrist view is
blind once the block is held, every frame of a 13 GB collection inherits a dead lens.

Three checks run automatically, one is Big Will's:

  a. FRAME FRESHNESS, per camera, every step, on a MOVING robot. EXP08 watched a second Camera
     sensor's buffer freeze after ~35 steps while the arm crossed its view, and lift_vision's
     smoke test (shapes only) would not have caught it. Reported as mean |frame_t - frame_t-1|;
     a camera whose stream is alive shows a clearly non-zero, varying number.
  b. NEAR-UNIFORM TRIPWIRE: per-frame std < 2.0 means the lens is pointing into the wrist
     housing or empty space.
  c. WRIST DEPTH-MIN in 2-5 cm every frame. This is the check that proves the RENDER tracks the
     link: the gripper housing sits permanently a few cm in front of the D405. It is also the
     only honest way to ask the question, because `Camera.data.pos_w` is frozen at the spawn
     transform for a link-mounted camera in Isaac Lab 3.0 (EXP08 retracted a batch of FOV
     numbers to exactly that).
  d. STILLS every --still-every steps from both cameras, for Big Will to review. The question
     that matters: once the block is in the gripper, does the wrist view still show the slot?

The champion drives, so the trajectory is the one the student will be distilled from -- not a
scripted wiggle that visits states collection never will.

    python scripts/vision_g0.py --ckpt runs/bc_armB_seed0/ckpt_final.pt --steps 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt", default="runs/bc_armB_seed0/ckpt_final.pt")
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--still-every", type=int, default=25)
parser.add_argument("--out", default="runs/vision_g0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
import slot_mdp as mdp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from slot_act.cameras import (  # noqa: E402
    CAMERA_NAMES,
    attach_cameras,
    frame_freshness,
    rgb,
    student_proprio,
    wrist_camera_pose,
)
from slot_act.eval_act import BatchedACTController, load_checkpoint  # noqa: E402

STD_TRIPWIRE = 2.0
DEPTH_MIN_OK = (0.015, 0.08)   # gripper housing a few cm in front of the D405


def main() -> None:
    device = "cuda:0"
    out = Path(args.out)
    (out / "wrist").mkdir(parents=True, exist_ok=True)
    (out / "workspace").mkdir(parents=True, exist_ok=True)

    policy, stats, cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    # Both halves, always: dropping_penalty/toppling_penalty are is_terminated_terms that look
    # their termination up BY NAME, so nulling only the terminations raises at manager build.
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.seed = args.seed
    attach_cameras(env_cfg)
    # depth only for this gate: it is check (c), and it is the render-side proof the camera
    # rides the link. Collection stays rgb-only.
    env_cfg.scene.wrist_cam.data_types = ["rgb", "distance_to_image_plane"]

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    obs = env.reset()[0]["policy"]
    assert obs.shape == (1, 34), obs.shape
    _ = student_proprio(obs)  # contract assert, before anything is written

    prev = {c: None for c in CAMERA_NAMES}
    rows: list[dict] = []
    for step in range(args.steps):
        frames = {c: rgb(u, c) for c in CAMERA_NAMES}
        depth = u.scene["wrist_cam"].data.output["distance_to_image_plane"]
        d = depth[0][torch.isfinite(depth[0])]
        block = mdp.object_pos_local(u, mdp.OBJECT_NAMES[0])[0]
        cam_pos, _ = wrist_camera_pose(u)
        rows.append({
            "step": step,
            "block_x": round(float(block[0]), 4),
            "block_z": round(float(block[2]), 4),
            "cam_z": round(float(cam_pos[0, 2]), 4),
            "depth_min_m": round(float(d.min()), 4) if d.numel() else None,
            **{f"{c}_std": round(float(frames[c][0].float().std()), 2) for c in CAMERA_NAMES},
            **{f"{c}_diff": round(frame_freshness(prev[c], frames[c]), 3) for c in CAMERA_NAMES},
        })
        if step % args.still_every == 0:
            for c in CAMERA_NAMES:
                Image.fromarray(frames[c][0].cpu().numpy()).save(
                    out / c.split("_")[0] / f"step_{step:04d}.png")
        prev = frames

        obs = env.step(controller.act(obs).to(u.device))[0]["policy"]

    live = [r for r in rows[1:]]
    summary = {
        "ckpt": args.ckpt, "seed": args.seed, "steps": args.steps,
        "block_x_range": [min(r["block_x"] for r in rows), max(r["block_x"] for r in rows)],
        "wrist_cam_z_moves": max(r["cam_z"] for r in rows) - min(r["cam_z"] for r in rows),
    }
    for c in CAMERA_NAMES:
        diffs = [r[f"{c}_diff"] for r in live]
        stds = [r[f"{c}_std"] for r in rows]
        summary[c] = {
            "frozen_steps": sum(1 for v in diffs if v < 1e-6),
            "diff_min": round(min(diffs), 3), "diff_med": round(sorted(diffs)[len(diffs) // 2], 3),
            "std_min": round(min(stds), 2),
            "near_uniform_frames": sum(1 for v in stds if v < STD_TRIPWIRE),
        }
    # STATIC-RENDER PROBE. The per-step diffs above conflate two things: the scene moving and
    # the renderer being non-deterministic. EXP08's live blocker is "frames are GPU-load-
    # dependent", so the split matters -- if a frozen scene still renders differently every
    # time, every collected frame carries that noise and the student trains through it.
    # Render twice with NO physics step and diff.
    static = {}
    for c in CAMERA_NAMES:
        sensor = u.scene[c]
        u.sim.render()
        sensor.update(dt=0.0, force_recompute=True)
        a = rgb(u, c)
        u.sim.render()
        sensor.update(dt=0.0, force_recompute=True)
        b = rgb(u, c)
        static[c] = round(float((b[0].float() - a[0].float()).abs().mean()), 3)
    summary["static_render_diff"] = static

    dm = [r["depth_min_m"] for r in rows if r["depth_min_m"] is not None]
    summary["wrist_depth_min"] = {"min": round(min(dm), 4), "max": round(max(dm), 4),
                                  "out_of_band": sum(1 for v in dm
                                                     if not DEPTH_MIN_OK[0] <= v <= DEPTH_MIN_OK[1])}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "per_step": rows}, indent=2))

    print("\n=== G0 ===")
    for c in CAMERA_NAMES:
        s = summary[c]
        verdict = "ALIVE" if s["frozen_steps"] == 0 and s["near_uniform_frames"] == 0 else "*** SUSPECT ***"
        print(f"  {c:<14} frozen {s['frozen_steps']}/{len(live)}  diff min/med "
              f"{s['diff_min']}/{s['diff_med']}  std min {s['std_min']}  "
              f"near-uniform {s['near_uniform_frames']}   {verdict}")
    st = summary["static_render_diff"]
    print(f"  static-render diff (no physics step): " +
          "  ".join(f"{c}={v}" for c, v in st.items()) +
          ("   deterministic render -- per-step diffs are real motion"
           if max(st.values()) < 1.0 else
           "   *** RENDER IS NON-DETERMINISTIC -- frames carry this much noise ***"))
    w = summary["wrist_depth_min"]
    print(f"  wrist depth-min {w['min']}-{w['max']} m, out of band {w['out_of_band']}/{len(dm)}"
          f"   {'RENDER TRACKS THE LINK' if w['out_of_band'] == 0 else '*** CHECK MOUNT ***'}")
    print(f"  block x {summary['block_x_range']}  (stage_x 0.165, slot 0.245)")
    print(f"  stills -> {out}/wrist/ and {out}/workspace/   <- Big Will reviews these")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
