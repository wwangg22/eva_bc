#!/usr/bin/env python
"""Does the render SHIMMER frame-to-frame on a static scene? (EXP08 renderer verdict, retested)

EXP08 closed its renderer investigation with: static-scene per-pixel **temporal** std is 31-37
on >99.6 % of pixels, identical for FXAA and Off, unchanged by `samples_per_pixel` -- the RTX
real-time path jitters its projection every frame, and only a temporal filter (DLSS) integrates
it away. They took DLSS + a no-GPU-work rule.

That finding puts my own G0 claim in doubt. G0's "static-render diff = 0.0" called `sim.render()`
twice **without advancing the frame index**, so a frame-index-deterministic jitter would produce
exactly 0.0 and I would have concluded "clean" from a probe that could not see the artifact. The
same class of mistake as the uint8 aliasing bug: an instrument that cannot fail.

So this measures the thing properly -- STEP the sim (which advances the frame index) while the
scene is physically static, and report per-pixel temporal std of the frames the POLICY consumes.

The reason it matters: EXP08 only compared non-temporal modes at 1x resolution. Supersampling is
a third option they did not test. If the jitter is per-pixel and roughly independent, box-
averaging k*k samples should cut its temporal std by ~k -- which would beat DLSS without any
temporal filter, and therefore without the GPU-load dependence that floored their DAgger driver.

    python scripts/vision_shimmer_probe.py --supersample 1
    python scripts/vision_shimmer_probe.py --supersample 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--supersample", type=int, default=8)
parser.add_argument("--aa", default="Off", choices=["Off", "FXAA", "DLSS", "TAA", "DLAA"])
parser.add_argument("--settle", type=int, default=120, help="steps of champion, then freeze")
parser.add_argument("--hold", type=int, default=40, help="frames captured while static")
parser.add_argument("--ckpt", default="runs/bc_armB_seed0/ckpt_final.pt")
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--out", default="runs/vision_shimmer")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from slot_act.cameras import CAMERA_NAMES, attach_cameras, rgb_native  # noqa: E402
from slot_act.eval_act import BatchedACTController, load_checkpoint  # noqa: E402


def main() -> None:
    device = "cuda:0"
    policy, stats, cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"],
                                      device, fixed_x0=torch.zeros(cfg["chunk_size"], 7))

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.seed = args.seed
    attach_cameras(env_cfg, antialiasing=args.aa, supersample=args.supersample)

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    obs = env.reset()[0]["policy"]
    for _ in range(args.settle):
        obs = env.step(controller.act(obs).to(u.device))[0]["policy"]

    # Freeze the command: repeat the last action so the arm holds pose. Physics keeps stepping
    # (that is the point -- the frame index must advance), so residual motion is measured, not
    # assumed, via joint velocity.
    hold_action = controller.act(obs).to(u.device)
    frames = {c: [] for c in CAMERA_NAMES}
    jv = []
    for _ in range(args.hold):
        obs = env.step(hold_action)[0]["policy"]
        jv.append(float(obs[0, 8:16].abs().max()))
        for c in CAMERA_NAMES:
            frames[c].append(rgb_native(u, c)[0].float())

    result = {"supersample": args.supersample, "aa": args.aa, "hold": args.hold,
              "max_joint_vel_rel_during_hold": round(max(jv), 5), "cams": {}}
    for c in CAMERA_NAMES:
        stack = torch.stack(frames[c])                      # (T, 90, 160, 3)
        tstd = stack.std(dim=0)                             # per-pixel temporal std
        drift = (stack[-1] - stack[0]).abs().mean()
        result["cams"][c] = {
            "temporal_std_mean": round(float(tstd.mean()), 3),
            "temporal_std_p99": round(float(tstd.flatten().quantile(0.99)), 3),
            "frac_pixels_std_gt_2": round(float((tstd > 2).float().mean()), 4),
            "first_to_last_drift": round(float(drift), 3),
        }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"ss{args.supersample}_{args.aa}.json").write_text(json.dumps(result, indent=2))

    print(f"\n[shimmer] supersample={args.supersample} aa={args.aa} "
          f"max|joint_vel| during hold {result['max_joint_vel_rel_during_hold']}")
    for c, m in result["cams"].items():
        print(f"  {c:<14} temporal std mean {m['temporal_std_mean']:>7.3f}  p99 "
              f"{m['temporal_std_p99']:>7.3f}  pixels>2 {m['frac_pixels_std_gt_2'] * 100:>5.1f}%"
              f"  drift {m['first_to_last_drift']:.3f}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
