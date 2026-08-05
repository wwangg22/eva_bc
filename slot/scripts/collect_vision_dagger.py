#!/usr/bin/env python
"""DAgger round: the VISION STUDENT drives, the champion labels — POST-HOC.

Why post-hoc and not in-loop, which is the obvious way to write this: EXP08's v1 collector
computed teacher features *inside* the driving loop and the student's success collapsed from
~80 % to ~40 % while driving. The extra GPU work between env steps perturbed the very frames it
was recording. Their v2 fix was to record raw state during the rollout and compute every label
after rolling stops, validated exact to 7.8e-8 against the live build. This is that design.

So the driving loop here does buffer reads and one host copy per step. Nothing else. The
champion never runs until the episode is over.

Labels are full 50-step chunks from the champion, taken at the STUDENT's 15-step window
boundaries — the states where the student actually re-decides, and therefore the only states
where a corrected chunk changes anything. 600 steps / 15 = exactly 40 boundaries per episode.

Shard (matches slot_act/dataset_vision.py's DAgger format, auto-detected via `label_chunks`):
    wrist_rgb / workspace_rgb  (40, 90, 160, 3) uint8   at the boundary states
    proprio                    (40, 23) float32
    label_chunks               (40, 50, 7) float32      CHAMPION actions, unnormalised
    obs34                      (40, 34) float32         teacher-only, never read by the student
    success                    bool                     the STUDENT's outcome

**Built-in audit.** The student's driving success is printed and must land near its clean
baseline (vision v1: 0.804 pooled). A materially lower number means the loop is perturbing the
rollout and the data must be thrown away, not trained on — that is exactly how EXP08 caught
their v1 problem.

    python scripts/collect_vision_dagger.py --episodes 128 --num-envs 16 --seed 303 \
        --out data/vision_dagger/seed303
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student", default="runs/vision_bc/v1/ckpt_final.pt")
parser.add_argument("--teacher", default="runs/bc_armB_seed0/ckpt_final.pt")
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=303)
parser.add_argument("--out", required=True)
parser.add_argument("--warmup-episodes", type=int, default=1)
parser.add_argument("--label-batch", type=int, default=64, help="post-hoc labelling batch size")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
import slot_mdp as mdp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from slot_act.cameras import SUPERSAMPLE, attach_cameras, rgb_native, student_proprio  # noqa: E402
from slot_act.dataset import ENV_STATE_SLICE, STATE_SLICE  # noqa: E402
from slot_act.eval_act import load_checkpoint  # noqa: E402
from slot_act.eval_flow_vision import VisionController, load_vision_checkpoint  # noqa: E402
from slot_act.normalize import MeanStdNormalizer  # noqa: E402


@torch.no_grad()
def champion_labels(policy, normalizer, obs34: torch.Tensor, batch: int) -> torch.Tensor:
    """(T, 34) privileged states -> (T, chunk, 7) champion action chunks, UNNORMALISED.

    Pure function of the observation and the flow's x0 draw -- no controller queue state is
    involved, which is what makes post-hoc labelling exactly equivalent to labelling live.
    """
    out = []
    for i in range(0, obs34.shape[0], batch):
        sub = obs34[i : i + batch]
        nb = normalizer.normalize({
            "observation.state": sub[:, STATE_SLICE],
            "observation.environment_state": sub[:, ENV_STATE_SLICE],
        })
        chunk = policy.predict_action_chunk(nb)
        out.append(normalizer.unnormalize("action", chunk).cpu())
    return torch.cat(out)


def main() -> None:
    device = "cuda:0"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    student, s_stats, s_cfg = load_vision_checkpoint(Path(args.student), device)
    assert not s_cfg.get("blind"), "the blind arm cannot drive a DAgger round"
    ss = int((s_cfg.get("render") or {}).get("supersample", SUPERSAMPLE))
    controller = VisionController(student, s_stats, s_cfg["n_action_steps"], s_cfg["chunk_size"],
                                  device)
    window = s_cfg["n_action_steps"]

    teacher, t_stats, t_cfg = load_checkpoint(Path(args.teacher), device)
    t_norm = MeanStdNormalizer(t_stats).to(device)
    print(f"[dagger] student {args.student} (render ss={ss}, window={window})")
    print(f"[dagger] teacher {args.teacher} (chunk={t_cfg['chunk_size']})")

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.seed = args.seed
    attach_cameras(env_cfg, supersample=ss)
    render = {"supersample": ss, "width": 160, "height": 90, "antialiasing": "Off",
              "update_period": env_cfg.decimation * env_cfg.sim.dt}

    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    n = args.num_envs
    obs = env.reset()[0]["policy"]
    assert u.max_episode_length % window == 0, u.max_episode_length

    ep_idx = torch.zeros(n, dtype=torch.long)
    buf = [{"wrist": [], "workspace": [], "obs34": []} for _ in range(n)]
    kept = n_succ = steps = 0
    t0 = time.time()

    while kept < args.episodes:
        # ---- DRIVING LOOP: buffer reads + host copies only. No teacher, no features, nothing.
        wrist = rgb_native(u, "wrist_cam")
        works = rgb_native(u, "workspace_cam")
        success_now = mdp.placed_mask(u).all(dim=1)
        boundary = steps % window == 0
        if boundary:
            for i in range(n):
                buf[i]["wrist"].append(wrist[i].cpu())
                buf[i]["workspace"].append(works[i].cpu())
                buf[i]["obs34"].append(obs[i].cpu())

        stu = {"joint_pos": obs[:, 0:8], "joint_vel": obs[:, 8:16], "actions": obs[:, 27:34],
               "wrist_rgb": wrist, "workspace_rgb": works}
        obs_next, _, term, trunc, _ = env.step(controller.act(stu).to(u.device))
        obs_next = obs_next["policy"]
        steps += 1

        done = (term | trunc).view(-1).nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            controller.reset(done.cpu())
            for i in done.tolist():
                b = buf[i]
                if ep_idx[i] >= args.warmup_episodes and kept < args.episodes:
                    o34 = torch.stack(b["obs34"])
                    # ---- POST-HOC. The rollout for this episode is over; the teacher may run.
                    labels = champion_labels(teacher, t_norm, o34.to(device), args.label_batch)
                    shard = {
                        "wrist_rgb": torch.stack(b["wrist"]),
                        "workspace_rgb": torch.stack(b["workspace"]),
                        "proprio": student_proprio(o34),
                        "label_chunks": labels,
                        "obs34": o34,
                        "success": bool(success_now[i]),
                        "render": render, "env": i, "seed": args.seed,
                        "episode_index_in_env": int(ep_idx[i]),
                    }
                    assert shard["proprio"].shape[1] == 23, shard["proprio"].shape
                    assert shard["label_chunks"].shape[1:] == (t_cfg["chunk_size"], 7), \
                        shard["label_chunks"].shape
                    assert shard["wrist_rgb"].shape[0] == labels.shape[0], "boundary count mismatch"
                    torch.save(shard, out / f"ep_{kept:04d}.pt")
                    kept += 1
                    n_succ += int(shard["success"])
                    if kept % 8 == 0 or kept == args.episodes:
                        el = time.time() - t0
                        print(f"[dagger] {kept}/{args.episodes} eps  STUDENT-DRIVING success "
                              f"{n_succ}/{kept} ({n_succ / kept:.3f})  {steps / el:.1f} vec-steps/s"
                              f"  {el / 60:.1f} min", flush=True)
                ep_idx[i] += 1
                for k in b:
                    b[k].clear()
        obs = obs_next

    rate = n_succ / max(kept, 1)
    meta = {"student": args.student, "teacher": args.teacher, "task": args.task,
            "seed": args.seed, "num_envs": n, "episodes": kept,
            "student_driving_success": rate, "render": render,
            "boundaries_per_episode": u.max_episode_length // window,
            "vec_steps_per_s": round(steps / (time.time() - t0), 2)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[dagger] DONE {kept} eps -> {out}")
    print(f"[dagger] AUDIT: student-driving success {rate:.3f}  (clean baseline 0.804)")
    if rate < 0.70:
        print("[dagger] *** AUDIT FAILED: the driving loop is perturbing the rollout. "
              "Do NOT train on this data -- EXP08 v1 looked exactly like this. ***")
    else:
        print("[dagger] audit OK: the post-hoc loop does not perturb the student.")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
