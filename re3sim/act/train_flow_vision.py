#!/usr/bin/env python
"""Vision flow-BC for the workstation pick-and-place, with NO privileged information.

The student sees exactly what the real rig can measure:

    observation.images.wrist      (3, 120, 160)  the D405 on the gripper
    observation.images.workspace  (3, 120, 160)  the workstation camera, 75 deg down
    observation.state             (23,)          joint_pos(8) + joint_vel(8) + last_action(7)

and nothing else. The cube pose, the box pose, the clutter offsets and the placed flag --
``obs41[16:34]``, which every state-based policy on this task has been fed -- are written into
the shards for the expert side and are *dropped at load time* by
``act.dataset_vision.VisionShardDataset`` (`_STUDENT_KEYS`). The 23-D proprio is byte-derived
from the same observation manager terms in ``re3sim/expert/collect_demos.py``, which asserts
the slicing at write time. Both halves of that contract are enforced in code rather than by
convention, because a leak is invisible in the loss: privileged features make training look
*better*.

Structure copied from ``act/train_flow_vision.py`` (EXP08 Gate C) and the recipe is unchanged,
because those numbers were paid for upstream: rectified flow rather than a CVAE, chunk 50,
n_action_steps 15, no temporal ensembling, dim 512, 10 Euler steps, AdamW at 1e-4. The dataset
loader is *imported* rather than copied: the shard contract (`wrist_rgb`, `workspace_rgb`,
`proprio`, `actions`, `label_chunks`) and both student dims (23, 7) are identical to
pick-place's, so a copy would only be a second thing to keep in sync.

What differs from the pick-place version, and why:

* **CAM_SHAPE is 4:3, not 16:9.** The wrist camera carries the D405's measured intrinsics,
  calibrated at 640x480; rendering it 16:9 changes the vertical FOV and silently decalibrates
  the mount that was matched against the live feed.
* **The checkpoint carries a task tag.** The student dims here are numerically identical to
  pick-place's, so without the tag a pick-place vision checkpoint loads into this task without
  complaint and simply performs badly.

Usage
-----
    python -u re3sim/act/train_flow_vision.py \\
        --data re3sim/expert/data/vision_s1 re3sim/expert/data/vision_s2 \\
        --out re3sim/runs/vbc_r1 --steps 100000 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_EVA_BC = Path(__file__).resolve().parents[2]      # .../eva_bc
sys.path.insert(0, str(_EVA_BC))

from act.configuration_act import ACTConfig, FeatureType, PolicyFeature  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))   # for the local dataset filter

from act.dataset_vision import ACTION_DIM, STUDENT_STATE_DIM  # noqa: E402
from dataset_vision import (  # noqa: E402
    WorkstationVisionDataset,
    compute_stats_vision,
)
from act.modeling_flow_vision import FlowMatchingVisionPolicy  # noqa: E402
from act.normalize import MeanStdNormalizer  # noqa: E402

IMAGE_KEYS = ("observation.images.wrist", "observation.images.workspace")
#: 4:3 -- see the module docstring. Asserted against the data, not assumed.
CAM_SHAPE = (3, 120, 160)
#: Guards against loading a pick-place vision checkpoint, whose dims are identical.
TASK_TAG = "re3sim_workstation_vision"


def make_config(args: argparse.Namespace, cam_shape: tuple[int, int, int]) -> ACTConfig:
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE,
                                               shape=(STUDENT_STATE_DIM,)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=cam_shape)
               for key in IMAGE_KEYS},
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=None,
        use_vae=False,
        dim_model=512,
        device=args.device,
    )
    config.num_inference_steps = args.num_inference_steps
    return config


def save_checkpoint(path: Path, policy, normalizer, config, step, cam_shape):
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "config": {
                "policy_type": "flow_vision",
                "task": TASK_TAG,
                "chunk_size": config.chunk_size,
                "n_action_steps": config.n_action_steps,
                "num_inference_steps": config.num_inference_steps,
                "state_dim": STUDENT_STATE_DIM,
                "image_keys": list(IMAGE_KEYS),
                "cam_shape": list(cam_shape),
                "action_dim": ACTION_DIM,
            },
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, nargs="+", help="shard dir(s) from collect_demos")
    p.add_argument("--out", required=True, help="output dir (checkpoints + log)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--n-action-steps", type=int, default=15)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--include-failures", action="store_true",
                   help="train on failed episodes too (default: successes only)")
    p.add_argument("--keep-black", action="store_true",
                   help="keep samples whose camera frame is black (default: drop them -- see "
                        "re3sim/act/dataset_vision.py)")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = WorkstationVisionDataset(args.data, chunk_size=args.chunk_size,
                                       success_only=not args.include_failures,
                                       drop_black=not args.keep_black)
    # Take the image shape FROM the data. Hardcoding it means a dataset collected at another
    # resolution trains a model whose position embeddings do not match the eval renderer, and
    # the only symptom is a policy that scores zero.
    probe = dataset[0]["observation.images.wrist"]
    cam_shape = tuple(int(v) for v in probe.shape)
    if cam_shape != CAM_SHAPE:
        print(f"[train] NOTE image shape {cam_shape} != default {CAM_SHAPE}; using the data's",
              flush=True)
    for ep in dataset.episodes:
        assert "obs41" not in ep, "privileged obs reached the student dataset"

    stats = compute_stats_vision(dataset)
    normalizer = MeanStdNormalizer(stats).to(device)

    config = make_config(args, cam_shape)
    policy = FlowMatchingVisionPolicy(config).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    log_path = out_dir / "train_log.jsonl"
    step, t0 = 0, time.time()
    with open(log_path, "a") as log_f:
        while step < args.steps:
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                batch = normalizer.normalize(batch)
                loss, loss_dict = policy.forward(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                if step % 100 == 0:
                    rec = {"step": step, "loss": loss.item(), **loss_dict,
                           "sec": round(time.time() - t0, 1)}
                    print(json.dumps(rec), flush=True)
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                if step % args.save_every == 0:
                    save_checkpoint(out_dir / f"ckpt_{step:07d}.pt", policy, normalizer,
                                    config, step, cam_shape)
                if step >= args.steps:
                    break

    save_checkpoint(out_dir / "ckpt_final.pt", policy, normalizer, config, step, cam_shape)
    print(f"done: {step} steps -> {out_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
