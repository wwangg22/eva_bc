#!/usr/bin/env python
"""Vision flow-BC trainer (EXP08 step 3, Gate C).

Structure copied from slot_act/train_flow.py; swaps in FlowMatchingVisionPolicy
(slot_act/modeling_flow_vision.py) and VisionShardDataset (slot_act/dataset_vision.py). Config
unchanged from the champion base recipe: chunk 50, n_action_steps 15, no ensembling,
hidden 512, 10 Euler steps, AdamW. Images use fixed ImageNet normalization inside the
model; the MeanStdNormalizer covers state + action only.

Usage:
    python slot_act/train_flow_vision.py --data data/exp08_vision/seed42 data/exp08_vision/seed123 \
        --out runs/exp08_bc/v1 --steps 100000
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slot_act.configuration_act import ACTConfig, FeatureType, PolicyFeature
from slot_act.dataset_vision import ACTION_DIM, STUDENT_STATE_DIM, VisionShardDataset, compute_stats_vision
from slot_act.modeling_flow_vision import FlowMatchingVisionPolicy
from slot_act.normalize import MeanStdNormalizer

IMAGE_KEYS = ("observation.images.wrist", "observation.images.workspace")
CAM_SHAPE = (3, 90, 160)


def make_config(args: argparse.Namespace) -> ACTConfig:
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STUDENT_STATE_DIM,)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=CAM_SHAPE) for key in IMAGE_KEYS},
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


def save_checkpoint(
    path: Path, policy: FlowMatchingVisionPolicy, normalizer: MeanStdNormalizer, config: ACTConfig,
    step: int, extra: dict | None = None,
):
    # `extra` carries the DATASET's identity -- the render config the shards were collected
    # under, and whether this is the blind arm. It is passed in rather than read from a
    # global: this function is module-level and `args` is local to main() (that exact
    # NameError killed the first blind run at its first checkpoint, 10k steps in).
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "config": {
                "policy_type": "flow_vision",
                "chunk_size": config.chunk_size,
                "n_action_steps": config.n_action_steps,
                "num_inference_steps": config.num_inference_steps,
                "state_dim": STUDENT_STATE_DIM,
                "image_keys": list(IMAGE_KEYS),
                **(extra or {}),
                "cam_shape": list(CAM_SHAPE),
                "action_dim": ACTION_DIM,
            },
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, nargs="+", help="exp08 collection shard dir(s)")
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
    p.add_argument("--blind", action="store_true",
                   help="BLIND CONTROL (VISION_PLAN G3): identical architecture and capacity, "
                        "images replaced by zeros. The slot never moves, so a blind policy can "
                        "still run the whole insertion -- it just cannot find the block. Any "
                        "visual number is only meaningful read against this one.")
    p.add_argument("--include-failures", action="store_true",
                   help="train on failed episodes too (default: successes only)")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = VisionShardDataset(args.data, chunk_size=args.chunk_size,
                                 success_only=not args.include_failures)
    stats = compute_stats_vision(dataset)
    normalizer = MeanStdNormalizer(stats).to(device)

    extra = {"render": dataset.render, "blind": bool(args.blind)}
    print(f"[train] render contract: {dataset.render}   blind={args.blind}")
    config = make_config(args)
    policy = FlowMatchingVisionPolicy(config).to(device)
    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
                if args.blind:
                    for k in IMAGE_KEYS:
                        batch[k] = torch.zeros_like(batch[k])
                loss, loss_dict = policy.forward(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1

                if step % 100 == 0:
                    rec = {"step": step, "loss": loss.item(), **loss_dict, "sec": round(time.time() - t0, 1)}
                    print(json.dumps(rec), flush=True)
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                if step % args.save_every == 0:
                    save_checkpoint(out_dir / f"ckpt_{step:07d}.pt", policy, normalizer,
                                    config, step, extra)
                if step >= args.steps:
                    break

    save_checkpoint(out_dir / "ckpt_final.pt", policy, normalizer, config, step, extra)
    print(f"done: {step} steps -> {out_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
