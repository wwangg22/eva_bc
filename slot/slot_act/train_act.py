#!/usr/bin/env python
"""Minimal ACT training script (PLAN.md Stage 2.3).

Config per plan: chunk 50, n_action_steps 15, temporal ensembling OFF, kl_weight 10,
hidden 512, AdamW. No eval loop here — batched sim eval is a separate script.

Usage:
    python train_act.py --data demos.hdf5 --out runs/act_nominal --steps 100000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slot_act.configuration_act import ACTConfig, FeatureType, PolicyFeature
from slot_act.dataset import (
    ACTION_DIM,
    ENV_STATE_DIM,
    STATE_DIM,
    RebotDemoDataset,
    compute_stats,
    nominal_dagger_pool_filter,
    nominal_pool_filter,
    recovery_pool_filter,
)
from slot_act.modeling_act import ACTPolicy
from slot_act.normalize import MeanStdNormalizer


def make_config(args: argparse.Namespace) -> ACTConfig:
    return ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(ENV_STATE_DIM,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=None,  # OFF per plan (stale pre-event chunks suppress corrections)
        kl_weight=10.0,
        dim_model=512,
        device=args.device,
    )


def save_checkpoint(path: Path, policy: ACTPolicy, normalizer: MeanStdNormalizer, config: ACTConfig, step: int):
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "config": {
                "chunk_size": config.chunk_size,
                "n_action_steps": config.n_action_steps,
                "state_dim": STATE_DIM,
                "env_state_dim": ENV_STATE_DIM,
                "action_dim": ACTION_DIM,
            },
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, nargs="+", help="demo HDF5 file(s)")
    p.add_argument("--pool", choices=["default", "nominal", "recovery", "nominal+dagger"], default="default",
                   help="demo filter: nominal = first-attempt-clean successes only; "
                        "nominal+dagger adds successful recovery_dagger takeovers")
    p.add_argument("--out", required=True, help="output dir (checkpoints + log)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--n-action-steps", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pool_filter = {"default": None, "nominal": nominal_pool_filter, "recovery": recovery_pool_filter,
                   "nominal+dagger": nominal_dagger_pool_filter}[args.pool]
    dataset = RebotDemoDataset(args.data, chunk_size=args.chunk_size, demo_filter=pool_filter)
    stats = compute_stats(dataset)
    normalizer = MeanStdNormalizer(stats).to(device)
    print(f"dataset: {len(dataset.demos)} demos, {len(dataset)} samples")

    config = make_config(args)
    policy = ACTPolicy(config).to(device)
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
                    save_checkpoint(out_dir / f"ckpt_{step:07d}.pt", policy, normalizer, config, step)
                if step >= args.steps:
                    break

    save_checkpoint(out_dir / "ckpt_final.pt", policy, normalizer, config, step)
    print(f"done: {step} steps -> {out_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
