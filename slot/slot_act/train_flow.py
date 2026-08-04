#!/usr/bin/env python
"""Minimal flow-matching training script (PLAN.md Stage 2.3, rectified-flow pivot).

Structure copied from act/train_act.py; builds FlowMatchingPolicy (act/modeling_flow.py)
instead of ACTPolicy. Config per plan: chunk 50, n_action_steps 15, temporal ensembling
OFF, hidden 512, no CVAE/KL, 10 Euler inference steps, AdamW. Default lr 1e-4 (FM /
pi0-style heads train with a higher lr than ACT's 1e-5). No eval loop here — batched
sim eval is a separate script (act/eval_act.py dispatches on policy_type == "flow").

Usage:
    python train_flow.py --data demos.hdf5 --out runs/flow_nominal --steps 100000
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
from slot_act.dataset import (
    ACTION_DIM,
    ENV_STATE_DIM,
    STATE_DIM,
    RebotDemoDataset,
    compute_stats,
    nominal_dagger_pool_filter,
    nominal_pool_filter,
    recovery_pool_filter,
    success_pool_filter,
)
from slot_act.modeling_flow import FlowMatchingPolicy
from slot_act.normalize import MeanStdNormalizer


def make_config(args: argparse.Namespace) -> ACTConfig:
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(ENV_STATE_DIM,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=None,  # OFF per plan (stale pre-event chunks suppress corrections)
        use_vae=False,  # no CVAE — the flow noise x0 is the stochasticity source (PLAN 2.3)
        dim_model=512,
        device=args.device,
    )
    # Not a vendored ACTConfig field (configuration_act.py is untouched); the policy
    # reads it with getattr(config, "num_inference_steps", 10).
    config.num_inference_steps = args.num_inference_steps
    return config


def save_checkpoint(
    path: Path, policy: FlowMatchingPolicy, normalizer: MeanStdNormalizer, config: ACTConfig, step: int
):
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "config": {
                "policy_type": "flow",  # eval_act.py dispatches on this
                "chunk_size": config.chunk_size,
                "n_action_steps": config.n_action_steps,
                "num_inference_steps": config.num_inference_steps,
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
    p.add_argument("--pool", choices=["default", "success", "nominal", "recovery", "nominal+dagger"],
                   default="default",
                   help="demo filter: success = every successful demo of any kind (use this for "
                        "the S3 arm comparison; 'nominal' would drop arm B's whole DART half); "
                        "nominal = first-attempt-clean successes only; "
                        "nominal+dagger adds successful recovery_dagger takeovers")
    p.add_argument("--out", required=True, help="output dir (checkpoints + log)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--n-action-steps", type=int, default=15)
    p.add_argument("--num-inference-steps", type=int, default=10,
                   help="Euler steps for flow sampling at inference (stored in the checkpoint config)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None,
                   help="seed python/numpy/torch RNGs (replica control); default: unseeded")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    pool_filter = {"default": None, "success": success_pool_filter, "nominal": nominal_pool_filter,
                   "recovery": recovery_pool_filter,
                   "nominal+dagger": nominal_dagger_pool_filter}[args.pool]
    dataset = RebotDemoDataset(args.data, chunk_size=args.chunk_size, demo_filter=pool_filter)
    stats = compute_stats(dataset)
    normalizer = MeanStdNormalizer(stats).to(device)
    kinds: dict[str, int] = {}
    for d in dataset.demos:
        k = str(d["attrs"].get("episode_kind", "?"))
        kinds[k] = kinds.get(k, 0) + 1
    censored = sum(int((d["train_mask"] == 0).any()) for d in dataset.demos)
    print(f"dataset: {len(dataset.demos)} demos, {len(dataset)} samples, pool={args.pool}, "
          f"kinds={kinds}, {censored} demos carry a loss censor")

    config = make_config(args)
    policy = FlowMatchingPolicy(config).to(device)
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
