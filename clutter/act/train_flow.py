#!/usr/bin/env python
"""Flow-matching BC for `Rebot-ClutterExtract-v0`.

Structurally identical to `eva_bc/act/train_flow.py`; the only differences are the dataset
module (42-D observation, `clutter/act/dataset.py`) and the pool filter, which is just
"successful" here -- clutter has no `episode_kind` and no per-can `outcomes`, so pick-place's
nominal / recovery / dagger pools have no analogue.

Everything else is deliberately unchanged, because every one of those numbers was paid for
upstream:

  * **rectified flow, not a CVAE.** `x_tau = (1-tau)*x0 + tau*x1`, target `v = x1 - x0`, 10
    Euler steps at inference. The flow noise `x0` is the stochasticity source, so `use_vae`
    is off.
  * **chunk 50, n_action_steps 15.** Chunk commitment is load-bearing, not a tunable: eva_bc
    measured 59.4 -> 32.8 -> 3.1 -> 0 -> 0 % at n_action_steps 15/8/4/2/1 (EXP02).
  * **temporal ensembling OFF.** Stale pre-event chunks suppress corrections.
  * **lr 1e-4**, an order above ACT's 1e-5: flow / pi0-style heads want it.

Usage
-----
    python -u clutter/act/train_flow.py --data runs/demos_v1.hdf5 --out runs/bc_s1 \
        --steps 100000 --seed 1
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from act.configuration_act import ACTConfig, FeatureType, PolicyFeature  # noqa: E402
from act.modeling_flow import FlowMatchingPolicy  # noqa: E402
from act.normalize import MeanStdNormalizer  # noqa: E402
from dataset import (  # noqa: E402
    ACTION_DIM,
    ENV_STATE_DIM,
    STATE_DIM,
    ClutterDemoDataset,
    compute_stats,
)


def make_config(args: argparse.Namespace) -> ACTConfig:
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV,
                                                           shape=(ENV_STATE_DIM,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=None,
        use_vae=False,
        dim_model=512,
        device=args.device,
    )
    # Not a vendored ACTConfig field; the policy reads it with getattr(..., 10).
    config.num_inference_steps = args.num_inference_steps
    return config


def save_checkpoint(path: Path, policy, normalizer, config, step: int):
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "normalizer_state_dict": normalizer.state_dict(),
            "config": {
                "policy_type": "flow",
                "chunk_size": config.chunk_size,
                "n_action_steps": config.n_action_steps,
                "num_inference_steps": config.num_inference_steps,
                "state_dim": STATE_DIM,
                "env_state_dim": ENV_STATE_DIM,
                "action_dim": ACTION_DIM,
                "task": "clutter",
            },
        },
        path,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, nargs="+", help="demo HDF5 file(s)")
    p.add_argument("--out", required=True, help="output dir (checkpoints + log)")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--n-action-steps", type=int, default=15,
                   help="NOT a tunable: eva_bc EXP02 measured 59.4/32.8/3.1/0/0 %% at 15/8/4/2/1")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = ClutterDemoDataset(args.data, chunk_size=args.chunk_size)
    stats = compute_stats(dataset)
    normalizer = MeanStdNormalizer(stats).to(device)
    lens = [d["obs"].shape[0] for d in dataset.demos]
    print(f"dataset: {len(dataset.demos)} demos kept, {dataset.n_rejected} rejected, "
          f"{len(dataset)} samples; episode length {min(lens)}-{max(lens)} env steps")

    config = make_config(args)
    policy = FlowMatchingPolicy(config).to(device)
    policy.train()
    n_par = sum(p_.numel() for p_ in policy.parameters())
    print(f"policy: FlowMatchingPolicy, {n_par / 1e3:.0f}k params, chunk {args.chunk_size}, "
          f"n_action_steps {args.n_action_steps}, {args.num_inference_steps} Euler steps")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True,
                        pin_memory=device.type == "cuda")

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
                                    config, step)
                if step >= args.steps:
                    break

    save_checkpoint(out_dir / "ckpt_final.pt", policy, normalizer, config, step)
    print(f"done: {step} steps -> {out_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
