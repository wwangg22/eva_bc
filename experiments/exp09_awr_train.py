#!/usr/bin/env python
"""EXP09 R3' AWR trainer: fit SteerHead (frozen v4 encoder tokens -> 7-D z) by
advantage-weighted regression on exp09_awr_collect.py shards.

Per-window advantage A = w1*(placed_after - placed_before) + w2*znorm(win_reward);
weights = exp(A/beta) clipped to [0.1, 10], beta = std(A), then normalized to
mean 1. Weighted MSE to the recorded PRE-tanh z (alpha is applied in the
collector). The collector imports SteerHead from here for --head-ckpt runs —
keep the class name and (n_tokens, dim) signature.

    python experiments/exp09_awr_train.py \
        --data data/exp09_awr/it0b_a025_seed3001 data/exp09_awr/it0b_a025_seed3002 \
        --out-dir runs/exp09/awr_it1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.eval_flow_vision import VisionController, load_vision_checkpoint


class SteerHead(nn.Module):
    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_tokens * dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, feats):  # (B, S, D)
        return self.net(feats.flatten(1))


@torch.no_grad()
def precompute(policy, normalizer, data_dirs, device):
    feats, zs, placed_delta, win_reward = [], [], [], []
    for data_dir in data_dirs:
        for p in sorted(Path(data_dir).glob("ep_*.pt")):
            sh = torch.load(p, map_location="cpu")
            T = sh["z"].shape[0]
            for i in range(0, T, 256):
                sl = slice(i, min(i + 256, T))
                batch = normalizer.normalize({
                    "observation.state": sh["proprio"][sl].float().to(device),
                    "observation.images.wrist": sh["wrist_rgb"][sl].permute(0, 3, 1, 2).float().to(device) / 255.0,
                    "observation.images.workspace": sh["workspace_rgb"][sl].permute(0, 3, 1, 2).float().to(device) / 255.0,
                })
                out, _ = policy.model.encode(policy._stack_images(batch))  # (S, B, D)
                feats.append(out.permute(1, 0, 2).cpu())
            zs.append(sh["z"].float())
            placed_delta.append((sh["placed_after"] - sh["placed_before"]).float())
            win_reward.append(sh["win_reward"].float())
    return (torch.cat(feats), torch.cat(zs),
            torch.cat(placed_delta), torch.cat(win_reward))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vision-ckpt", default="runs/exp08_bc/v4_dagger3/ckpt_final.pt")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--w-placed", type=float, default=1.0)
    p.add_argument("--w-reward", type=float, default=0.3)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, stats, cfg = load_vision_checkpoint(root / args.vision_ckpt, device)
    ctrl = VisionController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)

    print("[awr-train] precomputing frozen encoder tokens...", flush=True)
    feats, zs, placed_delta, win_reward = precompute(
        policy, ctrl.normalizer, [root / d for d in args.data], device)
    N, S, D = feats.shape[0], feats.shape[1], feats.shape[2]

    adv = args.w_placed * placed_delta \
        + args.w_reward * (win_reward - win_reward.mean()) / win_reward.std().clamp_min(1e-6)
    beta = adv.std().clamp_min(1e-6)
    w = torch.exp(adv / beta).clamp(0.1, 10.0)
    ess = (w.sum() ** 2 / (w ** 2).sum()).item()
    w = w / w.mean()
    print(f"[awr-train] {N} windows, tokens {S}x{D}, placed-delta windows "
          f"{(placed_delta != 0).sum().item()}, adv std {beta.item():.4f}, ESS {ess:.0f}", flush=True)

    perm = torch.randperm(N)
    n_val = max(1, int(N * args.val_frac))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    head = SteerHead(S, D).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    log = open(out_dir / "train_log.jsonl", "a")
    head.train()
    for step in range(1, args.steps + 1):
        idx = tr_idx[torch.randint(0, tr_idx.shape[0], (args.batch_size,))]
        pred = head(feats[idx].to(device))
        per = ((pred - zs[idx].to(device)) ** 2).mean(dim=1)
        loss = (w[idx].to(device) * per).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            print(json.dumps({"step": step, "wloss": loss.item(),
                              "uloss": per.mean().item()}), file=log, flush=True)
            print(f"[awr-train] step {step} wloss {loss.item():.4f} uloss {per.mean().item():.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        preds = torch.cat([head(feats[va_idx[i:i + 1024]].to(device)).cpu()
                           for i in range(0, n_val, 1024)])
        val_mse = ((preds - zs[va_idx]) ** 2).mean().item()
        z_std = preds.std(dim=0)
    result = {"vision_ckpt": args.vision_ckpt, "data": args.data, "n_windows": N,
              "tokens": [S, D], "steps": args.steps,
              "w_placed": args.w_placed, "w_reward": args.w_reward,
              "adv_std": beta.item(), "ess": ess,
              "n_placed_delta_windows": int((placed_delta != 0).sum()),
              "val_mse": val_mse, "head_z_std": z_std.tolist(),
              "head_z_std_mean": z_std.mean().item()}
    (out_dir / "awr_train_result.json").write_text(json.dumps(result, indent=2))
    torch.save({"head_state_dict": head.state_dict(), "tokens": [S, D],
                "vision_ckpt": args.vision_ckpt, "data": args.data},
               out_dir / "head.pt")
    print("[awr-train] " + json.dumps(result), flush=True)
    print("AWR TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
