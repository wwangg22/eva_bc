#!/usr/bin/env python
"""EXP09 S3v2: perception head on the FROZEN v4 student's encoder tokens.

S3v1 (fresh resnet18 + global-avg-pool) missed the bar (tgt-pos 5.5 cm mean).
v2 rides the v4 vision transformer's own 31 task-trained encoder tokens
(S=31, D=512), flattened into a small MLP -> 18-D obs41[16:34]. BCE on the two
placed flags, MSE elsewhere. Features are precomputed once (encoder frozen), so
head training is minutes. Doubles as validation of the integrated-forward
deploy design (steering rides the same encoder pass).

    python experiments/exp09_s3v2_perception.py --out-dir runs/exp09/s3v2_perception
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

TARGET_SLICE = slice(16, 34)
FLAG_DIMS = slice(14, 16)  # within the 18-D target
GROUPS = {
    "tgt_pos": slice(0, 3), "tgt_quat": slice(3, 7),
    "other_pos": slice(7, 10), "other_quat": slice(10, 14),
    "placed_flags": slice(14, 16), "basket_xy": slice(16, 18),
}


class TokenHead(nn.Module):
    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_tokens * dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 18),
        )

    def forward(self, feats):  # (B, S, D)
        return self.net(feats.flatten(1))


@torch.no_grad()
def precompute(policy, normalizer, data_dir: Path, device) -> tuple[torch.Tensor, torch.Tensor]:
    feats, targets = [], []
    for p in sorted(data_dir.glob("ep_*.pt")):
        sh = torch.load(p, map_location="cpu")
        T = sh["obs41"].shape[0]
        for i in range(0, T, 256):
            sl = slice(i, min(i + 256, T))
            batch = normalizer.normalize({
                "observation.state": sh["proprio"][sl].float().to(device),
                "observation.images.wrist": sh["wrist_rgb"][sl].permute(0, 3, 1, 2).float().to(device) / 255.0,
                "observation.images.workspace": sh["workspace_rgb"][sl].permute(0, 3, 1, 2).float().to(device) / 255.0,
            })
            out, _ = policy.model.encode(policy._stack_images(batch))  # (S, B, D)
            feats.append(out.permute(1, 0, 2).cpu())
        targets.append(sh["obs41"][:, TARGET_SLICE].float())
    return torch.cat(feats), torch.cat(targets)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vision-ckpt", default="runs/exp08_bc/v4_dagger3/ckpt_final.pt")
    p.add_argument("--train-data", default="data/exp08_vision/seed42")
    p.add_argument("--val-data", default="data/exp08_vision/seed123")
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/exp09/s3v2_perception")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, stats, cfg = load_vision_checkpoint(root / args.vision_ckpt, device)
    ctrl = VisionController(policy, stats, cfg["n_action_steps"], cfg["chunk_size"], device)

    print("[s3v2] precomputing frozen encoder features...", flush=True)
    tr_f, tr_t = precompute(policy, ctrl.normalizer, root / args.train_data, device)
    va_f, va_t = precompute(policy, ctrl.normalizer, root / args.val_data, device)
    n_tr, n_va = tr_t.shape[0], va_t.shape[0]
    S, D = tr_f.shape[1], tr_f.shape[2]
    print(f"[s3v2] train {n_tr}, val {n_va}, tokens {S}x{D}", flush=True)

    head = TokenHead(S, D).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()

    log = open(out_dir / "train_log.jsonl", "a")
    head.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_tr, (args.batch_size,))
        pred = head(tr_f[idx].to(device))
        tg = tr_t[idx].to(device)
        loss = nn.functional.mse_loss(pred[:, :14], tg[:, :14]) \
            + nn.functional.mse_loss(pred[:, 16:], tg[:, 16:]) \
            + 0.2 * bce(pred[:, FLAG_DIMS], tg[:, FLAG_DIMS])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            print(json.dumps({"step": step, "loss": loss.item()}), file=log, flush=True)
            print(f"[s3v2] step {step} loss {loss.item():.5f}", flush=True)

    head.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, n_va, 1024):
            sl = slice(i, min(i + 1024, n_va))
            pred = head(va_f[sl].to(device))
            pred[:, FLAG_DIMS] = torch.sigmoid(pred[:, FLAG_DIMS])
            errs.append((pred.cpu() - va_t[sl]).abs())
    errs = torch.cat(errs)

    metrics = {}
    for name, sl in GROUPS.items():
        e = errs[:, sl]
        vec = e.norm(dim=1) if "pos" in name or "xy" in name else e.mean(dim=1)
        metrics[name] = {"rmse": (e ** 2).mean().sqrt().item(),
                         "vec_mean": vec.mean().item(),
                         "vec_p95": vec.quantile(0.95).item()}
    result = {"vision_ckpt": args.vision_ckpt, "n_train": n_tr, "n_val": n_va,
              "steps": args.steps, "tokens": [S, D], "metrics": metrics}
    (out_dir / "s3v2_result.json").write_text(json.dumps(result, indent=2))
    torch.save({"head_state_dict": head.state_dict(), "tokens": [S, D],
                "vision_ckpt": args.vision_ckpt, "target_slice": [16, 34]},
               out_dir / "ckpt_final.pt")
    print("[s3v2] " + json.dumps(metrics, indent=2), flush=True)
    print("S3V2 DONE", flush=True)


if __name__ == "__main__":
    main()
