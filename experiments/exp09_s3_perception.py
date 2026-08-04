#!/usr/bin/env python
"""EXP09 smoke test S3: can obs41[16:34] be predicted from pixels + proprio?

Targets (18-D, the exact privileged slice): objects_canonical 16 (target-first
object poses in ROBOT-ROOT frame + placed flags) + basket_center_xy 2. Inputs:
wrist/workspace RGB 160x90 + 23-D proprio. Data: Gate B BC shards (per-step
images + obs41). Train seed42, validate cross-seed on seed123 (caveat logged in
EXP09 doc: both are training-stream seeds).

Report: per-group RMSE + p95 abs error on val (pos groups in meters). PASS
sanity bar (EXP09 doc section 4): target rel-pos RMSE <~ 2 cm, basket <~ 3 cm.

    python experiments/exp09_s3_perception.py --out-dir runs/exp09/s3_perception
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TARGET_SLICE = slice(16, 34)
GROUPS = {  # name -> dims within the 18-D target
    "tgt_pos": slice(0, 3), "tgt_quat": slice(3, 7),
    "other_pos": slice(7, 10), "other_quat": slice(10, 14),
    "placed_flags": slice(14, 16), "basket_xy": slice(16, 18),
}


class PerceptionHead(nn.Module):
    """Shared resnet18 trunk (ImageNet init) over both cameras + proprio MLP -> 18-D."""

    def __init__(self):
        super().__init__()
        trunk = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        trunk.fc = nn.Identity()
        self.trunk = trunk
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.proprio = nn.Sequential(nn.Linear(23, 128), nn.ReLU(), nn.Linear(128, 128))
        self.head = nn.Sequential(
            nn.Linear(512 * 2 + 128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 18),
        )

    def forward(self, wrist, work, proprio):
        feats = [self.trunk((im - self.im_mean) / self.im_std) for im in (wrist, work)]
        return self.head(torch.cat(feats + [self.proprio(proprio)], dim=1))


def load_split(data_dir: Path):
    wrist, work, proprio, target = [], [], [], []
    for p in sorted(data_dir.glob("ep_*.pt")):
        sh = torch.load(p, map_location="cpu")
        wrist.append(sh["wrist_rgb"])
        work.append(sh["workspace_rgb"])
        proprio.append(sh["proprio"])
        target.append(sh["obs41"][:, TARGET_SLICE])
    return (torch.cat(wrist), torch.cat(work),
            torch.cat(proprio).float(), torch.cat(target).float())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", default="data/exp08_vision/seed42")
    p.add_argument("--val-data", default="data/exp08_vision/seed123")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/exp09/s3_perception")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    root = Path(__file__).resolve().parent.parent
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = load_split(root / args.train_data)
    va = load_split(root / args.val_data)
    n_tr, n_va = tr[3].shape[0], va[3].shape[0]
    print(f"[s3] train {n_tr} samples, val {n_va}", flush=True)

    model = PerceptionHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch_from(split, idx):
        w, k, pr, tg = split
        return (w[idx].permute(0, 3, 1, 2).float().to(device) / 255.0,
                k[idx].permute(0, 3, 1, 2).float().to(device) / 255.0,
                pr[idx].to(device), tg[idx].to(device))

    log = open(out_dir / "train_log.jsonl", "a")
    model.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_tr, (args.batch_size,))
        w, k, pr, tg = batch_from(tr, idx)
        loss = nn.functional.mse_loss(model(w, k, pr), tg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "loss": loss.item()}), file=log, flush=True)
            print(f"[s3] step {step} loss {loss.item():.5f}", flush=True)

    # Validation: full pass, per-group metrics.
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, n_va, 256):
            idx = torch.arange(i, min(i + 256, n_va))
            w, k, pr, tg = batch_from(va, idx)
            errs.append((model(w, k, pr) - tg).abs().cpu())
    errs = torch.cat(errs)  # (n_va, 18)

    metrics = {}
    for name, sl in GROUPS.items():
        e = errs[:, sl]
        vec = e.norm(dim=1) if "pos" in name or "xy" in name else e.mean(dim=1)
        metrics[name] = {"rmse": (e ** 2).mean().sqrt().item(),
                         "vec_mean": vec.mean().item(),
                         "vec_p95": vec.quantile(0.95).item()}
    result = {"train": str(args.train_data), "val": str(args.val_data),
              "n_train": n_tr, "n_val": n_va, "steps": args.steps, "metrics": metrics}
    (out_dir / "s3_result.json").write_text(json.dumps(result, indent=2))
    torch.save({"model_state_dict": model.state_dict(), "target_slice": [16, 34]},
               out_dir / "ckpt_final.pt")
    print("[s3] " + json.dumps(metrics, indent=2), flush=True)
    print("S3 DONE", flush=True)


if __name__ == "__main__":
    main()
