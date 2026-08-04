#!/usr/bin/env python
"""EXP09 smoke test S1: does the distilled VISION flow base respond to x0?

EXP07 x0-steering works because the state base's flow maps x0 regions to distinct
action modes. The vision base was distilled from already-steered champion output,
so its x0-response may be baked out. This probe measures chunk spread under an
identical set of x0 draws for both bases on the SAME physical states (Gate B BC
shards carry images + obs41 per step).

Metric (action space, unnormalized): for each state and base, over K x0 draws,
mean over draw pairs of the mean-over-executed-steps L2 between chunks. PASS
(pre-registered, EXP09 doc section 4): vision spread >= ~50% of state spread.

Offline; no sim; C2-safe. Run only when the GPU is otherwise free:
    python experiments/exp09_s1_x0probe.py --out runs/exp09/s1_x0probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset import ENV_STATE_SLICE, STATE_SLICE
from act.eval_act import BatchedACTController, load_checkpoint
from act.eval_flow_vision import VisionController, load_vision_checkpoint

EXEC_STEPS = 15


@torch.no_grad()
def chunk_spread(policy, normalizer, batch: dict, x0s: torch.Tensor, unnorm) -> torch.Tensor:
    """(K draws) -> per-state mean pairwise L2 over the executed chunk prefix."""
    chunks = []
    for k in range(x0s.shape[0]):
        c = policy.predict_action_chunk(normalizer.normalize(dict(batch)), x0=x0s[k])
        chunks.append(unnorm("action", c)[:, :EXEC_STEPS])
    chunks = torch.stack(chunks)  # (K, B, EXEC, 7)
    K = chunks.shape[0]
    dists = [
        (chunks[i] - chunks[j]).norm(dim=-1).mean(dim=-1)
        for i in range(K)
        for j in range(i + 1, K)
    ]
    return torch.stack(dists).mean(dim=0)  # (B,)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vision-ckpt", default="runs/exp08_bc/v4_dagger3/ckpt_final.pt")
    p.add_argument("--state-ckpt", default="runs/exp03_N3/ckpt_final.pt")
    p.add_argument("--data", default="data/exp08_vision/seed42")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--stride", type=int, default=45, help="sample every Nth step per episode")
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp09/s1_x0probe.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(__file__).resolve().parent.parent

    v_policy, v_stats, v_cfg = load_vision_checkpoint(root / args.vision_ckpt, device)
    v_ctrl = VisionController(v_policy, v_stats, v_cfg["n_action_steps"], v_cfg["chunk_size"], device)
    s_policy, s_stats, s_cfg = load_checkpoint(root / args.state_ckpt, device)
    s_ctrl = BatchedACTController(s_policy, s_stats, s_cfg["n_action_steps"], s_cfg["chunk_size"], device)
    chunk_size = v_cfg["chunk_size"]
    assert chunk_size == s_cfg["chunk_size"], (v_cfg, s_cfg)

    # Sample matched states.
    wrist, work, proprio, obs41 = [], [], [], []
    shards = sorted((root / args.data).glob("ep_*.pt"))[: args.episodes]
    for sh_path in shards:
        sh = torch.load(sh_path, map_location="cpu")
        idx = torch.arange(0, sh["obs41"].shape[0], args.stride)
        wrist.append(sh["wrist_rgb"][idx])
        work.append(sh["workspace_rgb"][idx])
        proprio.append(sh["proprio"][idx])
        obs41.append(sh["obs41"][idx])
    wrist, work = torch.cat(wrist), torch.cat(work)
    proprio, obs41 = torch.cat(proprio), torch.cat(obs41)
    B = obs41.shape[0]
    print(f"[s1] {B} states from {len(shards)} episodes, {args.draws} x0 draws")

    v_batch = {
        "observation.state": proprio.float().to(device),
        "observation.images.wrist": wrist.permute(0, 3, 1, 2).float().to(device) / 255.0,
        "observation.images.workspace": work.permute(0, 3, 1, 2).float().to(device) / 255.0,
    }
    s_batch = {
        "observation.state": obs41[:, STATE_SLICE].float().to(device),
        "observation.environment_state": obs41[:, ENV_STATE_SLICE].float().to(device),
    }

    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn(args.draws, 7, generator=g)
    x0s = torch.tanh(z).unsqueeze(1).expand(-1, chunk_size, -1).to(device)  # (K, chunk, 7)

    v_spread = chunk_spread(v_policy, v_ctrl.normalizer, v_batch, x0s, v_ctrl.normalizer.unnormalize)
    s_spread = chunk_spread(s_policy, s_ctrl.normalizer, s_batch, x0s, s_ctrl.normalizer.unnormalize)
    ratio = v_spread / s_spread.clamp_min(1e-8)

    result = {
        "vision_ckpt": args.vision_ckpt,
        "state_ckpt": args.state_ckpt,
        "n_states": B,
        "draws": args.draws,
        "vision_spread_mean": v_spread.mean().item(),
        "vision_spread_median": v_spread.median().item(),
        "state_spread_mean": s_spread.mean().item(),
        "state_spread_median": s_spread.median().item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_median": ratio.median().item(),
        "ratio_p10": ratio.quantile(0.1).item(),
        "pass_rule": "vision spread >= 0.5 * state spread (median ratio)",
        "pass": ratio.median().item() >= 0.5,
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(f"[s1] {k}: {v}")


if __name__ == "__main__":
    main()
