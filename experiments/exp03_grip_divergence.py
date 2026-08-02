#!/usr/bin/env python
"""EXP03 — offline grip/arm divergence between flow checkpoints on identical nominal
hold states (POSTMORTEM §5 methodology, made reusable for replicas).

States: expert demo frames in lift/transport phases of GRASPED attempts (true hold
states, expert grip closed). Each policy predicts a chunk from the SAME normalized obs
with the SAME x0 (shared generator seed) — differences are purely weights.

Metrics per checkpoint (vs the first, the reference):
  - mean |Δ arm| and |Δ grip| over the executed 15-step horizon, per phase
  - open-rate: fraction of hold states where the unnormalized grip channel goes > 0
    (commanded open) anywhere in the executed 15 steps
  - divergent-open rate: reference holds closed, this ckpt opens

Usage: python exp03_grip_divergence.py CKPT_REF CKPT [CKPT ...] [--device cuda]
Writes exp03_divergence.json next to this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.dataset import STATE_SLICE, ENV_STATE_SLICE  # noqa: E402
from act.eval_act import load_checkpoint  # noqa: E402
from act.normalize import MeanStdNormalizer  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPERT_FILES = sorted((HERE.parent / "expert").glob("demos_nominal_s10*.h5"))
HORIZON = 15
X0_SEED = 1234
MAX_STATES = 1500
STRIDE = 3


def collect_hold_states():
    obs_list, phase_list = [], []
    for f in EXPERT_FILES:
        with h5py.File(f) as h:
            root = h["data"] if "data" in h else h
            for name in root.keys():
                ep = root[name]
                segs = json.loads(ep.attrs["segments"])
                outcomes = json.loads(ep.attrs["outcomes"])
                obs = None
                for i, s in enumerate(segs):
                    grasped = (s["seg"] is not None
                               and outcomes.get(s["seg"], {}).get("outcome") in ("grasped", "delivered"))
                    if s["phase"] not in ("lift", "transport") or not grasped:
                        continue
                    if obs is None:
                        obs = ep["obs/policy"][:]
                    t0 = s["t"]
                    t1 = segs[i + 1]["t"] if i + 1 < len(segs) else len(obs)
                    for t in range(t0, min(t1, len(obs)) - 1, STRIDE):
                        obs_list.append(obs[t])
                        phase_list.append(s["phase"])
    return np.stack(obs_list), np.array(phase_list)


def predict(policy, normalizer, obs, device, batch=256):
    """Predict UNNORMALIZED chunks for (N, 41) raw obs, fixed x0 seed per batch index."""
    chunks = []
    with torch.no_grad():
        for i in range(0, len(obs), batch):
            o = torch.tensor(obs[i:i + batch], dtype=torch.float32, device=device)
            b = {"observation.state": o[:, STATE_SLICE],
                 "observation.environment_state": o[:, ENV_STATE_SLICE]}
            b = normalizer.normalize(b)
            gen = torch.Generator(device=device).manual_seed(X0_SEED + i)
            chunk = policy.predict_action_chunk(b, generator=gen)[:, :HORIZON]
            chunk = normalizer.unnormalize("action", chunk)
            chunks.append(chunk.cpu())
    return torch.cat(chunks).numpy()  # (N, HORIZON, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="first = reference")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    obs, phases = collect_hold_states()
    if len(obs) > MAX_STATES:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(obs), MAX_STATES, replace=False))
        obs, phases = obs[idx], phases[idx]
    print(f"{len(obs)} hold states "
          f"({(phases == 'lift').sum()} lift / {(phases == 'transport').sum()} transport)")

    preds = {}
    for c in args.ckpts:
        policy, stats, cfg = load_checkpoint(Path(c), device)
        normalizer = MeanStdNormalizer(stats).to(device)
        preds[c] = predict(policy, normalizer, obs, device)
        open_rate = float((preds[c][:, :, 6] > 0).any(axis=1).mean())
        print(f"{c}: open-anywhere-in-{HORIZON} rate on hold states = {open_rate:.3f}")

    ref = args.ckpts[0]
    results = {"n_states": len(obs), "horizon": HORIZON, "reference": ref, "ckpts": {}}
    ref_open = (preds[ref][:, :, 6] > 0).any(axis=1)
    for c in args.ckpts:
        d = np.abs(preds[c] - preds[ref])  # (N, H, 7)
        this_open = (preds[c][:, :, 6] > 0).any(axis=1)
        entry = {
            "open_rate": float(this_open.mean()),
            "ref_holds_this_opens": float((~ref_open & this_open).mean()),
            "per_phase": {},
        }
        for ph in ("lift", "transport"):
            m = phases == ph
            entry["per_phase"][ph] = {
                "arm_mean_abs_delta": float(d[m][:, :, :6].mean()),
                "grip_mean_abs_delta": float(d[m][:, :, 6].mean()),
            }
        results["ckpts"][c] = entry
        if c != ref:
            print(f"{c} vs ref: lift arm|Δ|={entry['per_phase']['lift']['arm_mean_abs_delta']:.3f} "
                  f"grip|Δ|={entry['per_phase']['lift']['grip_mean_abs_delta']:.3f}  "
                  f"ref-holds-this-opens={entry['ref_holds_this_opens']:.3f}")

    out = HERE / "exp03_divergence.json"
    out.write_text(json.dumps(results, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
