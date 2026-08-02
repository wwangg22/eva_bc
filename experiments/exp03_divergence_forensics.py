#!/usr/bin/env python
"""Forensics: why doesn't exp03_grip_divergence.py reproduce POSTMORTEM §5?

Original analysis (recovered from session transcript) vs the rewritten tool differ in:
  (a) normalization: original normalized BOTH policies' obs with v1's stats
      (v3 trained/evals with its own pool stats — mis-normalized inputs for v3);
  (b) state selection: original included lift/transport frames of MISSED attempts
      (empty closed-gripper lifts — aliased states where v3 opening is learned
      recovery, not interference);
  (c) data: original used only s101, first 30 h5 keys, stride 5.

Conditions (each reports v1 open-rate, v3 open-rate, v1-holds-v3-opens):
  i    grasped-only holds, own stats            (the corrected method)
  ii   + missed-attempt lifts, own stats        (isolates state conflation)
  iii  grasped-only holds, v1 stats for both    (isolates normalization bug)
  iv   exact replication: s101[:30], stride 5, no filter, v1 stats, gen seed 7
       (should reproduce 0.121 / 0.203 / 0.087)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from act.dataset import STATE_SLICE, ENV_STATE_SLICE  # noqa: E402
from act.eval_act import load_checkpoint  # noqa: E402

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "runs/flow_nominal_v1/ckpt_final.pt"
V3 = HERE.parent / "runs/flow_dagger_v3/ckpt_final.pt"
HORIZON = 15


def norm(x, st):
    return (x - st["mean"]) / (st["std"] + 1e-8)


def collect(files, include_missed, stride, max_keys=None, missed_only=False):
    states = []
    for f in files:
        with h5py.File(f) as h:
            data = h["data"]
            keys = list(data.keys())[:max_keys] if max_keys else list(data.keys())
            for k in keys:
                g = data[k]
                obs = None
                segs = json.loads(g.attrs["segments"])
                outcomes = json.loads(g.attrs["outcomes"])
                for j, s in enumerate(segs):
                    if s["phase"] not in ("lift", "transport"):
                        continue
                    grasped = (s["seg"] is not None and
                               outcomes.get(s["seg"], {}).get("outcome") in ("grasped", "delivered"))
                    if missed_only and grasped:
                        continue
                    if not include_missed and not missed_only and not grasped:
                        continue
                    if obs is None:
                        obs = g["obs"]["policy"][:]
                    t1 = segs[j + 1]["t"] if j + 1 < len(segs) else obs.shape[0]
                    for t in range(s["t"], min(t1, obs.shape[0]), stride):
                        states.append(obs[t])
    return torch.tensor(np.stack(states), dtype=torch.float32)


def open_rates(S, p1, s1, p3, s3, stats_for_v3):
    b1 = {"observation.state": norm(S[:, STATE_SLICE], s1["observation.state"]),
          "observation.environment_state": norm(S[:, ENV_STATE_SLICE], s1["observation.environment_state"])}
    sv3 = stats_for_v3
    b3 = {"observation.state": norm(S[:, STATE_SLICE], sv3["observation.state"]),
          "observation.environment_state": norm(S[:, ENV_STATE_SLICE], sv3["observation.environment_state"])}
    gen = torch.Generator()
    with torch.no_grad():
        gen.manual_seed(7)
        a1 = p1.predict_action_chunk(b1, generator=gen)
        gen.manual_seed(7)
        a3 = p3.predict_action_chunk(b3, generator=gen)
    u1 = a1 * (s1["action"]["std"] + 1e-8) + s1["action"]["mean"]
    u3 = a3 * (s3["action"]["std"] + 1e-8) + s3["action"]["mean"]
    g1 = (u1[:, :HORIZON, 6] > 0).any(dim=1)
    g3 = (u3[:, :HORIZON, 6] > 0).any(dim=1)
    return float(g1.float().mean()), float(g3.float().mean()), float((~g1 & g3).float().mean())


def main():
    p1, s1, _ = load_checkpoint(V1, "cpu")
    p3, s3, _ = load_checkpoint(V3, "cpu")
    p1.eval(); p3.eval()
    all_files = sorted((HERE.parent / "expert").glob("demos_nominal_s10*.h5"))
    s101 = [HERE.parent / "expert/demos_nominal_s101.h5"]
    rng = np.random.default_rng(0)

    def cap(S, n=1500):
        if len(S) > n:
            S = S[np.sort(rng.choice(len(S), n, replace=False))]
        return S

    conds = {
        "i_grasped_ownstats": (cap(collect(all_files, False, 3)), s3),
        "ii_withmissed_ownstats": (cap(collect(all_files, True, 3)), s3),
        "iii_grasped_v1stats": (cap(collect(all_files, False, 3)), s1),
        "iv_exact_replication": (collect(s101, True, 5, max_keys=30), s1),
        "v_missed_only_ownstats": (cap(collect(all_files, True, 3, missed_only=True)), s3),
    }
    results = {}
    for name, (S, stats_v3) in conds.items():
        r = open_rates(S, p1, s1, p3, s3, stats_v3)
        results[name] = {"n": len(S), "v1_open": r[0], "v3_open": r[1], "v1holds_v3opens": r[2]}
        print(f"{name:24s} n={len(S):5d}  v1={r[0]:.3f}  v3={r[1]:.3f}  v1-holds-v3-opens={r[2]:.3f}")

    (HERE / "exp03_divergence_forensics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
