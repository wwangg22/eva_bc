#!/usr/bin/env python
"""EXP06 gate 1 — validated grasp-success bit for the residual policy's obs.

Candidates (features = EXP01 variant D: finger pos 6,7 + vel 14,15 + last-grip 40):
  rule : hand threshold on aperture, gated by commanded-closed (grid-searched on train)
  mlp  : the EXP01 D-probe itself (2x128 MLP), exported for runtime use

Gate (EXP06_residual_rl.md): 0% FPR on the 665 on-policy closed-on-air freeze states
AND >=95% accuracy on expert post-close test frames. Same episode-level split and
loaders as EXP01. Winner exported to exp06_grasp_bit.pt; runtime semantics:
bit = predictor(5 dims) AND last-grip-commanded-closed (open command => bit 0).

CPU-only. Writes exp06_grasp_bit_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from exp01_probe import (FINGER_DIMS, SEED, auc, load_expert_samples,
                         load_onpolicy_negatives, train_probe)

HERE = Path(__file__).resolve().parent


def rule_scores(X, theta):
    """X: (N,5) raw [pos6, pos7, vel14, vel15, grip40]. Score = aperture margin above
    theta, valid only when commanded closed; open command scores 'not grasped'."""
    ap = X[:, 0] + X[:, 1]
    closed = X[:, 4] < 0
    return np.where(closed, ap - theta, -1e9)


def main():
    rng = np.random.default_rng(SEED)
    expert = load_expert_samples()
    onpol = load_onpolicy_negatives()
    labels = np.array([s[2] for s in expert])
    uids = sorted({s[4] for s in expert})
    rng.shuffle(uids)
    test_uids = set(uids[: len(uids) // 5])
    is_te = np.array([s[4] in test_uids for s in expert])

    X = np.stack([s[0] for s in expert])[:, 0, FINGER_DIMS]   # (N,5) current frame
    Xo = np.stack([s[0] for s in onpol])[:, 0, FINGER_DIMS]
    closed_frac_expert = float((X[:, 4] < 0).mean())
    closed_frac_onpol = float((Xo[:, 4] < 0).mean())
    print(f"expert {len(X)} (pos {int(labels.sum())}), onpol {len(Xo)}; "
          f"commanded-closed frac: expert {closed_frac_expert:.3f} onpol {closed_frac_onpol:.3f}")
    print(f"aperture (pos6+pos7) grasped: {np.percentile(X[labels==1,0]+X[labels==1,1], [5,50,95])}")
    print(f"aperture missed:  {np.percentile(X[labels==0,0]+X[labels==0,1], [5,50,95])}")
    print(f"aperture onpol-freeze: {np.percentile(Xo[:,0]+Xo[:,1], [5,50,95])}")

    results = {}

    # --- candidate 1: threshold rule, theta grid-searched on TRAIN only ---
    thetas = np.linspace((X[:, 0] + X[:, 1]).min(), (X[:, 0] + X[:, 1]).max(), 400)
    best = None
    for th in thetas:
        s = rule_scores(X[~is_te], th)
        acc = float(((s > 0) == (labels[~is_te] == 1)).mean())
        if best is None or acc > best[1]:
            best = (float(th), acc)
    theta, _ = best
    ste = rule_scores(X[is_te], theta)
    so = rule_scores(Xo, theta)
    results["rule"] = {
        "theta": theta,
        "test_acc": float(((ste > 0) == (labels[is_te] == 1)).mean()),
        "test_tpr": float((ste[labels[is_te] == 1] > 0).mean()),
        "test_auc": auc(rule_scores(X[is_te], theta), labels[is_te]),
        "onpolicy_fpr": float((so > 0).mean()),
    }

    # --- candidate 2: the D-probe MLP (identical to EXP01) ---
    model, (mu, sd), str_, ste_m = train_probe(X[~is_te], labels[~is_te], X[is_te], labels[is_te])
    with torch.no_grad():
        so_m = model(torch.tensor((Xo - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
    # runtime semantics: open command forces bit 0
    pred_te = (ste_m > 0) & (X[is_te, 4] < 0)
    pred_o = (so_m > 0) & (Xo[:, 4] < 0)
    results["mlp"] = {
        "test_acc": float((pred_te == (labels[is_te] == 1)).mean()),
        "test_tpr": float(pred_te[labels[is_te] == 1].mean()),
        "test_auc": auc(ste_m, labels[is_te]),
        "onpolicy_fpr": float(pred_o.mean()),
    }

    for k, v in results.items():
        gate = v["onpolicy_fpr"] == 0.0 and v["test_acc"] >= 0.95
        v["passes_gate"] = bool(gate)
        print(k, json.dumps(v, default=float))

    # --- export winner (prefer rule if it passes; else mlp) retrained on ALL data ---
    if results["rule"]["passes_gate"]:
        artifact = {"kind": "rule", "theta": theta, "dims": FINGER_DIMS,
                    "semantics": "bit = (grip40<0) and (pos6+pos7 - theta > 0)"}
    else:
        model_all, (mu_a, sd_a), _, _ = train_probe(X, labels, X[:1], labels[:1])
        with torch.no_grad():
            so_all = model_all(torch.tensor((Xo - mu_a) / sd_a, dtype=torch.float32)).squeeze(-1).numpy()
        results["mlp_all_data_onpolicy_fpr"] = float(((so_all > 0) & (Xo[:, 4] < 0)).mean())
        artifact = {"kind": "mlp", "state_dict": model_all.state_dict(),
                    "mu": torch.tensor(mu_a, dtype=torch.float32),
                    "sd": torch.tensor(sd_a, dtype=torch.float32), "dims": FINGER_DIMS,
                    "semantics": "bit = (grip40<0) and (sigmoid(mlp((x-mu)/sd))>0.5)"}
    torch.save(artifact, HERE / "exp06_grasp_bit.pt")
    (HERE / "exp06_grasp_bit_results.json").write_text(json.dumps(results, indent=2, default=float))
    print("exported", artifact["kind"], "-> exp06_grasp_bit.pt")


if __name__ == "__main__":
    main()
