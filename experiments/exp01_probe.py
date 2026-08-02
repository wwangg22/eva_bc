#!/usr/bin/env python
"""EXP01 — grasp-success probe: can the policy's own obs distinguish
closed-on-can from closed-on-air? See EXP01_grasp_aliasing_probe.md.

Labels come free from h5 attrs: per-grasp `outcomes` (grasped/missed) + `segments`
phase timeline. Post-close frames (during the lift phase that follows every close,
grasped or missed) are the aliased states of interest. DAgger policy-phase frozen
stretches before takeover_t are on-policy label-0 transfer samples (never trained on).

CPU-only. Outputs exp01_results.json next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
EXPERT_FILES = sorted((HERE.parent / "expert").glob("demos_nominal_s10*.h5"))
DAGGER_FILES = [HERE.parent / "expert" / "dagger_r1.h5",
                HERE.parent / "expert" / "dagger_r1_failures.h5"]

FINGER_DIMS = [6, 7, 14, 15, 40]  # finger pos, finger vel, last commanded grip
KS = [0, 2, 5, 10]  # frames after close-end at which we probe
SEED = 0


def load_expert_samples():
    """One sample per (episode, grasp attempt, k): (obs_history[4, 41] stride-1,
    obs_history_s5[4, 41] stride-5, label, k, ep_uid)."""
    samples = []
    for f in EXPERT_FILES:
        with h5py.File(f) as h:
            root = h["data"] if "data" in h else h
            for name in root.keys():
                ep = root[name]
                uid = f"{f.name}:{name}"
                obs = ep["obs/policy"][:]
                segs = json.loads(ep.attrs["segments"])
                outcomes = json.loads(ep.attrs["outcomes"])
                for i, s in enumerate(segs):
                    if s["phase"] != "close" or s["seg"] is None:
                        continue
                    out = outcomes.get(s["seg"], {}).get("outcome")
                    if out not in ("grasped", "missed"):
                        continue
                    label = 1 if out == "grasped" else 0
                    t_end = segs[i + 1]["t"] if i + 1 < len(segs) else len(obs)
                    # window = the phase after close (lift), before the next boundary
                    t_next = segs[i + 2]["t"] if i + 2 < len(segs) else len(obs)
                    for k in KS:
                        t = t_end + k
                        if t >= min(t_next, len(obs)):
                            continue
                        h1 = np.stack([obs[max(t - m, 0)] for m in range(4)])       # stride 1
                        h5_ = np.stack([obs[max(t - 5 * m, 0)] for m in range(4)])  # stride 5
                        samples.append((h1, h5_, label, k, uid))
    return samples


def load_onpolicy_negatives():
    """On-policy closed-on-air states: last steps of the policy phase before a
    miss-gated takeover. All label 0. Transfer set only — never trained on."""
    samples = []
    for f in DAGGER_FILES:
        with h5py.File(f) as h:
            root = h["data"] if "data" in h else h
            for name in root.keys():
                ep = root[name]
                if ep.attrs.get("gate_reason") != "miss":
                    continue
                t_to = int(ep.attrs["takeover_t"])
                obs = ep["obs/policy"][:]
                for j in (1, 5, 15, 30, 45):
                    t = t_to - j
                    if t < 4 * 5:
                        continue
                    h1 = np.stack([obs[t - m] for m in range(4)])
                    h5_ = np.stack([obs[t - 5 * m] for m in range(4)])
                    samples.append((h1, h5_, 0, j, f"{f.name}:{name}"))
    return samples


def features(sample_arr, variant):
    """sample_arr: (N, 4, 41) history stack, index 0 = current frame."""
    x = sample_arr
    if variant == "A":            # single frame, full obs
        return x[:, 0, :]
    if variant == "B2":           # 2-frame history
        return x[:, :2, :].reshape(len(x), -1)
    if variant == "B4":           # 4-frame history
        return x.reshape(len(x), -1)
    if variant == "C":            # single frame, finger dims removed
        keep = [d for d in range(41) if d not in FINGER_DIMS]
        return x[:, 0, keep]
    if variant == "D":            # finger dims only
        return x[:, 0, FINGER_DIMS]
    if variant == "E":            # finger dims, 4-frame history
        return x[:, :, FINGER_DIMS].reshape(len(x), -1)
    if variant == "F":            # objects+basket block only — confound check:
        return x[:, 0, 16:34]     # can a probe predict miss from can CONFIG alone?
    if variant == "G":            # physical finger joints only (no last-grip action)
        return x[:, 0, [6, 7, 14, 15]]
    raise ValueError(variant)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_probe(Xtr, ytr, Xte, yte, seed=SEED):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    model = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    w1 = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)  # upweight positives if rarer... or negatives
    pos_weight = torch.tensor([float(w1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(Xt)
    for epoch in range(60):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        ste = model(torch.tensor(Xte, dtype=torch.float32)).squeeze(-1).numpy()
        str_ = model(Xt).squeeze(-1).numpy()
    return model, (mu, sd), str_, ste


def main():
    rng = np.random.default_rng(SEED)
    expert = load_expert_samples()
    onpol = load_onpolicy_negatives()
    labels = np.array([s[2] for s in expert])
    ks = np.array([s[3] for s in expert])
    print(f"expert samples: {len(expert)}  (pos {int(labels.sum())} / neg {int((labels == 0).sum())})")
    for k in KS:
        m = ks == k
        print(f"  k={k}: {m.sum()} samples, {labels[m].mean():.3f} pos-rate")
    print(f"on-policy negatives: {len(onpol)}")

    # episode-level split
    uids = sorted({s[4] for s in expert})
    rng.shuffle(uids)
    test_uids = set(uids[: len(uids) // 5])
    is_te = np.array([s[4] in test_uids for s in expert])
    H1 = np.stack([s[0] for s in expert])   # (N, 4, 41) stride-1
    H5 = np.stack([s[1] for s in expert])   # (N, 4, 41) stride-5
    OH1 = np.stack([s[0] for s in onpol]) if onpol else None

    results = {"n_expert": len(expert), "n_pos": int(labels.sum()),
               "n_onpolicy_neg": len(onpol), "variants": {}}
    for variant in ["A", "B2", "B4", "B4s5", "C", "D", "E", "F", "G"]:
        src, v = (H5, "B4") if variant == "B4s5" else (H1, variant)
        X = features(src, v)
        model, norm, str_, ste = train_probe(X[~is_te], labels[~is_te], X[is_te], labels[is_te])
        r = {"auc_train": auc(str_, labels[~is_te]), "auc_test": auc(ste, labels[is_te]),
             "auc_test_by_k": {}}
        for k in KS:
            m = is_te & (ks == k)
            with torch.no_grad():
                mu, sd = norm
                sk = model(torch.tensor((features(src[m], v) - mu) / sd,
                                        dtype=torch.float32)).squeeze(-1).numpy()
            r["auc_test_by_k"][k] = auc(sk, labels[m])
        if OH1 is not None and variant != "B4s5":
            Xo = features(OH1, v)
            with torch.no_grad():
                mu, sd = norm
                so = model(torch.tensor((Xo - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
            # all true label 0 → report fraction wrongly scored as "grasped"
            thr = 0.0  # logit 0 = p 0.5
            r["onpolicy_fpr_at_0.5"] = float((so > thr).mean())
            r["onpolicy_mean_p"] = float(torch.sigmoid(torch.tensor(so)).mean())
        results["variants"][variant] = r
        print(variant, json.dumps(r, indent=None, default=float))

    out = HERE / "exp01_results.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
