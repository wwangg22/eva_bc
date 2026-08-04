#!/usr/bin/env python
"""Offline audit of a clutter demo set -- and a direct measurement of the N2 risk.

No simulator. Everything here is arithmetic on the recorded tape, which is the point: the
biggest registered threat to Stage 2 is a property of the *dataset*, and it can be measured
before a single training step is spent.

The N2 risk, as registered (the N2 chunk-ambiguity risk; `REFERENCE.md` §6)
---------------------------------------------------------------------------------
    "47 % of the 382-step demo is a held pose -- 70 env steps of `close` alone, during which
     the 42-D observation is constant to numerical noise while the correct action chunk
     differs at every step. A memoryless chunk policy cannot recover phase from a constant
     observation; a flow head learns the distribution over 'steps until the lift' and samples
     one."

That is a hypothesis with a number attached, and the number is measurable. If two frames have
near-identical observations and different correct chunks, no memoryless predictor can get both
right, and the size of that disagreement is a **floor on the training loss** -- an irreducible
error that no amount of capacity or optimisation removes.

How it is measured
------------------
For a sample of query frames, find the `k` nearest frames in NORMALIZED observation space (the
space the network actually sees) and measure how far apart their action chunks are, also
normalized. Three numbers per phase:

    nn_chunk_rmse    RMS distance between a query's chunk and its neighbours' chunks.
                     Two independent draws from the same conditional differ by `sqrt(2)*sigma`,
                     so this is `sqrt(2)` times the conditional std of the correct chunk.
    uncond_rmse      the same distance between RANDOM pairs from the phase. The scale bar --
                     what the number would be if the observation carried no information at all.
    nn_dt            median |t_query - t_neighbour| in env steps, for neighbours drawn from the
                     SAME demo. If the observation encodes phase, this is ~1. If the hold is
                     genuinely static, it is a large fraction of the hold length.

`nn_chunk_rmse / uncond_rmse` is the fraction of the chunk's variation that survives
conditioning on the observation. Near 0 means the observation determines the chunk; near 1
means it does not, and the phase is where the policy will hallucinate.

The loss floor, derived rather than asserted
--------------------------------------------
The training log prints a **velocity** MSE, not a chunk MSE, so the two are not directly
comparable and it would be sloppy to put them side by side without doing the work.

Rectified flow draws `x0 ~ N(0, I)`, `tau ~ U[0,1]`, forms `x_tau = (1-tau)*x0 + tau*x1`, and
regresses `v = x1 - x0` on `(obs, x_tau, tau)`. Model one dimension of `p(x1 | obs)` as
`N(mu, sigma^2)`; everything is jointly Gaussian, so the Bayes-optimal predictor's residual is

    Var(v | obs, x_tau) = (sigma^2 + 1) - (tau*sigma^2 - (1-tau))^2
                                          / ((1-tau)^2 + tau^2 * sigma^2)

and integrating over `tau ~ U[0,1]` gives, to four decimals for every sigma from 0.01 to 1.0,

    floor(sigma) = sigma * pi / 2                                     (verified numerically)

which is exactly 0 when the observation determines the chunk, and grows linearly in the
ambiguity. Summing over the `chunk_size * action_dim` cells of the chunk, the achievable
training MSE is

    floor = (pi / 2) * mean_over_cells( sigma_cell )

Note the **mean**, not the RMS: ambiguity concentrated in a few cells costs far less than the
same total spread smeared over all of them, so `nn_chunk_rmse` alone would badly overstate it.
`sigma_cell` is estimated per cell as `rms_cell / sqrt(2)`.

**This is an UPPER bound on the flow objective, and here a loose one.** The derivation prices
every one of the `chunk_size * action_dim` cells independently, but the flow head is given
`x_tau` for all of them at once, and the ambiguity in this dataset is essentially ONE scalar --
"how many steps until the lift begins". 350 noisy readings of a single latent identify it
almost exactly, so the model resolves at training time an ambiguity the per-cell calculation
charges it for 350 times over. Measured: the bound is 0.099 and the observed plateau is 0.045.

What the bound is still good for: **localisation**. The per-phase contributions are computed
on the same footing, so "54 % of the ambiguity is in the `close`" is a like-for-like comparison
even though the absolute number is an over-estimate.

And the behavioural consequence is untouched by any of this. At INFERENCE there is no `x_tau`
-- sampling starts from `x0 ~ N(0, I)` -- so the mode really is drawn from the prior, and the
policy really does pick its own moment to lift. That is what the simulator eval measures, and
it is why `ratio` (a property of the data, not of the objective) is the column to read.

Usage
-----
    python -u clutter/act/analyse_demos.py --data runs/demos_v1.hdf5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import (  # noqa: E402
    ACTION_DIM,
    ENV_STATE_SLICE,
    OBS_DIM,
    STATE_SLICE,
    ClutterDemoDataset,
    compute_stats,
    load_segments,
)

ENV_STATE_NAMES = (["tgt_x", "tgt_y", "tgt_z", "tgt_qx", "tgt_qy", "tgt_qz", "tgt_qw"]
                   + [f"d{i}_{c}" for i in range(4) for c in ("dx", "dy", "up")]
                   + [f"last_a{j}" for j in range(ACTION_DIM)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--queries", type=int, default=4000)
    p.add_argument("--refs", type=int, default=30000)
    p.add_argument("--k", type=int, default=16,
                   help="neighbours per query; also the spread the zero-distance "
                        "extrapolation is fitted over, so do not set it too small")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = ClutterDemoDataset(args.data, chunk_size=args.chunk_size)
    stats = compute_stats(ds)
    lens = np.array([d["obs"].shape[0] for d in ds.demos])

    print("\n" + "=" * 100)
    print("CLUTTER DEMO AUDIT")
    print("=" * 100)
    print(f"   {len(ds.demos)} demos kept, {ds.n_rejected} rejected, {len(ds)} samples")
    print(f"   episode length {lens.min()}-{lens.max()} env steps (mean {lens.mean():.1f})")

    # ------------------------------------------------------------------ actions
    acts = np.concatenate([d["actions"] for d in ds.demos], axis=0)
    print("\n   ACTION RANGE   a = 2 * (q_desired - q_default); the env applies no clip")
    print("      joint       min      max    |a|max")
    for j in range(6):
        lo, hi = acts[:, j].min(), acts[:, j].max()
        m = max(abs(lo), abs(hi))
        print(f"      j{j + 1}      {lo:+8.3f} {hi:+8.3f}  {m:7.3f}"
              + ("   <-- outside [-1,1]" if m > 1.0 else ""))
    g = np.unique(acts[:, 6])
    print(f"      gripper  unique values {g}   "
          + ("BINARY, as required" if set(g.tolist()) <= {-1.0, 1.0}
             else "*** NOT BINARY -- a policy cannot emit this ***"))

    # ---------------------------------------------------------- degenerate channels
    sd = stats["observation.environment_state"]["std"].numpy()
    weak = [(i, ENV_STATE_NAMES[i], v) for i, v in enumerate(sd) if v < 1e-3]
    print("\n   NEAR-DEGENERATE OBSERVATION CHANNELS   (std < 1e-3 over the kept pool)")
    if not weak:
        print("      none")
    for i, nm, v in weak:
        print(f"      env_state[{i:2}] {nm:9} std {v:.2e}  -> normalization multiplies its "
              f"residual by {1 / v:.0f}")
    if weak:
        print("      These are distractor up-axes. Successful demos never topple, so the")
        print("      channel is ~1.0 throughout and normalization rescales its remaining")
        print("      micro-variation (real nudge signal) to unit variance. Recorded, not")
        print("      changed: it is the first thing to test if the policy underperforms.")

    # --------------------------------------------------------------- phase segments
    segs = load_segments(ds.demos[0]["attrs"])
    print("\n   PHASE SEGMENTS (identical across demos -- one frozen schedule)")
    tot = segs[-1][2]
    for ph, t0, t1 in segs:
        print(f"      {ph:<9} [{t0:4},{t1:4})  {t1 - t0:4} env steps  {(t1 - t0) / tot:5.1%}")
    holds = {"settle", "predwell", "close", "dwell", "release", "final"}
    static = sum(t1 - t0 for ph, t0, t1 in segs if ph in holds)
    print(f"      TOTAL {tot} env steps, {static} of them a held pose ({static / tot:.1%})")

    # ------------------------------------------------ N2: conditional chunk ambiguity
    #
    # Build normalized (obs, chunk) for a random subsample. The chunk for frame t is the
    # dataset's own chunk (edge-padded), so this measures exactly the target the loss sees.
    print("\n   N2 -- IS THE CORRECT CHUNK DETERMINED BY THE OBSERVATION?")
    print(f"      {args.queries} queries against {args.refs} reference frames, k = {args.k}, "
          f"normalized space")

    o_mean = torch.cat([stats["observation.state"]["mean"],
                        stats["observation.environment_state"]["mean"]]).to(dev)
    o_std = torch.cat([stats["observation.state"]["std"],
                       stats["observation.environment_state"]["std"]]).to(dev)
    a_mean = stats["action"]["mean"].to(dev)
    a_std = stats["action"]["std"].to(dev)
    assert o_mean.numel() == OBS_DIM and STATE_SLICE.start == 0 \
        and ENV_STATE_SLICE.stop == OBS_DIM

    def gather(idx):
        obs = np.stack([ds.demos[d]["obs"][t] for d, t in idx])
        ch = np.stack([np.concatenate([
            ds.demos[d]["actions"][t:min(t + args.chunk_size, ds.demos[d]["obs"].shape[0])],
            np.repeat(ds.demos[d]["actions"][-1][None],
                      max(0, t + args.chunk_size - ds.demos[d]["obs"].shape[0]), axis=0),
        ]) for d, t in idx])
        o = (torch.from_numpy(obs).to(dev) - o_mean) / (o_std + 1e-8)
        a = (torch.from_numpy(ch).to(dev) - a_mean) / (a_std + 1e-8)
        return o, a.reshape(len(idx), -1)

    all_idx = np.array(ds.index)
    ref_sel = rng.choice(len(all_idx), size=min(args.refs, len(all_idx)), replace=False)
    ref_idx = all_idx[ref_sel]
    ref_o, ref_a = gather(ref_idx)
    ref_demo = torch.from_numpy(ref_idx[:, 0]).to(dev)
    ref_t = torch.from_numpy(ref_idx[:, 1]).to(dev)

    print(f"\n      {'phase':<10} {'nn_chunk_rmse':>14} {'uncond_rmse':>12} {'ratio':>7} "
          f"{'nn_dt':>7} {'floor(hi)':>10} {'floor':>9}")
    rows = {}
    for ph, t0, t1 in segs:
        pool = np.where((all_idx[:, 1] >= t0) & (all_idx[:, 1] < t1))[0]
        q_sel = rng.choice(pool, size=min(args.queries, len(pool)), replace=False)
        q_idx = all_idx[q_sel]
        q_o, q_a = gather(q_idx)

        d2 = torch.cdist(q_o, ref_o)
        # drop the exact self-match (same demo AND same t)
        same = ((torch.from_numpy(q_idx[:, 0]).to(dev)[:, None] == ref_demo[None, :])
                & (torch.from_numpy(q_idx[:, 1]).to(dev)[:, None] == ref_t[None, :]))
        d2 = d2.masked_fill(same, float("inf"))
        nn = d2.topk(args.k, largest=False).indices                       # (Q, k)

        sq = (q_a[:, None, :] - ref_a[nn]).pow(2)                         # (Q, k, C*A)
        nn_rmse = float(sq.mean().sqrt())
        # per-cell conditional std: two independent draws differ by sqrt(2)*sigma
        sigma_cell = (sq.mean(dim=(0, 1)) / 2.0).clamp(min=0.0).sqrt()    # (C*A,)
        flow_floor_hi = float(sigma_cell.mean()) * math.pi / 2.0

        # ...but a nearest neighbour is not at zero distance, and whatever the observation
        # DOES determine varies between the query and its neighbour. That inflates the
        # estimate, so `flow_floor_hi` is an upper bound, not the floor. Debias by
        # extrapolating to zero observation distance: per cell, least-squares fit
        #     E[(chunk_q - chunk_nn)^2] = 2*sigma^2 + beta * d_obs^2
        # over all (query, neighbour) pairs, and read off the intercept.
        x = d2.gather(1, nn).reshape(-1).pow(2)                           # (Q*k,) obs dist^2
        Y = sq.reshape(-1, sq.shape[-1])                                  # (Q*k, C*A)
        xc = x - x.mean()
        den = float((xc * xc).sum())
        beta = (xc @ (Y - Y.mean(0))) / max(den, 1e-12)                   # (C*A,)
        a0 = (Y.mean(0) - beta * x.mean()).clamp(min=0.0)
        flow_floor = float((a0 / 2.0).sqrt().mean()) * math.pi / 2.0

        # unconditional scale bar: random pairs drawn from the SAME phase
        rp = rng.choice(len(q_sel), size=(min(2000, len(q_sel)), 2))
        unc = float((q_a[rp[:, 0]] - q_a[rp[:, 1]]).pow(2).mean(dim=1).sqrt().mean())

        sd_mask = ref_demo[nn] == torch.from_numpy(q_idx[:, 0]).to(dev)[:, None]
        dt = (ref_t[nn] - torch.from_numpy(q_idx[:, 1]).to(dev)[:, None]).abs().float()
        nn_dt = float(dt[sd_mask].median()) if bool(sd_mask.any()) else float("nan")

        rows[ph] = {"nn_chunk_rmse": nn_rmse, "uncond_rmse": unc,
                    "ratio": nn_rmse / unc if unc else float("nan"),
                    "flow_floor": flow_floor, "flow_floor_hi": flow_floor_hi,
                    "nn_dt_same_demo": nn_dt, "n_queries": len(q_sel),
                    "len": t1 - t0, "is_hold": ph in holds}
        print(f"      {ph:<10} {nn_rmse:14.4f} {unc:12.4f} {nn_rmse / unc:7.3f} "
              f"{nn_dt:7.1f} {flow_floor_hi:10.4f} {flow_floor:9.4f}")

    print("\n      ratio      = fraction of the chunk's variation surviving conditioning on obs")
    print("      floor(hi)  = (pi/2)*mean_cell(sigma_cell) straight from the k neighbours --")
    print("                   an UPPER bound: a neighbour is not at zero observation distance,")
    print("                   so real obs-driven variation leaks into the estimate")
    print("      floor      = the same quantity extrapolated to zero obs distance.")
    print("                   Still an UPPER bound on the flow objective (see the docstring):")
    print("                   it prices each of the 350 chunk cells independently, while the")
    print("                   flow head sees x_tau for ALL of them and this ambiguity is one")
    print("                   scalar -- `when does the lift start` -- so it aggregates 350")
    print("                   noisy readings of a single latent. Expect the training mse to")
    print("                   land BELOW this and above zero.")
    worst = max(rows.items(), key=lambda kv: kv[1]["flow_floor"])
    print(f"      worst phase: {worst[0]}, flow floor {worst[1]['flow_floor']:.4f} "
          f"over {worst[1]['len']} steps")
    hold_rows = {k: v for k, v in rows.items() if v["is_hold"]}
    move_rows = {k: v for k, v in rows.items() if not v["is_hold"]}
    hf = sum(v["flow_floor"] * v["len"] for v in hold_rows.values())
    mf = sum(v["flow_floor"] * v["len"] for v in move_rows.values())
    lh = sum(v["len"] for v in hold_rows.values())
    lm = sum(v["len"] for v in move_rows.values())
    tot_floor = (hf + mf) / (lh + lm)
    print(f"\n      holds       flow floor {hf / lh:.4f}  over {lh} steps")
    print(f"      moves       flow floor {mf / lm:.4f}  over {lm} steps")
    print(f"      WHOLE DEMO  flow floor {tot_floor:.4f}  (upper bound; see the note above)")
    print("      contribution to the floor, by phase:")
    for ph, v in sorted(rows.items(), key=lambda kv: -kv[1]["flow_floor"] * kv[1]["len"]):
        c = v["flow_floor"] * v["len"] / (hf + mf)
        print(f"         {ph:<10} {c:6.1%}   ({v['len']:3} steps, "
              f"{v['len'] / (lh + lm):5.1%} of the demo)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "n_demos": len(ds.demos),
                       "n_rejected": ds.n_rejected, "n_samples": len(ds),
                       "episode_len": [int(lens.min()), int(lens.max())],
                       "action_abs_max": [float(np.abs(acts[:, j]).max()) for j in range(7)],
                       "gripper_values": [float(x) for x in g],
                       "degenerate_channels": [{"i": i, "name": nm, "std": float(v)}
                                               for i, nm, v in weak],
                       "segments": segs, "n2": rows,
                       "flow_floor": {"holds": hf / lh, "moves": mf / lm, "all": tot_floor}}, f, indent=2)
        print(f"\n   wrote {args.out}")


if __name__ == "__main__":
    main()
