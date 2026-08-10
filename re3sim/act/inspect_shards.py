# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Look at a vision shard directory before spending hours training on it.

Every failure mode this catches is silent downstream -- the images still look like images, the
tensors still have the right shape, and the loss still goes down:

* **the desk missing** from all but one env (the `/World/Splats` prim is global unless
  `RE3SIM_SPLATS_PER_ENV=1`). Reported as the fraction of dark "ground plane" pixels, and by
  writing an actual contact sheet to look at.
* **privileged leakage** -- proprio must be exactly `obs41[0:16] ++ obs41[34:41]`.
* **the frames and the actions out of step by one** -- checked by correlating the commanded
  joint delta against the joint motion that follows it.
* **dead time**: what fraction of the kept steps the arm is not moving. Batch padding is
  supposed to have been dropped at write time; what is left should be the expert's deliberate
  settles, not another 40 %.

Usage
-----
    python -u re3sim/act/inspect_shards.py re3sim/expert/data/vision_s1 --sheet /tmp/sheet.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--sheet", default=None, help="write a contact sheet of sampled frames")
    p.add_argument("--max-shards", type=int, default=24)
    args = p.parse_args()

    paths = sorted(q for d in args.dirs for q in Path(d).glob("ep_*.pt"))
    if not paths:
        raise SystemExit(f"no ep_*.pt under {args.dirs}")
    print(f"{len(paths)} shards")

    lens, still, dark_w, dark_s, n_ok = [], [], [], [], 0
    lag_corr = []
    sample = paths[:: max(1, len(paths) // args.max_shards)][: args.max_shards]
    for q in sample:
        sh = torch.load(q, map_location="cpu", weights_only=False)
        w, s = sh["wrist_rgb"], sh["workspace_rgb"]
        obs, act, pro = sh["obs41"], sh["actions"], sh["proprio"]
        T = act.shape[0]
        lens.append(T)
        n_ok += int(bool(sh["success"]))
        assert w.dtype == torch.uint8 and s.dtype == torch.uint8, (q, w.dtype)
        assert w.shape[0] == T and s.shape[0] == T, (q, w.shape, act.shape)
        # the contract, re-checked on disk rather than trusted from the writer
        ref = torch.cat([obs[:, 0:16], obs[:, 34:41]], dim=1)
        assert torch.equal(pro, ref), f"{q}: proprio is not obs41[0:16]++obs41[34:41]"
        # dead time: the commanded arm action unchanged from the previous step
        d = (act[1:, :6] - act[:-1, :6]).abs().max(dim=1).values
        still.append(float((d < 1e-6).float().mean()))
        # "is there a desk": the bare ground plane is near-black, the desk is not
        dark_w.append(float((w.float().mean(-1) < 40).float().mean()))
        dark_s.append(float((s.float().mean(-1) < 40).float().mean()))
        # Frame/action alignment. The action is a joint-position target in relative units
        # (`a[:6] = (q_target - q_default) / 0.5`) and proprio[:, :6] is `q - q_default`, so
        # `act[t]*0.5 - pro[t]` is the position error the drive is being asked to close at t.
        # It must correlate with the motion actually measured between t and t+1; if the images
        # and actions were written one step apart, this collapses.
        err = act[:-1, :6] * 0.5 - pro[:-1, :6]
        mot = pro[1:, :6] - pro[:-1, :6]
        a, b = err.flatten().numpy(), mot.flatten().numpy()
        if a.std() > 0 and b.std() > 0:
            lag_corr.append(float(np.corrcoef(a, b)[0, 1]))

    print(f"  success in sample     {n_ok}/{len(sample)}")
    print(f"  steps per episode     mean {np.mean(lens):.0f}  min {min(lens)}  max {max(lens)}")
    print(f"  arm command unchanged {np.mean(still):.1%} of steps  "
          f"(expert settles; batch padding should already be gone)")
    print(f"  near-black pixels     wrist {np.mean(dark_w):.1%}   workspace {np.mean(dark_s):.1%}"
          f"   <- a workspace view with no desk runs ~40 %+")
    if lag_corr:
        print(f"  cmd-error vs motion   r = {np.mean(lag_corr):+.3f}  "
              f"(positive means the frame precedes the action it labels)")
    px = int(np.prod(torch.load(sample[0], map_location="cpu",
                                weights_only=False)["wrist_rgb"].shape[1:]))
    tot = sum(lens) * px * 2 / 1e9 * len(paths) / len(sample)
    print(f"  approx image bytes    {tot:.1f} GB for all {len(paths)} shards "
          f"(VisionShardDataset holds these in RAM)")

    if args.sheet:
        from PIL import Image
        rows = []
        for q in sample[:6]:
            sh = torch.load(q, map_location="cpu", weights_only=False)
            T = sh["actions"].shape[0]
            idx = np.linspace(0, T - 1, 6).astype(int)
            for key in ("workspace_rgb", "wrist_rgb"):
                rows.append(np.concatenate([sh[key][i].numpy() for i in idx], axis=1))
        Image.fromarray(np.concatenate(rows, axis=0)).save(args.sheet)
        print(f"  wrote {args.sheet}")


if __name__ == "__main__":
    main()
