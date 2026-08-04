# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Structural and semantic verification of a collected demo file. No Isaac Sim needed.

The point of this file is that a demo set can be perfectly well-*shaped* and still be wrong in
the one way that silently destroys behaviour cloning: an off-by-one between the observation
and the action it is supposed to label. Shape checks cannot see that. This can, because the
34-D observation carries ``last_action`` in its tail:

    **obs[t, 27:34] must equal actions[t-1]**, exactly.

If the collector had recorded the observation *after* stepping (the natural mistake, since
``env.step`` hands one back), that identity would read ``obs[t, 27:34] == actions[t]`` and the
policy would be trained to predict an action it can already see. This project has been burned
twice by harnesses that were self-consistent rather than correct, so the alignment is asserted
rather than assumed.

.. code-block:: bash

    python slot/scripts/verify_demos.py slot/data/demos_v0_n128x4_s0_noise0.hdf5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

OBS_DIM = 34
ACTION_DIM = 7
STATE_SLICE = slice(0, 16)        # joint_pos_rel(8) + joint_vel_rel(8)
ENV_STATE_SLICE = slice(16, 34)   # block pose(7) + slot_error(4) + last_action(7)
LAST_ACTION_SLICE = slice(27, 34)
SLOT_ERR_SLICE = slice(23, 27)    # depth, lateral, yaw, inserted
ARM_ACTION_SCALE = 0.5            # JointPositionActionCfg(scale=0.5, use_default_offset=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a collected demo HDF5.")
    ap.add_argument("path", type=str)
    ap.add_argument("--tol", type=float, default=1e-5, help="tolerance for the alignment identity")
    a = ap.parse_args()

    path = Path(a.path)
    passed = True
    with h5py.File(str(path), "r") as f:
        data = f["data"]
        keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
        total = int(data.attrs["total"])
        print(f"\n{path.name}: {len(keys)} demos, /data.attrs['total'] = {total}\n")

        passed &= check("demo group names are demo_<int>, contiguous from 0",
                        [int(k.split("_")[-1]) for k in keys] == list(range(len(keys))))
        passed &= check("attrs['total'] matches the group count", total == len(keys))

        shapes_ok = dtypes_ok = align_ok = t0_ok = grip_ok = track_ok = mask_ok = True
        worst_align = 0.0
        worst_track = 0.0
        lens, n_succ, n_masked, kinds, spawns = [], 0, 0, {}, []
        outcome_hist: dict[str, int] = {}

        for k in keys:
            g = data[k]
            obs = np.asarray(g["obs/policy"])
            act = np.asarray(g["actions"])
            msk = np.asarray(g["train_mask"])
            T = obs.shape[0]
            lens.append(T)
            shapes_ok &= obs.shape == (T, OBS_DIM) and act.shape == (T, ACTION_DIM) and msk.shape == (T,)
            dtypes_ok &= (obs.dtype == np.float32 and act.dtype == np.float32 and msk.dtype == np.uint8)
            if int(g.attrs["num_samples"]) != T:
                shapes_ok = False

            # --- the alignment identity: obs[t] is the observation the policy sees BEFORE act[t]
            worst_align = max(worst_align, float(np.abs(obs[1:, LAST_ACTION_SLICE] - act[:-1]).max()))
            t0_ok &= bool(np.abs(obs[0, LAST_ACTION_SLICE]).max() < a.tol)

            # --- the gripper channel is binary by construction (<0 closes, >=0 opens)
            grip_ok &= bool(np.all(np.isin(np.sign(act[:, 6]), (-1.0, 1.0))))
            grip_ok &= bool(np.abs(np.abs(act[:, 6]) - 1.0).max() < a.tol)

            # --- the arm tracked its own targets: joint_pos_rel should approach 0.5*action.
            # Judged on the last 20 frames, which are an idle hold, so any residual is real
            # steady-state error and not motion lag.
            resid = np.abs(obs[-20:, 0:6] - ARM_ACTION_SCALE * act[-21:-1, 0:6]).max()
            worst_track = max(worst_track, float(resid))

            n_succ += bool(g.attrs["success"])
            n_masked += int((msk == 0).any())
            kinds[str(g.attrs["episode_kind"])] = kinds.get(str(g.attrs["episode_kind"]), 0) + 1
            spawns.append(json.loads(g.attrs["spawn"]))
            for sid, oc in json.loads(g.attrs["outcomes"]).items():
                key = f"{sid}={oc['outcome']}"
                outcome_hist[key] = outcome_hist.get(key, 0) + 1
            # censored demos must be censored contiguously to the end (this expert never recovers)
            if (msk == 0).any():
                first0 = int(np.argmax(msk == 0))
                mask_ok &= bool((msk[first0:] == 0).all())

        passed &= check("dataset shapes (T,34)/(T,7)/(T,) and num_samples agree", shapes_ok)
        passed &= check("dtypes float32/float32/uint8", dtypes_ok)

        # On a DART pool the strict identity CANNOT hold, and that is the point: the executed
        # action is nominal + injected noise while the recorded label is nominal, so
        # last_action - actions[t-1] IS the noise. Checking the residual against the OU process
        # it is supposed to be verifies alignment AND the noise injection at once, which the
        # strict identity never could.
        noise_std = float(data[keys[0]].attrs.get("noise_std", 0.0))
        if noise_std == 0.0:
            passed &= check("ALIGNMENT obs[t,27:34] == actions[t-1]", worst_align < a.tol,
                            f"max abs difference {worst_align:.3e}")
        else:
            eps = np.concatenate([
                np.asarray(data[k]["obs/policy"])[1:, LAST_ACTION_SLICE][:, :6]
                - np.asarray(data[k]["actions"])[:-1, :6] for k in keys])
            segs = {s["seg"]: s["t"] for s in json.loads(data[keys[0]].attrs["segments"])}
            mag = np.abs(eps).max(axis=1)
            push_lo = segs["push"] - 1                 # eps index t corresponds to step t+1
            per = np.abs(np.stack([
                np.asarray(data[k]["obs/policy"])[1:, LAST_ACTION_SLICE][:, :6]
                - np.asarray(data[k]["actions"])[:-1, :6] for k in keys])).max(axis=2)
            active = per[:, : push_lo]
            frozen = per[:, push_lo:]
            # stationary std of an OU walk with this rho is exactly noise_std
            got = float(eps[np.abs(eps).sum(axis=1) > 0].std()) if (np.abs(eps) > 0).any() else 0.0
            passed &= check("DART residual std matches --noise_std (OU stationary std)",
                            abs(got - noise_std) < 0.35 * noise_std,
                            f"measured {got:.4f} vs declared {noise_std:.4f}")
            passed &= check("noise is ACTIVE before the push phase", active.max() > 0.2 * noise_std,
                            f"max |eps| = {active.max():.4f} over t < {push_lo}")
            passed &= check("noise is OFF from the push phase onward (phase restriction held)",
                            frozen.max() < 0.05 * noise_std,
                            f"max |eps| = {frozen.max():.2e} over t >= {push_lo}, "
                            f"i.e. {100 * frozen.max() / max(noise_std, 1e-9):.2f} % of noise_std")
            lag1 = float(np.corrcoef(eps[:-1].ravel(), eps[1:].ravel())[0, 1])
            passed &= check("noise is temporally correlated, not white", lag1 > 0.5,
                            f"lag-1 autocorrelation {lag1:.3f} (OU rho was 0.95)")
            print(f"    (max |residual| {mag.max():.4f} action units = the injected noise, "
                  f"NOT a misalignment -- the strict identity only applies to noise-free pools)")
        passed &= check("obs[0] last_action is zero (env reset cleared it)", t0_ok)
        passed &= check("gripper action channel is exactly +/-1", grip_ok)
        passed &= check("arm tracks its target at rest (|joint_pos_rel - 0.5*action|)",
                        worst_track < 0.02, f"worst {worst_track:.4f} rad")
        passed &= check("train_mask censors contiguously to the episode end", mask_ok)
        passed &= check("every demo has the same length", len(set(lens)) == 1, f"T = {sorted(set(lens))}")

        # Loss-censoring health. A mask that is all ones on every demo is not "clean data" --
        # on this task it means the detector cannot see the expert's actual failure mode, which
        # is exactly what a purely grip-based outcome check did (every phase reported "held"
        # on all 512 demos while 50 of them ended unseated).
        slips = np.array([float(data[k].attrs.get("slip_max_mm", np.nan)) for k in keys])
        cuts = np.array([int(data[k].attrs.get("slip_censor_t", -1)) for k in keys])
        succ = np.array([bool(data[k].attrs["success"]) for k in keys])
        if not np.isnan(slips).all():
            keep = np.array([(np.asarray(data[k]["train_mask"]) == 1).mean() for k in keys])
            print(f"\n  in-hand slip [mm]  p50 {np.nanpercentile(slips, 50):.2f}  "
                  f"p90 {np.nanpercentile(slips, 90):.2f}  max {np.nanmax(slips):.2f}")
            print(f"    seated p90 {np.nanpercentile(slips[succ], 90):.2f}  vs  "
                  f"failed p90 {np.nanpercentile(slips[~succ], 90) if (~succ).any() else float('nan'):.2f}"
                  "   <- raw/whole-span is an INVERTED statistic: successes score HIGHER,")
            print("       because the expert drives the block into the back stop and the gripper "
                  "keeps\n       advancing while it cannot move. Use calibrate_slip.py, which "
                  "windows the signal.")
            print(f"  slip-censored      {int((cuts >= 0).sum())}/{len(keys)} demos, "
                  f"{int(((cuts >= 0) & succ).sum())} of them successful")
            print(f"  trainable frames   {100 * keep.mean():.1f}% of all frames "
                  f"({100 * keep[succ].mean():.1f}% within successful demos)")

        sp = np.asarray(spawns)
        depth = np.asarray([np.asarray(data[k]["obs/policy"])[-1, SLOT_ERR_SLICE][0] for k in keys])
        print(f"\n  demos          {len(keys)}  (T = {lens[0]}, "
              f"{len(keys) * lens[0]:,} frames = BC samples)")
        print(f"  successful     {n_succ}/{len(keys)} = {100 * n_succ / len(keys):.1f}%")
        print(f"  loss-censored  {n_masked} demos have any train_mask == 0")
        print(f"  episode_kind   {kinds}")
        print(f"  spawn coverage x [{sp[:, 0].min():.3f}, {sp[:, 0].max():.3f}]  "
              f"y [{sp[:, 1].min():.3f}, {sp[:, 1].max():.3f}]  "
              f"yaw [{sp[:, 2].min():+.3f}, {sp[:, 2].max():+.3f}] rad")
        print(f"  final depth    mean {depth.mean() * 1000:.1f} mm  min {depth.min() * 1000:.1f}  "
              f"(success needs >= 40.0)")
        print("  outcomes       " + "  ".join(f"{k}:{v}" for k, v in sorted(outcome_hist.items())))

    print(f"\n{'ALL CHECKS PASSED' if passed else 'VERIFICATION FAILED'}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
