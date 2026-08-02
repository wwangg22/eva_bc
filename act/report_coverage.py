#!/usr/bin/env python
"""Stratification / coverage report over demo HDF5 files (PLAN.md section 2.2 quota axes).

Usage:
    python report_coverage.py file1.h5 [file2.h5 ...]

Per file (and a TOTAL row), computed from each demo's first observation and attrs:
  - r-band of each can's spawn radius (obs objects_canonical pos slices [16:19], [23:26]):
        near r < 0.24 | mid 0.24-0.28 | far r > 0.28
  - orientation class per can from the spawn quat ([19:23], [26:30], XYZW): the can's
    cylinder axis is body-Y, axis = (2(xy-wz), 1-2(x^2+z^2), 2(yz+wx)); lying if |axis_z| < 0.5
  - basket sector from basket_center_xy (obs [32:34]): azimuth terciles over +/-50 deg
        left az > +16.7 deg | center | right az < -16.7 deg
  - object-scale proxy: median resting can z at t=0 across the file's demos (scale pins per run)
  - counts: demos, success rate, nominal pool (nominal_pool_filter), recovery pool
    (recovery_pool_filter), trainable steps (train_mask sum) and masked fraction

Plain text output; deps: h5py + numpy (filters imported from dataset.py alongside).
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np

from dataset import _plain_attrs, nominal_pool_filter, recovery_pool_filter

SECTOR_DEG = 50.0 / 3.0  # tercile boundaries of the +/-50 deg basket azimuth range


def scan_file(path: str) -> dict:
    c = {
        "file": path, "demos": 0, "success": 0, "nominal": 0, "recovery": 0,
        "steps": 0, "train_steps": 0,
        "r_near": 0, "r_mid": 0, "r_far": 0,
        "upright": 0, "lying": 0,
        "bsk_l": 0, "bsk_c": 0, "bsk_r": 0,
        "rest_z": [],
    }
    with h5py.File(path, "r") as f:
        data = f["data"]
        for key in sorted(data.keys(), key=lambda k: int(k.split("_")[-1])):
            grp = data[key]
            attrs = _plain_attrs(grp.attrs)
            c["demos"] += 1
            c["success"] += bool(attrs.get("success", False))
            c["nominal"] += nominal_pool_filter(attrs)
            c["recovery"] += recovery_pool_filter(attrs)
            mask = np.asarray(grp["train_mask"])
            c["steps"] += mask.shape[0]
            c["train_steps"] += int(mask.sum())

            o0 = grp["obs/policy"][0]
            for base in (16, 23):  # can 1, can 2: pos(3) + quat(4, XYZW)
                px, py, pz = o0[base : base + 3]
                x, y, z, w = o0[base + 3 : base + 7]
                r = float(np.hypot(px, py))
                if r < 0.24:
                    c["r_near"] += 1
                elif r <= 0.28:
                    c["r_mid"] += 1
                else:
                    c["r_far"] += 1
                axis_z = 2.0 * (y * z + w * x)  # z-component of the body-Y (cylinder) axis
                c["lying" if abs(axis_z) < 0.5 else "upright"] += 1
                c["rest_z"].append(float(pz))
            bx, by = o0[32:34]
            az = float(np.degrees(np.arctan2(by, bx)))
            if az > SECTOR_DEG:
                c["bsk_l"] += 1
            elif az < -SECTOR_DEG:
                c["bsk_r"] += 1
            else:
                c["bsk_c"] += 1
    return c


COLUMNS = [
    ("file", 28), ("demos", 5), ("succ%", 6), ("nom", 4), ("rec", 4),
    ("steps", 7), ("train", 7), ("mask%", 6),
    ("r<.24", 6), ("r-mid", 6), ("r>.28", 6),
    ("upri", 5), ("lyin", 5),
    ("bskL", 5), ("bskC", 5), ("bskR", 5),
    ("medZ", 7),
]


def format_row(c: dict) -> str:
    succ = 100.0 * c["success"] / c["demos"] if c["demos"] else 0.0
    maskf = 100.0 * (1.0 - c["train_steps"] / c["steps"]) if c["steps"] else 0.0
    med_z = float(np.median(c["rest_z"])) if c["rest_z"] else float("nan")
    vals = [
        os.path.basename(c["file"]), c["demos"], f"{succ:.1f}", c["nominal"], c["recovery"],
        c["steps"], c["train_steps"], f"{maskf:.1f}",
        c["r_near"], c["r_mid"], c["r_far"],
        c["upright"], c["lying"],
        c["bsk_l"], c["bsk_c"], c["bsk_r"],
        f"{med_z:.4f}",
    ]
    return "  ".join(str(v).ljust(w) if i == 0 else str(v).rjust(w) for i, ((_, w), v) in enumerate(zip(COLUMNS, vals)))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    per_file = [scan_file(p) for p in argv]
    total = {"file": "TOTAL", "rest_z": []}
    for c in per_file:
        for k, v in c.items():
            if k == "file":
                continue
            if k == "rest_z":
                total["rest_z"].extend(v)
            else:
                total[k] = total.get(k, 0) + v

    header = "  ".join(name.ljust(w) if i == 0 else name.rjust(w) for i, (name, w) in enumerate(COLUMNS))
    print(header)
    print("-" * len(header))
    for c in per_file:
        print(format_row(c))
    print("-" * len(header))
    print(format_row(total))
    print(
        "\nbands: r-band of each can spawn (2 cans/demo); upri/lyin: |axis_z| of body-Y "
        ">= / < 0.5; bskL/C/R: basket azimuth terciles over +/-50 deg; "
        "medZ: median resting can z at t=0 (per-file object-scale proxy); "
        "mask%: fraction of steps loss-censored (train_mask == 0)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
