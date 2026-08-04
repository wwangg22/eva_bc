# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""The expert's schedule, expanded into per-env-step joint commands.

Extracted verbatim from `collect_demos.py` on 2026-08-03 so that the diagnostic probes and
the demo collector cannot drift apart. That matters more here than it usually would: the
whole value of a probe like P36 is that it instruments **the manoeuvre that was measured**,
and a re-implementation that differs by one interpolation step is measuring something else.

Same reasoning as `policy_runner.py`, which exists so `eval_flow.py` and `record_video.py`
share one `ChunkController`.

No argparse and no `AppLauncher` here, so it is importable from any script that has already
launched the app.
"""

from __future__ import annotations

import torch


def expand(ex, chain) -> list[tuple[str, torch.Tensor, bool]]:
    """`ClutterExpert.schedule` -> one entry per **env step**: (phase, q_arm (n,6), close).

    This mirrors `_hold` and `_move` exactly, which is the whole point -- if the two ever
    diverge, Gate 2a stops measuring the port and starts measuring a transcription error.

    * a `hold` argument is in **physics** steps; one env step is `decimation = 8` of them, so
      it expands to `steps // 8` identical commands. All six `HOLDS` values are multiples of
      8; a value that is not is a hard error rather than a silent rounding.
    * a `move` argument is already in env steps, and `_move` runs `substeps = 8` physics steps
      per control step with a linear target ramp `f = (s + 1) / steps`. `env.step` with
      `decimation = 8` is the same thing, so the commanded sequence is identical.
    """
    out: list[tuple[str, torch.Tensor, bool]] = []
    for item in ex.schedule(chain):
        phase, kind, close = item[0], item[1], bool(item[-1])
        if kind == "hold":
            q, nphys = item[2], item[3]
            if nphys % 8:
                raise ValueError(f"hold '{phase}': {nphys} physics steps is not a multiple "
                                 f"of decimation 8; the env.step port would round it")
            out += [(phase, q, close)] * (nphys // 8)
        else:
            a, b, k = item[2], item[3], item[4]
            out += [(phase, (1.0 - (s + 1) / k) * a + ((s + 1) / k) * b, close)
                    for s in range(k)]
    return out


def approach_prefix(K, appr_qs, chain0, q_nom0, total: int):
    """Expand the FROZEN `home -> chain[0]` approach into per-env-step commands.

    P29 measured three candidates and only one survives, so this is not a choice the caller
    makes:

    ==================================  =========  =========  ==============  =========
    candidate                           keep-out   seam |dq|  approach hazard  success
    ==================================  =========  =========  ==============  =========
    teleport (the Stage-1 baseline)      --         0.00 rad   0 %             74.2 %
    joint-space lerp, 40 steps           4.85 mm    0.00 rad   **100 %**        5.5 %
    forward dense Cartesian, 40 steps    0.00 mm    **1.90**   0 %             **0.0 %**
    backward dense Cartesian, 80 steps   0.00 mm    0.00 rad   0 %             **73.0 %**
    ==================================  =========  =========  ==============  =========

    Two independent ways to destroy the run, and the middle row is the instructive one. The
    forward Cartesian path is geometrically spotless -- 0.00 mm of penetration, every body
    above z = 105 mm -- and it scores **exactly zero**, with 100 % topple and 96 % of targets
    still delivered to the goal. It ends 1.90 rad from the frozen chain's `qs[0]` on `joint6`:
    a different IK branch at the same TCP. The schedule's first hold then commands that 1.90
    rad as a single step change with the gripper directly over the row, and the wrist sweeps
    every neighbour down on its way through. P17's branch flip, moved to the seam between two
    independently solved paths -- `_dense` enforces local solves *within* a path and nothing
    enforced it *across the join*.

    Solving backward from `qs[0]` and reversing makes the seam the identity by construction,
    which is what `plan()` has always done for the descent. The path is frozen in
    `pose_p33.json` rather than re-solved, for P34's reason: `_dense` runs a CEM and the CEM
    is not bit-reproducible even under a fixed seed.

    The per-env adaptation is a linear ramp of the grasp offset `chain[0] - qs[0]`, zero at
    home and full at the far end, so the last command is exactly `chain[0]`.
    """
    d = chain0 - q_nom0.unsqueeze(0)
    seq = [appr_qs[i].unsqueeze(0) + d * (i / (len(appr_qs) - 1)) for i in range(len(appr_qs))]
    sub = max(1, total // len(seq))
    out, prev = [], K.q_arm0.unsqueeze(0).repeat(K.n, 1)
    for tgt in seq:
        out += [("approach", (1.0 - (s + 1) / sub) * prev + ((s + 1) / sub) * tgt, False)
                for s in range(sub)]
        prev = tgt
    return out
