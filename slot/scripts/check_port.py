# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Static consistency check of the 41-D -> 34-D port. No Isaac Sim, no GPU, runs in a second.

The observation width appears as a constant in three files that never import each other's
numbers -- ``dataset.py`` (``OBS_DIM``), ``residual_core.py`` (``RES_OBS_DIM``, ``GRIP_CMD_DIM``)
and ``steer_core.py`` (``STEER_OBS_DIM``) -- and they are related only by arithmetic that a
human has to keep in their head. That is exactly the kind of coupling that survives a port
looking fine and then trains on a silently wrong layout.

Two of these were already caught the hard way: a bare ``obs41[:, 40]`` that no
``obs41 -> obs34`` rename could see, and a package-name collision that let ``import act``
resolve to the 41-D original with no error at all.

Run before any training run:

.. code-block:: bash

    python slot/scripts/check_port.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SLOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLOT))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> int:
    import slot_act
    import slot_act.dataset as D
    import slot_act.residual_core as R
    import slot_act.steer_core as S

    print("\nport consistency\n")

    # --- the package must be the port, not the tracked original
    pkg = Path(slot_act.__file__).resolve().parent
    check("slot_act resolves inside slot/", pkg.parent.name == "slot", str(pkg))
    check("the tracked eva_bc/act/ is NOT what got imported",
          "eva_bc/act" not in str(pkg).replace("\\", "/"), str(pkg))

    # --- observation layout arithmetic
    check("OBS_DIM == 34", D.OBS_DIM == 34, f"got {D.OBS_DIM}")
    check("STATE_SLICE + ENV_STATE_SLICE tile OBS_DIM exactly",
          D.STATE_SLICE.start == 0 and D.STATE_SLICE.stop == D.ENV_STATE_SLICE.start
          and D.ENV_STATE_SLICE.stop == D.OBS_DIM,
          f"{D.STATE_SLICE} + {D.ENV_STATE_SLICE} vs {D.OBS_DIM}")
    check("STATE_DIM matches STATE_SLICE",
          D.STATE_DIM == D.STATE_SLICE.stop - D.STATE_SLICE.start, f"got {D.STATE_DIM}")
    check("ENV_STATE_DIM matches ENV_STATE_SLICE",
          D.ENV_STATE_DIM == D.ENV_STATE_SLICE.stop - D.ENV_STATE_SLICE.start,
          f"got {D.ENV_STATE_DIM}")

    # --- the bare-integer trap: the commanded-grip channel is the LAST obs dim
    check("GRIP_CMD_DIM is the last observation dim (commanded grip)",
          R.GRIP_CMD_DIM == D.OBS_DIM - 1, f"GRIP_CMD_DIM={R.GRIP_CMD_DIM}, OBS_DIM={D.OBS_DIM}")
    check("every finger feature dim is in range",
          all(0 <= d < D.OBS_DIM for d in R.FINGER_FEATURE_DIMS), str(R.FINGER_FEATURE_DIMS))
    # SlotGraspBit mirrors precision_slot_env_cfg.RewardsCfg.lifting; if the env ever retunes
    # those, this feature and the reward the RL optimises would silently disagree.
    check("SlotGraspBit thresholds match the env's lifting reward term",
          (R.LIFT_MIN_Z, R.LIFT_EE_MAX_DIST) == (0.045, 0.08),
          f"{R.LIFT_MIN_Z}, {R.LIFT_EE_MAX_DIST}")

    # --- residual / steering widths, which are pure arithmetic over the feature tail
    tail = 16   # fingers 4 + grasp bit 1 + block-in-gripper pose 7 + signed goal delta 4
    check("RES_OBS_DIM == OBS_DIM + 7 + 1 + tail",
          R.RES_OBS_DIM == D.OBS_DIM + D.ACTION_DIM + 1 + tail,
          f"{R.RES_OBS_DIM} vs {D.OBS_DIM}+{D.ACTION_DIM}+1+{tail}")
    check("STEER_OBS_DIM == OBS_DIM + tail",
          S.STEER_OBS_DIM == D.OBS_DIM + tail, f"{S.STEER_OBS_DIM} vs {D.OBS_DIM}+{tail}")
    check("RES_ACTION_DIM == ACTION_DIM - 1 (arm only, grip passes through)",
          R.RES_ACTION_DIM == D.ACTION_DIM - 1, f"{R.RES_ACTION_DIM} vs {D.ACTION_DIM}")
    check("STEER_ACTION_DIM == ACTION_DIM (z has one column per action dim)",
          S.STEER_ACTION_DIM == D.ACTION_DIM, f"{S.STEER_ACTION_DIM} vs {D.ACTION_DIM}")

    # --- the flush rule this task deliberately does not use
    check("z-drop flush constants are gone from residual_core",
          not hasattr(R, "FLUSH_Z_DROP") and not hasattr(R, "FLUSH_Z_ABOVE"),
          "the block's legitimate 8 mm settle on release would trigger them")

    # --- window alignment: 600-step episode, 15-step execution window
    check("600-step episode is an exact multiple of the 15-step window", 600 % 15 == 0,
          f"{600 // 15} windows")

    # --- no pick-place task ids or mdp imports survive
    stale = []
    for f in sorted((SLOT / "slot_act").glob("*.py")):
        txt = f.read_text()
        for pat in ("Rebot-PickPlace", "manager_based.pick_place", "basket_centers_local"):
            if pat in txt:
                stale.append(f"{f.name}:{pat}")
    check("no pick-place task id / mdp path / basket_centers_local remains",
          not stale, "; ".join(stale) if stale else "clean")

    print(f"\n{'PORT OK' if not FAILS else 'PORT BROKEN: ' + ', '.join(FAILS)}\n")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
