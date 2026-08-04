# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``mdp`` surface ``act/`` expects, backed by the challenge task's own mdp module.

``act/`` was written against ``reBot_RL.tasks.manager_based.pick_place.mdp`` and calls exactly
four things on it: ``OBJECT_NAMES``, ``object_pos_local``, ``placed_mask`` and
``basket_centers_local``. This shim supplies them for ``Rebot-PrecisionSlot-*`` so that
**nothing in `eva_rl` has to be edited** -- it is a shared asset repo carrying someone else's
authored task and an existing 87.9 % pick-place result.

Everything not defined here falls through to the challenge mdp via PEP 562 ``__getattr__``,
so ``slot_mdp.insertion_depth``, ``slot_mdp.yaw_of``, ``slot_mdp.SLOT_CENTER`` and friends all
work unchanged.

Two deliberate departures from a mechanical rename
--------------------------------------------------
**1. ``placed_mask`` does NOT wrap ``is_inserted``.** The env's own success predicate bounds
the block's height only from *below* (``z > SLOT_FLOOR_Z - 0.005``, there to catch a dropped
block) and has no upper bound, so it also passes for a block resting on top of the 30 mm slot
walls and for a block still dangling in a closed gripper. Measured during Stage A: one probe
cell scored **93.8 % with 13.28 mm mean lateral error** -- geometrically impossible for a 15 mm
half-width block inside a 16.5 mm half-width channel -- and another scored 100 % with a
33.86 mm finger gap (the pads jammed on the wall tops, gripping nothing) and the block at
z = 67 mm.

``eval_act.py`` reduces ``placed_mask(env).all(dim=1)`` into the headline success number, so
wiring the raw predicate in here would inflate **every evaluation number for the rest of the
project**. It is wrapped with the seated-height guard instead. The raw predicate stays
available as ``is_inserted`` for reporting the gap.

**2. ``basket_centers_local`` is deliberately absent.** Pick-place aims at a movable basket;
the slot is welded to the table at ``SLOT_CENTER``. Any call site that still asks for a basket
centre has not been ported and should raise ``AttributeError`` rather than silently receive a
constant. Use :func:`goal_delta` -- the same shape, the right semantics.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from reBot_RL.tasks.manager_based.challenge import mdp as _mdp

# Explicitly re-exported so readers and linters see the surface act/ actually touches.
from reBot_RL.tasks.manager_based.challenge.mdp import (  # noqa: F401
    BLOCK_HALF,
    SLOT_CENTER,
    SLOT_DEPTH,
    SLOT_FLOOR_Z,
    SUCCESS_DEPTH,
    SUCCESS_YAW,
    TCP_OFFSET,
    insertion_depth,
    is_inserted,
    lateral_error,
    object_pos_local,
    object_quat,
    yaw_error,
    yaw_of,
)

#: The single manipulable object. Pick-place had two, and its target-selection logic
#: (``dists[1] < dists[0]``, nearest-unplaced) collapses to "the object" here.
OBJECT_NAMES = ("block",)

#: Block centre height when the block is properly seated on the slot floor: 20 + 35 mm.
SEAT_Z = SLOT_FLOOR_Z + BLOCK_HALF[2]

#: Half-window on SEAT_Z. 6 mm is loose enough to accept the block resting anywhere on the
#: floor and tight enough to reject the two false positives above: the wall tops sit 30 mm
#: high (block centre 85 mm) and a carried block rides at ~62 mm.
SEAT_TOL = 0.006

#: x of the slot mouth. Depth is measured from here, on the block CENTRE.
MOUTH_X = SLOT_CENTER[0] - SLOT_DEPTH / 2

#: Largest block-centre x before the nose reaches the back stop.
MAX_INSERT_X = SLOT_CENTER[0] + SLOT_DEPTH / 2 - BLOCK_HALF[0]


def __getattr__(name: str):
    """Fall through to the challenge mdp for everything not overridden here (PEP 562)."""
    return getattr(_mdp, name)


def seated(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """(N,) bool -- inserted **and** actually sitting on the slot floor.

    This is the success predicate for every number this project reports. See the module
    docstring for why ``is_inserted`` alone is not usable.
    """
    z = object_pos_local(env, name)[:, 2]
    return is_inserted(env, name) & ((z - SEAT_Z).abs() < SEAT_TOL)


def placed_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(N, 1) bool, one column per object -- the shape ``act/`` expects.

    ``eval_act.py`` reduces this with ``.all(dim=1)`` and ``.sum(dim=1)``; returning a column
    vector for the single block keeps both valid with no further edits.
    """
    return seated(env).unsqueeze(1)


def goal_delta(env: ManagerBasedRLEnv, name: str = "block") -> torch.Tensor:
    """(N, 4) block-to-goal error: ``(dx_to_min_depth, dy_to_centreline, dz_to_seat, -yaw)``.

    The replacement for pick-place's basket delta
    ``[goal_x - obj_x, goal_y - obj_y, CAN_REST_Z_IN_BASKET - obj_z]``. Two changes matter:

    * the x target is the **minimum passing depth**, not a point target, because depth is
      measured on the block centre and the useful signal is "how much further in";
    * yaw is included, because a horizontal insertion into a 1.5 mm per-side channel fails on
      yaw in a way a top-down drop into a basket never does (``SUCCESS_YAW`` = 0.12 rad).
    """
    p = object_pos_local(env, name)
    yaw = yaw_of(object_quat(env, name))
    return torch.stack([(MOUTH_X + SUCCESS_DEPTH) - p[:, 0],
                        SLOT_CENTER[1] - p[:, 1],
                        SEAT_Z - p[:, 2],
                        -yaw], dim=-1)
