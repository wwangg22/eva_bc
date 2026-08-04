# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""Stage-2 behaviour cloning for `Rebot-ClutterExtract-v0`.

Everything clutter-specific lives here. The vendored LeRobot transformer, the flow-matching
head, the normalizer and the ACT config are imported from `eva_bc/act/` **unmodified** --
those are tracked files and the standing constraint is that nothing outside
`eva_bc/clutter/` is touched.

What could NOT be reused, and why
---------------------------------
`act/dataset.py` hardcodes the pick-place observation layout as module-level constants
(`OBS_DIM = 41`, `ENV_STATE_SLICE = slice(16, 41)`). Clutter's policy observation is
**42-D**. Rebinding another module's globals at import time would work -- Python resolves
them at call time -- and is exactly the kind of action-at-a-distance that goes unnoticed
until it produces a wrong number. So `clutter/act/dataset.py` is a self-contained copy with
the clutter layout and without the four pick-place pool filters, which have no analogue here
(there is no `episode_kind`, no per-can `outcomes`).
"""
