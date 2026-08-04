"""Vendored LeRobot ACT policy (state-only) + reBot HDF5 dataset. See PROVENANCE.md.

**Slot-insertion port.** This is a copy of ``eva_bc/act/`` retargeted from the 41-D pick-place
observation to the 34-D ``Rebot-PrecisionSlot-*`` observation. The originals are git-tracked
and are not edited.

**Why this package is called ``slot_act`` and not ``act``.** The obvious copy keeps the name,
and that is a trap. Both directories would then be importable as ``act`` and which one wins is
decided by ``sys.path`` order, so a stray ``PYTHONPATH``, a ``python -m act.train_flow``
launched from ``eva_bc/``, or an ``act`` already imported earlier in the same process binds the
**41-D pick-place constants** to 34-D slot data. This was measured, not theorised:

.. code-block:: text

    cd /tmp && PYTHONPATH=.../eva_bc python -c "import act, act.dataset as D; print(D.OBS_DIM)"
    41

No exception, no warning -- just the wrong observation width, which is precisely the
"self-consistent rather than correct" failure this project has already hit twice. A guard
inside the copy cannot help, because in the failing case the copy is never imported at all.
Renaming removes the ambiguity instead of detecting it: there is no ``slot_act`` in ``eva_bc``.
"""
