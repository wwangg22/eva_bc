# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU validation of the x0-steering path. No Isaac Sim, no GPU, ~20 s.

EXP07 (pick-place) reached its result on the FIRST configured run and burned zero GPU runs,
where EXP06 burned two on an rl_games action-scaling default. `POSTMORTEM.md` section 9d
attributes that to pre-registered discipline, and the cheapest item on that list is a CPU
plumbing test run before any simulator boots. This is the slot equivalent.

Every check below fails *silently in a plausible direction* if it is wrong -- a mis-routed
per-env x0 does not raise, it trains a policy against the wrong env's steering and reports a
flat learning curve. The one that matters most is the bit-exactness gate:

    steer_x0 = zeros  MUST be action-for-action identical to  fixed_x0 = zeros

because EXP07's gate 1 (z=0 reproduces the frozen base episode-for-episode) is the thing that
makes the eventual number attributable to steering rather than to a wrapper artefact, and if
that identity is broken it is broken on CPU too -- 20 s here instead of 35 min of GPU there.

.. code-block:: bash

    python slot/scripts/test_steer_cpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SLOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLOT))

from slot_act.dataset import ACTION_DIM, ENV_STATE_DIM, OBS_DIM, STATE_DIM  # noqa: E402
from slot_act.eval_act import BatchedACTController  # noqa: E402
from slot_act.modeling_flow import FlowMatchingPolicy  # noqa: E402
from slot_act.noise import LAST_ACTION_SLICE, noise_sigmas  # noqa: E402
from slot_act.residual_core import (  # noqa: E402
    GRIP_CMD_DIM,
    LIFT_EE_MAX_DIST,
    LIFT_MIN_Z,
    SlotGraspBit,
)
from slot_act.steer_core import STEER_ACTION_DIM, STEER_OBS_DIM  # noqa: E402
from slot_act.train_flow import make_config  # noqa: E402

FAILS: list[str] = []
CHUNK, WINDOW, N = 50, 15, 6


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def build_controller() -> tuple[BatchedACTController, torch.Tensor]:
    """A real FlowMatchingPolicy (random weights) behind a real controller, on CPU."""
    torch.manual_seed(0)
    cfg = make_config(SimpleNamespace(chunk_size=CHUNK, n_action_steps=WINDOW,
                                      device="cpu", num_inference_steps=10))
    policy = FlowMatchingPolicy(cfg).eval()
    stats = {
        "observation.state": {"mean": torch.zeros(STATE_DIM), "std": torch.ones(STATE_DIM)},
        "observation.environment_state": {
            "mean": torch.zeros(ENV_STATE_DIM), "std": torch.ones(ENV_STATE_DIM)},
        "action": {"mean": torch.zeros(ACTION_DIM), "std": torch.ones(ACTION_DIM)},
    }
    ctrl = BatchedACTController(policy, stats, WINDOW, CHUNK, "cpu")
    torch.manual_seed(1)
    obs = torch.randn(N, OBS_DIM)
    return ctrl, obs


def roll(ctrl, obs, steps, flush_at=None, flush_ids=(2,)):
    """Run `steps` control steps, optionally flushing mid-window to desync the refills."""
    out = []
    for t in range(steps):
        if flush_at is not None and t == flush_at:
            ctrl.flush(list(flush_ids))
        out.append(ctrl.act(obs).clone())
    return torch.stack(out)


def main() -> int:
    print("\nx0-steering plumbing (CPU)\n")

    # ---------------------------------------------------------------- window arithmetic
    check("600-step episode is an exact multiple of the window",
          600 % WINDOW == 0, f"{600 // WINDOW} windows per episode")
    check("STEER_ACTION_DIM == ACTION_DIM (z has one column per action dim)",
          STEER_ACTION_DIM == ACTION_DIM, f"{STEER_ACTION_DIM} vs {ACTION_DIM}")
    check("STEER_OBS_DIM == OBS_DIM + 16-D feature tail",
          STEER_OBS_DIM == OBS_DIM + 16, f"{STEER_OBS_DIM} vs {OBS_DIM}+16")

    # ------------------------------------------------- x0 batching: (chunk,7) vs (B,chunk,7)
    # predict_action_chunk expands a 2-D x0 across the batch. The steering path always sends
    # 3-D. If the expand and the per-row path disagree, gate 1 can never be bit-exact.
    ctrl, obs = build_controller()
    x0_2d = torch.randn(CHUNK, ACTION_DIM)
    batch = ctrl.normalizer.normalize({
        "observation.state": obs[:, :STATE_DIM],
        "observation.environment_state": obs[:, STATE_DIM:],
    })
    a2 = ctrl.policy.predict_action_chunk(batch, x0=x0_2d)
    a3 = ctrl.policy.predict_action_chunk(batch, x0=x0_2d.unsqueeze(0).expand(N, -1, -1))
    check("x0 (chunk,7) and (B,chunk,7) paths are bit-identical for identical rows",
          torch.equal(a2, a3), f"max abs diff {(a2 - a3).abs().max():.3e}")

    # ------------------------------------------------------------ THE GATE: zeros == zeros
    # EXP07 gate 1, done on CPU. steer_x0=zeros must reproduce fixed_x0=zeros exactly,
    # INCLUDING across a mid-window flush that desyncs one env's refill boundary.
    z = torch.zeros(N, CHUNK, ACTION_DIM)
    ctrl_a, obs_a = build_controller()
    ctrl_a.fixed_x0 = torch.zeros(CHUNK, ACTION_DIM)
    ref = roll(ctrl_a, obs_a, 3 * WINDOW, flush_at=WINDOW + 4)

    ctrl_b, obs_b = build_controller()
    ctrl_b.steer_x0 = z
    got = roll(ctrl_b, obs_b, 3 * WINDOW, flush_at=WINDOW + 4)

    check("steer_x0=zeros is action-for-action identical to fixed_x0=zeros (incl. a "
          "desynced mid-window flush)",
          torch.equal(ref, got), f"max abs diff {(ref - got).abs().max():.3e}")

    # ------------------------------------------------- steer_x0 takes precedence over fixed_x0
    ctrl_c, obs_c = build_controller()
    ctrl_c.fixed_x0 = torch.full((CHUNK, ACTION_DIM), 5.0)   # would be very visible
    ctrl_c.steer_x0 = z
    got_c = roll(ctrl_c, obs_c, WINDOW)
    check("steer_x0 takes precedence over a conflicting fixed_x0",
          torch.equal(ref[:WINDOW], got_c), f"max abs diff {(ref[:WINDOW] - got_c).abs().max():.3e}")

    # ------------------------------------------------------------------- per-env routing
    # The failure this catches: steer_x0[empty] indexed with the wrong index set, so env k's
    # z lands on env j. Nothing raises; the run just does not learn.
    K = 3
    z_one = torch.zeros(N, CHUNK, ACTION_DIM)
    z_one[K] = 0.9
    ctrl_d, obs_d = build_controller()
    ctrl_d.steer_x0 = z_one
    got_d = roll(ctrl_d, obs_d, WINDOW)
    diff = (got_d - ref[:WINDOW]).abs().amax(dim=(0, 2))   # (N,) per-env max change
    check(f"steering only env {K} changes only env {K}",
          diff[K] > 1e-6 and torch.all(diff[torch.arange(N) != K] == 0),
          f"per-env max |delta| = {[round(float(v), 5) for v in diff]}")

    # ------------------------------------------------------ set_steer: alpha * tanh, broadcast
    core = SimpleNamespace(controller=ctrl_d, alpha_x0=1.0, device=torch.device("cpu"))
    from slot_act.steer_core import SteerCore
    zed = torch.randn(N, ACTION_DIM) * 3.0        # deliberately outside [-1, 1]
    SteerCore.set_steer(core, zed)
    applied = ctrl_d.steer_x0
    check("set_steer broadcasts z to (N, chunk, action_dim)",
          applied.shape == (N, CHUNK, ACTION_DIM), str(tuple(applied.shape)))
    check("set_steer is constant across chunk positions (broadcast, not per-position)",
          torch.equal(applied[:, 0, :].unsqueeze(1).expand(-1, CHUNK, -1), applied))
    check("set_steer applies alpha * tanh(z)",
          torch.allclose(applied[:, 0, :], torch.tanh(zed), atol=1e-6),
          f"max abs diff {(applied[:, 0, :] - torch.tanh(zed)).abs().max():.3e}")
    check("x0 stays inside +/-1 even for |z| >> 1 (tanh bound holds)",
          applied.abs().max() < 1.0, f"max |x0| = {applied.abs().max():.5f}")

    # alpha scales the bound
    core.alpha_x0 = 0.25
    SteerCore.set_steer(core, zed)
    check("alpha_x0 scales the bound", ctrl_d.steer_x0.abs().max() < 0.25,
          f"max |x0| = {ctrl_d.steer_x0.abs().max():.5f}")

    # --------------------------------------------------------------------- SlotGraspBit
    # Replaces the missing exp06_grasp_bit.pt (SESSION5_FINDINGS.md section 4). Faked env +
    # mdp: the only thing under test is the AND, the sign, and the threshold plumbing.
    lifted_flag = {"v": torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])}
    seen = {}

    def fake_block_lifted(env, minimal_height, ee_max_dist, ee_frame_cfg):
        seen.update(h=minimal_height, d=ee_max_dist, name=ee_frame_cfg.name)
        return lifted_flag["v"]

    bit = SlotGraspBit(SimpleNamespace(block_lifted=fake_block_lifted), torch.device("cpu"))
    o = torch.zeros(N, OBS_DIM)
    o[:, GRIP_CMD_DIM] = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0, -1.0])  # -1 = closed
    out = bit(o, env=None)
    #        lifted:   1    1    0    0    1    0
    #        closed:   Y    N    Y    N    Y    Y
    check("SlotGraspBit = block_lifted AND commanded-closed",
          torch.equal(out, torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])), str(out.tolist()))
    check("SlotGraspBit passes the env's own lifting thresholds through",
          (seen["h"], seen["d"]) == (LIFT_MIN_Z, LIFT_EE_MAX_DIST), str(seen))
    check("SlotGraspBit reads the ee_frame sensor", seen["name"] == "ee_frame", str(seen))
    check("SlotGraspBit returns float, one per env", out.dtype == torch.float32
          and out.shape == (N,), f"{out.dtype}, {tuple(out.shape)}")

    # the documented dead zone: closed on the block but not yet above 45 mm reads 0
    lifted_flag["v"] = torch.zeros(N)
    o[:, GRIP_CMD_DIM] = -1.0
    check("documented dead zone: closed-but-not-lifted reads 0 (not an error)",
          torch.equal(bit(o, env=None), torch.zeros(N)))

    # ---- 6. noise scales (EXP_STEER) -------------------------------------------------
    # The gate compares a rate measured by eval_act.py against one measured by
    # eval_steer.py. That only means something if "--action-noise 0.02" denotes the same
    # perturbation in both, so the sigma construction lives in ONE place and its semantics
    # are pinned here. A silent change (e.g. dropping the last_action mask) would not raise
    # anywhere -- it would just move a number the whole experiment is calibrated against.
    print("\n6. per-channel noise scales (slot_act/noise.py)")
    stats = {
        "observation.state": {"std": torch.arange(1.0, STATE_DIM + 1)},
        "observation.environment_state": {"std": torch.arange(100.0, 100.0 + ENV_STATE_DIM)},
        "action": {"std": torch.arange(1.0, ACTION_DIM + 1)},
    }
    check("magnitude 0 => no sigma at all (no perturbation object built)",
          noise_sigmas(stats, "cpu", 0.0, 0.0) == (None, None))
    os_, as_ = noise_sigmas(stats, "cpu", 0.2, 0.02, tag="test")
    check("obs sigma is the concatenated per-channel training std",
          os_.shape == (OBS_DIM,) and torch.equal(os_[:STATE_DIM], torch.arange(1.0, STATE_DIM + 1)),
          f"{tuple(os_.shape)}, head {os_[:3].tolist()}")
    check("obs sigma zeroes obs[27:34] = last_action (bookkeeping, not a sensor)",
          torch.equal(os_[LAST_ACTION_SLICE], torch.zeros(OBS_DIM - LAST_ACTION_SLICE.start))
          and bool((os_[:LAST_ACTION_SLICE.start] > 0).all()),
          f"tail {os_[LAST_ACTION_SLICE].tolist()}")
    check("action sigma is the per-channel action std, unmasked",
          as_.shape == (ACTION_DIM,) and torch.equal(as_, torch.arange(1.0, ACTION_DIM + 1)),
          str(as_.tolist()))
    check("sigmas are copies -- perturbing one harness cannot mutate the checkpoint stats",
          as_.data_ptr() != stats["action"]["std"].data_ptr())

    print()
    if FAILS:
        print(f"  {len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"    - {f}")
        print()
        return 1
    print("  STEERING PLUMBING OK\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
