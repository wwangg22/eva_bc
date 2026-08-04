"""EXP06 residual-RL core: frozen-base obs builder + residual blending.

Sim-side companion of ``BatchedACTController`` (act/eval_act.py). Per control step:

    obs58 = core.advance(obs34, env)     # flush check, chunk refill (peek), features
    full7 = core.compose(residual6)      # pops the peeked base action, blends arm dims
    env.step(full7)
    core.reset(done_env_ids)             # clear queues of reset envs

Residual observation layout (**58-D** for the slot port; was 64-D for pick-place):
    [0:34]   the base policy's own 34-D obs (raw env units)
    [34:41]  queued base action for THIS step (env action units, pre-residual)
    [41:42]  phase = index-in-execution-window / n_action_steps
    [42:46]  physical finger features re-surfaced: pos dims 6,7 + vel dims 14,15
    [46:47]  grasp bit — ANALYTIC here, not the EXP06 MLP (see :class:`SlotGraspBit`)
    [47:54]  block pose in the gripper (ee_frame) frame: pos 3 + quat XYZW 4
    [54:58]  **signed** block -> goal delta (env-local), see below

Why the tail is 16-D here and not 15-D (so 58/50, not the 57/49 in PORT_MAP section 2)
--------------------------------------------------------------------------------------
Pick-place's 3-D basket delta becomes a **4-D** slot-frame goal delta, because yaw is
load-bearing for a horizontal insertion into a 1.5 mm per-side channel in a way it never was
for dropping a can into a basket (``SUCCESS_YAW`` = 0.12 rad is a real failure mode).

The signs are the point. The env's own observation already carries the slot-frame error at
``obs[23:27]``, but ``mdp.lateral_error`` and ``mdp.yaw_error`` are **absolute values** — they
tell the policy how wrong it is and not which way to correct. ``slot_mdp.goal_delta`` returns
the signed quantities. (The depth component is an affine function of ``obs[23]`` and is
genuinely redundant; it is kept so the four channels stay one coherent vector.)

Action: 6-D residual on the arm joints, ``arm = base_arm + alpha * tanh(res)`` in env
action units (1 unit = 0.5 rad); grip channel passes through from the base untouched.
"""

from __future__ import annotations

import torch

RES_OBS_DIM = 58
RES_ACTION_DIM = 6
FINGER_FEATURE_DIMS = [6, 7, 14, 15]
GRIP_CMD_DIM = 33   # last dim of last_action = the commanded grip channel; -1 = closed

# block_lifted's parameters, copied from precision_slot_env_cfg.RewardsCfg.lifting so the
# feature and the env's own reward term cannot disagree.
LIFT_MIN_Z = 0.045
LIFT_EE_MAX_DIST = 0.08

# Discontinuity flush. The position-jump half ports directly: a 30 mm jump of the block
# between two 20 ms control steps is a dropped or flung block, not motion.
FLUSH_POS_JUMP = 0.03

# The pick-place z-drop half (`prev_z - z > 0.02` while `prev_z > 0.05`) is DELIBERATELY
# DROPPED. It was a gravity-drop slip proxy sized against a 40 mm basket rim. Here the block
# legitimately descends ~8 mm the instant the fingers open at the insert pose (measured: it
# rides at z = 63 mm while carried and settles to 55 mm on release, from a carry height above
# the 50 mm wall tops). That is the single most precision-critical moment in the episode, and
# the pick-place rule would fire a chunk flush exactly there — discarding the committed action
# chunk when chunk commitment is the load-bearing part of this policy (shortening the
# execution horizon collapsed success 59.4 -> 32.8 -> 3.1 -> 0 %).


class SlotGraspBit:
    """Analytic "the gripper is holding the block" bit. NOT the EXP06 MLP.

    EXP06 learned this from ``[6, 7, 14, 15, 33]`` (finger pos + vel + grip command) and
    exported ``experiments/exp06_grasp_bit.pt``. Two independent reasons that recipe cannot
    be reused here, both measured -- see docs/slot/SESSION5_FINDINGS.md section 4:

    1. **The artifact does not exist.** Only ``exp06_grasp_bit.py`` was ever committed to
       eva_bc; the ``.pt`` is in neither repo. Six call sites pointed at it, so every
       steering/residual entry point died on startup after paying Isaac Sim's boot cost.
    2. **The finger channels carry no grasp signal on this robot.** Over 707 802 frames
       where the block is provably held, the summed finger position is -0.04889 +/- 0.00003;
       over the closed-but-not-lifted frames it is -0.04887 +/- 0.00005. A 2e-5 difference
       against a 3e-5 spread: the fingers reach the same saturated closed position whether or
       not a 30 mm block is between them, so they cannot report the block. And the expert
       never fails to grasp (61/61 failures are ``release=unseated``), so there is no
       negative class in 2038 demos to train against even in principle.

    What replaces it is the env's OWN ``block_lifted`` reward term -- block off the table and
    still near the gripper -- ANDed with commanded-closed. Calling the env's function rather
    than reimplementing the threshold is deliberate: the feature then cannot drift from the
    reward the RL is being trained against. The commanded-closed conjunct is the one part of
    EXP06 that does port; it is what took the false-positive rate there from 27.1 % to 0 %,
    and the sign is verified against real demo data (``actions[:, 6]`` transitions at steps
    40 and 464; -1 = closed; ``obs[t, 33] == actions[t-1, 6]`` exactly).

    KNOWN DEAD ZONE, stated rather than hidden: between the gripper closing (t ~ 40) and the
    block clearing 45 mm (t ~ 85) this reads 0 while the block IS held. "Closed on the block
    but still on the table" is genuinely not a completed grasp, and the grasp phase is not
    where this task fails, so the dead zone is accepted.
    """

    def __init__(self, mdp, device):
        from isaaclab.managers import SceneEntityCfg

        self.mdp = mdp
        self.device = device
        self._ee_cfg = SceneEntityCfg("ee_frame")  # only .name is read; no resolve() needed

    @torch.no_grad()
    def __call__(self, obs34: torch.Tensor, env) -> torch.Tensor:
        lifted = self.mdp.block_lifted(
            env, minimal_height=LIFT_MIN_Z, ee_max_dist=LIFT_EE_MAX_DIST,
            ee_frame_cfg=self._ee_cfg,
        ) > 0
        closed = obs34[:, GRIP_CMD_DIM] < 0  # open command => not holding
        return (lifted.to(closed.device) & closed).float()


class ResidualCore:
    """Holds the frozen controller + feature machinery; owns no RL state."""

    def __init__(self, controller, mdp, alpha: float, flush: bool = True):
        self.controller = controller
        self.mdp = mdp  # slot/slot_mdp.py -- the challenge mdp plus the act/ adapters
        self.alpha = float(alpha)
        self.flush_enabled = flush
        self.device = controller.device
        self.grasp_bit = SlotGraspBit(mdp, self.device)
        self._pending_base: torch.Tensor | None = None
        self._prev_pos: torch.Tensor | None = None
        self._just_reset: torch.Tensor | None = None
        self.flush_count = 0

    def reset(self, env_ids) -> None:
        """Episode reset for the given envs: clear queues, suppress next flush."""
        self.controller.reset(env_ids)
        if self._just_reset is not None:
            for i in env_ids:
                self._just_reset[int(i)] = True

    def _can_pos_local(self, env) -> torch.Tensor:
        """(N, 1, 3). One object here, so the pick-place note about the canonical order
        swapping mid-episode no longer applies; the object axis is kept so that flush_check
        and every caller stay shape-compatible."""
        return torch.stack(
            [self.mdp.object_pos_local(env, name) for name in self.mdp.OBJECT_NAMES], dim=1
        )

    @torch.no_grad()
    def flush_check(self, env) -> None:
        """Discontinuity flush. NO LONGER identical to eval_act.py: the z-drop half is gone
        (module docstring), so this is position-jump only.

        Call exactly once per env step, BEFORE the controller serves that step's action.
        """
        pos = self._can_pos_local(env)
        if self._just_reset is None:
            self._just_reset = torch.ones(pos.shape[0], dtype=torch.bool, device=self.device)
        if self.flush_enabled and self._prev_pos is not None:
            # Position jump only — the z-drop half is intentionally absent, see the module
            # docstring: the block's legitimate 8 mm settle on release would trigger it.
            jump = torch.linalg.norm(pos - self._prev_pos, dim=-1) > FLUSH_POS_JUMP
            trig = (jump.any(dim=1) & ~self._just_reset).nonzero(as_tuple=False).squeeze(-1)
            if trig.numel():
                self.controller.flush(trig.tolist())
                self.flush_count += int(trig.numel())
        self._prev_pos = pos
        self._just_reset[:] = False

    @torch.no_grad()
    def advance(self, obs34: torch.Tensor, env) -> torch.Tensor:
        """Build the 58-D residual observation; refills base queues (peek, no pop)."""
        n = obs34.shape[0]
        self.flush_check(env)

        base, phase = self.controller.peek(obs34)  # refills empty queues
        self._pending_base = base

        obs58 = torch.cat(
            [
                obs34,
                base,
                (phase.float() / self.controller.n_action_steps).unsqueeze(1),
                self.task_features(obs34, env),
            ],
            dim=1,
        )
        assert obs58.shape == (n, RES_OBS_DIM), obs58.shape
        return obs58

    @torch.no_grad()
    def task_features(self, obs34: torch.Tensor, env) -> torch.Tensor:
        """(N, 16) shared feature tail: fingers 4, grasp bit 1, block-in-gripper pose 7,
        signed block->goal delta 4. Used by both the 58-D residual obs and the 50-D steering
        obs, whose widths are asserted against ``RES_OBS_DIM`` / ``STEER_OBS_DIM`` — so if this
        tail's composition changes, both constants must be adjusted by hand.

        Pick-place selected a target from two cans with a nearest-unplaced rule and a two-way
        ``torch.where`` on ``dists[1] < dists[0]``. There is exactly one object here, so the
        whole selection collapses; ``placed_mask`` is no longer needed to break the tie.
        """
        from isaaclab.utils.math import subtract_frame_transforms

        ee_w = env.scene["ee_frame"].data.target_pos_w.torch[..., 0, :]
        ee_q = env.scene["ee_frame"].data.target_quat_w.torch[..., 0, :]  # wxyz
        name = self.mdp.OBJECT_NAMES[0]
        tpos = env.scene[name].data.root_pos_w.torch
        tquat = env.scene[name].data.root_quat_w.torch

        rel_pos, rel_quat = subtract_frame_transforms(ee_w, ee_q, tpos, tquat)
        rel_quat_xyzw = rel_quat[:, [1, 2, 3, 0]]  # obs convention is XYZW

        return torch.cat(
            [
                obs34[:, FINGER_FEATURE_DIMS],
                self.grasp_bit(obs34, env).unsqueeze(1),
                rel_pos,
                rel_quat_xyzw,
                self.mdp.goal_delta(env, name),
            ],
            dim=1,
        )

    @torch.no_grad()
    def compose(self, residual6: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Blend the residual into the pending base action; consumes the peeked base.

        Returns (full 7-D env action, applied residual (N, 6) for the reward penalty).
        """
        base = self.controller.pop()
        assert self._pending_base is not None and torch.equal(base, self._pending_base)
        self._pending_base = None
        applied = self.alpha * torch.tanh(residual6.to(base.device, dtype=base.dtype))
        full = base.clone()
        full[:, :RES_ACTION_DIM] += applied
        return full, applied
