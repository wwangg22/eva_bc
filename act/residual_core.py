"""EXP06 residual-RL core: frozen-base obs builder + residual blending.

Sim-side companion of ``BatchedACTController`` (act/eval_act.py). Per control step:

    obs64 = core.advance(obs41, env)     # flush check, chunk refill (peek), features
    full7 = core.compose(residual6)      # pops the peeked base action, blends arm dims
    env.step(full7)
    core.reset(done_env_ids)             # clear queues of reset envs

Residual observation layout (64-D, Big Will's spec 2026-08-02, EXP06_residual_rl.md):
    [0:41]   the base policy's own 41-D obs (raw env units)
    [41:48]  queued base action for THIS step (env action units, pre-residual)
    [48:49]  phase = index-in-execution-window / n_action_steps
    [49:53]  physical finger features re-surfaced: pos dims 6,7 + vel dims 14,15
    [53:54]  validated grasp-success bit (EXP01 variant-D MLP, exp06_grasp_bit.pt;
             ANDed with commanded-closed — see EXP06 gate 1)
    [54:61]  target-can pose in the gripper (ee_frame) frame: pos 3 + quat XYZW 4
    [61:64]  target-can -> basket delta (env-local): [bx-cx, by-cy, z_rest-cz]

Action: 6-D residual on the arm joints, ``arm = base_arm + alpha * tanh(res)`` in env
action units (1 unit = 0.5 rad); grip channel passes through from the base untouched.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

RES_OBS_DIM = 64
RES_ACTION_DIM = 6
FINGER_FEATURE_DIMS = [6, 7, 14, 15]
BIT_DIMS = [6, 7, 14, 15, 40]
CAN_REST_Z_IN_BASKET = 0.015  # can root height at rest on the basket floor (env-local)

# section 4.2 flush thresholds — identical to act/eval_act.py
FLUSH_POS_JUMP = 0.03
FLUSH_Z_DROP = 0.02
FLUSH_Z_ABOVE = 0.05


class GraspBit:
    """Runtime wrapper for experiments/exp06_grasp_bit.pt (EXP06 gate 1)."""

    def __init__(self, artifact_path: Path, device):
        art = torch.load(artifact_path, map_location=device, weights_only=False)
        if art["kind"] != "mlp":
            raise ValueError(f"unsupported grasp-bit artifact kind: {art['kind']}")
        self.mlp = nn.Sequential(nn.Linear(5, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.mlp.load_state_dict(art["state_dict"])
        self.mlp.to(device).eval()
        self.mu = art["mu"].to(device)
        self.sd = art["sd"].to(device)

    @torch.no_grad()
    def __call__(self, obs41: torch.Tensor) -> torch.Tensor:
        x = (obs41[:, BIT_DIMS] - self.mu) / self.sd
        logit = self.mlp(x).squeeze(-1)
        closed = obs41[:, 40] < 0  # last commanded grip: open command => not holding
        return ((logit > 0) & closed).float()


class ResidualCore:
    """Holds the frozen controller + feature machinery; owns no RL state."""

    def __init__(self, controller, mdp, grasp_bit_path: Path, alpha: float,
                 flush: bool = True):
        self.controller = controller
        self.mdp = mdp  # reBot_RL.tasks.manager_based.pick_place.mdp
        self.alpha = float(alpha)
        self.flush_enabled = flush
        self.device = controller.device
        self.grasp_bit = GraspBit(grasp_bit_path, self.device)
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
        """(N, num_cans, 3) fixed a/b order (canonical order can swap mid-episode)."""
        return torch.stack(
            [self.mdp.object_pos_local(env, name) for name in self.mdp.OBJECT_NAMES], dim=1
        )

    @torch.no_grad()
    def flush_check(self, env) -> None:
        """Section 4.2 discontinuity flush (identical semantics to eval_act.py loop).

        Call exactly once per env step, BEFORE the controller serves that step's action.
        """
        pos = self._can_pos_local(env)
        if self._just_reset is None:
            self._just_reset = torch.ones(pos.shape[0], dtype=torch.bool, device=self.device)
        if self.flush_enabled and self._prev_pos is not None:
            jump = torch.linalg.norm(pos - self._prev_pos, dim=-1) > FLUSH_POS_JUMP
            z_drop = ((self._prev_pos[..., 2] - pos[..., 2]) > FLUSH_Z_DROP) & (
                self._prev_pos[..., 2] > FLUSH_Z_ABOVE
            )
            trig = ((jump | z_drop).any(dim=1) & ~self._just_reset).nonzero(as_tuple=False).squeeze(-1)
            if trig.numel():
                self.controller.flush(trig.tolist())
                self.flush_count += int(trig.numel())
        self._prev_pos = pos
        self._just_reset[:] = False

    @torch.no_grad()
    def advance(self, obs41: torch.Tensor, env) -> torch.Tensor:
        """Build the 64-D residual observation; refills base queues (peek, no pop)."""
        n = obs41.shape[0]
        self.flush_check(env)

        base, phase = self.controller.peek(obs41)  # refills empty queues
        self._pending_base = base

        obs64 = torch.cat(
            [
                obs41,
                base,
                (phase.float() / self.controller.n_action_steps).unsqueeze(1),
                self.task_features(obs41, env),
            ],
            dim=1,
        )
        assert obs64.shape == (n, RES_OBS_DIM), obs64.shape
        return obs64

    @torch.no_grad()
    def task_features(self, obs41: torch.Tensor, env) -> torch.Tensor:
        """(N, 15) shared feature tail: fingers 4, grasp bit 1, can-in-gripper pose 7,
        can->basket delta 3. Used by both the 64-D residual obs (EXP06) and the 56-D
        steering obs (EXP07)."""
        from isaaclab.utils.math import subtract_frame_transforms

        # target-can selection — mirrors mdp.observations.objects_canonical exactly
        ee_w = env.scene["ee_frame"].data.target_pos_w.torch[..., 0, :]
        ee_q = env.scene["ee_frame"].data.target_quat_w.torch[..., 0, :]  # wxyz
        placed = self.mdp.placed_mask(env)
        pos_w, quat_w, dists = [], [], []
        for i, name in enumerate(self.mdp.OBJECT_NAMES):
            p = env.scene[name].data.root_pos_w.torch
            pos_w.append(p)
            quat_w.append(env.scene[name].data.root_quat_w.torch)
            d = torch.linalg.norm(p - ee_w, dim=1)
            dists.append(torch.where(placed[:, i], torch.full_like(d, torch.inf), d))
        target_is_b = (dists[1] < dists[0]).unsqueeze(1)
        tpos = torch.where(target_is_b, pos_w[1], pos_w[0])
        tquat = torch.where(target_is_b, quat_w[1], quat_w[0])

        rel_pos, rel_quat = subtract_frame_transforms(ee_w, ee_q, tpos, tquat)
        rel_quat_xyzw = rel_quat[:, [1, 2, 3, 0]]  # obs convention is XYZW

        centers = self.mdp.basket_centers_local(env)  # (N, 2) env-local
        tpos_local = tpos - env.scene.env_origins
        basket_delta = torch.stack(
            [
                centers[:, 0] - tpos_local[:, 0],
                centers[:, 1] - tpos_local[:, 1],
                torch.full_like(tpos_local[:, 2], CAN_REST_Z_IN_BASKET) - tpos_local[:, 2],
            ],
            dim=1,
        )

        return torch.cat(
            [
                obs41[:, FINGER_FEATURE_DIMS],
                self.grasp_bit(obs41).unsqueeze(1),
                rel_pos,
                rel_quat_xyzw,
                basket_delta,
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
