# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Batched damped-least-squares IK over the 6 arm joints of the reBot arm.

Why this exists
---------------
eva_bc's Stage-1 expert is built on cuRobo, which is **not installed** in
``env_isaaclab6``. The planner surface it actually uses is tiny -- the runner only needs a
``(T, 6)`` joint array at 50 Hz -- so the replacement is a solver, not a motion-planning
stack.

It is specifically *not* CEM. The Stage-1 sweep in ``analysis/insertion_feasibility.py``
measured CEM silently jumping IK branch between two adjacent waypoints (158 mm of tracking
error with the arm in a mirrored elbow configuration), and the offending cell **moved between
two runs of the same script** -- 0.078 in one, 0.090 in the next. A solver that is not
reproducible cannot generate demonstrations. The fixes here are structural:

* **full 6-DOF pose targets.** Position alone is not a sufficient specification for this arm:
  an unconstrained wrist arrives holding the block along its 45 mm length instead of across
  its 30 mm width. CEM patched this with a soft finger-axis cost; DLS constrains the whole
  orientation, which also pins the elbow branch.
* **warm starting.** ``solve_path`` threads each waypoint's solution into the next seed, so
  the trajectory is continuous by construction rather than by luck.
* **per-env convergence reporting.** Every solve returns achieved position/rotation error and
  a ``converged`` mask. Callers reject non-converged envs instead of executing garbage.

Jacobian
--------
``root_physx_view.get_jacobians()`` exists here and returns ``(N, 10, 6, 14)`` -- the arm
reports ``is_fixed_base = False``, so the 14 columns are 6 floating-base DOFs followed by the
8 joints, and the body dimension is *not* offset. That layout is asserted at runtime against a
finite-difference Jacobian rather than trusted: PhysX DOF order need not match Isaac Lab's
sorted ``joint_names``, and a silently permuted Jacobian still converges, just to the wrong
branch. If the check fails the solver falls back to finite differences (7 ``sim.forward()``
calls per iteration instead of 1) and says so.

Quaternions are ``(x, y, z, w)`` in this build -- measured in ``analysis/conventions.py``,
where writing XYZW reads back unchanged and ``mdp.yaw_of`` and ``quat_apply`` agree on the
same tensor. Only ``quat_apply`` is used below, so no second convention is assumed: rotation
error is taken from rotation-matrix columns built with ``quat_apply``.

.. warning::
    Every method here calls ``write_joint_state_to_sim``, which **teleports the arm and
    resets the fingers**. Never run IK while an object is held -- plan the whole path first,
    then execute. Getting this ordering wrong once took a trial from 0 % to 100 %.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FINGER_JOINTS = ("joint_left", "joint_right")
END_BODY = "gripper_end"


def _skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
        ],
        dim=-2,
    )


def _rot_from_quat(q: torch.Tensor) -> torch.Tensor:
    """(n, 4) XYZW -> (n, 3, 3) with the basis vectors as COLUMNS.

    Built only from ``quat_apply`` so this file assumes exactly one measured convention.
    """
    n = q.shape[0]
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    cols = [quat_apply(q, eye[i].expand(n, 3)) for i in range(3)]
    return torch.stack(cols, dim=2)


def rot_err_vec(r_cur: torch.Tensor, r_tgt: torch.Tensor) -> torch.Tensor:
    """World-frame rotation vector taking ``r_cur`` to ``r_tgt``. (n,3,3),(n,3,3) -> (n,3)."""
    d = r_tgt @ r_cur.transpose(1, 2)
    v = 0.5 * torch.stack(
        [d[:, 2, 1] - d[:, 1, 2], d[:, 0, 2] - d[:, 2, 0], d[:, 1, 0] - d[:, 0, 1]], dim=1
    )
    # 0.5*(R - R^T) saturates at 90 deg and reverses beyond it; rescale by the true angle so
    # large initial errors still point the right way.
    cos = ((d[:, 0, 0] + d[:, 1, 1] + d[:, 2, 2]) - 1.0).mul(0.5).clamp(-1.0, 1.0)
    ang = torch.acos(cos)
    s = torch.sin(ang)
    scale = torch.where(s.abs() > 1e-6, ang / s.clamp(min=1e-6), torch.ones_like(ang))
    return v * scale.unsqueeze(1)


class ArmIK:
    """Damped-least-squares IK for the 6 arm joints, solved in parallel across envs.

    Args:
        env: the ``ManagerBasedRLEnv`` (wrapped or not).
        tcp_offset: TCP position in the ``gripper_end`` frame. Defaults to
            ``mdp.TCP_OFFSET`` = (-0.0419, 0, 0) -- measured, and *not* the -0.075 inherited
            from the lift task.
        damping: Levenberg damping lambda. 0.05 is stable through the wrist singularity;
            the arm's Jacobian condition number at the home pose is only 4.2.
        rot_weight: metres-per-radian weighting so position and orientation are commensurate
            in the stacked task vector.
    """

    def __init__(
        self,
        env,
        tcp_offset: tuple[float, float, float] = (-0.0419, 0.0, 0.0),
        damping: float = 0.05,
        rot_weight: float = 0.05,
        pos_tol: float = 5.0e-4,
        rot_tol: float = 5.0e-3,
    ) -> None:
        self.e = env.unwrapped
        self.robot = self.e.scene["robot"]
        self.dev = self.e.device
        self.n = self.e.num_envs
        self.damping = damping
        self.rot_weight = rot_weight
        self.pos_tol = pos_tol
        self.rot_tol = rot_tol

        jn = list(self.robot.joint_names)
        self.arm_dof = [jn.index(j) for j in ARM_JOINTS]
        self.fing_dof = [jn.index(j) for j in FINGER_JOINTS]
        self.end_idx = self.robot.body_names.index(END_BODY)
        self.left_idx = self.robot.body_names.index("gripper_left")
        self.right_idx = self.robot.body_names.index("gripper_right")
        self.q_default = torch.as_tensor(self.robot.data.default_joint_pos[0], device=self.dev).clone()
        lim = torch.as_tensor(self.robot.data.joint_pos_limits[0], device=self.dev)
        self.lo = lim[self.arm_dof, 0]
        self.hi = lim[self.arm_dof, 1]
        self.offs = torch.tensor(tcp_offset, device=self.dev)
        self.jac_mode: str | None = None
        self._jac_cols: list[int] | None = None

    # ------------------------------------------------------------------ forward kinematics
    def fk(self, q_arm: torch.Tensor, q_fing: float = 0.016) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teleport to ``q_arm`` and read back (tcp (n,3) env-local, finger axis (n,3), R (n,3,3)).

        The finger axis is ``normalize(gripper_left - gripper_right)`` -- the direction the
        pads squeeze along. It is the orientation quantity the task actually cares about: an
        unconstrained wrist arrives holding the block along its 45 mm length instead of
        across its 30 mm width.
        """
        q = self.q_default.unsqueeze(0).repeat(self.n, 1)
        q[:, self.arm_dof] = q_arm
        q[:, self.fing_dof] = q_fing
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.e.sim.forward()
        self.robot.update(0.0)
        bp = torch.as_tensor(self.robot.data.body_pos_w.torch, device=self.dev)
        bp = bp - self.e.scene.env_origins.unsqueeze(1)
        bq = torch.as_tensor(self.robot.data.body_quat_w.torch, device=self.dev)
        r = _rot_from_quat(bq[:, self.end_idx, :])
        tcp = bp[:, self.end_idx, :] + torch.einsum("nij,j->ni", r, self.offs)
        sep = bp[:, self.left_idx, :] - bp[:, self.right_idx, :]
        return tcp, sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9), r

    def tcp_now(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Achieved (tcp, finger axis, R) from the CURRENT physics state -- no joint writes,
        so a held object survives. Use this during execution; ``fk`` is for planning only."""
        bp = torch.as_tensor(self.robot.data.body_pos_w.torch, device=self.dev)
        bp = bp - self.e.scene.env_origins.unsqueeze(1)
        bq = torch.as_tensor(self.robot.data.body_quat_w.torch, device=self.dev)
        r = _rot_from_quat(bq[:, self.end_idx, :])
        sep = bp[:, self.left_idx, :] - bp[:, self.right_idx, :]
        return (bp[:, self.end_idx, :] + torch.einsum("nij,j->ni", r, self.offs),
                sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9), r)

    # ---------------------------------------------------------------------- Jacobian
    def _fd_jacobian(self, q_arm: torch.Tensor, q_fing: float, eps: float = 1e-4) -> torch.Tensor:
        """(n, 6, 6) finite-difference Jacobian of [tcp; finger axis] w.r.t. the arm joints."""
        p0, u0, _ = self.fk(q_arm, q_fing)
        cols = []
        for j in range(6):
            qp = q_arm.clone()
            qp[:, j] += eps
            p1, u1, _ = self.fk(qp, q_fing)
            cols.append(torch.cat([(p1 - p0) / eps, (u1 - u0) / eps], dim=1))
        return torch.stack(cols, dim=2)

    def _analytic_jacobian(self, r_end: torch.Tensor, axis: torch.Tensor) -> torch.Tensor | None:
        """(n, 6, 6) PhysX Jacobian mapped to [TCP velocity; finger-axis rate], or None."""
        if self._jac_cols is None:
            return None
        j = torch.as_tensor(self.robot.root_physx_view.get_jacobians(), device=self.dev)
        jv = j[:, self.end_idx, :3, :][:, :, self._jac_cols]
        jw = j[:, self.end_idx, 3:, :][:, :, self._jac_cols]
        # v_tcp = v_link + w x (R * offset);  du/dt = w x u = -skew(u) w
        rad = torch.einsum("nij,j->ni", r_end, self.offs)
        return torch.cat([jv - _skew(rad) @ jw, -_skew(axis) @ jw], dim=1)

    def calibrate_jacobian(self, q_arm: torch.Tensor, q_fing: float = 0.016, tol: float = 0.05) -> str:
        """Decide between the analytic and finite-difference Jacobian, by measurement.

        Tries the documented column layout (6 base DOFs, then joints in ``joint_names``
        order); if that disagrees with finite differences by more than ``tol`` relative, it
        searches for the permutation that matches, and failing that falls back to FD.
        """
        fd = self._fd_jacobian(q_arm, q_fing)
        _, axis, r_end = self.fk(q_arm, q_fing)
        try:
            j = torch.as_tensor(self.robot.root_physx_view.get_jacobians(), device=self.dev)
        except Exception as exc:  # noqa: BLE001
            self.jac_mode = f"finite-difference (get_jacobians unavailable: {exc!r})"
            return self.jac_mode
        ncol = j.shape[-1]
        base = ncol - len(self.robot.joint_names)  # 6 if floating, 0 if fixed

        def rel(cols: list[int]) -> float:
            self._jac_cols = cols
            an = self._analytic_jacobian(r_end, axis)
            scale = fd.abs().amax().clamp(min=1e-9)
            return float((an - fd).abs().amax() / scale)

        cand = [base + d for d in self.arm_dof]
        err = rel(cand)
        if err <= tol:
            self.jac_mode = f"analytic (cols {cand}, rel err {err:.4f})"
            return self.jac_mode
        # Column-by-column match against finite differences.
        self._jac_cols = list(range(ncol))
        full = self._analytic_jacobian(r_end, axis)
        picked, used = [], set()
        for c in range(6):
            d = (full - fd[:, :, c : c + 1]).abs().amax(dim=(0, 1))
            for idx in torch.argsort(d).tolist():
                if idx not in used:
                    picked.append(idx)
                    used.add(idx)
                    break
        err2 = rel(picked)
        if err2 <= tol:
            self.jac_mode = f"analytic (PERMUTED cols {picked}, rel err {err2:.4f})"
            return self.jac_mode
        self._jac_cols = None
        self.jac_mode = f"finite-difference (analytic mismatch {err:.3f}, best perm {err2:.3f})"
        return self.jac_mode

    # ---------------------------------------------------------------------- solve
    def solve(
        self,
        tgt_pos: torch.Tensor,
        tgt_axis: torch.Tensor,
        q_seed: torch.Tensor,
        q_fing: float = 0.016,
        iters: int = 60,
        q_bias: torch.Tensor | None = None,
        null_gain: float = 0.05,
    ) -> dict:
        """Solve for the 6 arm joints putting the TCP at ``tgt_pos`` (n,3) with the finger
        axis along ``tgt_axis`` (n,3, unit).

        This is a **5-DOF task**, not 6: position plus a direction. Pinning the full
        orientation was measured to over-constrain the arm -- demanding an exact wrist roll
        left 3.6 mm of residual position error at the nominal pose and 15.6 mm along a path,
        with 0/64 envs converged, because a 6R arm reaches an isolated set of full poses but a
        continuum of (position, axis) pairs. The leftover roll about the finger axis is
        genuinely free for this task, so it is steered by a nullspace bias toward ``q_bias``
        (default: the seed) rather than constrained. That keeps the wrist near a
        known-wall-clearing configuration while letting position be met exactly.

        Returns ``q`` (n,6), ``pos_err`` (n,) [m], ``axis_err`` (n,) [1 - |cos|],
        ``converged`` (n,) bool. ``q`` is the **best iterate seen per env**, so a solver that
        wanders late cannot return something worse than its own best.
        """
        if self.jac_mode is None:
            self.calibrate_jacobian(q_seed, q_fing)
        q = q_seed.clone().clamp(self.lo, self.hi)
        bias = (q_seed if q_bias is None else q_bias).clone()
        best_q = q.clone()
        best_c = torch.full((self.n,), float("inf"), device=self.dev)
        w = self.rot_weight
        eye6 = torch.eye(6, device=self.dev)
        for it in range(iters):
            p, u, r = self.fk(q, q_fing)
            ep = tgt_pos - p
            eu = tgt_axis - u
            cost = ep.norm(dim=1) + w * eu.norm(dim=1)
            upd = cost < best_c
            best_c = torch.where(upd, cost, best_c)
            best_q[upd] = q[upd]
            jac = self._analytic_jacobian(r, u)
            if jac is None:
                jac = self._fd_jacobian(q, q_fing)
            jw = jac.clone()
            jw[:, 3:, :] *= w
            err = torch.cat([ep, w * eu], dim=1).unsqueeze(2)          # (n,6,1)
            jjt = jw @ jw.transpose(1, 2) + (self.damping**2) * eye6.expand_as(jw @ jw.transpose(1, 2))
            pinv = jw.transpose(1, 2) @ torch.linalg.inv(jjt)          # (n,6,6)
            dq = (pinv @ err).squeeze(2)
            # The bias gain DECAYS to zero. With Levenberg damping, (I - J^+ J) is only an
            # approximate nullspace projector, so a constant bias leaks into task space and
            # fights position convergence -- measured: 23 % of a straight 74 mm push's
            # waypoints stuck above 0.5 mm, 17/64 envs converged. Decaying it keeps the
            # branch guidance early, where it decides the elbow, and leaves the endgame to
            # the task term alone.
            g = null_gain * (1.0 - it / max(1, iters - 1))
            null = (eye6.expand(self.n, 6, 6) - pinv @ jw) @ (bias - q).unsqueeze(2)
            q = (q + (dq + g * null.squeeze(2)).clamp(-0.20, 0.20)).clamp(self.lo, self.hi)
        p, u, _ = self.fk(best_q, q_fing)
        pe = (tgt_pos - p).norm(dim=1)
        ae = 1.0 - (u * tgt_axis).sum(dim=1).abs()
        return {
            "q": best_q,
            "pos_err": pe,
            "axis_err": ae,
            "converged": (pe < self.pos_tol) & (ae < self.rot_tol),
        }

    def solve_path(
        self,
        waypoints: list[torch.Tensor],
        tgt_axis: torch.Tensor,
        q_seed: torch.Tensor,
        q_fing: float = 0.016,
        first_iters: int = 200,
        iters: int = 50,
    ) -> dict:
        """Warm-started chain of solves -- the branch-continuity fix for CEM's failure mode.

        ``waypoints`` is a list of (n,3) TCP positions. Each solve seeds from the previous
        solution *and* biases its nullspace toward it, so adjacent waypoints cannot land in
        different elbow branches unless the path genuinely requires it -- which then shows up
        as a convergence failure rather than a silent 158 mm jump between two waypoints.
        """
        qs, pes, aes, conv = [], [], [], []
        q = q_seed.clone()
        axes = tgt_axis if isinstance(tgt_axis, (list, tuple)) else [tgt_axis] * len(waypoints)
        for i, wp in enumerate(waypoints):
            out = self.solve(wp, axes[i], q, q_fing,
                             iters=first_iters if i == 0 else iters, q_bias=q)
            q = out["q"]
            qs.append(q.clone())
            pes.append(out["pos_err"])
            aes.append(out["axis_err"])
            conv.append(out["converged"])
        return {
            "q": torch.stack(qs, dim=1),                # (n, T, 6)
            "pos_err": torch.stack(pes, dim=1),         # (n, T)
            "axis_err": torch.stack(aes, dim=1),
            "converged": torch.stack(conv, dim=1).all(dim=1),
        }

    # ---------------------------------------------------------------------- actions
    def action(self, q_arm: torch.Tensor, close: bool) -> torch.Tensor:
        """(n,6) joint targets -> the env's 7-D action.

        ``arm_action`` is ``JointPositionActionCfg(scale=0.5, use_default_offset=True)``, so
        the action is ``(q_desired - q_default) / 0.5``. The gripper channel is binary:
        ``< 0`` closes, ``>= 0`` opens.
        """
        a = torch.zeros((self.n, 7), device=self.dev)
        a[:, :6] = (q_arm - self.q_default[self.arm_dof].unsqueeze(0)) / 0.5
        a[:, 6] = -1.0 if close else 1.0
        return a

    def finger_gap_mm(self) -> torch.Tensor:
        """Measured finger opening [mm] from the two prismatic joints.

        ``gap = 1.0035 * (q_left + q_right) - 1.25 mm`` (CHALLENGE_SUITE C3, residual
        0.035 mm). This is the only retention test that cannot be faked: 30 mm of block
        between the pads reads as 30 mm, air reads as about -1 mm. "Object within 80 mm of
        the TCP" once scored an untouched 107 mm block in an 89 mm gripper at 100 %.
        """
        jp = torch.as_tensor(self.robot.data.joint_pos.torch, device=self.dev)
        return (1.0035 * (jp[:, self.fing_dof[0]] + jp[:, self.fing_dof[1]]) - 0.00125) * 1000.0
