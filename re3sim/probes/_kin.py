# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""Shared kinematics core for the clutter probes and (later) the scripted expert.

Everything here follows one rule, inherited from eva_rl's `slot_insertion_probe.py:12-17`
and paid for by eva_bc's most expensive lesson ("planner-valid != executable"):

    **candidates are scored by forward kinematics read back from the sim, and the achieved
    error is always reported.**

No IK solver, no motion planner. A search that scores on achieved FK cannot silently
converge on an unexecutable pose.

Frames and conventions
----------------------
* Quaternions are **XYZW** everywhere (Isaac Lab 3.0).
* `gripper_end` local **-X** is the *approach* axis (`a_hat`) -- the direction the gripper
  travels to engulf an object. This is the axis `reachability_map.py` calls "top-down
  capable" when it points down.
* `gripper_end` local **+Y** is the *opening* axis (`o_hat`) -- the direction the two fingers
  separate along. Perpendicular to `a_hat` by construction.
* TCP = `gripper_end` origin + `quat_apply(q_end, TCP_OFFSET)` with
  `TCP_OFFSET = (-0.0419, 0, 0)`. **One constant, named, with provenance** -- eva_bc lost a
  GPU run to having three live TCP conventions in one file.
* All positions returned in the **env frame** (table top is z = 0).

Action encoding (`clutter_env_cfg.ActionsCfg`)
----------------------------------------------
    arm     : JointPositionActionCfg(scale=0.5, use_default_offset=True)
              -> q_target[i] = q_default[i] + 0.5 * a[i]      (ABSOLUTE, not incremental)
    gripper : BinaryJointPositionActionCfg
              -> a[6] >= 0 : q = 0.045/finger (89.07 mm clear gap)
                 a[6] <  0 : q = 0.000/finger
              There is NO intermediate aperture. See docs/06_EXPERT_DESIGN.md S3.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply

#: Offset from `gripper_end` to the true tool centre point, in the link frame [m].
#: Measured (eva_rl CHALLENGE_SUITE C10): with the fingers shut the two finger body origins
#: coincide, and that point is the grasp point. NEVER -0.075 (the lift task's stale value,
#: 33.1 mm too far forward) and never -0.048.
TCP_OFFSET = (-0.0419, 0.0, 0.0)

#: gripper joint values reachable through the action manager. Binary -- nothing between.
Q_OPEN = 0.045
Q_CLOSE = 0.0

#: clear gap calibration, CHALLENGE_SUITE C3: gap = 1.0035*(q_L+q_R) - 1.25 mm, resid 0.035 mm
GAP_A, GAP_B = 1.0035, 0.00125


def clear_gap(q_sum: torch.Tensor | float):
    """Clear finger opening [m] from the summed finger joint values."""
    return GAP_A * q_sum - GAP_B


def _t(x):
    """Isaac Lab 3.0 wraps some buffers in a container exposing `.torch`."""
    return x.torch if hasattr(x, "torch") else x


class ArmKin:
    """FK / CEM / waypoint execution against a live `ManagerBasedRLEnv`.

    The env's `num_envs` doubles as the CEM population size (the pattern
    `grasp_geometry.py` uses): every env holds one candidate joint vector, one batched
    `write_joint_state_to_sim` + `sim.forward()` evaluates them all.

    `fk()` never steps physics, so it cannot disturb the blocks -- but callers that run a
    CEM search and then execute must still re-write the block states, because the arm was
    teleported through them and PhysX will resolve the overlap on the first real step.
    """

    def __init__(self, env, end_link: str = "gripper_end"):
        e = env.unwrapped
        self.env, self.e = env, e
        self.robot = e.scene["robot"]
        self.dev = e.device
        self.n = e.num_envs

        r = self.robot
        self.i_end = r.body_names.index(end_link)
        self.i_left = r.body_names.index("gripper_left")
        self.i_right = r.body_names.index("gripper_right")
        self.arm_dof = [r.joint_names.index(f"joint{i}") for i in range(1, 7)]
        self.fing_dof = [r.joint_names.index(x) for x in ("joint_left", "joint_right")]

        lim = torch.as_tensor(_t(r.data.joint_pos_limits)[0], device=self.dev)
        self.lo, self.hi = lim[self.arm_dof, 0].clone(), lim[self.arm_dof, 1].clone()
        self.q_default = torch.as_tensor(_t(r.data.default_joint_pos)[0], device=self.dev).clone()
        #: the action's zero point -- `use_default_offset=True` means a = 0 holds this pose
        self.q_arm0 = self.q_default[self.arm_dof].clone()

        #: Bodies used for the table-floor check. Deliberately NOT every body: `base_link`
        #: sits at z = 0, so a whole-arm minimum is identically zero and the floor penalty
        #: silently does nothing. That defect let the CEM return grasp poses with
        #: `gripper_end` at z = 17 mm -- inside the table -- which is why P03's first control
        #: run closed on air in every condition.
        self.floor_bodies = [r.body_names.index(b) for b in
                             ("link6", "gripper_end", "gripper_left", "gripper_right")
                             if b in r.body_names]

        self._offs = torch.tensor(TCP_OFFSET, device=self.dev).repeat(self.n, 1)
        self._ax_a = torch.tensor([-1.0, 0.0, 0.0], device=self.dev).repeat(self.n, 1)
        self._ax_o = torch.tensor([0.0, 1.0, 0.0], device=self.dev).repeat(self.n, 1)

    # ------------------------------------------------------------------ forward kinematics
    def fk(self, q_arm: torch.Tensor, q_fing: float = Q_OPEN) -> dict:
        """Write `q_arm` (n, 6) to the sim and read back the achieved geometry.

        Returns a dict of (n, ...) tensors in the **env frame**:
            tcp     (n,3)  tool centre point
            a_hat   (n,3)  approach axis, unit
            o_hat   (n,3)  finger opening axis, unit (measured from the two finger bodies,
                           not assumed from the link frame -- so it stays honest if the
                           fingers are asymmetric)
            end     (n,3)  gripper_end origin
            quat    (n,4)  gripper_end orientation, XYZW
            low_z   (n,)   lowest body-origin z over `self.floor_bodies` (table proxy)
            bodies  (n,B,3) every body origin, env frame -- for row-clearance checks
        """
        q = self.q_default.unsqueeze(0).repeat(self.n, 1)
        q[:, self.arm_dof] = q_arm
        q[:, self.fing_dof] = q_fing
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.e.sim.forward()
        self.robot.update(0.0)

        bp = _t(self.robot.data.body_pos_w)
        lq = _t(self.robot.data.body_quat_w)[:, self.i_end, :]
        org = self.e.scene.env_origins
        end = bp[:, self.i_end, :] - org
        tcp = end + quat_apply(lq, self._offs)
        a_hat = quat_apply(lq, self._ax_a)
        sep = bp[:, self.i_left, :] - bp[:, self.i_right, :]
        o_hat = sep / sep.norm(dim=1, keepdim=True).clamp(min=1e-9)
        low_z = (bp[:, self.floor_bodies, 2] - org[:, 2:3]).min(dim=1).values
        return {"tcp": tcp, "a_hat": a_hat, "o_hat": o_hat, "end": end, "quat": lq,
                "low_z": low_z, "bodies": bp - org.unsqueeze(1)}

    # ------------------------------------------------------------------- row clearance
    @staticmethod
    def box_penetration(bodies: torch.Tensor, boxes: torch.Tensor,
                        margin: float = 0.0) -> torch.Tensor:
        """Deepest penetration of any body origin into any keep-out box [m], per env.

        `bodies` (n, B, 3); `boxes` (M, 6) as `[cx, cy, cz, hx, hy, hz]`. Zero when every
        body is clear. Origins are not surfaces, so this understates real penetration --
        `margin` inflates the boxes to compensate and should be set from measured body
        extents, not guessed.

        This exists because of P13's most expensive lesson: a pose can satisfy
        `pos_err = 0.55 mm` and `o_align = 1.000` and still be catastrophic, because those
        two gates pin the *tool frame* and say nothing about where the rest of the arm goes.
        The pose they accepted put the wrist stub at y = +39 mm, z = 51 mm -- inside
        `distractor_2` -- and enclosure collapsed from 100 % to 32 %.
        """
        c, h = boxes[:, :3], boxes[:, 3:] + margin
        d = (bodies.unsqueeze(2) - c).abs() - h            # (n, B, M, 3)
        pen = (-d.amax(dim=3)).clamp(min=0.0)              # (n, B, M)
        return pen.amax(dim=2).amax(dim=1)

    # -------------------------------------------------------------------------------- CEM
    def cem(
        self,
        pos: torch.Tensor,
        seed: torch.Tensor,
        o_des: torch.Tensor | None = None,
        a_des: torch.Tensor | None = None,
        w_pos: float = 200.0,
        pos_tol: float = 0.001,
        w_o: float = 0.25,
        w_a: float = 0.25,
        iters: int = 45,
        std0: float = 0.45,
        floor_z: float = 0.012,
        w_floor: float = 20.0,
        restarts: int = 1,
        avoid: torch.Tensor | None = None,
        w_avoid: float = 400.0,
        avoid_margin: float = 0.010,
    ) -> dict:
        """Cross-entropy search for arm joints putting the TCP at `pos` [3] in the env frame.

        The cost is a **constrained** form, not a weighted sum, because a weighted sum failed
        in both directions and the failures were symmetric:

        * `w_pos = 1` (grasp_geometry.py's scaling) -- `|dp|` is in metres, so 1 mm costs
          0.001 against orientation terms of order 0.25. The search walked 137-520 mm away to
          chase an axis it could not otherwise reach.
        * `w_pos = 20` -- position dominated so completely that the orientation term became
          noise, and the search returned flipped wrist poses with `gripper_end` inside the
          table.

        So position enters as a **hinge**: free inside `pos_tol`, then priced at `w_pos` per
        metre (0.2 per mm at the default). Orientation terms, bounded by `w_o + 2*w_a`, can
        therefore buy a few millimetres of slack and no more. The floor term is the same shape.

        `restarts > 1` runs independent searches from uniform draws over the joint limits and
        keeps the best. CEM is a local method: seeded at this arm's folded home pose it cannot
        even find a grasp of an isolated block, which is what P03's control run demonstrated.

        `o_des` / `a_des` are optional unit directions for the opening / approach axes.
        The opening-axis term is **sign-free** (`|o.o_des|`): a parallel jaw is symmetric,
        so demanding a sign would reject half the valid solutions. The approach term is
        signed -- direction of travel matters.

        `floor_z` penalises any arm body dipping below that height, a cheap stand-in for
        "do not drive the wrist through the table". It is a *soft* term: the returned
        `low_z` lets the caller reject hard.

        `avoid` (M, 6) keep-out boxes `[cx, cy, cz, hx, hy, hz]` price the deepest
        penetration of any **body origin** into any box. Without it the search is free to
        route the forearm or the wrist stub straight through a neighbouring block: with
        `o_hat = x_hat` the approach axis is confined to the y-z plane, so the wrist sits at
        `TCP - 0.0419 * a_hat`, and the CEM will happily put it at y = +39 mm -- inside
        `distractor_2` -- while reporting a 0.55 mm position error and `o_align = 1.000`.
        See `box_penetration`.

        Returns the best candidate plus the *achieved* geometry for it -- never a
        commanded value.
        """
        best_cost, best_q = float("inf"), seed.clone()
        n_elite = max(8, self.n // 8)

        def score(g):
            c = w_pos * ((g["tcp"] - pos).norm(dim=1) - pos_tol).clamp(min=0.0)
            c = c + w_floor * (floor_z - g["low_z"]).clamp(min=0.0)
            if o_des is not None:
                c = c + w_o * (1.0 - (g["o_hat"] @ o_des).abs())
            if a_des is not None:
                c = c + w_a * (1.0 - (g["a_hat"] @ a_des))
            if avoid is not None:
                # Priced at twice `w_pos` per metre: the search must be willing to give up
                # position accuracy to stay out of a neighbour, never the other way round.
                c = c + w_avoid * self.box_penetration(g["bodies"], avoid, avoid_margin)
            return c

        for k in range(max(1, restarts)):
            mean = seed.clone() if k == 0 else \
                self.lo + (self.hi - self.lo) * torch.rand(6, device=self.dev)
            std = torch.full((6,), std0, device=self.dev)
            for _ in range(iters):
                cand = (mean + std * torch.randn(self.n, 6, device=self.dev)) \
                    .clamp(self.lo, self.hi)
                cand[0] = mean  # carry the incumbent, so the search is monotone
                cost = score(self.fk(cand))
                elite = cand[cost.topk(n_elite, largest=False).indices]
                mean, std = elite.mean(0), elite.std(0).clamp(min=0.01)
                j = int(cost.argmin())
                if float(cost[j]) < best_cost:
                    best_cost, best_q = float(cost[j]), cand[j].clone()

        g = self.fk(best_q.unsqueeze(0).repeat(self.n, 1))
        out = {
            "q": best_q,
            "cost": best_cost,
            "pos_err": float((g["tcp"][0] - pos).norm()),
            "tcp": g["tcp"][0].clone(),
            "a_hat": g["a_hat"][0].clone(),
            "o_hat": g["o_hat"][0].clone(),
            "low_z": float(g["low_z"][0]),
            "pen": float(self.box_penetration(g["bodies"][:1], avoid, avoid_margin)[0])
            if avoid is not None else 0.0,
        }
        out["o_align"] = float((out["o_hat"] @ o_des).abs()) if o_des is not None else float("nan")
        out["a_align"] = float(out["a_hat"] @ a_des) if a_des is not None else float("nan")
        return out

    def refine(self, q0: torch.Tensor, pos: torch.Tensor, iters: int = 3,
               eps: float = 2e-3, damp: float = 1e-4,
               o_des: torch.Tensor | None = None, k_rot: float = 0.06) -> torch.Tensor:
        """Per-env millimetre correction of `q0` (n,6) toward per-env TCP targets `pos` (n,3).

        A damped-least-squares step around a CEM-verified pose. This is **not** the DLS
        differential IK both repos warn about: that warning is about *servoing* a whole
        trajectory near the table, where the solver walks into singular configurations. Here
        the Jacobian is finite-differenced once at a pose the CEM already proved executable,
        the correction is millimetre-scale, and the achieved TCP is re-read from the sim
        afterwards so the caller can reject rather than trust.

        `o_des` (n,3) additionally steers the **opening axis** per env, which is what the
        target's spawn yaw requires: the reset draws `yaw ~ U(-0.20, +0.20)` rad, and an
        orthogonal jaw closing on a block yawed by 11.5 deg does not squeeze it, it *rotates*
        it. A 36 x 30 mm block turned back to the jaw sweeps its corners up to 8.4 mm in y,
        into free gaps whose measured median is 8.3 mm. P15's forensics put 52 % of all
        topple onsets inside the `close` phase with `distractor_1` as the victim in 66 of
        128 envs, which is that mechanism.

        The residual is stacked as `[tcp ; k_rot * o_hat]` against `[pos ; k_rot * o_des]`,
        sign-corrected per env because a parallel jaw is symmetric. `k_rot` sets the exchange
        rate between the two: at 0.06 a full 90 deg misalignment weighs the same as 60 mm of
        position error, so orientation can never win a real position argument. Aligning one
        axis leaves roll about it free, so the 6x6 system is rank-5 and the damping carries
        the null space -- which is correct, since roll about the opening axis is exactly the
        freedom the CEM already spent on clearance.
        """
        q = q0.clone()

        def feat(qq):
            g = self.fk(qq)
            if o_des is None:
                return g["tcp"]
            return torch.cat([g["tcp"], k_rot * g["o_hat"]], dim=1)

        if o_des is None:
            des = pos
        else:
            s = torch.sign((self.fk(q)["o_hat"] * o_des).sum(dim=1, keepdim=True))
            des = torch.cat([pos, k_rot * o_des * torch.where(s == 0, 1.0, s)], dim=1)

        m = des.shape[1]
        # numerical Jacobian at the nominal pose, 6 batched FK calls
        base = feat(q)
        J = torch.zeros(self.n, m, 6, device=self.dev)
        for k in range(6):
            dq = q.clone()
            dq[:, k] += eps
            J[:, :, k] = (feat(dq) - base) / eps
        JT = J.transpose(1, 2)
        A = JT @ J + damp * torch.eye(6, device=self.dev).unsqueeze(0)
        for _ in range(iters):
            err = (des - feat(q)).unsqueeze(-1)
            q = (q + (torch.linalg.solve(A, JT @ err)).squeeze(-1)).clamp(self.lo, self.hi)
        return q

    # -------------------------------------------------------------------------- execution
    def act(self, q_arm: torch.Tensor, close: bool) -> torch.Tensor:
        """Encode arm joints + gripper state as the env's 7-D action."""
        a = torch.zeros((self.n, 7), device=self.dev)
        a[:, :6] = (q_arm - self.q_arm0.unsqueeze(0)) / 0.5
        a[:, 6] = -1.0 if close else 1.0
        return a

    def hold(self, q_arm: torch.Tensor, steps: int, close: bool):
        for _ in range(steps):
            self.env.step(self.act(q_arm, close))

    def run(self, q_from: torch.Tensor, q_to: torch.Tensor, steps: int, close: bool,
            ramp: float = 0.8):
        """Interpolate the *commanded* joint target from `q_from` to `q_to` over `steps`.

        `ramp < 1` finishes the interpolation early so the last steps are a settle at the
        final command -- the position drive needs time to converge, and arriving at the
        waypoint exactly on the final step means arriving with tracking error.
        """
        for s in range(steps):
            f = min(1.0, (s + 1) / max(1.0, steps * ramp))
            self.env.step(self.act((1 - f) * q_from + f * q_to, close))

    # --------------------------------------------------------- physics-only execution
    # `env.step` runs the whole MDP, including the `distractor_toppled` termination and the
    # reset events behind it. In a contact probe that is a trap: a block that tips past 41.4
    # deg resets the scene, the block re-spawns UPRIGHT with fresh spawn jitter, and the probe
    # then reads `up_z = 1.0` and a random +/-10 mm displacement. The topple it was built to
    # detect is erased, and the jitter looks like measurement noise. P09's first run showed
    # exactly that -- distractors "moving" toward the robot under a push directed away from it.
    #
    # These bypass the MDP: set the drive target, step the simulator, refresh the buffers.
    # Use them for anything measuring contact; use `hold`/`run` when the MDP itself is the
    # object of study.
    def _drive(self, q_arm: torch.Tensor, q_fing: float):
        q = self.q_default.unsqueeze(0).repeat(self.n, 1)
        q[:, self.arm_dof] = q_arm
        q[:, self.fing_dof] = q_fing
        return q

    def hold_phys(self, q_arm: torch.Tensor, steps: int, q_fing: float = Q_OPEN):
        q = self._drive(q_arm, q_fing)
        for _ in range(steps):
            self.robot.set_joint_position_target(q)
            self.robot.write_data_to_sim()
            self.e.sim.step()
            self.e.scene.update(self.e.physics_dt)

    def run_phys(self, q_from: torch.Tensor, q_to: torch.Tensor, steps: int,
                 q_fing: float = Q_OPEN, substeps: int = 8):
        """Interpolate the drive target, `substeps` physics steps per control step.

        `substeps` mirrors the env's `decimation = 8` so speeds stay comparable to what a
        policy could command.
        """
        for s in range(steps):
            f = (s + 1) / steps
            q = self._drive((1 - f) * q_from + f * q_to, q_fing)
            for _ in range(substeps):
                self.robot.set_joint_position_target(q)
                self.robot.write_data_to_sim()
                self.e.sim.step()
                self.e.scene.update(self.e.physics_dt)

    def teleport_arm(self, q_arm: torch.Tensor, q_fing: float = Q_OPEN):
        """Put the arm at `q_arm` with zero velocity, for starting a trial cleanly."""
        q = self.q_default.unsqueeze(0).repeat(self.n, 1)
        q[:, self.arm_dof] = q_arm
        q[:, self.fing_dof] = q_fing
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.e.sim.forward()
        self.e.scene.update(self.e.physics_dt)

    # ------------------------------------------------------------------------- read-backs
    def gap(self) -> torch.Tensor:
        """Clear finger opening [m], per env. The ground truth for 'did it close on it'."""
        return clear_gap(_t(self.robot.data.joint_pos)[:, self.fing_dof].sum(dim=1))

    def tcp_now(self) -> torch.Tensor:
        bp = _t(self.robot.data.body_pos_w)
        lq = _t(self.robot.data.body_quat_w)[:, self.i_end, :]
        return bp[:, self.i_end, :] + quat_apply(lq, self._offs) - self.e.scene.env_origins

    def finger_pos(self) -> tuple[torch.Tensor, torch.Tensor]:
        bp = _t(self.robot.data.body_pos_w) - self.e.scene.env_origins.unsqueeze(1)
        return bp[:, self.i_left, :], bp[:, self.i_right, :]


def lerp_pts(a: torch.Tensor, b: torch.Tensor, k: int) -> list[torch.Tensor]:
    """k intermediate Cartesian waypoints from `a` (exclusive) to `b` (inclusive)."""
    return [a + (b - a) * (i + 1) / k for i in range(k)]
