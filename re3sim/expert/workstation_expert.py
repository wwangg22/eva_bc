# Copyright (c) 2026. Re3Sim workstation effort, eva_bc/re3sim.
"""Scripted expert for ``Rebot-Workstation-PickPlace1-v0``.

Grasp the rubix cube off the reconstructed desk and drop it into the open box, working around
two clutter bodies (a roll of tape and a tape measure) that are never manipulated.

Every choice below is either measured on this arm or inherited from a measurement that cost
someone a GPU run. Named so a later reader can check rather than trust.

The recipe
----------
==============  ==========================  ===============================================
grip height     **z = 0.025 m**             MEASURED, `probes/p01_grasp_feasibility.py`.
                                            The cube holds 99 % here. The neighbouring cells
                                            are 5 % at 20 mm and 77 % at 30 mm, so this is a
                                            genuine optimum and not a plateau. Note it is
                                            *below* the 44 mm "TCP floor" of CHALLENGE_SUITE
                                            C9 -- that number was measured with the fingers
                                            SHUT and never constrained grasping.
opening axis    NEAREST CUBE FACE NORMAL    A parallel jaw closing on a cube corner rotates
                                            it instead of squeezing it. The cube's yaw is
                                            drawn U[0, 2pi), so the jaw is snapped to
                                            whichever of the four face normals lies closest
                                            to the perpendicular of the approach. Sign-free
                                            (`|o.o_des|`), because a parallel jaw is
                                            symmetric.                    (LESSONS A5)
approach axis   UNCONSTRAINED               Pinning it to horizontal +x collapsed `o_align`
                                            to 0.40-0.69 and failed the p01 positive
                                            control. C1 says this arm's attainable approach
                                            at table height is tilted, not horizontal. The
                                            standoff direction is taken from the ACHIEVED
                                            `a_hat`, projected horizontal.
floor penalty   **`floor_z = 0.002`**       NOT the clutter expert's 0.012. Its blocks are
                                            70 mm tall; a 56 mm cube gripped at 25 mm needs
                                            the finger bodies far lower, and a 12 mm floor
                                            penalises the exact pose the grasp requires.
                                            This value only keeps the arm out of the table.
pose gate       pos <= 1.5 mm,              `o_align >= 0.99` (the clutter expert's gate) is
                o_align >= 0.90,            NOT attainable at this grip height -- p01 never
                low_z >= 2 mm               saw it. 0.90 is where held-rate goes to 99 %.
release         **z >= 0.150 m**            The rim is at 93 mm and the cube hangs ~28 mm
                over the box centre          below the TCP, so anything lower drags the cube
                                            through the near wall on the way in.
ordering        SOLVE EVERYTHING FIRST      The whole trajectory -- approach, grasp, lift,
                                            carry, release -- is solved BEFORE the fingers
                                            close. `cem`/`refine` call
                                            `write_joint_state_to_sim`, which teleports the
                                            arm and re-opens the fingers hundreds of times;
                                            searching after closing silently drops the cube
                                            and reads as a slip. Fixing this ordering alone
                                            took one control trial 0 % -> 100 %. (LESSONS A6)
clutter         KEEP-OUT BOXES              Passed to `cem(avoid=...)`. Body ORIGINS are
                                            tested, and origins are not surfaces, so the
                                            boxes are inflated. Without this the search
                                            routes the forearm straight through the tape
                                            roll while reporting a sub-mm TCP error.
==============  ==========================  ===============================================

What this expert deliberately does NOT do
-----------------------------------------
* **It does not ramp the gripper.** ``ActionsCfg.gripper_action`` is a
  ``BinaryJointPositionActionCfg`` -- there is no intermediate aperture. A demonstration
  containing an action the policy cannot emit is worse than useless, and the legal
  duty-cycled approximations measured *worse* than a plain binary close. (clutter P19/P20)
* **It does not touch the clutter.** They are obstacles to route around, not objects.
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "probes"))
from _kin import ArmKin, Q_OPEN, lerp_pts  # noqa: E402

# ---------------------------------------------------------------------------------- recipe
#: Measured optimum. 0.025 was measured by `p01_grasp_feasibility.py` against an ANALYTIC
#: 56 mm cube; the reconstructed mesh is 53.3 x 58.0 x 54.5 mm with rounded corners and a
#: convex-hull collider, which is a different grasp -- MEASURED, the expert fell from ~31 % to
#: 19.5 % on the swap. Overridable so the height can be re-swept per collider without editing
#: code; the sweep lives in `probes/sweep_grip_z.sh`.
#:
#: RE-MEASURED 2026-08-06 against the reconstructed mesh (`probes/sweep_grip_z.sh`,
#: 64 envs, teleport protocol, seed 11):
#:
#:     grip z [mm]   18     22     25     28     32     36
#:     success      17.2   10.9   21.9   25.0   29.7   26.6  %
#:     planned      37     28     34     40     42     38   /64
#:
#: 32 mm, not the 25 mm the analytic cube wanted -- and that recovers essentially all of the
#: 31 % -> 19.5 % drop the collider swap cost. The real cube is 54.5 mm tall with rounded
#: corners, so gripping above mid-height puts the fingers on more flat face and less bevel.
#:
#: RE-MEASURED AGAIN 2026-08-06, END TO END (no teleport shortcut, full transit from the env's
#: own reset pose) once the transit ordering bug was fixed:
#:
#:     grip z [mm]   32     40     48     56     64
#:     success        0.0    7.8   25.0   40.6   40.6  %
#:     planned       45     55     60     59     58   /64
#:
#: 56 mm -- ABOVE the cube's 54.5 mm top, and that is the point. The arm sags ~35 mm on the
#: 140 mm descent onto the object, so the commanded height has to sit high enough that the
#: SETTLED tool centre lands on mid-cube. Commanding the height you want is commanding the
#: wrong thing on a position-driven arm.
#:
#: Both earlier values were measured under protocols that hid this. 25 mm came from p01 on an
#: ANALYTIC cube; 32 mm came from a sweep that used `--teleport-pregrasp`, and under the
#: pre-fix waypoint ordering that put the arm effectively AT the cube -- so it measured a
#: grasp with no descent and therefore no sag. Both were honest measurements of the wrong
#: thing.
GRIP_Z = float(os.environ.get("GRIP_Z", "0.056"))
#: how far back along the (horizontal) approach axis the pre-grasp sits [m]. Env-overridable
#: for the same reason `GRIP_Z` is: it is swept end-to-end rather than reasoned about.
STANDOFF = float(os.environ.get("STANDOFF", "0.075"))
#: straight-up lift after the grasp, before any lateral motion [m]
LIFT_Z = 0.150
#: TCP height while carrying and at release. Rim 93 mm + cube half 28 mm + margin.
CARRY_Z = float(os.environ.get("CARRY_Z", "0.175"))
#: gates -- see the module docstring for why these differ from the clutter expert's
POS_TOL = 0.0015
O_ALIGN_MIN = float(os.environ.get("O_ALIGN_MIN", "0.90"))
LOW_Z_MIN = 0.002
#: Cartesian waypoint spacing [m]. clutter P17: independently solved waypoints can land in
#: different IK branches, and the joint-space line between them swept 108 mm off its
#: Cartesian path. Dense local solves cut that to 1.1 mm and took success 18 % -> 62.5 %.
MAX_STEP = 0.030
#: keep-out inflation for clutter bodies [m] -- body origins are not surfaces
AVOID_MARGIN = 0.020
#: MEASURED 2026-08-06. `_kin.cem`'s default `w_o = 0.25` prices a *total* orientation
#: collapse at 0.25, which the hinged position term matches after 1.25 mm -- so the search
#: buys a fifth of a millimetre with the entire opening axis. The first run of this expert
#: failed its gate on 15/32 envs with `pos_err` of 0.3-1.1 mm (well inside tolerance) and
#: `o_align` of 0.60-0.90, which is exactly that trade being taken. At 4.0 a drop to
#: `o_align = 0.90` costs 0.40, worth 2 mm -- competitive with position without dominating it.
W_O_GRASP = 4.0
#: raised with `W_O_GRASP`, and for the same reason in reverse. The two weights are only
#: meaningful against each other: at the stock `w_pos = 200` a millimetre past `pos_tol` costs
#: 0.2, so `w_o = 4.0` would let the search spend the ENTIRE position budget
#: (`POS_TOL - pos_tol` = 0.5 mm) to buy 0.025 of `o_align` -- trading a gate failure for a
#: different gate failure. At 600 the same millimetre costs 0.6, so alignment can buy about a
#: millimetre of slack and no more, which is the balance the hinge was designed for.
W_POS_GRASP = 600.0
#: The cube is symmetric under 90 deg, so BOTH face-normal pairs are valid opening axes and
#: they are not equally reachable: whichever one the radial heuristic happens to pick may put
#: the wrist somewhere this arm cannot align to. Trying both is the cheapest real widening of
#: the search available. z = 0.030 is the p01 runner-up cell (77 % vs 99 %) and is tried only
#: after both axes have failed at the measured optimum.
GRIP_Z_FALLBACKS = (GRIP_Z, 0.064)
#: Largest joint move between consecutive waypoints of the free-space home transit [rad].
#: Not a precision figure -- it only has to be small enough that the audit samples the path
#: densely enough to catch a body clipping the table, and that the executor's per-waypoint
#: ramp is not asked for more motion than the drive can deliver in its allotted steps.
MAX_JOINT_STEP = 0.08
#: Height of the up-and-over transit waypoint [m]. Above the box rim (93 mm) and above every
#: object, so the descent to the pre-grasp is the only part of the transit that is ever low.
TRANSIT_Z = 0.170
#: Largest per-joint difference allowed between the transit waypoint and the pre-grasp [rad].
#: Beyond this they are in different IK branches and the line between them is not followable.
MAX_TRANSIT_JOINT_JUMP = 1.2
#: Largest per-joint gap allowed between the env's reset pose and the first solved waypoint of
#: the transit [rad]. Anything larger is a different IK branch, not a move.
MAX_START_HOP = 0.45
#: How far a waypoint on a *chain* may miss its Cartesian target before the plan is thrown
#: away [m]. Far looser than the grasp gate's 1.5 mm -- these are transit waypoints and only
#: their endpoint has to be accurate -- but a chain that misses by tens of millimetres has not
#: solved the path it claims to have solved.
CHAIN_TOL = 0.010


def _chain(kin: ArmKin, start_q, pts, o_des, restarts=1, std0=0.10, avoid=None, iters=35):
    """Solve a dense Cartesian waypoint chain, seeding each solve from the previous one.

    Returns ``(qs, worst_pos_err)``. The error is not decoration: this function used to
    discard it, and a chain that quietly failed to reach its waypoints was indistinguishable
    from one that succeeded. MEASURED consequence -- the transit's outer waypoint came back
    **98 mm below** where it had been asked for, the arm tracked that pose perfectly, and the
    whole thing read as "the arm cannot follow the transit" when the truth was "the planner
    never solved it". The caller must gate on this.
    """
    qs, q, worst = [], start_q, 0.0
    for p in pts:
        s = kin.cem(p, q, o_des=o_des, iters=iters, restarts=restarts, std0=std0,
                    floor_z=LOW_Z_MIN, avoid=avoid)
        q = s["q"]
        worst = max(worst, s["pos_err"])
        qs.append(q)
    return qs, worst


def _joint_line(a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor]:
    """Joint-space waypoints from a (exclusive) to b (inclusive), at most MAX_JOINT_STEP apart.

    Smooth and single-branch by construction, which is the whole point -- see the transit
    comment in `plan_episode`. Clearance is NOT guaranteed and is audited separately.
    """
    k = max(1, int(math.ceil(float((b - a).abs().max()) / MAX_JOINT_STEP)))
    return [a + (b - a) * (i / k) for i in range(1, k + 1)]


def _densify(a: torch.Tensor, b: torch.Tensor):
    """Waypoints from a (exclusive) to b (inclusive), at most MAX_STEP apart."""
    k = max(1, int(math.ceil(float((b - a).norm()) / MAX_STEP)))
    return lerp_pts(a, b, k)


def face_normal_o_des(cube_xy: torch.Tensor, cube_yaw: float, dev) -> torch.Tensor:
    """Opening axis: the cube face normal closest to the perpendicular of the radial approach.

    A parallel jaw that closes on a cube *corner* rotates the cube rather than squeezing it,
    and the spawn draws yaw uniformly on [0, 2pi), so this has to be computed per episode
    rather than assumed. The four face normals are yaw + k*pi/2; the perpendicular of the
    approach is the direction the fingers should separate along.
    """
    return face_normal_candidates(cube_xy, cube_yaw, dev)[0]


def face_normal_candidates(cube_xy: torch.Tensor, cube_yaw: float, dev) -> list[torch.Tensor]:
    """BOTH valid opening axes for a cube, best-first.

    ``+n`` and ``-n`` are the same axis to a symmetric jaw, so the four face normals collapse
    to **two** distinct choices, ``yaw`` and ``yaw + pi/2``. Both squeeze face-to-face and are
    equally correct as *grasps*; they are not equally **reachable**, because they demand
    different wrist rolls at a fixed TCP.

    Ordered by the radial heuristic -- the axis closest to the perpendicular of the approach
    is tried first, since that is the one a straight-in approach naturally produces -- but the
    caller is expected to fall through to the second. MEASURED: doing so is what turned the
    planner's 53 % solve rate into something usable; a single axis leaves the arm no way out
    when the yaw it was handed happens to fight the wrist limit.
    """
    u = cube_xy / cube_xy.norm().clamp(min=1e-9)
    perp = torch.tensor([-u[1], u[0], 0.0], device=dev)
    axes = []
    for k in range(2):
        a = cube_yaw + k * math.pi / 2
        n = torch.tensor([math.cos(a), math.sin(a), 0.0], device=dev)
        axes.append((abs(float(n @ perp)), n))
    axes.sort(key=lambda t: -t[0])
    return [n for _, n in axes]


def clutter_avoid_boxes(clutter: list[tuple[torch.Tensor, float, float]], dev) -> torch.Tensor:
    """(M, 6) keep-out boxes ``[cx, cy, cz, hx, hy, hz]`` for the clutter bodies."""
    rows = []
    for pos, radius, height in clutter:
        rows.append([float(pos[0]), float(pos[1]), height / 2,
                     radius, radius, height / 2])
    return torch.tensor(rows, device=dev) if rows else None


def _passes(s: dict) -> bool:
    return (s["pos_err"] <= POS_TOL and s["o_align"] >= O_ALIGN_MIN
            and s["low_z"] >= LOW_Z_MIN and s.get("pen", 0.0) <= 1e-6)


def _gate_margin(s: dict) -> float:
    """How close a candidate is to passing, as the *worst* of the normalised margins.

    Ranking by `cost` was wrong: `cost` mixes the avoid and floor penalties in, so a pose that
    is beautifully aligned but 4 mm out can outrank one that misses the gate by 0.01 of
    `o_align`. What the caller wants is the candidate it would have to improve least.

    Keep-out penetration is subtracted rather than ranked, because it is not a margin to trade
    against: an arm inside the tape roll is not a nearly-good plan.
    """
    return min((POS_TOL - s["pos_err"]) / POS_TOL,
               (s["o_align"] - O_ALIGN_MIN) / max(1e-6, 1.0 - O_ALIGN_MIN),
               (s["low_z"] - LOW_Z_MIN) / max(1e-6, LOW_Z_MIN)) - 100.0 * s.get("pen", 0.0)


def _measure(kin: ArmKin, q: torch.Tensor, pos, o_des, avoid) -> dict:
    """Achieved geometry of a single arm pose, read back through the sim's own FK."""
    g = kin.fk(q.unsqueeze(0).repeat(kin.n, 1))
    return {
        "q": q, "tcp": g["tcp"][0].clone(),
        "a_hat": g["a_hat"][0].clone(), "o_hat": g["o_hat"][0].clone(),
        "pos_err": float((g["tcp"][0] - pos).norm()),
        "o_align": float((g["o_hat"][0] @ o_des).abs()),
        "low_z": float(g["low_z"][0]),
        "pen": float(kin.box_penetration(g["bodies"][:1], avoid, AVOID_MARGIN)[0])
        if avoid is not None else 0.0,
    }


def settle_bias(kin: ArmKin, q: torch.Tensor, rounds: int = 3, steps: int = 30) -> torch.Tensor:
    """The command offset that makes the arm SETTLE at `q` instead of near it.

    A position drive produces torque proportional to its own error, so any pose that requires
    torque is held with an error -- and `cem` cannot see this, because it evaluates candidates
    with `write_joint_state_to_sim`, which places the arm kinematically with no gravity and no
    contact. MEASURED (03_EXPERT_AND_BC.md 3a/3b): teleported exactly onto a solved pre-grasp
    the arm settles 19.8 mm away from it, and driven to the transit pose it parks ~100 mm
    below. The manoeuvre after the grasp is unaffected because those poses are high and
    retracted; the grasp itself is the most torque-demanding pose in the trajectory.

    A runtime integrator was tried and is the wrong instrument: tuned to converge inside a
    30-step descent it necessarily overshoots once it arrives, and at a +-0.80 rad clamp it
    drove the cube from z = 28 mm to z = 11.5 mm -- into the desk.

    So measure it instead, once, at plan time, where teleporting is already free. Drive the
    arm at `q + bias`, read where it actually settles, and add the shortfall. Converges in two
    or three rounds because the plant is a stiff proportional drive.

    `hold_phys` is used rather than `env.step` deliberately: it bypasses the MDP, so no
    termination fires and no reset erases the measurement (`REFERENCE.md` 5).
    """
    qb = q.unsqueeze(0).repeat(kin.n, 1)
    bias = torch.zeros_like(qb)
    for _ in range(rounds):
        # Teleport to the DESIRED pose and drive at the BIASED one. Teleporting to `qb + bias`
        # -- which the first version did -- starts the arm already at its own command, so it
        # barely moves, `q_now` comes back as `qb + bias`, and the update becomes
        # `bias += (qb - (qb + bias)) = -bias`: the correction cancels itself every round and
        # converges to zero. That is why enabling this made the grasp WORSE (52 -> 66 mm).
        kin.teleport_arm(qb, q_fing=Q_OPEN)
        kin.hold_phys(qb + bias, steps, q_fing=Q_OPEN)
        q_now = kin.robot.data.joint_pos[:, kin.arm_dof]
        bias = bias + (qb - q_now)
    return bias[0].clone()


def _solve_grasp(kin: ArmKin, grip, o_des, avoid, tries: int = 1) -> dict:
    """CEM at the re-priced orientation weight, then a DLS polish, keeping whichever wins.

    The first version of this expert stopped at the CEM. That leaves millimetre-and-degree
    residuals on the table for free: `refine` finite-differences one Jacobian at a pose the
    CEM has already proved executable and drives `[tcp ; k_rot * o_hat]` at the target, which
    is precisely the pair the gate tests. It is kept only when it actually improves the gate
    margin -- DLS near a rank-deficient configuration can walk *away*, and the achieved pose
    is re-read from the sim rather than assumed.
    """
    best = None
    for _ in range(tries):
        # `tries` defaults to 1 because `restarts` already supplies the diversity a repeat
        # would: each restart seeds from a fresh uniform draw over the joint limits, so a
        # second call is 8 more of the same draws, not a different kind of search. Diversity
        # that actually changes the problem comes from the caller's candidate loop -- a
        # different opening axis or grip height. Planning cost is the binding constraint on
        # how many demonstrations a night produces, and this halves the worst case.
        s = kin.cem(grip, kin.q_arm0, o_des=o_des, w_o=W_O_GRASP, w_pos=W_POS_GRASP,
                    iters=45, restarts=10, std0=0.45,
                    floor_z=LOW_Z_MIN, avoid=avoid, avoid_margin=AVOID_MARGIN)
        s = {k: s[k] for k in ("q", "tcp", "a_hat", "o_hat", "pos_err", "o_align",
                               "low_z", "pen")}
        q_ref = kin.refine(s["q"].unsqueeze(0).repeat(kin.n, 1),
                           grip.unsqueeze(0).repeat(kin.n, 1),
                           o_des=o_des.unsqueeze(0).repeat(kin.n, 1))[0]
        r = _measure(kin, q_ref, grip, o_des, avoid)
        cand = r if _gate_margin(r) > _gate_margin(s) else s
        if best is None or _gate_margin(cand) > _gate_margin(best):
            best = cand
        if _passes(best):
            break
    return best


def grasp_choices() -> list[tuple[float, int]]:
    """The goalset, as ``(grip height, opening-axis index)`` pairs, best-first.

    ``run_expert_v1`` hands its planner a **goalset** -- many candidate grasps at once -- and
    lets the planner pick a reachable one, instead of committing to a single solution and
    hoping. This expert cannot do that literally (its CEM solves one pose per call), so the
    goalset is enumerated instead: two opening axes at the measured grip height, then the same
    two at the runner-up height. Ordered so the 99 %-held cell is exhausted on both axes
    before the 77 % cell is touched.

    The point of naming them is that the *executor* can now retry with a DIFFERENT member
    after a grasp fails, which is the recovery the current expert has none of.
    """
    return [(gz, k) for gz in GRIP_Z_FALLBACKS for k in (0, 1)]


def plan_episode(kin: ArmKin, cube_xy, cube_yaw, box_xy, clutter, verbose=False,
                 choices=None) -> dict | None:
    """Solve the WHOLE trajectory before anything closes. Returns None if a gate fails.

    Returns a dict of joint-space segments:
        home    -- reset pose -> pre-grasp standoff, fingers OPEN. Action-driven, because
                   `env.reset()` leaves the arm here and a demonstration that starts anywhere
                   else begins in a state the env never produces
        pre     -- the pre-grasp standoff itself (== home[-1])
        grasp   -- standoff -> grasp pose, fingers OPEN
        lift    -- straight up, fingers CLOSED
        carry   -- lift top -> above the box centre, fingers CLOSED
        (release happens at carry[-1] by opening the fingers)
        retreat -- back toward the start pose, fingers OPEN
    """
    dev = kin.dev
    avoid = clutter_avoid_boxes(clutter, dev)

    # --- the grasp pose: search over BOTH opening axes, then both grip heights -------------
    # Order matters. The measured optimum (z = 25 mm) is exhausted on both axes before the
    # runner-up height is touched, so a plan only pays the 77 %-cell penalty when the 99 %
    # cell genuinely has no reachable pose for this layout.
    #
    # ``choices`` restricts the search to particular members of the goalset, which is how the
    # executor asks for a grasp *different from the one that just failed*. None means "the
    # whole goalset, best-first", the original behaviour.
    axes = face_normal_candidates(cube_xy, cube_yaw, dev)
    best, best_o_des, best_grip = None, None, None
    for gz, k in (choices if choices is not None else grasp_choices()):
        grip = torch.tensor([float(cube_xy[0]), float(cube_xy[1]), gz], device=dev)
        o_des = axes[k]
        s = _solve_grasp(kin, grip, o_des, avoid)
        if best is None or _gate_margin(s) > _gate_margin(best):
            best, best_o_des, best_grip = s, o_des, grip
        if _passes(s):
            break

    if not _passes(best):
        if verbose:
            print(f"    [plan] grasp gate FAILED: pos {best['pos_err'] * 1000:.2f} mm, "
                  f"o_align {best['o_align']:.3f}, low_z {best['low_z'] * 1000:.1f} mm")
        return None
    o_des, grip, q_grip = best_o_des, best_grip, best["q"]

    # --- descent: high transit pose -> standoff -> grasp, ALL solved backwards -----------
    # One Cartesian chain, solved from the grasp pose outwards, so every waypoint on the whole
    # descent inherits `q_grip`'s IK branch by construction. That is the P17 rule and it is the
    # right rule *here*, where the arm has to follow a particular path through space next to an
    # object it must not touch.
    #
    # What P17 does NOT cover is free-space transit, and conflating the two cost most of this
    # session. MEASURED, at 32 envs, primitive colliders, same seed:
    #
    #   Cartesian CEM chain home -> pre-grasp   arm plateaus 60.8 mm short   0/32 success
    #   joint line home -> pre-grasp            tracks to 4.5 mm             but clips the
    #                                                                        table: 30/32
    #                                                                        plans rejected
    #   joint line home -> q_high (separate IK) arm parks 53 mm ABOVE        0/32 success
    #                                           the pre-grasp
    #
    # The third failed because `q_high` was solved independently and landed in a different
    # branch -- so the fix is not to solve it at all. Extending the backwards chain up to the
    # transit height *produces* `q_high` in the correct branch as a by-product, and the only
    # joint-space leg left is home -> q_high, which is high, short and away from everything.
    f = best["a_hat"].clone()
    f[2] = 0.0
    f = f / f.norm().clamp(min=1e-9)
    pre = grip - STANDOFF * f
    high = pre.clone()
    high[2] = max(float(pre[2]), TRANSIT_Z)

    # NO reversal. `_densify(a, b)` already runs a -> b, so marching outward from `q_grip`
    # the order is simply grip -> pre -> high. Reversing each half (copied from an earlier
    # version that solved a single leg backwards) puts `high` in the MIDDLE of the chain and
    # leaves `q_up[-1]` sitting near the pre-grasp instead of at the transit height.
    #
    # MEASURED, and this is what made the transit look unfixable for a day: at the end of the
    # transit the arm was 101.6 mm from the Cartesian target `high` but only **7.5 mm** from
    # the FK of the pose it had actually been commanded to. The arm was going exactly where it
    # was sent; the plan was sending it 107 mm too low. Every "the drive cannot hold this pose"
    # hypothesis was chasing a planner bug.
    #
    # The `CHAIN_TOL` gate did not catch it because the chain DOES reach every waypoint it was
    # given -- it was given them in the wrong order.
    up_pts = _densify(grip, pre) + _densify(pre, high)
    q_up, e_up = _chain(kin, q_grip, up_pts, o_des, avoid=avoid)
    q_high = q_up[-1]
    seg_pre = q_up[len(_densify(grip, pre)) - 1]
    seg_grasp = list(reversed(q_up[:-1])) + [q_grip]

    # --- home -> the transit pose, ACTION-DRIVEN, in joint space --------------------------
    # `env.reset()` leaves the arm at its default pose. Teleporting to the pre-grasp and
    # recording from there produces demonstrations whose first observation the env never
    # generates, so the policy meets an unseen state at step 0 of every evaluation. Upstream
    # measured the fix and its price: an action-driven approach scored 73.0 % against a 74.2 %
    # teleport baseline (clutter P29) -- 1.2 points, and it is what ships, because the teleport
    # number is not a number about a deployable policy.
    #
    # A joint line is smooth, single-branch and inside the limits by construction, which is
    # exactly what a followable free-space move needs. It gives no clearance guarantee, so
    # that is audited rather than assumed.
    tcp_pre = kin.fk(seg_pre.unsqueeze(0).repeat(kin.n, 1))["tcp"][0].clone()
    tcp_home = kin.fk(kin.q_arm0.unsqueeze(0).repeat(kin.n, 1))["tcp"][0].clone()

    # TRANSIT: one Cartesian chain, still solved backwards from the grasp.
    #
    # Three joint-space transits were tried and all three failed the same way -- a linear
    # interpolation between two high configurations of a redundant arm is not itself high.
    # MEASURED dips, for a path whose endpoints sit at z = 164 mm and z = 170 mm:
    #
    #   home -> pre-grasp, one leg          -51 .. -86 mm   30/32 layouts rejected
    #   home -> q_high -> pre-grasp         same            18/32 rejected
    #   home -> turn base -> q_high         same            18/32 rejected
    #
    # Interpolating joints moves through configurations, not through positions, and this arm
    # folds under the table on the way. So the transit is Cartesian after all -- the *reason*
    # the first Cartesian attempt failed was never the geometry, it was that each waypoint was
    # solved independently and hopped IK branches. Extending the SAME backwards chain from the
    # grasp all the way out to the home TCP fixes that at the root: every waypoint on the whole
    # trajectory, transit included, is seeded from its neighbour and inherits `q_grip`'s branch.
    #
    # What is left over is the short joint hop from the arm's actual reset pose to the first
    # solved waypoint, which targets the home TCP itself and is therefore small.
    # Solved FORWARD from the reset pose, and with the opening axis LEFT FREE.
    #
    # The previous version ran the chain backwards from the grasp all the way out to the home
    # TCP, carrying `o_des` with it. That is what made every plan unusable: at the home pose
    # the gripper's opening axis is whatever the reset happens to give, so demanding the
    # GRASP's axis out there forces the CEM to twist the wrist into a different branch, and
    # the executor's first command then asks the arm to jump 2.15-2.52 rad. All 23 surviving
    # plans were rejected on exactly that.
    #
    # The transit does not need the grasp's orientation -- it only has to arrive. So it is
    # solved forward from `q_arm0` with `o_des=None`, which keeps it in the reset pose's own
    # branch, and the wrist is re-oriented afterwards at TRANSIT_Z, in free space 170 mm above
    # the desk where a re-orientation is both safe and followable. Doing that twist low down,
    # next to the cube, is what every earlier attempt was implicitly asking for.
    in_pts = _densify(tcp_home, high)
    q_in, e_in = _chain(kin, kin.q_arm0, in_pts, None, avoid=avoid, iters=25)
    seg_home = q_in + _joint_line(q_in[-1], q_high)

    # `fk` takes exactly `kin.n` rows (it is the CEM population width). Pad a short path by
    # repeating its endpoint; SUBSAMPLE a long one rather than truncating it, so the audit
    # covers the whole transit instead of silently checking only its first n waypoints and
    # passing a path that clips the table near the end.
    if len(seg_home) > kin.n:
        idx = torch.linspace(0, len(seg_home) - 1, kin.n).round().long().tolist()
        probe = [seg_home[i] for i in idx]
    else:
        probe = seg_home + [q_high] * (kin.n - len(seg_home))
    g = kin.fk(torch.stack(probe))
    m = min(len(seg_home), kin.n)
    if float(g["low_z"][:m].min()) < LOW_Z_MIN:
        if verbose:
            print(f"    [plan] home transit dips to {float(g['low_z'][:m].min()) * 1000:.1f} mm")
        return None
    if avoid is not None and float(kin.box_penetration(g["bodies"][:m], avoid,
                                                       AVOID_MARGIN).max()) > 1e-6:
        if verbose:
            print("    [plan] home transit passes through clutter")
        return None

    # --- lift straight up, then carry to above the box ------------------------------------
    top = torch.tensor([float(grip[0]), float(grip[1]), LIFT_Z], device=dev)
    seg_lift, e_lift = _chain(kin, q_grip, _densify(grip, top), o_des, avoid=avoid)

    over = torch.tensor([float(box_xy[0]), float(box_xy[1]), CARRY_Z], device=dev)
    # the carry runs high above the desk, so the clutter cannot be hit and the keep-out
    # boxes only over-constrain the search
    seg_carry, e_carry = _chain(kin, seg_lift[-1], _densify(top, over), o_des)

    # --- retreat, fingers open, straight back up and away ---------------------------------
    seg_retreat, _ = _chain(kin, seg_carry[-1],
                            _densify(over, over + torch.tensor([0.0, 0.0, 0.05], device=dev)),
                            o_des)

    return {"home": seg_home, "pre": seg_pre, "grasp": seg_grasp, "lift": seg_lift,
            "carry": seg_carry, "retreat": seg_retreat,
            # Cartesian targets for the executor's diagnostics. Recorded here because `kin.fk`
            # writes joint state to read geometry back, so they cannot be recomputed during a
            # run without teleporting the arm mid-episode.
            # `high` is the Cartesian target; `high_fk` is what the FINAL COMMANDED joint
            # configuration actually produces. Reporting both separates "the planner solved
            # for somewhere else" from "the arm did not get where it was sent" -- two
            # diagnoses that a single position error cannot tell apart, and that want
            # completely different fixes.
            "tcp": {"pre": tcp_pre, "high": high.clone(), "grip": grip.clone(),
                    "high_fk": kin.fk(q_high.unsqueeze(0).repeat(kin.n, 1))["tcp"][0].clone(),
                    "top": top.clone(), "over": over.clone()},
            # Measured command offset for the grasp pose. Ramped in over the descent by the
            # executor: it is a property of THAT pose, and applying it at the transit height
            # -- where the arm holds fine -- would only push it off a pose it was hitting.
            # Off by default: MEASURED, at 32 envs with the teleport protocol, the plan-time
            # bias made real contact where none existed before (`gap` -1.2 mm on air -> 14.3 mm
            # on the cube) but did NOT raise success -- 3/32 against 11/32 with no bias at all.
            # It is the right idea aimed at the right cause and it is not yet paying, so it is
            # kept, disabled, and left for the session that finishes the transit work.
            "bias": (settle_bias(kin, q_grip) if os.environ.get("SETTLE_BIAS") == "1"
                     else torch.zeros_like(q_grip)),
            "grasp_info": {k: best[k] for k in ("pos_err", "o_align", "low_z")}}


def execute(kin: ArmKin, plan: dict, n: int, settle_close: int = 70,
            steps_per_wp: int = 12, record=None):
    """Run a solved plan through the env's own action manager.

    ``record``, if given, is called as ``record(action)`` immediately BEFORE each
    ``env.step`` so an (obs, action) pair stays causal -- the observation must be the one the
    action was chosen from, not the one that resulted.
    """
    def go(segments, close):
        prev = None
        for q in segments:
            qq = q.unsqueeze(0).repeat(n, 1)
            if prev is None:
                kin.hold(qq, 4, close=close)
            else:
                _run(kin, prev, qq, steps_per_wp, close, record)
            prev = qq
        return prev

    # Start where `env.reset()` actually leaves the arm, and drive to the pre-grasp with
    # actions. Teleporting to `plan["pre"]` instead would record episodes beginning in a state
    # the env never produces -- see `plan_episode`'s `home` segment.
    q_home = kin.q_arm0.unsqueeze(0).repeat(n, 1)
    kin.teleport_arm(q_home, q_fing=Q_OPEN)
    _hold(kin, q_home, 10, False, record)
    last = go(plan["home"] + plan["grasp"], close=False)
    # close -- every search is already finished (LESSONS A6)
    _hold(kin, last, settle_close, True, record)
    last = go(plan["lift"], close=True)
    last = go(plan["carry"], close=True)
    # release
    _hold(kin, last, 25, False, record)
    go(plan["retreat"], close=False)


def _hold(kin, q, steps, close, record):
    for _ in range(steps):
        a = kin.act(q, close)
        if record is not None:
            record(a)
        kin.env.step(a)


def _run(kin, q_from, q_to, steps, close, record, ramp=0.8):
    for s in range(steps):
        f = min(1.0, (s + 1) / max(1.0, steps * ramp))
        a = kin.act((1 - f) * q_from + f * q_to, close)
        if record is not None:
            record(a)
        kin.env.step(a)
