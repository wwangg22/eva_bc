# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Planner and execution schedule for the scripted slot-insertion expert.

Extracted **verbatim** from ``scripts/run_expert.py`` once a second consumer appeared
(``scripts/collect_demos.py``). The extraction is deliberate rather than a copy: the runner's
measured 100 % result and the demos handed to behaviour cloning must come from the *same*
trajectory, and two copies of 120 lines of geometry would drift silently. The regression test
is cheap and was run: ``run_expert.py`` on ``Rebot-PrecisionSlot-v0`` at n=128 must still
report 128/128 seated after the refactor.

Every constant here was forced by a measurement; see ``docs/slot/EXPERT_RESULTS.md`` for the
evidence and ``scripts/run_expert.py``'s module docstring for the design argument.

The execution schedule is exposed as a **generator of per-env-step commands**
(:func:`action_stream`) rather than a run loop, because the two consumers need different
things wrapped around the same steps: the runner traces and prints phase statistics, the
collector records observations and labels segments. Neither may change the schedule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from reBot_RL.tasks.manager_based.challenge import mdp

# Motion phases, in execution order. "close" / "release" / "retreat" / "idle" are holds and
# appear in the stream but not here.
PHASES = ("reach", "lift", "back", "spin", "turn", "push")

#: sim steps spent per Cartesian waypoint. Two is the floor at which the position controller
#: still tracks; the turn is slower because it is the phase that historically lost the grip.
PER_WP = {"reach": 2, "lift": 2, "back": 2, "spin": 2, "turn": 3, "push": 2}

#: sim steps held at the last waypoint of each phase, to let the arm and the swinging block
#: settle before the next phase accelerates them again.
SETTLE = {"reach": 8, "lift": 10, "back": 10, "spin": 12, "turn": 25, "push": 15}

WARMUP_STEPS = 5      # arm eases onto the first reach waypoint before anything is recorded
CLOSE_STEPS = 30      # fingers close on the block at the reach end pose
RELEASE_STEPS = 45    # fingers open at the insert pose; the block drops ~8 mm onto the floor
RETREAT_STEP_M = 0.004
RETREAT_PER_WP = 2


@dataclass
class ExpertParams:
    """Trajectory geometry. Defaults are the values that measured 100 % on all three
    clearances; see the table in ``docs/slot/HANDOFF.md`` section 5 for what forced each."""

    grasp_h: float = 0.031      # TCP height above the block centre at the grasp
    carry_z: float = 0.095      # TCP z while carrying and inserting
    stage_x: float = 0.165      # x to retract to before traversing in y
    insert_x: float = 0.2545    # target TCP x at the end of the push
    turn_per_wp: int = 3
    cem_iters: int = 120
    retreat: bool = True        # back the open gripper out of the slot after releasing
    seed_file: str = "logs/expert/seed_q.json"

    def per_wp(self) -> dict[str, int]:
        return {**PER_WP, "turn": self.turn_per_wp}


def lerp_path(a: torch.Tensor, b: torch.Tensor, step_m: float = 0.004,
              ease: bool = False) -> list[torch.Tensor]:
    """Straight Cartesian segment from (n,3) ``a`` to (n,3) ``b``, at most ``step_m`` apart.

    ``ease`` distributes the waypoints by smootherstep (6t^5 - 15t^4 + 10t^3) instead of
    uniformly. Executed at a fixed number of sim steps per waypoint, that gives zero velocity
    *and zero acceleration* at both ends of the segment.

    This is not cosmetic. The carried block hangs 33 mm below the grip point -- a pendulum of
    period 2*pi*sqrt(0.033/9.81) = 0.365 s -- and the y-traverse crosses 128 mm in about 4.4
    of those periods. Driven with uniform waypoints (a velocity step every 40 ms) the block
    rolled ~5 degrees in the pads, which presents 30*cos(phi) + 70*sin(phi) = 36 mm across a
    36 mm Loose-v0 channel and jams at the mouth.
    """
    k = max(2, int(torch.ceil((b - a).norm(dim=1).max() / step_m).item()) + 1)
    out = []
    for i in range(k):
        t = i / (k - 1)
        s = t * t * t * (t * (6 * t - 15) + 10) if ease else t
        out.append(a + (b - a) * s)
    return out


def polar_turn(x: float, y0: torch.Tensor, z: float, dth: float = 0.018) -> list[torch.Tensor]:
    """Swing the TCP to y = 0 at fixed x by sweeping the BASE ANGLE uniformly.

    A straight line in y is not smooth in joint space: the base yaw obeys
    ``dtheta1/dy = x / (x^2 + y^2)``, which grows ~50 % as y -> 0 at x = 0.180. Kept because
    it is harmless and slightly smoother, but note its motivating hypothesis was **refuted** --
    the grip loss it was written to fix turned out to be a collision (HANDOFF section 5a).
    """
    th0 = torch.atan2(y0, torch.full_like(y0, x))
    k = max(2, int(torch.ceil(th0.abs().max() / dth).item()) + 1)
    out = []
    for i in range(k):
        t = i / (k - 1)
        s = t * t * t * (t * (6 * t - 15) + 10)
        th = th0 * (1.0 - s)
        out.append(torch.stack([torch.full_like(th, x), x * torch.tan(th),
                                torch.full_like(th, z)], dim=1))
    return out


def check_geometry(p: ExpertParams) -> None:
    """Refuse an insert target whose block nose would drive through the back stop.

    Depth is measured on the block CENTRE (``mdp.insertion_depth`` reads ``root_pos``), so the
    naive target ``mouth + SUCCESS_DEPTH`` is not the constraint -- the back stop is.
    """
    max_x = mdp.SLOT_CENTER[0] + mdp.SLOT_DEPTH / 2 - mdp.BLOCK_HALF[0]
    if p.insert_x > max_x:
        raise SystemExit(f"insert_x {p.insert_x} > {max_x:.4f} (block nose through the back stop)")


def solve_seed(env, ik, p: ExpertParams, verbose: bool = True) -> dict:
    """Return the canonical wrist/elbow branch: ``q_seed`` (n,6), ``axis_slot`` (n,3), ``sign``.

    CEM's only job is to hand DLS a seed in the right elbow branch; DLS does the rest. A
    branch jump between adjacent waypoints is what made the CEM-only sweep return 158 mm of
    tracking error, so nothing downstream trusts CEM for accuracy.

    CEM's population size is ``num_envs``, so a small-n run gets a badly seeded branch:
    measured, n=4 scored 50 % where n=128 scored 100 % on the SAME task, with the traverse
    ending at y = -12.9 mm instead of 0. The seed is a property of the robot, not of the batch,
    so it is validated once at large n and cached. That also makes the expert deterministic,
    which demo collection needs.
    """
    dev, n = ik.dev, ik.n
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)
    nominal = torch.tensor([p.insert_x, 0.0, p.carry_z], device=dev)

    seed_path = Path(p.seed_file)
    if not seed_path.is_absolute():
        seed_path = Path(__file__).resolve().parents[1] / seed_path
    cached = None
    if seed_path.exists():
        d = json.loads(seed_path.read_text())
        if d.get("insert_x") == p.insert_x and d.get("carry_z") == p.carry_z:
            cached = torch.tensor(d["q"], device=dev)
            if verbose:
                print(f"  seed: loaded from {seed_path}")

    mean, std = ik.q_default[ik.arm_dof].clone(), torch.full((6,), 0.40, device=dev)
    best = {"cost": 1e9}
    for _ in range(0 if cached is not None else p.cem_iters):
        q = (mean + std * torch.randn((n, 6), device=dev)).clamp(ik.lo, ik.hi)
        tcp, sep, _ = ik.fk(q, 0.016)
        cost = (tcp - nominal).norm(dim=1) + 0.25 * (1.0 - (sep @ y_axis).abs())
        elite = q[cost.argsort()[: max(8, n // 20)]]
        mean, std = elite.mean(0), elite.std(0).clamp(min=0.008)
        i = int(cost.argmin())
        if float(cost[i]) < best["cost"]:
            best = {"cost": float(cost[i]), "q": q[i].clone()}

    q_seed = (cached if cached is not None else best["q"]).unsqueeze(0).repeat(n, 1)
    _, sep, _ = ik.fk(q_seed, 0.016)
    sign = torch.sign((sep @ y_axis).mean()).clamp(min=-1.0, max=1.0)
    axis_slot = (sign * y_axis).expand(n, 3).contiguous()
    ref = ik.solve(nominal.expand(n, 3), axis_slot, q_seed, 0.016, iters=200)
    if cached is None:
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(json.dumps({"q": ref["q"][0].cpu().tolist(), "insert_x": p.insert_x,
                                         "carry_z": p.carry_z, "n": n}, indent=2))
        if verbose:
            print(f"  seed: solved by CEM at n={n} and cached to {seed_path}")
    if verbose:
        print(f"  jacobian: {ik.jac_mode}")
        print(f"  seed pose: pos err {float(ref['pos_err'].mean()) * 1000:.3f} mm, "
              f"axis err {float(ref['axis_err'].mean()):.6f}, "
              f"converged {int(ref['converged'].sum())}/{n}")
    return {"q_seed": ref["q"], "axis_slot": axis_slot, "sign": sign, "ref": ref}


def plan(ik, p: ExpertParams, bp0: torch.Tensor, byaw0: torch.Tensor,
         q_seed: torch.Tensor, axis_slot: torch.Tensor, sign: torch.Tensor) -> dict:
    """Solve every phase of the trajectory from the post-reset block pose (n,3)/(n,).

    **All IK runs before the block is touched.** ``write_joint_state_to_sim`` teleports the arm
    and re-opens the fingers, so any solve after the grasp destroys it.

    Returns ``{"plans": {phase: solve_path output}, "segs": {phase: [waypoints]}}``.
    """
    dev, n = ik.dev, ik.n

    # Grasp axis = the block's own local y, so the pads squeeze its 30 mm width, not its
    # 45 mm length. Measured on this arm: 0 % -> 55-81 % insert rate when this is enforced.
    axis_grasp = torch.stack([-torch.sin(byaw0), torch.cos(byaw0), torch.zeros_like(byaw0)], dim=1) * sign
    gz = bp0[:, 2] + p.grasp_h
    carry = torch.full((n,), p.carry_z, device=dev)
    p_hover = torch.stack([bp0[:, 0], bp0[:, 1], gz + 0.045], dim=1)
    p_grasp = torch.stack([bp0[:, 0], bp0[:, 1], gz], dim=1)
    p_lift = torch.stack([bp0[:, 0], bp0[:, 1], carry], dim=1)
    p_stage = torch.stack([torch.full((n,), p.stage_x, device=dev), bp0[:, 1], carry], dim=1)
    p_align = torch.tensor([p.stage_x, 0.0, p.carry_z], device=dev).expand(n, 3)
    p_ins = torch.tensor([p.insert_x, 0.0, p.carry_z], device=dev).expand(n, 3)

    def rot_axis(f: float) -> torch.Tensor:
        """Finger axis interpolated from the block's yaw to the slot axis."""
        a = byaw0 * (1.0 - f)
        return torch.stack([-torch.sin(a), torch.cos(a), torch.zeros_like(a)], dim=1) * sign

    segs = {}
    segs["reach"] = [p_hover] + lerp_path(p_hover, p_grasp, 0.004)
    segs["lift"] = lerp_path(p_grasp, p_lift, 0.004)
    segs["back"] = lerp_path(p_lift, p_stage, 0.005, ease=True)
    # Spin in place, THEN traverse. Doing both at once cost 17/64 grips: the block hangs 33 mm
    # below the grip point, so accelerating it 128 mm sideways swings it about the push axis
    # while the wrist is also twisting it. The pads then see a 35 mm presented width instead of
    # 30 mm and are forced open (measured gap 29.96 -> 35.00 mm).
    segs["spin"] = [p_stage] * 24
    segs["turn"] = polar_turn(p.stage_x, bp0[:, 1], p.carry_z)
    segs["push"] = lerp_path(p_align, p_ins, 0.002)

    axes = {"reach": axis_grasp, "lift": axis_grasp, "back": axis_grasp,
            "spin": [rot_axis(i / (len(segs["spin"]) - 1)) for i in range(len(segs["spin"]))],
            "turn": axis_slot, "push": axis_slot}

    plans, q = {}, q_seed
    for name in PHASES:
        plans[name] = ik.solve_path(segs[name], axes[name], q, 0.016)
        q = plans[name]["q"][:, -1, :]

    if p.retreat:
        # Back the OPEN gripper out of the slot so the episode ends with the arm clear of the
        # block. Zero interference risk in principle -- the pads sit at |y| = 45 mm, outside
        # the wall outer faces at 24.5 mm, and travel in -x away from the released block -- but
        # it is checked against the no-retreat result rather than asserted.
        segs["retreat"] = lerp_path(p_ins, p_align, RETREAT_STEP_M)
        plans["retreat"] = ik.solve_path(segs["retreat"], axis_slot, q, 0.045)

    return {"plans": plans, "segs": segs}


@dataclass
class Step:
    """One env step's command. ``wp`` is the waypoint index, or -1 during a settle/hold."""

    phase: str
    wp: int
    q: torch.Tensor   # (n, 6) arm joint targets
    close: bool


def action_stream(plans: dict, p: ExpertParams, budget: int | None = None) -> Iterator[Step]:
    """Yield one :class:`Step` per env step, in execution order.

    ``budget`` (env steps) pads the tail with an idle hold so every demo has the same length
    and no auto-reset fires mid-recording. The 600-step episode budget is HARD: an earlier
    expert ran 659 steps, timed out every env and read as 0 % success.
    """
    per_wp = p.per_wp()
    n_emitted = 0

    def emit(phase: str, wp: int, q: torch.Tensor, close: bool, k: int) -> Iterator[Step]:
        nonlocal n_emitted
        for _ in range(k):
            n_emitted += 1
            yield Step(phase, wp, q, close)

    reach = plans["reach"]["q"]
    yield from emit("warmup", -1, reach[:, 0, :], False, WARMUP_STEPS)
    for name in PHASES:
        qs = plans[name]["q"]
        close = name != "reach"
        for t in range(qs.shape[1]):
            yield from emit(name, t, qs[:, t, :], close, per_wp[name])
        yield from emit(name, -1, qs[:, -1, :], close, SETTLE.get(name, 8))
        if name == "reach":
            yield from emit("close", -1, qs[:, -1, :], True, CLOSE_STEPS)

    q_end = plans["push"]["q"][:, -1, :]
    yield from emit("release", -1, q_end, False, RELEASE_STEPS)
    if "retreat" in plans:
        qs = plans["retreat"]["q"]
        for t in range(qs.shape[1]):
            yield from emit("retreat", t, qs[:, t, :], False, RETREAT_PER_WP)
        q_end = qs[:, -1, :]
    if budget is not None and n_emitted < budget:
        yield from emit("idle", -1, q_end, False, budget - n_emitted)
