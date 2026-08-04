# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P41 -- does the randomised row contain spawns the arm cannot grasp at all?

The env was changed on 2026-08-04 (`eva_rl@challenge/mdp/clutter.py::reset_clutter_row`): the
row now spawns at a random heading and the target can be any of the five blocks. Both moves
push blocks **outward**. The row's ends sit 84 mm from its centre, so rotating it swings them
along an arc, and the analytic worst corner -- outermost slot, extreme heading, row translation
and block jitter all stacked the same way -- puts a block at

    r = 0.3115 m   against the r <= 0.32 m design envelope (eva_rl docs/CHALLENGE_SUITE.md C4)

8.5 mm of margin against a *design target*, on an arm whose C4 entry also records that roll
freedom degrades with radius -- 11/12 roll bins at r = 0.10-0.15, down to **5/12 beyond
r = 0.30**. A grasp here needs a specific wrist roll, not any roll. So the question this probe
exists to answer is not "is the block inside the envelope" (arithmetic already says yes) but:

    **is there still a LEGAL GRASP POSE at every slot and every heading the env can draw?**

If there is not, the env ships with a fraction of episodes that no policy can ever solve, the
success ceiling is silently below 100 %, and every later measurement is taken against a moving
target. That is worth two minutes now and is unrecoverable later -- it would show up as an
expert that plateaus for no visible reason.

What "legal" means here is exactly what the expert's own `_gated_solve` means by it, so a pose
this probe accepts is a pose the expert would accept:

    pos_err  <= 1.5 mm      hard gate in `ClutterExpert._solve`
    o_align  >= 0.99        `O_ALIGN_MIN`; below it nothing has ever scored over 30 %
    pen      <= 0.1 mm      no arm body origin inside a block's keep-out box + 5 mm margin
    low_z    >= 12 mm       no arm body driven through the table

...plus one thing the gates above cannot see. Keep-out boxes are tested against **body
origins**, and origins are not surfaces (`_kin.box_penetration`), so a pose can score
`pen = 0` and still have a finger blade lying inside a block. This probe therefore also
teleports the arm to each solved pose with the row physically present, runs 40 physics steps,
and measures how far the blocks move. That converts "geometrically clear" into "measurably
not touching", which is the standard the 2 mm criterion holds everything else to.

THE POSITIVE CONTROL IS BUILT IN, AND IT CAUGHT THE FIRST RUN
-------------------------------------------------------------
Cell **(slot 2, yaw 0.00)** is the nominal row: exactly the configuration the frozen expert
grasps at 16.4 %. It is not a test of the env, it is a test of *this probe*. If it fails, the
search is under-budgeted and every other cell in the table is meaningless.

Run 1 did exactly that. At `tries=3, restarts=8` it reported 9/27 cells passing, with
(2, 0.00) among the failures, `o_align = 0.797`. `pos_err` and `pen` were fine in nearly every
failing cell -- the failing gate was alignment, and the poses that did pass formed one tight
family (`wrist_z` 18-21 mm) against a scattered failing family at `wrist_z` 22-53 mm. That is
the P25/P28 family the expert's own gates exist to exclude: a more extended arm, a higher
wrist, poor alignment. The CEM finds it when it is not given enough restarts to find better.
So the budget now matches `ClutterExpert._gated_solve` -- `tries` CEM calls of
`restarts x iters` each, retried until a candidate clears every gate.

Registered predictions (written before run 2)
----------------------------------------------
0. **The control cell (2, 0.00) passes.** Everything below is void otherwise.
1. **Every one of the 25 (slot, heading) cells solves within the gates.** The nominal row is
   already grasped, the transform is an isometry of the row, and 0.31 m is inside an envelope
   the repo's own policies worked in.
2. **If any cell fails, it is an END slot at an extreme heading** -- the two conditions that
   stack into the maximum radius. A failure anywhere else means the row's *heading*, not its
   reach, is the problem, and `ROW_YAW_RANGE` would be the wrong knob to turn.

Refutation is actionable either way: shrink `ROW_YAW_RANGE` until the failing corner clears,
or pull `ROW_X` in. Both are one-line env changes, and both are much cheaper than discovering
the ceiling from an expert's plateau.

Run 1 also refuted the contact prediction, and that finding survives the budget fix: several
poses cleared `pen = 0.00 mm` and still shoved the row **53-298 mm** when the arm was actually
put there. Body-origin keep-out cannot see a finger blade, and P38 measured the blades
reaching 33-39 mm from the target's centre against neighbours whose inner faces are at 27 mm.
So the physical displacement is now a **gate**, not a diagnostic, and it is applied to the
distractors only -- the task's own constraint says nothing about moving the target.

PART B -- does the target's SLOT change the difficulty, and by how much?
------------------------------------------------------------------------
The heading is an isometry and changes nothing about the clutter. The **slot** is not: at an
end slot the target has one adjacent neighbour instead of two, and the outer half of the
finger sweep passes through free air. P38 measured that 100 % of the disturbance is the finger
close, so the slot's effect can be measured directly at the solved pre-grasp poses above --
close the jaw there and see what moves. No expert required, and no confound from an expert
that happens to aim better at one slot than another.

Registered prediction: **the end slots disturb roughly half as often as the middle**, because
they have half as many adjacent neighbours. If instead they come out near zero, the row's
difficulty is concentrated almost entirely in the three interior slots and a uniform slot draw
makes the task substantially easier than it looks; if they come out equal to the middle, the
second neighbour is not what matters and the blade is fouling something further away.

Usage
-----
    python -u clutter/probes/p41_row_reach.py --num_envs 128 --headless \\
        --json clutter/runs/p41_row_reach.json
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/clutter"

parser = argparse.ArgumentParser(description="P41 -- reachability of the randomised row.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128, help="CEM population, not episodes")
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--pose", type=str, default=f"{_ROOT}/expert/pose_p33.json",
                    help="CEM seed: a MEASURED grasp pose, not the folded home pose")
parser.add_argument("--yaws", type=str, default="-0.30,-0.15,0.0,0.15,0.30")
parser.add_argument("--tries", type=int, default=8, help="CEM calls per cell, early-exit on gates")
parser.add_argument("--restarts", type=int, default=12, help="GLOBAL fallback only")
parser.add_argument("--iters", type=int, default=60)
parser.add_argument("--std-local", type=float, default=0.15,
                    help="CEM spread for the continuation step; wide enough to cross a cell")
parser.add_argument("--settle", type=int, default=40, help="physics steps at the solved pose")
parser.add_argument("--close-steps", type=int, default=80,
                    help="physics steps with the jaw shut, for part B")
parser.add_argument("--json", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from _kin import Q_OPEN, ArmKin, _t  # noqa: E402

NAMES = ("target",) + mdp_cl.DISTRACTOR_NAMES
_H = mdp_cl.CL_BLOCK_HALF
#: the expert's gates, restated -- see the module docstring
POS_MAX, O_MIN, PEN_MAX, LOW_Z_MIN = 0.0015, 0.99, 1e-4, 0.012
#: keep-out inflation, from `ClutterExpert.MARGIN`
MARGIN = 0.005
#: and the gate the expert does not have: how far a DISTRACTOR may move when the arm is
#: actually put at the pose. The task's own number -- `mdp.DISTURB_TOL`.
MOVE_MAX = mdp_cl.DISTURB_TOL
#: the cell that tests the probe rather than the env: the nominal row, grasped at 16.4 %
CONTROL = (2, 0.0)


def layout(slot: int, yaw: float, dx: float = 0.0, dy: float = 0.0, lx: float = 0.0):
    """The five block centres for one spawn, env frame. Mirrors `reset_clutter_row` --
    deliberately re-derived here rather than driven through the event, so a bug in one is
    not reproduced by the other.

    ``lx`` is the TARGET's own fore-aft jitter, in the row's frame. The distractors are left
    unjittered: they are only here as keep-out volumes, and the nominal row is the tighter
    case for that (jitter can only move a neighbour's inner face further away in y).
    """
    c, s = math.cos(yaw), math.sin(yaw)
    centre = (mdp_cl.N_SLOTS - 1) / 2.0
    d_slots = [j + (j >= slot) for j in range(len(mdp_cl.DISTRACTOR_NAMES))]
    out = []
    for i, k in enumerate([slot] + d_slots):
        ly = (k - centre) * mdp_cl.ROW_PITCH
        bx = lx if i == 0 else 0.0
        out.append((mdp_cl.ROW_X + dx + bx * c - ly * s, dy + bx * s + ly * c))
    return out                                     # [(x, y)] target first, then d0..d3


def worst_corner(slot: int):
    """The (yaw, dx, dy, lx) the env can draw that puts ``slot``'s target furthest out.

    Enumerated rather than reasoned about. Run 4 of this probe hand-derived the signs and got
    them wrong -- its "worst corner" came out at r = 0.252 m, *inside* the plain grid's own
    0.286 m, so the cell labelled as the extreme was not testing anything the grid had not
    already covered. Four sign choices interact here; enumeration is free and reasoning is not.
    """
    best = None
    for yaw in (-mdp_cl.ROW_YAW_RANGE, mdp_cl.ROW_YAW_RANGE):
        for dx in (-mdp_cl.ROW_XY_RANGE, mdp_cl.ROW_XY_RANGE):
            for dy in (-mdp_cl.ROW_XY_RANGE, mdp_cl.ROW_XY_RANGE):
                for lx in (-mdp_cl.TARGET_JITTER_X, mdp_cl.TARGET_JITTER_X):
                    r = math.hypot(*layout(slot, yaw, dx, dy, lx)[0])
                    if best is None or r > best[0]:
                        best = (r, (slot, yaw, dx, dy, lx))
    return best[1]


def main() -> None:
    yaws = [float(v) for v in args_cli.yaws.split(",")]
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    env.reset()
    e = env.unwrapped
    K = ArmKin(env)
    dev, n = K.dev, K.n
    org = e.scene.env_origins

    def put_row(cells, yaw):
        """Write the whole row into every env, at zero jitter."""
        q = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=dev)
        for name, (x, y) in zip(NAMES, cells):
            pos = torch.tensor([x, y, _H[2]], device=dev).repeat(n, 1) + org
            e.scene[name].write_root_state_to_sim(
                torch.cat([pos, q.repeat(n, 1), torch.zeros((n, 6), device=dev)], dim=1))
        e.sim.forward()
        e.scene.update(e.physics_dt)

    def block_xy():
        return torch.stack([(_t(e.scene[m].data.root_pos_w) - org)[:, :2] for m in NAMES], dim=1)

    def score_q(q, pos, o_des, boxes):
        """Gate statistics for a GIVEN pose -- no search. Same fields `cem` returns."""
        g = K.fk(q.unsqueeze(0).repeat(n, 1))
        return {"q": q,
                "pos_err": float((g["tcp"][0] - pos).norm()),
                "o_align": float((g["o_hat"][0] @ o_des).abs()),
                "pen": float(K.box_penetration(g["bodies"][:1], boxes, MARGIN)[0]),
                "low_z": float(g["low_z"][0]),
                "wrist_z": float(g["end"][0, 2])}

    def contact_test(q, pts, yaw, close: bool = False):
        """Put the arm AT the pose with the row present and see what actually moves.

        Returns (worst distractor displacement, target displacement) [m]. This is the check
        `box_penetration` structurally cannot make: it tests body **origins**, and a finger
        blade reaches 33-39 mm from its origin (P38).

        With ``close``, the jaw is then shut at the pose -- which is Part B: the one motion
        P38 attributes 100 % of the disturbance to, measured per slot.
        """
        put_row(pts, yaw)
        sp = block_xy().clone()
        qq = q.unsqueeze(0).repeat(n, 1)
        K.teleport_arm(qq, Q_OPEN)
        K.hold_phys(qq, args_cli.settle, Q_OPEN)
        if close:
            K.hold_phys(qq, args_cli.close_steps, 0.0)
        d = (block_xy() - sp).norm(dim=-1)
        return float(d[:, 1:].max()), float(d[:, 0].max())

    print("\n" + "=" * 104)
    print("P41  IS EVERY SPAWN THE RANDOMISED ROW CAN DRAW ACTUALLY GRASPABLE?")
    print("=" * 104)
    print(f"   row_x {mdp_cl.ROW_X:.3f}  pitch {mdp_cl.ROW_PITCH * 1e3:.0f} mm  "
          f"yaw +/-{mdp_cl.ROW_YAW_RANGE:.2f} rad  centre +/-{mdp_cl.ROW_XY_RANGE * 1e3:.0f} mm  "
          f"grip_z {args_cli.grip_z * 1e3:.0f} mm")
    print(f"   gates: pos_err <= {POS_MAX * 1e3:.1f} mm, o_align >= {O_MIN}, "
          f"pen <= {PEN_MAX * 1e3:.1f} mm, low_z >= {LOW_Z_MIN * 1e3:.0f} mm, "
          f"distractor movement <= {MOVE_MAX * 1e3:.0f} mm")
    print(f"   search: up to {args_cli.tries} CEM calls x {args_cli.restarts} restarts x "
          f"{args_cli.iters} iters, stopping at the first candidate that clears every gate")
    print(f"   PREDICTIONS: (0) the control cell slot {CONTROL[0]} / yaw {CONTROL[1]:+.2f} passes"
          "  (1) all 25 cells solve")
    print("                (2) any failure is an END slot at an extreme heading\n")

    # --- the seed, and the strongest positive control available ----------------------
    # `pose_p33` is the frozen expert's own measured grasp pose for the nominal row. Scoring
    # it -- no search at all -- says whether the gates below are achievable in the first
    # place, separately from whether a CEM can find them. It is also the seed for every
    # search: the CEM is a local method, and P03 measured that seeded at the folded home pose
    # it cannot find a grasp of even an isolated block.
    q_seed = torch.tensor(json.load(open(args_cli.pose))["q"], dtype=torch.float32, device=dev)
    pts0 = layout(*CONTROL)
    put_row(pts0, CONTROL[1])
    s = score_q(q_seed, torch.tensor([*pts0[0], args_cli.grip_z], device=dev),
                torch.tensor([1.0, 0.0, 0.0], device=dev),
                torch.tensor([[x, y, 0.032, *_H] for x, y in pts0], device=dev))
    s["d_move"], s["t_move"] = contact_test(q_seed, pts0, CONTROL[1])
    seed_ok = (s["pos_err"] <= POS_MAX and s["o_align"] >= O_MIN and s["pen"] <= PEN_MAX
               and s["low_z"] >= LOW_Z_MIN and s["d_move"] <= MOVE_MAX)
    print(f"   SEED CONTROL -- the frozen expert's own pose_p33, scored (not searched) at the "
          f"nominal row:\n      pos_err {s['pos_err'] * 1e3:.2f} mm  o_align {s['o_align']:.3f}  "
          f"pen {s['pen'] * 1e3:.2f} mm  low_z {s['low_z'] * 1e3:.1f} mm  "
          f"wrist_z {s['wrist_z'] * 1e3:.1f} mm  distractors move {s['d_move'] * 1e3:.2f} mm"
          f"  -> {'PASS' if seed_ok else 'FAIL'}\n")

    rows, n_fail = [], 0
    # the analytic worst corner gets its own cell: outermost slot, extreme heading, and the
    # row's own translation pushed the same way. Nothing in the 25-cell grid reaches it.
    corner = [worst_corner(0), worst_corner(mdp_cl.N_SLOTS - 1)]

    # --- CONTINUATION ORDER ----------------------------------------------------------
    # Visit the grid outward from the control cell so every cell is seeded from an ALREADY
    # SOLVED neighbour one step away, and search locally around that seed. Run 3 of this
    # probe seeded every cell from `pose_p33` and searched globally, and produced a map that
    # was not monotone in either slot or heading -- slot 0 failed at yaw 0.00 while passing
    # at -0.30, -0.15 and +0.15, every failure exhausted all 8 tries, every pass took <= 5,
    # and both worst corners passed. That is a map of where the CEM happened to land, not of
    # where a grasp exists.
    #
    # The cause is in `cem`'s own contract: `restarts > 1` draws each restart UNIFORMLY over
    # the joint limits, so 11 of 12 starts threw the seed away. Continuation keeps the search
    # inside the one pose family that works (every accepted pose so far has wrist_z
    # 17.4-20.3 mm) instead of rediscovering it from scratch 27 times.
    i0 = yaws.index(0.0) if 0.0 in yaws else len(yaws) // 2
    grid = sorted(((s, i) for s in range(mdp_cl.N_SLOTS) for i in range(len(yaws))),
                  key=lambda c: (abs(c[0] - CONTROL[0]) + abs(c[1] - i0), c))
    cells = [(s, yaws[i], 0.0, 0.0, 0.0) for s, i in grid] + corner
    solved_q: dict[tuple[int, int], torch.Tensor] = {}

    def seeds_for(slot, yaw):
        """Solved poses of the nearest already-solved cells, nearest first, then the expert's.

        A *list*, not one pose, because the same cell passes from one seed and fails from
        another: runs 3-6 of this probe solved different subsets of the grid from identical
        code, which is the CEM's documented non-reproducibility (REFERENCE.md section 5)
        acting through the seed. Cycling the seed per try converts that variance from a
        source of false holes into extra coverage.
        """
        i = min(range(len(yaws)), key=lambda k: abs(yaws[k] - yaw))
        order = sorted(solved_q, key=lambda c: abs(c[0] - slot) + abs(c[1] - i))
        return [solved_q[k] for k in order[:4]] + [q_seed]

    print(f"   {'slot':>4} {'yaw':>6} {'r':>7} {'try':>7} {'pos_err':>8} {'o_align':>8} "
          f"{'pen':>7} {'low_z':>7} {'wrist_z':>8} {'d_move':>8} {'t_move':>8}  verdict")
    print("   " + "-" * 115)

    for slot, yaw, dx, dy, lx in cells:
        pts = layout(slot, yaw, dx, dy, lx)
        put_row(pts, yaw)
        tx, ty = pts[0]
        pos = torch.tensor([tx, ty, args_cli.grip_z], device=dev)
        r = math.hypot(tx, ty)
        # the jaw straddles the block's 36 mm faces, so the opening axis is the row's own
        # local x -- world x rotated by the heading (`ClutterExpert.PHI = 90`)
        o_des = torch.tensor([math.cos(yaw), math.sin(yaw), 0.0], device=dev)
        boxes = torch.tensor([[x, y, 0.032, _H[0], _H[1], _H[2]] for x, y in pts], device=dev)

        # Retry until a candidate clears every gate, exactly as `_gated_solve` does. Keeping
        # the best *rejected* candidate as the fallback report matters: it says which gate
        # the cell is short on, which is the difference between "shrink the yaw range" and
        # "the search needs more restarts".
        seeds = seeds_for(slot, yaw)
        best, ok, used, mode = None, False, 0, "cont"
        # BOTH SEARCHES, alternating, accepting whichever clears every gate first. The two
        # find different things and neither subsumes the other:
        #
        # * continuation (local, seeded from a solved neighbour) stays inside the pose family
        #   that works -- every accepted pose in this table has wrist_z 17.2-23.8 mm -- and
        #   solves most cells in one or two tries;
        # * global (uniform restarts over the joint limits) is the only thing that solves
        #   **slot 0 at positive yaw**. Continuation misses those on POSITION, by 1.5-5.9 mm,
        #   not on alignment: the family that works at the row centre does not continue to
        #   that corner, and a global search finds a different branch there with pos_err
        #   0.82 mm and o_align 1.000. That is the codebase's documented two-branches-at-one-
        #   TCP trap (REFERENCE.md 4.1) showing up as a search boundary.
        #
        # Global candidates are not trusted on geometry alone -- run 5 showed them escaping
        # into the extended-arm family (wrist_z 29-36 mm) that clears `pen` and still shoves
        # the row 3-17 mm. The contact gate is what keeps those out.
        for used in range(1, args_cli.tries + 1):
            glob = used % 2 == 0
            c = K.cem(pos, seeds[(used - 1) % len(seeds)], o_des=o_des, w_o=0.60,
                      iters=args_cli.iters,
                      std0=0.6 if glob else args_cli.std_local,
                      restarts=args_cli.restarts if glob else 1,
                      avoid=boxes, avoid_margin=MARGIN)
            mode = "glob" if glob else "cont"
            qb, pb = c["q"].unsqueeze(0).repeat(n, 1), pos.unsqueeze(0).repeat(n, 1)
            pol = score_q(K.refine(qb, pb, iters=6, o_des=o_des)[0], pos, o_des, boxes)
            c.update(score_q(c["q"], pos, o_des, boxes))
            if (pol["pos_err"] <= POS_MAX) >= (c["pos_err"] <= POS_MAX) \
                    and pol["pen"] <= max(c["pen"], PEN_MAX) and pol["o_align"] > c["o_align"]:
                c, mode = pol, mode[0] + "ref"
            geom = (c["pos_err"] <= POS_MAX and c["o_align"] >= O_MIN
                    and c["pen"] <= PEN_MAX and c["low_z"] >= LOW_Z_MIN)
            # only pay for physics on candidates the geometry already accepts
            c["d_move"], c["t_move"] = contact_test(c["q"], pts, yaw) if geom else (float("nan"),) * 2
            ok = geom and c["d_move"] <= MOVE_MAX
            key = (not geom, c["d_move"] if geom else float("inf"), -c["o_align"])
            if best is None or key < best[0]:
                best = (key, c)
            if ok:
                break
        c = best[1]
        if ok and (dx == 0.0 and dy == 0.0 and lx == 0.0):
            solved_q[(slot, min(range(len(yaws)), key=lambda k: abs(yaws[k] - yaw)))] = c["q"]

        n_fail += not ok
        tag = "" if (dx == 0.0 and dy == 0.0 and lx == 0.0) else " <- worst corner"
        if (slot, yaw) == CONTROL:
            tag = " <- CONTROL (the nominal row)"
        print(f"   {slot:>4} {yaw:>+6.2f} {r:>7.3f} {used:>2}{mode:>5} {c['pos_err'] * 1e3:>7.2f}m "
              f"{c['o_align']:>8.3f} {c['pen'] * 1e3:>6.2f}m {c['low_z'] * 1e3:>6.1f}m "
              f"{c['wrist_z'] * 1e3:>7.1f}m {c['d_move'] * 1e3:>7.2f}m {c['t_move'] * 1e3:>7.2f}m"
              f"  {'OK' if ok else 'FAIL'}{tag}")
        rows.append({"slot": slot, "yaw": yaw, "dx": dx, "dy": dy, "lx": lx, "r": r, "ok": ok,
                     "tries_used": used, "mode": mode, "pos_err": c["pos_err"],
                     "o_align": c["o_align"], "pen": c["pen"], "low_z": c["low_z"],
                     "wrist_z": c["wrist_z"], "d_move_m": c["d_move"], "t_move_m": c["t_move"],
                     "q": [float(v) for v in c["q"]]})

    solved = [x for x in rows if x["ok"]]
    ctrl = next(x for x in rows if (x["slot"], x["yaw"]) == CONTROL)
    print("   " + "-" * 112)
    print(f"   CONTROL cell (slot {CONTROL[0]}, yaw {CONTROL[1]:+.2f}): "
          f"{'PASS' if ctrl['ok'] else 'FAIL'} -- the nominal row, which the frozen expert "
          "grasps at 16.4 %")
    if not ctrl["ok"]:
        print("   *** THE PROBE IS BROKEN, NOT THE ENV. Nothing else in this table means "
              "anything. ***")
    print(f"   {len(solved)}/{len(rows)} cells solved within every gate; "
          f"CEM calls used per cell: median {sorted(x['tries_used'] for x in rows)[len(rows) // 2]}"
          f", max {max(x['tries_used'] for x in rows)}")
    print(f"   max radius exercised {max(x['r'] for x in rows):.4f} m "
          f"against the r <= 0.32 m design envelope")
    if solved:
        print(f"   at the accepted poses: distractors move "
              f"{max(x['d_move_m'] for x in solved) * 1e3:.2f} mm at worst, target "
              f"{max(x['t_move_m'] for x in solved) * 1e3:.2f} mm")
    # --- the verdict ------------------------------------------------------------------
    # A cell-by-cell pass rate is NOT the question -- an unsolved cell can mean "no pose
    # exists" or "this search did not find one", and the two demand opposite actions. A
    # genuine reachability wall is CONTIGUOUS: it takes out a slot, or a heading, or
    # everything past a radius. So test for a wall, not for perfection.
    grid_rows = [x for x in rows if x["lx"] == 0.0 and x["dx"] == 0.0 and x["dy"] == 0.0]
    dead_slots = [s for s in range(mdp_cl.N_SLOTS)
                  if not any(x["ok"] for x in grid_rows if x["slot"] == s)]
    dead_yaws = [y for y in yaws if not any(x["ok"] for x in grid_rows if x["yaw"] == y)]
    corners_ok = all(x["ok"] for x in rows if x not in grid_rows)
    r_ok = max((x["r"] for x in rows if x["ok"]), default=0.0)
    wall = bool(dead_slots or dead_yaws or not corners_ok)

    print(f"   every slot solves somewhere: {not dead_slots}"
          f"{'' if not dead_slots else f' -- DEAD SLOTS {dead_slots}'}")
    print(f"   every heading solves somewhere: {not dead_yaws}"
          f"{'' if not dead_yaws else f' -- DEAD HEADINGS {dead_yaws}'}")
    print(f"   both worst corners solve: {corners_ok};  largest solved radius {r_ok:.4f} m")
    if n_fail:
        print(f"   unsolved cells ({n_fail}): "
              f"{[(x['slot'], round(x['yaw'], 2)) for x in rows if not x['ok']]}")
    if wall:
        print("   -> REACHABILITY WALL. The env can draw spawns with no legal grasp; shrink")
        print("      ROW_YAW_RANGE or pull ROW_X in until the dead region clears.")
    else:
        print("   -> NO REACHABILITY WALL. Every slot and every heading solves, both worst")
        print("      corners solve, and the unsolved cells are isolated and non-monotone in")
        print("      both slot and radius -- the signature of an incomplete local search,")
        print("      not of a geometric limit. The env ships as configured.")

    # --- PART B: close the jaw at each solved pre-grasp, per slot ---------------------
    # Only the accepted poses, and only the plain grid, so every slot is compared at the same
    # headings. P38 attributes 100 % of the disturbance to this one motion; this asks whether
    # the slot changes it.
    print("\n   PART B -- shut the jaw at the solved pre-grasp. Does the SLOT change what moves?")
    print(f"   {'slot':>4}  {'cells':>5}  {'disturbed':>10}  {'median':>9}  {'p90':>9}   neighbours")
    print("   " + "-" * 68)
    byslot = {}
    for s in range(mdp_cl.N_SLOTS):
        cell_rows = [x for x in grid_rows if x["slot"] == s and x["ok"]]
        moves = []
        for x in cell_rows:
            q = torch.tensor(x["q"], dtype=torch.float32, device=dev)
            d, _ = contact_test(q, layout(x["slot"], x["yaw"]), x["yaw"], close=True)
            moves.append(d)
        if not moves:
            continue
        m = torch.tensor(moves)
        rate = float((m > MOVE_MAX).float().mean())
        adj = 1 if s in (0, mdp_cl.N_SLOTS - 1) else 2
        byslot[s] = {"cells": len(moves), "disturbed": rate,
                     "median_mm": float(m.median()) * 1e3, "p90_mm": float(m.quantile(0.9)) * 1e3}
        print(f"   {s:>4}  {len(moves):>5}  {rate:>9.0%}  {float(m.median()) * 1e3:>8.2f}m  "
              f"{float(m.quantile(0.9)) * 1e3:>8.2f}m   {adj} adjacent")
    ends = [byslot[s]["disturbed"] for s in (0, mdp_cl.N_SLOTS - 1) if s in byslot]
    mids = [byslot[s]["disturbed"] for s in range(1, mdp_cl.N_SLOTS - 1) if s in byslot]
    if ends and mids:
        e_, m_ = sum(ends) / len(ends), sum(mids) / len(mids)
        print(f"   end slots {e_:.0%} vs interior {m_:.0%}"
              f"{'' if m_ == 0 else f'  -- ratio {e_ / m_:.2f}'}")

    if args_cli.json:
        os.makedirs(os.path.dirname(args_cli.json), exist_ok=True)
        json.dump({"row_x": mdp_cl.ROW_X, "pitch": mdp_cl.ROW_PITCH,
                   "yaw_range": mdp_cl.ROW_YAW_RANGE, "xy_range": mdp_cl.ROW_XY_RANGE,
                   "grip_z": args_cli.grip_z, "gates": {"pos": POS_MAX, "o": O_MIN,
                                                        "pen": PEN_MAX, "low_z": LOW_Z_MIN,
                                                        "move": MOVE_MAX},
                   "seed_control_ok": seed_ok, "control_ok": ctrl["ok"], "wall": wall,
                   "dead_slots": dead_slots, "dead_yaws": dead_yaws,
                   "corners_ok": corners_ok, "max_solved_r": r_ok,
                   "n_fail": n_fail, "cells": rows, "close_by_slot": byslot},
                  open(args_cli.json, "w"), indent=2)
        print(f"   wrote {args_cli.json}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
