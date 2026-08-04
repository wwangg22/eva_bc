# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Pick the expert's insertion strategy by measurement, and price its grasp-accuracy budget.

Pre-registration (written before running)
-----------------------------------------
**What we know.** ``insertion_feasibility.py`` established that a gripper *holding the block*
only clears the slot walls at TCP ``z >= 0.090`` (wall clearance +0.4 mm at 0.090, -3.5 mm at
0.084, and the physics agrees exactly: finger gap 30.0 mm vs a jammed 35.4 mm). At 0.090 and
0.096 the block is genuinely held and the TCP reaches its commanded 45 mm depth to under
1 mm -- **but the block stops at 29.5 mm**, i.e. the pads slide ~15 mm forward along the
block. Insert rate 8-12 %.

**Two candidate explanations for the 15 mm loss**, which Part A separates:

* *slip* -- the grip cannot transmit the push force, so the block lags from the first
  millimetre and the gap ``tcp_x - block_x`` grows smoothly;
* *jam* -- the block tracks the TCP faithfully and then stops dead at a particular depth,
  after which the gap grows linearly at exactly the commanded rate.

The Stage 2b harness also drove the arm by **joint-space** interpolation, which does not move
the TCP in a straight line; a bowed path drags the block into a wall. Part A removes that
confound by driving dense Cartesian waypoints through the new DLS IK.

**Part B tests a different strategy entirely.** The env write-up assumed the block must be
pushed in horizontally, and warned that "the arm cannot release from above". That is true only
if you insist on releasing with the block *seated*. It does not have to be: the slot walls are
30 mm tall, so a block whose bottom is even a few mm below the wall tops is already captured
laterally and can simply be **dropped** the rest of the way. Releasing at TCP z = 0.100 with a
30 mm grip height puts the block's bottom 18 mm inside the channel while the finger tips are
still 10 mm above the wall tops. That converts a 39 mm friction-limited drag into a short
guided descent.

**Beliefs, so they can be wrong on the record.** I expect Part A to show *jam*, not slip
(2000 N/m finger stiffness x ~15.6 mm deflection is ~31 N per pad, far more than a 0.04 kg
block sliding on a floor can resist). I expect Part B to beat 8-12 % substantially, and I
expect its binding constraint to be **lateral** grasp error, since the block is 30 mm wide in
a 33 mm channel: 1.5 mm per side is the entire budget.

**Decision rule.** If Part B clears 60 % at realistic grasp scatter it becomes the expert's
insertion primitive and Part A's drag is abandoned. If neither clears 40 %, the task needs a
compliance or search behaviour (wiggle-on-contact), not a better open-loop plan.

Part B deliberately samples grasp error *continuously* and bins the outcome, so one run yields
the whole tolerance surface rather than a single number. That surface is the specification the
grasp stage has to meet.

.. code-block:: bash

    python slot/analysis/strategy_probe.py --num_envs 128
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare horizontal-drag vs vertical lower-in insertion.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--cem_iters", type=int, default=120)
parser.add_argument("--insert_x", type=float, default=0.253, help="target block CENTRE x [m]")
parser.add_argument("--grip_hs", type=float, nargs="+", default=[0.025, 0.033],
                    help="grip height above the block centre [m]")
parser.add_argument("--release_zs", type=float, nargs="+", default=[0.092, 0.100, 0.108],
                    help="TCP z at which the fingers open [m]")
parser.add_argument("--y_scatter", type=float, default=0.003, help="uniform +/- block y error [m]")
parser.add_argument("--yaw_scatter", type=float, default=0.06, help="uniform +/- block yaw error [rad]")
parser.add_argument("--x_scatter", type=float, default=0.002, help="uniform +/- block x error [m]")
parser.add_argument("--skip_a", action="store_true")
parser.add_argument("--out_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge import mdp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slot.expert.ik import ArmIK  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    block = e.scene["block"]
    n = e.num_envs
    ik = ArmIK(env)

    mouth_x = mdp.SLOT_CENTER[0] - mdp.SLOT_DEPTH / 2                 # 0.210
    wall_top = mdp.SLOT_FLOOR_Z + mdp.WALL_HEIGHT                     # 0.050
    seat_z = mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2]                     # 0.055
    max_x = mdp.SLOT_CENTER[0] + mdp.SLOT_DEPTH / 2 - mdp.BLOCK_HALF[0]   # 0.2575
    half_w = getattr(e.cfg, "slot_half_width", mdp.BLOCK_HALF[1] + 0.0015)
    if args_cli.insert_x > max_x:
        raise SystemExit(f"--insert_x {args_cli.insert_x} > {max_x:.4f} (nose through the back stop)")

    report = {"task": args_cli.task, "num_envs": n, "insert_x": args_cli.insert_x,
              "half_width_mm": half_w * 1000, "wall_top": wall_top, "seat_z": seat_z}

    print("\n" + "=" * 78)
    print(f"SETUP  task={args_cli.task}  n={n}  slot half-width={half_w * 1000:.2f} mm")
    print("=" * 78)
    print(f"  mouth x {mouth_x:.3f} | seated block centre z {seat_z:.3f} | wall tops {wall_top:.3f}")
    print(f"  success needs centre x >= {mouth_x + mdp.SUCCESS_DEPTH:.3f}, "
          f"max {max_x:.4f}; targeting {args_cli.insert_x:.3f}")

    # The episode is 600 steps and the env auto-resets on time_out, block_dropped and
    # block_toppled. A reset mid-phase teleports the block back to the table and silently
    # turns the rest of that phase into noise, so every step is watched and every phase
    # starts from a fresh episode budget. (Stage 2b in insertion_feasibility.py ran ~950
    # steps without this guard -- another reason not to trust its absolute numbers.)
    resets = torch.zeros(n, dtype=torch.bool, device=dev)

    def step(a: torch.Tensor) -> None:
        _, _, term, trunc, _ = env.step(a)
        nonlocal resets
        resets |= torch.as_tensor(term, device=dev).bool() | torch.as_tensor(trunc, device=dev).bool()

    def new_phase() -> None:
        nonlocal resets
        env.reset()
        resets = torch.zeros(n, dtype=torch.bool, device=dev)

    def run(q_traj: torch.Tensor, close: bool, per_wp: int = 4) -> None:
        """Execute a (n, T, 6) joint trajectory, `per_wp` env steps per waypoint."""
        for t in range(q_traj.shape[1]):
            for _ in range(per_wp):
                step(ik.action(q_traj[:, t, :], close=close))

    def hold(q_arm: torch.Tensor, close: bool, steps: int) -> None:
        for _ in range(steps):
            step(ik.action(q_arm, close=close))

    def park(q_arm: torch.Tensor, q_fing: float) -> None:
        q = ik.q_default.unsqueeze(0).repeat(n, 1)
        q[:, ik.arm_dof] = q_arm
        q[:, ik.fing_dof] = q_fing
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)

    def block_state(pos: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        quat = torch.zeros((n, 4), device=dev)                        # XYZW, measured
        quat[:, 2], quat[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        return torch.cat([pos + e.scene.env_origins, quat, torch.zeros((n, 6), device=dev)], dim=1)

    def block_now() -> torch.Tensor:
        return torch.as_tensor(block.data.root_pos_w.torch, device=dev) - e.scene.env_origins

    def seated() -> torch.Tensor:
        """Honest success: the env predicate AND the block actually at its seated height.

        ``mdp.is_inserted`` checks z only as a LOWER bound (``z > SLOT_FLOOR_Z - 0.005``, to
        catch a block that fell off the table), with no upper bound. So a block resting on
        top of the 30 mm walls, or still dangling in a closed gripper above the slot, passes
        it. Measured in the first run of this probe: a cell scored 93.8 % on ``is_inserted``
        with 13.28 mm of mean lateral error -- geometrically impossible *inside* a channel of
        16.5 mm half-width for a block of 15 mm half-width. Every headline number here is the
        seated one; the raw predicate is reported alongside so the gap is visible.
        """
        return mdp.is_inserted(e) & ((block_now()[:, 2] - seat_z).abs() < 0.006)

    def block_pitch() -> torch.Tensor:
        """Tilt of the block's own +z axis away from world +z [rad] -- catches the block
        rotating out of the fingers, which yaw alone cannot see."""
        q = torch.as_tensor(block.data.root_quat_w.torch, device=dev)
        up = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        return torch.acos(up[:, 2].clamp(-1.0, 1.0))

    with torch.inference_mode():
        env.reset()

        # ------------------------------------------------------------------- Part 0: setup
        # ALL kinematic search happens here, before any block is placed. write_joint_state_to_sim
        # teleports the arm and re-opens the fingers, so a search run after a grasp destroys it.
        print("\n" + "=" * 78)
        print("PART 0 -- Jacobian calibration and the canonical grasp orientation")
        print("=" * 78)
        seed = ik.q_default[ik.arm_dof].clone().unsqueeze(0).repeat(n, 1)
        print(f"  jacobian: {ik.calibrate_jacobian(seed)}")

        # CEM once, only to choose the elbow branch / wrist roll: position plus a
        # finger-axis-along-y cost. Its solution is then FROZEN as the target orientation for
        # every DLS solve, which is what makes the whole plan branch-continuous.
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=dev)
        nominal = torch.tensor([args_cli.insert_x, 0.0, 0.100], device=dev)
        mean, std = ik.q_default[ik.arm_dof].clone(), torch.full((6,), 0.40, device=dev)
        best = {"cost": 1e9}
        for _ in range(args_cli.cem_iters):
            q = (mean + std * torch.randn((n, 6), device=dev)).clamp(ik.lo, ik.hi)
            tcp, sep, _ = ik.fk(q, 0.016)
            pos_err = (tcp - nominal).norm(dim=1)
            align = 1.0 - (sep @ y_axis).abs()
            cost = pos_err + 0.25 * align
            elite = q[cost.argsort()[: max(8, n // 20)]]
            mean, std = elite.mean(0), elite.std(0).clamp(min=0.008)
            i = int(cost.argmin())
            if float(cost[i]) < best["cost"]:
                best = {"cost": float(cost[i]), "q": q[i].clone(),
                        "pos_err": float(pos_err[i]), "align": float(align[i])}
        q_nom = best["q"].unsqueeze(0).repeat(n, 1)
        _, sep, _ = ik.fk(q_nom, 0.016)
        print(f"  CEM seed: pos err {best['pos_err'] * 1000:.2f} mm, "
              f"finger-axis misalignment {best['align']:.5f}")
        print(f"  finger axis {[round(float(v), 4) for v in sep[0]]} (want +/-y)")
        # The target is the axis itself, exactly -- no orientation snapping and no rigid wrist
        # roll. CEM's only remaining job is to hand DLS the right elbow branch as a seed.
        sign = torch.sign((sep @ y_axis).mean()).clamp(min=-1.0, max=1.0)
        axis_tgt = (sign * y_axis).expand(n, 3).contiguous()
        chk = ik.solve(nominal.expand(n, 3), axis_tgt, q_nom, 0.016, iters=200)
        q_ref = chk["q"]
        _, sep2, _ = ik.fk(q_ref, 0.016)
        print(f"  DLS at the nominal pose: pos err {float(chk['pos_err'].mean()) * 1000:.3f} mm "
              f"(max {float(chk['pos_err'].max()) * 1000:.3f}), axis err "
              f"{float(chk['axis_err'].mean()):.6f}, converged {int(chk['converged'].sum())}/{n}")
        print(f"  finger axis achieved {[round(float(v), 5) for v in sep2[0]]}")
        report["setup"] = {"jacobian": ik.jac_mode, "cem_align": best["align"],
                           "dls_pos_err_mm": float(chk["pos_err"].mean()) * 1000,
                           "dls_axis_err": float(chk["axis_err"].mean()),
                           "dls_converged": int(chk["converged"].sum())}
        if not bool(chk["converged"].all()):
            print("  WARNING: DLS did not converge at the nominal pose -- plans below are suspect.")
        # Seed all later solves from a converged pose on the target axis.
        seed = q_ref.clone()

        # ------------------------------------------- Part A: trace the horizontal drag
        if not args_cli.skip_a:
            print("\n" + "=" * 78)
            print("PART A -- horizontal drag, traced per waypoint (Cartesian-straight path)")
            print("=" * 78)
            gz = 0.090
            xs = torch.arange(mouth_x + 0.006, args_cli.insert_x + 1e-6, 0.001, device=dev)
            wps = [torch.stack([x.expand(n), torch.zeros(n, device=dev),
                                torch.full((n,), gz, device=dev)], dim=1) for x in xs]
            plan = ik.solve_path(wps, axis_tgt, seed, q_fing=0.016)
            print(f"  {len(wps)} waypoints, 1 mm apart; IK converged in "
                  f"{int(plan['converged'].sum())}/{n} envs, "
                  f"max pos err {float(plan['pos_err'].max()) * 1000:.2f} mm")

            new_phase()
            park(plan["q"][:, 0, :], 0.045)
            tcp0, _, _ = ik.tcp_now()
            bpos = torch.stack([xs[0].expand(n), tcp0[:, 1], torch.full((n,), seat_z + 0.002, device=dev)], dim=1)
            block.write_root_state_to_sim(block_state(bpos, torch.zeros(n, device=dev)))
            hold(plan["q"][:, 0, :], close=True, steps=40)
            print(f"  after close: gap {float(ik.finger_gap_mm().mean()):.2f} mm "
                  f"(30 mm block reads 30 mm)")
            print(f"  {'wp':>4} {'tcp x':>8} {'blk x':>8} {'lag':>8} {'blk y':>8} {'blk z':>8} "
                  f"{'pitch':>7} {'yaw':>7} {'gap':>7}")
            trace = []
            for t in range(plan["q"].shape[1]):
                for _ in range(4):
                    step(ik.action(plan["q"][:, t, :], close=True))
                tcp, _, _ = ik.tcp_now()
                b = block_now()
                row = {"wp": t, "tcp_x": float(tcp[:, 0].mean()), "blk_x": float(b[:, 0].mean()),
                       "lag_mm": float((tcp[:, 0] - b[:, 0]).mean()) * 1000,
                       "blk_y_mm": float(b[:, 1].mean()) * 1000, "blk_z_mm": float(b[:, 2].mean()) * 1000,
                       "pitch": float(block_pitch().mean()), "yaw": float(mdp.yaw_error(e).mean()),
                       "gap_mm": float(ik.finger_gap_mm().mean())}
                trace.append(row)
                if t % 4 == 0 or t == plan["q"].shape[1] - 1:
                    print(f"  {t:4d} {row['tcp_x']:8.4f} {row['blk_x']:8.4f} {row['lag_mm']:7.2f}m "
                          f"{row['blk_y_mm']:7.2f}m {row['blk_z_mm']:7.2f}m {row['pitch']:7.4f} "
                          f"{row['yaw']:7.4f} {row['gap_mm']:6.2f}m")
            raw_held = mdp.is_inserted(e) & ~resets
            hold(plan["q"][:, -1, :], close=False, steps=60)     # release, then judge honestly
            ok = seated() & ~resets
            raw = mdp.is_inserted(e) & ~resets
            print(f"  RESULT while still gripped: is_inserted {int(raw_held.sum())}/{n}")
            print(f"  RESULT after release: is_inserted {int(raw.sum())}/{n}, "
                  f"SEATED {int(ok.sum())}/{n} = {float(ok.float().mean()) * 100:.1f}%, "
                  f"depth {float(mdp.insertion_depth(e).mean()) * 1000:.1f} mm, "
                  f"block z {float(block_now()[:, 2].mean()) * 1000:.1f} mm (seat 55.0), "
                  f"resets {int(resets.sum())}")
            # The lag's SHAPE is the diagnosis: smooth growth from wp 0 = slip;
            # flat then linear = a jam, and the knee says where.
            lag = torch.tensor([r["lag_mm"] for r in trace])
            knee = int((lag > lag[0] + 2.0).nonzero()[0]) if bool((lag > lag[0] + 2.0).any()) else -1
            print(f"  lag at wp0 {float(lag[0]):.2f} mm -> final {float(lag[-1]):.2f} mm; "
                  f"first exceeds +2 mm at wp {knee} "
                  f"(block x {trace[knee]['blk_x']:.4f})" if knee >= 0 else "  lag never grew")
            report["part_a"] = {"trace": trace, "seated": int(ok.sum()),
                                "raw_gripped": int(raw_held.sum()), "raw_released": int(raw.sum()),
                                "rate": float(ok.float().mean()), "knee_wp": knee,
                                "resets": int(resets.sum())}

        # ------------------------------------------- Part B: vertical lower-in + drop
        print("\n" + "=" * 78)
        print("PART B -- vertical lower-in, release above the seat, let the walls guide it")
        print("=" * 78)
        print(f"  grasp scatter: x +/-{args_cli.x_scatter * 1000:.1f} mm, "
              f"y +/-{args_cli.y_scatter * 1000:.1f} mm, yaw +/-{args_cli.yaw_scatter:.3f} rad")
        print(f"  {'h mm':>6} {'z_rel':>7} {'hover':>7} {'drop mm':>8} {'IK':>7} {'held':>9} "
              f"{'ins':>9} {'rate':>7} {'depth':>7} {'lat':>7} {'rst':>4}")
        cells = []
        for h in args_cli.grip_hs:
            for z_rel in args_cli.release_zs:
                # hover: block bottom 5 mm clear of the wall tops so lateral scatter cannot
                # catch a wall on the way in
                z_hov = wall_top + 0.005 + mdp.BLOCK_HALF[2] + h
                zs = torch.arange(z_hov, z_rel - 1e-6, -0.001, device=dev)
                wps = [torch.stack([torch.full((n,), args_cli.insert_x, device=dev),
                                    torch.zeros(n, device=dev), z.expand(n)], dim=1) for z in zs]
                plan = ik.solve_path(wps, axis_tgt, seed, q_fing=0.016)
                nconv = int(plan["converged"].sum())

                new_phase()
                park(plan["q"][:, 0, :], 0.045)
                tcp0, _, _ = ik.tcp_now()
                dx = (torch.rand(n, device=dev) - 0.5) * 2 * args_cli.x_scatter
                dy = (torch.rand(n, device=dev) - 0.5) * 2 * args_cli.y_scatter
                dyaw = (torch.rand(n, device=dev) - 0.5) * 2 * args_cli.yaw_scatter
                bpos = torch.stack([tcp0[:, 0] + dx, tcp0[:, 1] + dy, tcp0[:, 2] - h], dim=1)
                # Close the fingers with the block teleported each step: it is held in free
                # space here, so without this it free-falls before the pads arrive. The grasp
                # is then proven by 20 UNSUPPORTED steps and the finger gap, not by the
                # teleport -- a block that is not really gripped fails both.
                for _ in range(30):
                    block.write_root_state_to_sim(block_state(bpos, dyaw))
                    step(ik.action(plan["q"][:, 0, :], close=True))
                hold(plan["q"][:, 0, :], close=True, steps=20)
                gap = ik.finger_gap_mm()
                b = block_now()
                held = (gap > 26.0) & (gap < 34.0) & ((b[:, :2] - tcp0[:, :2]).norm(dim=1) < 0.02)

                run(plan["q"], close=True, per_wp=3)
                hold(plan["q"][:, -1, :], close=False, steps=60)     # open and let it drop
                ok = seated() & ~resets
                raw = mdp.is_inserted(e) & ~resets
                depth, lat = mdp.insertion_depth(e), mdp.lateral_error(e)
                rec = {"h": h, "z_rel": z_rel, "z_hover": float(z_hov),
                       "drop_mm": (z_rel - h - seat_z) * 1000, "resets": int(resets.sum()),
                       "ik_converged": nconv, "ik_err_mm": float(plan["pos_err"].max()) * 1000,
                       "tcp_y_off_mm": float(tcp0[:, 1].abs().mean()) * 1000,
                       "held": int(held.sum()), "seated": int(ok.sum()), "raw": int(raw.sum()),
                       "rate": float(ok.float().mean()), "depth_mm": float(depth.mean()) * 1000,
                       "lat_mm": float(lat.mean()) * 1000,
                       "blk_z_mm": float(block_now()[:, 2].mean()) * 1000,
                       "dy_mm": (dy * 1000).cpu().tolist(), "dyaw": dyaw.cpu().tolist(),
                       "ok": ok.cpu().tolist()}
                cells.append(rec)
                print(f"  {h * 1000:6.0f} {z_rel:7.3f} {z_hov:7.3f} {rec['drop_mm']:8.1f} "
                      f"{nconv:4d}/{n} {int(held.sum()):5d}/{n} {int(ok.sum()):5d}/{n} "
                      f"{rec['rate'] * 100:6.1f}% {rec['depth_mm']:6.1f}m {rec['lat_mm']:6.2f}m "
                      f"{int(resets.sum()):4d}  (raw {int(raw.sum())}, blk z "
                      f"{rec['blk_z_mm']:.1f}m, IK err {rec['ik_err_mm']:.2f}m)")
        report["part_b"] = cells

        # ------------------------------------------------- tolerance surface (the deliverable)
        good = max(cells, key=lambda c: c["rate"]) if cells else None
        if good:
            print("\n" + "=" * 78)
            print(f"TOLERANCE SURFACE for the best cell (h={good['h'] * 1000:.0f} mm, "
                  f"z_rel={good['z_rel']:.3f}) -- this is the grasp stage's spec")
            print("=" * 78)
            dy = torch.tensor(good["dy_mm"]).abs()
            dyaw = torch.tensor(good["dyaw"]).abs()
            ok = torch.tensor(good["ok"], dtype=torch.float32)
            print(f"  {'|dy| mm':>12} {'n':>5} {'success':>9}")
            edges = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
            bins_y = []
            for a, b_ in zip(edges[:-1], edges[1:]):
                m = (dy >= a) & (dy < b_)
                r = float(ok[m].mean()) if int(m.sum()) else float("nan")
                bins_y.append({"lo": a, "hi": b_, "n": int(m.sum()), "rate": r})
                print(f"  {a:5.1f}-{b_:<5.1f} {int(m.sum()):5d} {r * 100:8.1f}%")
            print(f"  {'|dyaw| rad':>12} {'n':>5} {'success':>9}")
            yedges = [0.0, 0.015, 0.03, 0.045, 0.06]
            bins_yaw = []
            for a, b_ in zip(yedges[:-1], yedges[1:]):
                m = (dyaw >= a) & (dyaw < b_)
                r = float(ok[m].mean()) if int(m.sum()) else float("nan")
                bins_yaw.append({"lo": a, "hi": b_, "n": int(m.sum()), "rate": r})
                print(f"  {a:5.3f}-{b_:<5.3f} {int(m.sum()):5d} {r * 100:8.1f}%")
            report["tolerance"] = {"cell": {k: good[k] for k in ("h", "z_rel", "rate")},
                                   "by_dy_mm": bins_y, "by_dyaw_rad": bins_yaw}

        print("\n" + "=" * 78)
        best_b = max((c["rate"] for c in cells), default=0.0)
        a_rate = report.get("part_a", {}).get("rate", float("nan"))
        print(f"  VERDICT: horizontal drag {a_rate * 100:.1f}%  vs  vertical lower-in "
              f"{best_b * 100:.1f}%  (Stage 2b baseline was 8.2-11.7%)")
        print("=" * 78)

    out = Path(args_cli.out_dir or Path(__file__).resolve().parents[1] / "logs" / "envelope")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"strategy_probe_{args_cli.task}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[probe] wrote {path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
