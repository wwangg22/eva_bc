# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end scripted expert for ``Rebot-PrecisionSlot-*``: grasp, transport, insert.

The trajectory itself lives in ``slot/expert/plan.py`` -- this file is the *measurement*
harness around it. Design, and why each choice was forced by a measurement:

**Insertion is a horizontal drag, not a top-down drop.** ``strategy_probe.py`` measured both:
a Cartesian-straight +x drag at TCP z = 0.090 scores **100 % seated**, while lowering the
block in from above and releasing scores 46.9 % -- not because of precision (success was flat
across every lateral and yaw bin) but because the 15 mm release drop costs ~3 mm of depth
against a 40 mm threshold. The drag also needs 30 mm less reach.

**The path must be straight in Cartesian space.** The original Stage 2b harness interpolated
in *joint* space and scored 8-12 %; the block climbed onto the wall tops (z rose from 55 to
69 mm) because a joint-linear path bows. Same geometry, same grip, straight path: 100 %.

**Grip height is pinned to a 2 mm window.** A gripper *holding the block* only clears the slot
walls at TCP z >= 0.090 (measured wall clearance +0.4 mm at 0.090, -3.5 mm at 0.084, and the
physics agrees: finger gap 30.0 mm vs a jammed 35.4 mm). The block sits at centre z = 0.055
when seated, so the TCP must be >= 35 mm above the block centre -- and the block's half-height
*is* 35 mm.

**Approach the slot from outside its mouth.** The block spawns at x in [0.200, 0.240], which
overlaps the slot's own x range [0.210, 0.280]. The expert retracts to ``stage_x`` first, then
traverses in y, then drives +x. The carried block rides ~5 mm *ahead* of the TCP in x, which is
why ``stage_x`` is 0.165 and not 0.180 -- at 0.180 the block's nose reached the wall faces.

.. code-block:: bash

    python slot/scripts/run_expert.py --task Rebot-PrecisionSlot-v0 --num_envs 128
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted grasp-transport-insert expert.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Loose-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--cem_iters", type=int, default=120)
parser.add_argument("--grasp_h", type=float, default=0.031, help="TCP height above block centre [m]")
parser.add_argument("--carry_z", type=float, default=0.095, help="TCP z while carrying/inserting [m]")
parser.add_argument("--stage_x", type=float, default=0.165, help="x to retract to before traversing [m]")
parser.add_argument("--insert_x", type=float, default=0.2545, help="target block CENTRE x [m]")
parser.add_argument("--turn_per_wp", type=int, default=3)
parser.add_argument("--no_retreat", action="store_true", help="skip backing the gripper out after release")
parser.add_argument("--trace", type=str, default="", help="phase to trace per-waypoint")
parser.add_argument("--slot_dx", type=float, default=0.0,
                    help="shift the slot along x by this many metres and move insert_x with it. "
                         "This is the EXPERT CONTROL for EXP_ROBUSTNESS: the learned policy is "
                         "flat from 0 to +10 mm (0.979 / 0.990 / 1.000) but loses ~10 points at "
                         "-10 mm (0.844) and +20 mm (0.885). An open-loop planner that is TOLD "
                         "the new slot position separates 'the ROBOT is worse at those slot "
                         "positions' from 'the POLICY is'. Only x: plan.py hardcodes y = 0 for the "
                         "align and insert waypoints (:203-204), so a dy control would need "
                         "real surgery rather than a flag.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--seed_file", type=str, default="logs/expert/seed_q.json",
                    help="cached CEM seed; written if absent, loaded if present")
parser.add_argument("--video", action="store_true", help="render an mp4 of the rollout")
parser.add_argument("--fps", type=int, default=25)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    # nothing renders without this, and it must be set before the app launches
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge import mdp
from reBot_RL.tasks.manager_based.lift.camera_cfg import WORKSPACE_CAM_CFG

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slot.expert import plan as P  # noqa: E402
from slot.expert.ik import ArmIK  # noqa: E402


def main() -> None:
    torch.manual_seed(args_cli.seed)
    insert_x = args_cli.insert_x
    if args_cli.slot_dx:
        # Same patch as slot_act/eval_act.py, for the same reason and with the same trap: the
        # env cfg reads challenge.mdp.SLOT_CENTER to place the boxes while insertion_depth reads
        # challenge.mdp.common.SLOT_CENTER, so patching one moves the walls and the other moves
        # the score. plan.py holds the same `challenge.mdp` module object, so it follows along.
        # Must happen before parse_env_cfg, which builds the fixture geometry.
        from reBot_RL.tasks.manager_based.challenge.mdp import common as _common

        old = tuple(_common.SLOT_CENTER)
        new = (old[0] + args_cli.slot_dx, old[1])
        for _m in (_common, mdp):
            if hasattr(_m, "SLOT_CENTER"):
                _m.SLOT_CENTER = new
        for _m, _n in ((_common, "mdp.common"), (mdp, "mdp")):
            assert getattr(_m, "SLOT_CENTER", None) == new, f"SLOT_CENTER patch missed {_n}"
        insert_x += args_cli.slot_dx   # keep the push target at the same slot-relative depth
        print(f"[expert] SLOT SHIFTED {old} -> {new}; insert_x {args_cli.insert_x:.4f} -> "
              f"{insert_x:.4f}")
        # write the EFFECTIVE value back, because the results JSON serialises args_cli: without
        # this every shifted run records insert_x = 0.2545 and four different runs look
        # byte-identical in the one field a reader would use to tell them apart. (slot_dx is
        # recorded either way, so the four runs collected before this line was added are
        # distinguishable and correct -- their logs show the right insert_x -- but their JSON
        # insert_x field is the pre-shift value and must not be quoted.)
        args_cli.insert_x = insert_x
    params = P.ExpertParams(grasp_h=args_cli.grasp_h, carry_z=args_cli.carry_z,
                            stage_x=args_cli.stage_x, insert_x=insert_x,
                            turn_per_wp=args_cli.turn_per_wp, cem_iters=args_cli.cem_iters,
                            retreat=not args_cli.no_retreat, seed_file=args_cli.seed_file)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.video:
        env_cfg.scene.workspace_cam = WORKSPACE_CAM_CFG.replace(width=960, height=540)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    block = e.scene["block"]
    n = e.num_envs
    ik = ArmIK(env)
    P.check_geometry(params)

    seat_z = mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2]
    report = {"task": args_cli.task, "n": n, "args": vars(args_cli)}

    resets = torch.zeros(n, dtype=torch.bool, device=dev)
    nsteps = 0
    frames: list = []
    cam = None
    if args_cli.video:
        import numpy as np  # noqa: PLC0415

        cam = e.scene["workspace_cam"]
        # The repo camera pose was framed for the lift scene and puts the slot out of shot,
        # so re-aim it. The cfg pose is env-relative but the look-at API takes world
        # coordinates, hence adding the env origin back in.
        eye = torch.tensor([0.62, -0.42, 0.34], device=dev).unsqueeze(0) + e.scene.env_origins
        tgt = torch.tensor([0.235, -0.045, 0.055], device=dev).unsqueeze(0) + e.scene.env_origins
        cam.set_world_poses_from_view(eye, tgt)

    def grab() -> None:
        out = cam.data.output["rgb"]
        out = getattr(out, "torch", out)
        if callable(out):
            out = out()
        tiles = [out[i, ..., :3].detach().cpu().numpy().astype("uint8") for i in range(min(4, n))]
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        frames.append(np.concatenate([np.concatenate(tiles[0:2], axis=1),
                                      np.concatenate(tiles[2:4], axis=1)], axis=0))

    def step(a: torch.Tensor) -> None:
        nonlocal resets, nsteps
        nsteps += 1
        _, _, term, trunc, _ = env.step(a)
        resets |= torch.as_tensor(term, device=dev).bool() | torch.as_tensor(trunc, device=dev).bool()
        if cam is not None and nsteps % 2 == 0:
            grab()

    def block_pose() -> tuple[torch.Tensor, torch.Tensor]:
        p = torch.as_tensor(block.data.root_pos_w.torch, device=dev) - e.scene.env_origins
        q = torch.as_tensor(block.data.root_quat_w.torch, device=dev)
        return p, mdp.yaw_of(q)

    def seated() -> torch.Tensor:
        """``mdp.is_inserted`` bounds block z only from BELOW, so a block resting on the wall
        tops or dangling in a closed gripper passes it. Require the seated height too."""
        p, _ = block_pose()
        return mdp.is_inserted(e) & ((p[:, 2] - seat_z).abs() < 0.006)

    with torch.inference_mode():
        env.reset()
        seed = P.solve_seed(env, ik, params)

        # -------------------------------------------------------------- plan (block untouched)
        bp0, byaw0 = block_pose()
        print(f"  spawn: x {float(bp0[:, 0].min()):.3f}-{float(bp0[:, 0].max()):.3f}, "
              f"y {float(bp0[:, 1].min()):.3f}-{float(bp0[:, 1].max()):.3f}, "
              f"|yaw| <= {float(byaw0.abs().max()):.3f} rad")
        out = P.plan(ik, params, bp0, byaw0, seed["q_seed"], seed["axis_slot"], seed["sign"])
        plans, segs = out["plans"], out["segs"]

        print(f"  {'phase':>8} {'wps':>5} {'max pos err':>12} {'max axis err':>13} {'converged':>10}")
        for k, pl in plans.items():
            # Peak per-waypoint joint step. The IK can converge perfectly at every waypoint and
            # still produce a trajectory that jerks: near a singularity a 3 mm Cartesian step
            # costs a large joint step, and the block slips in the pads.
            dq = (pl["q"][:, 1:, :] - pl["q"][:, :-1, :]).abs()
            jmax = int(dq.amax(dim=(0, 1)).argmax()) if dq.numel() else 0
            frac = float(((pl["pos_err"] < ik.pos_tol) & (pl["axis_err"] < ik.rot_tol)).float().mean())
            report.setdefault("jerk", {})[k] = {"max_dq_rad": float(dq.max()) if dq.numel() else 0.0,
                                                "worst_joint": jmax + 1}
            print(f"  {k:>8} {pl['q'].shape[1]:5d} {float(pl['pos_err'].max()) * 1000:11.3f}m "
                  f"{float(pl['axis_err'].max()):13.6f} {int(pl['converged'].sum()):6d}/{n} "
                  f"(waypoints ok {frac * 100:.1f}%, max |dq| "
                  f"{float(dq.max()) if dq.numel() else 0.0:.4f} rad on joint{jmax + 1})")
        report["plan"] = {k: {"waypoints": int(pl["q"].shape[1]),
                              "max_pos_err_mm": float(pl["pos_err"].max()) * 1000,
                              "max_axis_err": float(pl["axis_err"].max()),
                              "converged": int(pl["converged"].sum())} for k, pl in plans.items()}
        # the retreat runs with the block already released, so it cannot fail the episode
        plan_ok = torch.stack([plans[k]["converged"] for k in P.PHASES], dim=1).all(dim=1)

        # ------------------------------------------------------------------------- execute
        env.reset()
        resets = torch.zeros(n, dtype=torch.bool, device=dev)
        # the reset re-randomises the block, so restore the pose the plan was built for
        block.write_root_state_to_sim(torch.cat([
            bp0 + e.scene.env_origins,
            torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev),
                         torch.sin(byaw0 / 2), torch.cos(byaw0 / 2)], dim=1),
            torch.zeros((n, 6), device=dev)], dim=1))

        stats: dict = {}
        steps = list(P.action_stream(plans, params))
        traced_wp = {i for i, s in enumerate(steps)
                     if s.phase == args_cli.trace and s.wp >= 0
                     and (i + 1 == len(steps) or steps[i + 1].wp != s.wp)}
        if traced_wp:
            print(f"\n  TRACE {args_cli.trace}: {'wp':>4} {'cmd y':>8} {'tcp y':>8} {'axis err':>9} "
                  f"{'blk yaw':>8} {'slip':>8} {'gap':>7}")

        for i, s in enumerate(steps):
            step(ik.action(s.q, close=s.close))
            nxt = steps[i + 1].phase if i + 1 < len(steps) else None
            if i in traced_wp and (s.wp % 6 == 0 or nxt != s.phase):
                # Separate two indistinguishable stories for a grip loss in this segment:
                #   * in-hand slip -- the gripper holds its axis, the BLOCK rotates inside the
                #     pads, so block yaw diverges from the finger axis and the gap opens;
                #   * tracking lag -- the wrist itself lags its commanded axis under load and
                #     carries the block with it, so block yaw tracks the axis, gap stays 30 mm.
                tcp, ax, _ = ik.tcp_now()
                _, byaw = block_pose()
                ax_ang = torch.atan2(ax[:, 0], ax[:, 1].abs().clamp(min=1e-9))
                print(f"  {'':>13}{s.wp:4d} {float(segs[s.phase][s.wp][:, 1].mean()) * 1000:7.2f}m "
                      f"{float(tcp[:, 1].mean()) * 1000:7.2f}m "
                      f"{float(1.0 - (ax * seed['axis_slot']).sum(1).abs().mean()):9.6f} "
                      f"{float(byaw.mean()):8.4f} {float((byaw + ax_ang).abs().mean()):8.4f} "
                      f"{float(ik.finger_gap_mm().mean()):6.2f}m")
            if nxt == s.phase:
                continue

            # ------ phase just ended: record what the arm and the block actually did
            gap = ik.finger_gap_mm()
            held = (gap > 26.0) & (gap < 34.0)
            bp, byaw = block_pose()
            tcp, _, _ = ik.tcp_now()
            if s.phase == "close":
                stats["grasp"] = {
                    "gap_mm_mean": float(gap.mean()), "gap_mm_p10": float(torch.quantile(gap, 0.1)),
                    "grasped": int(held.sum()),
                    "dx_mm": float((bp[:, 0] - tcp[:, 0]).abs().mean()) * 1000,
                    "dy_mm": float((bp[:, 1] - tcp[:, 1]).abs().mean()) * 1000,
                    "dz_mm": float((tcp[:, 2] - bp[:, 2]).mean()) * 1000,
                    "dyaw": float((byaw - byaw0).abs().mean())}
                print(f"\n  GRASP: gap {gap.mean():.2f} mm (p10 "
                      f"{float(torch.quantile(gap, 0.1)):.2f}), grasped {int(held.sum())}/{n}, "
                      f"in-hand offset dx {stats['grasp']['dx_mm']:.2f} "
                      f"dy {stats['grasp']['dy_mm']:.2f} mm, TCP is "
                      f"{stats['grasp']['dz_mm']:.1f} mm above the block centre")
            elif s.phase in P.PHASES or s.phase in ("release", "retreat"):
                stats[s.phase] = {"gap_mm": float(gap.mean()), "still_held": int(held.sum()),
                                  "blk_x_mm": float(bp[:, 0].mean()) * 1000,
                                  "blk_y_mm": float(bp[:, 1].mean()) * 1000,
                                  "blk_z_mm": float(bp[:, 2].mean()) * 1000,
                                  "yaw_mean": float(byaw.abs().mean()),
                                  "lag_mm": float((tcp[:, 0] - bp[:, 0]).mean()) * 1000,
                                  "seated": int((seated() & ~resets).sum())}
                print(f"  {s.phase.upper():>8}: block ({stats[s.phase]['blk_x_mm']:.1f}, "
                      f"{stats[s.phase]['blk_y_mm']:.1f}, {stats[s.phase]['blk_z_mm']:.1f}) mm, "
                      f"|yaw| {stats[s.phase]['yaw_mean']:.4f}, gap {stats[s.phase]['gap_mm']:.2f} mm, "
                      f"held {stats[s.phase]['still_held']}/{n}, "
                      f"seated {stats[s.phase]['seated']}/{n}")
            if s.phase == "push":
                stats["raw_gripped"] = int(mdp.is_inserted(e).sum())

        ok = seated() & ~resets
        raw = mdp.is_inserted(e)
        gap = ik.finger_gap_mm()
        bpf, byawf = block_pose()
        depth, lat = mdp.insertion_depth(e), mdp.lateral_error(e)

        print("\n" + "=" * 78)
        print(f"  RESULT  task={args_cli.task}  n={n}")
        print("=" * 78)
        print(f"  seated success        {int(ok.sum())}/{n} = {float(ok.float().mean()) * 100:.1f}%")
        print(f"  env predicate (raw)   {int(raw.sum())}/{n} gripped {stats.get('raw_gripped', -1)}/{n}")
        print(f"  depth   mean {float(depth.mean()) * 1000:.1f} mm  p10 "
              f"{float(torch.quantile(depth, 0.1)) * 1000:.1f}  (need >= 40.0)")
        print(f"  lateral mean {float(lat.mean()) * 1000:.2f} mm  p90 "
              f"{float(torch.quantile(lat, 0.9)) * 1000:.2f}")
        print(f"  |yaw|   mean {float(byawf.abs().mean()):.4f} rad  p90 "
              f"{float(torch.quantile(byawf.abs(), 0.9)):.4f}  (need <= 0.12)")
        print(f"  block z mean {float(bpf[:, 2].mean()) * 1000:.1f} mm (seat 55.0)")
        budget = int(e.max_episode_length)
        print(f"  plan converged {int(plan_ok.sum())}/{n}, envs reset mid-episode {int(resets.sum())}")
        print(f"  trajectory used {nsteps}/{budget} env steps"
              + ("  <-- OVER BUDGET, every env times out" if nsteps >= budget else ""))
        # Attribute every failure to the first predicate it breaks.
        fail_depth = int(((depth < mdp.SUCCESS_DEPTH) & ~ok).sum())
        fail_yaw = int(((byawf.abs() > mdp.SUCCESS_YAW) & (depth >= mdp.SUCCESS_DEPTH) & ~ok).sum())
        fail_seat = int((((bpf[:, 2] - seat_z).abs() >= 0.006) & (depth >= mdp.SUCCESS_DEPTH)
                         & (byawf.abs() <= mdp.SUCCESS_YAW) & ~ok).sum())
        fail_drop = int((~((gap > 26.0) & (gap < 34.0)) & ~ok).sum())
        print(f"  failures: too shallow {fail_depth}, yawed {fail_yaw}, not seated {fail_seat}, "
              f"lost grip before release {fail_drop}")
        print("=" * 78)
        stats["result"] = {"seated": int(ok.sum()), "rate": float(ok.float().mean()),
                           "raw": int(raw.sum()), "depth_mm": float(depth.mean()) * 1000,
                           "lat_mm": float(lat.mean()) * 1000, "yaw": float(byawf.abs().mean()),
                           "plan_converged": int(plan_ok.sum()), "resets": int(resets.sum()),
                           "fail_depth": fail_depth, "fail_yaw": fail_yaw,
                           "fail_seat": fail_seat, "fail_grip": fail_drop,
                           "steps_used": nsteps, "step_budget": budget}
        report["stats"] = stats

    out_dir = Path(args_cli.out_dir or Path(__file__).resolve().parents[1] / "logs" / "expert")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"expert_{args_cli.task}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[expert] wrote {path}")
    if frames:
        import imageio.v2 as imageio  # noqa: PLC0415

        vpath = out_dir / f"expert_{args_cli.task}.mp4"
        imageio.mimsave(vpath, frames, fps=args_cli.fps, macro_block_size=1)
        print(f"[expert] wrote {vpath}  ({len(frames)} frames, "
              f"{len(frames) / args_cli.fps:.1f} s, {frames[0].shape[1]}x{frames[0].shape[0]})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
