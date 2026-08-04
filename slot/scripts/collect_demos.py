# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

r"""Collect behaviour-cloning demonstrations from the scripted slot expert.

Writes eva_bc's HDF5 schema (``act/dataset.py`` lines 141-163) so the existing loader can read
them with only the 41 -> 34 observation-dimension edit.

Why noise injection is not optional
-----------------------------------
The expert plans its entire trajectory from the post-reset block pose and then executes
**open-loop**. Its nominal demonstrations therefore contain no corrective behaviour at all:
every state in them lies on one deterministic manifold indexed by the block spawn, and a
cloned policy that drifts a millimetre off that manifold has never seen anything like where it
now is. eva_bc's own postmortem attributes part of its BC plateau to exactly this.

The fix here is DART-style (Laskey et al. 2017) and exploits a property of *this* action
space rather than of the expert: the action is a **joint position target**, so the expert's
command at step ``t`` is the correct thing to do from *any* nearby state -- commanding ``q_t``
pulls the arm to ``q_t`` whether it started on the nominal path or 2 mm off it. So we

* perturb the action that is actually **executed**, and
* record the **nominal** action as the label.

That labels a tube of off-manifold states with correct recovery commands, at zero planning
cost and with no IK run after the grasp (which would teleport the arm and drop the block --
``expert/ik.py`` warns about this, and it once took a trial from 0 % to 100 %).

The noise is an Ornstein-Uhlenbeck walk, not white noise: at 50 Hz a position controller with
stiffness 2000 filters independent per-step noise into a state deviation far smaller than the
command deviation, so white noise would move the *labels* without moving the *states* -- the
opposite of what is wanted. It is also **decayed to zero during every settle/hold**, so each
phase boundary re-converges to nominal and deviations cannot compound across a 560-step
episode into a dropped block. The cost of that choice is bounded coverage, and it is what
makes the largest usable ``--noise_std`` large.

.. code-block:: bash

    python slot/scripts/collect_demos.py --task Rebot-PrecisionSlot-v0 --num_envs 128 \
        --rollouts 4 --noise_std 0.02 --out data/demos_v0_dart.hdf5
"""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect BC demos from the scripted slot expert.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--rollouts", type=int, default=4, help="batches of num_envs demos to collect")
parser.add_argument("--noise_std", type=float, default=0.0,
                    help="OU std on the EXECUTED arm action, in env action units (1.0 = 0.5 rad)")
parser.add_argument("--noise_rho", type=float, default=0.95, help="OU correlation, ~20-step timescale")
parser.add_argument("--noise_phases", type=str, default="reach,lift,back,spin,turn,push",
                    help="phases to inject noise into. DART's label-correctness argument holds "
                         "only in free space, so 'push' should normally be excluded -- see "
                         "docs/slot/EXP_NOISE_SWEEP.md.")
parser.add_argument("--hold_decay", type=float, default=0.7, help="per-step noise decay during holds")
parser.add_argument("--slip_tol_mm", type=float, default=0.0,
                    help="in-hand offset drift [mm] past which the expert's plan is stale and "
                         "every later frame is loss-censored. **Default 0 = off**: the raw "
                         "per-step signal is always stored as the `slip_mm` dataset, so the "
                         "threshold is calibrated against real failures offline "
                         "(scripts/calibrate_slip.py) instead of being guessed at collection "
                         "time. A guessed 3.0 mm censored 100%% of demos, successes included, "
                         "because nominal drift already reaches 4.7 mm by the end of push.")
parser.add_argument("--grasp_h", type=float, default=0.031)
parser.add_argument("--carry_z", type=float, default=0.095)
parser.add_argument("--stage_x", type=float, default=0.165)
parser.add_argument("--insert_x", type=float, default=0.2545)
parser.add_argument("--turn_per_wp", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None, help="output .hdf5 (default: auto-named)")
parser.add_argument("--dry_run", action="store_true", help="roll out and report, write nothing")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge import mdp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slot.expert import plan as P  # noqa: E402
from slot.expert.ik import ArmIK  # noqa: E402

OBS_DIM = 34
ACTION_DIM = 7
GRIP_LO, GRIP_HI = 26.0, 34.0   # finger gap [mm] that means "a 30 mm block is between the pads"

#: segment id -> the phase whose END decides its outcome. Ids follow eva_bc's
#: "<name>:<tag>:g<n>" convention where the dataset's nominal-pool filter parses ":g<n>".
SEGMENTS = [("reach:grasp:g0", "close"), ("lift", "lift"), ("back", "back"), ("spin", "spin"),
            ("turn", "turn"), ("push", "push"), ("release", "release")]


def slip_censor(slip_mm: np.ndarray, tol: float) -> int | None:
    """First frame at which the block has drifted ``tol`` mm from where the plan assumes it is.

    Why this is the right detector for THIS expert
    ----------------------------------------------
    The expert is **open-loop after the grasp**: it plans the whole trajectory from the
    post-reset block pose and never looks again. So if the block shifts in the pads, every
    later frame pairs an observation showing the *new* in-hand pose with an action computed
    for the *old* one. That is a wrong label, not a noisy one, and training on it teaches the
    policy to ignore in-hand error -- precisely the error we need it to correct.

    Measuring ``block_pos - TCP`` against its value at the end of the grasp catches both of
    this task's failure modes with one number:

    * the block **sliding** in the pads (the offset drifts), and
    * the block **jamming** on a slot wall while the gripper keeps advancing (the offset
      drifts just as much, because the gripper moves and the block does not).

    It is also **noise-agnostic**, which matters because half the pool is DART data: injected
    noise moves the arm and the block it is holding together, so the offset is unchanged. That
    keeps the censor from deleting exactly the corrective labels the DART pool exists to
    provide -- an injected deviation is a *correct* label, an expert's own uncorrected
    deviation is not.

    Returns the first offending frame index, or ``None``.
    """
    if tol <= 0.0:
        return None
    bad = np.nonzero(slip_mm > tol)[0]
    return int(bad[0]) if bad.size else None


def build_train_mask(T: int, segments: list, outcomes: dict) -> np.ndarray:
    """BC loss mask: 0 over failed-grasp / lost-transport segments.

    Ported unchanged from ``expert/run_expert_v1.py:267``. The boundary is failure
    **detection**, not failure: post-detection segments are recovery skill and stay trainable.
    This expert has no recovery, so a censored demo is censored to its end -- and is dropped
    anyway by ``dataset.default_demo_filter``, which keeps only successes. The machinery is
    here so that DAgger data collected later lands in the same format.
    """
    mask = np.ones(T, dtype=np.uint8)
    for k, s in enumerate(segments):
        oc = outcomes.get(s["seg"])
        if oc is None:
            continue
        if oc["outcome"] in ("missed", "lost"):
            end = segments[k + 1]["t"] if k + 1 < len(segments) else T
            mask[s["t"]:end] = 0
    return mask


def main() -> None:
    torch.manual_seed(args_cli.seed)
    params = P.ExpertParams(grasp_h=args_cli.grasp_h, carry_z=args_cli.carry_z,
                            stage_x=args_cli.stage_x, insert_x=args_cli.insert_x,
                            turn_per_wp=args_cli.turn_per_wp)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    block = e.scene["block"]
    robot = e.scene["robot"]
    n = e.num_envs
    ik = ArmIK(env)
    P.check_geometry(params)

    noise_phases = {p.strip() for p in args_cli.noise_phases.split(",") if p.strip()}
    unknown = noise_phases - set(P.PHASES)
    if unknown:
        raise SystemExit(f"--noise_phases has unknown phases {sorted(unknown)}; valid: {P.PHASES}")

    seat_z = mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2]
    # One short of the episode limit, so the recording never straddles an auto-reset.
    budget = int(e.max_episode_length) - 1

    def block_pose() -> tuple[torch.Tensor, torch.Tensor]:
        p = torch.as_tensor(block.data.root_pos_w.torch, device=dev) - e.scene.env_origins
        q = torch.as_tensor(block.data.root_quat_w.torch, device=dev)
        return p, mdp.yaw_of(q)

    def seated() -> torch.Tensor:
        p, _ = block_pose()
        return mdp.is_inserted(e) & ((p[:, 2] - seat_z).abs() < 0.006)

    def policy_obs(d) -> torch.Tensor:
        o = d["policy"]
        return torch.as_tensor(getattr(o, "torch", o), device=dev)

    demos: list[dict] = []
    summary = {"task": args_cli.task, "args": vars(args_cli), "rollouts": []}

    with torch.inference_mode():
        env.reset()
        seed = P.solve_seed(env, ik, params)

        for r in range(args_cli.rollouts):
            env.reset()
            bp0, byaw0 = block_pose()
            out = P.plan(ik, params, bp0, byaw0, seed["q_seed"], seed["axis_slot"], seed["sign"])
            plans = out["plans"]
            plan_ok = torch.stack([plans[k]["converged"] for k in P.PHASES], dim=1).all(dim=1)

            env.reset()
            # The reset re-randomises the block, so restore the pose the plan was built for.
            block.write_root_state_to_sim(torch.cat([
                bp0 + e.scene.env_origins,
                torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev),
                             torch.sin(byaw0 / 2), torch.cos(byaw0 / 2)], dim=1),
                torch.zeros((n, 6), device=dev)], dim=1))
            # Flush that write into the physics view and refresh the data buffers, so the FIRST
            # recorded observation already carries the restored block pose. Without this the
            # t=0 observation reports the re-randomised pose the plan was NOT built for --
            # a single mislabelled frame per demo, and exactly the kind of self-consistent
            # harness error that this project has been bitten by twice.
            e.sim.forward()
            robot.update(0.0)
            block.update(0.0)
            obs = policy_obs(e.observation_manager.compute())
            assert obs.shape == (n, OBS_DIM), f"expected (n,{OBS_DIM}) obs, got {tuple(obs.shape)}"

            steps = list(P.action_stream(plans, params, budget=budget))
            T = len(steps)
            ep_obs = torch.empty((T, n, OBS_DIM), device=dev)
            ep_act = torch.empty((T, n, ACTION_DIM), device=dev)
            # in-hand offset (block position relative to the TCP) at every step; the reference
            # is taken at the end of the grasp, and drift from it means the expert's open-loop
            # plan no longer describes the scene. See slip_censor().
            ep_inhand = torch.empty((T, n, 3), device=dev)
            inhand_ref: torch.Tensor | None = None
            close_end = 0
            resets = torch.zeros(n, dtype=torch.bool, device=dev)
            eps = torch.zeros((n, 6), device=dev)
            sigma_root = (1.0 - args_cli.noise_rho ** 2) ** 0.5 * args_cli.noise_std
            # Tracking deviation is measured PER PHASE, reset at each boundary. Measured
            # globally it is dominated by the warmup, where the arm starts at its default pose
            # and is commanded straight to the hover waypoint: 1262 mrad at zero noise, which
            # says nothing about how far the injected noise actually moved the arm.
            dev_phase = torch.zeros(n, device=dev)
            # The peak alone cannot tell "the noise moved the arm everywhere by a little"
            # (which is the state diversity we are paying for) from "the noise did nothing and
            # the peak is still the systematic tracking lag". The running mean can.
            dev_sum, dev_cnt = torch.zeros(n, device=dev), 0

            seg_t: dict[str, int] = {}
            held_prev = torch.zeros(n, dtype=torch.bool, device=dev)
            outcomes: list[dict] = [{} for _ in range(n)]
            phase_log: dict[str, dict] = {}

            for i, s in enumerate(steps):
                seg_t.setdefault(s.phase, i)
                a_nom = ik.action(s.q, close=s.close)
                if args_cli.noise_std > 0.0:
                    if s.wp >= 0 and s.phase in noise_phases:
                        eps = args_cli.noise_rho * eps + sigma_root * torch.randn((n, 6), device=dev)
                    else:
                        eps = eps * args_cli.hold_decay   # ring down, do not step-discontinue
                a_exec = a_nom.clone()
                a_exec[:, :6] += eps

                ep_obs[i] = obs
                ep_act[i] = a_nom
                obs_d, _, term, trunc, _ = env.step(a_exec)
                obs = policy_obs(obs_d)
                resets |= torch.as_tensor(term, device=dev).bool() | torch.as_tensor(trunc, device=dev).bool()

                # How far the noise actually pushed the ARM, not the command -- the quantity
                # that decides whether the tube of covered states is worth anything.
                dq = (torch.as_tensor(robot.data.joint_pos.torch, device=dev)[:, ik.arm_dof]
                      - s.q).abs().amax(dim=1)
                dev_phase = torch.maximum(dev_phase, dq)
                dev_sum, dev_cnt = dev_sum + dq, dev_cnt + 1

                # tcp_now() reads physics without writing, so it is safe while holding
                tcp_i, _, _ = ik.tcp_now()
                bp_i, _ = block_pose()
                ep_inhand[i] = bp_i - tcp_i

                if i + 1 < len(steps) and steps[i + 1].phase == s.phase:
                    continue
                # ---- phase just ended: judge the segment it closes, if any
                gap = ik.finger_gap_mm()
                # Exclude envs that have already reset: Isaac Lab hands them a fresh scene, so
                # their finger gap is measured on a NEW episode and reads as a healthy grip.
                # Uncorrected this made a run with 30/128 mid-episode failures report
                # "grip held 126/128" -- a number that flatly contradicted its own 131 mm
                # lateral p90, and would have been read as "the grip is fine at this level".
                held = (gap > GRIP_LO) & (gap < GRIP_HI) & ~resets
                bp, byaw = block_pose()
                tcp, _, _ = ik.tcp_now()
                phase_log[s.phase] = {"t_end": i + 1, "gap_mm": float(gap.mean()),
                                      "held": int(held.sum()),
                                      "blk_mm": [float(bp[:, j].mean()) * 1000 for j in range(3)],
                                      "arm_dev_mrad": float(dev_phase.mean()) * 1000,
                                      "arm_dev_mrad_p90": float(torch.quantile(dev_phase, 0.9)) * 1000,
                                      "arm_dev_mean_mrad": float((dev_sum / max(1, dev_cnt)).mean()) * 1000,
                                      "slip_mm": (float((ep_inhand[i] - inhand_ref).norm(dim=1).mean()) * 1000
                                                  if inhand_ref is not None else 0.0)}
                dev_phase = torch.zeros(n, device=dev)
                dev_sum, dev_cnt = torch.zeros(n, device=dev), 0
                for sid, judge_phase in SEGMENTS:
                    if judge_phase != s.phase:
                        continue
                    if sid.startswith("reach"):
                        clean = (((bp[:, :2] - tcp[:, :2]).norm(dim=1) < 0.003)
                                 & ((byaw - byaw0).abs() < 0.05) & held).tolist()
                        for k, (h, c) in enumerate(zip(held.tolist(), clean)):
                            outcomes[k][sid] = {"outcome": "grasped" if h else "missed", "clean": c}
                        held_prev = held.clone()
                        # the grasp is complete: this is where the plan's assumption about
                        # where the block sits in the hand is established
                        inhand_ref, close_end = ep_inhand[i].clone(), i + 1
                    elif sid == "release":
                        for k, h in enumerate(seated().tolist()):
                            outcomes[k][sid] = {"outcome": "seated" if h else "unseated"}
                    else:
                        for k, h in enumerate((held & held_prev).tolist()):
                            outcomes[k][sid] = {"outcome": "held" if h else "lost"}
                        held_prev = held_prev & held

            ok = seated() & ~resets
            depth, lat = mdp.insertion_depth(e), mdp.lateral_error(e)
            _, byawf = block_pose()
            # A segment spans from where its motion begins to where the next one does, so a
            # censored grasp censors the whole approach that produced it, not just the close.
            segments = [{"t": seg_t["warmup"] if sid.startswith("reach") else seg_t[jp], "seg": sid}
                        for sid, jp in SEGMENTS]
            obs_np = ep_obs.permute(1, 0, 2).cpu().numpy()
            act_np = ep_act.permute(1, 0, 2).cpu().numpy()
            # Drift of the in-hand offset from its post-grasp reference, per step, in mm.
            # Zeroed before the grasp completes: there is nothing in the hand to drift.
            if inhand_ref is None:
                raise RuntimeError("no 'close' phase in the action stream; slip reference unset")
            # The offset is only meaningful WHILE THE BLOCK IS HELD -- from the end of the
            # grasp to the moment the fingers open. Outside that span there is nothing in the
            # hand, so "in-hand offset" just measures how far the gripper has travelled away
            # from the block: with the retreat phase included that reads ~92 mm and censored
            # 128/128 demos, successes included, on the first run of this detector.
            slip = (ep_inhand - inhand_ref.unsqueeze(0)).norm(dim=2) * 1000.0
            slip[:close_end] = 0.0
            slip[seg_t["release"]:] = 0.0
            slip_np = slip.permute(1, 0).cpu().numpy()
            kept = 0
            for k in range(n):
                if bool(resets[k]):
                    continue      # the scene teleported mid-episode; the obs stream is garbage
                demos.append({
                    "obs": obs_np[k], "act": act_np[k],
                    "success": bool(ok[k]), "segments": segments, "outcomes": outcomes[k],
                    "kind": "nominal" if args_cli.noise_std == 0.0 else "dart",
                    "spawn": [float(bp0[k, 0]), float(bp0[k, 1]), float(byaw0[k])],
                    "plan_ok": bool(plan_ok[k]),
                    "slip": slip_np[k],
                    "slip_max_mm": float(slip_np[k].max()),
                })
                kept += 1
            # Headline deviation = the worst carry phase. The warmup and reach are excluded
            # because the arm is still catching up to the trajectory there whatever the noise.
            carry_dev = max(phase_log[p]["arm_dev_mrad"] for p in ("lift", "back", "spin", "turn", "push"))
            rs = {"rollout": r, "T": T, "kept": kept, "reset": int(resets.sum()),
                  "seated": int(ok.sum()), "rate": float(ok.float().mean()),
                  "plan_converged": int(plan_ok.sum()),
                  "carry_dev_mrad": carry_dev, "phases": phase_log,
                  "depth_p10_mm": float(torch.quantile(depth, 0.1)) * 1000,
                  "lat_p90_mm": float(torch.quantile(lat, 0.9)) * 1000,
                  "yaw_p90": float(torch.quantile(byawf.abs(), 0.9)),
                  "grip_lost": n - int(phase_log["push"]["held"])}
            summary["rollouts"].append(rs)
            print(f"  [rollout {r}] T={T} seated {int(ok.sum())}/{n} = "
                  f"{float(ok.float().mean()) * 100:.1f}%  reset {int(resets.sum())}")
            print(f"      arm dev peak [mrad] " + " ".join(f"{p}:{phase_log[p]['arm_dev_mrad']:.0f}"
                                                           for p in ("lift", "back", "spin", "turn", "push"))
                  + f"   worst-carry {carry_dev:.0f}")
            print(f"      arm dev mean [mrad] " + " ".join(f"{p}:{phase_log[p]['arm_dev_mean_mrad']:.1f}"
                                                           for p in ("lift", "back", "spin", "turn", "push")))
            print(f"      grip held    " + " ".join(f"{p}:{phase_log[p]['held']}"
                                                    for p in ("lift", "back", "spin", "turn", "push")))
            print(f"      in-hand slip [mm] " + " ".join(f"{p}:{phase_log[p]['slip_mm']:.2f}"
                                                         for p in ("lift", "back", "spin", "turn", "push", "release")))
            print(f"      depth p10 {rs['depth_p10_mm']:.1f} mm (need >=40)   "
                  f"lateral p90 {rs['lat_p90_mm']:.2f} mm   |yaw| p90 {rs['yaw_p90']:.4f}")

    n_ok = sum(d["success"] for d in demos)
    slips = np.array([d["slip_max_mm"] for d in demos])
    cens = np.array([slip_censor(d["slip"], args_cli.slip_tol_mm) is not None for d in demos])
    ok_arr = np.array([d["success"] for d in demos])
    summary["slip_max_mm_p50"] = float(np.percentile(slips, 50))
    summary["slip_max_mm_p90"] = float(np.percentile(slips, 90))
    summary["slip_censored_demos"] = int(cens.sum())
    summary["slip_censored_successful"] = int((cens & ok_arr).sum())
    print(f"\n  in-hand slip [mm]: p50 {np.percentile(slips, 50):.2f}  p90 "
          f"{np.percentile(slips, 90):.2f}  max {slips.max():.2f}   "
          f"(tol {args_cli.slip_tol_mm})")
    print(f"  loss-censored by slip: {int(cens.sum())}/{len(demos)} demos "
          f"({int((cens & ok_arr).sum())} of them successful)")
    if cens.mean() > 0.5:
        # A censor that fires on most demos, successes included, is a broken detector rather
        # than a dirty dataset. The first version of this one measured the offset past the
        # release, where there is nothing in the hand, and reported a 92 mm "slip" on 128/128.
        print(f"  *** WARNING: the slip censor fired on {100 * cens.mean():.0f}% of demos, "
              f"{100 * (cens & ok_arr).mean():.0f}% of them successful. That is a detector "
              f"fault, not dirty data -- check the gripped span before training on this. ***")
    # NOT labelled "separates outcomes": measured over the whole gripped span this statistic is
    # INVERTED (AUC 0.301 -- successes score higher), because the expert deliberately drives the
    # block into the back stop and the gripper then keeps advancing while the block cannot move.
    # High slip is the signature of a fully seated insert. scripts/calibrate_slip.py windows the
    # signal properly and decides whether to censor at all.
    print(f"  slip by outcome (raw, whole span -- an INVERTED statistic, see calibrate_slip.py): "
          f"seated p90 {np.percentile(slips[ok_arr], 90):.2f} mm vs "
          f"failed p90 {np.percentile(slips[~ok_arr], 90) if (~ok_arr).any() else float('nan'):.2f} mm")
    summary["total_demos"] = len(demos)
    summary["successful"] = n_ok
    summary["success_rate"] = n_ok / max(1, len(demos))
    attempted = sum(r["kept"] + r["reset"] for r in summary["rollouts"])
    summary["attempted"] = attempted
    summary["seated_rate_of_attempted"] = n_ok / max(1, attempted)
    print("\n" + "=" * 78)
    print(f"  COLLECTED  {len(demos)} demos kept of {attempted} attempted, {n_ok} successful")
    # Two rates, because they answer different questions and only one is the expert's score.
    # The conditional rate is over KEPT episodes and silently excludes every env that reset
    # mid-episode -- at noise_std=0.10 that read 98.0 % while the expert had actually seated
    # only 75.0 % of the blocks it was given.
    print(f"    seated / attempted   {n_ok}/{attempted} = "
          f"{100 * n_ok / max(1, attempted):.1f}%   <- the expert's real score")
    print(f"    seated / kept        {n_ok}/{len(demos)} = "
          f"{100 * n_ok / max(1, len(demos)):.1f}%   (conditional on not resetting)")
    print(f"    noise_std={args_cli.noise_std} on phases {sorted(noise_phases)}")
    print("=" * 78)

    if not args_cli.dry_run:
        tag = args_cli.task.replace("Rebot-PrecisionSlot-", "").replace("-v0", "")
        default = (f"data/demos_{tag}_n{args_cli.num_envs}x{args_cli.rollouts}"
                   f"_s{args_cli.seed}_noise{args_cli.noise_std:g}.hdf5")
        path = Path(args_cli.out or default)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(path), "w") as f:
            grp = f.create_group("data")
            for i, d in enumerate(demos):
                g = grp.create_group(f"demo_{i}")
                g.create_dataset("obs/policy", data=d["obs"].astype(np.float32), compression="gzip")
                g.create_dataset("actions", data=d["act"].astype(np.float32), compression="gzip")
                mask = build_train_mask(d["obs"].shape[0], d["segments"], d["outcomes"])
                cut = slip_censor(d["slip"], args_cli.slip_tol_mm)
                if cut is not None:
                    mask[cut:] = 0
                g.create_dataset("train_mask", data=mask)
                # The raw signal rides along so the censor threshold can be re-derived offline
                # without re-running the simulator. 599 floats per demo.
                g.create_dataset("slip_mm", data=d["slip"].astype(np.float32), compression="gzip")
                g.attrs["slip_max_mm"] = d["slip_max_mm"]
                g.attrs["slip_censor_t"] = -1 if cut is None else cut
                g.attrs["success"] = d["success"]
                g.attrs["num_samples"] = d["obs"].shape[0]
                g.attrs["episode_kind"] = d["kind"]
                g.attrs["perturb_steps"] = json.dumps([])
                g.attrs["segments"] = json.dumps(d["segments"])
                g.attrs["outcomes"] = json.dumps(d["outcomes"])
                g.attrs["spawn"] = json.dumps(d["spawn"])
                g.attrs["plan_ok"] = d["plan_ok"]
                g.attrs["noise_std"] = args_cli.noise_std
                g.attrs["task"] = args_cli.task
            grp.attrs["total"] = len(demos)
        summary["path"] = str(path)
        print(f"[h5] wrote {len(demos)} demos -> {path} "
              f"({path.stat().st_size / 1e6:.1f} MB)")
        (path.with_suffix(".json")).write_text(json.dumps(summary, indent=2))
        print(f"[h5] wrote summary -> {path.with_suffix('.json')}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
