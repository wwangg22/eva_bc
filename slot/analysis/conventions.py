# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Settle three conventions empirically before any IK is written on top of them.

Why
---
The repo carries a **contradiction** about quaternion order that is load-bearing for IK:

* ``challenge/mdp/common.object_quat`` documents its return as ``(x, y, z, w)`` "as
  everywhere in Isaac Lab 3.0", and ``yaw_of`` unpacks it in that order;
* ``challenge/mdp/terminations.block_toppled`` feeds the **same** tensor to
  ``isaaclab.utils.math.quat_apply``, which is conventionally WXYZ;
* ``slot_insertion_probe.py`` feeds ``body_quat_w`` to ``quat_apply`` and its CEM converges
  to 0.13-2.1 mm, which only works if that pairing is right.

At most one reading of that set is correct. A constant-offset or swapped-order quaternion is
exactly the class of error that stays invisible in a reach reward and is fatal to a scripted
grasp -- the same shape as the 33 mm TCP error that produced several flatly wrong "this
object cannot be grasped" measurements. So measure it, do not reason about it.

Checks
------
1. **Quaternion order** -- rotate the block by a known yaw with ``write_root_state_to_sim``
   in both orders, read back with ``yaw_of`` and with ``quat_apply``, see which agrees.
2. **Jacobian API** -- whether ``root_physx_view.get_jacobians()`` is available, its shape,
   and the body-index convention for a fixed-base articulation; validated against a
   **finite-difference** Jacobian computed from the sim's own FK.
3. **TCP consistency** -- that ``ee_frame`` (the env's FrameTransformer, bound to
   ``mdp.TCP_OFFSET``) and a hand-rolled ``gripper_end + quat_apply(offset)`` agree, and that
   both land on the finger-body midpoint when the fingers are shut (the definition of the
   grasp point per CHALLENGE_SUITE C10).

.. code-block:: bash

    python slot/analysis/conventions.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify quaternion / Jacobian / TCP conventions.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
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


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev = e.device
    robot = e.scene["robot"]
    block = e.scene["block"]
    n = e.num_envs
    ok = []

    with torch.inference_mode():
        env.reset()

        # ---------------------------------------------------------- 1. quaternion order
        print("\n" + "=" * 74)
        print("CHECK 1 -- quaternion order of root_quat_w / body_quat_w")
        print("=" * 74)
        yaw = 0.30
        pos = torch.tensor([0.220, -0.130, 0.035], device=dev).repeat(n, 1) + e.scene.env_origins
        s, c = torch.sin(torch.tensor(yaw / 2)), torch.cos(torch.tensor(yaw / 2))
        for name, q in (("XYZW", [0.0, 0.0, float(s), float(c)]),
                        ("WXYZ", [float(c), 0.0, 0.0, float(s)])):
            quat = torch.tensor(q, device=dev).repeat(n, 1)
            block.write_root_state_to_sim(torch.cat([pos, quat, torch.zeros((n, 6), device=dev)], dim=1))
            e.sim.forward()
            block.update(0.0)
            back = torch.as_tensor(block.data.root_quat_w.torch, device=dev)
            y_of = float(mdp.yaw_of(back).mean())
            up = quat_apply(back, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
            xa = quat_apply(back, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
            print(f"  wrote as {name}: read back {[round(float(v), 4) for v in back[0]]}")
            print(f"      mdp.yaw_of -> {y_of:+.4f} (wrote {yaw:+.4f})   "
                  f"quat_apply(+z).z -> {float(up[:, 2].mean()):+.4f} (upright wants +1)")
            print(f"      quat_apply(+x) -> ({float(xa[0, 0]):+.3f}, {float(xa[0, 1]):+.3f}), "
                  f"atan2 -> {float(torch.atan2(xa[:, 1], xa[:, 0]).mean()):+.4f}")
        print("  READING: whichever write order comes back unchanged is the storage order;")
        print("           whichever function then reports yaw=+0.30 and up.z=+1 is consistent.")

        # ------------------------------------------------------------- 2. Jacobian API
        print("\n" + "=" * 74)
        print("CHECK 2 -- analytic Jacobian availability, shape and body indexing")
        print("=" * 74)
        arm_dof = [robot.joint_names.index(f"joint{i}") for i in range(1, 7)]
        fing_dof = [robot.joint_names.index(x) for x in ("joint_left", "joint_right")]
        end_idx = robot.body_names.index("gripper_end")
        q_default = torch.as_tensor(robot.data.default_joint_pos[0], device=dev).clone()
        is_fixed = robot.is_fixed_base
        print(f"  is_fixed_base = {is_fixed}, num bodies = {len(robot.body_names)}, "
              f"num joints = {len(robot.joint_names)}")
        jac = None
        try:
            jac = torch.as_tensor(robot.root_physx_view.get_jacobians(), device=dev)
            print(f"  root_physx_view.get_jacobians() -> {tuple(jac.shape)}")
            print(f"  expected jacobian row for '{'gripper_end'}': body index {end_idx} -> "
                  f"jacobian index {end_idx - 1 if is_fixed else end_idx}")
        except Exception as exc:  # noqa: BLE001
            print(f"  UNAVAILABLE: {exc!r}  -> the expert must use a finite-difference Jacobian")

        def fk_tcp(q_arm: torch.Tensor) -> torch.Tensor:
            q = q_default.unsqueeze(0).repeat(n, 1)
            q[:, arm_dof] = q_arm
            q[:, fing_dof] = 0.0
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            e.sim.forward()
            robot.update(0.0)
            bp = torch.as_tensor(robot.data.body_pos_w.torch, device=dev) - e.scene.env_origins.unsqueeze(1)
            bq = torch.as_tensor(robot.data.body_quat_w.torch, device=dev)
            offs = torch.tensor(mdp.TCP_OFFSET, device=dev).repeat(n, 1)
            return bp[:, end_idx, :] + quat_apply(bq[:, end_idx, :], offs)

        # finite-difference Jacobian of TCP position w.r.t. the 6 arm joints, at the start pose
        q0 = q_default[arm_dof].unsqueeze(0).repeat(n, 1)
        base = fk_tcp(q0)
        eps = 1e-4
        cols = []
        for j in range(6):
            qp = q0.clone()
            qp[:, j] += eps
            cols.append((fk_tcp(qp) - base) / eps)
        jac_fd = torch.stack(cols, dim=2)  # (n, 3, 6)
        print(f"  finite-difference dTCP/dq at the start pose (env 0), rows x/y/z:")
        for r, lbl in enumerate("xyz"):
            print(f"      d{lbl}: {[round(float(v), 4) for v in jac_fd[0, r, :]]}")
        cond = torch.linalg.svdvals(jac_fd[0])
        print(f"  singular values: {[round(float(v), 4) for v in cond]}  "
              f"(cond {float(cond[0] / cond[-1].clamp(min=1e-9)):.1f})")
        ok.append(("finite-difference Jacobian usable", bool(torch.isfinite(jac_fd).all())))
        if jac is not None:
            ji = end_idx - 1 if is_fixed else end_idx
            try:
                ja = jac[:, ji, :3, :][:, :, arm_dof]  # (n,3,6) linear rows, arm columns
                rel = (ja[0] - jac_fd[0]).abs().max() / jac_fd[0].abs().max().clamp(min=1e-9)
                print(f"  analytic vs finite-difference (gripper_end LINK, not TCP): "
                      f"max rel diff {float(rel):.3f}")
                print("      (a nonzero diff is expected -- the analytic Jacobian is for the")
                print("       link origin, the finite-difference one is for the offset TCP)")
            except Exception as exc:  # noqa: BLE001
                print(f"  could not slice analytic jacobian: {exc!r}")

        # --------------------------------------------------------- 3. TCP consistency
        print("\n" + "=" * 74)
        print("CHECK 3 -- TCP definition: ee_frame vs hand-rolled vs finger midpoint")
        print("=" * 74)
        q = q_default.unsqueeze(0).repeat(n, 1)
        q[:, fing_dof] = 0.0  # shut: the two finger-body origins coincide at the grasp point
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        e.sim.forward()
        robot.update(0.0)
        e.scene["ee_frame"].update(0.0)
        bp = torch.as_tensor(robot.data.body_pos_w.torch, device=dev) - e.scene.env_origins.unsqueeze(1)
        bq = torch.as_tensor(robot.data.body_quat_w.torch, device=dev)
        offs = torch.tensor(mdp.TCP_OFFSET, device=dev).repeat(n, 1)
        tcp_manual = bp[:, end_idx, :] + quat_apply(bq[:, end_idx, :], offs)
        tcp_frame = (torch.as_tensor(e.scene["ee_frame"].data.target_pos_w.torch, device=dev)[:, 0, :]
                     - e.scene.env_origins)
        li = robot.body_names.index("gripper_left")
        ri = robot.body_names.index("gripper_right")
        mid = 0.5 * (bp[:, li, :] + bp[:, ri, :])
        d_frame = float((tcp_manual - tcp_frame).norm(dim=1).mean()) * 1000
        d_mid = float((tcp_manual - mid).norm(dim=1).mean()) * 1000
        print(f"  TCP (manual)        {[round(float(v), 4) for v in tcp_manual[0]]}")
        print(f"  TCP (ee_frame)      {[round(float(v), 4) for v in tcp_frame[0]]}   "
              f"delta {d_frame:.3f} mm")
        print(f"  finger midpoint     {[round(float(v), 4) for v in mid[0]]}   delta {d_mid:.3f} mm")
        print(f"  finger separation at q=0: "
              f"{float((bp[:, li, :] - bp[:, ri, :]).norm(dim=1).mean()) * 1000:.3f} mm "
              f"(C10 says the origins coincide)")
        ok.append(("ee_frame agrees with manual TCP (<1 mm)", d_frame < 1.0))
        ok.append(("TCP is the shut-finger midpoint (<2 mm)", d_mid < 2.0))

        print("\n" + "=" * 74)
        for label, passed in ok:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        print("=" * 74)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
