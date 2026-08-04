# Copyright (c) 2026. Slot-insertion port of the eva_bc pipeline.
# SPDX-License-Identifier: BSD-3-Clause

"""Is the scripted expert's PLAN a pure function of the block pose? Measured, not assumed.

Why this exists
---------------
Collecting 4 x 128 demos in one process gave 128/128 seated in rollout 0 and 87.5 / 84.4 /
89.1 % in rollouts 1-3. Everything that should explain that was ruled out with the recorded
data:

* initial conditions are **bit-identical** -- ``|q0 - q_default| = 0``, joint velocity 0,
  block placed at exactly its planned spawn (dx/dy std 0.0000 mm, z 35.000 mm);
* the robot root frame does not drift (0.0000 mm across rollouts);
* plan convergence does not predict success -- rollout 0 had the **worst** convergence
  (107/128) and the best outcome;
* failures are not a fixed per-env property: the overlap of failing env indices across
  rollouts is at chance (5 observed vs 2.5 expected, 1 vs 1.8, 2 vs 2.2), so it is not the
  startup friction randomisation;
* spawns are iid across rollouts (per-rollout means within 1 SEM).

What is left is the planner. The commanded joint target at the end of ``push`` is
**spawn-independent** by construction -- the target pose is ``(insert_x, 0, carry_z)`` with the
slot axis, identical for every env and every rollout -- and yet its mean drifts monotonically
with rollout index (j2: 1.85554 -> 1.85595 -> 1.87826 -> 1.88988, up to 17 mrad).

So: does planning the *same* block pose twice in one process, separated by a full 599-step
execution, produce the same trajectory? ``expert/ik.py`` computes forward kinematics by
``write_joint_state_to_sim`` + ``sim.forward()`` + ``robot.update(0)`` and then reading
``body_pos_w`` -- if that read is stale or order-dependent, every IK solution in this project
inherits it. This project has already been bitten once by a "measurement" that never saw
physics poses (``UsdGeom.BBoxCache``), so the read path is tested rather than trusted.

.. code-block:: bash

    python slot/analysis/plan_determinism.py --num_envs 32
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test whether the expert's plan is deterministic.")
parser.add_argument("--task", type=str, default="Rebot-PrecisionSlot-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import reBot_RL.tasks  # noqa: F401
from reBot_RL.tasks.manager_based.challenge import mdp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slot.expert import plan as P  # noqa: E402
from slot.expert.ik import ArmIK  # noqa: E402


def main() -> None:
    torch.manual_seed(args_cli.seed)
    params = P.ExpertParams()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    dev, n = e.device, e.num_envs
    block, robot = e.scene["block"], e.scene["robot"]
    ik = ArmIK(env)

    def block_pose():
        p = torch.as_tensor(block.data.root_pos_w.torch, device=dev) - e.scene.env_origins
        q = torch.as_tensor(block.data.root_quat_w.torch, device=dev)
        return p, mdp.yaw_of(q)

    def write_block(bp, byaw):
        block.write_root_state_to_sim(torch.cat([
            bp + e.scene.env_origins,
            torch.stack([torch.zeros(n, device=dev), torch.zeros(n, device=dev),
                         torch.sin(byaw / 2), torch.cos(byaw / 2)], dim=1),
            torch.zeros((n, 6), device=dev)], dim=1))

    with torch.inference_mode():
        env.reset()
        seed = P.solve_seed(env, ik, params, verbose=False)
        q_seed, axis_slot, sign = seed["q_seed"], seed["axis_slot"], seed["sign"]

        # TEST 3 runs FIRST, and that ordering is the whole point. The hypothesis under test is
        # "the first executed episode in a process behaves differently from later ones", so its
        # run 0 must be a genuine first execution. An earlier version of this file ran TEST 3
        # last, after TEST 2 had already executed a full episode -- every repeat was therefore a
        # "later" run and the test could not see the effect it was written for. It reported
        # 114 / 109 / 109 seated and I nearly read that as evidence of no effect.
        env.reset()
        bp0, byaw0 = block_pose()
        plan_a = P.plan(ik, params, bp0, byaw0, q_seed, axis_slot, sign)["plans"]

        print("\n" + "=" * 78)
        print("  TEST 3 -- same plan, same initial state, repeated. run 0 IS the process's first.")
        print("=" * 78)
        seat_z = mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2]
        budget = int(e.max_episode_length) - 1
        runs = []
        for rep in range(3):
            env.reset()
            write_block(bp0, byaw0)
            e.sim.forward()
            robot.update(0.0)
            block.update(0.0)
            q_start = torch.as_tensor(robot.data.joint_pos.torch, device=dev).clone()
            resets = torch.zeros(n, dtype=torch.bool, device=dev)
            for s in P.action_stream(plan_a, params, budget=budget):
                _, _, term, trunc, _ = env.step(ik.action(s.q, close=s.close))
                resets |= torch.as_tensor(term, device=dev).bool() | torch.as_tensor(trunc, device=dev).bool()
            p, _ = block_pose()
            ok = mdp.is_inserted(e) & ((p[:, 2] - seat_z).abs() < 0.006) & ~resets
            depth = mdp.insertion_depth(e)
            runs.append({"ok": ok.clone(), "depth": depth.clone()})
            print(f"    run {rep}{' (FIRST in process)' if rep == 0 else '':>20}: seated "
                  f"{int(ok.sum())}/{n}   depth mean {float(depth.mean()) * 1000:.2f} mm  "
                  f"min {float(depth.min()) * 1000:.2f}   resets {int(resets.sum())}   "
                  f"|q_start - default| max "
                  f"{float((q_start - ik.q_default.unsqueeze(0)).abs().max()):.2e}")
        print()
        for i in (1, 2):
            dd = (runs[i]["depth"] - runs[0]["depth"]).abs()
            print(f"    run {i} vs run 0:  max |depth diff| {float(dd.max()) * 1000:8.3f} mm   "
                  f"mean {float(dd.mean()) * 1000:6.3f} mm   outcome flips "
                  f"{int((runs[i]['ok'] != runs[0]['ok']).sum())}/{n}")
        d12 = (runs[2]["depth"] - runs[1]["depth"]).abs()
        print(f"    run 2 vs run 1:  max |depth diff| {float(d12.max()) * 1000:8.3f} mm   "
              f"mean {float(d12.mean()) * 1000:6.3f} mm   outcome flips "
              f"{int((runs[2]['ok'] != runs[1]['ok']).sum())}/{n}")
        s0 = int(runs[0]["ok"].sum())
        later = (int(runs[1]["ok"].sum()) + int(runs[2]["ok"].sum())) / 2
        print(f"\n    first execution {s0}/{n} = {100 * s0 / n:.1f}%   vs   later mean "
              f"{later:.1f}/{n} = {100 * later / n:.1f}%   delta {100 * (s0 - later) / n:+.1f} pts")
        print("    (binomial SE at this n is about "
              f"{100 * (0.87 * 0.13 / n) ** 0.5:.1f} pts, so a real effect needs a few times that)")
        print("=" * 78 + "\n")

        # ---------------------------------------------------------------- TEST 1: fk order
        # Same q, asked twice, with a different q in between. A pure function of q gives the
        # same answer both times.
        print("\n" + "=" * 78)
        print("  TEST 1 -- is fk() a pure function of the joint vector?")
        print("=" * 78)
        qa = q_seed.clone()
        qb = (q_seed + 0.20).clamp(ik.lo, ik.hi)
        pa1, ua1, _ = ik.fk(qa, 0.016)
        pb1, _, _ = ik.fk(qb, 0.016)
        pa2, ua2, _ = ik.fk(qa, 0.016)
        d_pos = (pa1 - pa2).abs().max()
        d_ax = (ua1 - ua2).abs().max()
        print(f"    fk(qa) then fk(qb) then fk(qa):  max |dpos| = {float(d_pos) * 1e6:.3f} um, "
              f"max |daxis| = {float(d_ax):.3e}")
        print(f"    -> fk is {'PURE' if float(d_pos) < 1e-9 else 'ORDER-DEPENDENT (STALE READ)'}")
        print(f"    (sanity: fk(qa) vs fk(qb) differ by {float((pa1 - pb1).abs().max()) * 1000:.1f} mm, "
              f"so the probe is not trivially reading the same thing)")

        # ---------------------------------------------------------------- TEST 2: plan drift
        print("\n" + "=" * 78)
        print("  TEST 2 -- does planning the SAME block pose twice give the same trajectory?")
        print("=" * 78)
        # replan immediately, nothing in between except the planning itself
        plan_b = P.plan(ik, params, bp0, byaw0, q_seed, axis_slot, sign)["plans"]

        # now execute a full episode, then replan the very same block pose
        env.reset()
        write_block(bp0, byaw0)
        e.sim.forward()
        robot.update(0.0)
        for s in P.action_stream(plan_a, params, budget=int(e.max_episode_length) - 1):
            env.step(ik.action(s.q, close=s.close))
        env.reset()
        plan_c = P.plan(ik, params, bp0, byaw0, q_seed, axis_slot, sign)["plans"]

        print(f"\n    {'phase':>9} {'A vs B (back-to-back)':>24} {'A vs C (after 599 steps)':>26}")
        worst_ab = worst_ac = 0.0
        for name in P.PHASES:
            ab = float((plan_a[name]["q"] - plan_b[name]["q"]).abs().max())
            ac = float((plan_a[name]["q"] - plan_c[name]["q"]).abs().max())
            worst_ab, worst_ac = max(worst_ab, ab), max(worst_ac, ac)
            print(f"    {name:>9} {ab * 1000:>21.4f} mrad {ac * 1000:>23.4f} mrad")
        print(f"\n    worst A-B {worst_ab * 1000:.4f} mrad   worst A-C {worst_ac * 1000:.4f} mrad")

        # The push END pose is spawn-independent, so it is the cleanest single number.
        for lab, pl in (("A", plan_a), ("B", plan_b), ("C", plan_c)):
            q_end = pl["push"]["q"][:, -1, :]
            print(f"    plan {lab}: push-end joint mean "
                  + " ".join(f"{float(v):.6f}" for v in q_end.mean(dim=0)))

        print()
        if worst_ab < 1e-6 and worst_ac < 1e-6:
            print("    VERDICT: the plan is a pure function of the block pose. The rollout")
            print("             effect lies in EXECUTION, not planning.")
        elif worst_ab < 1e-6 <= worst_ac:
            print("    VERDICT: planning is repeatable back-to-back but CHANGES after an")
            print("             episode has been executed -- the planner reads simulator state")
            print("             that a rollout mutates. This is the rollout effect.")
        else:
            print("    VERDICT: planning is not even repeatable back-to-back.")
        print("=" * 78 + "\n")


    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
