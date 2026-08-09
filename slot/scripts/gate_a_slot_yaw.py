#!/usr/bin/env python
"""GATE A -- the angled slot's geometry and its success predicate agree.

Runs before any expert, any policy and any deletion. Four checks, each of which can fail:

**A1 regression at theta = 0.** The three predicates were rewritten from world axes into the
slot frame. With the yaw pinned to zero they must reproduce the *old* arithmetic on a batch of
random block poses. The old formulas are inlined here rather than imported, so this compares
against what the code used to say and not against a shared helper that could be wrong twice.

  Note on "bit-for-bit": the plan asked for exact equality and that is not achievable.
  ``(x - cx) + D/2`` and ``x - (cx - D/2)`` are the same real number and different floats, so
  the depth check reports the measured deviation instead. The *predicate* ``is_inserted`` is
  boolean and IS checked for exact agreement, which is what actually matters.

**A2 analytic placement at random theta.** Blocks are teleported to exact slot-frame
coordinates and the predicates must read those coordinates back. Four cells: seated, too
shallow, too far off the centreline, too far off square. A predicate that ignored the angle
would pass "seated" only at theta = 0.

**A3 the fixture really moved.** Predicates agreeing with themselves proves nothing about the
geometry. The four fixture bodies' poses are read back FROM THE SIM and compared against the
closed-form rotation about ``SLOT_CENTER``.

**A4 the collision moved with it.** A3 reads a pose; this one reads a consequence. A block is
parked where a *rotated* wall now stands -- a spot that is empty floor in the unrotated
fixture -- and the sim is stepped. If the collider followed the pose, PhysX ejects the block;
if only the render moved, the block sits still.

    python slot/scripts/gate_a_slot_yaw.py --num_envs 256
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--yaw", type=float, default=0.5, help="|theta| bound for the A2/A3 cells [rad]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from reBot_RL.tasks.manager_based.challenge import mdp  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def yaw_quat(th: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion for a yaw about +z."""
    z = torch.zeros_like(th)
    return torch.stack([z, z, torch.sin(0.5 * th), torch.cos(0.5 * th)], dim=-1)


def place(env, pos_local: torch.Tensor, quat: torch.Tensor) -> None:
    """Teleport the block to an env-local position; verify the write actually took."""
    ids = torch.arange(env.num_envs, device=env.device)
    pose = torch.cat([pos_local + env.scene.env_origins, quat], dim=-1)
    env.scene["block"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    env.scene["block"].write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(env.num_envs, 6, device=env.device), env_ids=ids)
    back = mdp.object_pos_local(env, "block")
    err = (back - pos_local).abs().max().item()
    assert err < 1e-5, f"block pose write did not land: max err {err:.2e} m"


def slot_to_world(env, along, across, extra_yaw, z):
    """Slot-frame (along, across, dyaw) -> env-frame position and quaternion."""
    th = mdp.slot_yaw(env)
    c, s = torch.cos(th), torch.sin(th)
    x = mdp.SLOT_CENTER[0] + along * c - across * s
    y = mdp.SLOT_CENTER[1] + along * s + across * c
    return torch.stack([x, y, z], dim=-1), yaw_quat(th + extra_yaw)


def main() -> None:
    device = "cuda:0"
    n = args.num_envs
    torch.manual_seed(args.seed)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=n)
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    env_cfg.events.reset_slot.params["yaw_range"] = (0.0, 0.0)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    half = env.cfg.slot_half_width
    zero_act = torch.zeros(n, env.action_space.shape[1], device=device)

    # ---------------------------------------------------------------- A1
    print("\nA1  theta = 0 reproduces the axis-aligned arithmetic")
    th = mdp.slot_yaw(env)
    check("slot_yaw is exactly zero", bool((th == 0).all()), f"max |theta| = {th.abs().max():.3e}")

    # random poses, deliberately centred on the slot so both outcomes of is_inserted occur
    pos = torch.empty(n, 3, device=device)
    pos[:, 0].uniform_(mdp.SLOT_CENTER[0] - 0.06, mdp.SLOT_CENTER[0] + 0.04)
    pos[:, 1].uniform_(-0.03, 0.03)
    pos[:, 2].uniform_(0.030, 0.070)
    q = yaw_quat(torch.empty(n, device=device).uniform_(-math.pi, math.pi))
    place(env, pos, q)

    old_depth = mdp.object_pos_local(env, "block")[:, 0] - (mdp.SLOT_CENTER[0] - mdp.SLOT_DEPTH / 2)
    old_lat = (mdp.object_pos_local(env, "block")[:, 1] - mdp.SLOT_CENTER[1]).abs()
    old_yaw = mdp.yaw_of(mdp.object_quat(env, "block")).abs()
    old_ins = ((old_depth >= mdp.SUCCESS_DEPTH) & (old_lat <= half) & (old_yaw <= mdp.SUCCESS_YAW)
               & (mdp.object_pos_local(env, "block")[:, 2] > mdp.SLOT_FLOOR_Z - 0.005))

    d_err = (mdp.insertion_depth(env) - old_depth).abs().max().item()
    l_err = (mdp.lateral_error(env) - old_lat).abs().max().item()
    y_err = (mdp.yaw_error(env) - old_yaw).abs().max().item()
    new_ins = mdp.is_inserted(env)
    check("insertion_depth matches", d_err < 1e-6, f"max dev {d_err:.3e} m  (tolerance 1.5e-3 m)")
    check("lateral_error is bit-exact", l_err == 0.0, f"max dev {l_err:.3e} m")
    check("yaw_error is bit-exact", y_err == 0.0, f"max dev {y_err:.3e} rad")
    check("is_inserted identical", bool((new_ins == old_ins).all()),
          f"{int(old_ins.sum())}/{n} inserted under the old rule, "
          f"{int(new_ins.sum())}/{n} under the new one")
    if int(old_ins.sum()) == 0:
        FAILS.append("A1 vacuous: no sampled pose was ever inserted")
        print("  [FAIL] A1 is vacuous -- widen the sampling")

    # ---------------------------------------------------------------- A2
    print(f"\nA2  analytic placement at theta ~ U(-{args.yaw}, {args.yaw})")
    ids = torch.arange(n, device=device)
    mdp.set_slot_yaw(env, ids, torch.empty(n, device=device).uniform_(-args.yaw, args.yaw))
    th = mdp.slot_yaw(env)
    check("theta spans the range", float(th.max() - th.min()) > 1.5 * args.yaw,
          f"[{th.min():+.3f}, {th.max():+.3f}] rad")

    seated_z = torch.full((n,), mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2], device=device)
    along_ok = mdp.SUCCESS_DEPTH - mdp.SLOT_DEPTH / 2 + 0.002      # 2 mm past the depth threshold
    zeros = torch.zeros(n, device=device)

    cells = {
        "seated": (along_ok, 0.0, 0.0, True),
        "too shallow": (along_ok - 0.010, 0.0, 0.0, False),
        "off centreline": (along_ok, half + 0.002, 0.0, False),
        "off square": (along_ok, 0.0, mdp.SUCCESS_YAW + 0.02, False),
    }
    for label, (a, x, dy, want) in cells.items():
        p, qq = slot_to_world(env, zeros + a, zeros + x, zeros + dy, seated_z)
        place(env, p, qq)
        got = mdp.is_inserted(env)
        check(f"{label:<15} -> is_inserted == {want}", bool((got == want).all()),
              f"{int(got.sum())}/{n} true")
        if label == "seated":
            e_d = (mdp.insertion_depth(env) - (a + mdp.SLOT_DEPTH / 2)).abs().max().item()
            e_l = mdp.lateral_error(env).abs().max().item()
            e_y = mdp.yaw_error(env).abs().max().item()
            check("slot-frame coordinates read back", max(e_d, e_l) < 2e-6 and e_y < 2e-6,
                  f"depth {e_d:.2e} m, lateral {e_l:.2e} m, yaw {e_y:.2e} rad")

    # ---------------------------------------------------------------- A3
    print("\nA3  the fixture bodies actually rotated")
    cx, cy = mdp.SLOT_CENTER
    c, s = torch.cos(th), torch.sin(th)
    worst_pos = worst_rot = 0.0
    for part in mdp.SLOT_PARTS:
        obj = env.scene[part]
        d0 = obj.data.default_root_pose.torch
        got = obj.data.root_pose_w.torch.clone()
        got[:, 0:3] -= env.scene.env_origins
        ox, oy = d0[:, 0] - cx, d0[:, 1] - cy
        want = torch.stack([cx + ox * c - oy * s, cy + ox * s + oy * c, d0[:, 2]], dim=-1)
        worst_pos = max(worst_pos, (got[:, 0:3] - want).abs().max().item())
        worst_rot = max(worst_rot, (mdp.yaw_of(got[:, 3:7]) - th).abs().max().item())
    check("fixture positions follow the rotation", worst_pos < 1e-5, f"max dev {worst_pos:.2e} m")
    check("fixture orientations follow theta", worst_rot < 1e-5, f"max dev {worst_rot:.2e} rad")
    # a rotation that does nothing would also pass the two checks above
    swept = (torch.stack([env.scene[p].data.root_pose_w.torch[:, 0:2] for p in mdp.SLOT_PARTS])
             - torch.stack([env.scene[p].data.default_root_pose.torch[:, 0:2] for p in mdp.SLOT_PARTS])
             - env.scene.env_origins[None, :, 0:2]).norm(dim=-1).max().item()
    check("the fixture moved at all", swept > 0.005, f"max part displacement {swept * 1000:.1f} mm")

    # ---------------------------------------------------------------- A4
    print("\nA4  the COLLIDER moved, not just the pose")
    # theta = 1.2 rad on purpose: it puts the +y wall's new centre over what is bare floor in
    # the unrotated fixture, so "block ejected" and "block sits still" are 30 mm apart.
    big = torch.full((n,), 1.2, device=device)
    mdp.set_slot_yaw(env, ids, big)
    wall_off = env.scene["slot_wall_py"].data.default_root_pose.torch[:, 0:2] - torch.tensor(
        [cx, cy], device=device)
    cb, sb = torch.cos(big), torch.sin(big)
    wx = cx + wall_off[:, 0] * cb - wall_off[:, 1] * sb
    wy = cy + wall_off[:, 0] * sb + wall_off[:, 1] * cb
    start = torch.stack([wx, wy, torch.full((n,), mdp.SLOT_FLOOR_Z + mdp.BLOCK_HALF[2],
                                            device=device)], dim=-1)
    place(env, start, yaw_quat(big))
    for _ in range(20):
        env.step(zero_act)
    moved = (mdp.object_pos_local(env, "block") - start).norm(dim=-1)
    check("block is ejected by the rotated wall", float(moved.median()) > 0.010,
          f"median displacement {float(moved.median()) * 1000:.1f} mm "
          f"(unrotated floor contact would be < 2 mm)")

    env.close()
    print("\n" + ("GATE A PASSED" if not FAILS else f"GATE A FAILED: {FAILS}"))
    app.close()
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
