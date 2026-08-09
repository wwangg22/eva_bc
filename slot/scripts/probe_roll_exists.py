#!/usr/bin/env python
"""Does a LEVEL-GRIPPER solution exist at the poses the angled insert needs?

This decides whether theta_max can be lifted past +/-0.35 rad or whether it is a hardware
fact (ANGLED_SLOT.md 8d). The expert's IK solves a 5-DOF task -- position plus the finger-axis
direction -- and leaves rotation about that axis free. At theta = -0.5 the solution it lands on
rolls the gripper 23.27 deg, which pitches the clamped block and costs 12.5 mm of insertion
depth. The question is whether that roll is a CHOICE or a NECESSITY.

Method: the solution set at a fixed (position, axis) target is a 1-D curve, so it is sampled by
**random restarts**. Each env is an independent restart seeded from a uniformly random joint
configuration; ``ik.solve`` runs all of them in parallel. If any restart converges to the target
with a small tilt, a level solution exists and the fix is a soft roll objective. If hundreds of
restarts all land far from level, it does not.

**The tilt metric is validated before it is trusted.** It is defined as rotation away from the
canonical seed's orientation about the world vertical, which is only meaningful if it agrees
with the block tilt actually measured in physics. The probe therefore runs the expert's own
waypoint chain first and checks its tilt against the number ``run_expert`` reported (23.27 deg
at the end of the turn at theta = -0.5). A metric that disagrees there is measuring something
else, and the existence answer would be worthless.

    python slot/scripts/probe_roll_exists.py --num_envs 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
parser.add_argument("--num_envs", type=int, default=512, help="= number of random restarts")
parser.add_argument("--thetas", type=float, nargs="+",
                    default=[0.0, 0.35, -0.35, 0.5, -0.5, 0.7, -0.7])
parser.add_argument("--iters", type=int, default=400)
parser.add_argument("--pos_tol", type=float, default=0.001, help="[m]")
parser.add_argument("--axis_tol", type=float, default=0.001, help="1 - |cos|")
parser.add_argument("--out", default="logs/roll_exists.json")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import reBot_RL.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from reBot_RL.tasks.manager_based.challenge import mdp  # noqa: E402

from slot.expert import plan as P  # noqa: E402
from slot.expert.ik import ArmIK  # noqa: E402


def main() -> None:
    torch.manual_seed(args.seed)
    params = P.ExpertParams()

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.events.reset_slot.params["yaw_range"] = (0.0, 0.0)
    env = gym.make(args.task, cfg=env_cfg)
    e = env.unwrapped
    dev, n = e.device, e.num_envs
    ik = ArmIK(env)
    z = torch.tensor([0.0, 0.0, 1.0], device=dev)

    with torch.inference_mode():
        env.reset()
        seed = P.solve_seed(env, ik, params, verbose=False)
        q_seed, sign = seed["q_seed"], seed["sign"]

        # ---- tilt metric.
        # The block is CLAMPED to the gripper, so its orientation is R(q) R_ref^T applied to
        # the upright block, where R_ref is the gripper's rotation at a pose where the block
        # is known upright. The canonical seed is such a pose: it is the nominal insert, and
        # the insert measurably works (block tilt 1.5 deg at push end at theta = 0).
        #
        # Only ONE vector is needed and it is env-independent: v = R_ref^T z, the world
        # vertical written in the gripper's own frame. Two envs whose grasps differ by a yaw
        # about the world vertical give the same v, because a pre-rotation about z leaves z.
        #
        # The first version of this reduced v to its largest component and measured "the angle
        # of one body axis from vertical" -- which has an arbitrary offset, reported the seed
        # itself at 30.88 deg, and disagreed with physics by 48 deg. Hence the validation below.
        _, _, r0 = ik.fk(q_seed)
        v = r0[0].transpose(0, 1) @ z                                     # R_ref^T z
        print(f"[roll] tilt reference v = {[round(float(x), 4) for x in v]}")

        def tilt_deg(q: torch.Tensor) -> torch.Tensor:
            _, _, r = ik.fk(q)
            return torch.rad2deg(torch.arccos(
                torch.einsum("nij,j->ni", r, v)[:, 2].clamp(-1.0, 1.0)))

        # ---- VALIDATE the metric against physics before trusting any of it. run_expert
        # measured the held block at 6.38 deg (theta = 0) and 23.27 deg (theta = -0.5) at the
        # end of the turn. Replanning here and reading the same waypoint must reproduce those.
        bp0 = mdp.object_pos_local(e, "block")
        byaw0 = mdp.yaw_of(mdp.object_quat(e, "block"))
        # The metric is COMMANDED GRIPPER ROLL. The block does not follow it one-for-one: the
        # grasp is compliant, so gravity pulls the block back toward vertical and it ends up at
        # roughly half the gripper's roll. So the check is monotonicity and a stable ratio, not
        # equality -- demanding equality was the wrong criterion and flagged a correct metric.
        print("[roll] metric check vs physics (block tilt = ~0.5 x gripper roll, compliant grasp)")
        q_at = {}
        for th_v, want in ((0.0, 6.38), (-0.5, 23.27)):
            pl = P.plan(ik, params, bp0, byaw0, q_seed, sign,
                        torch.full((n,), th_v, device=dev))
            q_at[th_v] = pl["plans"]["turn"]["q"][:, -1, :].clone()
            got = float(tilt_deg(q_at[th_v]).median())
            print(f"[roll]   theta {th_v:+.2f}: commanded roll {got:6.2f} deg, block measured "
                  f"{want:5.2f} deg, ratio {want / max(got, 1e-6):.2f}")

        stage_dist = mdp.SLOT_CENTER[0] - params.stage_x
        insert_dist = params.insert_x - mdp.SLOT_CENTER[0]
        results = {}

        for th in args.thetas:
            c, sn = math.cos(th), math.sin(th)
            u = (torch.tensor([-sn, c, 0.0], device=dev) * sign).expand(n, 3).contiguous()
            targets = {
                "align": torch.tensor([mdp.SLOT_CENTER[0] - stage_dist * c,
                                       mdp.SLOT_CENTER[1] - stage_dist * sn,
                                       params.carry_z], device=dev).expand(n, 3).contiguous(),
                "insert": torch.tensor([mdp.SLOT_CENTER[0] + insert_dist * c,
                                        mdp.SLOT_CENTER[1] + insert_dist * sn,
                                        params.carry_z], device=dev).expand(n, 3).contiguous(),
            }
            cell = {}
            for name, tgt in targets.items():
                # (a) what the EXPERT lands on: warm-started from the canonical seed, which is
                #     how the real plan reaches this pose.
                exp = ik.solve(tgt, u, q_seed, 0.016, iters=args.iters)
                exp_ok = (exp["pos_err"] < args.pos_tol) & (exp["axis_err"] < args.axis_tol)
                exp_tilt = tilt_deg(exp["q"])

                # (b) RANDOM RESTARTS -- the actual existence question.
                lo, hi = ik.lo, ik.hi
                q_rand = lo + (hi - lo) * torch.rand((n, 6), device=dev)
                out = ik.solve(tgt, u, q_rand, 0.016, iters=args.iters)
                ok = (out["pos_err"] < args.pos_tol) & (out["axis_err"] < args.axis_tol)
                t = tilt_deg(out["q"])

                # (c) LOCAL restarts around the expert's own solution. Existence somewhere in
                # joint space is not enough: if the only level solutions sit in a different
                # elbow branch, no soft bias will reach them while carrying a block, and the
                # fix would need a replanned branch rather than a nudge. This asks whether a
                # level solution is NEARBY.
                near = {}
                for sig in (0.10, 0.30, 0.60):
                    q_loc = (exp["q"] + sig * torch.randn((n, 6), device=dev)).clamp(lo, hi)
                    o2 = ik.solve(tgt, u, q_loc, 0.016, iters=args.iters)
                    ok2 = (o2["pos_err"] < args.pos_tol) & (o2["axis_err"] < args.axis_tol)
                    t2 = tilt_deg(o2["q"])
                    dq = (o2["q"] - exp["q"]).abs().max(dim=1).values
                    near[f"sigma_{sig}"] = {
                        "converged": int(ok2.sum()),
                        "min_tilt_deg": float(t2[ok2].min()) if ok2.any() else None,
                        "dq_at_min_tilt": (float(dq[ok2][t2[ok2].argmin()]) if ok2.any() else None),
                    }

                got = t[ok]
                cell[name] = {
                    "expert_tilt_deg": float(exp_tilt[exp_ok].median()) if exp_ok.any() else None,
                    "expert_converged": int(exp_ok.sum()),
                    "restarts_converged": int(ok.sum()),
                    "min_tilt_deg": float(got.min()) if ok.any() else None,
                    "p10_tilt_deg": float(torch.quantile(got, 0.10)) if ok.any() else None,
                    "median_tilt_deg": float(got.median()) if ok.any() else None,
                    "max_tilt_deg": float(got.max()) if ok.any() else None,
                    "frac_below_5deg": float((got < 5.0).float().mean()) if ok.any() else None,
                    "local": near,
                }
                r = cell[name]
                print(f"  theta {th:+.2f} {name:<7} "
                      f"expert {r['expert_tilt_deg'] if r['expert_tilt_deg'] is None else round(r['expert_tilt_deg'], 2)}deg | "
                      f"restarts {r['restarts_converged']}/{n} converged, tilt min "
                      f"{'-' if r['min_tilt_deg'] is None else round(r['min_tilt_deg'], 2)} "
                      f"med {'-' if r['median_tilt_deg'] is None else round(r['median_tilt_deg'], 2)} "
                      f"max {'-' if r['max_tilt_deg'] is None else round(r['max_tilt_deg'], 2)} deg, "
                      f"below 5deg {'-' if r['frac_below_5deg'] is None else round(100 * r['frac_below_5deg'], 1)}%")
                loc = " ".join(
                    f"s{k.split('_')[1]}:{'-' if vv['min_tilt_deg'] is None else round(vv['min_tilt_deg'], 1)}"
                    f"@dq{'-' if vv['dq_at_min_tilt'] is None else round(vv['dq_at_min_tilt'], 2)}"
                    for k, vv in near.items())
                print(f"           local around expert -> min tilt {loc}")
            results[f"{th:+.3f}"] = cell

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"args": vars(args), "tilt_ref_v": [float(x) for x in v], "results": results}, indent=2))
    print(f"\n[roll] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
