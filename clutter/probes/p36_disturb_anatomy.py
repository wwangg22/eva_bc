# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P36 -- the anatomy of the disturbance. WHEN, WHICH block, WHICH WAY, and HOW MUCH.

Under the 2 mm rule the expert scores 16.4 % and `distractor_disturbed` is the **only**
non-zero failure bucket -- 83.6 %, with `time_out`, `target_dropped` and `distractor_toppled`
all at 0.0 % (`15_STRICT_METRIC.md` §4). So the entire remaining task is this one mechanism,
and 56.9 points depend on understanding it before choosing a fix.

What is already known, and what is not
--------------------------------------
Known: the hazard concentrates in the `close` phase. Every hazard table since P17 reports
71-76 % there, and P22 localised it to the finger blades. Known geometrically, from
`target_axis()`'s docstring: the blade spans only +/-19.2 mm perpendicular to the opening
axis against neighbour faces at +/-27 mm -- **7.8 mm of margin** -- but reaches ~47 mm ALONG
the opening axis, so yaw-matching the jaw by the target's full 11.4 deg of spawn yaw swings a
blade corner **9.3 mm**, more than the margin.

**Not** known, and each answer selects a different fix:

* Does the 2 mm crossing happen during `close`, or during `descend` before the fingers even
  move, or during `carry` after the block is lifted? A hazard *rate* per phase counts
  episodes that were ever disturbed in that phase; it does not say where the threshold was
  first crossed, and the phases are not independent.
* Which block? If it is the **inner** pair (`distractor_1`, `distractor_2`, at +/-42 mm) it is
  the blade sweep. If the **outer** pair moves too, something much larger is wrong.
* Which direction? Motion along **y** (across the row) is a blade pushing sideways. Motion
  along **x** (toward or away from the robot) is the blade's 47 mm reach dragging the block
  fore-and-aft. These call for opposite fixes.
* Is it a shove (one large increment) or a drag (accumulating over many steps)?

Method
------
Run the shipping expert -- frozen pose, frozen chain, frozen approach, `--close 40
--holds-scale 0.25`, the exact configuration behind the 16.4 % -- through `env.step`, and
record every distractor's displacement at every step with its phase label.

**On `-Lenient-v0`, deliberately.** The strict env terminates at the moment of the crossing
and auto-resets the scene from inside `env.step` (R23), so it can report *that* the threshold
was crossed and nothing about what happened afterwards. The lenient variant is the same
physics with the termination off, so the whole trajectory is visible. The crossing step is
then computed offline, which gives exactly what the strict env would have terminated on plus
the rest of the story.

`schedule_utils.expand` / `approach_prefix` are imported from `act/`, not re-implemented, so
this instruments the manoeuvre that was measured.

Registered predictions (written before the run)
-----------------------------------------------
1. **> 80 % of first crossings are in `close`.** If `descend` or `approach` carries a
   meaningful share, the fix is not the close at all and P22's attribution is incomplete.
2. **The inner pair accounts for > 90 % of first crossings.** The outer pair sits 84 mm out,
   57 mm beyond the blade's perpendicular half-span.
3. **Displacement is predominantly along y.** The jaw opens along x and closes along x; the
   blade's *perpendicular* half-span is what fouls the neighbour, so the neighbour should be
   pushed sideways, away from the target.

Prediction 3 is the one I am least sure of, and the one that most changes the fix: if the
motion is along **x** instead, the mechanism is the 47 mm reach sweeping fore-and-aft as the
jaw closes, and the lever is jaw *clocking*, not grip height.

Usage
-----
    python -u clutter/probes/p36_disturb_anatomy.py --num_envs 128 \\
        --seeds 88000,88001,88002,88003,88004,88005 --headless \\
        --json clutter/runs/p36_disturb_anatomy.json
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/clutter"

parser = argparse.ArgumentParser(description="P36 -- where the disturbance comes from.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Lenient-v0",
                    help="lenient on purpose: the strict env terminates at the crossing and "
                         "auto-resets, hiding everything after it")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seeds", type=str, default="88000,88001,88002,88003,88004,88005")
parser.add_argument("--pose", type=str, default=f"{_ROOT}/expert/pose_p33.json")
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--close", type=int, default=40, help="the shipping value (P27)")
parser.add_argument("--holds-scale", type=float, default=0.25, help="the shipping value (P27)")
parser.add_argument("--approach-steps", type=int, default=80)
parser.add_argument("--json", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
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
sys.path.insert(0, os.path.join(_HERE, "..", "act"))
from _kin import Q_OPEN, _t  # noqa: E402
from clutter_expert import HOLDS, ClutterExpert  # noqa: E402
from schedule_utils import approach_prefix, expand  # noqa: E402

DIST = mdp_cl.DISTRACTOR_NAMES
TOL = mdp_cl.DISTURB_TOL


def spawn_xy(e) -> torch.Tensor:
    """(N, 4, 2) recorded spawn positions -- the reference `record_spawn_xy` wrote."""
    return e._clutter_spawn_xy


def cur_xy(e) -> torch.Tensor:
    org = e.scene.env_origins
    return torch.stack([(_t(e.scene[d].data.root_pos_w) - org)[:, :2] for d in DIST], dim=1)


def main() -> None:
    spec = json.load(open(args_cli.pose))
    seeds = [int(s) for s in args_cli.seeds.split(",")]
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    e = env.unwrapped
    env.reset()

    holds = {"close": args_cli.close}
    for k in ("settle", "predwell", "dwell", "release", "final"):
        holds[k] = max(8, int(round(HOLDS[k] * args_cli.holds_scale / 8)) * 8)
    ex = ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"], holds=holds,
                       chain=spec.get("chain"), approach=spec.get("approach"), verbose=True)
    K, n = ex.K, ex.K.n
    dev = K.dev

    print("\n" + "=" * 100)
    print("P36  ANATOMY OF THE DISTURBANCE -- when, which block, which way, how much")
    print("=" * 100)
    print(f"   task {args_cli.task}  (lenient: nothing terminates, so the whole story shows)")
    print(f"   {n} envs x {len(seeds)} seeds = {n * len(seeds)} episodes")
    print(f"   holds {holds}   DISTURB_TOL {TOL * 1e3:.1f} mm")
    print("   PREDICTIONS: (1) >80 % of first crossings in `close`, (2) >90 % on the inner")
    print("                pair, (3) displacement predominantly along y\n")

    # accumulators over all seeds
    cross_phase: dict[str, int] = {}
    cross_block = torch.zeros(4, dtype=torch.long)
    n_cross = n_clean = 0
    dx_at, dy_at = [], []          # signed components at the crossing, of the crossing block
    away_flags = []                # did that block move OUTWARD from the row centreline?
    step_of_cross, gap_cross, gap_clean = [], [], []
    phase_worst: dict[str, float] = {}   # summed max-increment attributed to each phase
    final_disp = []                       # (E, 4) end-of-episode displacement

    for s in seeds:
        env.reset(seed=s)
        chain = ex.adapt()
        steps = expand(ex, chain)
        pre = approach_prefix(K, ex.approach_qs, chain[0], ex.qs[0], args_cli.approach_steps)
        steps = pre + steps
        K.teleport_arm(K.q_arm0.unsqueeze(0).repeat(n, 1), Q_OPEN)

        sp = spawn_xy(e).clone()
        prev = torch.zeros(n, 4, device=dev)
        crossed = torch.zeros(n, dtype=torch.bool, device=dev)
        c_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        c_block = torch.zeros(n, dtype=torch.long, device=dev)
        c_vec = torch.zeros(n, 2, device=dev)
        c_spawn_y = torch.zeros(n, device=dev)
        # per-phase, the largest single-step increment of the worst block, summed over envs
        inc_by_phase: dict[str, torch.Tensor] = {}

        for t, (phase, q, close) in enumerate(steps):
            env.step(K.act(q, close))   # the collector's own encoder, not a copy of it
            d = (cur_xy(e) - sp).norm(dim=-1)             # (n, 4)
            inc = (d - prev).max(dim=1).values.clamp(min=0.0)
            inc_by_phase[phase] = inc_by_phase.get(
                phase, torch.zeros(n, device=dev)) + inc
            worst, wi = d.max(dim=1)
            newly = (worst > TOL) & ~crossed
            if bool(newly.any()):
                c_step[newly] = t
                c_block[newly] = wi[newly]
                ar = torch.arange(n, device=dev)
                v = (cur_xy(e) - sp)[ar, wi]
                c_vec[newly] = v[newly]
                c_spawn_y[newly] = sp[ar, wi, 1][newly]
                crossed |= newly
                for k in torch.nonzero(newly).flatten().tolist():
                    cross_phase[phase] = cross_phase.get(phase, 0) + 1
            prev = d

        n_cross += int(crossed.sum())
        n_clean += int((~crossed).sum())
        cross_block += torch.bincount(c_block[crossed].cpu(), minlength=4)
        dx_at += c_vec[crossed, 0].tolist()
        dy_at += c_vec[crossed, 1].tolist()
        away_flags += (c_vec[crossed, 1] * c_spawn_y[crossed] > 0).long().tolist()
        step_of_cross += c_step[crossed].tolist()
        final_disp.append(d.cpu())
        for k, v in inc_by_phase.items():
            phase_worst[k] = phase_worst.get(k, 0.0) + float(v.sum())
        # spawn tightness, split by outcome

        tgt_y = torch.zeros(n, device=dev)
        gg = ((sp[:, :, 1] - tgt_y.unsqueeze(1)).abs() - 2 * mdp_cl.CL_BLOCK_HALF[1]).min(dim=1).values
        gap_cross += gg[crossed].mul(1e3).tolist()
        gap_clean += gg[~crossed].mul(1e3).tolist()
        print(f"   seed {s}: crossed {int(crossed.sum()):3}/{n}  "
              f"({int(crossed.sum()) / n:5.1%})   "
              f"median crossing step {int(c_step[crossed].median()) if bool(crossed.any()) else -1}")

    tot = n_cross + n_clean
    print("\n" + "-" * 100)
    print(f"   DISTURBED {n_cross}/{tot} = {n_cross / tot:.1%}   CLEAN {n_clean / tot:.1%}")

    print("\n   (1) WHICH PHASE crosses 2 mm first")
    for k, v in sorted(cross_phase.items(), key=lambda kv: -kv[1]):
        print(f"        {k:<10} {v:5}  {v / max(n_cross, 1):6.1%}")

    print("\n   (2) WHICH BLOCK crosses first   (inner = d1,d2 at +/-42 mm; outer = d0,d3 at +/-84)")
    for i in range(4):
        tag = "outer" if i in (0, 3) else "INNER"
        print(f"        {DIST[i]:<14} {tag}  {int(cross_block[i]):5}  "
              f"{int(cross_block[i]) / max(n_cross, 1):6.1%}")
    inner = int(cross_block[1] + cross_block[2])
    print(f"        inner pair total: {inner / max(n_cross, 1):.1%}")

    dxa = torch.tensor(dx_at).abs() if dx_at else torch.zeros(1)
    dya = torch.tensor(dy_at).abs() if dy_at else torch.zeros(1)
    print("\n   (3) WHICH DIRECTION, at the moment of crossing  [mm]")
    print(f"        |dx| (toward/away from robot)  median {float(dxa.median()) * 1e3:6.3f}  "
          f"mean {float(dxa.mean()) * 1e3:6.3f}")
    print(f"        |dy| (across the row)          median {float(dya.median()) * 1e3:6.3f}  "
          f"mean {float(dya.mean()) * 1e3:6.3f}")
    share_y = float((dya > dxa).float().mean())
    print(f"        y-dominant in {share_y:.1%} of crossings")
    # "away" = the block moved further from the row centreline, i.e. a blade pushed it
    # outward. "toward" = it was dragged INTO the gap the target came out of, which is a
    # different mechanism and would point at the lift rather than the close.
    if away_flags:
        af = torch.tensor(away_flags, dtype=torch.float)
        print(f"        pushed AWAY from the row centre {float(af.mean()):.1%}   "
              f"dragged TOWARD it {1 - float(af.mean()):.1%}")

    print("\n   (4) HOW MUCH each phase contributes  [total mm of worst-block motion]")
    for k, v in sorted(phase_worst.items(), key=lambda kv: -kv[1]):
        print(f"        {k:<10} {v * 1e3:10.1f} mm   {v / max(sum(phase_worst.values()), 1e-9):6.1%}")

    if gap_cross and gap_clean:
        gc = torch.tensor(gap_cross)
        gk = torch.tensor(gap_clean)
        print("\n   (5) DOES SPAWN TIGHTNESS PREDICT IT?  min free gap at spawn [mm]")
        print(f"        disturbed  median {float(gc.median()):6.2f}   mean {float(gc.mean()):6.2f}")
        print(f"        clean      median {float(gk.median()):6.2f}   mean {float(gk.mean()):6.2f}")
        print(f"        difference {float(gk.mean() - gc.mean()):+.2f} mm "
              f"(clean spawns wider by this much)")

    sc = torch.tensor(step_of_cross, dtype=torch.float) if step_of_cross else torch.zeros(1)
    print(f"\n   crossing step: p10 {float(sc.quantile(0.1)):.0f}  median "
          f"{float(sc.median()):.0f}  p90 {float(sc.quantile(0.9)):.0f}   "
          f"(approach ends at {len(pre)}, schedule total {len(steps)})")
    print("=" * 100 + "\n")

    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump({
                "task": args_cli.task, "seeds": seeds, "n_envs": n, "tol_m": TOL,
                "holds": holds, "n_disturbed": n_cross, "n_total": tot,
                "cross_phase": cross_phase,
                "cross_block": {DIST[i]: int(cross_block[i]) for i in range(4)},
                "dx_at_cross_m": dx_at, "dy_at_cross_m": dy_at,
                "step_of_cross": step_of_cross,
                "phase_worst_m": phase_worst,
                "gap_cross_mm": gap_cross, "gap_clean_mm": gap_clean, "away_flags": away_flags,
                "n_approach_steps": len(pre), "n_total_steps": len(steps),
            }, f)
        print(f"   wrote {args_cli.json}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
