# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""P38 -- WHAT touches the neighbour? Two ablations that assume no geometry at all.

P36 established *when* (the first step of `close`), *which block* (the inner pair, 100 %),
and *which way* (fore-aft, |dx| = 9.2x |dy|). P37 then tested the mechanism I inferred from
that -- a yaw-swung blade corner intruding into the neighbour's footprint -- by sweeping the
jaw's yaw-matching to zero. It bought **+2.9 points** (16.4 -> 19.3 %) and left 80 % of
episodes still failing.

So the inferred mechanism is at best a minor contributor, and the inference was built on a
docstring's geometry numbers (blade +/-19.2 mm perpendicular, ~47 mm along the opening axis)
that have never been checked against the asset. Rather than refine a model I do not trust,
these two ablations measure the answer directly.

Ablation B -- the gripper never closes
--------------------------------------
Run the identical commanded arm trajectory with `close` forced False everywhere. The arm
goes to exactly the same places; the only thing removed is the finger motion.

* neighbour still disturbed  -> it is the ARM or the WRIST, and every finger-geometry idea
  (yaw gain, blade clocking, grip height, narrower blades) is aimed at the wrong body
* neighbour left alone       -> it is the fingers, and the search stays where it is

This is the control that should have been run before P37.

Ablation P -- widen the row until it stops
------------------------------------------
Sweep `ROW_PITCH` from the shipping 42 mm upward, with everything else identical, and find
the pitch at which the disturbance vanishes. That converts "something fouls the neighbour"
into a **number in millimetres for how far it reaches from the target's centre**, with no
geometry model at all -- and that number identifies the culprit by elimination:

    reach ~ 21 mm   the target block's own half-width (15 mm) plus jitter -- the TARGET is
                    being pushed sideways into its neighbour
    reach ~ 27 mm   the blade's perpendicular half-span as documented (19.2 mm) plus the
                    yaw swing -- the FINGER, as assumed
    reach > 35 mm   something much larger: the wrist, or a finger body far off its origin

Only a diagnostic. The shipping task keeps its 42 mm pitch and its 12 mm gap; `-Tight-v0`
already exists at 36 mm and this says nothing about changing either.

Registered predictions (before the run)
---------------------------------------
1. **Ablation B leaves the neighbour alone** (< 10 % disturbed vs 83.3 % baseline). If it
   does not, P22's eight-stage attribution to the finger blades is wrong.
2. **The disturbance falls below 20 % somewhere between 50 and 56 mm of pitch**, i.e. an
   effective reach of 25-28 mm, consistent with the documented blade.

Prediction 2 is a genuine guess. If the reach comes out near 21 mm the mechanism is the
target, not the finger, and every candidate fix on the list is wrong.

Usage
-----
    python -u clutter/probes/p38_disturb_ablation.py --num_envs 128 \\
        --seeds 88000,88001,88002 --headless --json clutter/runs/p38_ablation.json
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/clutter"

parser = argparse.ArgumentParser(description="P38 -- what actually touches the neighbour.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Lenient-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seeds", type=str, default="88000,88001,88002")
parser.add_argument("--pose", type=str, default=f"{_ROOT}/expert/pose_p33.json")
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--close", type=int, default=40)
parser.add_argument("--holds-scale", type=float, default=0.25)
parser.add_argument("--approach-steps", type=int, default=80)
parser.add_argument("--yaw-gain", type=float, default=1.0)
parser.add_argument("--pitches", type=str, default="42,48,54,60,70",
                    help="row pitch sweep [mm]; 42 is the shipping value")
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
_H = mdp_cl.CL_BLOCK_HALF
ROW_X = 0.250


def cur_xy(e) -> torch.Tensor:
    org = e.scene.env_origins
    return torch.stack([(_t(e.scene[d].data.root_pos_w) - org)[:, :2] for d in DIST], dim=1)


def build(task: str, pitch_mm: float, n_envs: int):
    """Env at a given row pitch. 42 mm reproduces the shipping scene exactly."""
    cfg = parse_env_cfg(task, device=args_cli.device, num_envs=n_envs)
    if abs(pitch_mm - 42.0) > 1e-9:
        p = pitch_mm / 1000.0
        for i, s in enumerate((-2, -1, 1, 2)):
            getattr(cfg.scene, f"distractor_{i}").init_state.pos = [ROW_X, s * p, _H[2]]
    return gym.make(task, cfg=cfg)


def rollout(env, ex, seeds, force_open: bool, upto: str | None):
    """Run the manoeuvre; return the fraction of episodes whose worst neighbour passes TOL.

    `force_open` submits `close=False` at every step -- the arm goes to identical places and
    only the finger motion is removed. `upto` stops after the named phase, so the ablations
    are not paying for a 114-step carry that cannot happen without a grip.
    """
    K, n = ex.K, ex.K.n
    dev = K.dev
    tot_cross = tot = 0
    worst_all = []
    for s in seeds:
        env.reset(seed=s)
        chain = ex.adapt(yaw_gain=args_cli.yaw_gain)
        steps = expand(ex, chain)
        pre = approach_prefix(K, ex.approach_qs, chain[0], ex.qs[0], args_cli.approach_steps)
        steps = pre + steps
        if upto:
            last = max(i for i, (ph, _, _) in enumerate(steps) if ph == upto)
            steps = steps[:last + 1]
        K.teleport_arm(K.q_arm0.unsqueeze(0).repeat(n, 1), Q_OPEN)
        sp = e_spawn(env).clone()
        worst = torch.zeros(n, device=dev)
        for phase, q, close in steps:
            env.step(K.act(q, False if force_open else close))
            worst = torch.maximum(worst, (cur_xy(env.unwrapped) - sp).norm(dim=-1).max(dim=1).values)
        tot_cross += int((worst > TOL).sum())
        tot += n
        worst_all.append(worst.cpu())
    w = torch.cat(worst_all)
    return tot_cross / tot, float(w.median()) * 1e3, float(w.quantile(0.9)) * 1e3


def e_spawn(env):
    return env.unwrapped._clutter_spawn_xy


def make_expert(env):
    spec = json.load(open(args_cli.pose))
    holds = {"close": args_cli.close}
    for k in ("settle", "predwell", "dwell", "release", "final"):
        holds[k] = max(8, int(round(HOLDS[k] * args_cli.holds_scale / 8)) * 8)
    return ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"], holds=holds,
                         chain=spec.get("chain"), approach=spec.get("approach"), verbose=False)


def main() -> None:
    seeds = [int(s) for s in args_cli.seeds.split(",")]
    pitches = [float(p) for p in args_cli.pitches.split(",")]
    out: dict = {"seeds": seeds, "tol_m": TOL, "yaw_gain": args_cli.yaw_gain}

    print("\n" + "=" * 100)
    print("P38  WHAT TOUCHES THE NEIGHBOUR -- two assumption-free ablations")
    print("=" * 100)
    print(f"   {args_cli.num_envs} envs x {len(seeds)} seeds = {args_cli.num_envs * len(seeds)}"
          f" episodes per cell   TOL {TOL * 1e3:.1f} mm   yaw_gain {args_cli.yaw_gain}")
    print("   PREDICTIONS: (B) forcing the gripper open leaves the neighbour alone (<10 %)")
    print("                (P) disturbance drops below 20 % at 50-56 mm pitch\n")

    # ---- Ablation B: same arm trajectory, fingers never close --------------------
    env = build(args_cli.task, 42.0, args_cli.num_envs)
    env.reset()
    ex = make_expert(env)
    print("   --- ablation B: does the ARM alone disturb the row? ------------------")
    for tag, force_open in (("close as normal", False), ("gripper FORCED OPEN", True)):
        r, med, p90 = rollout(env, ex, seeds, force_open, upto="close")
        out[f"B_{'open' if force_open else 'normal'}"] = {"rate": r, "median_mm": med, "p90_mm": p90}
        print(f"      {tag:<22} disturbed {r:6.1%}   worst displacement median {med:6.3f} mm"
              f"   p90 {p90:7.3f} mm")
    b_n = out["B_normal"]["rate"]
    b_o = out["B_open"]["rate"]
    print(f"      -> the fingers account for {b_n - b_o:+.1%} of the {b_n:.1%}; "
          f"the arm alone for {b_o:.1%}")
    env.close()

    # ---- Ablation P: widen the row until it stops -------------------------------
    print("\n   --- ablation P: how far does the culprit REACH? ---------------------")
    print("      pitch  free gap   disturbed   median    p90       reach from target centre")
    rows = []
    for p in pitches:
        env = build(args_cli.task, p, args_cli.num_envs)
        env.reset()
        ex = make_expert(env)
        r, med, p90 = rollout(env, ex, seeds, False, upto="close")
        gap = p - 2 * _H[1] * 1000
        inner_face = p - _H[1] * 1000        # nominal y of the neighbour's inner face
        rows.append({"pitch_mm": p, "gap_mm": gap, "rate": r, "median_mm": med, "p90_mm": p90,
                     "inner_face_mm": inner_face})
        print(f"      {p:5.0f}  {gap:7.1f}   {r:8.1%}   {med:7.3f}  {p90:8.3f}"
              f"        <= {inner_face:.1f} mm")
        env.close()
    out["P_pitch"] = rows

    below = [r for r in rows if r["rate"] < 0.20]
    if below:
        first = min(below, key=lambda r: r["pitch_mm"])
        print(f"\n      -> drops below 20 % at pitch {first['pitch_mm']:.0f} mm, so whatever "
              f"fouls the row reaches\n         roughly {first['inner_face_mm']:.0f} mm from "
              f"the target centre.")
    else:
        print(f"\n      -> STILL above 20 % at {max(p for p in pitches):.0f} mm pitch. The "
              f"culprit reaches\n         further than {max(rows, key=lambda r: r['pitch_mm'])['inner_face_mm']:.0f} mm "
              f"-- larger than any finger geometry in the docs.")
    print("=" * 100 + "\n")

    if args_cli.json:
        with open(args_cli.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"   wrote {args_cli.json}")


if __name__ == "__main__":
    main()
    simulation_app.close()
