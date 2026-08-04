# Copyright (c) 2026. Clutter-extraction effort, eva_bc/clutter.
"""Stage 2 -- drive the FROZEN expert through `env.step` and record demonstrations.

This file is both the demo generator and the instrument for **Gate 2**. That is deliberate:
the thing that has to be trusted is the recorded trajectory, so the equivalence check has to
run through the same code path that writes the data, not through a parallel implementation
that could agree with the expert while the recorder disagrees with both.

Why a port gate exists at all
-----------------------------
Every number in Stages 0 and 1 -- 25 %, 57.7 %, 72.1 % -- was measured with
`ClutterExpert.run_physics`, which sets joint drive targets and calls `sim.step()` directly.
It does that on purpose: `env.step` runs the whole MDP, and the `distractor_toppled`
termination resets the scene and re-spawns the blocks upright, so a topple erases the evidence
of itself (`07_STAGE0_RESULTS.md` §7.5, and P09's first run, where distractors appeared to
move *toward* a push directed away from them).

Demonstrations cannot be collected that way. A demo has to be a sequence of legal 7-D actions
submitted to `env.step`, starting from where `env.reset()` leaves the arm. So the manoeuvre has
to cross from one execution path to the other, and the crossing is measured rather than
assumed.

The four arms, each differing from its neighbour by ONE thing
------------------------------------------------------------
======  ==========================================================  ====================
`phys`  settle + `run_physics`                                      the 72.1 % reference
`tele`  settle + teleport to `chain[0]`, then `env.step`            MDP vs physics only
`tele0` teleport to `chain[0]`, then `env.step`, **no settle**      the settle's own effect
`appr`  from the reset pose, action-driven approach, then `env.step`  Gate 2b -- what ships
======  ==========================================================  ====================

`phys` -> `tele` is Gate 2a. `tele0` -> `appr` is Gate 2b. `tele` -> `tele0` exists because
the 30 free physics steps every probe has run since P01 are *not* available to a policy at
deployment, and nobody has ever measured what they are worth. Pass band for each adjacent
pair: **+/- 5 points**.

Bit-exactness is NOT the pass criterion and asking for it would be self-deception: P34
measured **43 of 512 episodes flipping between two runs of a bit-identical frozen chain**,
while the aggregate agreed to 0.2 points. ~8 % episode churn is the paired-comparison noise
floor of this stack.

The four things that were predicted to bite (`HANDOFF.md` §10.1a Step 1)
------------------------------------------------------------------------
1. **Action encoding.** `JointPositionActionCfg(scale=0.5, use_default_offset=True)` means
   `q_target = q_default + 0.5*a`, so `a = 2*(q_desired - q_default)` and the expert's
   `joint1` command lands near `|a| = 1.57`. Nothing clips it here -- `cfg.clip is None`, so
   `JointAction` skips its clamp entirely -- but rl_games' `clip_actions` would, which is why
   `max |a|` is measured, printed and stored rather than trusted.
2. **The gripper is binary.** `BinaryJointPositionAction.process_actions` is
   `where(a[6] > 0, open, close)`; there is no intermediate aperture and no rate limit. Only
   the two states are ever emitted. A demo containing a ramp teaches an action the policy
   cannot submit (P19/P20).
3. **`run_physics` teleports; `env.step` cannot.** Hence the `appr` arm, whose approach is
   FK-audited for keep-out penetration and floor clearance before it is executed.
4. **Observations are recorded BEFORE the step**, and the action recorded is the one actually
   submitted -- not the commanded joint target it decodes to.

Determinism
-----------
The expert is built once, from `expert/pose_p33.json`, with both `pose_q` and `chain` frozen,
so no CEM runs and every episode in the dataset is the same manoeuvre adapted to its spawn.
P34 is why: freezing the grasp pose alone does not freeze the chain, because `_dense` redraws
22 waypoints from a CEM and a seed cannot fix it (GPU reductions are not bit-stable; 1 320
iterations amplify 1e-7 into 0.109 rad).

Usage
-----
    # Gate 2 -- all four arms, paired on the same spawns, nothing written
    python -u clutter/act/collect_demos.py --num_envs 128 --arms phys,tele,tele0,appr

    # dataset -- one arm, recorded
    python -u clutter/act/collect_demos.py --num_envs 128 --arms appr --record appr \
        --seeds 30000,30001,30002,30003,30004,30005,30006,30007 --out runs/demos_v1.hdf5
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

_ROOT = "/home/eva/Desktop/isaacLab/eva_bc/clutter"

parser = argparse.ArgumentParser(description="Stage-2 demo collector and port gate.")
parser.add_argument("--task", type=str, default="Rebot-ClutterExtract-Play-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--grip_z", type=float, default=0.055)
parser.add_argument("--pose", type=str, default=f"{_ROOT}/expert/pose_p33.json")
parser.add_argument("--arms", type=str, default="phys,tele,tele0,appr",
                    help="comma-separated subset of phys,tele,tele0,appr")
parser.add_argument("--seeds", type=str, default="77000,77001,77002,77003",
                    help="spawn batch seeds; every arm sees the same ones (paired)")
parser.add_argument("--approach-steps", type=int, default=80,
                    help="env steps for the reset-pose -> chain[0] approach (`appr` arm). "
                         "80 is P29's C80: 73.0 %% vs a 74.2 %% teleport baseline; C40 gave "
                         "71.5 %%, so slower is mildly better and 80 is what shipped")
parser.add_argument("--settle", type=int, default=30,
                    help="free physics steps after reset, for the `phys`/`tele` arms only")
parser.add_argument("--record", type=str, default="",
                    help="arm whose episodes are written to --out; empty = record nothing")
parser.add_argument("--keep-failures", action="store_true",
                    help="write failed episodes too (success attr is False; the dataset "
                         "filter drops them). Default: write them, for the taxonomy")
parser.add_argument("--out", type=str, default="", help="demo HDF5 path")
parser.add_argument("--multimodal", action="store_true",
                    help="THE UNIMODALITY CONTROL. Re-solve the grasp pose (and its own "
                         "backward approach) for EVERY spawn batch, instead of using the "
                         "frozen one -- i.e. Stage-1 `plan()` behaviour, which P28 showed "
                         "produces six structurally different pose clusters from eight "
                         "draws. Produces a deliberately multi-modal dataset, to test "
                         "whether unimodality is what bought flow BC its 2.9-point gap. "
                         "Single-arm runs only: the arms would no longer be paired.")
parser.add_argument("--close", type=int, default=None,
                    help="close-hold duration in PHYSICS steps, overriding HOLDS['close'] "
                         "(560). P27 swept 560->40 with the frozen pose and found it FLAT: "
                         "72.9 %% to 71.1 %%, enclosure 100.0 %% at every duration, largest "
                         "single-step drop 2.3 %%. The fingers need at most 5 env steps and "
                         "70 were being spent.")
parser.add_argument("--holds-scale", type=float, default=None,
                    help="multiply settle/predwell/dwell/release/final by this (rounded to a "
                         "multiple of the decimation, min 8). P27 arm 2, with close=40: "
                         "1.00 -> 72.1 %%, 0.50 -> 76.0 %%, 0.25 -> 75.8 %%, i.e. shortening "
                         "the holds makes the EXPERT better by ~4 points while cutting the "
                         "demo from 394 to 247 env steps and its static fraction from 45.7 %% "
                         "to 13.4 %%.")
parser.add_argument("--screen", type=int, default=4,
                    help="candidate poses screened in-sim per re-solve (--multimodal only)")
parser.add_argument("--grip-open", type=float, default=None,
                    help="override the gripper's OPEN command, per finger [m]. Default 0.045, "
                         "i.e. 90 mm of separation to grasp a 36 mm block -- 27 mm of excess "
                         "travel per finger, every millimetre of which sweeps the blade "
                         "through the row. P38 measured the fouling reach at 33-39 mm from "
                         "the target centre against neighbour faces at 27 mm, and showed the "
                         "arm alone disturbs NOTHING (0.0 %): it is entirely the finger "
                         "closing motion.")
parser.add_argument("--yaw-gain", type=float, default=1.0,
                    help="how far to rotate the jaw toward the target's own yaw. 1.0 meets "
                         "the 36 mm faces squarely and is what every number before P37 used; "
                         "0.0 keeps the jaw on the nominal axis. P36 found the disturbance is "
                         "a FORE-AFT drag (|dx| 9.2x |dy|) starting at the first close step, "
                         "which is the signature of a yaw-swung blade corner intruding into "
                         "the neighbour's footprint and then travelling along the opening "
                         "axis -- 47*sin(11.4 deg) = 9.3 mm against 7.8 mm of margin")
parser.add_argument("--no-match-yaw", dest="match_yaw", action="store_false",
                    help="drop the orientation constraint entirely, rather than pinning it to "
                         "the nominal axis as --yaw-gain 0 does. Different thing: `refine` "
                         "then has no o_des at all and is free to pick any wrist roll")
parser.add_argument("--json", type=str, default=f"{_ROOT}/runs/gate2.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import reBot_RL.tasks  # noqa: F401,E402
from reBot_RL.tasks.manager_based.challenge.mdp import clutter as mdp_cl  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "probes"))
sys.path.insert(0, os.path.join(_HERE, "..", "expert"))
from _kin import Q_OPEN, _t  # noqa: E402
from clutter_expert import HOLDS, ClutterExpert  # noqa: E402

sys.path.insert(0, _HERE)
from schedule_utils import approach_prefix, expand  # noqa: E402

_HOLD_OVERRIDE: dict = {}
DIST = mdp_cl.DISTRACTOR_NAMES
HY = mdp_cl.CL_BLOCK_HALF[1]
OBS_DIM = 42
ACTION_DIM = 7


# --------------------------------------------------------------------------- the schedule
# `expand` and `approach_prefix` live in `schedule_utils.py` (imported above) so the
# diagnostic probes run the identical manoeuvre rather than a re-implementation of it.


def derive_approach(ex):
    """Solve the backward dense-Cartesian approach for an expert whose pose is NOT frozen.

    Only `--multimodal` needs this: the frozen approach in `pose_p33.json` ends exactly on the
    frozen `qs[0]`, and pairing it with any other pose would recreate the branch seam that
    scored 0.0 % (P29). A re-solved pose therefore needs a re-solved approach, and it has to be
    solved **backward from that pose's own `qs[0]`** for the same reason.

    Runs a CEM, so it is not reproducible (P34) -- which is exactly the property the
    multimodality control is trying to introduce, so here it is a feature.
    """
    K = ex.K
    q_home = K.q_arm0.unsqueeze(0).repeat(K.n, 1)
    tcp_home = K.fk(q_home)["tcp"][0].clone()
    tcp_0 = K.fk(ex.qs[0].unsqueeze(0).repeat(K.n, 1))["tcp"][0].clone()
    _, back_qs = ex._dense([tcp_0, tcp_home], ex.qs[0])
    return list(reversed(back_qs))


def audit(K, steps, boxes, margin):
    """FK the commanded path and report the worst keep-out penetration / floor clearance.

    Costs one batched `fk` per env step and no physics, so it is nearly free -- and it is the
    difference between finding out that the approach sweeps through `distractor_2` here,
    before any data is written, or finding it out from a demo set that quietly contains it.
    Leaves the arm teleported; the caller must restore.
    """
    pen, low = 0.0, float("inf")
    for _, q, _ in steps:
        g = K.fk(q)
        pen = max(pen, float(K.box_penetration(g["bodies"], boxes, margin).max()))
        low = min(low, float(g["low_z"].min()))
    return pen, low


# --------------------------------------------------------------------------- scene helpers
def settle_phys(e, k):
    for _ in range(k):
        e.sim.step()
        e.scene.update(e.physics_dt)


def free_gaps(e):
    """Min free gap [m] between the target and each neighbour, axis-aligned approximation.

    Recorded per episode so a later analysis can ask whether the failures concentrate in the
    tight spawns. Yaw is ignored on purpose -- this is a covariate, not a gate, and a rotated
    30 x 36 mm block has no single well-defined y-extent.
    """
    org = e.scene.env_origins
    ty = (_t(e.scene["target"].data.root_pos_w) - org)[:, 1]
    gaps = [((_t(e.scene[d].data.root_pos_w) - org)[:, 1] - ty).abs() - 2 * HY for d in DIST]
    return torch.stack(gaps, dim=1).min(dim=1).values


def fingerprint(e):
    """Scene identity hash: every block's planar position to 0.1 mm.

    Two arms that share a fingerprint saw the same spawns, so their difference is the arm and
    not the batch. P32 introduced this after a "paired" comparison turned out not to be.
    """
    org = e.scene.env_origins
    p = [(_t(e.scene[k].data.root_pos_w) - org)[:, :2] for k in ("target",) + DIST]
    return round(float(torch.stack(p).mul(1e4).round().sum()), 1)


# --------------------------------------------------------------------------- the rollouts
def run_env_step(env, ex, steps, record: bool):
    """Submit `steps` as 7-D actions through `env.step`. Returns per-env outcome + tape.

    Terminations are the reason this is not a simple loop. `distractor_toppled` and
    `block_dropped` both fire mid-episode, and Isaac Lab auto-resets those envs at the top of
    the next step -- so everything recorded after a done is a *different episode's* spawn.
    `done_at` freezes each env at its first done and the tape is truncated there.

    That auto-reset also breaks the end-of-episode scoring that every physics-only probe uses.
    `ex.score()` reads the scene, and a toppled env's scene has already been re-spawned
    upright, so it reports `topple = False` -- the failure erases itself, exactly the trap
    `run_physics` exists to avoid. The taxonomy therefore comes from the termination manager
    at the step the env actually died: `_term_dones` is written by `compute()` every step and
    `reset()` does not clear it, so reading it straight after `env.step` gives which term
    fired *this* step, before the re-spawn.
    """
    e, K, n = ex.e, ex.K, ex.K.n
    dev = K.dev
    T = len(steps)
    obs_tape = np.zeros((T, n, OBS_DIM), dtype=np.float32) if record else None
    act_tape = np.zeros((T, n, ACTION_DIM), dtype=np.float32) if record else None
    done_at = torch.full((n,), -1, dtype=torch.long, device=dev)
    tnames = list(e.termination_manager.active_terms)
    why = torch.zeros((n, len(tnames)), dtype=torch.bool, device=dev)
    # Latched success, so this arm can be compared to `eval_flow.py` on the SAME footing.
    # `ex.score()` reads the scene once, at the end of the schedule; the policy evaluator
    # latches `target_at_goal` at every step of a 700-step episode. Those are different
    # questions, and the difference runs in the expert's disfavour -- a block that is in the
    # goal circle at step 400 and rolls 2 mm out of it by step 472 counts for the policy and
    # not for the expert. Both are reported; the latched pair is the honest comparison.
    latched = torch.zeros(n, dtype=torch.bool, device=dev)
    # worst planar displacement of any NON-target block from spawn; see eval_flow.py -- the
    # benchmark's success predicate never looks at this.
    disturb = torch.zeros(n, device=dev)
    gap_stall = None
    max_abs_a = 0.0

    obs = e.observation_manager.compute()["policy"]
    for t, (phase, q, close) in enumerate(steps):
        a = K.act(q, close)
        max_abs_a = max(max_abs_a, float(a[:, :6].abs().max()))
        if record:
            obs_tape[t] = obs.detach().cpu().numpy()
            act_tape[t] = a.detach().cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(a)
        obs = obs["policy"] if isinstance(obs, dict) else obs
        latched |= mdp_cl.target_at_goal(e) & (done_at < 0)
        if hasattr(e, "_clutter_spawn_xy"):
            cur = torch.stack([mdp_cl.common.object_pos_local(e, nm)[:, :2] for nm in DIST], dim=1)
            dd = (cur - e._clutter_spawn_xy).norm(dim=-1).max(dim=1).values
            disturb = torch.where((done_at < 0) & ~(terminated | truncated),
                                  torch.maximum(disturb, dd), disturb)
        newly = (terminated | truncated) & (done_at < 0)
        if bool(newly.any()):
            done_at[newly] = t
            for k, nm in enumerate(tnames):
                why[:, k] |= newly & e.termination_manager.get_term(nm)
        if phase == "close" and (t + 1 == T or steps[t + 1][0] != "close"):
            gap_stall = K.gap().clone()

    alive = done_at < 0
    sc = ex.score(gap_stall if gap_stall is not None else K.gap())
    sc = {k: v & alive for k, v in sc.items()}
    # a topple that terminated the episode is still a topple, whatever the re-spawned scene says
    i_top = tnames.index("distractor_toppled") if "distractor_toppled" in tnames else None
    if i_top is not None:
        sc["topple"] = sc["topple"] | why[:, i_top]
    return {"score": sc, "done_at": done_at, "max_abs_a": max_abs_a,
            "latched": float(latched.float().mean()),
            "disturb_mm": (disturb * 1000).tolist(),
            "latched_mask": latched.tolist(),
            "why": {nm: float(why[:, k].float().mean()) for k, nm in enumerate(tnames)},
            "obs": obs_tape, "act": act_tape, "T": T}


def run_arm(env, ex, arm, seed, spec, recorder):
    """One (arm, spawn batch) cell. Every arm resets the same way, so the spawns are paired."""
    e, K = ex.e, ex.K
    env.reset(seed=seed)
    if arm in ("phys", "tele") or args_cli.multimodal:
        settle_phys(e, args_cli.settle)      # `_screen` needs a settled scene
    if args_cli.multimodal:
        ex = ClutterExpert(env, grip_z=args_cli.grip_z, screen=args_cli.screen,
                           holds=_HOLD_OVERRIDE or None, verbose=False)
        ex.approach_qs = derive_approach(ex)
        print(f"      [multimodal] re-solved: o_align {ex.pose['o_align']:.4f}, "
              f"wrist_y {float(ex.pose['wrist'][1]) * 1000:+.1f} mm, "
              f"j6 {float(ex.pose['q'][5]):+.3f}, screen "
              f"{ex.pose.get('screen_score', float('nan')):.1%}")
        K = ex.K

    # `adapt()` teleports the arm through `refine`'s finite-difference FK, so it must run
    # before recording and the arm must then be put back where the arm's protocol says.
    chain = ex.adapt(match_yaw=args_cli.match_yaw, yaw_gain=args_cli.yaw_gain)
    steps = expand(ex, chain)
    fp, gaps = fingerprint(e), free_gaps(e).clone()

    if arm == "appr":
        q_start = K.q_arm0.unsqueeze(0).repeat(K.n, 1)
        pre = approach_prefix(K, ex.approach_qs, chain[0], ex.qs[0], args_cli.approach_steps)
        pen, low = audit(K, pre, ex.boxes, ex.margin)
        steps = pre + steps
        K.teleport_arm(q_start, Q_OPEN)
    else:
        pen, low = float("nan"), float("nan")
        K.teleport_arm(chain[0], Q_OPEN)

    if arm == "phys":
        sc = ex.run_physics(chain)
        out = {"score": sc, "done_at": None, "max_abs_a": float("nan"), "T": len(steps),
               "why": {}, "latched": float("nan"), "disturb_mm": None, "latched_mask": None}
    else:
        out = run_env_step(env, ex, steps, record=(arm == args_cli.record))
        if out["obs"] is not None:
            recorder.add(out, seed, steps, gaps)

    sc = out["score"]
    cell = {"seed": seed, "fingerprint": fp, "T": out["T"],
            "encl": float(sc["held"].float().mean()),
            "at_goal": float(sc["at_goal"].float().mean()),
            "topple": float(sc["topple"].float().mean()),
            "success": float(sc["success"].float().mean()),
            "succ_mask": sc["success"].tolist(),
            "max_abs_a": out["max_abs_a"],
            "why": out["why"], "latched": out["latched"],
            "disturb_mm": out["disturb_mm"], "latched_mask": out["latched_mask"],
            "approach_pen_mm": pen * 1000 if pen == pen else None,
            "approach_low_z_mm": low * 1000 if low == low else None,
            "min_free_gap_mm": float(gaps.min()) * 1000}
    if out["done_at"] is not None:
        d = out["done_at"]
        cell["terminated_early"] = float((d >= 0).float().mean())
        cell["first_done_step"] = float(d[d >= 0].float().mean()) if (d >= 0).any() else None
    return cell


# --------------------------------------------------------------------------- the recorder
class Recorder:
    """Accumulates episodes and writes the `act/dataset.py` HDF5 contract.

        /data/demo_<i>/obs/policy      (T, 42) float32
        /data/demo_<i>/actions         (T,  7) float32
        /data/demo_<i>/train_mask      (T,)    uint8
        /data/demo_<i>.attrs["success"]  bool

    `train_mask` is written all-ones. It is the mechanism for censoring the ambiguous tail of
    the 70-step `close` hold (`09_STAGE2_BC_PLAN.md` N2) and the censor rides the same
    `action_is_pad` channel as end-of-episode padding, so a censored step contributes no
    gradient. Whether to use it is P27's call; the segment boundaries are stored per episode
    so the decision can be made without regenerating anything.
    """

    def __init__(self, path: str, keep_failures: bool):
        self.path, self.keep = path, keep_failures
        self.demos: list[dict] = []

    def add(self, out, seed, steps, gaps):
        if not self.path:
            return
        sc, d = out["score"], out["done_at"]
        n = sc["success"].shape[0]
        segs, cur, start = [], None, 0
        for t, (phase, _, _) in enumerate(steps):
            if phase != cur:
                if cur is not None:
                    segs.append([cur, start, t])
                cur, start = phase, t
        segs.append([cur, start, len(steps)])
        for i in range(n):
            ok = bool(sc["success"][i])
            if not ok and not self.keep:
                continue
            T = int(d[i]) + 1 if int(d[i]) >= 0 else out["T"]
            self.demos.append({
                "obs": out["obs"][:T, i], "act": out["act"][:T, i],
                "success": ok, "seed": int(seed), "env_index": i,
                "at_goal": bool(sc["at_goal"][i]), "topple": bool(sc["topple"][i]),
                "held": bool(sc["held"][i]), "done_at": int(d[i]),
                "min_free_gap_mm": float(gaps[i]) * 1000,
                "segments": json.dumps(segs),
            })

    def write(self):
        if not self.path or not self.demos:
            return 0
        import h5py
        with h5py.File(self.path, "w") as f:
            data = f.create_group("data")
            for i, dm in enumerate(self.demos):
                g = data.create_group(f"demo_{i}")
                g.create_dataset("obs/policy", data=dm["obs"], compression="gzip")
                g.create_dataset("actions", data=dm["act"], compression="gzip")
                g.create_dataset("train_mask",
                                 data=np.ones(dm["obs"].shape[0], dtype=np.uint8))
                for k in ("success", "seed", "env_index", "at_goal", "topple", "held",
                          "done_at", "min_free_gap_mm", "segments"):
                    g.attrs[k] = dm[k]
                g.attrs["num_samples"] = dm["obs"].shape[0]
        return len(self.demos)


# --------------------------------------------------------------------------------- driver
def main() -> None:
    spec = json.load(open(args_cli.pose))
    arms = [a.strip() for a in args_cli.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args_cli.seeds.split(",")]
    for a in arms:
        assert a in ("phys", "tele", "tele0", "appr"), a
    assert not args_cli.record or args_cli.record in arms, "--record names an arm not run"
    assert not args_cli.record or args_cli.out, "--record needs --out"
    assert not args_cli.multimodal or len(arms) == 1, (
        "--multimodal re-solves a different pose per batch, so the arms stop being paired; "
        "run one arm at a time")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.grip_open is not None:
        env_cfg.actions.gripper_action.open_command_expr = {
            "joint_left": args_cli.grip_open, "joint_right": args_cli.grip_open}
        print(f"   GRIPPER OPEN OVERRIDE: {args_cli.grip_open:.3f} per finger "
              f"({args_cli.grip_open * 2000:.0f} mm separation, default 90 mm)")
    env = gym.make(args_cli.task, cfg=env_cfg)
    e = env.unwrapped
    env.reset()

    print("\n" + "=" * 100)
    print("STAGE 2 -- env.step PORT GATE / DEMO COLLECTOR")
    print("=" * 100)
    print(f"   task {args_cli.task}   num_envs {args_cli.num_envs}   "
          f"episode_length {e.max_episode_length} env steps")
    print(f"   pose {spec['name']}  ({spec['source']})")
    print(f"   arms {arms}   seeds {seeds}")

    # No action term may clip: the expert commands |a| ~ 1.57 and a silent clamp would turn
    # the port gate into a measurement of the clamp.
    for name, term in e.action_manager._terms.items():
        assert getattr(term.cfg, "clip", None) is None, f"action term {name} clips: {term.cfg.clip}"
    print("   action manager: no clip on any term  (checked, not assumed)")

    holds = {}
    if args_cli.close is not None:
        holds["close"] = args_cli.close
    if args_cli.holds_scale is not None:
        for k in ("settle", "predwell", "dwell", "release", "final"):
            holds[k] = max(8, int(round(HOLDS[k] * args_cli.holds_scale / 8)) * 8)
    if holds:
        _HOLD_OVERRIDE.update(holds)
        print(f"   HOLD OVERRIDE (P27): {holds}")

    ex = ClutterExpert(env, grip_z=args_cli.grip_z, pose_q=spec["q"], holds=holds or None,
                       chain=spec.get("chain"), approach=spec.get("approach"), verbose=True)
    if args_cli.multimodal:
        print("   *** MULTIMODAL CONTROL: the frozen pose is IGNORED. A fresh pose and a")
        print("   *** fresh backward approach are solved for every spawn batch.")
    if "appr" in arms and not args_cli.multimodal:
        assert ex.approach_qs is not None, (
            f"{args_cli.pose} has no frozen `approach`. P29 measured the improvised "
            f"alternatives at 5.5 % (joint lerp) and 0.0 % (forward Cartesian); do not "
            f"substitute one here.")
    if "appr" in arms and not args_cli.multimodal:
        ap = spec["approach"]
        print(f"   approach: {len(ap['qs'])} frozen waypoints over {args_cli.approach_steps} "
              f"env steps, seam {ap['measured']['seam_rad']:.4f} rad "
              f"({ap['measured']['c80_success']:.1%} vs {ap['measured']['teleport_baseline']:.1%} "
              f"teleport, P29)")
    est = ex.env_steps(ex.adapt(match_yaw=args_cli.match_yaw, yaw_gain=args_cli.yaw_gain))
    print(f"   schedule: {est['TOTAL']} env steps, {est['STATIC']} of them a held pose "
          f"({est['STATIC'] / est['TOTAL']:.1%})")
    print("   " + "  ".join(f"{k}={v}" for k, v in est.items() if k not in ("TOTAL", "STATIC")))

    rec = Recorder(args_cli.out, keep_failures=True)
    out: dict[str, dict] = {}
    for arm in arms:
        print(f"\n   --- arm {arm} " + "-" * 70)
        cells = []
        for s in seeds:
            c = run_arm(env, ex, arm, s, spec, rec)
            cells.append(c)
            extra = ""
            if c["max_abs_a"] == c["max_abs_a"]:
                extra += f" | max|a| {c['max_abs_a']:.3f}"
            if c.get("terminated_early") is not None:
                extra += f" | early-term {c['terminated_early']:5.1%}"
            if c["approach_pen_mm"] is not None:
                extra += (f" | appr pen {c['approach_pen_mm']:.1f} mm"
                          f" low_z {c['approach_low_z_mm']:.0f} mm")
            if c["latched"] == c["latched"]:
                extra += f" | latched {c['latched']:6.1%}"
            print(f"      seed {s}: encl {c['encl']:6.1%} | goal {c['at_goal']:6.1%} | "
                  f"topple {c['topple']:6.1%} | SUCCESS {c['success']:6.1%}{extra}")
        m = sum(c["success"] for c in cells) / len(cells)
        out[arm] = {"cells": cells, "mean": m, "n": len(seeds) * args_cli.num_envs}
        dm = [x for c in cells if c["disturb_mm"] for x in c["disturb_mm"]]
        lm = [x for c in cells if c["latched_mask"] for x in c["latched_mask"]]
        if dm:
            import statistics
            ok = [d for d, k in zip(dm, lm) if k]
            print(f"      neighbour displacement among SUCCESSES: median "
                  f"{statistics.median(ok):.2f} mm, max {max(ok):.2f} mm")
            for t in (2.0, 5.0, 10.0):
                k = sum(1 for d, g in zip(dm, lm) if g and d < t) / len(dm)
                print(f"         STRICT success < {t:4.0f} mm: {k:6.1%}")
            out[arm]["strict"] = {f"{int(t)}mm": sum(1 for d, g in zip(dm, lm) if g and d < t) / len(dm)
                                  for t in (2.0, 5.0, 10.0)}
        lat = [c["latched"] for c in cells if c["latched"] == c["latched"]]
        if lat:
            out[arm]["latched_mean"] = sum(lat) / len(lat)
            print(f"      {arm} MEAN {m:.1%} over {out[arm]['n']} episodes   "
                  f"(LATCHED {out[arm]['latched_mean']:.1%} -- the eval_flow.py protocol)")
        else:
            print(f"      {arm} MEAN {m:.1%} over {out[arm]['n']} episodes")
        if cells[0]["why"]:
            tax = {k: sum(c["why"][k] for c in cells) / len(cells) for k in cells[0]["why"]}
            print("      termination taxonomy: "
                  + ", ".join(f"{k} {v:.1%}" for k, v in tax.items()))
            out[arm]["taxonomy"] = tax

    # Pairing check: every arm must have seen the same spawns, seed for seed.
    #
    # Grouped by whether the arm settles, because the fingerprint is read AFTER the settle and
    # 30 free physics steps do move the blocks -- by well under 0.1 mm, but the hash rounds to
    # 0.1 mm and a coordinate sitting on a rounding boundary flips the sum by 1. Run 1 saw
    # exactly that on one seed of four (1601041 vs 1601040) and calling it a mismatch would be
    # wrong: it is the settle, which is the very thing the tele/tele0 arm measures.
    print("\n   pairing check (identical scene fingerprints within a settle group):")
    for settled in (True, False):
        grp = [a for a in arms if (a in ("phys", "tele")) == settled]
        if len(grp) < 2:
            continue
        for j, s in enumerate(seeds):
            fps = {arm: out[arm]["cells"][j]["fingerprint"] for arm in grp}
            ok = len(set(fps.values())) == 1
            print(f"      settle={settled} seed {s}: "
                  f"{'OK' if ok else '*** MISMATCH ***'}  {fps}")
    if any(a in arms for a in ("phys", "tele")) and any(a in arms for a in ("tele0", "appr")):
        a0 = "phys" if "phys" in arms else "tele"
        a1 = "tele0" if "tele0" in arms else "appr"
        d = [abs(out[a0]["cells"][j]["fingerprint"] - out[a1]["cells"][j]["fingerprint"])
             for j in range(len(seeds))]
        print(f"      settle vs no-settle hash drift: max {max(d):.0f} units of 0.1 mm "
              f"over {5 * args_cli.num_envs * 2} coordinates -- the settle itself")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    pairs = [("phys", "tele", "2a  MDP vs physics-only"),
             ("tele", "tele0", "    the 30-step free settle"),
             ("tele0", "appr", "2b  action-driven approach")]
    for a, b, label in pairs:
        if a in out and b in out:
            d = (out[b]["mean"] - out[a]["mean"]) * 100
            flips = sum(int(x != y) for c1, c2 in zip(out[a]["cells"], out[b]["cells"])
                        for x, y in zip(c1["succ_mask"], c2["succ_mask"]))
            n = out[a]["n"]
            print(f"   {label}: {a} {out[a]['mean']:6.1%} -> {b} {out[b]['mean']:6.1%}  "
                  f"delta {d:+5.2f} pts  {'PASS' if abs(d) <= 5 else '*** FAIL ***'}"
                  f"   episode flips {flips}/{n} ({flips / n:.1%}, noise floor ~8 %)")
    if "phys" in out:
        ref = spec["measured"]["p34_verification"]["quote"]
        print(f"\n   phys arm {out['phys']['mean']:.1%} vs P34's quoted {ref:.1%} "
              f"(delta {(out['phys']['mean'] - ref) * 100:+.2f} pts) -- the reference check")

    n_written = rec.write()
    if n_written:
        kept = sum(1 for d in rec.demos if d["success"])
        print(f"\n   wrote {n_written} episodes ({kept} successful, "
              f"{kept / n_written:.1%}) -> {args_cli.out}")

    with open(args_cli.json, "w") as f:
        json.dump({"args": vars(args_cli), "pose": spec["name"], "arms": out,
                   "schedule": est}, f)
    print(f"   wrote {args_cli.json}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
