#!/usr/bin/env python
"""Stage-3 gated planner-DAgger collector (PLAN.md section 3.2, HG-DAgger-style).

Per rollout (single env): the frozen flow policy (act/eval_act.py BatchedACTController)
drives the env; every policy step is checked against simple failure gates (miss / stall /
drop / timeout, GateMonitor below). On a trip the cuRobo expert takes over from the
CURRENT sim state (sync_world -> open + retreat home -> fetch_one per unplaced can,
mirroring run_expert_v1.Driver.episode()'s conventions incl. the lost-refetch path) and
finishes the task. Episodes where the policy places both cans before any gate trips are
counted as policy successes and NOT written.

Labels (Big Will's directive: train ONLY on recovery behavior, never the failure):
h5 layout is identical to the demo files (RebotDemoDataset-compatible); train_mask is
run_expert_v1.build_train_mask over the takeover's segments/outcomes (masks the expert's
own missed/lost sub-attempts), then HARD-ZEROED over [0, takeover_t) — every
policy-driven step is unsupervised, no exceptions. Successful takeovers ->
episode_kind "recovery_dagger" in --out; failed takeovers -> "<out>_failures.h5"
with success=False (Stage 5 material). Extra per-demo attrs: takeover_t, gate_reason.

Run (env_isaaclab6, GPU):
    python collect_dagger.py --ckpt ../runs/flow_nominal_v1/ckpt_final.pt \
        --out dagger_r1.h5 --seed 201

Structure note: run_expert_v1.py launches the AppLauncher at module scope, so it is
imported INSIDE main() (with a synthesized argv) and its AppLauncher is the one launch;
this module itself keeps all Isaac work in main() (eval_act.py's pattern) and stays
CPU-importable for gate/mask/h5 unit tests. The expert runner disables the env's
time_out termination (no auto-resets mid scripted control may re-randomize the scene);
--episode-length-s is still applied to the env cfg for parity with eval_act.py, but the
operative policy-phase backstop is GateMonitor's 900-step timeout gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.eval_act import FLUSH_Z_ABOVE, FLUSH_Z_DROP, BatchedACTController, load_checkpoint  # noqa: F401

# gate thresholds (PLAN.md section 3.2; drop matches eval_act.py's flush trigger)
MISS_STEPS = 45  # consecutive closed-gripper steps with no can RISE -> air-close.
# Must exceed close(21) + the ~10 lift steps a real grasp needs to raise the can
# 2 cm (v1 shakedown: 25 steps tripped on every successful mid-lift grasp).
MISS_RISE = 0.02  # "lifted": some can rose this far above its z at streak start
STALL_STEPS = 450  # steps without an increase in placed-can count (shakedown: healthy
# rollouts still make first-can progress past t=500; 300 preempted them)
TIMEOUT_STEPS = 900  # episode-step backstop
DROP_Z = FLUSH_Z_DROP  # 0.02 m one-step z fall ...
DROP_ABOVE = FLUSH_Z_ABOVE  # ... from above 0.05 m (settling into the basket never trips)


class GateMonitor:
    """Per-policy-step failure gates. update() takes the executed grip command, both
    cans' env-local z, the placed-can count, and the post-step index; returns the
    gate_reason string on the first trip, else None. Pure python (CPU-tested)."""

    def __init__(self, miss_steps=MISS_STEPS, miss_rise=MISS_RISE, stall_steps=STALL_STEPS,
                 drop_z=DROP_Z, drop_above=DROP_ABOVE, timeout_steps=TIMEOUT_STEPS):
        self.miss_steps = miss_steps
        self.miss_rise = miss_rise
        self.stall_steps = stall_steps
        self.drop_z = drop_z
        self.drop_above = drop_above
        self.timeout_steps = timeout_steps
        self.close_streak = 0
        self.window_max_z = float("-inf")
        self.streak_base_z = None
        self.best_placed = 0
        self.steps_since_progress = 0
        self.prev_zs = None

    def update(self, grip, can_zs, placed_count, step_idx, over_basket=None):
        # a release into the basket is a legitimate >drop_z fall — exempt cans over it
        over = over_basket if over_basket is not None else [False] * len(can_zs)
        dropped = self.prev_zs is not None and any(
            pz > self.drop_above and (pz - z) > self.drop_z and not ob
            for pz, z, ob in zip(self.prev_zs, can_zs, over)
        )
        self.prev_zs = [float(z) for z in can_zs]
        if grip < 0:
            if self.close_streak == 0:
                self.streak_base_z = max(can_zs)
            self.close_streak += 1
            self.window_max_z = max(self.window_max_z, max(can_zs))
        else:
            self.close_streak = 0
            self.window_max_z = float("-inf")
            self.streak_base_z = None
        if placed_count > self.best_placed:
            self.best_placed = placed_count
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        if (self.close_streak >= self.miss_steps
                and self.window_max_z - self.streak_base_z <= self.miss_rise):
            return "miss"
        if self.steps_since_progress >= self.stall_steps:
            return "stall"
        if dropped:
            return "drop"
        if step_idx > self.timeout_steps:
            return "timeout"
        return None


def write_demos_h5(path, demos):
    """run_expert_v1.main()'s h5 writer + the DAgger attrs (takeover_t, gate_reason).

    demos: [{"obs": (T,41), "act": (T,7), "train_mask": (T,), "meta": {...}}].
    """
    with h5py.File(str(path), "w") as f:
        grp = f.create_group("data")
        for i, d in enumerate(demos):
            g = grp.create_group(f"demo_{i}")
            g.create_dataset("obs/policy", data=d["obs"], compression="gzip")
            g.create_dataset("actions", data=d["act"], compression="gzip")
            g.create_dataset("train_mask", data=d["train_mask"])
            m = d["meta"]
            g.attrs["success"] = bool(m["success"])
            g.attrs["num_samples"] = len(d["act"])
            g.attrs["episode_kind"] = m["episode_kind"]
            g.attrs["perturb_steps"] = json.dumps(m.get("perturb_steps", []))
            g.attrs["segments"] = json.dumps(m["segments"])
            g.attrs["outcomes"] = json.dumps(m["outcomes"])
            g.attrs["takeover_t"] = int(m["takeover_t"])
            g.attrs["gate_reason"] = m["gate_reason"]
        grp.attrs["total"] = len(demos)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="flow checkpoint (act/train_flow.py .pt)")
    parser.add_argument("--rollouts", type=int, default=300, help="max policy rollouts")
    parser.add_argument("--target-takeovers", type=int, default=100,
                        help="stop early once this many successful takeovers are written")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", required=True, help="output h5 path (relative to this dir, like --record-h5)")
    parser.add_argument("--task", default="Rebot-PickPlace-Play-v1")
    parser.add_argument("--episode-length-s", type=float, default=30.0,
                        help="env horizon override for cfg parity with eval_act.py (the expert "
                             "Driver disables time_out; the 900-step timeout gate is the backstop)")
    args = parser.parse_args()

    # run_expert_v1 parses sys.argv and launches the AppLauncher at import time; hand it
    # an empty argv, import it (THE app launch), then set its args_cli fields directly.
    sys.argv = [sys.argv[0]]
    import run_expert_v1 as rex

    rex.args_cli.task = args.task
    rex.args_cli.seed = args.seed
    rex.args_cli.record_h5 = "<in-memory>"  # truthy: Driver records obs/act; h5 written here
    rex.args_cli.video = False
    rex.args_cli.perturb = False
    rex.args_cli.diversify = False

    # apply --episode-length-s before Driver.__init__ consumes the cfg (see docstring)
    orig_parse_env_cfg = rex.parse_env_cfg

    def parse_env_cfg_with_horizon(task, **kwargs):
        env_cfg = orig_parse_env_cfg(task, **kwargs)
        env_cfg.episode_length_s = args.episode_length_s
        return env_cfg

    rex.parse_env_cfg = parse_env_cfg_with_horizon

    device = "cuda:0"
    policy, stats, ckpt_cfg = load_checkpoint(Path(args.ckpt), device)
    controller = BatchedACTController(
        policy, stats, ckpt_cfg["n_action_steps"], ckpt_cfg["chunk_size"], device)

    CANS = ("object_a", "object_b")

    class DaggerDriver(rex.Driver):
        def step_policy_action(self, a7):
            """Step the env with the policy's 7-D action verbatim, recording the
            (pre-step obs, action) pair exactly like step_action."""
            a = torch.as_tensor(a7, device="cuda", dtype=torch.float32).view(1, 7)
            if self.h5_path is not None:
                self.ep_obs.append(self.last_obs)
                self.ep_act.append(a[0].cpu().numpy().astype("float32"))
            obs = self.env.step(a)[0]
            if self.h5_path is not None:
                self.last_obs = obs["policy"][0].cpu().numpy().astype("float32")
            self.step_idx += 1

        def _retreat_home(self, seg):
            self.mark("retreat", seg)
            self.sync_world()
            hres = self.expert.plan_home(self.q6())
            if hres is not None and bool(hres.success.any().item()):
                self.run_traj(rex.traj_to_qs(hres), +1)

        def dagger_episode(self, i):
            obs = self.env.reset()[0]
            self.last_obs = obs["policy"][0].cpu().numpy().astype("float32")
            self.ep_obs = []
            self.ep_act = []
            self.q_default = self.robot.data.default_joint_pos[0, :6].clone()
            self.step_idx = 0
            self.segments = []
            self.outcomes = {}
            self.perturb_log = []
            self.pending_event = None
            self.grip_override = 0
            controller.reset([0])
            self.mark("policy", seg=None)

            # -- policy phase --
            gate = GateMonitor()
            reason = None
            while reason is None:
                if all(self.placed(n) for n in CANS):
                    return {"result": "policy_success", "steps": self.step_idx}
                a = controller.act(torch.from_numpy(self.last_obs).view(1, -1))[0].cpu().numpy()
                self.step_policy_action(a)
                bxy = self.basket_xy()
                zs, over = [], []
                for n in CANS:
                    p = self.can_pos(n)
                    zs.append(float(p[2]))
                    over.append(float(np.linalg.norm(p[:2] - bxy)) < 0.10)
                placed_count = sum(self.placed(n) for n in CANS)
                reason = gate.update(float(a[6]), zs, placed_count, self.step_idx, over_basket=over)

            # -- expert takeover from the CURRENT state --
            takeover_t = self.step_idx
            self.mark("reopen", "takeover:r0")  # first act of recovery: release + settle
            self.outcomes["takeover:r0"] = {"outcome": "recovery", "gate_reason": reason}
            self.hold(15, +1)
            if all(self.placed(n) for n in CANS):  # e.g. drop gate on a can falling into the basket
                return {"result": "policy_success", "steps": self.step_idx, "late_gate": reason}
            m = {"plans": 0, "plan_fail": 0, "grasps": 0, "failed_grasps": 0,
                 "clean_grasps": 0, "close_disp": [], "place_plan_fail": 0}
            # Scripted retreat to Q_HOME before any planning: policy poses sit within
            # cuRobo's sphere margins of the table and/or cans (v5 ablation: 3/3 failed
            # takeovers were start-state collisions — with table, can, or both — while
            # the SAME scenes plan fine from Q_HOME). A blind joint-delta lift is not
            # enough; go up first (clear the immediate can), then interp to the proven
            # planning pose. Any can bumped by the sweep is re-synced before planning
            # and is exactly the kind of state the expert recovers from.
            q_now = np.array(self.q6())
            q_up = q_now + np.array([0.0, -0.35, -0.15, 0.0, 0.0, 0.0])
            self.run_traj(np.linspace(q_now, q_up, 30), +1)
            self.run_traj(np.linspace(q_up, np.array(rex.Q_HOME), 40), +1)
            self.hold(10, +1)
            self._retreat_home("takeover:recover")
            order = sorted((n for n in CANS if not self.placed(n)),
                           key=lambda n: float(np.linalg.norm(self.can_pos(n)[:2])))
            m["order"] = list(order)
            for name in order:
                r = self.fetch_one(name, m, tag="d0")
                if r == "lost":  # dropped mid-transport: recover by refetching (episode() convention)
                    self._retreat_home(f"{name}:recover")
                    self.fetch_one(name, m, tag="d1")
            success = all(self.placed(n) for n in CANS)

            T = len(self.ep_act)
            mask = rex.build_train_mask(T, self.segments, self.outcomes)
            mask[:takeover_t] = 0  # policy-driven steps are NEVER supervised
            m.update({
                "success": success,
                "placed": {n: bool(self.placed(n)) for n in CANS},
                "steps": self.step_idx,
                "segments": self.segments,
                "outcomes": self.outcomes,
                "perturb_steps": [],
                "episode_kind": "recovery_dagger",
                "takeover_t": takeover_t,
                "gate_reason": reason,
            })
            return {"result": "takeover", "gate_reason": reason, "takeover_t": takeover_t,
                    "success": success, "steps": self.step_idx,
                    "obs": np.stack(self.ep_obs), "act": np.stack(self.ep_act),
                    "train_mask": mask, "meta": m}

    drv = DaggerDriver()
    # cuRobo CUDA-graph capture order must match run_expert_v1: the process's FIRST
    # solver call must be a pose-goalset plan (plan_grasp), never plan_cspace — the
    # takeover's retreat-home otherwise captures cspace shapes first and the next
    # goalset plan dies with "CUDA graph reset is not available". Warm up + discard.
    drv.env.reset()
    warm = drv.can_pos(CANS[0])
    drv.expert.plan_grasp(drv.q6(), warm[:2], can_z=float(warm[2]))
    out_path = Path(rex.HERE) / args.out
    fail_path = out_path.with_name(out_path.stem + "_failures.h5")
    summary_path = out_path.with_suffix(".json")

    demos, failures, episodes = [], [], []
    policy_successes = 0
    gate_counts = {}
    rollouts = 0
    for i in range(args.rollouts):
        rollouts += 1
        r = drv.dagger_episode(i)
        if r["result"] == "policy_success":
            policy_successes += 1
            print(f"[ep {i}] policy_success steps={r['steps']}"
                  + (f" (late gate: {r['late_gate']})" if "late_gate" in r else ""))
        else:
            gate_counts[r["gate_reason"]] = gate_counts.get(r["gate_reason"], 0) + 1
            (demos if r["success"] else failures).append(r)
            print(f"[ep {i}] gate={r['gate_reason']} takeover_t={r['takeover_t']} "
                  f"success={r['success']} steps={r['steps']}")
        episodes.append({k: r[k] for k in ("result", "gate_reason", "takeover_t", "success", "steps")
                         if k in r})
        if len(demos) >= args.target_takeovers:
            break

    write_demos_h5(out_path, demos)
    print(f"[h5] wrote {len(demos)} recovery_dagger demos -> {out_path}")
    if failures:
        write_demos_h5(fail_path, failures)
        print(f"[h5] wrote {len(failures)} failed takeovers -> {fail_path}")
    summary = {
        "ckpt": str(args.ckpt),
        "seed": args.seed,
        "rollouts": rollouts,
        "policy_successes": policy_successes,
        "takeovers_written": len(demos),
        "failures": len(failures),
        "gate_reasons": gate_counts,
    }
    print("SUMMARY:", json.dumps(summary, indent=2))
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "episodes": episodes}, f, indent=2)

    drv.env.close()
    rex.app.close()


if __name__ == "__main__":
    main()
