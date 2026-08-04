#!/usr/bin/env python
"""Batched sim evaluation for the vendored ACT policy (PLAN.md sections 2.3 / 4.2).

Rolls a trained ACT checkpoint in Rebot-PrecisionSlot-v0 across N parallel envs with
per-env action queues (receding horizon, temporal ensembling OFF) and the section 4.2
privileged flush trigger: on a detected can discontinuity (teleport-size position jump,
or a fast z drop while the can is above the table) the env's queue is cleared so the
next control step re-predicts from fresh observations.

Env terminations vs the expert runner (expert/run_expert_v1.py): the runner disables
BOTH time_out and object_dropping for its scripted control. Here time_out is KEPT
(episodes must end on the fixed horizon) and object_dropping is DISABLED to match how
the demos were generated (a dropped can does not re-randomize the scene mid-episode;
the fixed horizon ends the episode instead).

Success predicate: ``slot_mdp.placed_mask(env)`` = the env's ``is_inserted`` **AND** the block
actually seated on the slot floor at z = 55 +/- 6 mm. The seat guard is not optional. The env's
bare predicate bounds block height only from below (``z > SLOT_FLOOR_Z - 0.005``, there to
catch a drop) and therefore also passes for a block resting on the 30 mm wall tops or still
dangling in a closed gripper: during Stage A one probe cell scored 93.8 % at a mean lateral
error of 13.28 mm, geometrically impossible for a 15 mm half-width block in a 16.5 mm channel.
Every record also carries ``inserted_raw`` so the gap between the two stays visible.

Because Isaac Lab's ManagerBasedRLEnv resets done envs *before* computing the returned
observations, the scene state at the exact termination step is unreadable after ``env.step``;
success is therefore sampled one control step (20 ms) before episode end. On a 12 s /
600-step horizon this is inconsequential.

Run (env_isaaclab6):
    python eval_act.py --ckpt runs/act_nominal/ckpt_final.pt --episodes 64 --num-envs 16

The policy/queue/normalization logic lives in ``BatchedACTController`` (sim-free,
CPU-testable); all Isaac Sim imports happen inside ``main()`` after the AppLauncher.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slot_act.dataset import ACTION_DIM, ENV_STATE_SLICE, OBS_DIM, STATE_SLICE
from slot_act.noise import noise_sigmas
from slot_act.normalize import MeanStdNormalizer

# Discontinuity flush thresholds (privileged, sim-side).
FLUSH_POS_JUMP = 0.03  # m, per-step block position jump (teleport / nudge detector)
# The pick-place z-drop half is DELIBERATELY ABSENT here, matching residual_core.py. It was a
# slip proxy sized against a 40 mm basket rim. On this task the block legitimately descends
# ~8 mm the instant the fingers open at the insert pose (it rides at z = 63 mm while carried
# and settles to 55 mm), which is the most precision-critical moment in the episode — the
# pick-place rule would flush the committed chunk exactly there. Chunk commitment is
# load-bearing: shortening the execution horizon collapsed success 59.4 -> 32.8 -> 3.1 -> 0 %.


class BatchedACTController:
    """Per-env action-queue manager around an ACT policy for batched sim eval.

    ``policy`` needs only ``predict_action_chunk(batch) -> (B, chunk_size, action_dim)``
    (see act/modeling_act.py); normalization happens HERE, outside the policy — this
    LeRobot version keeps normalization in an external processor (act/PROVENANCE.md):
    observations are mean/std-normalized before the forward and the predicted chunk is
    unnormalized before queueing.

    Each env has its own deque of upcoming actions. ``act`` refills every empty queue
    with ONE batched forward over exactly those envs, then pops one action per env.
    ``reset``/``flush`` clear only the targeted envs' queues (episode reset / section
    4.2 discontinuity trigger), forcing a fresh prediction on their next ``act``.
    """

    def __init__(self, policy, stats, n_action_steps: int, chunk_size: int, device,
                 fixed_x0: torch.Tensor | None = None, x0_scale: float = 1.0):
        if not (1 <= n_action_steps <= chunk_size):
            raise ValueError(f"n_action_steps={n_action_steps} must be in [1, chunk_size={chunk_size}]")
        self.policy = policy
        self.normalizer = MeanStdNormalizer(stats).to(device)
        self.n_action_steps = n_action_steps
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        # Optional (chunk_size, action_dim) noise shared by every prediction: makes the
        # base chunk a deterministic function of the observation (residual-RL stages,
        # PLAN section 6). None = draw x0 fresh per refill (historical eval behavior).
        self.fixed_x0 = fixed_x0.to(self.device) if fixed_x0 is not None else None
        # Optional per-env steered noise (EXP07): (N, chunk_size, action_dim), set by
        # the steering wrapper each RL window. Takes precedence over fixed_x0.
        self.steer_x0: torch.Tensor | None = None
        # Scale on a FRESHLY DRAWN x0 (ignored when fixed_x0/steer_x0 is set, which are already
        # scaled by the caller). EXP_STEER §14: success rises monotonically with ||x0|| inside
        # the structured family, and this is the version of that knob which requires no search.
        self.x0_scale = float(x0_scale)
        # Vectorized queue (no per-env python containers — required for 1000+ envs):
        # _buf (N, n_action_steps, 7) holds each env's committed window, _idx (N,) the
        # position of its NEXT action; _idx == n_action_steps means empty.
        self._buf: torch.Tensor | None = None  # lazily sized on first act()/peek()
        self._idx: torch.Tensor | None = None

    @torch.no_grad()
    def _refill(self, obs_batch: torch.Tensor) -> None:
        """Refill every empty queue with ONE batched forward over exactly those envs
        (ascending env order — preserves the historical forward-batch composition)."""
        n = obs_batch.shape[0]
        if self._buf is None:
            self._buf = torch.zeros(n, self.n_action_steps, ACTION_DIM, device=self.device)
            self._idx = torch.full((n,), self.n_action_steps, dtype=torch.long, device=self.device)
        if self._buf.shape[0] != n:
            raise ValueError(f"obs batch size {n} != controller size {self._buf.shape[0]}")

        empty = (self._idx >= self.n_action_steps).nonzero(as_tuple=False).squeeze(-1)
        if empty.numel():
            sub = obs_batch[empty]
            batch = self.normalizer.normalize(
                {
                    "observation.state": sub[:, STATE_SLICE],
                    "observation.environment_state": sub[:, ENV_STATE_SLICE],
                }
            )
            if self.steer_x0 is not None:
                chunk = self.policy.predict_action_chunk(batch, x0=self.steer_x0[empty])
            elif self.fixed_x0 is not None:
                chunk = self.policy.predict_action_chunk(batch, x0=self.fixed_x0)
            elif self.x0_scale != 1.0:
                # Fresh draw per refill, as usual, but scaled off the prior's shell. This is the
                # NON-hardcoded version of the norm result (EXP_STEER §14): no latent is chosen,
                # only its magnitude, so it needs no per-checkpoint search.
                x0 = self.x0_scale * torch.randn(
                    (sub.shape[0], self.chunk_size, ACTION_DIM), device=self.device)
                chunk = self.policy.predict_action_chunk(batch, x0=x0)
            else:
                chunk = self.policy.predict_action_chunk(batch)
            chunk = chunk[:, : self.n_action_steps]
            chunk = self.normalizer.unnormalize("action", chunk)  # (B, n_action_steps, 7)
            self._buf[empty] = chunk
            self._idx[empty] = 0

    @torch.no_grad()
    def act(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """(N, 34) observation batch -> (N, 7) actions, refilling empty queues first."""
        obs_batch = obs_batch.to(self.device, dtype=torch.float32)
        self._refill(obs_batch)
        actions = self._buf[torch.arange(self._buf.shape[0], device=self.device), self._idx]
        self._idx += 1
        return actions

    @torch.no_grad()
    def peek(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Refill empties, then return (base_actions (N, 7), phase (N,)) WITHOUT popping.

        ``phase`` is the 0-based index of the returned action inside its execution
        window (0 = fresh chunk, n_action_steps-1 = last committed step). Call
        ``pop()`` afterwards to consume exactly these actions (residual-RL wrapper).
        """
        obs_batch = obs_batch.to(self.device, dtype=torch.float32)
        self._refill(obs_batch)
        base = self._buf[torch.arange(self._buf.shape[0], device=self.device), self._idx]
        return base, self._idx.clone()

    def pop(self) -> torch.Tensor:
        """Consume the actions the last ``peek`` returned (one per env)."""
        actions = self._buf[torch.arange(self._buf.shape[0], device=self.device), self._idx]
        self._idx += 1
        return actions

    def reset(self, env_ids) -> None:
        """Clear the action queues of the given envs (episode reset)."""
        if self._idx is None:
            return
        ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, dtype=torch.long)
        self._idx[ids.to(self.device)] = self.n_action_steps

    def flush(self, env_ids) -> None:
        """Section 4.2 discontinuity flush — same mechanics as reset."""
        self.reset(env_ids)


def load_checkpoint(ckpt_path: Path, device):
    """Load a train_act.py checkpoint -> (policy, stats, config dict).

    Checkpoint format (act/train_act.py save_checkpoint): {step, policy_state_dict,
    normalizer_state_dict, config:{chunk_size, n_action_steps, state_dim, env_state_dim,
    action_dim}}. Normalizer stats are recovered from the buffer names MeanStdNormalizer
    registers ("<key with . -> __>_mean"/"_std"). Checkpoints from act/train_flow.py
    additionally carry config["policy_type"] == "flow" (+ num_inference_steps) and are
    dispatched to FlowMatchingPolicy (same predict_action_chunk contract).
    """
    from types import SimpleNamespace

    from slot_act.modeling_act import ACTPolicy
    from slot_act.train_act import make_config

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg["state_dim"] == STATE_SLICE.stop - STATE_SLICE.start, cfg
    assert cfg["env_state_dim"] == ENV_STATE_SLICE.stop - ENV_STATE_SLICE.start, cfg
    assert cfg["action_dim"] == ACTION_DIM, cfg

    stats = {}
    for name, tensor in ckpt["normalizer_state_dict"].items():
        if name.endswith("_mean"):
            key = name[: -len("_mean")].replace("__", ".")
            stats[key] = {"mean": tensor, "std": ckpt["normalizer_state_dict"][name[: -len("_mean")] + "_std"]}

    if cfg.get("policy_type") == "flow":
        # identical architecture construction to training (act/train_flow.py make_config)
        from slot_act.modeling_flow import FlowMatchingPolicy
        from slot_act.train_flow import make_config as make_flow_config

        config = make_flow_config(
            SimpleNamespace(
                chunk_size=cfg["chunk_size"],
                n_action_steps=cfg["n_action_steps"],
                num_inference_steps=cfg.get("num_inference_steps", 10),
                device=str(device),
            )
        )
        policy = FlowMatchingPolicy(config)
    else:
        # identical architecture construction to training (act/train_act.py make_config)
        config = make_config(
            SimpleNamespace(chunk_size=cfg["chunk_size"], n_action_steps=cfg["n_action_steps"], device=str(device))
        )
        policy = ACTPolicy(config)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device).eval()
    return policy, stats, {**cfg, "step": ckpt.get("step")}


def main() -> None:
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="train_act.py checkpoint (.pt)")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--task", default="Rebot-PrecisionSlot-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-action-steps", type=int, default=None, help="override ckpt config's n_action_steps")
    parser.add_argument("--flush", action=argparse.BooleanOptionalAction, default=True,
                        help="section 4.2 discontinuity flush trigger (default on)")
    parser.add_argument("--out", default=None, help="results JSON (default: eval_act_results.json next to ckpt)")
    parser.add_argument("--episode-length-s", type=float, default=12.0,
                        help="episode horizon. 12.0 s = 599 control steps at dt = decimation/sim_dt "
                             "= 8/400 = 20 ms, which is EXACTLY the task default and exactly the "
                             "length of every collected demo (T = 599). Do not raise it: the "
                             "inherited pick-place default of 30 s rolls the policy 900 steps past "
                             "anything in its training distribution, with last_action feeding back "
                             "into the observation the whole way.")
    parser.add_argument("--fixed-x0", default=None, metavar="MODE",
                        help="freeze the flow's integration noise instead of drawing it fresh "
                             "per refill. 'zeros' = the distribution's mode, making each chunk a "
                             "DETERMINISTIC function of the observation; an integer = a fixed "
                             "N(0,I) draw at that seed, shared by every env and every refill. "
                             "'b<seed>' = the SAME draw broadcast across all chunk positions, "
                             "i.e. one 7-vector repeated 50 times -- which is exactly the family "
                             "x0-steering can express (x0 = alpha*tanh(z), one z per action dim). "
                             "b<seed> is row 0 of <seed>'s matrix, so the pair is directly "
                             "comparable and isolates whether per-chunk-position STRUCTURE is "
                             "what a good latent needs. "
                             "The controller has always supported this (BatchedACTController "
                             "fixed_x0); only the flag was missing. Use it to separate 'the "
                             "policy cannot do this' from 'the policy's own sampling noise "
                             "eats the clearance' -- see docs/slot/EXP_TIGHT.md.")
    parser.add_argument("--x0-scale", type=float, default=1.0, metavar="K",
                        help="multiply --fixed-x0 by K. Only meaningful with --fixed-x0, and it "
                             "exists to test ONE thing: a d=350 standard normal concentrates on "
                             "a shell of radius sqrt(350) = 18.7, so x0 = zeros is not 'the "
                             "average sample' -- it is nowhere near the typical set the flow was "
                             "trained to integrate from. x0-steering's x0 = alpha*tanh(z) sits at "
                             "norm ~5.6 at initialisation and cannot exceed 14.2, both short of "
                             "18.7. If success tracks ||x0|| this flag will show it.")
    # ---- robustness perturbations. The env randomises ONE thing per episode (the block spawn,
    # x +/-20 mm, y +/-30 mm, yaw +/-0.35 rad); friction is startup-only and the slot is welded
    # to SLOT_CENTER = (0.245, 0.0). These flags widen that, WITHOUT editing eva_rl.
    parser.add_argument("--slot-dx", type=float, default=0.0,
                        help="shift the slot along x by this many metres (+ = further from the "
                             "robot). insertion_depth is monotone in x, so the policy has a "
                             "usable signed signal for this axis.")
    parser.add_argument("--slot-dy", type=float, default=0.0,
                        help="shift the slot along y by this many metres. NOTE lateral_error is "
                             "an ABSOLUTE value -- the policy is told how far off it is, never "
                             "which way -- so this axis is the harder test by construction.")
    parser.add_argument("--spawn-scale", type=float, default=1.0,
                        help="multiply the block spawn pose_range by this. 1.0 = the training "
                             "distribution; >1 is out-of-distribution.")
    parser.add_argument("--arm-jitter", type=float, default=0.0, metavar="RAD",
                        help="uniform +/- offset [rad] on joint1-6 at every reset. The challenge "
                             "env has NO arm-start randomisation -- every episode the policy has "
                             "ever seen began from the identical _START_POSE -- so this is the "
                             "one axis on which it has literally zero coverage. pick_place uses "
                             "0.1 rad for exactly this reason (pick_place_v1_env_cfg.py:137).")
    parser.add_argument("--obs-noise", type=float, default=0.0, metavar="FRAC",
                        help="gaussian sensor noise on the observation, as a fraction of each "
                             "channel's OWN training std (so one scalar is meaningful across "
                             "radians, metres and quaternion components). Applied to obs[0:27] "
                             "only: [27:34] is the policy's own last commanded action, which is "
                             "internal state, not a measurement, and corrupting it would test "
                             "something else. The env's enable_corruption is False, so the "
                             "policy trained on noiseless observations.")
    parser.add_argument("--action-noise", type=float, default=0.0, metavar="FRAC",
                        help="gaussian noise on the ACTION, as a fraction of each channel's own "
                             "training std. The counterpart to --obs-noise: that one degrades "
                             "sensing, this one degrades actuation, and they are different "
                             "questions. Worth asking separately because every perturbation "
                             "measured so far -- slot dx, wider spawns, sensor noise -- does its "
                             "damage to the PUSH phase, and the push is the part an actuator "
                             "error acts on most directly.")
    parser.add_argument("--video", action="store_true", help="record an rgb_array video (offscreen render)")
    parser.add_argument("--video-length", type=int, default=3000, help="video length in env steps")
    parser.add_argument("--viewer-eye", type=float, nargs=3, default=None,
                        help="viewer camera eye xyz (close-up framing for failure review)")
    parser.add_argument("--viewer-lookat", type=float, nargs=3, default=None,
                        help="viewer camera lookat xyz")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.video:
        args.enable_cameras = True
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym

    import reBot_RL.tasks  # noqa: F401  (registers Rebot-PrecisionSlot-*)
    import slot_mdp as mdp  # the act/ mdp surface for the slot task; see slot/slot_mdp.py
    from isaaclab_tasks.utils import parse_env_cfg

    device = "cuda:0"
    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out) if args.out else ckpt_path.parent / "eval_act_results.json"

    policy, stats, ckpt_cfg = load_checkpoint(ckpt_path, device)
    n_action_steps = args.n_action_steps if args.n_action_steps is not None else ckpt_cfg["n_action_steps"]
    fixed_x0 = None
    if args.fixed_x0 is not None:
        shape = (ckpt_cfg["chunk_size"], ACTION_DIM)
        if args.fixed_x0 == "zeros":
            fixed_x0 = torch.zeros(shape)
        elif args.fixed_x0.startswith("b"):
            # One 7-vector repeated across all chunk positions: the family x0-steering CAN
            # express. Row 0 is taken from the FULL matrix rather than drawn as (1, ACTION_DIM)
            # -- torch's CPU normal_ uses a different fill path below 16 elements, so the two
            # do not agree (checked: they differ) and only this way does `b7` share a value
            # with `7`, leaving per-position structure as the single difference.
            g = torch.Generator().manual_seed(int(args.fixed_x0[1:]))
            fixed_x0 = torch.randn(shape, generator=g)[0].unsqueeze(0).expand(shape).contiguous()
        else:
            g = torch.Generator().manual_seed(int(args.fixed_x0))  # CPU generator => portable
            fixed_x0 = torch.randn(shape, generator=g)
        if args.x0_scale != 1.0:
            fixed_x0 = fixed_x0 * args.x0_scale
        print(f"[eval_act] fixed x0 = {args.fixed_x0} x {args.x0_scale:g}: the chunk is now a "
              f"deterministic function of the observation; ||x0|| = "
              f"{float(fixed_x0.norm()):.2f} (typical for N(0,I) at d={fixed_x0.numel()} is "
              f"{fixed_x0.numel() ** 0.5:.2f})")
    controller = BatchedACTController(policy, stats, n_action_steps, ckpt_cfg["chunk_size"],
                                      device, fixed_x0=fixed_x0, x0_scale=args.x0_scale)
    if fixed_x0 is None and args.x0_scale != 1.0:
        print(f"[eval_act] x0 SCALE {args.x0_scale:g} on a fresh draw per refill: ||x0|| ~ "
              f"{args.x0_scale * (ckpt_cfg['chunk_size'] * ACTION_DIM) ** 0.5:.2f} "
              f"(prior shell {(ckpt_cfg['chunk_size'] * ACTION_DIM) ** 0.5:.2f}). No latent is "
              f"chosen -- only its magnitude.")

    # Per-channel scales in units of each channel's own training std (slot_act/noise.py), so a
    # single --obs-noise number means the same thing for a joint angle and a quaternion
    # component -- and so this magnitude is the same perturbation the steering harness uses.
    obs_sigma, act_sigma = noise_sigmas(stats, device, args.obs_noise, args.action_noise,
                                        tag="eval_act")

    # ---- slot shift, applied BEFORE parse_env_cfg because _fixture_parts() reads SLOT_CENTER
    # at cfg-construction time (precision_slot_env_cfg.py:86, called from __post_init__:303).
    #
    # THREE bindings must move together or the geometry and the success predicate silently
    # disagree -- which would score a perfectly-inserted block as a failure, or vice versa:
    #   * challenge.mdp.common.SLOT_CENTER  -- read at call time by insertion_depth (:94),
    #     lateral_error (:100) and rewards.py (:49)
    #   * challenge.mdp.SLOT_CENTER         -- a SEPARATE binding created by `from .common
    #     import *`; this is the one the env cfg reads to place the boxes
    #   * slot_mdp.SLOT_CENTER              -- re-exported again for the act/ surface
    # Patching only the first is the trap: the predicate would move and the walls would not.
    if args.slot_dx or args.slot_dy:
        from reBot_RL.tasks.manager_based.challenge import mdp as _cmdp
        from reBot_RL.tasks.manager_based.challenge.mdp import common as _common

        old = tuple(_common.SLOT_CENTER)
        new = (old[0] + args.slot_dx, old[1] + args.slot_dy)
        for _m in (_common, _cmdp, mdp):
            if hasattr(_m, "SLOT_CENTER"):
                setattr(_m, "SLOT_CENTER", new)
        # assert the patch took on EVERY binding rather than trusting setattr
        for _m, _n in ((_common, "mdp.common"), (_cmdp, "mdp"), (mdp, "slot_mdp")):
            got = getattr(_m, "SLOT_CENTER", None)
            assert got == new, f"SLOT_CENTER patch missed {_n}: {got} != {new}"
        print(f"[eval_act] SLOT SHIFTED {old} -> {new}  (dx={args.slot_dx:+.3f}, "
              f"dy={args.slot_dy:+.3f} m)")

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)

    if args.spawn_scale != 1.0:
        pr = env_cfg.events.reset_block.params["pose_range"]
        env_cfg.events.reset_block.params["pose_range"] = {
            k: (v[0] * args.spawn_scale, v[1] * args.spawn_scale) for k, v in pr.items()}
        print(f"[eval_act] SPAWN RANGE x{args.spawn_scale}: {pr} -> "
              f"{env_cfg.events.reset_block.params['pose_range']}")

    if args.arm_jitter:
        # Appended to the instance, so it lands LAST in EventCfg.__dict__ and therefore runs
        # after reset_all (= reset_scene_to_default), which would otherwise put the arm straight
        # back on _START_POSE. reset_joints_by_offset offsets from default_joint_pos, so it does
        # not accumulate across episodes.
        from isaaclab.envs.mdp import reset_joints_by_offset
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        env_cfg.events.reset_arm_jitter = EventTermCfg(
            func=reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"]),
                "position_range": (-args.arm_jitter, args.arm_jitter),
                "velocity_range": (0.0, 0.0),
            },
        )
        assert list(env_cfg.events.__dict__)[-1] == "reset_arm_jitter", (
            f"arm jitter must run last or reset_all undoes it: "
            f"{list(env_cfg.events.__dict__)}")
        print(f"[eval_act] ARM START JITTER +/-{args.arm_jitter} rad on joint1-6 "
              f"(the env normally has none)")

    # KEEP time_out (fixed horizon ends episodes); DISABLE object_dropping + its reward
    # hook to match demo generation (see module docstring / run_expert_v1.py Driver).
    assert env_cfg.terminations.time_out is not None
    # Slot task term names differ from pick-place's, and nulling them is a DECISION, not a
    # rename: a dropped or toppled block must FAIL the episode, not silently re-randomise the
    # scene and let the policy be scored on a second attempt.
    env_cfg.terminations.block_dropped = None
    env_cfg.terminations.block_toppled = None
    env_cfg.rewards.dropping_penalty = None
    env_cfg.rewards.toppling_penalty = None
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    env_cfg.seed = args.seed
    if args.viewer_eye is not None:
        env_cfg.viewer.eye = tuple(args.viewer_eye)
    if args.viewer_lookat is not None:
        env_cfg.viewer.lookat = tuple(args.viewer_lookat)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(ckpt_path.parent / "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    u = env.unwrapped
    n = args.num_envs

    def obj_pos_local() -> torch.Tensor:
        """(N, n_obj, 3) env-local object root positions. One object here; the object axis is
        kept so the flush check and the record schema stay shape-compatible."""
        return torch.stack([mdp.object_pos_local(u, name) for name in mdp.OBJECT_NAMES], dim=1)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    assert obs.shape == (n, OBS_DIM), obs.shape

    # ---- prove the slot shift moved the GEOMETRY, not just the predicate.
    # A patch that reached mdp.common but not the env cfg would leave the walls where they were
    # while lateral_error measured from the new centre: every insert would score as a failure
    # and the run would look like a clean collapse. Read the actual wall prim back instead.
    if args.slot_dx or args.slot_dy:
        want_cx, want_cy = mdp.SLOT_CENTER
        floor = u.scene["floor"] if "floor" in u.scene.keys() else None
        if floor is not None:
            got = (floor.data.root_pos_w.torch[0, :2] - u.scene.env_origins[0, :2]).tolist()
            assert abs(got[0] - want_cx) < 1e-4 and abs(got[1] - want_cy) < 1e-4, (
                f"slot GEOMETRY did not move: floor prim at {got}, predicate at "
                f"{(want_cx, want_cy)} -- the SLOT_CENTER patch reached the mdp module but "
                f"not the env cfg")
            print(f"[eval_act] verified: slot floor prim at ({got[0]:.4f}, {got[1]:.4f}) "
                  f"matches the shifted predicate centre")
        else:
            print("[eval_act] WARNING: no 'floor' asset in the scene -- geometry shift is "
                  "UNVERIFIED; treat the numbers as provisional")

    prev_pos = obj_pos_local()
    just_reset = torch.ones(n, dtype=torch.bool, device=u.device)  # no flush across resets
    ep_len = torch.zeros(n, dtype=torch.long)
    ep_flushes = torch.zeros(n, dtype=torch.long)
    ep_max_placed = torch.zeros(n, dtype=torch.long)
    ep_final_placed = torch.zeros(n, dtype=torch.long)
    ep_max_obj_z = torch.zeros(n, len(mdp.OBJECT_NAMES))
    # Which episode this is FOR THIS ENV. The first episode a process executes scores about
    # 13 points higher than every later one (measured: 128/128 vs 114/128 and 109/128 from an
    # identical plan and a bit-identical initial state, 4.3 sigma -- see HANDOFF section 1).
    # Without this tag the headline number silently depends on --num_envs: 128 episodes over
    # 128 envs is ENTIRELY first-episode and maximally biased, while 128 over 16 envs is mostly
    # not. Both cohorts are reported so the bias is visible instead of baked in.
    ep_index = torch.zeros(n, dtype=torch.long)
    # Runtime observation distribution vs the checkpoint's training normaliser (EXP_STEER 8d).
    obs_mean = torch.cat([stats["observation.state"]["mean"],
                          stats["observation.environment_state"]["mean"]]).to(device)
    # + eps exactly as MeanStdNormalizer does it (normalize.py:54): a channel that never moved
    # in the demos has std 0, and the z the POLICY sees is the one divided by (std + 1e-8).
    obs_std = torch.cat([stats["observation.state"]["std"],
                         stats["observation.environment_state"]["std"]]).to(device) + 1e-8
    assert obs_mean.shape == obs_std.shape == (OBS_DIM,), (obs_mean.shape, obs_std.shape)
    obs_abs_z = torch.zeros(OBS_DIM, device=device, dtype=torch.float64)
    obs_sum = torch.zeros(OBS_DIM, device=device, dtype=torch.float64)
    obs_sq = torch.zeros(OBS_DIM, device=device, dtype=torch.float64)
    obs_n = 0
    records: list[dict] = []
    flush_count = 0

    # Spawn of the episode currently running in each env. Recorded so that pairing between two
    # eval runs can be VERIFIED rather than assumed. The env seed fixes the reset draws, so two
    # runs at the same seed *should* see identical spawns and the arm comparison would then be
    # paired (much more powerful than unpaired) -- but only if the policy does not consume the
    # same global RNG the reset events draw from. It does: flow sampling calls torch.randn for
    # x0 on every refill. Consumption is identical between arms only while every env refills in
    # lockstep and no env flushes, which is true but fragile. So: record it and check.
    ep_spawn = obj_pos_local()[:, 0].clone()
    # Spawn YAW, recorded separately from spawn_pos so the pairing check above keeps comparing
    # exactly what it always compared. --spawn-scale widens the yaw range too (+/-0.35 ->
    # +/-0.70 rad at 2.0), so without this an "inside the nominal box" cohort split on x/y alone
    # silently mixes in-distribution positions with out-of-distribution yaws -- which is exactly
    # what made the spawn20 in-box cell ambiguous the first time it was read (EXP_ROBUSTNESS 6e).
    ep_spawn_yaw = mdp.yaw_of(mdp.object_quat(u, mdp.OBJECT_NAMES[0])).clone()

    while len(records) < args.episodes:
        pos = obj_pos_local()
        fresh = (ep_len == 0).nonzero(as_tuple=False).squeeze(-1)
        if fresh.numel():
            ep_spawn[fresh.to(u.device)] = pos[fresh.to(u.device), 0]
            ep_spawn_yaw[fresh.to(u.device)] = mdp.yaw_of(
                mdp.object_quat(u, mdp.OBJECT_NAMES[0]))[fresh.to(u.device)]
        if args.flush:
            jump = torch.linalg.norm(pos - prev_pos, dim=-1) > FLUSH_POS_JUMP  # (N, n_obj)
            trig = (jump.any(dim=1) & ~just_reset).nonzero(as_tuple=False).squeeze(-1)
            if trig.numel():
                ids = trig.tolist()
                controller.flush(ids)
                flush_count += len(ids)
                ep_flushes[ids] += 1
        prev_pos = pos
        # Success predicate, sampled pre-step (see module docstring). NOTE this is
        # slot_mdp.placed_mask, which is `is_inserted AND seated at z = 55 +/- 6 mm` -- NOT the
        # env's bare is_inserted, which bounds block height only from below and so also passes
        # for a block resting on the 30 mm wall tops or still dangling in a closed gripper.
        # The bare predicate is recorded alongside so the gap stays visible.
        placed_now = mdp.placed_mask(u)
        raw_now = mdp.is_inserted(u)
        last_placed = placed_now.all(dim=1)
        # Slot-frame error, sampled HERE and not after env.step. Isaac Lab resets done envs
        # inside step(), so anything read afterwards describes the NEXT episode's spawn, not
        # the outcome just scored. That is not a subtle few-millimetre skew: the instrument
        # run reported a median |lateral| of 126.79 mm on episodes whose block was actually
        # seated at y = 0.0001 +/- 0.0008 m -- it was reporting the fresh block's distance from
        # the slot. success/placed_max/final_obj_pos were always read pre-step; these three
        # were the only stragglers.
        depth_now = mdp.insertion_depth(u).cpu()
        lat_now = mdp.lateral_error(u).cpu()
        yaw_now = mdp.yaw_error(u).cpu()
        ep_max_placed = torch.maximum(ep_max_placed, placed_now.sum(dim=1).cpu())
        ep_final_placed = placed_now.sum(dim=1).cpu()
        ep_max_obj_z = torch.maximum(ep_max_obj_z, pos[..., 2].cpu())
        just_reset[:] = False

        # Per-channel distribution shift of what the policy is ACTUALLY shown, in units of the
        # checkpoint's own normaliser (EXP_STEER 8d). Always on: it costs three tensor ops per
        # step, and when a perturbation costs 83 points the first question is *which input
        # channel stopped looking like training data*. Accumulated on obs, not obs_in, so
        # --obs-noise measures the shift the SIMULATOR produced, not the one we injected.
        z_now = (obs - obs_mean) / obs_std
        obs_abs_z += z_now.abs().sum(dim=0)
        obs_sum += obs.sum(dim=0)
        obs_sq += obs.square().sum(dim=0)
        obs_n += n

        obs_in = obs if obs_sigma is None else obs + args.obs_noise * obs_sigma * torch.randn_like(obs)
        actions = controller.act(obs_in).to(u.device)
        if act_sigma is not None:
            # after the controller, so the noise corrupts the COMMAND rather than the plan.
            # NOTE the policy DOES see the corruption on the next step: obs[27:34] is
            # mdp.last_action, which returns env.action_manager.action -- the action actually
            # passed to step(), i.e. the noisy one -- and obs[0:16] carries its physical
            # consequences anyway. (An earlier version of this comment claimed the policy saw a
            # clean last_action. It does not.)
            actions = actions + args.action_noise * act_sigma * torch.randn_like(actions)
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]
        ep_len += 1

        done = (terminated | truncated).view(-1).cpu().nonzero(as_tuple=False).squeeze(-1)
        if done.numel():
            # No basket here -- the slot is welded to the table at mdp.SLOT_CENTER. Record the
            # slot-frame error instead, which is what actually decides success. These come
            # from the PRE-step sample above; re-reading them now would read the reset scene.
            depth, lat, yaw = depth_now, lat_now, yaw_now
            raw_c = raw_now.cpu()
            for i in done.tolist():
                records.append(
                    {
                        "episode": len(records),
                        "env": i,
                        "episode_index_in_env": int(ep_index[i]),
                        "length": int(ep_len[i]),
                        "success": bool(last_placed[i]),
                        "flushes": int(ep_flushes[i]),
                        "placed_final": int(ep_final_placed[i]),
                        "placed_max": int(ep_max_placed[i]),
                        "max_obj_z": [round(float(z), 3) for z in ep_max_obj_z[i]],
                        # last pre-step sample (one control step stale — diagnostics only)
                        "final_obj_pos": [[round(float(v), 3) for v in c] for c in pos[i].cpu()],
                        "final_placed_per_obj": [bool(x) for x in placed_now[i].cpu()],
                        "inserted_raw": bool(raw_c[i]),   # env predicate WITHOUT the seat guard
                        # block pose at the START of this episode -- lets two runs be checked
                        # for spawn-identity, which decides whether they can be compared paired
                        "spawn_pos": [round(float(v), 5) for v in ep_spawn[i].cpu()],
                        "spawn_yaw": round(float(ep_spawn_yaw[i]), 5),
                        "depth_mm": round(float(depth[i]) * 1000, 2),
                        "lateral_mm": round(float(lat[i]) * 1000, 2),
                        "yaw_rad": round(float(yaw[i]), 4),
                    }
                )
            controller.reset(done.tolist())
            ep_len[done] = 0
            ep_flushes[done] = 0
            ep_max_placed[done] = 0
            ep_final_placed[done] = 0
            ep_max_obj_z[done] = 0.0
            ep_index[done] += 1
            just_reset[done.to(u.device)] = True

    records = records[: args.episodes]
    n_ep = len(records)
    first = [r for r in records if r["episode_index_in_env"] == 0]
    later = [r for r in records if r["episode_index_in_env"] > 0]
    rate = lambda rs: (sum(r["success"] for r in rs) / len(rs)) if rs else None  # noqa: E731
    result = {
        "ckpt": str(ckpt_path),
        "task": args.task,
        "seed": args.seed,
        "episodes": n_ep,
        "success_rate": sum(r["success"] for r in records) / n_ep,
        # Split by position-in-process. Use `success_rate_later` for A/B comparisons: it is the
        # steady-state number and does not move when --num_envs changes.
        "success_rate_first_episode": rate(first),
        "success_rate_later": rate(later),
        "n_first_episode": len(first),
        "n_later": len(later),
        "mean_ep_len": sum(r["length"] for r in records) / n_ep,
        "flush_count": flush_count,
        "flush_enabled": bool(args.flush),
        # fixed_x0 MUST be recorded: a deterministic-mode run is otherwise byte-indistinguishable
        # from a stochastic one, and the two are not comparable. None = fresh draw per refill.
        "config": {**ckpt_cfg, "n_action_steps_effective": n_action_steps, "num_envs": n,
                   "fixed_x0": args.fixed_x0, "x0_scale": args.x0_scale,
                   # perturbations, recorded for the same reason as fixed_x0: a shifted-slot run
                   # is otherwise byte-indistinguishable from a nominal one.
                   "slot_dx": args.slot_dx, "slot_dy": args.slot_dy,
                   "spawn_scale": args.spawn_scale, "arm_jitter": args.arm_jitter,
                   "obs_noise": args.obs_noise, "action_noise": args.action_noise,
                   "episode_length_s": args.episode_length_s},
        # EXP_STEER 8d: what the policy was actually shown, per channel, against its own
        # training normaliser. `mean_abs_z` is the headline -- ~0.8 is what an in-distribution
        # channel looks like (E|z| = 0.798 for a standard normal).
        "obs_dist": {
            "n_samples": obs_n,
            "mean_abs_z": [round(v, 4) for v in (obs_abs_z / max(obs_n, 1)).tolist()],
            "runtime_mean": [round(v, 5) for v in (obs_sum / max(obs_n, 1)).tolist()],
            "runtime_std": [round(v, 5) for v in
                            (obs_sq / max(obs_n, 1) - (obs_sum / max(obs_n, 1)).square())
                            .clamp_min(0.0).sqrt().tolist()],
            "train_mean": [round(v, 5) for v in obs_mean.tolist()],
            "train_std": [round(v, 5) for v in obs_std.tolist()],
        },
        "per_episode": records,
    }
    out_path.write_text(json.dumps(result, indent=2))
    def _f(v):
        return "  n/a" if v is None else f"{v:.3f}"

    print(
        f"[eval_act] {n_ep} eps: success_rate={result['success_rate']:.3f} "
        f"mean_ep_len={result['mean_ep_len']:.1f} flushes={flush_count} -> {out_path}"
    )
    print(f"[eval_act]   first-episode {_f(result['success_rate_first_episode'])} "
          f"(n={len(first)})   later {_f(result['success_rate_later'])} (n={len(later)})"
          f"   <- compare arms on 'later'")
    if len(later) == 0:
        print("[eval_act]   *** every episode is a first episode (--episodes <= --num_envs), so "
              "this number carries the full ~13-point first-episode bias. Raise --episodes. ***")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
