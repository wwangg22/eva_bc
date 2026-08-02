"""EXP07 x0-steering core: frozen base + chunk-level steering machinery.

Chunk-window control model (experiments/EXP07_x0_steering.md): the RL policy acts once
per execution window (n_action_steps = 15 env steps). At each window boundary:

    obs56 = core.build_obs(obs41, env)          # features only, no controller state
    z = policy(obs56)                           # (N, 7), rl_games-clamped to [-1, 1]
    core.set_steer(z)                           # x0 = alpha_x0 * tanh(z), broadcast
    for _ in range(15):                         # the window
        core.flush_check(env)                   # inherited section 4.2 flush
        action = core.controller.act(obs41)     # refills use the CURRENT steer_x0
        obs41 = env.step(action)
    core.reset(done_env_ids)

The controller stays free-running (refill on empty queue, exactly as in every ladder
eval); z only enters through the x0 of refills that happen while it is held. With the
window-aligned protocol (30 s = 1500 steps = 100 windows, no mid-episode terminations)
every refill lands on a window boundary; only a section 4.2 flush desyncs an env
(documented, accepted — its z is applied up to 14 steps stale).

Steering observation layout (56-D = EXP06's 64-D minus queued-base-action and
chunk-index, which do not exist at a window boundary):
    [0:41]   the base policy's own 41-D obs (raw env units)
    [41:45]  physical finger features: pos dims 6,7 + vel dims 14,15
    [45:46]  validated grasp-success bit (EXP06 gate-1 MLP)
    [46:53]  target-can pose in the gripper frame: pos 3 + quat XYZW 4
    [53:56]  target-can -> basket delta (env-local)

Action: z in R^7 (6 arm + 1 grip column of x0), one value per action dim broadcast
across all chunk positions; x0_steer = alpha_x0 * tanh(z).
"""

from __future__ import annotations

from pathlib import Path

import torch

from act.residual_core import ResidualCore

STEER_OBS_DIM = 56
STEER_ACTION_DIM = 7


class SteerCore(ResidualCore):
    """Reuses ResidualCore's flush logic + feature tail; the additive alpha/compose
    path is unused (alpha pinned to 0)."""

    def __init__(self, controller, mdp, grasp_bit_path: Path, alpha_x0: float = 1.0,
                 flush: bool = True):
        super().__init__(controller, mdp, grasp_bit_path, alpha=0.0, flush=flush)
        self.alpha_x0 = float(alpha_x0)

    @torch.no_grad()
    def set_steer(self, z: torch.Tensor) -> torch.Tensor:
        """(N, 7) z -> hold x0 = alpha_x0 * tanh(z) for refills until the next call.

        Returns the applied (N, 7) x0 values (pre-broadcast) for logging.
        """
        x0 = self.alpha_x0 * torch.tanh(z.to(self.device, dtype=torch.float32))
        self.controller.steer_x0 = x0.unsqueeze(1).expand(-1, self.controller.chunk_size, -1)
        return x0

    @torch.no_grad()
    def build_obs(self, obs41: torch.Tensor, env) -> torch.Tensor:
        """56-D steering observation at a window boundary (no controller side effects)."""
        obs56 = torch.cat([obs41, self.task_features(obs41, env)], dim=1)
        assert obs56.shape == (obs41.shape[0], STEER_OBS_DIM), obs56.shape
        return obs56
