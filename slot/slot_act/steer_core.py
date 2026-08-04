"""EXP07 x0-steering core: frozen base + chunk-level steering machinery.

Chunk-window control model (experiments/EXP07_x0_steering.md): the RL policy acts once
per execution window (n_action_steps = 15 env steps). At each window boundary:

    obs50 = core.build_obs(obs34, env)          # features only, no controller state
    z = policy(obs50)                           # (N, 7), rl_games-clamped to [-1, 1]
    core.set_steer(z)                           # x0 = alpha_x0 * tanh(z), broadcast
    for _ in range(15):                         # the window
        core.flush_check(env)                   # inherited section 4.2 flush
        action = core.controller.act(obs34)     # refills use the CURRENT steer_x0
        obs34 = env.step(action)
    core.reset(done_env_ids)

The controller stays free-running (refill on empty queue, exactly as in every ladder
eval); z only enters through the x0 of refills that happen while it is held. With the
window-aligned protocol (slot: 12 s = 600 steps = exactly 40 windows, no episode-length
override needed -- pick-place had to force 30 s) every refill lands on a window boundary; only a section 4.2 flush desyncs an env
(documented, accepted — its z is applied up to 14 steps stale).

Steering observation layout (50-D = the residual's 58-D minus queued-base-action and
chunk-index, which do not exist at a window boundary):
    [0:34]   the base policy's own 34-D obs (raw env units)
    [34:38]  physical finger features: pos dims 6,7 + vel dims 14,15
    [38:39]  grasp bit -- ANALYTIC (residual_core.SlotGraspBit), not the EXP06 MLP
    [39:46]  block pose in the gripper frame: pos 3 + quat XYZW 4
    [46:50]  signed block -> goal delta (env-local): see residual_core for why 4 and not 3

Action: z in R^7 (6 arm + 1 grip column of x0), one value per action dim broadcast
across all chunk positions; x0_steer = alpha_x0 * tanh(z).
"""

from __future__ import annotations

import torch

from slot_act.residual_core import ResidualCore

STEER_OBS_DIM = 50
STEER_ACTION_DIM = 7


class SteerCore(ResidualCore):
    """Reuses ResidualCore's flush logic + feature tail; the additive alpha/compose
    path is unused (alpha pinned to 0)."""

    def __init__(self, controller, mdp, alpha_x0: float = 1.0, flush: bool = True):
        super().__init__(controller, mdp, alpha=0.0, flush=flush)
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
    def build_obs(self, obs34: torch.Tensor, env) -> torch.Tensor:
        """50-D steering observation at a window boundary (no controller side effects)."""
        obs50 = torch.cat([obs34, self.task_features(obs34, env)], dim=1)
        assert obs50.shape == (obs34.shape[0], STEER_OBS_DIM), obs50.shape
        return obs50
