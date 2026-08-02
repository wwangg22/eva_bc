"""EXP07 — rl_games vec-env wrapper for chunk-level x0-steering.

One rl_games step = one execution window (n_action_steps env steps) with the window's
summed raw reward. Import only AFTER AppLauncher (isaaclab_rl pulls sim-side modules).
See act/steer_core.py for the control model and experiments/EXP07_x0_steering.md for
the pre-registered design.
"""

from __future__ import annotations

import gym as ogym  # rl_games-facing spaces module
import torch

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

from act.steer_core import STEER_ACTION_DIM, STEER_OBS_DIM, SteerCore


class SteerRlGamesWrapper(RlGamesVecEnvWrapper):
    """56-D obs / 7-D z action; step() runs a full 15-env-step window."""

    def __init__(self, env, rl_device, clip_obs, clip_actions, core: SteerCore):
        super().__init__(env, rl_device, clip_obs, clip_actions)
        self.core = core
        self.window = core.controller.n_action_steps
        self._obs41: torch.Tensor | None = None

    @property
    def observation_space(self):
        return ogym.spaces.Box(-self._clip_obs, self._clip_obs, (STEER_OBS_DIM,))

    @property
    def action_space(self):
        return ogym.spaces.Box(-self._clip_actions, self._clip_actions, (STEER_ACTION_DIM,))

    def _obs56(self):
        obs56 = self.core.build_obs(self._obs41, self.unwrapped)
        return {"obs": torch.clamp(obs56, -self._clip_obs, self._clip_obs).to(self._rl_device)}

    def reset(self):  # noqa: D102
        obs_dict, _ = self.env.reset()
        self._obs41 = obs_dict["policy"]
        return self._obs56()

    def step(self, z):  # noqa: D102
        self.core.set_steer(z.detach())
        u = self.unwrapped
        rew_sum = torch.zeros(u.num_envs, device=self._sim_device)
        term_any = torch.zeros(u.num_envs, dtype=torch.bool, device=self._sim_device)
        trunc_any = torch.zeros_like(term_any)
        extras = {}
        for _ in range(self.window):
            self.core.flush_check(u)
            action = self.core.controller.act(self._obs41)
            obs_dict, rew, terminated, truncated, extras = self.env.step(action.to(self._sim_device))
            self._obs41 = obs_dict["policy"]
            rew_sum += rew
            term_any |= terminated
            trunc_any |= truncated
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel():
                # env auto-reset already happened; post-reset steps of this window run
                # on a fresh chunk under the CURRENT z (window-aligned protocol makes
                # mid-window dones impossible; this is robustness only)
                self.core.reset(done_ids.tolist())

        dones = term_any | trunc_any
        if not u.cfg.is_finite_horizon:
            extras["time_outs"] = trunc_any.to(device=self._rl_device)
        rew_sum = rew_sum.to(device=self._rl_device)
        dones = dones.to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in extras.items()
        }
        if "log" in extras:
            extras["episode"] = extras.pop("log")
        return self._obs56(), rew_sum, dones, extras
