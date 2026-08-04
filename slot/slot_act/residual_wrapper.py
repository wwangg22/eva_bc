"""EXP06 — rl_games vec-env wrapper with the frozen base + residual blending.

Import only AFTER AppLauncher (isaaclab_rl pulls sim-side modules). Split out of
train_residual.py so diagnostics can import it without triggering that script's
module-level argparse.
"""

from __future__ import annotations

import gym as ogym  # rl_games-facing spaces module
import torch

from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

from slot_act.residual_core import RES_ACTION_DIM, RES_OBS_DIM, ResidualCore


class ResidualRlGamesWrapper(RlGamesVecEnvWrapper):
    """RlGamesVecEnvWrapper with the frozen base + residual blending inside step()."""

    def __init__(self, env, rl_device, clip_obs, clip_actions, core: ResidualCore,
                 res_penalty: float):
        super().__init__(env, rl_device, clip_obs, clip_actions)
        self.core = core
        self.res_penalty = float(res_penalty)

    @property
    def observation_space(self):
        return ogym.spaces.Box(-self._clip_obs, self._clip_obs, (RES_OBS_DIM,))

    @property
    def action_space(self):
        return ogym.spaces.Box(-self._clip_actions, self._clip_actions, (RES_ACTION_DIM,))

    def _obs64(self, obs_dict):
        obs64 = self.core.advance(obs_dict["policy"], self.unwrapped)
        return {"obs": torch.clamp(obs64, -self._clip_obs, self._clip_obs).to(self._rl_device)}

    def reset(self):  # noqa: D102
        obs_dict, _ = self.env.reset()
        return self._obs64(obs_dict)

    def step(self, actions):  # noqa: D102
        res = actions.detach().clone().to(device=self._sim_device)
        full, applied = self.core.compose(res)
        obs_dict, rew, terminated, truncated, extras = self.env.step(full)
        rew = rew - self.res_penalty * applied.square().sum(dim=1)

        dones = terminated | truncated
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel():
            self.core.reset(done_ids.tolist())

        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self._rl_device)
        obs = self._obs64(obs_dict)
        rew = rew.to(device=self._rl_device)
        dones = dones.to(device=self._rl_device)
        extras = {
            k: v.to(device=self._rl_device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in extras.items()
        }
        if "log" in extras:
            extras["episode"] = extras.pop("log")
        return obs, rew, dones, extras
