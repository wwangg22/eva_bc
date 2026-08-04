"""Per-channel perturbation scales, shared by every harness that injects noise.

A magnitude is a fraction of each channel's **own training std**, so one number means the
same thing for a joint angle (rad) and a quaternion component (dimensionless). That is a
convenience in ``eval_act.py``; in EXP_STEER it is load-bearing. The steering gate compares
a success rate measured through ``eval_act.py`` against one measured through
``eval_steer.py``/``steer_wrapper.py``, and that comparison is only meaningful if
``--action-noise 0.02`` denotes the *identical* perturbation in all three. It was written
twice by hand before this module existed; a third copy is how the two numbers quietly stop
being comparable.

``stats`` is the checkpoint's own normaliser dict (``load_checkpoint``'s second return), so
the scales follow the base policy that is being perturbed, not a global constant.
"""

from __future__ import annotations

import torch

from slot_act.dataset import ACTION_DIM, OBS_DIM

# obs[27:34] is mdp.last_action -- the command the harness itself issued last step. It is
# internal bookkeeping, not a measurement, so a *sensor*-noise sweep must not touch it.
# (Under --action-noise it does carry the corrupted command: last_action returns
# env.action_manager.action, i.e. the noisy action actually passed to step().)
LAST_ACTION_SLICE = slice(27, OBS_DIM)


def noise_sigmas(
    stats: dict,
    device: str | torch.device,
    obs_noise: float = 0.0,
    action_noise: float = 0.0,
    tag: str = "noise",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return ``(obs_sigma, act_sigma)``; either is ``None`` when its magnitude is 0.

    Callers multiply by their own magnitude at use time:
    ``obs + obs_noise * obs_sigma * randn_like(obs)``.
    """
    obs_sigma = None
    if obs_noise:
        obs_sigma = torch.cat([stats["observation.state"]["std"],
                               stats["observation.environment_state"]["std"]]).to(device).clone()
        assert obs_sigma.shape == (OBS_DIM,), obs_sigma.shape
        obs_sigma[LAST_ACTION_SLICE] = 0.0
        print(f"[{tag}] OBS NOISE {obs_noise:g} x per-channel training std on "
              f"obs[0:{LAST_ACTION_SLICE.start}]; "
              f"median sigma {float(obs_sigma[:LAST_ACTION_SLICE.start].median()):.4g}")

    act_sigma = None
    if action_noise:
        act_sigma = stats["action"]["std"].to(device).clone()
        assert act_sigma.shape == (ACTION_DIM,), act_sigma.shape
        print(f"[{tag}] ACTION NOISE {action_noise:g} x per-channel training std; "
              f"median sigma {float(act_sigma.median()):.4g}")

    return obs_sigma, act_sigma
