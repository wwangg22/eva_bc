# 00 — Machine, software and repo environment

*Recorded 2026-08-02. Everything here is measured on this box, not assumed.*

## Hardware

| | |
|---|---|
| GPU | **NVIDIA GeForce RTX 3080, 10 GiB**, sm_86, driver 595.84, CUDA 13.2 |
| CPU | 24 logical cores |
| RAM | 62 GiB total, ~60 GiB available |

> **Deviation from eva_bc's provenance.** eva_bc's numbers (2048 envs at 10–15k steps/s in
> ~7 GB VRAM) were measured on a different card. eva_rl's `CHALLENGE_SUITE.md` C7 assumed an
> **11 GB RTX 2080 Ti**. We have **10 GiB**, so every VRAM budget in both repos must be
> re-measured, not inherited. C7's advice — "budget 512–1024 envs for contact-rich tasks" —
> applies here with less headroom, and the clutter env is contact-rich (6 rigid bodies per
> env, `gpu_max_rigid_patch_count=2**20`).

## Software

Conda env: **`env_isaaclab6`** (the user confirmed this is the one to use).

```
source /home/eva/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab6
```

| package | status |
|---|---|
| python | 3.12.13 |
| torch | 2.11.0+cu128, CUDA available |
| isaacsim | installed (site-packages) |
| isaaclab | **6.1.17**, editable from `/home/eva/Desktop/isaacLab/IsaacLab` |
| isaaclab_tasks | 1.10.9 |
| rl_games | installed |
| rsl_rl | installed |
| skrl | 2.1.0 |
| gymnasium | 1.2.1 |
| warp | 1.13.0 |
| h5py | 3.16.0 |
| numpy | 2.3.1 |
| **curobo** | **NOT INSTALLED** |

### The cuRobo gap — a load-bearing fact

eva_bc **Stage 1** (the scripted planner expert, `expert/run_expert_v1.py`) is built on
cuRobo (`curobo.motion_planner.MotionPlanner`), and eva_bc's README says the expert imports
"a local motion-planner wrapper API — adapt to your cuRobo install". There is no cuRobo
install here, and the expert scripts additionally carry machine-specific absolute paths from
the original author's box (`/home/william/...`).

Consequence for the plan: **do not attempt to port `run_expert_v1.py` verbatim.**

> **CORRECTED 2026-08-02 — see `06_EXPERT_DESIGN.md §1.** I first wrote here that the expert
> should be rewritten against `isaaclab.controllers.DifferentialIKController`. **That is
> wrong**, and both repos say so: eva_bc `PLAN.md:41` — *"DLS differential IK diverges from
> table-level configs — never trust raw DLS near the table"*; eva_rl
> `generate_pick_place.py:7-9` — *"differential IK stalls at a z-floor ~0.045 m"*. That
> z-floor is exactly the band this task lives in.
>
> The correct instrument is **FK-scored CEM over the 6 arm joints driving Cartesian waypoint
> chains**, the pattern already implemented in eva_rl's `scripts/analysis/grasp_geometry.py`
> and `scripts/challenge/slot_insertion_probe.py`. Its decisive property is that candidates
> are scored by forward kinematics *read back from the sim*, so it cannot silently converge
> on an unexecutable pose — the direct answer to eva_bc's "planner-valid ≠ executable" lesson.

cuRobo is separately the wrong tool for this task even if it were installed (no collision-free
grasp pose exists at t = 0; the core skill is deliberate contact; its finger collision spheres
are wider than the gap). Full argument in `06_EXPERT_DESIGN.md §2`.

## Repos

| repo | remote | branch | HEAD at session start |
|---|---|---|---|
| `eva_bc` | `git@github.com:wwangg22/eva_bc.git` | `main` | `1c04eca` (already up to date) |
| `eva_rl` | `git@github.com:wwangg22/eva_rl.git` | `master` | pulled to `05f0fb3` "Add challenge env suite, hardware measurement harnesses, and smoke tests" |
| `IsaacLab` | `git@github.com:isaac-sim/IsaacLab.git` | `release/3.0.0-beta2` | `6a7acb032` (already up to date) |

`eva_rl`'s pull is what brought in the whole `challenge/` package — the ClutterExtract task
did not exist in the tree before this session.

## Working conventions for this effort

- All new work lives under `eva_bc/clutter/`. Nothing outside that subfolder is modified
  without saying so explicitly, so the eva_bc tracked tree stays clean until Big Will asks
  for a push.
- `eva_bc/.gitignore` already ignores `runs/`, `*.h5`, `*.pt`, `*.mp4`, `*.log`, so
  `clutter/runs/` and any checkpoints/demos are excluded automatically. Docs, scripts and
  probes under `clutter/` are the tracked deliverable.
- **One GPU job at a time** (eva_bc HANDOFF rule, and with 10 GiB it is enforced by physics).
  Check with `nvidia-smi` + `ps aux | grep "[t]rain_\|[e]val_"`.
</content>
