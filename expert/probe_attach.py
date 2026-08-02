"""Inspect attached-object spheres for a dumped place-refusal case.

Prints: the fitted sphere tensor (obstacle frame), the attached spheres in WORLD
frame at the case's q6 (via FK), the held can's true world position, min sphere
z (table top is at z=-0.002), and overlaps vs other robot spheres.
Run (env_isaaclab6, GPU free): python probe_attach.py [case_idx]
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from replay_place_fail import Expert, build_scene  # noqa: E402

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cases = torch.load(os.path.join(HERE, "place_fail_cases.pt"),
                   map_location="cpu", weights_only=False)
case = cases[idx]
q6, bxy, held = case["q6"], case["basket_xy"], case["held"]
print(f"case {idx}: held={held} q6={np.round(q6, 3).tolist()}")
print(f"  can world pos (dump): {np.round(case['cans'][held], 4).tolist()}")

ex = Expert()
full = {n: (xyz, False) for n, xyz in case["cans"].items()}
ex.planner.update_world(build_scene(bxy, full))

kparams = ex.planner.kinematics.config.kinematics_config
att_idx = kparams.get_sphere_index_from_link_name("attached_object").cpu().numpy()

st = ex.planner.kinematics.compute_kinematics(ex.js(q6))
before = st.get_link_spheres().reshape(-1, 4).cpu().numpy()
print(f"\nattached slots BEFORE attach (world): idx {att_idx.tolist()}")
for i in att_idx:
    print(f"  [{i}] c=({before[i,0]:+.4f},{before[i,1]:+.4f},{before[i,2]:+.4f}) r={before[i,3]:+.4f}")

from curobo.types import Pose  # noqa: E402
ex.planner.attachment_manager.attach_from_scene(
    ex.js(q6), [held],
    world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0]))
fit = ex.planner.attachment_manager._last_fit_result
print(f"\nfit result: {fit.num_spheres} spheres (obstacle frame):")
sph = torch.cat([fit.centers, fit.radii.unsqueeze(-1)], dim=-1).cpu().numpy()
for s in sph:
    print(f"  c=({s[0]:+.4f},{s[1]:+.4f},{s[2]:+.4f}) r={s[3]:.4f}")

st = ex.planner.kinematics.compute_kinematics(ex.js(q6))
world = st.get_link_spheres().reshape(-1, 4).cpu().numpy()
print(f"\nattached spheres AFTER attach (world frame at q6):")
act = world[att_idx]
for i, s in zip(att_idx, act):
    print(f"  [{i}] c=({s[0]:+.4f},{s[1]:+.4f},{s[2]:+.4f}) r={s[3]:+.4f}")
live = act[act[:, 3] > 0]
if len(live):
    print(f"  min sphere-bottom z: {(live[:, 2] - live[:, 3]).min():+.4f}  (table top at -0.002)")

# overlaps vs the rest of the robot's spheres
others = np.delete(world, att_idx, axis=0)
others = others[others[:, 3] > 0]
print("\noverlaps vs other robot spheres (dist < r1+r2):")
none = True
for i, s in zip(att_idx, act):
    if s[3] <= 0:
        continue
    d = np.linalg.norm(others[:, :3] - s[:3], axis=1) - (others[:, 3] + s[3])
    for j in np.where(d < 0)[0]:
        none = False
        print(f"  att[{i}] r={s[3]:.4f} <-> robot sphere c={np.round(others[j,:3],3).tolist()} "
              f"r={others[j,3]:.4f} pen={-d[j]:.4f}")
if none:
    print("  none")
