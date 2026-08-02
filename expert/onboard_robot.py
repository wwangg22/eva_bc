"""Onboard the RS-rebot-dev-arm into cuRoboV2 (PLAN.md ledger item 2).

Builds a cuRobo robot config from the original URDF via RobotBuilder (MorphIt
auto sphere fitting), then patches in the pieces the builder does not know:
locked finger joints, retract config matching the sim start pose, attached-object
link for in-hand collision checking, grasp contact links.

Outputs (next to this script):
  rs_rebot.yml          — final cuRobo robot config
  rs_rebot_spheres.png  — sphere fit visualization for Big Will's review
Run:  conda activate env_isaaclab6 && python onboard_robot.py
"""

import math
import os

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")
URDF = os.path.join(ASSETS, "00-arm-rs_asm-v3", "urdf", "00-arm-rs_asm-v3.urdf")
OUT_YML = os.path.join(HERE, "rs_rebot.yml")
OUT_PNG = os.path.join(HERE, "rs_rebot_spheres.png")

# The env's OPERATING start pose (_START_POSE in lift/rebot_lift_env_cfg.py, applied by
# pick_place cfgs via init_state override). NOT rebot_arm.py's USD init (joint2=-pi/2):
# that config puts the gripper 13 cm inside the table — PhysX shoves it out at reset.
SIM_START = {
    "joint1": 0.0,
    "joint2": -1.35,
    "joint3": -0.3,
    "joint4": -0.85,
    "joint5": 0.0,
    "joint6": 0.0,
}
GRIPPER_OPEN = 0.045  # sim's open target for joint_left / joint_right
TCP_OFFSET = (-0.075, 0.0, 0.0)  # TCP in gripper_end local frame (env ee_frame OffsetCfg)


def build():
    from curobo.robot_builder import RobotBuilder

    builder = RobotBuilder(URDF, ASSETS, tool_frames=["gripper_end"])
    print("[1/5] fitting collision spheres (MorphIt)...")
    builder.fit_collision_spheres()
    for link, spheres in builder.collision_spheres.items():
        print(f"    {link}: {len(spheres)} spheres")
    print("[2/5] computing self-collision matrix...")
    builder.compute_collision_matrix()
    cfg = builder.build()
    builder.save(cfg, OUT_YML)
    print(f"[3/5] saved base config -> {OUT_YML}")


def patch():
    with open(OUT_YML) as f:
        d = yaml.safe_load(f)
    kin = d.get("robot_cfg", d)["kinematics"]  # v2 builder writes kinematics at top level

    kin["lock_joints"] = {"joint_left": GRIPPER_OPEN, "joint_right": GRIPPER_OPEN}
    kin["grasp_contact_link_names"] = [
        "gripper_end", "gripper_left", "gripper_right", "attached_object",
    ]
    kin["extra_links"] = {
        "attached_object": {
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_name": "attach_joint",
            "joint_type": "FIXED",
            "link_name": "attached_object",
            "parent_link_name": "gripper_end",
        }
    }
    if not kin.get("extra_collision_spheres"):
        kin["extra_collision_spheres"] = {}
    kin["extra_collision_spheres"]["attached_object"] = 8
    if "attached_object" not in kin["collision_link_names"]:
        kin["collision_link_names"].append("attached_object")
    sci = kin.get("self_collision_ignore") or {}
    kin["self_collision_ignore"] = sci
    for parent in ("gripper_end", "gripper_left", "gripper_right", "link6", "link5"):
        lst = sci.setdefault(parent, [])
        if "attached_object" not in lst:
            lst.append("attached_object")
    scb = kin.get("self_collision_buffer") or {}
    kin["self_collision_buffer"] = scb
    scb["attached_object"] = 0.0

    # Clamp base_link spheres above the table plane: MorphIt protrudes them to
    # z=-0.027 although the base mesh sits ON the table (z>=0). Left unclamped,
    # ANY table obstacle permanently collides with the base and every IK/plan
    # fails (verified 2026-08-01: proven grasp configs 24/24 IK in empty world,
    # 0/24 with any table until this clamp).
    for s in kin["collision_spheres"]["base_link"]:
        cz, r = s["center"][2], s["radius"]
        if cz - r < -0.001:
            if cz >= 0.004:
                s["radius"] = round(cz + 0.001, 4)
            else:
                s["center"][2] = 0.005
                s["radius"] = min(r, 0.006)

    cs = kin["cspace"]
    default = []
    for name in cs["joint_names"]:
        if name in SIM_START:
            default.append(SIM_START[name])
        else:  # finger joints (present only if not excluded by lock_joints)
            default.append(GRIPPER_OPEN)
    cs["default_joint_position"] = default

    with open(OUT_YML, "w") as f:
        yaml.safe_dump(d, f, default_flow_style=None, sort_keys=False)
    print(f"[4/5] patched config (lock_joints, retract, attached_object) -> {OUT_YML}")
    print("    cspace joints:", cs["joint_names"])
    print("    retract:", [round(v, 4) for v in default])


def verify():
    from curobo.robot_builder import RobotDebugger

    dbg = RobotDebugger(OUT_YML)
    res = dbg.check_default_joint_configuration_collision()
    print("[5/5] retract (default config) collision check:", res)

    # FK sanity at sim start pose + sphere viz for Big Will (no GUI).
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState

    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(OUT_YML))
    jn = kin.joint_names
    q = torch.tensor([[SIM_START.get(n, GRIPPER_OPEN) for n in jn]], device="cuda", dtype=torch.float32)
    state = kin.compute_kinematics(JointState.from_position(q, joint_names=jn))
    ee = state.tool_poses
    print("    joint order:", jn)
    print("    tool_frames:", state.tool_frames)
    print("    gripper_end pos @ sim start:", ee.position.cpu().numpy().round(4).tolist())
    print("    gripper_end quat @ sim start:", ee.quaternion.cpu().numpy().round(4).tolist())

    spheres = state.get_link_spheres().cpu().numpy()
    spheres = spheres.reshape(-1, 4) if spheres.shape[-1] == 4 else spheres.reshape(4, -1).T
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))
    views = [(90, -90, "top (xy)"), (0, -90, "front (xz)"), (0, 0, "side (yz)")]
    for i, (elev, azim, title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        valid = spheres[spheres[:, 3] > 0]
        ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2], s=(valid[:, 3] * 800) ** 2,
                   alpha=0.4, edgecolors="k", linewidths=0.3)
        p = ee.position.reshape(-1, 3)[0].cpu().numpy()
        ax.scatter(*p, color="red", s=60, marker="x")
        ax.set_title(f"{title} — {len(valid)} spheres, red x = gripper_end")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        for f in (ax.set_xlim, ax.set_ylim): f(-0.45, 0.45)
        ax.set_zlim(0, 0.9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110)
    print(f"    sphere viz saved -> {OUT_PNG} (for Big Will — not viewed here)")


if __name__ == "__main__":
    import sys

    if "--skip-build" not in sys.argv or not os.path.exists(OUT_YML):
        build()
    patch()
    verify()
    print("DONE")
