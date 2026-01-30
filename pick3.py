# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a custom robot in an Isaac Lab scene.

Usage:
    ./isaaclab.sh -p scripts/tutorials/01_assets/add_new_robot.py
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spawn a custom robot in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.0, help="Environment spacing.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms


# ----------------------------
#  PhysX material helpers
# ----------------------------
def _try_bind_physx_material(stage, material_path: str, prim_paths: list[str],
                             static_friction=1.5, dynamic_friction=1.2, restitution=0.0):
    """Create (or reuse) a physics material and bind it to a list of prim paths.

    This is written with multiple fallbacks because Isaac Sim / IsaacLab versions differ.
    If it fails, it prints a warning but won't crash the script.
    """
    try:
        import omni.usd
        from pxr import Sdf
    except Exception as e:
        print(f"[WARN] Cannot import omni.usd/pxr; skip friction binding. err={e}")
        return

    stage = stage
    if stage is None:
        try:
            stage = omni.usd.get_context().get_stage()
        except Exception as e:
            print(f"[WARN] Cannot get USD stage; skip friction binding. err={e}")
            return

    # Create material prim
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        try:
            stage.DefinePrim(material_path, "Material")
            material_prim = stage.GetPrimAtPath(material_path)
        except Exception as e:
            print(f"[WARN] Cannot define material prim; skip friction binding. err={e}")
            return

    # Try create physx material via physicsUtils (best)
    created = False
    try:
        from omni.physx.scripts import physicsUtils
        # create_physics_material exists in many versions
        physicsUtils.create_physics_material(
            stage,
            material_path,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=restitution,
        )
        created = True
    except Exception:
        created = False

    # Fallback: apply PhysxMaterialAPI directly if available
    if not created:
        try:
            from pxr import PhysxSchema
            mat_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
            mat_api.CreateStaticFrictionAttr().Set(float(static_friction))
            mat_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
            mat_api.CreateRestitutionAttr().Set(float(restitution))
            created = True
        except Exception as e:
            print(f"[WARN] Cannot author PhysX material attrs; friction binding may not work. err={e}")

    # Bind to prims
    bound_ok = 0
    for p in prim_paths:
        prim = stage.GetPrimAtPath(p)
        if not prim.IsValid():
            continue

        # Prefer physicsUtils binding if present
        try:
            from omni.physx.scripts import physicsUtils
            physicsUtils.add_physics_material_to_prim(stage, prim, material_prim)
            bound_ok += 1
            continue
        except Exception:
            pass

        # Fallback: write binding relationship commonly used by PhysX
        # Some versions use "physxMaterial:binding" or "physics:material:binding"
        try:
            # Most common in PhysX pipeline
            rel = prim.GetRelationship("physxMaterial:binding")
            if not rel:
                rel = prim.CreateRelationship("physxMaterial:binding", False)
            rel.SetTargets([material_path])
            bound_ok += 1
            continue
        except Exception:
            pass

        try:
            rel = prim.GetRelationship("physics:material:binding")
            if not rel:
                rel = prim.CreateRelationship("physics:material:binding", False)
            rel.SetTargets([material_path])
            bound_ok += 1
            continue
        except Exception:
            pass

    print(f"[INFO] Friction material prepared at {material_path} "
          f"(mu_s={static_friction}, mu_d={dynamic_friction}, e={restitution}). "
          f"Bound to {bound_ok}/{len(prim_paths)} prims.")


def _find_collision_prims_under(stage, root_prim_path: str, name_keywords: tuple[str, ...] = ()):
    """Return collision prim paths under a root. Optionally filter by keywords in prim path."""
    try:
        from pxr import UsdPhysics
    except Exception:
        return []

    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        return []

    out = []
    it = root.GetDescendants()
    for prim in it:
        try:
            # A collision prim usually has UsdPhysics.CollisionAPI applied
            if UsdPhysics.CollisionAPI(prim):
                p = prim.GetPath().pathString
                if name_keywords:
                    s = p.lower()
                    if any(k.lower() in s for k in name_keywords):
                        out.append(p)
                else:
                    out.append(p)
        except Exception:
            continue
    return out


# ----------------------------
#  Robot config (tuned)
# ----------------------------
G1_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/xixi/IsaacLab/assets/G1_omnipicker/G1_omnipicker.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # 稍微放宽 depenetration，接触更稳（不要太大，避免弹飞）
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # solver 迭代稍微提高一点
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=6,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "idx01_body_joint1": 0.35,
            "idx02_body_joint2": 0.449,
            "idx11_head_joint1": 0.0,
            "idx12_head_joint2": 0.384,
            "idx21_arm_l_joint1": -0.6696,
            "idx61_arm_r_joint1": 0.6699,
            "idx22_arm_l_joint2": 0.201,
            "idx62_arm_r_joint2": -0.201,
            "idx23_arm_l_joint3": 0.27,
            "idx63_arm_r_joint3": -0.27,
            "idx24_arm_l_joint4": -1.2,
            "idx64_arm_r_joint4": 1.2,
            "idx25_arm_l_joint5": 0.8,
            "idx65_arm_r_joint5": -0.8,
            "idx26_arm_l_joint6": 1.57,
            "idx66_arm_r_joint6": -1.57,
            "idx27_arm_l_joint7": -0.18,
            "idx67_arm_r_joint7": 0.18,
            "idx31_gripper_l_inner_joint1": 0.0,
            "idx41_gripper_l_outer_joint1": 0.0,
            "idx71_gripper_r_inner_joint1": 0.0,
            "idx81_gripper_r_outer_joint1": 0.0,
            "idx32_gripper_l_inner_joint3": 0.1,
            "idx42_gripper_l_outer_joint3": 0.1,
            "idx72_gripper_r_inner_joint3": -0.1,
            "idx82_gripper_r_outer_joint3": 0.1,
            "idx33_gripper_l_inner_joint4": 0.0,
            "idx43_gripper_l_outer_joint4": 0.0,
            "idx73_gripper_r_inner_joint4": 0.0,
            "idx83_gripper_r_outer_joint4": 0.0,
            "idx54_gripper_l_inner_joint0": 0.0,
            "idx53_gripper_l_outer_joint0": 0.0,
            "idx94_gripper_r_inner_joint0": 0.0,
            "idx93_gripper_r_outer_joint0": 0.0,
        },
    ),
    actuators={
        "body_joints": ImplicitActuatorCfg(
            joint_names_expr=["idx0[1-2]_body_joint.*"],
            stiffness=100000.0,
            damping=100.0,
            velocity_limit_sim=None,
        ),
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["idx[26][1-7]_arm_[lr]_joint.*"],
            # 手臂可保持原样（IK 算目标，执行跟随为主）
            stiffness=None,
            damping=None,
            velocity_limit_sim=None,
            effort_limit_sim=None,
        ),
        # ✅ 关键：夹爪变“硬”和“有力”
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*gripper.*"],
            stiffness=15000.0,       # 夹紧刚度（可再加：2e4~5e4）
            damping=300.0,           # 阻尼避免抖动/弹开
            velocity_limit_sim=6.0,  # 关合速度限制（太快会撞飞物体）
            effort_limit_sim=200.0,  # 夹紧力上限（太小夹不住）
        ),
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-2]_head_joint.*"],
            stiffness=None,
            damping=None,
            velocity_limit_sim=None,
            effort_limit_sim=None,
        ),
    },
)


@configclass
class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/xixi/IsaacLab/assets/scene/table.usd",
            scale=(0.004, 0.006, 0.008),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=False,
                rigid_body_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )
    # Cube (0.06m x 0.06m x 0.06m)
    cube = AssetBaseCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.83), rot=(0, 0, 0, 1)),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # robot
    G1bot: ArticulationCfg = G1_CONFIG.replace(prim_path="{ENV_REGEX_NS}/G1bot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Runs the simulation loop.

    This version demonstrates a simple pick-and-place pipeline:
    1) read cube pose (ground-truth in simulation)
    2) move left end-effector above cube -> descend -> close gripper -> lift
    3) move to a fixed place position -> descend -> open -> retreat
    """
    robot = scene["G1bot"]
    cube = scene["cube"]
    sim_dt = sim.get_physics_dt()

    # Reset robot state
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()

    # Differential IK controller for left arm
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    left_arm_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    left_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current"))
    left_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/left_ee_goal"))
    right_ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_current"))
    right_goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/right_ee_goal"))

    # Configure arm entities (joint names and end-effector bodies)
    left_arm_cfg = SceneEntityCfg(
        "G1bot",
        joint_names=[
            "idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3",
            "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7"
        ],
        body_names=["arm_l_end_link"],
    )
    left_arm_cfg.resolve(scene)

    right_arm_cfg = SceneEntityCfg(
        "G1bot",
        joint_names=[
            "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
            "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7"
        ],
        body_names=["arm_r_end_link"],
    )
    right_arm_cfg.resolve(scene)

    # Jacobian indices
    if robot.is_fixed_base:
        left_ee_jacobi_idx = left_arm_cfg.body_ids[0] - 1
        right_ee_jacobi_idx = right_arm_cfg.body_ids[0] - 1
    else:
        left_ee_jacobi_idx = left_arm_cfg.body_ids[0]
        right_ee_jacobi_idx = right_arm_cfg.body_ids[0]

    # IK command buffers (controller expects pose in base frame)
    left_ik_commands = torch.zeros(scene.num_envs, left_arm_ik_controller.action_dim, device=robot.device)

    # Gripper joints
    left_gripper_joint_names = ["idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"]
    right_gripper_joint_names = ["idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"]
    left_gripper_joint_ids = [robot.data.joint_names.index(n) for n in left_gripper_joint_names]
    right_gripper_joint_ids = [robot.data.joint_names.index(n) for n in right_gripper_joint_names]

    # Gripper open/close targets (may need tuning)
    GRIP_OPEN = 0.6
    GRIP_CLOSE = 0.0

    # Fixed place position (world frame)
    place_pos_w = torch.tensor([0.60, 0.20, 0.83], device=robot.device)

    # Fixed grasp orientation (tool-down guess). If wrist orientation is wrong, tune this quaternion.
    goal_quat_w = torch.tensor([0.707, 0.0, 0.707, 0.0], device=robot.device)  # [w, x, y, z]

    # FSM parameters (✅ 微调：更高 approach、更合理 grasp z，避免“压住推走”)
    state = "APPROACH_ABOVE"
    hold_counter = 0
    APPROACH_Z = 0.15       # was 0.12
    GRASP_Z_OFFSET = 0.035  # was 0.02  （更接近“从上方轻贴近”而不是压到侧面推走）
    PLACE_Z_OFFSET = 0.04   # was 0.03
    CLOSE_STEPS = 60        # close gripper more gently
    SETTLE_STEPS = 30

    # Target smoothing
    MAX_TARGET_STEP = 0.01
    target_pos_w_smooth = None

    # Base alignment (only if robot has a free base)
    BASE_ALIGN_KP = 1.5
    BASE_ALIGN_MAX_SPEED = 0.2
    BASE_TARGET_OFFSET = torch.tensor([0.15, 0.0], device=robot.device)

    # Initialize desired joint positions to current
    right_joint_pos_des = robot.data.joint_pos[:, right_arm_cfg.joint_ids].clone()
    right_joint_hold = right_joint_pos_des.clone()

    def get_cube_pos_w() -> torch.Tensor:
        """Return cube position in world frame with robust fallbacks."""
        if hasattr(cube, "data"):
            if hasattr(cube.data, "root_pose_w"):
                pose = cube.data.root_pose_w
                return pose[:, 0:3]
            if hasattr(cube.data, "body_pose_w"):
                pose = cube.data.body_pose_w[:, 0, :]
                return pose[:, 0:3]
        if hasattr(cube, "get_world_poses"):
            pos_w, quat_w = cube.get_world_poses()
            return pos_w
        for attr in ("_positions", "positions", "translations"):
            if hasattr(cube, attr):
                v = getattr(cube, attr)
                if torch.is_tensor(v) and v.shape[-1] == 3:
                    return v
        raise RuntimeError("Cannot read cube world pose.")

    # Right arm stays fixed
    right_goal_pos_w = torch.tensor([0.50, -0.30, 1.00], device=robot.device).unsqueeze(0)
    right_goal_quat_w = goal_quat_w.unsqueeze(0)

    count = 0
    grasp_pos_w = None
    while simulation_app.is_running():
        # Read states
        root_pose_w = robot.data.root_pose_w
        left_ee_pose_w = robot.data.body_pose_w[:, left_arm_cfg.body_ids[0]]
        right_ee_pose_w = robot.data.body_pose_w[:, right_arm_cfg.body_ids[0]]

        # Compute jacobians
        left_jacobian = robot.root_physx_view.get_jacobians()[:, left_ee_jacobi_idx, :, left_arm_cfg.joint_ids]

        # Current joint positions
        left_joint_pos = robot.data.joint_pos[:, left_arm_cfg.joint_ids]

        # EE poses in base frame
        left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7]
        )

        cube_pos_w = get_cube_pos_w()

        # Optional base alignment
        if not robot.is_fixed_base:
            base_pos_w = root_pose_w[:, 0:3]
            base_target_xy = cube_pos_w[:, 0:2] - BASE_TARGET_OFFSET.unsqueeze(0)
            base_error_xy = base_target_xy - base_pos_w[:, 0:2]
            base_vel_xy = torch.clamp(BASE_ALIGN_KP * base_error_xy, -BASE_ALIGN_MAX_SPEED, BASE_ALIGN_MAX_SPEED)
            base_vel_cmd = torch.zeros(scene.num_envs, 6, device=robot.device)
            base_vel_cmd[:, 0:2] = base_vel_xy
            robot.write_root_velocity_to_sim(base_vel_cmd)

        # FSM
        if state == "APPROACH_ABOVE":
            target_pos_w = cube_pos_w + torch.tensor([0.0, 0.0, APPROACH_Z], device=robot.device)
            grip_target = GRIP_OPEN
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < 0.02:
                state = "DESCEND"

        elif state == "DESCEND":
            target_pos_w = cube_pos_w + torch.tensor([0.0, 0.0, GRASP_Z_OFFSET], device=robot.device)
            grip_target = GRIP_OPEN
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < 0.012:
                state = "CLOSE_GRIPPER"
                hold_counter = 0
                grasp_pos_w = cube_pos_w.clone()

        elif state == "CLOSE_GRIPPER":
            target_pos_w = (grasp_pos_w if grasp_pos_w is not None else cube_pos_w) + torch.tensor(
                [0.0, 0.0, GRASP_Z_OFFSET], device=robot.device
            )
            close_ratio = min(1.0, hold_counter / max(1, CLOSE_STEPS))
            grip_target = GRIP_OPEN + (GRIP_CLOSE - GRIP_OPEN) * close_ratio
            hold_counter += 1
            if hold_counter > (CLOSE_STEPS + SETTLE_STEPS):
                state = "LIFT"

        elif state == "LIFT":
            target_pos_w = (grasp_pos_w if grasp_pos_w is not None else cube_pos_w) + torch.tensor(
                [0.0, 0.0, APPROACH_Z], device=robot.device
            )
            grip_target = GRIP_CLOSE
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < 0.025:
                state = "MOVE_TO_PLACE_ABOVE"

        elif state == "MOVE_TO_PLACE_ABOVE":
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, APPROACH_Z], device=robot.device)
            grip_target = GRIP_CLOSE
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < 0.035:
                state = "DESCEND_PLACE"

        elif state == "DESCEND_PLACE":
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, PLACE_Z_OFFSET], device=robot.device)
            grip_target = GRIP_CLOSE
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < 0.018:
                state = "OPEN_GRIPPER"
                hold_counter = 0

        elif state == "OPEN_GRIPPER":
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, PLACE_Z_OFFSET], device=robot.device)
            grip_target = GRIP_OPEN
            hold_counter += 1
            if hold_counter > 25:
                state = "RETREAT"

        else:  # "RETREAT"
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, APPROACH_Z], device=robot.device)
            grip_target = GRIP_OPEN

        # Smooth target
        if target_pos_w_smooth is None:
            target_pos_w_smooth = target_pos_w.clone()
        else:
            delta = target_pos_w - target_pos_w_smooth
            step = torch.clamp(delta, -MAX_TARGET_STEP, MAX_TARGET_STEP)
            target_pos_w_smooth = target_pos_w_smooth + step
        target_pos_w = target_pos_w_smooth

        # WORLD -> BASE target pose for IK
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos_w, goal_quat_w.unsqueeze(0).repeat(scene.num_envs, 1)
        )

        left_ik_commands[:, 0:3] = target_pos_b
        left_ik_commands[:, 3:7] = target_quat_b
        left_arm_ik_controller.set_command(left_ik_commands)

        # Compute IK joint targets
        left_joint_pos_des = left_arm_ik_controller.compute(
            left_ee_pos_b, left_ee_quat_b, left_jacobian, left_joint_pos
        )

        # Blend toward IK solution
        MAX_JOINT_STEP = 0.02
        joint_delta = left_joint_pos_des - left_joint_pos
        joint_step = torch.clamp(joint_delta, -MAX_JOINT_STEP, MAX_JOINT_STEP)
        left_joint_pos_des = left_joint_pos + joint_step

        # Keep right arm fixed
        right_joint_pos_des = right_joint_hold

        # Full joint target
        all_joint_pos_des = robot.data.default_joint_pos.clone()
        all_joint_pos_des[:, left_arm_cfg.joint_ids] = left_joint_pos_des
        all_joint_pos_des[:, right_arm_cfg.joint_ids] = right_joint_pos_des

        # Apply gripper targets
        for jid in left_gripper_joint_ids:
            all_joint_pos_des[:, jid] = grip_target
        for jid in right_gripper_joint_ids:
            all_joint_pos_des[:, jid] = GRIP_OPEN

        # Send commands
        robot.set_joint_position_target(all_joint_pos_des)

        # Step sim
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        count += 1

        # Update markers
        left_ee_marker.visualize(left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7])
        left_goal_marker.visualize(target_pos_w, goal_quat_w.unsqueeze(0).repeat(scene.num_envs, 1))
        right_ee_marker.visualize(right_ee_pose_w[:, 0:3], right_ee_pose_w[:, 3:7])
        right_goal_marker.visualize(right_goal_pos_w, right_goal_quat_w)


def main() -> None:
    """Main function."""
    # 240 Hz
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 240.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))

    scene_cfg = NewRobotsSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    # ----------------------------
    # ✅ 运行时：检查碰撞体 + 绑定摩擦材质
    # ----------------------------
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        # 1) cube 摩擦
        cube_path = "/World/Cube"
        _try_bind_physx_material(
            stage,
            material_path="/World/PhysicsMaterials/HighFriction",
            prim_paths=[cube_path],
            static_friction=1.8,
            dynamic_friction=1.5,
            restitution=0.0,
        )

        # 2) gripper 碰撞体检查 + 绑定摩擦
        # 注意：机器人 prim 在 env 下面，因此我们用 scene 的实际 prim_path
        robot_prim_path = scene["G1bot"].prim_path  # like /World/envs/env_0/G1bot
        gripper_collision_prims = _find_collision_prims_under(
            stage, robot_prim_path, name_keywords=("gripper", "finger")
        )

        print(f"[INFO] Found {len(gripper_collision_prims)} gripper/finger collision prims under {robot_prim_path}:")
        for p in gripper_collision_prims[:30]:
            print(f"   - {p}")
        if len(gripper_collision_prims) == 0:
            print("[WARN] No gripper collision prims found! If you cannot grasp at all, "
                  "your USD gripper may have no collision shapes or collisions are disabled.")

        _try_bind_physx_material(
            stage,
            material_path="/World/PhysicsMaterials/HighFriction",
            prim_paths=gripper_collision_prims,
            static_friction=1.8,
            dynamic_friction=1.5,
            restitution=0.0,
        )

    except Exception as e:
        print(f"[WARN] Friction/collision setup skipped due to error: {e}")

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
