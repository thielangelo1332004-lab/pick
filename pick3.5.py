# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4,
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
            "idx41_gripper_l_outer_joint1": 0.0,  # 左臂控制张合
            "idx71_gripper_r_inner_joint1": 0.0,
            "idx81_gripper_r_outer_joint1": 0.0,  # 右臂控制张合
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
            stiffness=400.0,
            damping=40.0,
            velocity_limit_sim=None,
            effort_limit_sim=200.0,
        ),
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*gripper.*"],
            stiffness=1200.0,
            damping=80.0,
            velocity_limit_sim=None,
            effort_limit_sim=80.0,
        ),
        "head_joints": ImplicitActuatorCfg(
            joint_names_expr=["idx1[1-2]_head_joint.*"],
            stiffness=None,
            damping=None,
            velocity_limit_sim=None,
            effort_limit_sim=None,
        ),
    },
    # actuator_value_resolution_debug_print=True,
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
            size=(0.035, 0.035, 0.035),
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

    Notes:
    - This is a minimal, practical example. For more robust grasping you may need to tune gripper joints,
      contact/friction, or implement an "attach" (kinematic follow) after closing the gripper.
    """
    robot = scene["G1bot"]
    cube = scene["cube"]  # spawned in NewRobotsSceneCfg as name "cube"
    sim_dt = sim.get_physics_dt()

    # Reset robot state
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()

    # Differential IK controllers for left and right arms
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    left_arm_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    right_arm_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

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
        body_names=["gripper_l_center_link"],
    )
    left_arm_cfg.resolve(scene)

    right_arm_cfg = SceneEntityCfg(
        "G1bot",
        joint_names=[
            "idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3",
            "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7"
        ],
        body_names=["gripper_r_center_link"],
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
    right_ik_commands = torch.zeros(scene.num_envs, right_arm_ik_controller.action_dim, device=robot.device)

    # Gripper joints (left + right).
    left_gripper_joint_names = [
        "idx31_gripper_l_inner_joint1",
        "idx41_gripper_l_outer_joint1",
        "idx32_gripper_l_inner_joint3",
        "idx42_gripper_l_outer_joint3",
        "idx33_gripper_l_inner_joint4",
        "idx43_gripper_l_outer_joint4",
        "idx54_gripper_l_inner_joint0",
        "idx53_gripper_l_outer_joint0",
    ]
    right_gripper_joint_names = [
        "idx71_gripper_r_inner_joint1",
        "idx81_gripper_r_outer_joint1",
        "idx72_gripper_r_inner_joint3",
        "idx82_gripper_r_outer_joint3",
        "idx73_gripper_r_inner_joint4",
        "idx83_gripper_r_outer_joint4",
        "idx94_gripper_r_inner_joint0",
        "idx93_gripper_r_outer_joint0",
    ]
    left_gripper_joint_ids = [robot.data.joint_names.index(n) for n in left_gripper_joint_names]
    right_gripper_joint_ids = [robot.data.joint_names.index(n) for n in right_gripper_joint_names]

    # Gripper open/close targets (tuned against the initial configuration).
    GRIP_OPEN = {
        "idx31_gripper_l_inner_joint1": 0.0,
        "idx41_gripper_l_outer_joint1": 0.0,
        "idx32_gripper_l_inner_joint3": 0.1,
        "idx42_gripper_l_outer_joint3": 0.1,
        "idx33_gripper_l_inner_joint4": 0.0,
        "idx43_gripper_l_outer_joint4": 0.0,
        "idx54_gripper_l_inner_joint0": 0.0,
        "idx53_gripper_l_outer_joint0": 0.0,
        "idx71_gripper_r_inner_joint1": 0.0,
        "idx81_gripper_r_outer_joint1": 0.0,
        "idx72_gripper_r_inner_joint3": -0.1,
        "idx82_gripper_r_outer_joint3": 0.1,
        "idx73_gripper_r_inner_joint4": 0.0,
        "idx83_gripper_r_outer_joint4": 0.0,
        "idx94_gripper_r_inner_joint0": 0.0,
        "idx93_gripper_r_outer_joint0": 0.0,
    }
    GRIP_CLOSE = {name: 0.0 for name in GRIP_OPEN}

    # Fixed place position (world frame). Adjust as needed.
    place_pos_w = torch.tensor([0.60, 0.20, 0.83], device=robot.device)

    # FSM parameters
    state = "APPROACH_ABOVE"
    hold_counter = 0
    APPROACH_Z = 0.12       # approach height above cube/place
    GRASP_Z_OFFSET = 0.02   # grasp height above cube center
    PLACE_Z_OFFSET = 0.03   # place height above place point
    SMOOTH_ALPHA = 0.2
    CUBE_POS_ALPHA = 0.7
    ATTACH_DISTANCE = 0.035
    ATTACH_HOLD_STEPS = 10
    STATE_TIMEOUT = 200
    REACH_TOL_APPROACH = 0.02
    REACH_TOL_DESCEND = 0.008
    REACH_TOL_PLACE = 0.02

    # Initialize desired joint positions to current
    left_joint_pos_des = robot.data.joint_pos[:, left_arm_cfg.joint_ids].clone()
    right_joint_pos_des = robot.data.joint_pos[:, right_arm_cfg.joint_ids].clone()

    # Start with right arm holding default pose (optional)
    # We still run IK so you can add tasks later.
    right_goal_pos_w = torch.tensor([0.50, -0.30, 1.00], device=robot.device).unsqueeze(0)
    right_goal_quat_w = None
    left_goal_quat_w = None
    smoothed_target_pos_w = None
    smoothed_cube_pos_w = None
    cube_attached = False
    attach_offset_b = None
    attach_counter = 0
    state_steps = 0
    grasp_xy_w = None

    def get_cube_pose_w() -> tuple[torch.Tensor, torch.Tensor]:
        """Return cube (pos, quat) in world frame with robust fallbacks."""
        if hasattr(cube, "data"):
            if hasattr(cube.data, "root_pose_w"):
                pose = cube.data.root_pose_w
                return pose[:, 0:3], pose[:, 3:7]
            if hasattr(cube.data, "body_pose_w"):
                pose = cube.data.body_pose_w[:, 0, :]
                return pose[:, 0:3], pose[:, 3:7]
        if hasattr(cube, "get_world_poses"):
            pos_w, quat_w = cube.get_world_poses()
            return pos_w, quat_w
        raise RuntimeError("Cannot find cube pose on cube.data. Check available fields on cube.data.")

    def write_cube_pose_w(pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
        """Write cube pose into sim if the API exists."""
        if hasattr(cube, "write_root_pose_to_sim"):
            cube.write_root_pose_to_sim(torch.cat([pos_w, quat_w], dim=-1))
            if hasattr(cube, "write_root_velocity_to_sim"):
                zero_vel = torch.zeros(pos_w.shape[0], 6, device=pos_w.device)
                cube.write_root_velocity_to_sim(zero_vel)
            return
        if hasattr(cube, "set_world_poses"):
            cube.set_world_poses(pos_w, quat_w)
            return
        if hasattr(cube, "set_world_pose"):
            cube.set_world_pose(pos_w, quat_w)
            return

    count = 0
    while simulation_app.is_running():
        # Read states
        root_pose_w = robot.data.root_pose_w
        left_ee_pose_w = robot.data.body_pose_w[:, left_arm_cfg.body_ids[0]]
        right_ee_pose_w = robot.data.body_pose_w[:, right_arm_cfg.body_ids[0]]

        # Compute jacobians
        left_jacobian = robot.root_physx_view.get_jacobians()[:, left_ee_jacobi_idx, :, left_arm_cfg.joint_ids]
        right_jacobian = robot.root_physx_view.get_jacobians()[:, right_ee_jacobi_idx, :, right_arm_cfg.joint_ids]

        # Current joint positions
        left_joint_pos = robot.data.joint_pos[:, left_arm_cfg.joint_ids]
        right_joint_pos = robot.data.joint_pos[:, right_arm_cfg.joint_ids]

        # EE poses in base frame
        left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7]
        )
        right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            right_ee_pose_w[:, 0:3], right_ee_pose_w[:, 3:7]
        )

        # Cache initial end-effector orientation as target to avoid elbow flipping.
        if left_goal_quat_w is None:
            left_goal_quat_w = left_ee_pose_w[:, 3:7].clone()
        if right_goal_quat_w is None:
            right_goal_quat_w = right_ee_pose_w[:, 3:7].clone()

        # ==== FSM: compute left target in WORLD frame ====
        cube_pos_w, cube_quat_w = get_cube_pose_w()
        if smoothed_cube_pos_w is None:
            smoothed_cube_pos_w = cube_pos_w.clone()
        smoothed_cube_pos_w = smoothed_cube_pos_w + CUBE_POS_ALPHA * (cube_pos_w - smoothed_cube_pos_w)
        cube_pos_w = smoothed_cube_pos_w
        state_steps += 1

        if state == "APPROACH_ABOVE":
            target_pos_w = cube_pos_w + torch.tensor([0.0, 0.0, APPROACH_Z], device=robot.device)
            grip_target = GRIP_OPEN
            # transition condition
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < REACH_TOL_APPROACH:
                state = "DESCEND"
                state_steps = 0
                grasp_xy_w = cube_pos_w[:, 0:2].clone()

        elif state == "DESCEND":
            if grasp_xy_w is None:
                grasp_xy_w = cube_pos_w[:, 0:2].clone()
            target_pos_w = cube_pos_w + torch.tensor([0.0, 0.0, GRASP_Z_OFFSET], device=robot.device)
            target_pos_w[:, 0:2] = grasp_xy_w
            grip_target = GRIP_OPEN
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < REACH_TOL_DESCEND:
                state = "CLOSE_GRIPPER"
                hold_counter = 0
                state_steps = 0

        elif state == "CLOSE_GRIPPER":
            if grasp_xy_w is None:
                grasp_xy_w = cube_pos_w[:, 0:2].clone()
            target_pos_w = cube_pos_w + torch.tensor([0.0, 0.0, GRASP_Z_OFFSET], device=robot.device)
            target_pos_w[:, 0:2] = grasp_xy_w
            grip_target = GRIP_CLOSE
            hold_counter += 1
            if torch.norm(left_ee_pose_w[:, 0:3] - cube_pos_w, dim=-1).max().item() < ATTACH_DISTANCE:
                attach_counter += 1
                if attach_counter > ATTACH_HOLD_STEPS and not cube_attached:
                    ee_pos_w = left_ee_pose_w[:, 0:3]
                    attach_offset_b = cube_pos_w - ee_pos_w
                    cube_attached = True
            else:
                attach_counter = 0
            if hold_counter > 50:
                state = "LIFT"
                state_steps = 0
                grasp_xy_w = None

        elif state == "LIFT":
            target_pos_w = cube_pos_w + torch.tensor([0.0, 0.0, APPROACH_Z], device=robot.device)
            grip_target = GRIP_CLOSE
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < REACH_TOL_APPROACH:
                state = "MOVE_TO_PLACE_ABOVE"
                state_steps = 0

        elif state == "MOVE_TO_PLACE_ABOVE":
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, APPROACH_Z], device=robot.device)
            grip_target = GRIP_CLOSE
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < REACH_TOL_PLACE:
                state = "DESCEND_PLACE"
                state_steps = 0

        elif state == "DESCEND_PLACE":
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, PLACE_Z_OFFSET], device=robot.device)
            grip_target = GRIP_CLOSE
            if torch.norm(left_ee_pose_w[:, 0:3] - target_pos_w, dim=-1).max().item() < 0.015:
                state = "OPEN_GRIPPER"
                hold_counter = 0
                state_steps = 0

        elif state == "OPEN_GRIPPER":
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, PLACE_Z_OFFSET], device=robot.device)
            grip_target = GRIP_OPEN
            hold_counter += 1
            if cube_attached:
                cube_attached = False
                attach_offset_b = None
            if hold_counter > 20:
                state = "RETREAT"
                state_steps = 0

        else:  # "RETREAT" or any unknown
            target_pos_w = place_pos_w.unsqueeze(0) + torch.tensor([0.0, 0.0, APPROACH_Z], device=robot.device)
            grip_target = GRIP_OPEN
            if state_steps > STATE_TIMEOUT:
                state = "APPROACH_ABOVE"
                state_steps = 0

        if state_steps > STATE_TIMEOUT and state in {"DESCEND", "CLOSE_GRIPPER"}:
            state = "APPROACH_ABOVE"
            state_steps = 0
            hold_counter = 0
            attach_counter = 0
            grasp_xy_w = None

        if smoothed_target_pos_w is None:
            smoothed_target_pos_w = target_pos_w.clone()
        smoothed_target_pos_w = smoothed_target_pos_w + SMOOTH_ALPHA * (target_pos_w - smoothed_target_pos_w)
        target_pos_w = smoothed_target_pos_w

        # Convert WORLD target pose -> BASE pose for IK controller
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            target_pos_w, left_goal_quat_w
        )

        left_ik_commands[:, 0:3] = target_pos_b
        left_ik_commands[:, 3:7] = target_quat_b
        left_arm_ik_controller.set_command(left_ik_commands)

        # Right arm: keep a fixed target (optional)
        right_pos_b, right_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            right_goal_pos_w, right_goal_quat_w
        )
        right_ik_commands[:, 0:3] = right_pos_b
        right_ik_commands[:, 3:7] = right_quat_b
        right_arm_ik_controller.set_command(right_ik_commands)

        # Compute IK joint targets
        left_joint_pos_des = left_arm_ik_controller.compute(left_ee_pos_b, left_ee_quat_b, left_jacobian, left_joint_pos)
        right_joint_pos_des = right_arm_ik_controller.compute(right_ee_pos_b, right_ee_quat_b, right_jacobian, right_joint_pos)

        # Full joint target (start from default)
        all_joint_pos_des = robot.data.default_joint_pos.clone()
        all_joint_pos_des[:, left_arm_cfg.joint_ids] = left_joint_pos_des
        all_joint_pos_des[:, right_arm_cfg.joint_ids] = right_joint_pos_des

        # Apply gripper targets
        for jid, name in zip(left_gripper_joint_ids, left_gripper_joint_names):
            all_joint_pos_des[:, jid] = grip_target.get(name, 0.0)
        for jid, name in zip(right_gripper_joint_ids, right_gripper_joint_names):
            all_joint_pos_des[:, jid] = GRIP_OPEN.get(name, 0.0)  # keep right gripper open

        # Send commands
        robot.set_joint_position_target(all_joint_pos_des)

        # Step sim
        scene.write_data_to_sim()
        if cube_attached and attach_offset_b is not None:
            desired_cube_pos = left_ee_pose_w[:, 0:3] + attach_offset_b
            write_cube_pose_w(desired_cube_pos, cube_quat_w)
        sim.step()
        scene.update(sim_dt)
        count += 1

        # Update markers (markers expect WORLD pose)
        left_ee_marker.visualize(left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7])
        left_goal_marker.visualize(target_pos_w, left_goal_quat_w)

        right_ee_marker.visualize(right_ee_pose_w[:, 0:3], right_ee_pose_w[:, 3:7])
        right_goal_marker.visualize(right_goal_pos_w, right_goal_quat_w)


def main() -> None:
    """Main function."""
    # 使用更高的物理步长 (240 Hz) 提高控制精度
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0/240.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view((3.5, 0.0, 3.2), (0.0, 0.0, 0.5))

    scene_cfg = NewRobotsSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
