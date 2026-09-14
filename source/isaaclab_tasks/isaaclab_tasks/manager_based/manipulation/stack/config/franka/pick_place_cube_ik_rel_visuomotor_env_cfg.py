# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from .stack_ik_rel_visuomotor_env_cfg import (
    FrankaCubeStackVisuomotorEnvCfg,
    ObservationsCfg as BaseObservationsCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class ThreeCameraObservationsCfg(BaseObservationsCfg):
    """Policy observations for the gripper, side, and top cameras."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        table_cam = None
        wrist_cam = None
        gripper_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("gripper_cam"), "data_type": "rgb", "normalize": False},
        )
        side_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("side_cam"), "data_type": "rgb", "normalize": False},
        )
        top_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def reset_fixed_pick_place_cubes(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Restore both cubes to the fixed task poses on every environment reset."""

    if env_ids is None:
        return
    fixed_poses = {
        "cube_1": (0.40, 0.00, 0.0203),
        "cube_2": (0.55, 0.05, 0.0203),
    }
    for asset_name, position in fixed_poses.items():
        asset: RigidObject = env.scene[asset_name]
        pose = torch.zeros((len(env_ids), 7), device=env.device)
        pose[:, :3] = torch.tensor(position, device=env.device) + env.scene.env_origins[env_ids]
        pose[:, 3] = 1.0
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def reset_blue_and_randomize_red_cube(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Restore blue to its fixed pose and sample a new red pose on every reset."""

    if env_ids is None:
        return
    blue: RigidObject = env.scene["cube_1"]
    blue_pose = torch.zeros((len(env_ids), 7), device=env.device)
    blue_pose[:, :3] = torch.tensor((0.40, 0.00, 0.0203), device=env.device) + env.scene.env_origins[env_ids]
    blue_pose[:, 3] = 1.0
    blue.write_root_pose_to_sim(blue_pose, env_ids=env_ids)
    blue.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)

    franka_stack_events.randomize_object_pose(
        env=env,
        env_ids=env_ids,
        asset_cfgs=[SceneEntityCfg("cube_2")],
        pose_range={
            # Wider training distribution while keeping the closest possible
            # red-blue center distance at 0.12 m (outside success at reset).
            "x": (0.52, 0.66),
            "y": (-0.20, 0.20),
            "z": (0.0203, 0.0203),
            "yaw": (-3.14159, 3.14159),
        },
    )


def red_cube_placed_next_to_blue(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    blue_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    red_cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    min_center_distance: float = 0.055,
    max_center_distance: float = 0.12,
    height_threshold: float = 0.015,
    max_linear_speed: float = 0.10,
) -> torch.Tensor:
    """Return success when the red cube is stably placed next to the blue cube."""

    robot: Articulation = env.scene[robot_cfg.name]
    blue_cube: RigidObject = env.scene[blue_cube_cfg.name]
    red_cube: RigidObject = env.scene[red_cube_cfg.name]

    center_distance = torch.linalg.vector_norm(
        red_cube.data.root_pos_w[:, :2] - blue_cube.data.root_pos_w[:, :2], dim=1
    )
    height_error = torch.abs(red_cube.data.root_pos_w[:, 2] - blue_cube.data.root_pos_w[:, 2])
    linear_speed = torch.linalg.vector_norm(red_cube.data.root_lin_vel_w, dim=1)

    beside_blue = (center_distance > min_center_distance) & (center_distance < max_center_distance)
    on_table = height_error < height_threshold
    stable = linear_speed < max_linear_speed

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    gripper_open = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, device=env.device),
            atol=env.cfg.gripper_threshold,
        ),
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, device=env.device),
            atol=env.cfg.gripper_threshold,
        ),
    )

    return beside_blue & on_table & stable & gripper_open


@configclass
class FrankaPickPlaceCubeFixedEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """Pick the red cube and place it beside the fixed blue cube."""

    observations: ThreeCameraObservationsCfg = ThreeCameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Convert the stock three-cube scene into this self-contained two-cube scene.
        self.scene.cube_3 = None
        self.observations.policy.object = None
        self.observations.policy.cube_positions = None
        self.observations.policy.cube_orientations = None
        self.observations.subtask_terms.grasp_2 = None
        self.terminations.cube_3_dropping = None

        # Replace the inherited cameras with the three views used by the dataset.
        self.scene.table_cam = None
        self.scene.wrist_cam = None
        self.scene.gripper_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/gripper_cam",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.015, 1.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),
                convention="ros",
            ),
        )
        self.scene.side_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/side_cam",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=16.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.5, -0.8, 0.35),
                rot=(-0.58319, 0.81234, 0.0, 0.0),
                convention="ros",
            ),
        )
        self.scene.top_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/top_cam",
            update_period=0.0,
            height=84,
            width=84,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.05, 0.30, 1.10),
                rot=(-0.14364, 0.49648, 0.82235, -0.23792),
                convention="ros",
            ),
        )
        self.image_obs_list = ["gripper_cam", "side_cam", "top_cam"]

        # Disabling the parent randomizer alone leaves moved cubes in place
        # across resets. Explicitly restore both fixed poses every episode.
        self.events.randomize_cube_positions = EventTerm(
            func=reset_fixed_pick_place_cubes,
            mode="reset",
        )

        self.terminations.success = DoneTerm(
            func=red_cube_placed_next_to_blue,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "blue_cube_cfg": SceneEntityCfg("cube_1"),
                "red_cube_cfg": SceneEntityCfg("cube_2"),
                "min_center_distance": 0.055,
                "max_center_distance": 0.12,
                "height_threshold": 0.015,
                "max_linear_speed": 0.10,
            },
        )

        # The old red-on-blue stacking annotation is not part of this task.
        self.observations.subtask_terms.stack_1 = None


@configclass
class FrankaPickPlaceCubeRandomEnvCfg(FrankaPickPlaceCubeFixedEnvCfg):
    """Pick-and-place task with only the red cube start position randomized."""

    def __post_init__(self):
        super().__post_init__()

        self.events.randomize_cube_positions = EventTerm(
            func=reset_blue_and_randomize_red_cube,
            mode="reset",
        )


@configclass
class FrankaPickPlaceCubeActJointPosFixedEnvCfg(FrankaPickPlaceCubeFixedEnvCfg):
    """ACT evaluation environment accepting absolute joint targets (7 arm + 1 gripper)."""

    def __post_init__(self):
        super().__post_init__()

        # The converted LeRobot action contains absolute arm joint positions.
        # Do not apply the offset/scale used by the stock delta joint controller.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )
        # The recorded command uses +1 for open and -1 for close.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


@configclass
class FrankaPickPlaceCubeActJointPosRandomEnvCfg(FrankaPickPlaceCubeActJointPosFixedEnvCfg):
    """ACT evaluation environment with the training-time red-cube randomization."""

    def __post_init__(self):
        super().__post_init__()

        self.events.randomize_cube_positions = EventTerm(
            func=reset_blue_and_randomize_red_cube,
            mode="reset",
        )
