"""OpenArm-left adaptation of the existing Forge PegInsert environment."""

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

from fire_lab.tasks.direct.base_line.factory import factory_control

from .forge_env import ForgeEnv
from .forge_env_cfg import ForgeTaskPegInsertCfg


@configclass
class OpenArmForgePegInsertCfg(ForgeTaskPegInsertCfg):
    """Use the left hand in the reachable region validated by the OpenArm Reach task."""

    table_translation: tuple[float, float, float] = (0.22, 0.20, 0.25)
    arm_joint_names: list[str] = [f"openarm_left_joint{i}" for i in range(1, 8)]
    gripper_joint_names: list[str] = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
    left_finger_body_name: str = "openarm_left_hand"
    right_finger_body_name: str = "openarm_left_hand"
    fingertip_body_name: str = "openarm_left_hand"
    force_sensor_body_name: str = "openarm_left_hand"
    tcp_offset_pos: tuple[float, float, float] = (0.0, 0.0, 0.015)
    # Hand-down Reach orientation multiplied by this rotation yields Forge's downward TCP frame.
    tcp_offset_quat: tuple[float, float, float, float] = (-0.70710678, 0.0, 0.70710678, 0.0)
    reset_ik_pos_tolerance: float = 0.003
    reset_ik_rot_tolerance: float = 0.02

    def __post_init__(self) -> None:
        super().__post_init__()
        from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import OPEN_ARM_HIGH_PD_CFG

        self.scene.clone_in_fabric = False
        self.robot = OPEN_ARM_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.robot.spawn.activate_contact_sensors = True
        self.task.fixed_asset.init_state.pos = (0.22, 0.20, 0.30)
        self.task.fixed_asset_init_pos_noise = [0.02, 0.02, 0.0]
        self.task.hand_init_orn_noise = [0.0, 0.0, 0.20]
        self.task.robot_cfg.franka_fingerpad_length = 0.010
        self.ctrl.reset_joints = [0.0784, -0.1756, 0.0141, 1.0588, -0.0150, -0.2837, -0.6117]


class OpenArmForgePegInsertEnv(ForgeEnv):
    cfg: OpenArmForgePegInsertCfg

    def __init__(self, cfg: OpenArmForgePegInsertCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.force_bias = torch.zeros((self.num_envs, 6), device=self.device)

    def _init_tensors(self) -> None:
        super()._init_tensors()
        self.ik_controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.05},
            ),
            num_envs=self.num_envs,
            device=self.device,
        )

    def _read_force_sensor_world(self) -> torch.Tensor:
        raw = super()._read_force_sensor_world()
        return raw - self.force_bias

    def randomize_initial_state(self, env_ids: torch.Tensor) -> None:
        super().randomize_initial_state(env_ids)
        self.force_bias[env_ids] = self._robot.root_physx_view.get_link_incoming_joint_force()[
            env_ids, self.force_sensor_body_idx
        ]

    def _set_franka_to_default_pose(self, joints: list[float], env_ids: torch.Tensor) -> None:
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos[:, self.arm_joint_ids] = torch.tensor(joints, device=self.device)
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2.0 * 1.25
        joint_pos[:, self.gripper_joint_ids] = gripper_width
        joint_vel = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.reset(env_ids)
        self.step_sim_no_action()

    def _command_tcp_pose(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        gripper_pos: float,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        target_world_pos = target_pos + self.scene.env_origins
        target_root_pos, target_root_quat = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            target_world_pos,
            target_quat,
        )
        current_world_pos = self.fingertip_midpoint_pos + self.scene.env_origins
        current_root_pos, current_root_quat = subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            current_world_pos,
            self.fingertip_midpoint_quat,
        )
        root_rotation = matrix_from_quat(quat_inv(self._robot.data.root_quat_w))
        jacobian = self.fingertip_midpoint_jacobian.clone()
        jacobian[:, :3] = root_rotation @ jacobian[:, :3]
        jacobian[:, 3:] = root_rotation @ jacobian[:, 3:]
        self.ik_controller.set_command(torch.cat((target_root_pos, target_root_quat), dim=-1))
        desired = self.ik_controller.compute(
            current_root_pos,
            current_root_quat,
            jacobian,
            self._robot.data.joint_pos[:, self.arm_joint_ids],
        )
        limits = self._robot.data.soft_joint_pos_limits[:, self.arm_joint_ids]
        desired = desired.clamp(min=limits[..., 0] + 0.01, max=limits[..., 1] - 0.01)
        if not torch.isfinite(desired).all():
            raise RuntimeError("OpenArm IK produced non-finite joint targets")
        if env_ids is None:
            self._robot.set_joint_position_target(desired, joint_ids=self.arm_joint_ids)
            self._robot.set_joint_position_target(
                torch.full((self.num_envs, 2), gripper_pos, device=self.device),
                joint_ids=self.gripper_joint_ids,
            )
        else:
            self._robot.set_joint_position_target(desired[env_ids], joint_ids=self.arm_joint_ids, env_ids=env_ids)
            self._robot.set_joint_position_target(
                torch.full((len(env_ids), 2), gripper_pos, device=self.device),
                joint_ids=self.gripper_joint_ids,
                env_ids=env_ids,
            )

    def set_pos_inverse_kinematics(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2.0 * 1.25
        for _ in range(round(0.75 / self.physics_dt)):
            self._command_tcp_pose(
                ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat,
                gripper_width,
                env_ids,
            )
            self.step_sim_no_action()
        pos_error, rot_error = factory_control.get_pose_error(
            self.fingertip_midpoint_pos[env_ids],
            self.fingertip_midpoint_quat[env_ids],
            ctrl_target_fingertip_midpoint_pos[env_ids],
            ctrl_target_fingertip_midpoint_quat[env_ids],
            jacobian_type="geometric",
            rot_error_type="axis_angle",
        )
        return pos_error, rot_error

    def generate_ctrl_signals(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        ctrl_target_gripper_dof_pos: float,
    ) -> None:
        self._command_tcp_pose(
            ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos,
        )
