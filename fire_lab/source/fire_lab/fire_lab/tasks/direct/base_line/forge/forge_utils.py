# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


DOWNWARD_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] without a discontinuous remainder operation."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quat_from_euler_xyz_wxyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert XYZ Euler angles to WXYZ quaternions without Isaac Sim dependencies."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        (cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
         cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp),
        dim=-1,
    )


def euler_xyz_from_quat_wxyz(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert WXYZ quaternions to XYZ Euler angles in their principal ranges."""
    qw, qx, qy, qz = quat.unbind(dim=-1)
    roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    sin_pitch = (2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0)
    pitch = torch.asin(sin_pitch)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def project_quat_to_downward_yaw(quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Project WXYZ quaternions onto the roll=pi, pitch=0, free-yaw manifold.

    A valid downward quaternion has the WXYZ form ``[0, x, y, 0]``.  The
    projection is normalized and degenerate/non-finite rows fall back to the
    zero-yaw downward pose ``[0, 1, 0, 0]``.
    """
    projected = quat.clone()
    projected[..., 0] = 0.0
    projected[..., 3] = 0.0
    norm = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    valid = torch.isfinite(projected).all(dim=-1, keepdim=True) & torch.isfinite(norm) & (norm > eps)
    normalized = projected / norm.clamp_min(eps)
    fallback = torch.tensor(DOWNWARD_QUAT_WXYZ, dtype=quat.dtype, device=quat.device).expand_as(quat)
    return torch.where(valid, normalized, fallback)


def downward_quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Construct normalized WXYZ quaternions with roll=pi and pitch=0.

    The resulting world-frame Euler yaw is ``-yaw``. This preserves FORGE's
    existing bolt-frame action convention (roll-pi quaternion multiplied by
    the policy's yaw quaternion).
    """
    half_yaw = 0.5 * yaw
    quat = torch.stack(
        (torch.zeros_like(yaw), torch.cos(half_yaw), -torch.sin(half_yaw), torch.zeros_like(yaw)), dim=-1
    )
    return torch.nn.functional.normalize(quat, dim=-1)


def clip_downward_orientation(
    current_quat: torch.Tensor,
    desired_yaw: torch.Tensor,
    rot_threshold: torch.Tensor,
    yaw_wrap_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip a downward orientation target and return it with continuous yaw error.

    ``current_quat`` and the returned quaternion use Isaac Sim 5.1's WXYZ
    convention.  Roll/pitch errors are wrapped before clipping, so equivalent
    -pi/+pi representations do not generate a full-turn target.
    """
    curr_roll, curr_pitch, curr_yaw = euler_xyz_from_quat_wxyz(current_quat)
    curr_yaw = yaw_wrap_fn(curr_yaw)
    desired_yaw = yaw_wrap_fn(desired_yaw)

    delta_yaw = desired_yaw - curr_yaw
    clipped_yaw = curr_yaw + torch.clamp(delta_yaw, -rot_threshold[:, 2], rot_threshold[:, 2])

    delta_roll = wrap_to_pi(torch.full_like(curr_roll, torch.pi) - curr_roll)
    delta_pitch = wrap_to_pi(-curr_pitch)
    clipped_roll = curr_roll + torch.clamp(delta_roll, -rot_threshold[:, 0], rot_threshold[:, 0])
    clipped_pitch = curr_pitch + torch.clamp(delta_pitch, -rot_threshold[:, 1], rot_threshold[:, 1])

    clipped_quat = quat_from_euler_xyz_wxyz(clipped_roll, clipped_pitch, clipped_yaw)
    return torch.nn.functional.normalize(clipped_quat, dim=-1), delta_yaw


def get_random_prop_gains(default_values, noise_levels, num_envs, device):
    """Helper function to randomize controller gains."""
    c_param_noise = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device)
    c_param_noise = c_param_noise @ torch.diag(torch.tensor(noise_levels, dtype=torch.float32, device=device))
    c_param_multiplier = 1.0 + c_param_noise
    decrease_param_flag = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device) > 0.5
    c_param_multiplier = torch.where(decrease_param_flag, 1.0 / c_param_multiplier, c_param_multiplier)

    prop_gains = default_values * c_param_multiplier

    return prop_gains


def change_FT_frame(source_F, source_T, source_frame, target_frame):
    """Convert force/torque reading from source to target frame."""
    # Import lazily so the pure quaternion helpers can be unit-tested without
    # starting the Isaac Sim application and loading its extension modules.
    import isaacsim.core.utils.torch as torch_utils

    # Modern Robotics eq. 3.95
    source_frame_inv = torch_utils.tf_inverse(source_frame[0], source_frame[1])
    target_T_source_quat, target_T_source_pos = torch_utils.tf_combine(
        source_frame_inv[0], source_frame_inv[1], target_frame[0], target_frame[1]
    )
    target_F = torch_utils.quat_apply(target_T_source_quat, source_F)
    target_T = torch_utils.quat_apply(
        target_T_source_quat, (source_T + torch.cross(target_T_source_pos, source_F, dim=-1))
    )
    return target_F, target_T
