"""Keep the Franka end effector at its last commanded pose between gamepad inputs."""

from __future__ import annotations

import torch


class RelativePoseHold:
    """Turn joystick deltas into relative IK commands with a stationary pose target."""

    def __init__(self, arm_action, math_utils):
        self.arm_action = arm_action
        self.math = math_utils
        self.scale = arm_action._scale.clone()
        if self.scale.shape != (1, 6) or torch.any(self.scale <= 0):
            raise ValueError("Pose hold requires one Franka environment with positive six-axis IK scales")
        self.reset()

    def reset(self) -> None:
        """Latch the current IK tool pose after a scene reset."""
        position, orientation = self.arm_action._compute_frame_pose()
        self.target_position = position.clone()
        self.target_orientation = orientation.clone()

    def command(self, gamepad_action: tuple[float, ...]) -> torch.Tensor:
        """Preserve joystick motion and correct any drift when the sticks return to center."""
        action = torch.tensor(gamepad_action, dtype=self.scale.dtype, device=self.scale.device).unsqueeze(0)
        position, orientation = self.arm_action._compute_frame_pose()

        # Re-anchor only commanded axes: moving X/Y must still hold the Z height.
        if torch.any(action[:, :6] != 0).item():
            new_position, new_orientation = self.math.apply_delta_pose(position, orientation, action[:, :6] * self.scale)
            self.target_position = torch.where(action[:, :3] != 0, new_position, self.target_position)
            if torch.any(action[:, 3:6] != 0).item():
                self.target_orientation = new_orientation

        position_error, rotation_error = self.math.compute_pose_error(
            position, orientation, self.target_position, self.target_orientation
        )
        # Bound each correction to avoid a sudden jump after a large external disturbance.
        position_error = torch.clamp(position_error, -0.01, 0.01)
        rotation_error = torch.clamp(rotation_error, -0.02, 0.02)
        action[:, :6] = torch.cat((position_error, rotation_error), dim=1) / self.scale
        return action
