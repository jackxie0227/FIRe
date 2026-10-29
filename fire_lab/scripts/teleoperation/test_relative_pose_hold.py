"""Check that neutral gamepad input holds a fixed Franka tool pose."""

from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation

from relative_pose_hold import RelativePoseHold


def _rotation(quaternion: torch.Tensor) -> Rotation:
    return Rotation.from_quat(quaternion[0, [1, 2, 3, 0]].numpy())


def _quaternion(rotation: Rotation) -> torch.Tensor:
    xyzw = rotation.as_quat()
    return torch.tensor([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]], dtype=torch.float32)


class FakeMath:
    @staticmethod
    def apply_delta_pose(position, orientation, delta):
        rotation = Rotation.from_rotvec(delta[0, 3:6].numpy()) * _rotation(orientation)
        return position + delta[:, :3], _quaternion(rotation)

    @staticmethod
    def compute_pose_error(position, orientation, target_position, target_orientation):
        rotation = _rotation(target_orientation) * _rotation(orientation).inv()
        return target_position - position, torch.tensor(rotation.as_rotvec(), dtype=torch.float32).unsqueeze(0)


class FakeArmAction:
    def __init__(self):
        self._scale = torch.full((1, 6), 0.5)
        self.position = torch.tensor([[0.4, 0.0, 0.3]])
        self.orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    def _compute_frame_pose(self):
        return self.position, self.orientation


def test_neutral_input_corrects_downward_drift_and_preserves_gripper():
    arm = FakeArmAction()
    hold = RelativePoseHold(arm, FakeMath)
    arm.position[:, 2] -= 0.004

    command = hold.command((0, 0, 0, 0, 0, 0, -1))

    assert torch.allclose(command[0, :3], torch.tensor([0.0, 0.0, 0.008]), atol=1e-6)
    assert command[0, 6] == -1


def test_horizontal_motion_keeps_correcting_uncommanded_height():
    arm = FakeArmAction()
    hold = RelativePoseHold(arm, FakeMath)
    arm.position[:, 2] -= 0.004

    command = hold.command((0.01, 0, 0, 0, 0, 0, 1))

    assert torch.allclose(command[0, :3], torch.tensor([0.01, 0.0, 0.008]), atol=1e-6)


def test_moving_reanchors_target_and_neutral_input_holds_orientation():
    arm = FakeArmAction()
    hold = RelativePoseHold(arm, FakeMath)
    moving = hold.command((0, 0, 0.01, 0, 0, 0.01, 1))
    assert torch.allclose(moving[0, 2:6], torch.tensor([0.01, 0.0, 0.0, 0.01]), atol=1e-5)

    arm.position[:, 2] += 0.005
    arm.orientation = _quaternion(Rotation.from_rotvec([0, 0, 0.005]))
    neutral = hold.command((0, 0, 0, 0, 0, 0, 1))
    assert torch.allclose(neutral[0, :6], torch.zeros(6), atol=1e-5)

    arm.position[:, 2] -= 0.002
    arm.orientation = _quaternion(Rotation.from_rotvec([0, 0, 0.002]))
    correction = hold.command((0, 0, 0, 0, 0, 0, 1))
    assert correction[0, 2] > 0
    assert correction[0, 5] > 0


def test_reset_recaptures_pose_after_scene_reset():
    arm = FakeArmAction()
    hold = RelativePoseHold(arm, FakeMath)
    arm.position[:, 2] += 0.1
    hold.reset()
    assert torch.allclose(hold.command((0, 0, 0, 0, 0, 0, 1))[0, :6], torch.zeros(6), atol=1e-6)
