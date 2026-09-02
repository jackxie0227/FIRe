# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[1] / "fire_lab/tasks/direct/base_line/forge/forge_utils.py"
_SPEC = importlib.util.spec_from_file_location("forge_utils_under_test", _MODULE_PATH)
forge_utils = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(forge_utils)


def _wrap_forge_yaw(angle):
    return torch.where(angle > math.radians(235.0), angle - 2.0 * math.pi, angle)


@pytest.mark.parametrize("yaw", [-math.pi, -2.1, -0.1, 0.0, 1.7, math.pi])
def test_downward_projection_is_normalized_and_preserves_yaw(yaw):
    source = forge_utils.downward_quat_from_yaw(torch.tensor([yaw])) * 3.0
    projected = forge_utils.project_quat_to_downward_yaw(source)
    assert torch.allclose(torch.linalg.vector_norm(projected, dim=-1), torch.ones(1), atol=1.0e-6)
    assert torch.allclose(projected.abs(), forge_utils.downward_quat_from_yaw(torch.tensor([yaw])).abs(), atol=1.0e-6)


def test_projection_handles_upright_degenerate_and_nonfinite_inputs():
    source = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0, 0.0]]
    )
    projected = forge_utils.project_quat_to_downward_yaw(source)
    expected = torch.tensor(forge_utils.DOWNWARD_QUAT_WXYZ).repeat(3, 1)
    assert torch.isfinite(projected).all()
    assert torch.allclose(projected, expected)


def test_roll_boundary_clipping_is_continuous():
    eps = 1.0e-4
    current = forge_utils.quat_from_euler_xyz_wxyz(
        torch.tensor([-math.pi + eps]), torch.zeros(1), torch.zeros(1)
    )
    threshold = torch.full((1, 3), 0.05)
    clipped, _ = forge_utils.clip_downward_orientation(
        current, torch.zeros(1), threshold, _wrap_forge_yaw
    )
    roll, pitch, _ = forge_utils.euler_xyz_from_quat_wxyz(clipped)
    assert torch.abs(forge_utils.wrap_to_pi(roll - math.pi)).item() < 5.0e-4
    assert torch.abs(forge_utils.wrap_to_pi(pitch)).item() < 5.0e-4


def test_yaw_step_is_thresholded():
    current = forge_utils.downward_quat_from_yaw(torch.tensor([0.0]))
    threshold = torch.tensor([[0.05, 0.05, 0.1]])
    clipped, delta_yaw = forge_utils.clip_downward_orientation(
        current, torch.tensor([1.0]), threshold, _wrap_forge_yaw
    )
    _, _, clipped_yaw = forge_utils.euler_xyz_from_quat_wxyz(clipped)
    assert delta_yaw.item() == pytest.approx(1.0, abs=1.0e-5)
    assert abs(forge_utils.wrap_to_pi(clipped_yaw).item()) <= 0.10001


def test_only_yaw_changes_downward_target():
    yaw = torch.tensor([-1.2, 0.7])
    first = forge_utils.downward_quat_from_yaw(yaw)
    second = forge_utils.downward_quat_from_yaw(yaw)
    assert torch.allclose(first, second)
    roll, pitch, recovered_yaw = forge_utils.euler_xyz_from_quat_wxyz(first)
    assert torch.all(torch.abs(forge_utils.wrap_to_pi(roll - math.pi)) < 1.0e-5)
    assert torch.all(torch.abs(forge_utils.wrap_to_pi(pitch)) < 1.0e-5)
    assert torch.all(torch.abs(forge_utils.wrap_to_pi(recovered_yaw + yaw)) < 1.0e-5)
