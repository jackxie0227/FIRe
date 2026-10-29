"""Behavior checks for the Linux Xbox-to-Franka input mapping."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

import linux_xbox_gamepad as gamepad_module
from linux_xbox_gamepad import JoystickDisconnected, LinuxXboxGamepad, XboxState


AXES = [0, 1, 2, 5, 9, 10, 16, 17]
BUTTONS = list(range(304, 319))


def event(value: int, kind: int, number: int) -> bytes:
    return struct.pack("<IhBB", 0, value, kind, number)


def test_six_motion_axes_and_neutral() -> None:
    state = XboxState(AXES, BUTTONS)
    expected = {
        0: (1, 1),   # left stick right -> -Y
        1: (0, -1),  # left stick up -> +X
        2: (5, 1),   # right stick right -> +yaw
        3: (2, -1),  # right stick up -> +Z
        6: (3, 1),   # D-pad right -> -roll
        7: (4, -1),  # D-pad up -> -pitch
    }
    for axis_index, (action_index, raw_sign) in expected.items():
        state.process_event(event(32767 * raw_sign, 0x02, axis_index))
        action = state.action()
        assert action[action_index] != 0
        if axis_index in (0, 6, 7):
            assert action[action_index] < 0
        else:
            assert action[action_index] > 0
        state.process_event(event(0, 0x02, axis_index))
        assert state.action()[:6] == pytest.approx((0.0,) * 6)


def test_dead_zone_and_x_press_edge() -> None:
    state = XboxState(AXES, BUTTONS)
    state.process_event(event(1000, 0x02, 0))
    assert state.action()[:6] == pytest.approx((0.0,) * 6)
    x_index = BUTTONS.index(0x133)
    state.process_event(event(0, 0x01 | 0x80, x_index))
    assert state.action()[6] == 1.0
    state.process_event(event(1, 0x01, x_index))
    state.process_event(event(1, 0x01, x_index))
    assert state.action()[6] == -1.0
    state.process_event(event(0, 0x01, x_index))
    state.process_event(event(1, 0x01, x_index))
    assert state.action()[6] == 1.0


def test_missing_control_rejected() -> None:
    with pytest.raises(ValueError, match="lacks required"):
        XboxState(AXES[:-1], BUTTONS)


def test_disconnect_stops_input(monkeypatch: pytest.MonkeyPatch) -> None:
    device = LinuxXboxGamepad.__new__(LinuxXboxGamepad)
    device.path = Path("/dev/input/js0")
    device._fd = 1234
    monkeypatch.setattr(gamepad_module.os, "read", lambda _fd, _size: b"")
    with pytest.raises(JoystickDisconnected):
        device.poll()
