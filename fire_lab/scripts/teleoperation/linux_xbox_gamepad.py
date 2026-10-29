"""Read an Xbox controller through Linux joydev, bypassing Kit's GLFW mapping."""

from __future__ import annotations

import errno
import fcntl
import glob
import os
import struct
from pathlib import Path

from scipy.spatial.transform import Rotation

_EVENT = struct.Struct("<IhBB")
_BUTTON = 0x01
_AXIS = 0x02
_INIT = 0x80

# Linux input-event-codes.h values. Resolve their joydev indices at runtime.
_LEFT_X = 0x00
_LEFT_Y = 0x01
_RIGHT_X = 0x02
_RIGHT_Y = 0x05
_DPAD_X = 0x10
_DPAD_Y = 0x11
_X_BUTTON = 0x133
_REQUIRED_AXES = (_LEFT_X, _LEFT_Y, _RIGHT_X, _RIGHT_Y, _DPAD_X, _DPAD_Y)

_GET_AXES = 0x80016A11
_GET_BUTTONS = 0x80016A12
_GET_AXIS_MAP = 0x80406A32
_GET_BUTTON_MAP = 0x84006A34
_GET_NAME_128 = 0x80806A13


class JoystickDisconnected(RuntimeError):
    """The selected joystick is no longer available."""


def _device_info(fd: int) -> tuple[str, list[int], list[int]]:
    """Read a joydev device name and its axis/button event-code maps."""
    name_buf = bytearray(128)
    axis_count_buf = bytearray(1)
    button_count_buf = bytearray(1)
    axis_map_buf = bytearray(64)
    button_map_buf = bytearray(1024)
    for request, buf in (
        (_GET_NAME_128, name_buf),
        (_GET_AXES, axis_count_buf),
        (_GET_BUTTONS, button_count_buf),
        (_GET_AXIS_MAP, axis_map_buf),
        (_GET_BUTTON_MAP, button_map_buf),
    ):
        fcntl.ioctl(fd, request, buf, True)
    name = name_buf.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    axis_codes = list(axis_map_buf[: axis_count_buf[0]])
    button_codes = list(struct.unpack_from(f"<{button_count_buf[0]}H", button_map_buf))
    return name, axis_codes, button_codes


def _is_xbox(name: str) -> bool:
    lowered = name.lower()
    return "xbox" in lowered and "controller" in lowered


def find_xbox_joystick() -> Path:
    """Find the first Linux joystick whose reported name is an Xbox controller."""
    for candidate in sorted(glob.glob("/dev/input/js*")):
        try:
            fd = os.open(candidate, os.O_RDONLY | os.O_NONBLOCK)
            try:
                name, _, _ = _device_info(fd)
            finally:
                os.close(fd)
        except OSError:
            continue
        if _is_xbox(name):
            return Path(candidate)
    raise FileNotFoundError("No Xbox joystick found under /dev/input/js*. Check the Bluetooth connection.")


class XboxState:
    """Translate joydev events into the seven actions expected by Franka lift IK."""

    def __init__(
        self,
        axis_codes: list[int],
        button_codes: list[int],
        sensitivity: float = 0.1,
        dead_zone: float = 0.08,
    ) -> None:
        if sensitivity <= 0:
            raise ValueError("sensitivity must be positive")
        if not 0 <= dead_zone < 1:
            raise ValueError("dead_zone must be in [0, 1)")
        missing = [code for code in _REQUIRED_AXES if code not in axis_codes]
        if missing or _X_BUTTON not in button_codes:
            raise ValueError(f"Joystick lacks required Xbox controls: axes={missing}, X={_X_BUTTON in button_codes}")
        self.axis_codes: list[int] = axis_codes
        self.button_codes: list[int] = button_codes
        self.sensitivity: float = sensitivity
        self.dead_zone: float = dead_zone
        self.axes: dict[int, float] = {code: 0.0 for code in _REQUIRED_AXES}
        self.close_gripper: bool = False
        self._x_down: bool = False

    def process_event(self, raw: bytes) -> None:
        """Update state from one eight-byte Linux ``js_event``."""
        if len(raw) != _EVENT.size:
            raise ValueError(f"Expected {_EVENT.size} joystick event bytes, got {len(raw)}")
        _, value, event_type, number = _EVENT.unpack(raw)
        is_initial = bool(event_type & _INIT)
        event_type &= ~_INIT
        if event_type == _AXIS and number < len(self.axis_codes):
            code = self.axis_codes[number]
            if code in self.axes:
                normalized = max(-1.0, min(1.0, value / 32767.0))
                self.axes[code] = 0.0 if abs(normalized) < self.dead_zone else normalized
        elif event_type == _BUTTON and number < len(self.button_codes):
            if self.button_codes[number] == _X_BUTTON:
                pressed = value != 0
                if pressed and not self._x_down and not is_initial:
                    self.close_gripper = not self.close_gripper
                self._x_down = pressed

    def action(self) -> tuple[float, float, float, float, float, float, float]:
        """Return (dx, dy, dz, rx, ry, rz, gripper) for the relative IK task."""
        pos_gain = 0.1 * self.sensitivity
        rot_gain = 0.1 * self.sensitivity
        dx = -self.axes[_LEFT_Y] * pos_gain
        dy = -self.axes[_LEFT_X] * pos_gain
        dz = -self.axes[_RIGHT_Y] * pos_gain
        roll = -self.axes[_DPAD_X] * rot_gain * 0.8
        pitch = self.axes[_DPAD_Y] * rot_gain * 0.8
        yaw = self.axes[_RIGHT_X] * rot_gain
        rx, ry, rz = Rotation.from_euler("XYZ", (roll, pitch, yaw)).as_rotvec()
        gripper = -1.0 if self.close_gripper else 1.0
        return dx, dy, dz, float(rx), float(ry), float(rz), gripper


class LinuxXboxGamepad:
    """Own a nonblocking joydev handle and produce Franka teleoperation actions."""

    def __init__(self, path: str | None = None, sensitivity: float = 0.1, dead_zone: float = 0.08) -> None:
        self.path: Path = Path(path) if path is not None else find_xbox_joystick()
        self._fd: int = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        try:
            self.name, axis_codes, button_codes = _device_info(self._fd)
            if not _is_xbox(self.name):
                raise ValueError(f"{self.path} is {self.name!r}, not an Xbox controller")
            self.state: XboxState = XboxState(axis_codes, button_codes, sensitivity, dead_zone)
        except BaseException:
            os.close(self._fd)
            raise

    def poll(self) -> None:
        """Drain all pending input without blocking the simulation loop."""
        while True:
            try:
                raw = os.read(self._fd, _EVENT.size)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno in (errno.ENODEV, errno.EIO, errno.EBADF):
                    raise JoystickDisconnected(f"Xbox controller disconnected: {self.path}") from exc
                raise
            if not raw:
                raise JoystickDisconnected(f"Xbox controller disconnected: {self.path}")
            self.state.process_event(raw)

    def close(self) -> None:
        """Release the joystick device."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
