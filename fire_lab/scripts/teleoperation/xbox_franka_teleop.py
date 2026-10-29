"""Teleoperate Isaac Lab's Franka cube lift scene with a Linux Xbox controller.

The Linux joydev input path bypasses Isaac Sim 5.1's outdated GLFW gamepad map.
Run from the FIRe repository root with the FIRe Conda environment activated.
"""

from __future__ import annotations

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Xbox teleoperation of a simulated Franka Panda arm.")
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-IK-Rel-v0", help="Isaac Lab task ID (must accept 7D SE(3) + gripper actions).")
parser.add_argument("--joystick", default=None, help="Linux /dev/input/js* path; auto-detect Xbox when omitted.")
parser.add_argument("--sensitivity", type=float, default=0.1, help="Motion sensitivity (default: 0.1).")
parser.add_argument("--dead-zone", type=float, default=0.08, help="Analog stick dead zone in [0, 1) (default: 0.08).")
parser.add_argument("--debug-input", action="store_true", help="Print current 7D input at most twice per second.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.headless:
    parser.error("This teleoperation script needs the Isaac Sim GUI; omit --headless.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from omni.kit.viewport.utility import create_viewport_window  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils.math import convert_camera_frame_orientation_convention  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.lift import mdp  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from linux_xbox_gamepad import JoystickDisconnected, LinuxXboxGamepad  # noqa: E402
from relative_pose_hold import RelativePoseHold  # noqa: E402

WRIST_CAMERA_PATH = "/World/envs/env_0/Robot/panda_hand/wrist_camera"


def add_wrist_camera() -> None:
    """Create a viewport camera on the hand without a second sensor render product."""
    wrist_pose_ros = torch.tensor([[0.7032332, -0.0739128, -0.0739128, 0.7032332]], dtype=torch.float32)
    wrist_pose_usd = convert_camera_frame_orientation_convention(
        wrist_pose_ros, origin="ros", target="opengl"
    )[0].tolist()
    camera_cfg = sim_utils.PinholeCameraCfg(
        focal_length=24.0,
        focus_distance=400.0,
        horizontal_aperture=20.955,
        vertical_aperture=20.955 * 360 / 480,
        clipping_range=(0.01, 100.0),
        lock_camera=False,
    )
    camera_cfg.func(WRIST_CAMERA_PATH, camera_cfg, translation=(0.05, 0.0, 0.0), orientation=wrist_pose_usd)


def open_wrist_viewport():
    """Show the moving wrist camera in a separate, movable viewport."""
    wrist_viewport = create_viewport_window(
        name="Franka Wrist Camera", width=480, height=360, position_x=850, position_y=80
    )
    if wrist_viewport is None:
        raise RuntimeError("Could not create the wrist camera viewport")
    wrist_viewport.viewport_api.camera_path = WRIST_CAMERA_PATH
    return wrist_viewport


def main() -> int:
    """Run the graphical Franka scene and feed it Xbox pose/gripper commands."""
    env = None
    gamepad = None
    wrist_viewport = None
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        env_cfg.env_name = args.task
        env_cfg.terminations.time_out = None
        if "Lift" in args.task:
            env_cfg.commands.object_pose.debug_vis = False
            env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
            env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        if env.action_manager.total_action_dim != 7:
            raise ValueError(f"Task {args.task} has {env.action_manager.total_action_dim} actions; expected 7")

        gamepad = LinuxXboxGamepad(args.joystick, args.sensitivity, args.dead_zone)
        print(f"Reading {gamepad.name} from {gamepad.path}", flush=True)
        print("Left stick: X/Y; right stick: Z/yaw; D-pad: roll/pitch; X: gripper toggle.", flush=True)
        print("Initializing Franka scene...", flush=True)
        env.reset()
        pose_hold = RelativePoseHold(env.action_manager.get_term("arm_action"), math_utils)
        print("Franka scene ready; creating wrist camera...", flush=True)
        add_wrist_camera()
        print("Wrist camera ready; opening its viewport...", flush=True)
        wrist_viewport = open_wrist_viewport()
        print("Wrist camera: separate Franka Wrist Camera viewport (movable).", flush=True)
        next_debug_time = 0.0

        while simulation_app.is_running():
            with torch.inference_mode():
                gamepad.poll()
                action = gamepad.state.action()
                actions = pose_hold.command(action)
                if args.debug_input and time.monotonic() >= next_debug_time:
                    print(
                        "joystick:", tuple(round(value, 4) for value in action),
                        "IK:", tuple(round(value, 4) for value in actions[0].tolist()),
                        flush=True,
                    )
                    next_debug_time = time.monotonic() + 0.5
                _, _, terminated, truncated, _ = env.step(actions)
                if torch.any(terminated | truncated).item():
                    pose_hold.reset()
    except JoystickDisconnected as exc:
        print(f"Input stopped: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("Teleoperation interrupted by keyboard.", flush=True)
        return 0
    finally:
        if wrist_viewport is not None:
            wrist_viewport.destroy()
        if gamepad is not None:
            gamepad.close()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
