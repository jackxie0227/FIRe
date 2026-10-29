# Xbox control of the Franka cube-lift simulation

Connect the Xbox controller over Bluetooth before launching. From the FIRe repository root:

```bash
source /home/rim/miniconda3/etc/profile.d/conda.sh
conda activate FIRe
python fire_lab/scripts/teleoperation/xbox_franka_teleop.py --debug-input
```

The command opens the Isaac Sim GUI with one `Isaac-Lift-Cube-Franka-IK-Rel-v0` environment. The main viewport shows the full scene, and a separate **Franka Wrist Camera** viewport follows the gripper in real time. You can move or resize that viewport while teleoperating. It reads the Xbox controller directly from Linux joydev, so Isaac Sim 5.1's `unknown remapping ... will be ignored` warning does not block teleoperation. Input values printed by `--debug-input` are ordered as `(dx, dy, dz, rx, ry, rz, gripper)`.

The task's goal-pose debug arrows are hidden in both viewports during teleoperation.
When the motion controls return to center, the controller keeps the last commanded end-effector pose and corrects drift under load. The X button still changes only the gripper state. With `--debug-input`, the terminal shows both the joystick input and the IK command, which may be nonzero while correcting drift.

| Control | Robot command |
| --- | --- |
| Left stick up/down | End effector +X/-X |
| Left stick right/left | End effector -Y/+Y |
| Right stick up/down | End effector +Z/-Z |
| Right stick right/left | Positive/negative yaw |
| D-pad left/right | Positive/negative roll |
| D-pad down/up | Positive/negative pitch |
| X | Toggle gripper open/closed |

A, B, Y, LB, RB, LT, and RT are unbound. The directions above use the robot frame, which may differ from the camera view. Release a stick to stop that motion. The gripper starts open.

The default sensitivity is `0.1` and analog dead zone is `0.08`. Use `--sensitivity VALUE`, `--dead-zone VALUE`, or `--joystick /dev/input/jsN` to adjust them. Disconnecting the controller ends the run with an input error; reconnect it and relaunch the command.
