# Linux Xbox 手柄遥操作 Franka：实现与排障记录

本文记录 2026-10-25 在 FIRe 中使用蓝牙 Xbox Wireless Controller 遥操作 Isaac Sim 5.1 的 Franka Panda 方块抓取场景时，已经验证的做法。后续 agent 编写其他遥操作脚本时，先检查本文件和 `scripts/teleoperation/` 中的现有实现。本方案只控制仿真机械臂，不连接真实机械臂，也不修改 Isaac Lab 或 Isaac Sim 的安装文件。

## 直接启动

先连接手柄。从任何终端目录运行以下命令；这里使用脚本相对路径，因此需要先 `cd` 到仓库根目录（也可以直接使用脚本绝对路径）：

```bash
source /home/rim/miniconda3/etc/profile.d/conda.sh
conda activate FIRe
cd /home/rim/experiment/FIRe
python fire_lab/scripts/teleoperation/xbox_franka_teleop.py --debug-input
```

默认任务是 `Isaac-Lift-Cube-Franka-IK-Rel-v0`，只创建一个环境。Isaac Sim GUI 中保留主场景视口，并另开可移动的 **Franka Wrist Camera** 视口。可用 `--sensitivity 0.05` 降低速度，或用 `--sensitivity 0.2` 提高速度；`--dead-zone` 默认 `0.08`，`--joystick /dev/input/jsN` 可指定设备。`--debug-input` 每约半秒打印一次原始七维手柄动作和最终 IK 命令。手柄断开时脚本报错退出，重新连接后再启动。

## 代码入口与职责

| 文件 | 职责 |
| --- | --- |
| `scripts/teleoperation/xbox_franka_teleop.py` | 启动 GUI、配置场景、接入手柄、创建腕部视口、逐步调用 `env.step()`，在环境自动重置后重新锁定末端位姿。 |
| `scripts/teleoperation/linux_xbox_gamepad.py` | 通过 Linux joydev 读取手柄，处理设备查找、轴和按键编号、死区、X 键沿触发、断连，并生成七维动作。 |
| `scripts/teleoperation/relative_pose_hold.py` | 保存末端目标位姿；摇杆回中时持续给相对 IK 发送误差修正，避免负载下末端缓慢下沉。 |
| `scripts/teleoperation/test_linux_xbox_gamepad.py`、`scripts/teleoperation/test_relative_pose_hold.py` | 输入映射和位姿保持的行为测试。 |

## 为什么直接读取 `/dev/input/js*`

本机手柄通过蓝牙显示为 `Xbox Wireless Controller`，曾报告设备 ID `045e:0b13`。Isaac Sim 5.1 的 GLFW 层曾输出 `joystick with unknown remapping ... will be ignored`，导致手柄已连接、Linux 能读到输入，但 Isaac Sim 内置手柄接口没有动作。设置 `SDL_GAMECONTROLLERCONFIG` 并不能保证修复 GLFW 路径。

当前实现绕过 GLFW，用非阻塞方式打开 `/dev/input/js*`。通过 ioctl 读取设备名称、轴代码表和按键代码表，再把 Linux **事件代码**映射为该设备当前的 joydev 序号。不要把 `js_event.number` 当作固定键位：序号可能随设备和驱动变化。事件格式是 8 字节 `<IhBB`；轴值归一化到 `[-1, 1]`，小于死区的值置零。X 键只在非初始化事件的按下沿切换一次夹爪，避免长按时反复开合。`poll()` 每步排空当前事件但不阻塞仿真；读取返回空数据或 `ENODEV`、`EIO`、`EBADF` 时视为断连。

手柄无可用陀螺仪或空间位置输入，本路径只能读取摇杆、扳机、方向键和按钮；不能通过在空中移动或旋转这只 Xbox 手柄直接获得 6DoF 位姿。

## 当前键位和七维动作

动作顺序是 `(dx, dy, dz, rx, ry, rz, gripper)`。前三维是机械臂根坐标系中的平移增量；`rx, ry, rz` 是轴角旋转增量；第七维 `+1` 为张开，`-1` 为闭合。屏幕或腕部摄像头中的方向不一定与机械臂根坐标系一致。

| 输入 | 末端动作 |
| --- | --- |
| 左摇杆上／下 | `+X`／`-X` 平移 |
| 左摇杆右／左 | `-Y`／`+Y` 平移；2026-10-25 按使用偏好反转过此轴 |
| 右摇杆上／下 | `+Z`／`-Z` 平移 |
| 右摇杆右／左 | 正／负偏航 |
| 方向键左／右 | 正／负滚转 |
| 方向键下／上 | 正／负俯仰 |
| X | 单次按下切换夹爪开合 |

`A`、`B`、`Y`、`LB`、`RB`、`LT`、`RT` 当前未绑定。更改键位时先修改 `XboxState.action()`，再同步测试和本表；不要修改 Isaac Lab 内置手柄代码。

默认 `sensitivity=0.1`。脚本将摇杆量乘以 `0.1 * sensitivity`；任务中的 `arm_action.scale=0.5` 再缩放一次。满量程单轴平移或旋转指令因此分别约为每控制步 `0.005 m`、`0.005 rad`。场景 `sim.dt=0.01`、`decimation=2`，标称 50 个控制步/秒，对应约 `0.25 m/s` 或 `0.25 rad/s` 的**指令变化率**；实际运动受 IK、关节限制、PD 跟踪和实时仿真速度影响。`--dead-zone` 改变回中阈值，不改变速度。

## 末端位姿保持：不要把零增量当成保持命令

任务的 `DifferentialInverseKinematicsActionCfg` 使用 `command_type="pose"` 和 `use_relative_mode=True`。每次 `env.step()`，相对 IK 都从**当前测得**的末端位姿解释输入增量。若直接把回中后的六维零动作重复送入，目标会随末端下沉而重新锚定，不能保证持续保持夹起方块时的高度。这不是简单地增大摇杆速度或关闭重力能解决的问题；当前 Franka 高 PD 资产配置本身也设置了 `disable_gravity=True`。

`RelativePoseHold` 在 `env.reset()` 后读取 `arm_action._compute_frame_pose()` 得到与 IK 一致的机器人根坐标系及工具偏置位姿。对有摇杆输入的平移轴，以当前位姿加增量更新相应目标轴；未输入的轴保持先前目标，例如只移动 `X/Y` 时仍保持 `Z`。有姿态输入时更新姿态目标。然后计算当前位姿到目标位姿的误差，除以 IK 的 `_scale`，把修正量作为相对动作送入 `env.step()`。每步平移修正按轴限制在 `±0.01 m`、姿态修正限制在 `±0.02 rad`，以免外部扰动后突然跳动。夹爪动作原样透传。环境因终止条件自动重置后，必须调用 `pose_hold.reset()` 捕获新位姿，避免追逐旧场景中的目标。

这段实现使用了 Isaac Lab 动作项的私有接口 `_compute_frame_pose()` 和 `_scale`。升级 Isaac Lab 后，先核对它们的形状、坐标系和缩放语义，并运行相关测试。`--debug-input` 下，摇杆六维为零但 IK 六维非零，可能正是位姿保持在纠偏。

若**夹爪和方块一起下降**，优先检查末端保持与关节跟踪；若**夹爪停住、只有方块在指间下滑**，再检查夹爪闭合、接触和摩擦。不要把两种现象混为重力补偿问题。位姿保持已通过逻辑测试和 GUI 启动检查；夹取后长时间保持的效果仍需在实际遥操作中观察。

## 腕部摄像头与 GUI 的注意事项

腕部视角直接在 `/World/envs/env_0/Robot/panda_hand/wrist_camera` 创建 USD `PinholeCameraCfg`，沿用 FIRe 现有 Franka 腕部相机安装偏置：位置 `(0.05, 0, 0)`，ROS 四元数 `(0.7032332, -0.0739128, -0.0739128, 0.7032332)`。设置 USD 相机前要把 ROS 朝向转换到 OpenGL/USD 相机约定。相机在 `env.reset()` 后创建，第二视口的 `viewport_api.camera_path` 指向该 prim。此处只需要 GUI 视口，不需要额外的 `CameraCfg` 传感器和 Replicator 渲染产品，也不需要强制 `--enable_cameras`。

在这台机器的 Isaac Sim 5.1 上，创建第二视口后立即执行 `wrist_viewport.dock_in(main_viewport, ...)` 曾让应用直接进入关闭流程，终端没有 Python traceback。已改为独立可拖动视口；不要未经实测就恢复自动停靠。为避免腕部画面被红绿黄的目标坐标箭头遮挡，在创建环境前设置 `env_cfg.commands.object_pose.debug_vis=False`。只修改这个遥操作运行的配置，保持其他任务的可视化设置不变。

## 编写新遥操作脚本时的检查顺序

1. 确认新任务的动作维度、动作顺序、IK 相对／绝对模式、机器人根坐标系和动作缩放；当前 `RelativePoseHold` 只支持单环境、六维相对位姿加一维夹爪。
2. 复用 joydev 设备查找及事件代码映射，先用 `--debug-input` 证明动作进入 Python，再检查机械臂响应。`unknown remapping` 警告可能继续出现，但不会阻断此输入路径。
3. 逐轴验证方向，特别是已反转的左摇杆左右；松开后验证零输入和负载下的末端保持。
4. 若有自动重置，重置后重新捕获末端目标；断开手柄时停止输入并释放设备句柄。
5. 在 GUI 中逐轴观察主视口和腕部视口，再尝试夹取。若新增相机视口，先避免自动停靠和不必要的额外传感器渲染。

运行静态行为测试：

```bash
cd /home/rim/experiment/FIRe
/home/rim/miniconda3/envs/FIRe/bin/python -m pytest -q \
  fire_lab/scripts/teleoperation/test_linux_xbox_gamepad.py \
  fire_lab/scripts/teleoperation/test_relative_pose_hold.py
```

这 8 项测试覆盖六个运动方向、回中、X 键单次切换、断连、负载位姿纠偏和环境重置。真实 GUI 画面、夹持稳定性及长期保持仍需人工观察。
