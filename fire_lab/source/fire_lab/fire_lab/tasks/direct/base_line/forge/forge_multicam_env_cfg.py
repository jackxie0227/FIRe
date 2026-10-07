# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Camera-enabled configuration for visualizing the PPO teacher policy."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .forge_env_cfg import ForgeTaskPegInsertCfg


@configclass
class ForgeTaskPegInsertMulticamCfg(ForgeTaskPegInsertCfg):
    """Baseline PegInsert task with cameras that do not enter policy observations."""

    left_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/left_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.6, 0.6, 0.4),
            rot=(0.0, 0.0, -0.46175, -0.88701),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=256,
        height=256,
    )
    right_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/right_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.6, -0.6, 0.4),
            rot=(0.88701, 0.46175, 0.0, 0.0),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=256,
        height=256,
    )
    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.0),
            rot=(0.7032332, -0.0739128, -0.0739128, 0.7032332),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=256,
        height=256,
    )
