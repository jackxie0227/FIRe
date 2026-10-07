# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from fire_lab.tasks.direct.base_line.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class ForgeGr00tTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]
    force_shaping_lambda = 0.01


@configclass
class ForgeGr00tPegInsert(PegInsert, ForgeGr00tTask):
    contact_penalty_scale: float = 0.0003  # 0.0025  # 0.01


@configclass
class ForgeGr00tPegInsertOfficial(PegInsert, ForgeGr00tTask):
    """PegInsert terms disclosed on the FIRe project page."""

    contact_penalty_scale: float = 1.0e-5
    contact_penalty_alpha: float = 1.0e-5
    force_shaping_lambda: float = 1.0e-5
    force_potential_c: float = 1.0
    contact_penalty_threshold_range = [5.0, 10.0]
    success_lateral_tolerance_m: float = 0.005
    success_depth_tolerance_m: float = 0.025


@configclass
class ForgeGr00tGearMesh(GearMesh, ForgeGr00tTask):
    contact_penalty_scale: float = 0.0025


@configclass
class ForgeGr00tNutThread(NutThread, ForgeGr00tTask):
    contact_penalty_scale: float = 0.0025
