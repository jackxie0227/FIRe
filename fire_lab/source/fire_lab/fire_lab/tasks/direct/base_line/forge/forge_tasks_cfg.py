# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from fire_lab.tasks.direct.base_line.factory.factory_tasks_cfg import FactoryTask, GearMesh, NutThread, PegInsert


@configclass
class ForgeTask(FactoryTask):
    curriculum_stage: int = 0
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    engaged_reward_scale: float = 1.0
    success_reward_scale: float = 1.0
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2


@configclass
class ForgePegInsertCurriculumStage1(ForgePegInsert):
    """Learn basic insertion from easy alignment and permissive force limits."""

    curriculum_stage: int = 1
    hand_init_pos_noise: list = [0.004, 0.004, 0.005]
    held_asset_pos_noise: list = [0.001, 0.0, 0.001]
    contact_penalty_threshold_range: list = [15.0, 20.0]
    contact_penalty_scale: float = 0.05
    engaged_reward_scale: float = 3.0
    success_reward_scale: float = 10.0


@configclass
class ForgePegInsertCurriculumStage2(ForgePegInsert):
    """Expand alignment variation while restoring moderate force penalties."""

    curriculum_stage: int = 2
    hand_init_pos_noise: list = [0.01, 0.01, 0.008]
    held_asset_pos_noise: list = [0.002, 0.0, 0.002]
    contact_penalty_threshold_range: list = [10.0, 15.0]
    contact_penalty_scale: float = 0.1
    engaged_reward_scale: float = 3.0
    success_reward_scale: float = 10.0


@configclass
class ForgePegInsertCurriculumStage3(ForgePegInsert):
    """Restore full pose variation and near-baseline contact penalties."""

    curriculum_stage: int = 3
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    held_asset_pos_noise: list = [0.003, 0.0, 0.003]
    contact_penalty_threshold_range: list = [8.0, 12.0]
    contact_penalty_scale: float = 0.2
    engaged_reward_scale: float = 3.0
    success_reward_scale: float = 10.0


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05
