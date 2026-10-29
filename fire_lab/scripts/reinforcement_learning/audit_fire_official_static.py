#!/usr/bin/env python3
"""Load Isaac and audit the registered FIRe-official task contracts."""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import gymnasium as gym
import yaml

from fire_lab.tasks.direct.base_line.factory.factory_env_cfg import OBS_DIM_CFG
from fire_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_env_cfg import (
    ForgeTaskPegInsertGr00tCfg,
    ForgeTaskPegInsertGr00tDemoSaveCfg,
    ForgeTaskPegInsertGr00tOfficialCfg,
)
import fire_lab.tasks  # noqa: F401,E402


TASK_EXACT = "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-Direct-v1"
TASK_SAMPLE_EQ = "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-SampleEq4-Direct-v1"


def _actor_dim(cfg) -> int:
    return sum(OBS_DIM_CFG[name] for name in cfg.obs_order) + cfg.action_space


def _load_agent(task_id: str) -> dict:
    entry = Path(gym.spec(task_id).kwargs["rl_games_cfg_entry_point"].split(":", 1)[1])
    agents_dir = Path(__file__).parents[2] / "source/fire_lab/fire_lab/tasks/direct/vla/gr00t/forge/agents"
    return yaml.safe_load((agents_dir / entry.name).read_text())["params"]["config"]


def main() -> None:
    official = ForgeTaskPegInsertGr00tOfficialCfg()
    legacy = ForgeTaskPegInsertGr00tCfg()
    demo = ForgeTaskPegInsertGr00tDemoSaveCfg()
    exact_agent = _load_agent(TASK_EXACT)
    sample_agent = _load_agent(TASK_SAMPLE_EQ)

    audit = {
        "tasks_registered": [TASK_EXACT in gym.registry, TASK_SAMPLE_EQ in gym.registry],
        "official": {
            "action_space": official.action_space,
            "actor_observation_space": _actor_dim(official),
            "actor_observation_order": official.obs_order + ["prev_actions"],
            "vla_chunk_size": official.vla_chunk_size,
            "residual_action_limit": official.residual_action_limit,
            "force_reference_training_range_n": official.task.contact_penalty_threshold_range,
            "force_reference_evaluation_n": 10.0,
            "reward": {
                "alpha": official.task.contact_penalty_alpha,
                "lambda_f": official.task.force_shaping_lambda,
                "c_f": official.task.force_potential_c,
                "gamma": exact_agent["gamma"],
            },
            "success_lateral_tolerance_m": official.task.success_lateral_tolerance_m,
            "success_depth_tolerance_m": official.task.success_depth_tolerance_m,
        },
        "legacy": {
            "action_space": legacy.action_space,
            "actor_observation_space": _actor_dim(legacy),
            "demo_action_space": demo.action_space,
            "demo_save_enabled": demo.is_demo_save,
        },
        "exact_batch": {
            "num_envs": exact_agent["num_actors"],
            "horizon": exact_agent["horizon_length"],
            "frames": exact_agent["num_actors"] * exact_agent["horizon_length"],
            "minibatches": exact_agent["num_actors"]
            * exact_agent["horizon_length"]
            // exact_agent["minibatch_size"],
            "minibatch_updates": exact_agent["num_actors"]
            * exact_agent["horizon_length"]
            // exact_agent["minibatch_size"]
            * exact_agent["mini_epochs"],
        },
        "sample_equivalent_batch": {
            "num_envs": sample_agent["num_actors"],
            "horizon": sample_agent["horizon_length"],
            "frames": sample_agent["num_actors"] * sample_agent["horizon_length"],
            "minibatches": sample_agent["num_actors"]
            * sample_agent["horizon_length"]
            // sample_agent["minibatch_size"],
            "minibatch_updates": sample_agent["num_actors"]
            * sample_agent["horizon_length"]
            // sample_agent["minibatch_size"]
            * sample_agent["mini_epochs"],
        },
    }

    assert all(audit["tasks_registered"])
    assert audit["official"]["action_space"] == 6
    assert audit["official"]["actor_observation_space"] == 22
    assert "force_threshold" not in official.obs_order
    assert "force_threshold" in official.state_order
    assert audit["legacy"]["action_space"] == 7
    assert audit["legacy"]["demo_action_space"] == 7
    assert audit["legacy"]["demo_save_enabled"] is True
    assert audit["exact_batch"]["frames"] == 8192
    assert audit["sample_equivalent_batch"]["frames"] == 8192
    assert audit["exact_batch"]["minibatches"] == 16
    assert audit["sample_equivalent_batch"]["minibatch_updates"] == 64
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
