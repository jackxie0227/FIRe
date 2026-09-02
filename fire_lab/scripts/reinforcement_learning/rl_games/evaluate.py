# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RL-Games checkpoint for a fixed number of episodes."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate an RL-Games checkpoint with finite episodes.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Registered Isaac Lab task name (required).")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint (required).")
parser.add_argument("--seed", type=int, default=1000, help="Environment evaluation seed.")
parser.add_argument("--episodes", type=int, default=100, help="Number of completed episodes to collect.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Global safety limit in environment steps. Defaults to enough steps for the requested episodes.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Metrics JSON path. Defaults below experiments/evaluations/forge/peg_insert.",
)
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample policy actions instead of using deterministic actions.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import fire_lab.tasks  # noqa: F401


FIRE_LAB_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = FIRE_LAB_ROOT / "experiments" / "evaluations" / "forge" / "peg_insert"


def _summary(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    if args_cli.task is None:
        raise ValueError("--task is required.")
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required.")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be greater than zero.")
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be greater than zero.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    seed = random.randint(0, 10000) if args_cli.seed == -1 else args_cli.seed
    agent_cfg["params"]["seed"] = seed
    env_cfg.seed = seed

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    task_env = raw_env.unwrapped

    required_fields = ("ep_succeeded", "ep_success_times", "force_sensor_smooth")
    missing_fields = [name for name in required_fields if not hasattr(task_env, name)]
    if missing_fields:
        raw_env.close()
        raise RuntimeError(f"Task does not expose required evaluation fields: {missing_fields}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = RlGamesVecEnvWrapper(
        raw_env, rl_device, clip_obs, clip_actions, obs_groups, concatenate_obs_groups
    )

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["num_actors"] = task_env.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    episode_rewards = torch.zeros(task_env.num_envs, device=task_env.device)
    force_sums = torch.zeros(task_env.num_envs, device=task_env.device)
    force_peaks = torch.zeros(task_env.num_envs, device=task_env.device)
    episode_steps = torch.zeros(task_env.num_envs, dtype=torch.long, device=task_env.device)

    rewards_out: list[float] = []
    success_steps_out: list[float] = []
    mean_forces_out: list[float] = []
    peak_forces_out: list[float] = []
    successes_out: list[int] = []
    total_steps = 0
    default_max_steps = int(np.ceil(args_cli.episodes / task_env.num_envs) + 1) * task_env.max_episode_length
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else default_max_steps
    if max_steps <= 0:
        env.close()
        raise ValueError("--max_steps must be greater than zero.")

    while len(rewards_out) < args_cli.episodes and total_steps < max_steps and simulation_app.is_running():
        with torch.inference_mode():
            obs_tensor = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_tensor, is_deterministic=not args_cli.stochastic)
            obs, rewards, dones, _ = env.step(actions)

        reward_tensor = torch.as_tensor(rewards, device=task_env.device).reshape(-1)
        done_tensor = torch.as_tensor(dones, device=task_env.device, dtype=torch.bool).reshape(-1)
        force_norm = torch.linalg.vector_norm(task_env.force_sensor_smooth[:, :3], dim=-1)
        episode_rewards += reward_tensor
        force_sums += force_norm
        force_peaks = torch.maximum(force_peaks, force_norm)
        episode_steps += 1
        total_steps += 1

        done_ids = done_tensor.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            for env_id in done_ids.tolist():
                if len(rewards_out) >= args_cli.episodes:
                    break
                succeeded = int(task_env.ep_succeeded[env_id].item() > 0)
                successes_out.append(succeeded)
                rewards_out.append(float(episode_rewards[env_id].item()))
                mean_forces_out.append(float((force_sums[env_id] / episode_steps[env_id]).item()))
                peak_forces_out.append(float(force_peaks[env_id].item()))
                if succeeded:
                    success_steps_out.append(float(task_env.ep_success_times[env_id].item()))
            episode_rewards[done_ids] = 0.0
            force_sums[done_ids] = 0.0
            force_peaks[done_ids] = 0.0
            episode_steps[done_ids] = 0
            if agent.is_rnn and agent.states is not None:
                with torch.inference_mode():
                    for state in agent.states:
                        state[:, done_tensor, :] = 0.0

    completed = len(rewards_out)
    success_count = sum(successes_out)
    reward_mean, reward_std = _summary(rewards_out)
    force_mean, _ = _summary(mean_forces_out)
    success_step_mean, _ = _summary(success_steps_out)
    metrics = {
        "task": args_cli.task,
        "checkpoint": str(Path(checkpoint).resolve()),
        "seed": seed,
        "deterministic": not args_cli.stochastic,
        "requested_episodes": args_cli.episodes,
        "completed_episodes": completed,
        "success_count": success_count,
        "success_rate": success_count / completed if completed else 0.0,
        "mean_episode_reward": reward_mean,
        "std_episode_reward": reward_std,
        "mean_success_steps": success_step_mean,
        "mean_contact_force": force_mean,
        "peak_contact_force": max(peak_forces_out) if peak_forces_out else None,
        "timeout_count": completed - success_count,
        "global_step_limit_reached": total_steps >= max_steps and completed < args_cli.episodes,
        "environment_steps": total_steps,
    }

    output = args_cli.output
    if output is None:
        checkpoint_stem = Path(checkpoint).stem
        output = DEFAULT_OUTPUT_DIR / f"{checkpoint_stem}_seed{seed}.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=False)
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"[INFO] Metrics saved to: {output}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
