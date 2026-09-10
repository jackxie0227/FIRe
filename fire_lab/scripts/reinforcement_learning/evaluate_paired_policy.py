#!/usr/bin/env python3
"""Evaluate PPO-only or GR00T-only on reproducible single-environment conditions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", choices=("ppo", "gr00t"), required=True)
parser.add_argument("--task", required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", help="Required for --policy=ppo.")
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--condition-seed-start", type=int, default=100000)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--resume", action="store_true", help="Resume completed trials from --output.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import fire_lab.tasks  # noqa: F401


INITIAL_STATE_FIELDS = (
    "fixed_pos",
    "fixed_quat",
    "held_pos",
    "held_quat",
    "fingertip_midpoint_pos",
    "fingertip_midpoint_quat",
    "joint_pos",
    "init_fixed_pos_obs_noise",
    "task_prop_gains",
    "ema_factor",
    "pos_threshold",
    "rot_threshold",
    "contact_penalty_thresholds",
    "dead_zone_thresholds",
)


def _snapshot_initial_state(task_env) -> dict[str, list[float] | float]:
    snapshot = {}
    for field in INITIAL_STATE_FIELDS:
        if not hasattr(task_env, field):
            continue
        value = getattr(task_env, field)
        if isinstance(value, torch.Tensor):
            value = value[0].detach().cpu()
            snapshot[field] = value.item() if value.ndim == 0 else value.tolist()
    return snapshot


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return center - margin, center + margin


def _build_result(trials: list[dict], complete: bool) -> dict:
    successes = sum(trial["success"] for trial in trials)
    interval = _wilson_interval(successes, len(trials))
    return {
        "schema_version": 1,
        "policy": args_cli.policy,
        "task": args_cli.task,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()) if args_cli.checkpoint else None,
        "episodes": len(trials),
        "requested_episodes": args_cli.episodes,
        "condition_seed_start": args_cli.condition_seed_start,
        "complete": complete,
        "success_count": successes,
        "success_rate": successes / len(trials) if trials else 0.0,
        "success_rate_wilson_95": list(interval),
        "trials": trials,
    }


def _save_result(output: Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    temporary.replace(output)


def _load_ppo(raw_env, task_env, agent_cfg: dict) -> tuple[RlGamesVecEnvWrapper, BasePlayer]:
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for --policy=ppo")

    checkpoint = retrieve_file_path(args_cli.checkpoint)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concatenate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = RlGamesVecEnvWrapper(
        raw_env, rl_device, clip_obs, clip_actions, obs_groups, concatenate_obs_groups
    )

    vecenv.register(
        "PairedIsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "paired_rlgpu", {"vecenv_type": "PairedIsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = checkpoint
    agent_cfg["params"]["config"]["env_name"] = "paired_rlgpu"
    agent_cfg["params"]["config"]["num_actors"] = 1
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()
    return env, agent


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict) -> None:
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args_cli.policy == "gr00t" and args_cli.checkpoint:
        raise ValueError("--checkpoint is only valid for --policy=ppo")

    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    output = args_cli.output.expanduser().resolve()
    trials = []
    if args_cli.resume and output.exists():
        with output.open(encoding="utf-8") as file:
            previous = json.load(file)
        expected = {
            "policy": args_cli.policy,
            "task": args_cli.task,
            "condition_seed_start": args_cli.condition_seed_start,
        }
        incompatible = {
            key: (previous.get(key), value)
            for key, value in expected.items()
            if previous.get(key) != value
        }
        if incompatible:
            raise ValueError(f"Cannot resume incompatible output: {incompatible}")
        trials = previous.get("trials", [])
        if len(trials) > args_cli.episodes:
            raise ValueError(
                f"Output already has {len(trials)} trials, more than requested {args_cli.episodes}"
            )
        for index, trial in enumerate(trials):
            expected_seed = args_cli.condition_seed_start + index
            if trial.get("condition_index") != index or trial.get("condition_seed") != expected_seed:
                raise ValueError(f"Cannot resume non-contiguous trial at index {index}")
        print(f"[INFO] Resuming {args_cli.policy} from {len(trials)}/{args_cli.episodes} trials")

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    task_env = raw_env.unwrapped
    required = ("ep_succeeded", "ep_success_times")
    missing = [field for field in required if not hasattr(task_env, field)]
    if missing:
        raw_env.close()
        raise RuntimeError(f"Environment is missing paired-evaluation fields: {missing}")

    wrapped_env = None
    agent = None
    if args_cli.policy == "ppo":
        wrapped_env, agent = _load_ppo(raw_env, task_env, agent_cfg)

    success_count = sum(trial["success"] for trial in trials)
    try:
        for index in range(len(trials), args_cli.episodes):
            condition_seed = args_cli.condition_seed_start + index
            if args_cli.policy == "ppo":
                task_env.seed(condition_seed)
                obs = wrapped_env.reset()
                if isinstance(obs, dict):
                    obs = obs["obs"]
                agent.reset()
                _ = agent.get_batch_size(obs, 1)
                if agent.is_rnn:
                    agent.init_rnn()
            else:
                obs, _ = raw_env.reset(seed=condition_seed)

            initial_state = _snapshot_initial_state(task_env)
            missing_state = [field for field in INITIAL_STATE_FIELDS[:7] if field not in initial_state]
            if missing_state:
                raise RuntimeError(
                    f"Environment is missing initial-state fields after reset: {missing_state}"
                )
            episode_reward = 0.0
            step_count = 0
            done = False
            max_steps = task_env.max_episode_length + 1
            while not done and step_count < max_steps and simulation_app.is_running():
                # Isaac Lab keeps some tensors created during env.step() and mutates them
                # on the next reset, which is incompatible with torch.inference_mode().
                with torch.no_grad():
                    if args_cli.policy == "ppo":
                        obs_tensor = agent.obs_to_torch(obs)
                        action = agent.get_action(obs_tensor, is_deterministic=True)
                        obs, rewards, dones, _ = wrapped_env.step(action)
                        done = bool(torch.as_tensor(dones).reshape(-1)[0].item())
                    else:
                        action = torch.zeros(raw_env.action_space.shape, device=task_env.device)
                        obs, rewards, terminated, truncated, _ = raw_env.step(action)
                        done = bool(
                            (torch.as_tensor(terminated) | torch.as_tensor(truncated)).reshape(-1)[0].item()
                        )
                episode_reward += float(torch.as_tensor(rewards).reshape(-1)[0].item())
                step_count += 1

            if not done:
                raise RuntimeError(f"Condition {condition_seed} did not terminate within {max_steps} steps")

            succeeded = int(task_env.ep_succeeded[0].item() > 0)
            success_count += succeeded
            trial = {
                "condition_index": index,
                "condition_seed": condition_seed,
                "success": succeeded,
                "steps": step_count,
                "first_success_step": int(task_env.ep_success_times[0].item()) if succeeded else None,
                "episode_reward": episode_reward,
                "initial_state": initial_state,
            }
            trials.append(trial)
            _save_result(output, _build_result(trials, complete=len(trials) == args_cli.episodes))
            print(
                f"[{args_cli.policy}] {index + 1}/{args_cli.episodes} "
                f"seed={condition_seed} success={succeeded} running={success_count / (index + 1):.3f}",
                flush=True,
            )
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        else:
            raw_env.close()

    result = _build_result(trials, complete=len(trials) == args_cli.episodes)
    _save_result(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}, indent=2))
    print(f"[INFO] Paired evaluation saved to: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
