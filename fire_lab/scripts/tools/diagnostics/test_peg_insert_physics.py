"""Test whether a perfectly aligned Forge peg can be inserted without a learned policy."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="FireLab-BaseLine-Forge-PegInsert-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--descent_steps", type=int, default=100)
parser.add_argument("--hold_steps", type=int, default=40)
parser.add_argument("--target_depth", type=float, default=0.002, help="Desired peg-base depth below the hole base [m].")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import fire_lab.tasks  # noqa: F401
from fire_lab.tasks.direct.base_line.factory import factory_utils
from fire_lab.tasks.direct.base_line.forge import forge_utils


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, _agent_cfg: dict):
    if args_cli.num_envs <= 0 or args_cli.descent_steps <= 0 or args_cli.hold_steps < 0:
        raise ValueError("Environment and step counts must be positive.")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.fixed_asset_init_orn_range_deg = 0.0
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.obs_rand.fixed_asset_pos = [0.0, 0.0, 0.0]
    env_cfg.obs_rand.fingertip_pos = 0.0
    env_cfg.obs_rand.fingertip_rot_deg = 0.0
    env_cfg.obs_rand.ft_force = 0.0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = raw_env.unwrapped
    raw_env.reset()

    held_base_pos, _ = factory_utils.get_held_base_pose(
        env.held_pos, env.held_quat, env.cfg_task.name, env.cfg_task.fixed_asset_cfg, env.num_envs, env.device
    )
    target_base_pos, _ = factory_utils.get_target_held_base_pose(
        env.fixed_pos, env.fixed_quat, env.cfg_task.name, env.cfg_task.fixed_asset_cfg, env.num_envs, env.device
    )
    fingertip_to_peg = env.fingertip_midpoint_pos - held_base_pos
    initial_gap = held_base_pos[:, 2] - target_base_pos[:, 2]

    pos_bounds = torch.tensor(env.cfg.ctrl.pos_action_bounds, device=env.device)
    actions = torch.zeros((env.num_envs, 7), device=env.device)
    actions[:, 6] = -1.0
    force_peak = torch.zeros(env.num_envs, device=env.device)
    min_xy_error = torch.full((env.num_envs,), float("inf"), device=env.device)
    min_z_gap = torch.full((env.num_envs,), float("inf"), device=env.device)
    ever_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    total_steps = args_cli.descent_steps + args_cli.hold_steps
    for step in range(total_steps):
        fraction = min((step + 1) / args_cli.descent_steps, 1.0)
        desired_gap = initial_gap * (1.0 - fraction) - args_cli.target_depth * fraction
        desired_fingertip = target_base_pos + fingertip_to_peg
        desired_fingertip[:, 2] += desired_gap
        actions[:, :3] = (desired_fingertip - env.fixed_pos_obs_frame) / pos_bounds

        _, _, current_yaw = forge_utils.euler_xyz_from_quat_wxyz(env.fingertip_midpoint_quat)
        actions[:, 5] = ((-current_yaw + 0.25 * torch.pi) / (0.75 * torch.pi)).clamp(-1.0, 1.0)
        raw_env.step(actions)

        held_base_pos, _ = factory_utils.get_held_base_pose(
            env.held_pos, env.held_quat, env.cfg_task.name, env.cfg_task.fixed_asset_cfg, env.num_envs, env.device
        )
        target_base_pos, _ = factory_utils.get_target_held_base_pose(
            env.fixed_pos, env.fixed_quat, env.cfg_task.name, env.cfg_task.fixed_asset_cfg, env.num_envs, env.device
        )
        xy_error = torch.linalg.vector_norm(held_base_pos[:, :2] - target_base_pos[:, :2], dim=-1)
        z_gap = held_base_pos[:, 2] - target_base_pos[:, 2]
        successes = env._get_curr_successes(env.cfg_task.success_threshold, check_rot=False)
        force = torch.linalg.vector_norm(env.force_sensor_smooth[:, :3], dim=-1)
        min_xy_error = torch.minimum(min_xy_error, xy_error)
        min_z_gap = torch.minimum(min_z_gap, z_gap)
        force_peak = torch.maximum(force_peak, force)
        ever_success |= successes

    final_xy_error = torch.linalg.vector_norm(held_base_pos[:, :2] - target_base_pos[:, :2], dim=-1)
    final_z_gap = held_base_pos[:, 2] - target_base_pos[:, 2]
    metrics = {
        "task": args_cli.task,
        "num_envs": env.num_envs,
        "seed": args_cli.seed,
        "fingerpad_length_m": float(env.cfg_task.robot_cfg.franka_fingerpad_length),
        "target_depth_m": args_cli.target_depth,
        "initial_gap_mean_m": float(initial_gap.mean().item()),
        "final_xy_error_mean_m": float(final_xy_error.mean().item()),
        "final_z_gap_mean_m": float(final_z_gap.mean().item()),
        "min_xy_error_m": float(min_xy_error.min().item()),
        "min_z_gap_m": float(min_z_gap.min().item()),
        "peak_contact_force_n": float(force_peak.max().item()),
        "success_count": int(ever_success.sum().item()),
        "physical_insertion_possible": bool(ever_success.any().item()),
    }
    output = args_cli.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"[INFO] Metrics saved to: {output}")
    raw_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
