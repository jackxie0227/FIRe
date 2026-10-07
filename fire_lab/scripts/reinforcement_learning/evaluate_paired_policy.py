#!/usr/bin/env python3
"""Evaluate teacher PPO, GR00T-only, or GR00T+residual on recorded conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
OFFICIAL_SUCCESS_CRITERION = "fire_official_peg_insert_xy_le_0.005_z_le_0.025_ever"

parser.add_argument("--policy", choices=("ppo", "gr00t", "residual"), required=True)
parser.add_argument("--task", required=True, help="Registered Isaac Lab task name.")
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", help="Required for --policy=ppo or --policy=residual.")
parser.add_argument(
    "--success-criterion",
    choices=("native", OFFICIAL_SUCCESS_CRITERION),
    default="native",
    help="Use the same FIRe-official ever-success geometry for all three policies.",
)
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--condition-seed-start", type=int, default=100000)
parser.add_argument("--base-seeds", nargs="+", type=int)
parser.add_argument("--episodes-per-seed", type=int)
parser.add_argument("--eval-num-envs", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--resume", action="store_true", help="Resume completed trials from --output.")
parser.add_argument(
    "--record-multicam-dir",
    type=Path,
    help=(
        "Root directory for synchronized left, right, wrist, and three-view "
        "composite MP4s. Recordings are separated automatically by policy and "
        "condition seed. Requires a camera-enabled environment, "
        "--eval-num-envs=1, and --enable_cameras."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from torchvision.io import write_video

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import fire_lab.tasks  # noqa: F401
from fire_lab.tasks.direct.base_line.factory import factory_utils
from fire_lab.tasks.direct.vla.gr00t.forge.fire_official import peg_insert_success_mask


INITIAL_STATE_FIELDS = (
    "fixed_pos",
    "fixed_quat",
    "held_pos",
    "held_quat",
    "fingertip_midpoint_pos",
    "fingertip_midpoint_quat",
    "fingertip_midpoint_linvel",
    "fingertip_midpoint_angvel",
    "joint_pos",
    "joint_vel",
    "init_fixed_pos_obs_noise",
    "task_prop_gains",
    "ema_factor",
    "pos_threshold",
    "rot_threshold",
    "contact_penalty_thresholds",
    "dead_zone_thresholds",
)

MULTICAM_SENSORS = {
    "left": "_left_camera",
    "right": "_right_camera",
    "wrist": "_wrist_camera",
}

MULTICAM_POLICY_DIRS = {
    "ppo": "ppo_teacher",
    "gr00t": "vla_only",
    "residual": "vla_residual_ppo",
}


def _capture_multicam_frame(task_env) -> dict[str, torch.Tensor]:
    frames = {}
    for name, attribute in MULTICAM_SENSORS.items():
        if not hasattr(task_env, attribute):
            raise RuntimeError(f"Environment is missing the {name} camera ({attribute})")
        camera = getattr(task_env, attribute)
        rgb = camera.data.output.get("rgb")
        if rgb is None or rgb.shape[0] != 1:
            raise RuntimeError(f"Invalid {name} camera RGB output: {getattr(rgb, 'shape', None)}")
        frames[name] = rgb[0, ..., :3].detach().cpu().to(torch.uint8).clone()
    return frames


def _write_multicam_videos(
    buffers: dict[str, list[torch.Tensor]],
    *,
    output_dir: Path,
    condition_seed: int,
    fps: int,
) -> None:
    run_dir = output_dir / MULTICAM_POLICY_DIRS[args_cli.policy] / f"seed_{condition_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    videos = {name: torch.stack(frames) for name, frames in buffers.items()}
    videos["three_view"] = torch.cat(
        (videos["left"], videos["right"], videos["wrist"]), dim=2
    )
    for name, video in videos.items():
        output = run_dir / f"{name}.mp4"
        write_video(str(output), video, fps=fps, video_codec="h264")
        print(f"[INFO] Recorded {name} video: {output}", flush=True)


def _snapshot_initial_state(task_env, env_index: int = 0) -> dict[str, list[float] | float]:
    snapshot = {}
    for field in INITIAL_STATE_FIELDS:
        if not hasattr(task_env, field):
            continue
        value = getattr(task_env, field)
        if isinstance(value, torch.Tensor):
            value = value[env_index].detach().cpu()
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


@lru_cache(maxsize=4)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_success_mask(task_env) -> torch.Tensor:
    """Score all policies with the FIRe project-page PegInsert geometry."""

    held_base_pos, _ = factory_utils.get_held_base_pose(
        task_env.held_pos,
        task_env.held_quat,
        task_env.cfg_task.name,
        task_env.cfg_task.fixed_asset_cfg,
        task_env.num_envs,
        task_env.device,
    )
    target_base_pos, _ = factory_utils.get_target_held_base_pose(
        task_env.fixed_pos,
        task_env.fixed_quat,
        task_env.cfg_task.name,
        task_env.cfg_task.fixed_asset_cfg,
        task_env.num_envs,
        task_env.device,
    )
    xy_error = torch.linalg.vector_norm(held_base_pos[:, :2] - target_base_pos[:, :2], dim=-1)
    z_displacement = held_base_pos[:, 2] - target_base_pos[:, 2]
    return peg_insert_success_mask(xy_error, z_displacement)


class _OfficialSuccessTracker:
    """Observe the geometry during reward calculation, before terminal auto-reset."""

    def __init__(self, task_env):
        self.task_env = task_env
        self.active = False
        self.step = 0
        self.seen = torch.zeros(task_env.num_envs, dtype=torch.bool)
        self.first_step = torch.zeros(task_env.num_envs, dtype=torch.long)
        self.finite = torch.ones(task_env.num_envs, dtype=torch.bool)
        self.force_sum = torch.zeros(task_env.num_envs, dtype=torch.float64)
        self.force_max = torch.zeros(task_env.num_envs, dtype=torch.float64)
        self.over_10n_steps = torch.zeros(task_env.num_envs, dtype=torch.long)
        original_get_rewards = task_env._get_rewards

        def observed_get_rewards():
            rewards = original_get_rewards()
            if self.active:
                self.finite &= (
                    torch.isfinite(task_env.held_pos).all(dim=-1)
                    & torch.isfinite(task_env.fixed_pos).all(dim=-1)
                    & torch.isfinite(task_env.held_quat).all(dim=-1)
                    & torch.isfinite(task_env.fixed_quat).all(dim=-1)
                ).detach().cpu()
                current = _official_success_mask(task_env).detach().cpu()
                first = current & ~self.seen
                self.first_step[first] = self.step
                self.seen |= current
                force = torch.linalg.vector_norm(
                    task_env.force_sensor_smooth[:, :3], dim=-1
                ).detach().cpu().double()
                self.finite &= torch.isfinite(force)
                self.force_sum += force
                self.force_max = torch.maximum(self.force_max, force)
                self.over_10n_steps += force.gt(10.0).long()
            return rewards

        task_env._get_rewards = observed_get_rewards

    def begin(self):
        self.seen.zero_()
        self.first_step.zero_()
        self.finite.fill_(True)
        self.force_sum.zero_()
        self.force_max.zero_()
        self.over_10n_steps.zero_()
        self.step = 0
        self.active = True

    def before_step(self):
        self.step += 1

    def end(self):
        self.active = False


def _build_result(trials: list[dict], complete: bool) -> dict:
    successes = sum(trial["success"] for trial in trials)
    interval = _wilson_interval(successes, len(trials))
    per_seed = {}
    for trial in trials:
        if trial.get("base_seed") is None:
            continue
        key = str(trial["base_seed"])
        summary = per_seed.setdefault(key, {"episodes": 0, "success_count": 0})
        summary["episodes"] += 1
        summary["success_count"] += trial["success"]
    for summary in per_seed.values():
        summary["success_rate"] = summary["success_count"] / summary["episodes"]
    return {
        "schema_version": 1,
        "policy": args_cli.policy,
        "task": args_cli.task,
        "success_criterion": args_cli.success_criterion,
        "checkpoint": str(Path(args_cli.checkpoint).resolve()) if args_cli.checkpoint else None,
        "checkpoint_sha256": (
            _sha256(Path(args_cli.checkpoint).resolve()) if args_cli.checkpoint else None
        ),
        "episodes": len(trials),
        "requested_episodes": args_cli.episodes,
        "condition_seed_start": args_cli.condition_seed_start,
        "base_seeds": args_cli.base_seeds,
        "episodes_per_seed": args_cli.episodes_per_seed,
        "per_seed": per_seed,
        "complete": complete,
        "success_count": successes,
        "success_rate": successes / len(trials) if trials else 0.0,
        "success_rate_wilson_95": list(interval),
        "all_outputs_finite": all(trial.get("all_outputs_finite", False) for trial in trials),
        "all_zero_residual_compositions_exact": (
            all(trial.get("zero_residual_composition_exact", False) for trial in trials)
            if args_cli.policy == "gr00t" else None
        ),
        "vla_request_error_count": sum(trial.get("vla_request_error_count", 0) for trial in trials),
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
        raise ValueError("--checkpoint is required for PPO or residual policy")

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
    agent_cfg["params"]["config"]["num_actors"] = task_env.num_envs
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
    if args_cli.eval_num_envs <= 0 or args_cli.episodes % args_cli.eval_num_envs != 0:
        raise ValueError("--eval-num-envs must be positive and divide --episodes")
    if args_cli.record_multicam_dir is not None and args_cli.eval_num_envs != 1:
        raise ValueError("--record-multicam-dir requires --eval-num-envs=1")
    if (args_cli.base_seeds is None) != (args_cli.episodes_per_seed is None):
        raise ValueError("--base-seeds and --episodes-per-seed must be provided together")
    if args_cli.base_seeds is not None:
        if args_cli.episodes_per_seed <= 0:
            raise ValueError("--episodes-per-seed must be positive")
        if len(args_cli.base_seeds) * args_cli.episodes_per_seed != args_cli.episodes:
            raise ValueError("--episodes must equal len(base_seeds) * episodes_per_seed")
        if args_cli.episodes_per_seed % args_cli.eval_num_envs != 0:
            raise ValueError("--eval-num-envs must divide --episodes-per-seed")
        if args_cli.resume:
            raise ValueError("Grouped base-seed evaluation does not support --resume")
    if args_cli.policy == "gr00t" and args_cli.checkpoint:
        raise ValueError("--checkpoint is only valid for PPO or residual policy")
    if args_cli.policy in ("ppo", "residual") and not args_cli.checkpoint:
        raise ValueError("--checkpoint is required for PPO or residual policy")
    if args_cli.policy == "residual" and "FIReOfficial" not in args_cli.task:
        raise ValueError("Residual checkpoint requires a FIReOfficial task")
    if args_cli.success_criterion == OFFICIAL_SUCCESS_CRITERION and "PegInsert" not in args_cli.task:
        raise ValueError("FIRe-official PegInsert success criterion requires a PegInsert task")

    env_cfg.scene.num_envs = args_cli.eval_num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    if args_cli.success_criterion == OFFICIAL_SUCCESS_CRITERION:
        # Isaac Lab also randomizes startup properties while gym.make constructs
        # the environment. Reset-time task_env.seed() alone cannot align those
        # properties across fresh teacher/VLA/residual simulator processes.
        construction_seed = (
            args_cli.base_seeds[0]
            if args_cli.base_seeds is not None
            else args_cli.condition_seed_start
        )
        env_cfg.seed = construction_seed
        if hasattr(env_cfg, "fixed_contact_penalty_threshold"):
            env_cfg.fixed_contact_penalty_threshold = 10.0
        elif hasattr(env_cfg, "task") and hasattr(env_cfg.task, "contact_penalty_threshold_range"):
            # The baseline teacher observes F_ref. Keep the random draw in
            # reset while fixing its value to FIRe's evaluation setting.
            env_cfg.task.contact_penalty_threshold_range = [10.0, 10.0]

    output = args_cli.output.expanduser().resolve()
    trials = []
    if args_cli.resume and output.exists():
        with output.open(encoding="utf-8") as file:
            previous = json.load(file)
        expected = {
            "policy": args_cli.policy,
            "task": args_cli.task,
            "condition_seed_start": args_cli.condition_seed_start,
            "success_criterion": args_cli.success_criterion,
            "checkpoint": str(Path(args_cli.checkpoint).resolve()) if args_cli.checkpoint else None,
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
    official_tracker = (
        _OfficialSuccessTracker(task_env)
        if args_cli.success_criterion == OFFICIAL_SUCCESS_CRITERION
        else None
    )
    required = ("ep_succeeded", "ep_success_times")
    missing = [field for field in required if not hasattr(task_env, field)]
    if missing:
        raw_env.close()
        raise RuntimeError(f"Environment is missing paired-evaluation fields: {missing}")

    wrapped_env = None
    agent = None
    if args_cli.policy in ("ppo", "residual"):
        wrapped_env, agent = _load_ppo(raw_env, task_env, agent_cfg)

    success_count = sum(trial["success"] for trial in trials)
    carried_obs = None
    try:
        for index in range(len(trials), args_cli.episodes, args_cli.eval_num_envs):
            if args_cli.base_seeds is None:
                base_seed = None
                seed_episode_index = None
                condition_seed = args_cli.condition_seed_start + index
                reset_seed = condition_seed
            else:
                base_seed_index = index // args_cli.episodes_per_seed
                seed_episode_index = index % args_cli.episodes_per_seed
                base_seed = args_cli.base_seeds[base_seed_index]
                condition_seed = base_seed
                reset_seed = base_seed if seed_episode_index == 0 else None
            if args_cli.policy in ("ppo", "residual"):
                if reset_seed is not None:
                    task_env.seed(reset_seed)
                if reset_seed is not None or carried_obs is None:
                    obs = wrapped_env.reset()
                else:
                    obs = carried_obs
                if isinstance(obs, dict):
                    obs = obs["obs"]
                agent.reset()
                _ = agent.get_batch_size(obs, 1)
                if agent.is_rnn:
                    agent.init_rnn()
            else:
                if reset_seed is not None or carried_obs is None:
                    obs, _ = raw_env.reset(seed=reset_seed)
                else:
                    obs = carried_obs

            initial_states = [
                _snapshot_initial_state(task_env, env_index)
                for env_index in range(args_cli.eval_num_envs)
            ]
            missing_state = [field for field in INITIAL_STATE_FIELDS[:7] if field not in initial_states[0]]
            if missing_state:
                raise RuntimeError(
                    f"Environment is missing initial-state fields after reset: {missing_state}"
                )
            episode_reward = torch.zeros(args_cli.eval_num_envs, dtype=torch.float64)
            step_count = 0
            output_is_finite = torch.ones(args_cli.eval_num_envs, dtype=torch.bool)
            zero_residual_composition_exact = torch.ones(args_cli.eval_num_envs, dtype=torch.bool)
            vla_action_abs_sum = torch.zeros(args_cli.eval_num_envs, dtype=torch.float64)
            vla_action_elements = 0
            force_sum = torch.zeros(args_cli.eval_num_envs, dtype=torch.float64)
            force_max = torch.zeros(args_cli.eval_num_envs, dtype=torch.float64)
            over_10n_steps = torch.zeros(args_cli.eval_num_envs, dtype=torch.long)
            request_errors_before = int(getattr(task_env, "_vla_request_error_count", 0))
            multicam_buffers = None
            if args_cli.record_multicam_dir is not None:
                # Record the synchronized pre-action images for every control
                # step.  This matches the policy's observation timeline and
                # avoids capturing the auto-reset scene after the terminal step.
                multicam_buffers = {name: [] for name in MULTICAM_SENSORS}
            if official_tracker is not None:
                official_tracker.begin()
            done = False
            max_steps = task_env.max_episode_length + 1
            while not done and step_count < max_steps and simulation_app.is_running():
                # Isaac Lab keeps some tensors created during env.step() and mutates them
                # on the next reset, which is incompatible with torch.inference_mode().
                with torch.no_grad():
                    if multicam_buffers is not None:
                        for name, frame in _capture_multicam_frame(task_env).items():
                            multicam_buffers[name].append(frame)
                    if official_tracker is not None:
                        official_tracker.before_step()
                    if args_cli.policy in ("ppo", "residual"):
                        obs_tensor = agent.obs_to_torch(obs)
                        action = agent.get_action(obs_tensor, is_deterministic=True)
                        obs, rewards, dones, _ = wrapped_env.step(action)
                        done = bool(torch.as_tensor(dones).all().item())
                        finite_step = torch.isfinite(torch.as_tensor(action)).all(dim=-1).detach().cpu()
                        finite_step &= torch.isfinite(torch.as_tensor(rewards)).detach().cpu().flatten()
                        if args_cli.policy == "residual":
                            vla_action = task_env._last_vla_action[..., :6]
                            combined_action = task_env._last_combined_action[..., :6]
                            finite_step &= torch.isfinite(vla_action).all(dim=-1).detach().cpu()
                            finite_step &= torch.isfinite(combined_action).all(dim=-1).detach().cpu()
                            vla_action_abs_sum += vla_action.abs().sum(dim=-1).detach().cpu().double()
                            vla_action_elements += vla_action.shape[-1]
                        output_is_finite &= finite_step
                    else:
                        action = torch.zeros(raw_env.action_space.shape, device=task_env.device)
                        obs, rewards, terminated, truncated, _ = raw_env.step(action)
                        done = bool((torch.as_tensor(terminated) | torch.as_tensor(truncated)).all().item())
                        vla_action = task_env._last_vla_action[..., :6]
                        combined_action = task_env._last_combined_action[..., :6]
                        finite_step = torch.logical_and(
                            torch.isfinite(vla_action).all(dim=-1),
                            torch.isfinite(combined_action).all(dim=-1),
                        ).detach().cpu()
                        finite_step &= torch.isfinite(torch.as_tensor(rewards)).detach().cpu().flatten()
                        output_is_finite &= finite_step
                        zero_step = torch.eq(vla_action, combined_action).all(dim=-1).detach().cpu()
                        zero_residual_composition_exact &= zero_step
                        vla_action_abs_sum += vla_action.abs().sum(dim=-1).detach().cpu().double()
                        vla_action_elements += vla_action.shape[-1]
                    if hasattr(task_env, "force_sensor_smooth"):
                        force = torch.linalg.vector_norm(
                            task_env.force_sensor_smooth[:, :3], dim=-1
                        ).detach().cpu().double()
                        output_is_finite &= torch.isfinite(force)
                        force_sum += force
                        force_max = torch.maximum(force_max, force)
                        over_10n_steps += force.gt(10.0).long()
                episode_reward += torch.as_tensor(rewards).detach().cpu().flatten().double()
                step_count += 1

            if official_tracker is not None:
                official_tracker.end()
                output_is_finite &= official_tracker.finite
                force_sum = official_tracker.force_sum
                force_max = official_tracker.force_max
                over_10n_steps = official_tracker.over_10n_steps

            if not done:
                raise RuntimeError(f"Condition {condition_seed} did not terminate within {max_steps} steps")

            if multicam_buffers is not None:
                _write_multicam_videos(
                    multicam_buffers,
                    output_dir=args_cli.record_multicam_dir.expanduser().resolve(),
                    condition_seed=condition_seed,
                    fps=round(1.0 / task_env.step_dt),
                )

            native_success_batch = task_env.ep_succeeded.detach().cpu().gt(0).int()
            succeeded_batch = (
                official_tracker.seen.int()
                if args_cli.success_criterion == OFFICIAL_SUCCESS_CRITERION
                else native_success_batch
            )
            success_count += int(succeeded_batch.sum().item())
            request_error_delta = int(
                getattr(task_env, "_vla_request_error_count", 0)
            ) - request_errors_before
            for env_index in range(args_cli.eval_num_envs):
                succeeded = int(succeeded_batch[env_index].item())
                trial = {
                    "condition_index": index + env_index,
                    "condition_seed": condition_seed,
                    "base_seed": base_seed,
                    "seed_episode_index": (
                        seed_episode_index + env_index if seed_episode_index is not None else None
                    ),
                    "success": succeeded,
                    "native_success": int(native_success_batch[env_index].item()),
                    "steps": step_count,
                    "first_success_step": (
                        int(official_tracker.first_step[env_index].item())
                        if succeeded and args_cli.success_criterion == OFFICIAL_SUCCESS_CRITERION
                        else int(task_env.ep_success_times[env_index].item()) if succeeded else None
                    ),
                    "contact_force_mean_n": float(force_sum[env_index].item() / step_count),
                    "contact_force_max_n": float(force_max[env_index].item()),
                    "over_10n_fraction": float(over_10n_steps[env_index].item() / step_count),
                    "episode_reward": float(episode_reward[env_index].item()),
                    "all_outputs_finite": bool(output_is_finite[env_index].item()),
                    "zero_residual_composition_exact": (
                        bool(zero_residual_composition_exact[env_index].item())
                        if args_cli.policy == "gr00t" else None
                    ),
                    "vla_action_abs_mean": float(
                        vla_action_abs_sum[env_index].item() / max(vla_action_elements, 1)
                    ),
                    "vla_request_error_count": request_error_delta if env_index == 0 else 0,
                    "initial_state": initial_states[env_index],
                }
                trials.append(trial)
            carried_obs = obs
            _save_result(output, _build_result(trials, complete=len(trials) == args_cli.episodes))
            print(
                f"[{args_cli.policy}] {len(trials)}/{args_cli.episodes} "
                f"seed={condition_seed} seed_episode={seed_episode_index} "
                f"batch_success={int(succeeded_batch.sum().item())}/{args_cli.eval_num_envs} "
                f"running={success_count / len(trials):.3f}",
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
