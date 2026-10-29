"""Bounded GearMesh FIRe6D environment and GR00T-only smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="FireLab-VLA-Gr00t-Forge-GearMesh-FIRe6D-Direct-v1")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import fire_lab.tasks  # noqa: F401


def finite(name, value):
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=cfg)
    task = env.unwrapped
    try:
        obs, _ = env.reset()
        if env.action_space.shape != (args.num_envs, 6):
            raise RuntimeError(f"wrong action space: {env.action_space}")
        if obs["policy"].shape != (args.num_envs, 23) or obs["critic"].shape != (args.num_envs, 60):
            raise RuntimeError(f"wrong observation shapes: {[(k, tuple(v.shape)) for k, v in obs.items()]}")
        images = {}
        for name in ("left", "right", "wrist"):
            rgb = getattr(task, f"_{name}_camera").data.output["rgb"]
            images[name] = list(rgb.shape)
            finite(f"{name} camera", rgb.float())
        if task._robot.body_names[task.force_sensor_body_idx] != "force_sensor":
            raise RuntimeError("force sensor body does not resolve to force_sensor")
        result = {
            "task": args.task,
            "num_envs": args.num_envs,
            "steps_requested": args.steps,
            "action_shape": list(env.action_space.shape),
            "policy_shape": list(obs["policy"].shape),
            "critic_shape": list(obs["critic"].shape),
            "camera_shapes": images,
            "force_sensor_body": task._robot.body_names[task.force_sensor_body_idx],
            "vla_abs_max": 0.0,
            "combined_out_of_range_max": 0.0,
            "force_abs_max": 0.0,
            "episodes_completed": 0,
            "reset_refresh_count": 0,
            "post_reset_first_action_verified": 0,
            "camera_changed_fraction_after_reset": [],
        }
        finite("policy reset", obs["policy"])
        finite("critic reset", obs["critic"])
        expected_first_chunk_action = None
        for _ in range(args.steps):
            left_before = task._left_camera.data.output["rgb"].clone()
            with torch.inference_mode():
                action = torch.zeros(env.action_space.shape, device=task.device)
                obs, reward, terminated, truncated, _ = env.step(action)
            if expected_first_chunk_action is not None:
                if not torch.allclose(task._last_vla_action, expected_first_chunk_action):
                    raise RuntimeError("first action after reset does not match refreshed VLA chunk")
                result["post_reset_first_action_verified"] += 1
                expected_first_chunk_action = None
            for key, tensor in obs.items():
                finite(key, tensor)
            finite("reward", reward)
            finite("VLA action", task._last_vla_action)
            finite("combined action", task._last_combined_action)
            finite("force", task.force_sensor_smooth)
            result["vla_abs_max"] = max(result["vla_abs_max"], task._last_vla_action.abs().max().item())
            result["combined_out_of_range_max"] = max(
                result["combined_out_of_range_max"],
                task._last_combined_out_of_range_fraction.max().item(),
            )
            result["force_abs_max"] = max(result["force_abs_max"], task.force_sensor_smooth.abs().max().item())
            completed = int((terminated | truncated).sum().item())
            result["episodes_completed"] += completed
            if completed:
                left_after = task._left_camera.data.output["rgb"]
                changed = (left_before != left_after).float().mean().item()
                result["camera_changed_fraction_after_reset"].append(changed)
                if changed == 0.0:
                    raise RuntimeError("left RGB camera did not change after episode reset")
                expected_first_chunk_action = task.processed_vla_actions[:, 0, :].clone()
        result["inference_error_count"] = task._vla_request_error_count
        result["reset_refresh_count"] = task._vla_reset_refresh_count
        if args.steps >= 301 and result["post_reset_first_action_verified"] == 0:
            raise RuntimeError("did not verify a post-reset VLA chunk")
        if result["vla_abs_max"] == 0.0:
            raise RuntimeError("GR00T pose actions remained zero in smoke")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
