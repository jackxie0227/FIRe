# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

from fire_lab.tasks.direct.base_line.factory import factory_utils
from fire_lab.tasks.direct.base_line.forge import forge_utils
from fire_lab.tasks.direct.base_line.forge.forge_env import ForgeEnv
from fire_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_env_cfg import ForgeGr00tEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import TiledCamera
import time
import os
import pandas as pd
import sys
from pathlib import Path
from torchvision.io import write_video


class ForgeGr00tDemoSaveEnv(ForgeEnv):
    cfg: ForgeGr00tEnvCfg

    def __init__(self, cfg: ForgeGr00tEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)

        self.demo_save_cfg = self.cfg.demo_save_cfg

        if self.demo_save_cfg:
            self.gr00t_data_buffers = [{} for _ in range(self.num_envs)]
            self.gr00t_img_buffers = [{} for _ in range(self.num_envs)]
            self._pending_gr00t_transition = None
            self._saved_episode_count = 0

            self._initialize_parquet_buffers()
            self._initialize_image_buffers()

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self._wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self._left_camera = TiledCamera(self.cfg.left_camera)
        self._right_camera = TiledCamera(self.cfg.right_camera)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self.scene.sensors["left_camera"] = self._left_camera
        self.scene.sensors["right_camera"] = self._right_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Add success prediction rewards.
        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_buf = torch.where(true_successes, 1.0, 0.0)
        if self.cfg.is_demo_save:
            self._commit_gr00t_transition(true_successes)
        # TODO: Add force limit penalty

        return rew_buf
    
    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        if self.cfg.is_demo_save:
            self._capture_gr00t_transition(action)
        super()._pre_physics_step(action) # This changes action to smoothed action

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": self.noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ema_factor": self.ema_factor,
            "ft_force": self.force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        return {"policy": obs_tensors, "critic": state_tensors}


    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)

        if self.cfg.is_demo_save:
            self._save_gr00t_observations(
                data_dir=self.demo_save_cfg.data_dir,
                video_dir=self.demo_save_cfg.video_dir,
            )
            self._initialize_parquet_buffers()
            self._initialize_image_buffers()

    def _get_gr00t_observations(self):
        # This is for gr00t observations

        observations = {
            "video.left_view": np.expand_dims(self._left_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "video.right_view": np.expand_dims(self._right_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "video.wrist_view": np.expand_dims(self._wrist_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "state.eef_position": np.expand_dims(self.fingertip_midpoint_pos.cpu().numpy().astype(np.float64), axis=1),
            "state.eef_quaternion": np.expand_dims(self.fingertip_midpoint_quat.cpu().numpy().astype(np.float64), axis=1),
            "state.gripper_qpos": np.expand_dims(self.joint_pos[:, 7:9].cpu().numpy().astype(np.float64), axis=1),
        }
        return observations

    def _initialize_parquet_buffers(self):
        """Clears and initializes the data buffer for a specific environment."""
        for env_id in range(self.num_envs):
            self.gr00t_data_buffers[env_id] = {
                "observation.state": [],
                "observation.force": [],
                "action": [],
                "timestamp": [],
                "annotation.human.action.task_description": [],
                "task_index": [],
                "annotation.human.validity": [],
                "episode_index": [],
                "index": [],
                "next.reward": [],
                "next.done": [],
                "episode.success": [],
            }

    def _initialize_image_buffers(self):
        """Clears and initializes the image buffer for a specific environment."""
        for env_id in range(self.num_envs):
            self.gr00t_img_buffers[env_id] = {
                "left_camera": [],
                "right_camera": [],
                "wrist_camera": [],
            }

    def _save_gr00t_observations(self, data_dir: str, video_dir: str):
        """Saves the collected gr00t data and videos to the specified directories."""
        expected_frames = self.max_episode_length - 1
        frame_counts = [len(buffer["observation.state"]) for buffer in self.gr00t_data_buffers]
        if not any(frame_counts):
            print("No data collected for this episode. Skipping save.")
            return

        incomplete = [env_id for env_id, count in enumerate(frame_counts) if count != expected_frames]
        if incomplete:
            raise RuntimeError(
                f"Incomplete demo buffers for environments {incomplete}: "
                f"frame counts={frame_counts}, expected={expected_frames}."
            )

        successful_env_ids = [
            env_id
            for env_id, buffer in enumerate(self.gr00t_data_buffers)
            if any(buffer["next.reward"])
        ]
        if not successful_env_ids:
            print(f"No successful episodes in this batch ({self.num_envs} attempted); nothing saved.")
            if self.demo_save_cfg.target_successful_episodes <= 0:
                sys.exit(0)
            return

        data_path = Path(data_dir)
        existing_ids = []
        for path in data_path.glob("episode_*.parquet"):
            try:
                existing_ids.append(int(path.stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        next_episode_index = max(existing_ids, default=-1) + 1

        for output_offset, env_id in enumerate(successful_env_ids):
            episode_index = next_episode_index + output_offset
            # Save Datas
            df = pd.DataFrame(self.gr00t_data_buffers[env_id])
            df["annotation.human.validity"] = True
            df["episode.success"] = True
            df["episode_index"] = episode_index
            df["index"] = episode_index * expected_frames + np.arange(len(df), dtype=np.int64)

            gr00t_data_path = os.path.join(
                data_dir, f"episode_{episode_index:06d}.parquet"
            )
            os.makedirs(os.path.dirname(gr00t_data_path), exist_ok=True)

            df.to_parquet(gr00t_data_path, index=False)

            # Save Videos
            for camera_name, img_list in self.gr00t_img_buffers[env_id].items():

                img_tensor = torch.stack(img_list, dim=0).to(device="cpu", dtype=torch.uint8) # (T, H, W, C)

                gr00t_video_path = os.path.join(
                    video_dir[camera_name], f"episode_{episode_index:06d}.mp4"
                )
                os.makedirs(os.path.dirname(gr00t_video_path), exist_ok=True)

                write_video(gr00t_video_path, img_tensor, fps=int(1/self.step_dt), video_codec="h264")

        print(
            f"Saved {len(successful_env_ids)} successful episodes out of "
            f"{self.num_envs} attempts to {data_dir} and {video_dir}"
        )
        self._saved_episode_count += len(successful_env_ids)
        print(f"number of frames: {len(self.gr00t_data_buffers[0]['observation.state'])}")
        target = self.demo_save_cfg.target_successful_episodes
        if target > 0 and self._saved_episode_count < target:
            print(f"collection progress: {self._saved_episode_count} / {target} successful episodes")
            return
        print("Exiting program after reaching the demonstration collection target...")

        # Quit Program
        sys.exit(0)

    def _capture_gr00t_transition(self, action):
        """Capture the observation and action before physics so they are time-aligned."""
        state = torch.cat(
            (self.noisy_fingertip_pos, self.noisy_fingertip_quat, self.joint_pos[:, 7:9]), dim=-1
        )
        arm_action = action[:, :6]
        gripper_action = torch.ones((self.num_envs, 1), device=self.device) # This Task forces gripper to be closed
        gr00t_action = torch.cat((arm_action, gripper_action), dim=-1)

        self._pending_gr00t_transition = {
            "state": state.detach().cpu(),
            "force": self.noisy_force.detach().cpu(),
            "action": gr00t_action.detach().cpu(),
            "step": self.episode_length_buf.detach().cpu().clone(),
            "left_camera": self._left_camera.data.output["rgb"].detach().cpu().clone(),
            "right_camera": self._right_camera.data.output["rgb"].detach().cpu().clone(),
            "wrist_camera": self._wrist_camera.data.output["rgb"].detach().cpu().clone(),
        }

    def _commit_gr00t_transition(self, true_successes):
        """Attach post-action outcome fields and append the pending transition."""
        if self._pending_gr00t_transition is None:
            raise RuntimeError("Demo transition outcome was computed without a pending observation/action.")

        pending = self._pending_gr00t_transition
        successes = true_successes.detach().cpu()

        if pending["step"][0].item() % 10 == 0 or true_successes.any():
            print("================== Parquet Data Stats =================")
            print(f"current / max: {pending['step'][0].item()} / {self.max_episode_length - 2}")
            print(f"success / total: {true_successes.sum().item()} / {self.num_envs}")

        for i in range(self.num_envs):
            current_step_in_ep = pending["step"][i].item()

            # update data buffers
            self.gr00t_data_buffers[i]["observation.state"].append(pending["state"][i].numpy().tolist())
            self.gr00t_data_buffers[i]["observation.force"].append(pending["force"][i].numpy().tolist())
            self.gr00t_data_buffers[i]["action"].append(pending["action"][i].numpy().tolist())
            self.gr00t_data_buffers[i]["timestamp"].append(round(current_step_in_ep * self.step_dt, 3))
            self.gr00t_data_buffers[i]["annotation.human.action.task_description"].append(0) # Task index 0
            self.gr00t_data_buffers[i]["task_index"].append(0)
            self.gr00t_data_buffers[i]["annotation.human.validity"].append(False)
            self.gr00t_data_buffers[i]["episode_index"].append(i)
            self.gr00t_data_buffers[i]["index"].append(i * (self.max_episode_length - 1) + current_step_in_ep - 1)
            self.gr00t_data_buffers[i]["next.reward"].append(float(successes[i].item()))
            if current_step_in_ep == self.max_episode_length - 2: # Final Step of episode
                self.gr00t_data_buffers[i]["next.done"].append(True)
            else:
                self.gr00t_data_buffers[i]["next.done"].append(False)
            self.gr00t_data_buffers[i]["episode.success"].append(False)

            # update image buffers
            self.gr00t_img_buffers[i]["left_camera"].append(pending["left_camera"][i])
            self.gr00t_img_buffers[i]["right_camera"].append(pending["right_camera"][i])
            self.gr00t_img_buffers[i]["wrist_camera"].append(pending["wrist_camera"][i])

        self._pending_gr00t_transition = None
