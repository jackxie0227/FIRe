# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch

from fire_lab.tasks.direct.base_line.factory import factory_utils
from fire_lab.tasks.direct.vla.common.forge_env import ForgeEnv
from fire_lab.tasks.direct.vla.gr00t.forge.fire_official import force_reward_terms
from fire_lab.tasks.direct.vla.gr00t.forge.forge_gr00t_env_cfg import ForgeGr00tEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import TiledCamera
import wandb


class ForgeGr00tEnv(ForgeEnv):
    cfg: ForgeGr00tEnvCfg

    def __init__(self, cfg: ForgeGr00tEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize additional randomization and logging tensors."""
        super().__init__(cfg, render_mode, **kwargs)
        self._init_episode_audit_tensors()

    def _init_episode_audit_tensors(self) -> None:
        """Allocate complete-episode accumulators used by TensorBoard."""

        zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_steps = zeros()
        self._episode_force_sum = zeros()
        self._episode_force_max = zeros()
        self._episode_over_force_steps = zeros()
        self._episode_r_s = zeros()
        self._episode_r_penalty = zeros()
        self._episode_r_shaping = zeros()
        self._episode_vla_abs = zeros()
        self._episode_residual_abs = zeros()
        self._episode_combined_abs = zeros()
        self._episode_residual_clip = zeros()
        self._episode_combined_out_of_range = zeros()
        self._episode_force_history = torch.zeros(
            (self.num_envs, self.max_episode_length + 1),
            dtype=torch.float32,
            device=self.device,
        )

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
        rew_buf = self.get_sparse_rewards()

        return rew_buf

    def get_ft_force(self):
        """Return the one force sample produced for the current physical state.

        ``FactoryEnv._get_dones`` refreshes intermediate values after the final
        physics sub-step.  Reward, critic, and actor observations all reuse that
        sample; only the actor receives its already-computed noisy view.
        """

        return self.noisy_force, self.force_sensor_smooth[:, 0:3]

    def _get_alignment_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        held_base_pos, _ = factory_utils.get_held_base_pose(
            self.held_pos,
            self.held_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )
        xy_error = torch.linalg.vector_norm(
            target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1
        )
        z_displacement = held_base_pos[:, 2] - target_held_base_pos[:, 2]
        return xy_error, z_displacement

    def _update_episode_audit(
        self,
        *,
        contact_force: torch.Tensor,
        contact_over: torch.Tensor,
        r_s: torch.Tensor,
        r_penalty: torch.Tensor,
        r_shaping: torch.Tensor,
    ) -> None:
        self._episode_steps += 1.0
        self._episode_force_sum += contact_force
        self._episode_force_max = torch.maximum(self._episode_force_max, contact_force)
        self._episode_over_force_steps += contact_over.gt(0.0).float()
        self._episode_r_s += r_s
        self._episode_r_penalty += r_penalty
        self._episode_r_shaping += r_shaping

        history_idx = (self._episode_steps.long() - 1).clamp(
            min=0, max=self._episode_force_history.shape[1] - 1
        )
        self._episode_force_history[
            torch.arange(self.num_envs, device=self.device), history_idx
        ] = contact_force

        self._episode_vla_abs += self._last_vla_action.abs().mean(dim=-1)
        self._episode_residual_abs += self._last_residual_action.abs().mean(dim=-1)
        self._episode_combined_abs += self._last_combined_action.abs().mean(dim=-1)
        self._episode_residual_clip += self._last_residual_clip_fraction
        self._episode_combined_out_of_range += self._last_combined_out_of_range_fraction

    def _publish_completed_episode_audit(self) -> None:
        if not torch.any(self.reset_buf):
            return

        completed_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        steps = self._episode_steps[completed_ids].clamp_min(1.0)
        force_p95 = torch.stack(
            [
                torch.quantile(
                    self._episode_force_history[env_id, : int(self._episode_steps[env_id].item())],
                    0.95,
                )
                for env_id in completed_ids
            ]
        )
        xy_error, z_displacement = self._get_alignment_errors()
        success_time_steps = torch.where(
            self.ep_succeeded[completed_ids].bool(),
            self.ep_success_times[completed_ids].float(),
            steps,
        )

        # ``episode`` is handled by IsaacAlgoObserver and receives the standard
        # Episode/ prefix.  Failed episodes are right-censored at episode end for
        # success_time, keeping the aggregate finite and interpretable.
        self.extras["episode"] = {
            "success_rate": self.ep_succeeded[completed_ids].float().detach(),
            "success_time": (success_time_steps * self.step_dt).detach(),
            "contact_force_mean": (self._episode_force_sum[completed_ids] / steps).detach(),
            "contact_force_max": self._episode_force_max[completed_ids].detach(),
            "contact_force_p95": force_p95.detach(),
            "over_force_fraction": (
                self._episode_over_force_steps[completed_ids] / steps
            ).detach(),
            "xy_error": xy_error[completed_ids].detach(),
            "z_displacement": z_displacement[completed_ids].detach(),
        }
        # The custom observer logs these pre-prefixed names exactly as listed in
        # the stage-eight metric contract.
        self.extras["fire_episode"] = {
            "Reward/r_s": (self._episode_r_s[completed_ids] / steps).detach(),
            "Reward/r_penalty": (self._episode_r_penalty[completed_ids] / steps).detach(),
            "Reward/r_shaping": (self._episode_r_shaping[completed_ids] / steps).detach(),
            "Action/vla_abs_mean": (self._episode_vla_abs[completed_ids] / steps).detach(),
            "Action/residual_abs_mean": (
                self._episode_residual_abs[completed_ids] / steps
            ).detach(),
            "Action/combined_abs_mean": (
                self._episode_combined_abs[completed_ids] / steps
            ).detach(),
            "Action/residual_clip_fraction": (
                self._episode_residual_clip[completed_ids] / steps
            ).detach(),
            "Action/combined_out_of_range_fraction": (
                self._episode_combined_out_of_range[completed_ids] / steps
            ).detach(),
        }

    def get_sparse_rewards(self) -> torch.Tensor:
        """
        Reward structure (paper-aligned):
            r = r_s + r_penalty + r_shaping
            r_s        : sparse success reward
            r_penalty  : -alpha * max(0, ||F|| - F_limit)
            r_shaping  : lambda_f * (gamma * phi(s_{t+1}) - phi(s_t))
            phi(s)     : -log(c_f * max(0, ||F|| - F_limit) + 1)
            gamma      : fixed to 0.995
        """
        self.extras.pop("episode", None)
        self.extras.pop("fire_episode", None)
        # ------------------------------------------------------------
        # Initialize reward buffer
        # ------------------------------------------------------------
        rew_buf = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        # ============================================================
        # (1) Success reward: r_s
        # ============================================================
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold,
            check_rot=check_rot,
        )
        first_successes = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1
        first_success_ids = first_successes.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        r_s = curr_successes.float()
        # ============================================================
        # (2) Contact force penalty: r_penalty
        #     r_penalty = -alpha * max(0, ||F|| - F_limit)
        # ============================================================
        _, force_sensor_smooth = self.get_ft_force()
        contact_force = torch.norm(
            force_sensor_smooth[:, 0:3], p=2, dim=-1
        )

        F_limit = self.contact_penalty_thresholds
        alpha = getattr(self.cfg_task, "contact_penalty_alpha", None)
        if alpha is None:
            alpha = getattr(self.cfg_task, "contact_penalty_scale", 1.0)
        # ============================================================
        # (3) Potential-based shaping: r_shaping
        #     r_shaping = lambda_f * (gamma * phi(s_{t+1}) - phi(s_t))
        # ============================================================
        gamma = 0.995  # FIXED as requested
        lambda_f = getattr(self.cfg_task, "force_shaping_lambda", 0.01)
        c_f = getattr(self.cfg_task, "force_potential_c", 1.0)
        contact_over, r_penalty, r_shaping, phi_curr = force_reward_terms(
            contact_force,
            F_limit,
            self._prev_phi_f,
            alpha=alpha,
            shaping_lambda=lambda_f,
            gamma=gamma,
            potential_c=c_f,
        )
        # update buffer for next step
        self._prev_phi_f = phi_curr.detach()
        # ============================================================
        # (4) Total reward
        # ============================================================
        rew_buf += r_s
        rew_buf += r_penalty
        rew_buf += r_shaping

        self._update_episode_audit(
            contact_force=contact_force,
            contact_over=contact_over,
            r_s=r_s,
            r_penalty=r_penalty,
            r_shaping=r_shaping,
        )
        self._publish_completed_episode_audit()

        # Evaluation utilities such as ``zero_agent.py`` do not initialize a
        # W&B run.  Logging unconditionally makes the environment fail on its
        # first step, before a VLA-only rollout can be observed.  Keep direct
        # W&B logging for tracked PPO runs, but make it optional for inference.
        if wandb.run is not None:
            wandb.log({
                "success rate": curr_successes.float().mean() * 100,
                "r_penalty": r_penalty.float().mean(),
                "r_shaping": r_shaping.float().mean(),
            })
        # print(f"======= {self.episode_length_buf[0]} / {self.max_episode_length} =======")
        # print(f"success rate: {curr_successes.float().mean()*100}%")

        return rew_buf

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "_episode_steps"):
            return
        for name in (
            "_episode_steps",
            "_episode_force_sum",
            "_episode_force_max",
            "_episode_over_force_steps",
            "_episode_r_s",
            "_episode_r_penalty",
            "_episode_r_shaping",
            "_episode_vla_abs",
            "_episode_residual_abs",
            "_episode_combined_abs",
            "_episode_residual_clip",
            "_episode_combined_out_of_range",
        ):
            getattr(self, name)[env_ids] = 0.0
        self._episode_force_history[env_ids, :] = 0.0

    def _get_observations(self):
        """Add additional FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        noisy_force, force_sensor_smooth = self.get_ft_force()

        obs_dict.update({
            "fingertip_pos": self.noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": self.noisy_fingertip_quat,
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "ft_force": noisy_force,
            "prev_actions": prev_actions,
        })

        state_dict.update({
            "ema_factor": self.ema_factor,
            "ft_force": force_sensor_smooth[:, 0:3],
            "force_threshold": self.contact_penalty_thresholds[:, None],
            "prev_actions": prev_actions,
        })

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        return {"policy": obs_tensors, "critic": state_tensors}
