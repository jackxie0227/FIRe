# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).parents[3]
PKG = ROOT / "source/fire_lab/fire_lab"
FORGE = PKG / "tasks/direct/vla/gr00t/forge"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_official_action_composition_clips_only_residual():
    module = _load_module("vla_action", PKG / "envs/vla_action.py")
    vla = torch.tensor([[0.25, -0.5, 1.2, 0.0, 0.4, -1.4]])
    zero = torch.zeros_like(vla)
    residual, combined = module.compose_vla_residual_actions(vla, zero, 1.0)
    assert torch.equal(residual, zero)
    assert torch.equal(combined, vla)

    raw = torch.tensor([[2.0, -3.0, 0.5, 0.0, 1.1, -1.0]])
    residual, combined = module.compose_vla_residual_actions(vla, raw, 1.0)
    assert residual.tolist() == [[1.0, -1.0, 0.5, 0.0, 1.0, -1.0]]
    assert combined[0, 2].item() == pytest.approx(1.7)
    assert combined[0, 5].item() == pytest.approx(-2.4)


def test_official_force_reward_and_inclusive_success_equations():
    module = _load_module("fire_official", FORGE / "fire_official.py")
    force = torch.tensor([4.0, 10.0, 12.0])
    reference = torch.tensor([5.0, 10.0, 10.0])
    previous = torch.zeros(3)
    over, penalty, shaping, potential = module.force_reward_terms(force, reference, previous)
    assert torch.equal(over, torch.tensor([0.0, 0.0, 2.0]))
    assert torch.equal(penalty[:2], torch.zeros(2))
    assert torch.equal(shaping[:2], torch.zeros(2))
    assert penalty[2].item() == pytest.approx(-2.0e-5)
    assert potential[2].item() == pytest.approx(-torch.log(torch.tensor(3.0)).item())
    assert shaping[2].item() == pytest.approx(1.0e-5 * 0.995 * potential[2].item())

    success = module.peg_insert_success_mask(
        torch.tensor([0.005, 0.00501]), torch.tensor([0.025, 0.0])
    )
    assert success.tolist() == [True, False]


def test_official_task_dimensions_and_legacy_contract_are_isolated():
    cfg_source = (FORGE / "forge_gr00t_env_cfg.py").read_text()
    tree = ast.parse(cfg_source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    official = classes["ForgeTaskPegInsertGr00tOfficialCfg"]
    official_text = ast.get_source_segment(cfg_source, official)
    assert "action_space: int = 6" in official_text
    assert '"force_threshold"' not in official_text.split("obs_order", 1)[1]
    assert '"ft_force"' in official_text
    assert "refresh_vla_on_episode_reset: bool = True" in official_text
    assert "self.sim.render_interval = self.decimation" in official_text

    legacy_source = (ROOT / "source/fire_lab/fire_lab/tasks/direct/base_line/forge/forge_env_cfg.py").read_text()
    assert "action_space: int = 7" in legacy_source
    assert "class ForgeTaskPegInsertGr00tDemoSaveCfg" in cfg_source

    register_source = (FORGE / "__init__.py").read_text()
    assert "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-Direct-v1" in register_source
    assert "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-SampleEq4-Direct-v1" in register_source
    assert "FireLab-VLA-Gr00t-Forge-PegInsert-Demo-Save-Direct-v0" in register_source


@pytest.mark.parametrize(
    ("filename", "num_actors", "horizon"),
    [
        ("rl_games_ppo_cfg_peg_insert_fire_official.yaml", 64, 128),
        ("rl_games_ppo_cfg_peg_insert_fire_official_sampleeq4.yaml", 4, 2048),
    ],
)
def test_official_ppo_batch_and_hyperparameters(filename, num_actors, horizon):
    config = yaml.safe_load((FORGE / "agents" / filename).read_text())["params"]["config"]
    assert config["num_actors"] == num_actors
    assert config["horizon_length"] == horizon
    assert num_actors * horizon == 8192
    assert config["minibatch_size"] == 512
    assert (num_actors * horizon) // config["minibatch_size"] == 16
    assert config["mini_epochs"] == 4
    assert config["gamma"] == pytest.approx(0.995)
    assert config["tau"] == pytest.approx(0.95)
    assert config["learning_rate"] == pytest.approx(1.0e-4)
    assert config["e_clip"] == pytest.approx(0.2)
    assert config["entropy_coef"] == pytest.approx(0.0)
    assert config["max_epochs"] == 200
    assert config["save_frequency"] == 10
    assert config["save_best_after"] > config["max_epochs"]
    assert config["success_checkpoint"]["window_size"] == 100


def test_force_sample_is_reused_and_complete_episode_tags_exist():
    source = (FORGE / "forge_gr00t_env.py").read_text()
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ForgeGr00tEnv")
    methods = {node.name: node for node in class_node.body if isinstance(node, ast.FunctionDef)}
    get_force_text = ast.get_source_segment(source, methods["get_ft_force"])
    assert "get_link_incoming_joint_force" not in get_force_text
    assert "return self.noisy_force, self.force_sensor_smooth[:, 0:3]" in get_force_text

    required = {
        '"success_rate"',
        '"success_time"',
        '"contact_force_mean"',
        '"contact_force_max"',
        '"contact_force_p95"',
        '"over_force_fraction"',
        '"xy_error"',
        '"z_displacement"',
        '"Reward/r_s"',
        '"Reward/r_penalty"',
        '"Reward/r_shaping"',
        '"Action/vla_abs_mean"',
        '"Action/residual_abs_mean"',
        '"Action/combined_abs_mean"',
        '"Action/residual_clip_fraction"',
        '"Action/combined_out_of_range_fraction"',
    }
    assert all(tag in source for tag in required)
    assert 'self._prev_phi_f[env_ids] = 0.0' in (
        ROOT / "source/fire_lab/fire_lab/tasks/direct/vla/common/forge_env.py"
    ).read_text()


def test_success_observer_embeds_resume_state_and_writes_best_metadata(tmp_path):
    module = _load_module("fire_success_observer", PKG / "utils/rl_games_success_observer.py")
    cfg = {
        "params": {
            "seed": 999,
            "config": {
                "success_checkpoint": {
                    "enabled": True,
                    "window_size": 4,
                    "checkpoint_stem": "Forge_best_success",
                },
                "fire_task_id": "official-test",
                "num_actors": 4,
                "horizon_length": 128,
            },
        }
    }

    class Writer:
        def add_scalar(self, *args):
            pass

    class Meter:
        current_size = 0

    class Algo:
        nn_dir = str(tmp_path)
        device = torch.device("cpu")
        last_lr = 1.0e-4
        entropy_coef = 0.0
        scheduler = type("Scheduler", (), {"kl_threshold": 0.008})()

        def save(self, stem):
            torch.save(self.get_full_state_weights(), stem + ".pth")

        def get_full_state_weights(self):
            return {"epoch": 3, "frame": 512}

        def set_full_state_weights(self, weights, set_epoch=True):
            self.restored = weights

        def update_lr(self, learning_rate):
            self.updated_lr = learning_rate

    observer = module.FireSuccessObserver(cfg)
    observer.algo = Algo()
    observer.writer = Writer()
    observer.ep_infos = []
    observer.direct_info = {}
    observer.mean_scores = Meter()
    observer._install_resume_state_hooks()

    infos = {
        "episode": {"success_rate": torch.tensor([0.0, 1.0, 1.0, 0.0])},
        "fire_episode": {"Reward/r_s": torch.tensor([0.0, 0.1, 0.2, 0.0])},
    }
    observer.process_infos(infos, torch.arange(4))
    observer.after_print_stats(frame=512, epoch_num=3, total_time=1.0)

    checkpoint = tmp_path / "Forge_best_success.pth"
    metadata = tmp_path / "Forge_best_success.json"
    assert checkpoint.is_file()
    assert metadata.is_file()
    checkpoint_state = torch.load(checkpoint, weights_only=False)
    assert module.FireSuccessObserver.STATE_KEY in checkpoint_state
    payload = yaml.safe_load(metadata.read_text())
    assert payload["window_success_rate"] == pytest.approx(0.5)
    assert payload["sha256"] == module.sha256_file(checkpoint)

    observer.rolling_successes.clear()
    observer.best_success_rate = -1.0
    observer.algo.last_lr = 9.0e-4
    observer.algo.set_full_state_weights(checkpoint_state)
    assert list(observer.rolling_successes) == [0.0, 1.0, 1.0, 0.0]
    assert observer.best_success_rate == pytest.approx(0.5)
    assert observer.algo.last_lr == pytest.approx(1.0e-4)
