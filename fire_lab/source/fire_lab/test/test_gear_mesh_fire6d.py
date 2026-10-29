"""GearMesh-only inference and configuration contracts without Isaac Sim startup."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).parents[3]
PKG = ROOT / "source/fire_lab/fire_lab"
FORGE = PKG / "tasks/direct/vla/gr00t/forge"
spec = importlib.util.spec_from_file_location("strict_gr00t", PKG / "envs/strict_gr00t.py")
strict = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(strict)


def response(batch=4, horizon=16):
    return {
        "action.eef_position_delta": torch.ones(batch, horizon, 3),
        "action.eef_rotation_delta": torch.full((batch, horizon, 3), 0.25),
    }


def test_exact_six_dimensional_chunk_and_no_cross_env_copy():
    out = response()
    out["action.eef_position_delta"][:, :, 0] = torch.arange(4)[:, None]
    chunk = strict.validate_pose_chunk(out, num_envs=4, chunk_size=16, device="cpu")
    assert chunk.shape == (4, 16, 6)
    assert chunk[:, 0, 0].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (None, "mapping"),
        ({}, "missing action keys"),
        (response(batch=1), "expected (4, 16, 3)"),
        (response(horizon=15), "expected (4, 16, 3)"),
    ],
)
def test_rejects_missing_and_wrong_shape(bad, message):
    with pytest.raises(ValueError, match=message.replace("(", "\\(").replace(")", "\\)")):
        strict.validate_pose_chunk(bad, num_envs=4, chunk_size=16, device="cpu")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_pose(bad_value):
    out = response()
    out["action.eef_rotation_delta"][2, 3, 1] = bad_value
    with pytest.raises(ValueError, match="NaN or Inf"):
        strict.validate_pose_chunk(out, num_envs=4, chunk_size=16, device="cpu")


def test_reset_discards_pending_old_episode_chunk_and_uses_new_observations():
    class Client:
        def __init__(self):
            self.calls = []

        def get_result(self):
            self.calls.append("drain")
            return response()

        def get_action_sync(self, obs):
            self.calls.append(("sync", obs["episode_id"]))
            out = response()
            out["action.eef_position_delta"] *= obs["episode_id"]
            return out

    client = Client()
    returned, chunk = strict.refresh_episode_chunk(
        client, {"episode_id": 2},
        lambda out: strict.validate_pose_chunk(out, num_envs=4, chunk_size=16, device="cpu"),
        pending=True,
    )
    assert client.calls == ["drain", ("sync", 2)]
    assert returned["action.eef_position_delta"][0, 0, 0] == 2
    assert torch.all(chunk[:, :, :3] == 2)
    # 300 control steps and 16-step chunks leave an in-flight request at timeout.
    assert 300 % 16 == 12
    strict.require_synchronized_reset(torch.arange(4), 4)
    with pytest.raises(RuntimeError, match="partial reset"):
        strict.require_synchronized_reset(torch.tensor([1, 3]), 4)


def test_gear_mesh_only_registration_config_and_ppo_batch():
    cfg_source = (FORGE / "forge_gr00t_env_cfg.py").read_text()
    registry = (FORGE / "__init__.py").read_text()
    assert "class ForgeTaskGearMeshFIRe6DCfg" in cfg_source
    assert "action_space: int = 6" in cfg_source.split("class ForgeTaskGearMeshFIRe6DCfg", 1)[1]
    assert "strict_gr00t_response: bool = True" in cfg_source
    assert "refresh_vla_on_episode_reset: bool = True" in cfg_source
    assert "rerender_on_reset: bool = True" in cfg_source
    assert "FireLab-VLA-Gr00t-Forge-GearMesh-FIRe6D-Direct-v1" in registry
    assert "FireLab-VLA-Gr00t-Forge-GearMesh-Direct-v1" in registry
    config = yaml.safe_load((FORGE / "agents/rl_games_ppo_cfg_gear_mesh_fire6d.yaml").read_text())["params"]["config"]
    assert (config["num_actors"], config["horizon_length"], config["minibatch_size"], config["seq_length"]) == (4, 2048, 512, 128)
    assert config["success_checkpoint"]["enabled"] is True
    assert config["success_checkpoint"]["window_size"] == 100
