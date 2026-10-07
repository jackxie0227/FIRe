"""CPU-only checks for the strict three-policy batch launcher."""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts/reinforcement_learning/run_three_policy_paired_500.py"
SPEC = importlib.util.spec_from_file_location("run_three_policy_paired_500", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_policy_commands_are_fresh_one_episode_processes(tmp_path):
    spec = {
        "evaluator": "/fake/evaluate_paired_policy.py",
        "teacher_checkpoint": "/fake/teacher.pth",
        "residual_checkpoint": "/fake/residual.pth",
        "vla_host": "127.0.0.1",
        "vla_port": 5555,
    }
    for policy in runner.POLICIES:
        command = runner.command_for_trial(policy, 100000, tmp_path / f"{policy}.json", spec)
        assert command[0] == runner.sys.executable
        assert command[command.index("--episodes") + 1] == "1"
        assert command[command.index("--eval-num-envs") + 1] == "1"
        assert command[command.index("--condition-seed-start") + 1] == "100000"
        assert command[command.index("--success-criterion") + 1] == runner.CRITERION
        assert "--headless" in command and "--enable_cameras" in command
        if policy == "teacher":
            assert "--checkpoint" in command
            assert not any(argument.startswith("env.vla_host=") for argument in command)
        elif policy == "vla":
            assert "--checkpoint" not in command
            assert "env.vla_host=127.0.0.1" in command
        else:
            assert command[command.index("--checkpoint") + 1] == "/fake/residual.pth"
            assert "env.vla_host=127.0.0.1" in command


def _payload(policy, seed, spec, state=None, success=1):
    checkpoint = spec["teacher_checkpoint"] if policy == "teacher" else (
        spec["residual_checkpoint"] if policy == "residual" else None
    )
    checkpoint_sha256 = spec["teacher_sha256"] if policy == "teacher" else (
        spec["residual_sha256"] if policy == "residual" else None
    )
    return {
        "policy": runner.EVALUATOR_POLICIES[policy],
        "task": runner.TASKS[policy],
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "episodes": 1,
        "requested_episodes": 1,
        "condition_seed_start": seed,
        "success_criterion": runner.CRITERION,
        "complete": True,
        "all_outputs_finite": True,
        "vla_request_error_count": 0,
        "trials": [{
            "condition_seed": seed,
            "condition_index": 0,
            "success": success,
            "initial_state": state or {"fixed_pos": [0.6, 0.0, 0.04], "contact_penalty_thresholds": 10.0},
            "all_outputs_finite": True,
            "vla_request_error_count": 0,
            "zero_residual_composition_exact": policy == "vla",
        }],
    }


def test_trial_validation_and_strict_three_way_pairing():
    spec = {"teacher_checkpoint": "/fake/teacher.pth", "teacher_sha256": "aa", "residual_checkpoint": "/fake/residual.pth", "residual_sha256": "bb"}
    group = {
        policy: runner.validate_trial(_payload(policy, 100000, spec), policy=policy, seed=100000, spec=spec)
        for policy in runner.POLICIES
    }
    assert runner.check_triple_states(group, seed=100000) == 0.0

    displaced = dict(group["residual"])
    displaced["initial_state"] = {"fixed_pos": [0.6, 0.0, 0.04002], "contact_penalty_thresholds": 10.0}
    group["residual"] = displaced
    with pytest.raises(ValueError, match="NOT a strict three-policy matched scene"):
        runner.check_triple_states(group, seed=100000)


def test_wrong_checkpoint_and_criterion_are_rejected():
    spec = {"teacher_checkpoint": "/fake/teacher.pth", "teacher_sha256": "aa", "residual_checkpoint": "/fake/residual.pth", "residual_sha256": "bb"}
    payload = _payload("residual", 100000, spec)
    payload["checkpoint"] = "/fake/other.pth"
    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_trial(payload, policy="residual", seed=100000, spec=spec)
    payload["checkpoint"] = spec["residual_checkpoint"]
    payload["success_criterion"] = "legacy"
    with pytest.raises(ValueError, match="success_criterion"):
        runner.validate_trial(payload, policy="residual", seed=100000, spec=spec)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("checkpoint_sha256", "wrong"), ("all_outputs_finite", False), ("vla_request_error_count", 1)],
)
def test_bad_result_integrity_fields_are_rejected(field, bad_value):
    spec = {"teacher_checkpoint": "/fake/teacher.pth", "teacher_sha256": "aa", "residual_checkpoint": "/fake/residual.pth", "residual_sha256": "bb"}
    payload = _payload("residual", 100000, spec)
    payload[field] = bad_value
    with pytest.raises(ValueError, match=field):
        runner.validate_trial(payload, policy="residual", seed=100000, spec=spec)


def test_manifest_requires_explicit_resume_and_same_spec(tmp_path):
    spec = {"schema_version": 1, "model_dir": "/fake/model"}
    runner.ensure_manifest(tmp_path, spec, write=True, resume=False)
    with pytest.raises(ValueError, match="--resume"):
        runner.ensure_manifest(tmp_path, spec, write=True, resume=False)
    runner.ensure_manifest(tmp_path, spec, write=True, resume=True)
    with pytest.raises(ValueError, match="incompatible"):
        runner.ensure_manifest(tmp_path, {**spec, "model_dir": "/changed"}, write=True, resume=True)


def test_one_group_then_resume_without_reexecuting_it(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "EPISODES", 2)
    monkeypatch.setattr(runner, "check_no_other_simulator", lambda: None)
    monkeypatch.setattr(runner, "check_service", lambda host, port: None)
    spec = {
        "schema_version": 1,
        "teacher_checkpoint": "/fake/teacher.pth",
        "teacher_sha256": "aa",
        "residual_checkpoint": "/fake/residual.pth",
        "residual_sha256": "bb",
        "condition_seed_start": 100000,
    }
    monkeypatch.setattr(runner, "model_and_checkpoint_spec", lambda args: spec)
    calls = []

    def fake_run_one(policy, seed, output_dir, spec):
        path = runner.trial_path(output_dir, seed, policy)
        if not path.exists():
            calls.append((seed, policy))
            runner.atomic_json(path, _payload(policy, seed, spec))
        return runner.validate_trial(runner.load_json(path), policy=policy, seed=seed, spec=spec)

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    arguments = Namespace(
        output_dir=tmp_path,
        teacher_checkpoint=Path("/fake/teacher.pth"),
        residual_checkpoint=Path("/fake/residual.pth"),
        model_dir=Path("/fake/model"),
        vla_host="127.0.0.1",
        vla_port=5555,
        condition_seed_start=100000,
        count=1,
        resume=False,
        preflight_only=False,
    )
    runner.run(arguments)
    assert len(calls) == 3
    assert runner.load_json(tmp_path / "teacher_500.json")["episodes"] == 1
    arguments.count = 2
    arguments.resume = True
    runner.run(arguments)
    assert len(calls) == 6
    for policy in runner.POLICIES:
        aggregate = runner.load_json(tmp_path / f"{policy}_500.json")
        assert aggregate["episodes"] == 2
        assert aggregate["complete"] is True
        assert [trial["condition_index"] for trial in aggregate["trials"]] == [0, 1]
    pairing = runner.load_json(tmp_path / "pairing_audit.json")
    assert pairing["matched_groups"] == 2
    assert pairing["complete"] is True


def test_failed_pair_is_not_aggregated(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "EPISODES", 1)
    monkeypatch.setattr(runner, "check_no_other_simulator", lambda: None)
    monkeypatch.setattr(runner, "check_service", lambda host, port: None)
    spec = {
        "schema_version": 1,
        "teacher_checkpoint": "/fake/teacher.pth",
        "teacher_sha256": "aa",
        "residual_checkpoint": "/fake/residual.pth",
        "residual_sha256": "bb",
        "condition_seed_start": 100000,
    }
    monkeypatch.setattr(runner, "model_and_checkpoint_spec", lambda args: spec)

    def fake_run_one(policy, seed, output_dir, spec):
        state = {"fixed_pos": [0.6, 0.0, 0.04 if policy != "residual" else 0.041]}
        return runner.validate_trial(
            _payload(policy, seed, spec, state=state), policy=policy, seed=seed, spec=spec
        )

    monkeypatch.setattr(runner, "run_one", fake_run_one)
    arguments = Namespace(
        output_dir=tmp_path,
        teacher_checkpoint=Path("/fake/teacher.pth"),
        residual_checkpoint=Path("/fake/residual.pth"),
        model_dir=Path("/fake/model"),
        vla_host="127.0.0.1",
        vla_port=5555,
        condition_seed_start=100000,
        count=1,
        resume=False,
        preflight_only=False,
    )
    with pytest.raises(ValueError, match="NOT a strict three-policy matched scene"):
        runner.run(arguments)
    assert runner.load_json(tmp_path / "pairing_failure.json")["counted"] is False
    assert not (tmp_path / "teacher_500.json").exists()
