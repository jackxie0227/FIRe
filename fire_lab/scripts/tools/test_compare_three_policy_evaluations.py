"""CPU-only contract tests for the strict three-policy comparator."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("compare_three_policy_evaluations.py")
spec = importlib.util.spec_from_file_location("compare_three_policy_evaluations", SCRIPT)
assert spec is not None and spec.loader is not None
comparator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparator)


def _state(seed: int) -> dict:
    return {name: [float(seed % 23) * 0.01] for name in comparator.REQUIRED_STATE_FIELDS}


def _run(name: str, successes: set[int]) -> dict:
    trials = [
        {
            "condition_index": index,
            "condition_seed": 100000 + index,
            "initial_state": _state(100000 + index),
            "success": int(index in successes),
            "steps": 149,
            "all_outputs_finite": True,
            "vla_request_error_count": 0,
        }
        for index in range(500)
    ]
    return {
        "policy": comparator.POLICY_LABELS[name],
        "task": comparator.TASK_IDS[name],
        "checkpoint": None if name == "vla" else f"{name}.pth",
        "checkpoint_sha256": None if name == "vla" else "a" * 64,
        "success_criterion": comparator.EXPECTED_CRITERION,
        "episodes": 500,
        "requested_episodes": 500,
        "condition_seed_start": 100000,
        "complete": True,
        "success_count": len(successes),
        "success_rate": len(successes) / 500,
        "all_outputs_finite": True,
        "vla_request_error_count": 0,
        "trials": trials,
    }


def _runs() -> dict:
    return {
        "teacher": _run("teacher", set(range(400))),
        "vla": _run("vla", set(range(100))),
        "residual": _run("residual", set(range(350))),
    }


def test_exact_500_paired_success_statistics() -> None:
    result = comparator.compare(_runs())
    assert result["strictly_paired"] is True
    assert result["paired_trials"] == len(result["trials"]) == 500
    assert result["max_initial_state_abs_difference"] == 0.0
    assert result["policies"]["teacher"]["success_rate"] == 0.8
    assert result["policies"]["vla"]["success_rate"] == 0.2
    assert result["policies"]["residual"]["success_rate"] == 0.7
    assert result["pairwise"]["teacher_vs_vla"]["left_only"] == 300
    assert result["pairwise"]["teacher_vs_vla"]["mcnemar_exact_p"] < 1e-80
    assert result["pairwise"]["vla_vs_residual"]["right_only"] == 250
    for policy in result["policies"].values():
        assert len(policy["success_rate_wilson_95"]) == 2


def test_state_mismatch_fails_instead_of_dropping_trial() -> None:
    runs = _runs()
    runs["vla"]["trials"][123]["initial_state"]["joint_pos"][0] += 0.001
    with pytest.raises(ValueError, match="Condition 123.*initial states differ.*no trials were dropped"):
        comparator.compare(runs)


def test_seed_and_index_mismatches_fail() -> None:
    runs = _runs()
    runs["residual"]["trials"][7]["condition_seed"] += 1000
    with pytest.raises(ValueError, match="condition_seed must equal 100007"):
        comparator.compare(runs)
    runs = _runs()
    runs["residual"]["trials"][7]["condition_index"] = 9
    with pytest.raises(ValueError, match="non-contiguous condition_index"):
        comparator.compare(runs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda r: r["teacher"].update(complete=False), "complete must be true"),
        (lambda r: r["teacher"]["trials"].pop(), "exactly 500 trials"),
        (lambda r: r["teacher"].update(requested_episodes=499), "requested_episodes"),
        (lambda r: r["vla"].update(success_criterion="legacy"), "success_criterion"),
        (lambda r: r["vla"]["trials"][0].update(all_outputs_finite=False), "non-finite rollout"),
        (lambda r: r["residual"]["trials"][0].update(vla_request_error_count=1), "VLA request error"),
        (lambda r: r["teacher"]["trials"][0]["initial_state"].pop("joint_pos"), "missing initial-state"),
        (lambda r: r["teacher"].update(success_count=0), "success_count does not match"),
        (lambda r: r["teacher"].update(success_rate=0.1), "success_rate does not match"),
        (lambda r: r["residual"]["trials"][0].update(steps=0), "steps must be a positive integer"),
        (lambda r: r["vla"].update(condition_seed_start=100001), "condition_seed_start"),
        (lambda r: r["teacher"].update(policy="gr00t"), "policy label"),
        (lambda r: r["vla"].update(task="legacy"), "task must be"),
        (lambda r: r["residual"].update(checkpoint_sha256="bad"), "SHA-256 checkpoint hash"),
        (lambda r: r["vla"].update(checkpoint="wrong.pth"), "checkpoint and hash must be null"),
    ],
)
def test_all_integrity_gates_fail_closed(mutation, message: str) -> None:
    runs = _runs()
    mutation(runs)
    with pytest.raises(ValueError, match=message):
        comparator.compare(runs)


def test_nonfinite_nested_diagnostic_fails() -> None:
    runs = _runs()
    runs["residual"]["trials"][5]["episode_reward"] = math.nan
    with pytest.raises(ValueError, match="Non-finite number"):
        comparator.compare(runs)


def test_initial_state_schema_and_shapes_must_match() -> None:
    runs = _runs()
    runs["residual"]["trials"][3]["initial_state"]["joint_pos"].append(0.0)
    with pytest.raises(ValueError, match="shape differs"):
        comparator.compare(runs)
    runs = _runs()
    runs["residual"]["trials"][3]["initial_state"]["extra"] = [0.0]
    with pytest.raises(ValueError, match="fields differ"):
        comparator.compare(runs)


def test_episode_horizon_must_match_for_each_scenario() -> None:
    runs = _runs()
    runs["teacher"]["trials"][19]["steps"] = 100
    with pytest.raises(ValueError, match="Condition 19: episode horizon differs"):
        comparator.compare(runs)


def test_comparator_cannot_weaken_physical_state_tolerance() -> None:
    with pytest.raises(ValueError, match="state_atol must be within"):
        comparator.compare(_runs(), state_atol=1e-3)
