#!/usr/bin/env python3
"""Strictly compare teacher, VLA, and VLA+residual on the same 500 conditions.

Unlike the older two-policy comparison, this tool never drops mismatched trials.
An output is written only when all 500 conditions pass the pairing and data gates.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


POLICIES = ("teacher", "vla", "residual")
POLICY_LABELS = {"teacher": "ppo", "vla": "gr00t", "residual": "residual"}
CONDITION_SEED_START = 100000
TASK_IDS = {
    "teacher": "FireLab-BaseLine-Forge-PegInsert-Direct-v0",
    "vla": "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-SampleEq4-Direct-v1",
    "residual": "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-SampleEq4-Direct-v1",
}
PAIR_NAMES = (("teacher", "vla"), ("teacher", "residual"), ("vla", "residual"))
EXPECTED_CRITERION = "fire_official_peg_insert_xy_le_0.005_z_le_0.025_ever"
REQUIRED_STATE_FIELDS = frozenset(
    (
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
)


def _finite_tree(value: Any, *, path: str) -> None:
    """Reject JSON NaN/Infinity, including nested trial diagnostics."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite number at {path}: {value}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, path=f"{path}.{key}")
        return
    raise ValueError(f"Unsupported value at {path}: {type(value).__name__}")


def _state_difference(left: Any, right: Any, *, path: str) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Initial-state fields differ at {path}: {sorted(left.keys() ^ right.keys())}")
        return max(
            (_state_difference(left[key], right[key], path=f"{path}.{key}") for key in left),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"Initial-state shape differs at {path}: {len(left)} != {len(right)}")
        return max(
            (_state_difference(a, b, path=f"{path}[{i}]") for i, (a, b) in enumerate(zip(left, right))),
            default=0.0,
        )
    if isinstance(left, bool) or isinstance(right, bool):
        raise ValueError(f"Initial-state value must be numeric at {path}")
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"Non-finite initial-state value at {path}")
        return abs(left - right)
    raise ValueError(f"Initial-state types differ at {path}: {type(left).__name__}, {type(right).__name__}")


def _wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [center - margin, center + margin]


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _validate_run(label: str, result: dict[str, Any], episodes: int, criterion: str) -> list[dict]:
    _finite_tree(result, path=label)
    if result.get("policy") != POLICY_LABELS[label]:
        raise ValueError(f"{label}: policy label must be {POLICY_LABELS[label]!r}")
    if result.get("task") != TASK_IDS[label]:
        raise ValueError(f"{label}: task must be {TASK_IDS[label]!r}")
    checkpoint = result.get("checkpoint")
    digest = result.get("checkpoint_sha256")
    if label == "vla":
        if checkpoint is not None or digest is not None:
            raise ValueError("vla: PPO checkpoint and hash must be null")
    else:
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"{label}: PPO checkpoint path is required")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"{label}: a SHA-256 checkpoint hash is required")
    trials = result.get("trials")
    if not isinstance(trials, list) or len(trials) != episodes:
        raise ValueError(f"{label}: expected exactly {episodes} trials")
    if result.get("complete") is not True:
        raise ValueError(f"{label}: complete must be true")
    for key in ("episodes", "requested_episodes"):
        if result.get(key) != episodes:
            raise ValueError(f"{label}: {key} must equal {episodes}")
    if result.get("success_criterion") != criterion:
        raise ValueError(f"{label}: unexpected success_criterion {result.get('success_criterion')!r}")
    if result.get("condition_seed_start") != CONDITION_SEED_START:
        raise ValueError(f"{label}: condition_seed_start must be {CONDITION_SEED_START}")
    if result.get("all_outputs_finite") is not True:
        raise ValueError(f"{label}: all_outputs_finite must be true")
    if result.get("vla_request_error_count") != 0:
        raise ValueError(f"{label}: vla_request_error_count must be zero")

    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(f"{label}[{index}]: trial must be an object")
        if trial.get("condition_index") != index:
            raise ValueError(f"{label}[{index}]: non-contiguous condition_index")
        seed = trial.get("condition_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed != CONDITION_SEED_START + index:
            raise ValueError(
                f"{label}[{index}]: condition_seed must equal {CONDITION_SEED_START + index}"
            )
        if trial.get("success") not in (0, 1) or isinstance(trial.get("success"), bool):
            raise ValueError(f"{label}[{index}]: success must be 0 or 1")
        steps = trial.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(f"{label}[{index}]: steps must be a positive integer")
        if trial.get("all_outputs_finite") is not True:
            raise ValueError(f"{label}[{index}]: non-finite rollout was reported")
        if trial.get("vla_request_error_count") != 0:
            raise ValueError(f"{label}[{index}]: VLA request error was reported")
        state = trial.get("initial_state")
        if not isinstance(state, dict) or not REQUIRED_STATE_FIELDS.issubset(state):
            missing = sorted(REQUIRED_STATE_FIELDS - state.keys()) if isinstance(state, dict) else "all"
            raise ValueError(f"{label}[{index}]: missing initial-state fields: {missing}")
        _state_difference(state, state, path=f"{label}[{index}].initial_state")

    successes = sum(trial["success"] for trial in trials)
    if result.get("success_count") != successes:
        raise ValueError(f"{label}: success_count does not match trials")
    if not math.isclose(result.get("success_rate", float("nan")), successes / episodes, abs_tol=1e-12):
        raise ValueError(f"{label}: success_rate does not match trials")
    return trials


def compare(
    results: dict[str, dict[str, Any]],
    *,
    episodes: int = 500,
    state_atol: float = 1.0e-5,
) -> dict[str, Any]:
    if episodes <= 0 or not math.isfinite(state_atol) or not 0 <= state_atol <= 1.0e-5:
        raise ValueError("episodes must be positive and state_atol must be within [0, 1e-5]")
    if set(results) != set(POLICIES):
        raise ValueError(f"Expected exactly these policies: {POLICIES}")
    trials_by_policy = {
        name: _validate_run(name, results[name], episodes, EXPECTED_CRITERION) for name in POLICIES
    }
    paired_trials: list[dict[str, Any]] = []
    maximum_difference = 0.0
    for index in range(episodes):
        trials = {name: trials_by_policy[name][index] for name in POLICIES}
        seeds = {trial["condition_seed"] for trial in trials.values()}
        if len(seeds) != 1:
            raise ValueError(f"Condition {index}: seed mismatch: {seeds}")
        steps = {trial["steps"] for trial in trials.values()}
        if len(steps) != 1:
            raise ValueError(f"Condition {index}: episode horizon differs: {steps}")
        differences = {
            f"{left}_vs_{right}": _state_difference(
                trials[left]["initial_state"],
                trials[right]["initial_state"],
                path=f"condition[{index}].{left}_vs_{right}",
            )
            for left, right in PAIR_NAMES
        }
        difference = max(differences.values())
        maximum_difference = max(maximum_difference, difference)
        if difference > state_atol:
            raise ValueError(
                f"Condition {index} seed {next(iter(seeds))}: initial states differ by "
                f"{difference:.9g} > {state_atol:.9g}; no trials were dropped"
            )
        paired_trials.append(
            {
                "condition_index": index,
                "condition_seed": next(iter(seeds)),
                "steps": next(iter(steps)),
                "success": {name: trials[name]["success"] for name in POLICIES},
                "initial_state_max_abs_difference": difference,
            }
        )

    summaries = {}
    for name in POLICIES:
        successes = sum(trial["success"][name] for trial in paired_trials)
        summaries[name] = {
            "success_count": successes,
            "success_rate": successes / episodes,
            "success_rate_wilson_95": _wilson(successes, episodes),
            "task": results[name].get("task"),
            "checkpoint": results[name].get("checkpoint"),
            "checkpoint_sha256": results[name].get("checkpoint_sha256"),
        }
    pairs = {}
    for left, right in PAIR_NAMES:
        both = left_only = right_only = neither = 0
        for trial in paired_trials:
            a, b = trial["success"][left], trial["success"][right]
            if a and b:
                both += 1
            elif a:
                left_only += 1
            elif b:
                right_only += 1
            else:
                neither += 1
        pairs[f"{left}_vs_{right}"] = {
            "both_success": both,
            "left_only": left_only,
            "right_only": right_only,
            "both_fail": neither,
            "success_rate_difference_right_minus_left": (right_only - left_only) / episodes,
            "mcnemar_exact_p": _mcnemar_exact(left_only, right_only),
        }
    return {
        "schema_version": 1,
        "strictly_paired": True,
        "paired_trials": episodes,
        "condition_seed_start": CONDITION_SEED_START,
        "success_criterion": EXPECTED_CRITERION,
        "state_atol": state_atol,
        "max_initial_state_abs_difference": maximum_difference,
        "policies": summaries,
        "pairwise": pairs,
        "trials": paired_trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in POLICIES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-atol", type=float, default=1.0e-5)
    args = parser.parse_args()
    results = {}
    for name in POLICIES:
        with getattr(args, name).expanduser().open(encoding="utf-8") as file:
            results[name] = json.load(file)
    comparison = compare(results, state_atol=args.state_atol)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, allow_nan=False)
    temporary.replace(output)
    print(json.dumps({key: value for key, value in comparison.items() if key != "trials"}, indent=2))
    print(f"[INFO] Strict 500-condition comparison saved to: {output}")


if __name__ == "__main__":
    main()
