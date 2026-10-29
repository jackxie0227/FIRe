#!/usr/bin/env python3
"""Validate paired initial states and compare two binary success results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo", type=Path, required=True)
    parser.add_argument("--gr00t", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-atol", type=float, default=1e-5)
    parser.add_argument(
        "--drop-state-mismatches",
        action="store_true",
        help="Exclude non-matching initial states instead of failing; report exclusions explicitly.",
    )
    return parser.parse_args()


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield float(value)


def state_max_abs_difference(left: dict, right: dict) -> tuple[float, list[str]]:
    fields = sorted(set(left) | set(right))
    missing = [field for field in fields if field not in left or field not in right]
    maximum = 0.0
    for field in fields:
        if field in missing:
            continue
        lhs = list(flatten(left[field]))
        rhs = list(flatten(right[field]))
        if len(lhs) != len(rhs):
            missing.append(field)
            continue
        if lhs:
            maximum = max(maximum, max(abs(a - b) for a, b in zip(lhs, rhs)))
    return maximum, missing


def wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [center - margin, center + margin]


def exact_mcnemar(ppo_only: int, gr00t_only: int) -> float:
    discordant = ppo_only + gr00t_only
    if discordant == 0:
        return 1.0
    lower = min(ppo_only, gr00t_only)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def main() -> None:
    args = parse_args()
    with args.ppo.open(encoding="utf-8") as file:
        ppo = json.load(file)
    with args.gr00t.open(encoding="utf-8") as file:
        gr00t = json.load(file)

    ppo_trials = ppo["trials"]
    gr00t_trials = gr00t["trials"]
    if len(ppo_trials) != len(gr00t_trials):
        raise ValueError(f"Trial count differs: PPO={len(ppo_trials)}, GR00T={len(gr00t_trials)}")

    both_success = ppo_only = gr00t_only = both_fail = 0
    max_state_difference = 0.0
    max_kept_state_difference = 0.0
    mismatched_fields = set()
    excluded_seeds = []
    for ppo_trial, gr00t_trial in zip(ppo_trials, gr00t_trials):
        if ppo_trial["condition_seed"] != gr00t_trial["condition_seed"]:
            raise ValueError(
                f"Condition seed differs at index {ppo_trial['condition_index']}: "
                f"{ppo_trial['condition_seed']} != {gr00t_trial['condition_seed']}"
            )
        difference, missing = state_max_abs_difference(
            ppo_trial["initial_state"], gr00t_trial["initial_state"]
        )
        max_state_difference = max(max_state_difference, difference)
        mismatched_fields.update(missing)

        if difference > args.state_atol:
            excluded_seeds.append(ppo_trial["condition_seed"])
            if args.drop_state_mismatches:
                continue
        else:
            max_kept_state_difference = max(max_kept_state_difference, difference)

        pair = (ppo_trial["success"], gr00t_trial["success"])
        if pair == (1, 1):
            both_success += 1
        elif pair == (1, 0):
            ppo_only += 1
        elif pair == (0, 1):
            gr00t_only += 1
        else:
            both_fail += 1

    if mismatched_fields:
        raise ValueError(f"Initial-state fields differ or have incompatible shapes: {sorted(mismatched_fields)}")
    if excluded_seeds and not args.drop_state_mismatches:
        raise ValueError(
            f"Initial conditions are not paired: max abs difference {max_state_difference} "
            f"> tolerance {args.state_atol}"
        )

    trials = len(ppo_trials) - len(excluded_seeds)
    if trials == 0:
        raise ValueError("No trials remain after excluding initial-state mismatches")
    ppo_successes = both_success + ppo_only
    gr00t_successes = both_success + gr00t_only
    all_ppo_successes = sum(trial["success"] for trial in ppo_trials)
    all_gr00t_successes = sum(trial["success"] for trial in gr00t_trials)
    result = {
        "schema_version": 1,
        "paired_trials": trials,
        "source_trials": len(ppo_trials),
        "state_atol": args.state_atol,
        "max_initial_state_abs_difference": max_state_difference,
        "max_kept_initial_state_abs_difference": max_kept_state_difference,
        "all_initial_conditions_match": not excluded_seeds,
        "excluded_state_mismatch_count": len(excluded_seeds),
        "excluded_condition_seeds": excluded_seeds,
        "ppo": {
            "success_count": ppo_successes,
            "success_rate": ppo_successes / trials,
            "wilson_95": wilson(ppo_successes, trials),
        },
        "gr00t": {
            "success_count": gr00t_successes,
            "success_rate": gr00t_successes / trials,
            "wilson_95": wilson(gr00t_successes, trials),
        },
        "success_rate_difference_gr00t_minus_ppo": (gr00t_successes - ppo_successes) / trials,
        "paired_table": {
            "both_success": both_success,
            "ppo_only": ppo_only,
            "gr00t_only": gr00t_only,
            "both_fail": both_fail,
        },
        "mcnemar_exact_p": exact_mcnemar(ppo_only, gr00t_only),
        "all_trials_descriptive_only": {
            "warning": "Includes trials whose settled physical initial states did not match.",
            "ppo_success_count": all_ppo_successes,
            "ppo_success_rate": all_ppo_successes / len(ppo_trials),
            "gr00t_success_count": all_gr00t_successes,
            "gr00t_success_rate": all_gr00t_successes / len(gr00t_trials),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"[INFO] Comparison saved to: {output}")


if __name__ == "__main__":
    main()
