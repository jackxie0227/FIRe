#!/usr/bin/env python3
"""Summarize FIRe smoke TensorBoard files and evaluate pilot gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


EPISODE_TAGS = [
    "Episode/success_rate",
    "Episode/success_time",
    "Episode/contact_force_mean",
    "Episode/contact_force_max",
    "Episode/contact_force_p95",
    "Episode/over_force_fraction",
    "Episode/xy_error",
    "Episode/z_displacement",
]
FIRE_TAGS = [
    "Reward/r_s",
    "Reward/r_penalty",
    "Reward/r_shaping",
    "Action/vla_abs_mean",
    "Action/residual_abs_mean",
    "Action/combined_abs_mean",
    "Action/residual_clip_fraction",
    "Action/combined_out_of_range_fraction",
]
PPO_TAGS = [
    "losses/a_loss",
    "losses/c_loss",
    "losses/entropy",
    "losses/bounds_loss",
    "info/kl",
    "info/last_lr",
]

EXPECTED_EPOCH_EVENTS = {
    "chain": 1,
    "batch": 1,
    "pilot": 20,
}


def _series(accumulator: EventAccumulator, tag: str) -> list[dict]:
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [
        {"step": int(event.step), "wall_time": event.wall_time, "value": float(event.value)}
        for event in accumulator.Scalars(tag)
    ]


def _summary(events: list[dict]) -> dict:
    values = [event["value"] for event in events]
    return {
        "count": len(values),
        "finite": all(math.isfinite(value) for value in values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "events": events,
    }


def _max_consecutive(events: list[dict], threshold: float) -> int:
    longest = current = 0
    for event in events:
        if event["value"] > threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def audit_run(label: str, run_dir: Path) -> dict:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = sorted(accumulator.Tags().get("scalars", []))
    requested = EPISODE_TAGS + FIRE_TAGS + PPO_TAGS + [
        "Success/rolling_100",
        "performance/step_inference_rl_update_fps",
    ]
    series = {tag: _series(accumulator, tag) for tag in requested}
    result = {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "available_tags": available,
        "metrics": {tag: _summary(events) for tag, events in series.items()},
        "missing_episode_tags": [tag for tag in EPISODE_TAGS if not series[tag]],
        "missing_fire_tags": [tag for tag in FIRE_TAGS if not series[tag]],
        "missing_ppo_tags": [tag for tag in PPO_TAGS if not series[tag]],
    }
    all_events = [event for events in series.values() for event in events]
    result["all_requested_metrics_finite"] = all(
        math.isfinite(event["value"]) for event in all_events
    )

    expected_events = EXPECTED_EPOCH_EVENTS.get(label)
    ppo_epoch_events = len(series["info/kl"])
    result["execution_gates"] = {
        "expected_epoch_events": expected_events,
        "observed_epoch_events": ppo_epoch_events,
        "epoch_count_reached": expected_events is None or ppo_epoch_events >= expected_events,
        "ppo_tags_present": not result["missing_ppo_tags"],
        "all_available_metrics_finite": result["all_requested_metrics_finite"],
    }

    if label in {"batch", "pilot"}:
        successes = series["Episode/success_rate"]
        residual_clip = series["Action/residual_clip_fraction"]
        result["execution_gates"].update(
            {
                "complete_episode_tags_present": not result["missing_episode_tags"],
                "reward_action_tags_present": not result["missing_fire_tags"],
                "at_least_one_success_episode": any(event["value"] > 0.0 for event in successes),
                "residual_not_fully_clipped_every_epoch": bool(residual_clip)
                and any(event["value"] < 1.0 for event in residual_clip),
            }
        )

    if label == "pilot":
        successes = series["Episode/success_rate"]
        rolling = series["Success/rolling_100"]
        kl = series["info/kl"]
        kl_streak = _max_consecutive(kl, 0.1)
        result["pilot_gates"] = {
            "success_by_epoch_5": any(
                event["step"] <= 5 and event["value"] > 0.0 for event in successes
            ),
            "rolling_100_success_at_epoch_20_ge_5pct": bool(
                rolling and rolling[-1]["step"] >= 20 and rolling[-1]["value"] >= 0.05
            ),
            "max_consecutive_epochs_kl_gt_0_1": kl_streak,
            "kl_gate": kl_streak < 5,
            "all_metrics_finite": result["all_requested_metrics_finite"],
            "complete_episode_tags_present": not result["missing_episode_tags"],
            "reward_action_tags_present": not result["missing_fire_tags"],
            "ppo_tags_present": not result["missing_ppo_tags"],
        }
    return result


def _read_exit_codes(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        try:
            values[key.strip()] = int(raw_value.strip())
        except ValueError:
            continue
    return values


def _read_vla_result(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "complete": bool(payload.get("complete", False)),
        "episodes": int(payload.get("episodes", 0)),
        "requested_episodes": int(payload.get("requested_episodes", 0)),
        "success_count": int(payload.get("success_count", 0)),
        "success_rate": float(payload.get("success_rate", 0.0)),
        "all_outputs_finite": bool(payload.get("all_outputs_finite", False)),
        "zero_residual_exact": bool(payload.get("all_zero_residual_compositions_exact", False)),
        "request_error_count": int(payload.get("vla_request_error_count", 0)),
        "minimal_smoke_evidence_accepted": (
            int(payload.get("episodes", 0)) >= 388
            and int(payload.get("success_count", 0)) >= 67
            and bool(payload.get("all_outputs_finite", False))
            and bool(payload.get("all_zero_residual_compositions_exact", False))
            and int(payload.get("vla_request_error_count", 0)) == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run label and RL-Games run directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--vla-result", type=Path)
    args = parser.parse_args()

    results = []
    for item in args.run:
        label, path = item.split("=", 1)
        results.append(audit_run(label, Path(path)))
    exit_codes = _read_exit_codes(args.status_file)
    required_exit_codes = {
        "CHAIN_SMOKE_RC": 0,
        "BATCH_SMOKE_RC": 0,
        "PILOT_20EPOCH_RC": 0,
    }
    exit_code_gate = all(exit_codes.get(key) == expected for key, expected in required_exit_codes.items())
    run_gates = {}
    for result in results:
        if result["label"] not in EXPECTED_EPOCH_EVENTS:
            continue
        checks = [
            value for value in result["execution_gates"].values() if type(value) is bool
        ]
        checks.extend(
            value for value in result.get("pilot_gates", {}).values() if type(value) is bool
        )
        run_gates[result["label"]] = all(checks)
    expected_labels_present = set(EXPECTED_EPOCH_EVENTS).issubset(run_gates)
    vla_result = _read_vla_result(args.vla_result)
    vla_gate = bool(vla_result and vla_result["minimal_smoke_evidence_accepted"])
    conditional_pass = (
        exit_code_gate
        and expected_labels_present
        and all(run_gates.values())
        and vla_gate
    )
    payload = {
        "schema_version": 2,
        "runs": results,
        "exit_codes": exit_codes,
        "required_exit_codes": required_exit_codes,
        "vla_only_partial_evidence": vla_result,
        "gates": {
            "training_exit_codes_zero": exit_code_gate,
            "expected_run_labels_present": expected_labels_present,
            "run_gates": run_gates,
            "vla_partial_evidence_accepted": vla_gate,
        },
        "conditional_pass": conditional_pass,
        "readiness": (
            "CONDITIONAL_GO_MINIMAL_SMOKE" if conditional_pass else "NO_GO"
        ),
        "limitations": [
            "VLA-only regression was accepted at 388/500 episodes and is not a complete regression.",
            "Checkpoint optimizer/scheduler restoration was not exercised by a resumed training epoch.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
