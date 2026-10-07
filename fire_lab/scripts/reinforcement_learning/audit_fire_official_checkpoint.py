#!/usr/bin/env python3
"""Audit FIRe epoch/frame and optimizer/scheduler/normalizer/RNN resume state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


STATE_KEY = "fire_success_resume_state"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_summary(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = state.get("model", {})
    model_keys = list(model.keys())
    optimizer = state.get("optimizer", {})
    fire_state = state.get(STATE_KEY, {})
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "epoch": int(state.get("epoch", -1)),
        "frame": int(state.get("frame", -1)),
        "has_optimizer": bool(optimizer and "param_groups" in optimizer),
        "optimizer_state_entries": len(optimizer.get("state", {})),
        "has_scheduler_state": "scheduler" in fire_state and "last_lr" in fire_state,
        "last_lr": fire_state.get("last_lr"),
        "has_success_metadata": bool(fire_state),
        "completed_episodes": fire_state.get("completed_episodes"),
        "rolling_window_count": len(fire_state.get("rolling_successes", [])),
        "best_success_rate": fire_state.get("best_success_rate"),
        "has_input_normalizer": any("running_mean_std" in key for key in model_keys),
        "has_value_normalizer": any("value_mean_std" in key for key in model_keys),
        "has_rnn_weights": any("rnn" in key.lower() or "lstm" in key.lower() for key in model_keys),
        "has_central_value_state": "assymetric_vf_nets" in state,
        "top_level_keys": sorted(state.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--best-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = _checkpoint_summary(args.pilot_checkpoint)
    resumed = _checkpoint_summary(args.resume_checkpoint)
    best = json.loads(args.best_json.read_text())
    best_path = Path(best["checkpoint"])
    checks = {
        "pilot_checkpoint_epoch_10": pilot["epoch"] == 10,
        "resume_checkpoint_epoch_11": resumed["epoch"] == 11,
        "frame_continuity_8192": resumed["frame"] - pilot["frame"] == 8192,
        "optimizer_present_both": pilot["has_optimizer"] and resumed["has_optimizer"],
        "scheduler_present_both": pilot["has_scheduler_state"] and resumed["has_scheduler_state"],
        "normalizers_present_both": (
            pilot["has_input_normalizer"]
            and resumed["has_input_normalizer"]
            and pilot["has_value_normalizer"]
            and resumed["has_value_normalizer"]
        ),
        "rnn_present_both": pilot["has_rnn_weights"] and resumed["has_rnn_weights"],
        "success_metadata_continuous": (
            pilot["has_success_metadata"]
            and resumed["has_success_metadata"]
            and resumed["completed_episodes"] >= pilot["completed_episodes"]
        ),
        "best_checkpoint_sha_matches_json": (
            best_path.is_file() and best.get("sha256") == _sha256(best_path)
        ),
    }
    payload = {
        "schema_version": 1,
        "pilot_epoch10": pilot,
        "resumed_epoch11": resumed,
        "best_success_metadata": best,
        "checks": checks,
        "pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
