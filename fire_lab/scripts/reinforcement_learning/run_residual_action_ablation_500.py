#!/usr/bin/env python3
"""Resume-safe, fresh-process residual action ablation on the original 500 seeds."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_three_policy_paired_500 import (
    CRITERION, OFFICIAL_TASK, STATE_ATOL, check_no_other_simulator,
    check_service, load_json, model_and_checkpoint_spec, sha256_file,
    state_difference, atomic_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vla-host", default="127.0.0.1")
    parser.add_argument("--vla-port", type=int, default=5555)
    parser.add_argument("--condition-seed-start", type=int, default=100000)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.count <= 500:
        parser.error("--count must be 1..500")
    check_no_other_simulator()
    spec = model_and_checkpoint_spec(args)
    check_service(args.vla_host, args.vla_port)
    reference = load_json(args.reference.resolve())
    if reference.get("paired_trials") != 500 or not reference.get("strictly_paired"):
        raise ValueError("Reference must be the completed strict 500-condition comparison")
    if reference.get("condition_seed_start") != args.condition_seed_start:
        raise ValueError("Reference seed start differs")
    if reference["policies"]["residual"]["checkpoint_sha256"] != spec["residual_sha256"]:
        raise ValueError("Reference used a different residual checkpoint")
    reference_trials = reference["trials"]
    if len(reference_trials) != 500:
        raise ValueError("Reference has fewer than 500 trials")
    reference_residual = load_json(args.reference.resolve().with_name("residual_500.json"))
    if (reference_residual.get("episodes") != 500
            or reference_residual.get("checkpoint_sha256") != spec["residual_sha256"]):
        raise ValueError("Reference residual_500.json is incomplete or uses another checkpoint")
    output = args.output_dir.resolve()
    manifest = {"schema_version": 1, "reference": str(args.reference.resolve()),
                "reference_sha256": sha256_file(args.reference.resolve()), "spec": spec,
                "modes": ["direct", "post_sum_clip"]}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if load_json(manifest_path) != manifest:
            raise ValueError("Existing output manifest differs")
    else:
        # The one-command launcher starts GR00T before this preflight and writes
        # its service log here. That log is safe to keep on the first run.
        if output.exists():
            unexpected = [path.name for path in output.iterdir() if path.name != "gr00t_server.log"]
            if unexpected:
                raise ValueError(f"Nonempty output has no manifest; unexpected entries: {unexpected}")
        atomic_json(manifest_path, manifest)
    for index in range(args.count):
        seed = args.condition_seed_start + index
        original = reference_trials[index]
        if original.get("condition_seed") != seed:
            raise ValueError(f"Reference seed mismatch at {index}")
        reference_state = reference_residual["trials"][index]["initial_state"]
        for mode in ("direct", "post_sum_clip"):
            result_path = output / "trials" / str(seed) / f"{mode}.json"
            if not result_path.exists():
                result_path.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", spec["evaluator"], "--policy", "residual",
                           "--task", OFFICIAL_TASK, "--checkpoint", spec["residual_checkpoint"],
                           "--episodes", "1", "--condition-seed-start", str(seed),
                           "--eval-num-envs", "1", "--success-criterion", CRITERION,
                           "--output", str(result_path), "--headless", "--enable_cameras",
                           f"env.vla_host={args.vla_host}", f"env.vla_port={args.vla_port}"]
                if mode == "post_sum_clip":
                    command.append("--clip-combined-action")
                log = output / "logs" / f"{seed}_{mode}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                print(f"[START] {seed} {mode}", flush=True)
                with log.open("w") as stream:
                    stream.write("COMMAND: " + json.dumps(command) + "\n")
                    stream.flush()
                    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                if completed.returncode:
                    raise RuntimeError(f"Evaluator failed: {log}")
            payload = load_json(result_path)
            if (payload.get("complete") is not True or payload.get("episodes") != 1
                    or payload.get("checkpoint_sha256") != spec["residual_sha256"]
                    or payload.get("condition_seed_start") != seed
                    or payload.get("success_criterion") != CRITERION):
                raise ValueError(f"Invalid result: {result_path}")
            trial = payload["trials"][0]
            if trial.get("combined_action_clip_mode") != ("none" if mode == "direct" else "post_sum"):
                raise ValueError(f"Incorrect action mode: {result_path}")
            if not trial.get("all_outputs_finite") or trial.get("vla_request_error_count") != 0:
                raise ValueError(f"Invalid policy output: {result_path}")
            diff = state_difference(reference_state, trial["initial_state"])
            if diff > STATE_ATOL:
                raise ValueError(f"Initial state mismatch seed={seed} mode={mode}: {diff}")
        print(f"[MATCHED] {index + 1}/{args.count} seed={seed}", flush=True)
    results = {}
    for mode in ("direct", "post_sum_clip"):
        trials = [load_json(output / "trials" / str(args.condition_seed_start + i) / f"{mode}.json")["trials"][0]
                  for i in range(args.count)]
        results[mode] = {"success_count": sum(t["success"] for t in trials),
                         "success_rate": sum(t["success"] for t in trials) / len(trials),
                         "mean_out_of_range_fraction": sum(t["combined_action_out_of_range_fraction"] for t in trials) / len(trials)}
    results["count"] = args.count
    results["success_rate_difference_clip_minus_direct"] = results["post_sum_clip"]["success_rate"] - results["direct"]["success_rate"]
    atomic_json(output / "summary.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
