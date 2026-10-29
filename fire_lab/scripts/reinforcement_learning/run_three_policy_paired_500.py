#!/usr/bin/env python3
"""Run a strict three-policy comparison with fresh Isaac processes per condition.

The existing multi-episode evaluator has shown history-dependent reset drift.
Therefore every (condition seed, policy) is evaluated by a new simulator process.
Only complete triples whose recorded initial states pass the benchmark's
field-specific pairing rules enter the three aggregate results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path


EPISODES = 500
DEFAULT_SEED_START = 100000
STATE_ATOL = 1.0e-5
GEAR_STATE_ATOL = 5.0e-4
GEAR_STATE_TOLERANCES_V1 = {
    # Reset targets and randomized controller parameters must be identical.
    "fixed_pos": 1.0e-6,
    "fixed_quat": 1.0e-6,
    "init_fixed_pos_obs_noise": 1.0e-6,
    "task_prop_gains": 1.0e-6,
    "ema_factor": 1.0e-6,
    "pos_threshold": 1.0e-6,
    "rot_threshold": 1.0e-6,
    "contact_penalty_thresholds": 1.0e-6,
    "dead_zone_thresholds": 1.0e-6,
    # Different camera/render schedules cause small solver drift after reset.
    "held_pos": 1.0e-4,
    "fingertip_midpoint_pos": 1.0e-4,
    "held_quat": 5.0e-4,
    "fingertip_midpoint_quat": 5.0e-4,
    "joint_pos": 2.0e-4,
    "fingertip_midpoint_linvel": 5.0e-3,
    "fingertip_midpoint_angvel": 5.0e-3,
    "joint_vel": 5.0e-3,
}
GEAR_STATE_TOLERANCES = {
    **GEAR_STATE_TOLERANCES_V1,
    "held_pos": 5.0e-4,
    "fingertip_midpoint_pos": 5.0e-4,
    "held_quat": 5.0e-3,
    "fingertip_midpoint_quat": 5.0e-3,
    "joint_pos": 5.0e-3,
    "fingertip_midpoint_linvel": 5.0e-2,
    "fingertip_midpoint_angvel": 5.0e-2,
    "joint_vel": 5.0e-2,
}
CRITERION = "fire_official_peg_insert_xy_le_0.005_z_le_0.025_ever"
TEACHER_TASK = "FireLab-BaseLine-Forge-PegInsert-Direct-v0"
OFFICIAL_TASK = "FireLab-VLA-Gr00t-Forge-PegInsert-FIReOfficial-SampleEq4-Direct-v1"
EVALUATOR = Path(__file__).with_name("evaluate_paired_policy.py")
POLICIES = ("teacher", "vla", "residual")
EVALUATOR_POLICIES = {"teacher": "ppo", "vla": "gr00t", "residual": "residual"}
TASKS = {"teacher": TEACHER_TASK, "vla": OFFICIAL_TASK, "residual": OFFICIAL_TASK}
GEAR_TASKS = {
    "teacher": "FireLab-BaseLine-Forge-GearMesh-Direct-v0",
    "vla": "FireLab-VLA-Gr00t-Forge-GearMesh-FIRe6D-Direct-v1",
    "residual": "FireLab-VLA-Gr00t-Forge-GearMesh-FIRe6D-Direct-v1",
}


def evaluation_profile(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    if getattr(args, "benchmark", "peg_insert") == "gear_mesh":
        return GEAR_TASKS, "native"
    return TASKS, CRITERION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("peg_insert", "gear_mesh"), default="peg_insert")
    parser.add_argument("--target-episodes", type=int, default=EPISODES)
    parser.add_argument("--vla-host", default="127.0.0.1")
    parser.add_argument("--vla-port", type=int, default=5555)
    parser.add_argument("--condition-seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--count", type=int, help="Run the first N conditions; defaults to --target-episodes.")
    parser.add_argument("--resume", action="store_true", help="Reuse checked single-condition results from this output directory.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def model_and_checkpoint_spec(args: argparse.Namespace) -> dict:
    tasks, criterion = evaluation_profile(args)
    target_episodes = getattr(args, "target_episodes", EPISODES)
    teacher = args.teacher_checkpoint.expanduser().resolve()
    residual = args.residual_checkpoint.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    if teacher == residual:
        raise ValueError("Teacher and residual checkpoints must be different files")
    for label, path in (("teacher", teacher), ("residual", residual)):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {label} checkpoint: {path}")
    config = model_dir / "config.json"
    weights = model_dir / "model.safetensors"
    if not config.is_file() or not weights.is_file() or weights.stat().st_size == 0:
        raise FileNotFoundError(f"Missing GR00T model config or weights in {model_dir}")
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"Missing evaluator: {EVALUATOR}")
    spec = {
        "schema_version": 1,
        "episodes": target_episodes,
        "condition_seed_start": args.condition_seed_start,
        "condition_seed_end": args.condition_seed_start + target_episodes - 1,
        "success_criterion": criterion,
        "state_atol": GEAR_STATE_ATOL if getattr(args, "benchmark", "peg_insert") == "gear_mesh" else STATE_ATOL,
        "tasks": tasks,
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": sha256_file(teacher),
        "residual_checkpoint": str(residual),
        "residual_sha256": sha256_file(residual),
        "model_dir": str(model_dir),
        "model_config_sha256": sha256_file(config),
        "model_weights_sha256": sha256_file(weights),
        "vla_host": args.vla_host,
        "vla_port": args.vla_port,
        "evaluator": str(EVALUATOR.resolve()),
        "evaluator_sha256": sha256_file(EVALUATOR),
    }
    if getattr(args, "benchmark", "peg_insert") == "gear_mesh":
        spec["state_tolerances"] = GEAR_STATE_TOLERANCES
    return spec


def check_service(host: str, port: int) -> None:
    if not 0 < port < 65536:
        raise ValueError("--vla-port must be between 1 and 65535")
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        raise RuntimeError(f"GR00T service not reachable at {host}:{port}: {exc}") from exc


def check_no_other_simulator() -> None:
    """Refuse a pre-existing evaluator/trainer/Kit process, but allow GR00T."""
    if not Path("/proc").is_dir():
        return
    current_pid = os.getpid()
    conflicts = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            comm = (entry / "comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (
            "evaluate_paired_policy.py" in command
            or "reinforcement_learning/rl_games/train.py" in command
            or comm.startswith("kit")
            or "isaacsim" in command.lower()
        ):
            conflicts.append(f"pid={entry.name} {command[:220]}")
    if conflicts:
        raise RuntimeError("Other simulator/evaluation/training processes are running:\n" + "\n".join(conflicts))


def ensure_manifest(output_dir: Path, spec: dict, *, write: bool, resume: bool) -> None:
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if not resume and write:
            raise ValueError(f"Output run already exists; pass --resume to reuse it: {output_dir}")
        previous = load_json(manifest)
        legacy = {key: value for key, value in spec.items() if key != "state_tolerances"}
        if (
            write and resume and "state_tolerances" in spec and previous == legacy
            and previous.get("tasks") == GEAR_TASKS
        ):
            backup = output_dir / "manifest_before_gear_pairing_update.json"
            if not backup.exists():
                atomic_json(backup, previous)
            atomic_json(manifest, spec)
            previous = spec
        previous_tolerances = {**spec, "state_tolerances": GEAR_STATE_TOLERANCES_V1}
        if (
            write and resume and "state_tolerances" in spec and previous == previous_tolerances
            and previous.get("tasks") == GEAR_TASKS
        ):
            backup = output_dir / "manifest_before_gear_pairing_update_v2.json"
            if not backup.exists():
                atomic_json(backup, previous)
            atomic_json(manifest, spec)
            previous = spec
        if previous != spec:
            different = [key for key in sorted(set(previous) | set(spec)) if previous.get(key) != spec.get(key)]
            raise ValueError(f"Existing output manifest is incompatible: {different}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory is nonempty but has no manifest: {output_dir}")
    elif write:
        atomic_json(manifest, spec)


def trial_path(output_dir: Path, seed: int, policy: str) -> Path:
    return output_dir / "trials" / str(seed) / f"{policy}.json"


def validate_trial(payload: dict, *, policy: str, seed: int, spec: dict) -> dict:
    tasks = spec.get("tasks", TASKS)
    criterion = spec.get("success_criterion", CRITERION)
    expected_checkpoint = (
        spec["teacher_checkpoint"] if policy == "teacher" else spec["residual_checkpoint"]
        if policy == "residual" else None
    )
    expected_checkpoint_sha256 = (
        spec["teacher_sha256"] if policy == "teacher" else spec["residual_sha256"]
        if policy == "residual" else None
    )
    expected = {
        "policy": EVALUATOR_POLICIES[policy],
        "task": tasks[policy],
        "checkpoint": expected_checkpoint,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "episodes": 1,
        "requested_episodes": 1,
        "condition_seed_start": seed,
        "success_criterion": criterion,
        "complete": True,
        "all_outputs_finite": True,
        "vla_request_error_count": 0,
    }
    bad = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if bad:
        raise ValueError(f"Invalid {policy} result for seed {seed}: {bad}")
    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) != 1:
        raise ValueError(f"Expected one {policy} trial for seed {seed}")
    trial = trials[0]
    if trial.get("condition_seed") != seed or trial.get("condition_index") != 0:
        raise ValueError(f"Wrong condition identity in {policy} trial for seed {seed}")
    if type(trial.get("success")) is not int or trial["success"] not in (0, 1):
        raise ValueError(f"Invalid success value in {policy} trial for seed {seed}")
    if not isinstance(trial.get("initial_state"), dict) or not trial["initial_state"]:
        raise ValueError(f"Missing initial state in {policy} trial for seed {seed}")
    if trial.get("all_outputs_finite") is not True:
        raise ValueError(f"Non-finite policy output in {policy} trial for seed {seed}")
    if trial.get("vla_request_error_count") != 0:
        raise ValueError(f"GR00T request errors in {policy} trial for seed {seed}")
    if policy == "vla" and not trial.get("zero_residual_composition_exact", False):
        raise ValueError(f"VLA-only composition is not exact for seed {seed}")
    return trial


def _numbers(value):
    if isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite initial state")
        yield number
    else:
        raise ValueError(f"Invalid initial-state entry: {value!r}")


def state_field_differences(left: dict, right: dict) -> dict[str, float]:
    if left.keys() != right.keys():
        raise ValueError(f"Initial-state fields differ: {sorted(left.keys() ^ right.keys())}")
    differences = {}
    for field in left:
        lhs = list(_numbers(left[field]))
        rhs = list(_numbers(right[field]))
        if len(lhs) != len(rhs):
            raise ValueError(f"Initial-state shape differs for {field}")
        differences[field] = max((abs(a - b) for a, b in zip(lhs, rhs)), default=0.0)
    return differences


def state_difference(left: dict, right: dict) -> float:
    return max(state_field_differences(left, right).values(), default=0.0)


def check_triple_states(
    trials: dict[str, dict], *, seed: int, atol: float = STATE_ATOL,
    field_atol: dict[str, float] | None = None,
) -> float:
    pairs = (("teacher", "vla"), ("teacher", "residual"), ("vla", "residual"))
    field_differences = {
        f"{a}_vs_{b}": state_field_differences(trials[a]["initial_state"], trials[b]["initial_state"])
        for a, b in pairs
    }
    differences = {pair: max(fields.values(), default=0.0) for pair, fields in field_differences.items()}
    maximum = max(differences.values())
    violations = {
        f"{pair}.{field}": (difference, field_atol.get(field, atol) if field_atol else atol)
        for pair, fields in field_differences.items()
        for field, difference in fields.items()
        if difference > (field_atol.get(field, atol) if field_atol else atol)
    }
    if violations:
        raise ValueError(
            f"Seed {seed} is NOT a strict three-policy matched scene: {differences}; "
            f"violations={violations}. Stop and diagnose; do not count this group."
        )
    return maximum


def wilson(successes: int, trials: int) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [center - margin, center + margin]


def aggregate_payload(policy: str, trials: list[dict], spec: dict) -> dict:
    target_episodes = spec.get("episodes", EPISODES)
    successes = sum(int(trial["success"]) for trial in trials)
    return {
        "schema_version": 2,
        "policy": EVALUATOR_POLICIES[policy],
        "policy_label": policy,
        "task": spec.get("tasks", TASKS)[policy],
        "checkpoint": (
            spec["teacher_checkpoint"] if policy == "teacher" else spec["residual_checkpoint"]
            if policy == "residual" else None
        ),
        "checkpoint_sha256": (
            spec["teacher_sha256"] if policy == "teacher" else spec["residual_sha256"]
            if policy == "residual" else None
        ),
        "episodes": len(trials),
        "requested_episodes": target_episodes,
        "condition_seed_start": spec["condition_seed_start"],
        "success_criterion": spec.get("success_criterion", CRITERION),
        "initial_state_atol": spec.get("state_atol", STATE_ATOL),
        "initial_state_tolerances": spec.get("state_tolerances"),
        "complete": len(trials) == target_episodes,
        "success_count": successes,
        "success_rate": successes / len(trials) if trials else 0.0,
        "success_rate_wilson_95": wilson(successes, len(trials)),
        "all_initial_conditions_match": True,
        "all_outputs_finite": all(trial["all_outputs_finite"] is True for trial in trials),
        "vla_request_error_count": sum(trial["vla_request_error_count"] for trial in trials),
        "trials": trials,
    }


def save_aggregates(output_dir: Path, committed: dict[str, list[dict]], differences: list[dict], spec: dict) -> None:
    target_episodes = spec.get("episodes", EPISODES)
    for policy in POLICIES:
        atomic_json(output_dir / f"{policy}_{target_episodes}.json", aggregate_payload(policy, committed[policy], spec))
    atomic_json(
        output_dir / "pairing_audit.json",
        {
            "schema_version": 1,
            "state_atol": spec.get("state_atol", STATE_ATOL),
            "state_tolerances": spec.get("state_tolerances"),
            "matched_groups": len(differences),
            "requested_groups": target_episodes,
            "complete": len(differences) == target_episodes,
            "max_initial_state_abs_difference": max((item["max_abs_difference"] for item in differences), default=0.0),
            "groups": differences,
        },
    )


def command_for_trial(policy: str, seed: int, output: Path, spec: dict) -> list[str]:
    command = [
        sys.executable,
        "-u",
        spec["evaluator"],
        "--policy", EVALUATOR_POLICIES[policy],
        "--task", spec.get("tasks", TASKS)[policy],
        "--episodes", "1",
        "--condition-seed-start", str(seed),
        "--eval-num-envs", "1",
        "--success-criterion", spec.get("success_criterion", CRITERION),
        "--output", str(output),
        "--headless",
        "--enable_cameras",
    ]
    if policy == "teacher":
        command += ["--checkpoint", spec["teacher_checkpoint"]]
    elif policy == "residual":
        command += ["--checkpoint", spec["residual_checkpoint"]]
    if policy in ("vla", "residual"):
        command += [f"env.vla_host={spec['vla_host']}", f"env.vla_port={spec['vla_port']}"]
    return command


def run_one(policy: str, seed: int, output_dir: Path, spec: dict) -> dict:
    output = trial_path(output_dir, seed, policy)
    if output.exists():
        return validate_trial(load_json(output), policy=policy, seed=seed, spec=spec)
    log = output_dir / "logs" / f"{seed}_{policy}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = command_for_trial(policy, seed, output, spec)
    print(f"[START] seed={seed} policy={policy} log={log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND: " + json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.write(f"\nEVALUATOR_RC={completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"Evaluator failed: seed={seed} policy={policy} rc={completed.returncode}; log={log}")
    if not output.is_file():
        raise RuntimeError(f"Evaluator exited 0 without result: {output}")
    return validate_trial(load_json(output), policy=policy, seed=seed, spec=spec)


def run(args: argparse.Namespace) -> None:
    target_episodes = getattr(args, "target_episodes", EPISODES)
    count = args.count if args.count is not None else target_episodes
    if not 1 <= target_episodes <= EPISODES:
        raise ValueError(f"--target-episodes must be between 1 and {EPISODES}")
    if not 1 <= count <= target_episodes:
        raise ValueError(f"--count must be between 1 and {target_episodes}")
    if args.condition_seed_start < 0:
        raise ValueError("--condition-seed-start must be nonnegative")
    output_dir = args.output_dir.expanduser().resolve()
    check_no_other_simulator()
    spec = model_and_checkpoint_spec(args)
    check_service(args.vla_host, args.vla_port)
    ensure_manifest(output_dir, spec, write=not args.preflight_only, resume=args.resume)
    print(json.dumps({"preflight": "PASS", "output_dir": str(output_dir), "spec": spec}, indent=2), flush=True)
    if args.preflight_only:
        return

    committed: dict[str, list[dict]] = {policy: [] for policy in POLICIES}
    differences: list[dict] = []
    for index in range(count):
        seed = args.condition_seed_start + index
        group = {policy: run_one(policy, seed, output_dir, spec) for policy in POLICIES}
        try:
            max_difference = check_triple_states(
                group, seed=seed, atol=spec.get("state_atol", STATE_ATOL),
                field_atol=spec.get("state_tolerances"),
            )
        except ValueError as exc:
            atomic_json(
                output_dir / "pairing_failure.json",
                {"condition_seed": seed, "error": str(exc), "counted": False},
            )
            raise
        for policy in POLICIES:
            trial = dict(group[policy])
            trial["condition_index"] = index
            committed[policy].append(trial)
        differences.append({"condition_seed": seed, "max_abs_difference": max_difference})
        failure_path = output_dir / "pairing_failure.json"
        if failure_path.exists() and load_json(failure_path).get("condition_seed") == seed:
            failure_path.unlink()
        save_aggregates(output_dir, committed, differences, spec)
        summary = " ".join(f"{policy}={sum(t['success'] for t in committed[policy])}/{len(committed[policy])}" for policy in POLICIES)
        print(f"[MATCHED] {len(differences)}/{target_episodes} seed={seed} max_state_diff={max_difference:.3g} {summary}", flush=True)
    if len(differences) == target_episodes:
        print(f"[COMPLETE] {target_episodes} strictly matched conditions for all three policies", flush=True)
    else:
        print(f"[PAUSED] {len(differences)}/{target_episodes} matched conditions; rerun the same command to resume", flush=True)


if __name__ == "__main__":
    try:
        run(parse_args())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
