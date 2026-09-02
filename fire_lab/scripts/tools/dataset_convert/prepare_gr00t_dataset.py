"""Validate FIRe demonstrations and prepare a GR00T/LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from get_stats import compute_statistics, write_statistics


FIRE_LAB_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(
    os.environ.get("FIRE_EXPERIMENT_ROOT", FIRE_LAB_ROOT / "experiments")
).expanduser()
DEFAULT_SOURCE = EXPERIMENT_ROOT / "sim_demos" / "forge" / "peg_insert"
DEFAULT_TARGET = EXPERIMENT_ROOT / "datasets" / "gr00t" / "forge" / "peg_insert"

CAMERA_KEYS = (
    "observation.images.left_view",
    "observation.images.right_view",
    "observation.images.wrist_view",
)
EXPECTED_COLUMNS = (
    "observation.state",
    "observation.force",
    "action",
    "timestamp",
    "annotation.human.action.task_description",
    "task_index",
    "annotation.human.validity",
    "episode_index",
    "index",
    "next.reward",
    "next.done",
    "episode.success",
)
EXPECTED_FRAMES = 89
FPS = 15
CHUNK_SIZE = 1000
TASK_DESCRIPTION = "Insert peg into the socket"


def parse_episode_id(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid episode filename: {path}") from error


def indexed_files(directory: Path, suffix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(directory.glob(f"episode_*{suffix}")):
        episode_id = parse_episode_id(path)
        if episode_id in files:
            raise ValueError(f"Duplicate episode {episode_id} below {directory}.")
        files[episode_id] = path
    return files


def discover_source(source: Path) -> tuple[dict[int, Path], dict[str, dict[int, Path]]]:
    parquet = indexed_files(source / "data" / "chunk-000", ".parquet")
    videos = {
        camera: indexed_files(source / "videos" / "chunk-000" / camera, ".mp4")
        for camera in CAMERA_KEYS
    }
    if not parquet:
        raise ValueError(f"No source Parquet files found below {source}.")
    expected_ids = set(range(len(parquet)))
    if set(parquet) != expected_ids:
        missing = sorted(expected_ids - set(parquet))
        raise ValueError(f"Source episode IDs are not contiguous from zero; missing={missing[:10]}.")
    for camera, camera_files in videos.items():
        if set(camera_files) != expected_ids:
            missing = sorted(expected_ids - set(camera_files))
            extra = sorted(set(camera_files) - expected_ids)
            raise ValueError(
                f"Video/Parquet mismatch for {camera}: missing={missing[:10]}, extra={extra[:10]}."
            )
    return parquet, videos


def validate_arrow_schema(parquet: dict[int, Path]) -> None:
    first_id = min(parquet)
    reference = pq.read_schema(parquet[first_id]).remove_metadata()
    if tuple(reference.names) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns in {parquet[first_id]}: {reference.names}; "
            f"expected {list(EXPECTED_COLUMNS)}."
        )
    for episode_id, path in parquet.items():
        if not pq.read_schema(path).remove_metadata().equals(reference):
            raise ValueError(f"Arrow schema mismatch in episode {episode_id}: {path}.")


def validate_parquet_file(episode_id: int, path: Path) -> None:
    frame = pd.read_parquet(path)
    if len(frame) != EXPECTED_FRAMES:
        raise ValueError(f"Episode {episode_id} has {len(frame)} rows; expected {EXPECTED_FRAMES}.")
    if tuple(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Episode {episode_id} has unexpected columns.")
    state = np.stack(frame["observation.state"].to_numpy())
    force = np.stack(frame["observation.force"].to_numpy())
    action = np.stack(frame["action"].to_numpy())
    if state.shape != (EXPECTED_FRAMES, 9):
        raise ValueError(f"Episode {episode_id} state shape is {state.shape}, expected (89, 9).")
    if force.shape != (EXPECTED_FRAMES, 3):
        raise ValueError(f"Episode {episode_id} force shape is {force.shape}, expected (89, 3).")
    if action.shape != (EXPECTED_FRAMES, 7):
        raise ValueError(f"Episode {episode_id} action shape is {action.shape}, expected (89, 7).")
    if not all(np.isfinite(values).all() for values in (state, force, action)):
        raise ValueError(f"Episode {episode_id} contains non-finite state, force, or action values.")
    timestamp = frame["timestamp"].to_numpy(dtype=np.float64)
    if not np.isfinite(timestamp).all() or timestamp[0] != 0.0 or not np.all(np.diff(timestamp) > 0):
        raise ValueError(f"Episode {episode_id} has invalid timestamps.")
    if not np.allclose(np.diff(timestamp), 1.0 / FPS, atol=0.0007):
        raise ValueError(f"Episode {episode_id} timestamps do not match {FPS} FPS.")
    done = frame["next.done"].to_numpy(dtype=bool)
    if done.sum() != 1 or not done[-1]:
        raise ValueError(f"Episode {episode_id} must have exactly one terminal flag on its last row.")
    if not frame["episode.success"].astype(bool).all():
        raise ValueError(f"Episode {episode_id} is not consistently marked successful.")
    if not frame["annotation.human.validity"].astype(bool).all():
        raise ValueError(f"Episode {episode_id} contains invalid frames.")
    if not (frame["next.reward"].to_numpy(dtype=np.float64) > 0).any():
        raise ValueError(f"Episode {episode_id} never reaches the success reward.")
    if set(frame["episode_index"].astype(int)) != {episode_id}:
        raise ValueError(f"Episode {episode_id} has an incorrect episode_index column.")
    expected_indices = np.arange(
        episode_id * EXPECTED_FRAMES, (episode_id + 1) * EXPECTED_FRAMES, dtype=np.int64
    )
    if not np.array_equal(frame["index"].to_numpy(dtype=np.int64), expected_indices):
        raise ValueError(f"Episode {episode_id} has non-contiguous global frame indices.")


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream in {path}; found {len(streams)}.")
    return streams[0]


def validate_video(path: Path) -> None:
    stream = probe_video(path)
    expected = {
        "codec_name": "h264", "width": 256, "height": 256,
        "pix_fmt": "yuv420p", "r_frame_rate": "15/1",
        "nb_read_frames": str(EXPECTED_FRAMES),
    }
    mismatches = {
        key: (stream.get(key), value)
        for key, value in expected.items()
        if stream.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Video metadata/frame mismatch in {path}: {mismatches}.")


def validate_source(
    parquet: dict[int, Path], videos: dict[str, dict[int, Path]], video_check: str
) -> None:
    validate_arrow_schema(parquet)
    for episode_id, path in parquet.items():
        validate_parquet_file(episode_id, path)
    if video_check == "none":
        return
    if video_check == "sample":
        candidates = {0, len(parquet) // 2, 999, 1000, len(parquet) - 1}
        episode_ids = sorted(episode_id for episode_id in candidates if episode_id in parquet)
    else:
        episode_ids = sorted(parquet)
    for episode_id in episode_ids:
        for camera in CAMERA_KEYS:
            validate_video(videos[camera][episode_id])


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise FileExistsError(f"Target exists but is not the expected hard link: {target}")
    try:
        os.link(source, target)
    except OSError as error:
        raise OSError(f"Could not hard-link {source} to {target}: {error}") from error


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def feature_schema() -> dict:
    features = {
        camera: {
            "dtype": "video", "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(FPS), "video.codec": "h264",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "has_audio": False,
            },
        }
        for camera in CAMERA_KEYS
    }
    features.update({
        "observation.state": {
            "dtype": "float64", "shape": [9],
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_qpos1", "gripper_qpos2"],
        },
        "observation.force": {"dtype": "float64", "shape": [3], "names": ["fx", "fy", "fz"]},
        "action": {
            "dtype": "float64", "shape": [7],
            "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_close"],
        },
        "timestamp": {"dtype": "float64", "shape": [1]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "annotation.human.validity": {"dtype": "bool", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.reward": {"dtype": "float64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "episode.success": {"dtype": "bool", "shape": [1]},
    })
    return features


def build_metadata(episode_count: int) -> dict[str, str]:
    info = {
        "codebase_version": "v2.0", "robot_type": "franka_emika_panda",
        "total_episodes": episode_count, "total_frames": episode_count * EXPECTED_FRAMES,
        "total_tasks": 1, "total_videos": len(CAMERA_KEYS),
        "total_chunks": math.ceil(episode_count / CHUNK_SIZE), "chunks_size": CHUNK_SIZE,
        "fps": float(FPS), "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": feature_schema(),
    }
    modality = {
        "state": {
            "eef_position": {"start": 0, "end": 3},
            "eef_quaternion": {"start": 3, "end": 7, "rotation_type": "quaternion"},
            "gripper_qpos": {"start": 7, "end": 9},
        },
        "action": {
            "eef_position_delta": {"start": 0, "end": 3},
            "eef_rotation_delta": {"start": 3, "end": 6, "rotation_type": "axis_angle"},
            "gripper_close": {"start": 6, "end": 7},
        },
        "video": {
            "left_view": {"original_key": "observation.images.left_view"},
            "right_view": {"original_key": "observation.images.right_view"},
            "wrist_view": {"original_key": "observation.images.wrist_view"},
        },
        "annotation": {"human.action.task_description": {}, "human.validity": {}},
    }
    episodes = "".join(
        json.dumps({"episode_index": episode_id, "tasks": [TASK_DESCRIPTION], "length": EXPECTED_FRAMES}) + "\n"
        for episode_id in range(episode_count)
    )
    return {
        "info.json": json.dumps(info, indent=4) + "\n",
        "modality.json": json.dumps(modality, indent=4) + "\n",
        "episodes.jsonl": episodes,
        "tasks.jsonl": json.dumps({"task_index": 0, "task": TASK_DESCRIPTION}) + "\n",
    }


def prepare(source: Path, target: Path, dry_run: bool, video_check: str) -> None:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    parquet, videos = discover_source(source)
    if source == target:
        raise ValueError("Source and target must be separate directories.")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev != target.parent.stat().st_dev:
            raise OSError("Source and target are on different filesystems; hard links are not possible.")
    print(f"Validating {len(parquet)} Parquet files and {len(parquet) * len(CAMERA_KEYS)} videos...")
    validate_source(parquet, videos, video_check)
    print(
        f"Plan: {len(parquet)} Parquet links, {len(parquet) * len(CAMERA_KEYS)} video links, "
        f"{math.ceil(len(parquet) / CHUNK_SIZE)} chunks, and 5 metadata files."
    )
    if dry_run:
        print("Dry run complete; no files were written.")
        return
    for episode_id, source_path in parquet.items():
        chunk = f"chunk-{episode_id // CHUNK_SIZE:03d}"
        hardlink(source_path, target / "data" / chunk / source_path.name)
        for camera in CAMERA_KEYS:
            video_source = videos[camera][episode_id]
            hardlink(video_source, target / "videos" / chunk / camera / video_source.name)
    for filename, contents in build_metadata(len(parquet)).items():
        write_text_atomic(target / "meta" / filename, contents)
    statistics, stats_episodes, stats_frames = compute_statistics(target)
    if stats_episodes != len(parquet) or stats_frames != len(parquet) * EXPECTED_FRAMES:
        raise ValueError(f"Statistics input mismatch: episodes={stats_episodes}, frames={stats_frames}.")
    write_statistics(target / "meta" / "stats.json", statistics)
    print(f"Prepared dataset at {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--video-check", choices=("none", "sample", "all"), default="sample")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        prepare(args.source, args.target, args.dry_run, args.video_check)
    except (FileNotFoundError, FileExistsError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
