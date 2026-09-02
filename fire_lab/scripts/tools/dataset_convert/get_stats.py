"""Compute normalization statistics for a FIRe GR00T dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FIRE_LAB_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(
    os.environ.get("FIRE_EXPERIMENT_ROOT", FIRE_LAB_ROOT / "experiments")
).expanduser()
DEFAULT_DATASET = Path(
    os.environ.get(
        "FIRE_GR00T_DATASET",
        EXPERIMENT_ROOT / "datasets" / "gr00t" / "forge" / "peg_insert",
    )
).expanduser()


def discover_parquet_files(dataset: Path, chunk_ids: list[str] | None = None) -> list[Path]:
    data_dir = dataset / "data"
    if chunk_ids:
        return [
            path
            for chunk_id in chunk_ids
            for path in sorted((data_dir / chunk_id).glob("episode_*.parquet"))
        ]
    return sorted(data_dir.glob("chunk-*/episode_*.parquet"))


def validate_parquet_schemas(parquet_files: list[Path]) -> None:
    """Require every Parquet file to use the same Arrow schema."""
    if not parquet_files:
        raise ValueError("No Parquet files were provided for schema validation.")
    reference_path = parquet_files[0]
    reference_schema = pq.read_schema(reference_path).remove_metadata()
    for path in parquet_files[1:]:
        schema = pq.read_schema(path).remove_metadata()
        if not schema.equals(reference_schema):
            raise ValueError(
                f"Parquet schema mismatch: {path} differs from {reference_path}."
            )


def _numeric_statistics(values: np.ndarray) -> dict[str, float | list[float]]:
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute_statistics(
    dataset: Path,
    chunk_ids: list[str] | None = None,
    validate_schema: bool = True,
) -> tuple[dict[str, dict[str, float | list[float]]], int, int]:
    """Compute exact statistics across all selected chunks."""
    dataset = dataset.expanduser().resolve()
    parquet_files = discover_parquet_files(dataset, chunk_ids)
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found below {dataset / 'data'}.")
    if validate_schema:
        validate_parquet_schemas(parquet_files)

    accumulated: dict[str, list[np.ndarray]] = {}
    total_frames = 0
    for path in parquet_files:
        frame = pd.read_parquet(path)
        total_frames += len(frame)
        for column in frame.columns:
            sample = frame[column].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                values = np.stack(frame[column].to_numpy())
            elif np.issubdtype(type(sample), np.number) and not np.issubdtype(
                type(sample), np.bool_
            ):
                values = frame[column].to_numpy(dtype=np.float64)
            else:
                continue
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values found in {column} of {path}.")
            accumulated.setdefault(column, []).append(values)

    statistics = {
        column: _numeric_statistics(np.concatenate(parts, axis=0))
        for column, parts in accumulated.items()
    }
    return statistics, len(parquet_files), total_frames


def write_statistics(path: Path, statistics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(statistics, indent=4) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute stats.json from every Parquet chunk in a GR00T dataset."
    )
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--chunk-id",
        action="append",
        dest="chunk_ids",
        help="Only include this chunk (repeatable). Defaults to every chunk-* directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <dataset>/meta/stats.json.",
    )
    parser.add_argument("--skip-schema-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output or dataset / "meta" / "stats.json"
    try:
        statistics, episode_count, frame_count = compute_statistics(
            dataset, args.chunk_ids, validate_schema=not args.skip_schema_check
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1
    write_statistics(output, statistics)
    print(
        f"Computed {len(statistics)} numeric fields from {episode_count} episodes "
        f"and {frame_count} frames."
    )
    print(f"Saved stats.json to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
