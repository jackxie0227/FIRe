"""Inspect one Parquet episode from a prepared FIRe GR00T dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


FIRE_LAB_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(
    os.environ.get("FIRE_EXPERIMENT_ROOT", FIRE_LAB_ROOT / "experiments")
).expanduser()
DEFAULT_DATASET = Path(
    os.environ.get(
        "FIRE_GR00T_DATASET",
        EXPERIMENT_ROOT / "datasets" / "gr00t" / "forge" / "peg_insert",
    )
).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--head", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episode < 0 or args.chunk_size <= 0 or args.head <= 0:
        raise ValueError("episode must be non-negative; chunk-size and head must be positive.")
    chunk = args.episode // args.chunk_size
    path = (
        args.dataset.expanduser().resolve()
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{args.episode:06d}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    print(f"path: {path}")
    print(f"rows: {len(frame)}")
    print("dtypes:")
    print(frame.dtypes.to_string())
    print("array shapes:")
    for column in frame.columns:
        value = frame[column].iloc[0]
        if isinstance(value, (list, np.ndarray)):
            print(f"  {column}: {np.stack(frame[column].to_numpy()).shape}")
    print("preview:")
    print(frame.head(args.head).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
