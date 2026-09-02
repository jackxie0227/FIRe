#!/usr/bin/env python3
"""Smoke-test a prepared FIRe dataset with the NVIDIA GR00T N1 loader."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Prepared LeRobot/GR00T dataset root")
    parser.add_argument("--data-config", default="franka_triple_cam")
    parser.add_argument("--embodiment-tag", default="franka")
    parser.add_argument("--video-backend", default="decord")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=[0, 44410, 88910, -1],
        help="Global frame indices to sample; -1 means the final frame",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path("experiments/gr00t_finetune/cache/huggingface"),
    )
    return parser.parse_args()


def describe(value: Any) -> str:
    shape = tuple(value.shape) if hasattr(value, "shape") else None
    dtype = str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
    finite = "n/a"
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        finite = str(bool(np.isfinite(value).all()))
    elif isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        finite = str(bool(torch.isfinite(value).all()))
    return f"shape={shape}, dtype={dtype}, finite={finite}"


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home.resolve()))
    os.environ.setdefault("ALBUMENTATIONS_OFFLINE", "1")

    # Delay GR00T imports until after cache/offline environment variables are set.
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.transforms import DefaultDataCollatorGR00T

    data_config = DATA_CONFIG_MAP[args.data_config]
    transforms = data_config.transform()
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset.resolve(),
        modality_configs=data_config.modality_config(),
        transforms=transforms,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        video_backend=args.video_backend,
    )

    total_frames = int(dataset.trajectory_lengths.sum())
    print(
        f"dataset={dataset.dataset_path} episodes={len(dataset.trajectory_ids)} "
        f"frames={total_frames} samples={len(dataset)}"
    )
    for requested_index in args.indices:
        index = requested_index if requested_index >= 0 else len(dataset) + requested_index
        sample = dataset[index]
        print(f"sample[{index}]")
        for key, value in sample.items():
            print(f"  {key}: {describe(value)}")

    final_transform = transforms.transforms[-1]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=DefaultDataCollatorGR00T(final_transform.vlm_processor),
    )
    batch = next(iter(loader))
    print(f"batch_size={args.batch_size}")
    for key, value in batch.items():
        print(f"  {key}: {describe(value)}")


if __name__ == "__main__":
    main()
