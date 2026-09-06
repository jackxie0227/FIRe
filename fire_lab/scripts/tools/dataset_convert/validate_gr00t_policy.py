#!/usr/bin/env python3
"""Validate a saved FIRe GR00T policy's offline action contract and memory stability."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


ACTION_SHAPES = {
    "action.eef_position_delta": (16, 3),
    "action.eef_rotation_delta": (16, 3),
    # Gr00tPolicy squeezes every singleton dimension for an unbatched request.
    "action.gripper_close": (16,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--data-config", default="franka_triple_cam")
    parser.add_argument("--embodiment-tag", default="franka")
    parser.add_argument("--video-backend", default="decord")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path("experiments/gr00t_finetune/cache/huggingface"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.requests < 2:
        raise ValueError("--requests must be at least 2 so memory stability can be checked")
    os.environ.setdefault("HF_HOME", str(args.hf_home.resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((args.hf_home / "transformers").resolve()))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("ALBUMENTATIONS_OFFLINE", "1")

    import torch

    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GR00T policy preflight")

    config = DATA_CONFIG_MAP[args.data_config]
    policy = Gr00tPolicy(
        model_path=str(args.model.resolve()),
        modality_config=config.modality_config(),
        modality_transform=config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda",
    )
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset.resolve(),
        modality_configs=policy.get_modality_config(),
        transforms=None,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        video_backend=args.video_backend,
    )
    trajectory_id, base_index = dataset.all_steps[args.sample_index]
    observation = dataset.get_step_data(trajectory_id, base_index)

    timings_ms = []
    allocated_gib = []
    final_action = None
    for request_index in range(args.requests):
        torch.cuda.synchronize()
        started = time.perf_counter()
        action = policy.get_action(observation)
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - started) * 1000.0)
        allocated_gib.append(torch.cuda.memory_allocated() / 2**30)
        for key, expected_shape in ACTION_SHAPES.items():
            value = np.asarray(action[key])
            if value.shape != expected_shape:
                raise RuntimeError(f"{key} shape is {value.shape}; expected {expected_shape}")
            if not np.isfinite(value).all():
                raise RuntimeError(f"{key} contains NaN or Inf")
        final_action = np.concatenate(
            [
                np.asarray(action["action.eef_position_delta"]),
                np.asarray(action["action.eef_rotation_delta"]),
                np.asarray(action["action.gripper_close"])[..., None],
            ],
            axis=-1,
        )
        if final_action.shape != (16, 7):
            raise RuntimeError(f"Combined action shape is {final_action.shape}; expected (16, 7)")
        print(
            f"request={request_index + 1}, latency_ms={timings_ms[-1]:.2f}, "
            f"allocated_gib={allocated_gib[-1]:.3f}"
        )

    post_warmup_growth_gib = max(allocated_gib[1:]) - min(allocated_gib[1:])
    if post_warmup_growth_gib > 0.25:
        raise RuntimeError(
            f"Post-warmup allocated memory varied by {post_warmup_growth_gib:.3f} GiB (> 0.25 GiB)"
        )

    report = {
        "status": "pass",
        "dataset": str(args.dataset.resolve()),
        "model": str(args.model.resolve()),
        "requests": args.requests,
        "action_shapes": {key: list(value) for key, value in ACTION_SHAPES.items()},
        "combined_action_shape": list(final_action.shape),
        "action_dtype": str(final_action.dtype),
        "action_min": float(final_action.min()),
        "action_max": float(final_action.max()),
        "latency_ms": timings_ms,
        "post_warmup_mean_latency_ms": float(np.mean(timings_ms[1:])),
        "allocated_gib": allocated_gib,
        "post_warmup_growth_gib": post_warmup_growth_gib,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
