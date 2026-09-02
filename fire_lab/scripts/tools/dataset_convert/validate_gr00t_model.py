#!/usr/bin/env python3
"""Run a one-batch GR00T N1 forward/backward smoke test on a FIRe dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("model", help="Hugging Face model ID or local snapshot directory")
    parser.add_argument("--data-config", default="franka_triple_cam")
    parser.add_argument("--embodiment-tag", default="franka")
    parser.add_argument("--video-backend", default="decord")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path("experiments/gr00t_finetune/cache/huggingface"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(args.hf_home.resolve()))
    os.environ.setdefault("ALBUMENTATIONS_OFFLINE", "1")

    import torch
    from torch.utils.data import DataLoader

    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.data.schema import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.gr00t_n1 import GR00T_N1
    from gr00t.model.transforms import DefaultDataCollatorGR00T

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    config = DATA_CONFIG_MAP[args.data_config]
    transforms = config.transform()
    dataset = LeRobotSingleDataset(
        args.dataset.resolve(),
        config.modality_config(),
        EmbodimentTag(args.embodiment_tag),
        video_backend=args.video_backend,
        transforms=transforms,
    )
    collator = DefaultDataCollatorGR00T(transforms.transforms[-1].vlm_processor)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collator)))

    model = GR00T_N1.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        tune_llm=False,
        tune_visual=False,
        tune_projector=args.backward,
        tune_diffusion_model=args.backward,
    ).to(args.device)
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.train(args.backward)

    context = torch.enable_grad() if args.backward else torch.inference_mode()
    with context:
        outputs = model(batch)
        loss = outputs["loss"]
        print(f"loss={float(loss.detach().cpu()):.8f}, finite={bool(torch.isfinite(loss))}")
        if args.backward:
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients:
                raise RuntimeError("Backward completed without producing trainable gradients")
            gradients_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
            print(f"backward=true, gradient_tensors={len(gradients)}, finite={gradients_finite}")
            if not gradients_finite:
                raise RuntimeError("Backward produced NaN or Inf gradients")

    if args.device == "cuda":
        print(
            f"cuda_peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}, "
            f"cuda_peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}"
        )


if __name__ == "__main__":
    main()
