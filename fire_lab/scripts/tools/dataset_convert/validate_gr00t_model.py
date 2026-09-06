#!/usr/bin/env python3
"""Run a finite forward/backward/optimizer smoke test on a FIRe GR00T dataset."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=0,
        help="Run this many AdamW updates; implies --backward.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path("experiments/gr00t_finetune/cache/huggingface"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.optimizer_steps < 0:
        raise ValueError("--optimizer-steps cannot be negative")
    run_backward = args.backward or args.optimizer_steps > 0
    os.environ.setdefault("HF_HOME", str(args.hf_home.resolve()))
    os.environ.setdefault("TRANSFORMERS_CACHE", str((args.hf_home / "transformers").resolve()))
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
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

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
    batch = next(
        iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collator))
    )

    model = GR00T_N1.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        tune_llm=False,
        tune_visual=False,
        tune_projector=run_backward,
        tune_diffusion_model=run_backward,
    ).to(args.device)
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    model.train(run_backward)

    optimizer = None
    parameter_probes = {}
    if args.optimizer_steps:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            betas=(0.95, 0.999),
            eps=1e-8,
            weight_decay=1e-5,
        )
        parameter_probes = {
            name: parameter.detach().reshape(-1)[:16].float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    losses = []
    gradient_tensors = 0
    passes = max(1, args.optimizer_steps)
    context = torch.enable_grad() if run_backward else torch.inference_mode()
    with context:
        for step in range(passes):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = outputs["loss"]
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            loss_finite = bool(torch.isfinite(loss))
            print(f"step={step + 1}, loss={loss_value:.8f}, finite={loss_finite}")
            if not loss_finite:
                raise RuntimeError("Forward produced a NaN or Inf loss")
            if run_backward:
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients:
                    raise RuntimeError("Backward completed without producing trainable gradients")
                gradient_tensors = len(gradients)
                gradients_finite = all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
                print(
                    f"backward=true, gradient_tensors={gradient_tensors}, "
                    f"finite={gradients_finite}"
                )
                if not gradients_finite:
                    raise RuntimeError("Backward produced NaN or Inf gradients")
            if optimizer is not None:
                optimizer.step()

    changed_parameter_tensors = 0
    if optimizer is not None:
        changed_parameter_tensors = sum(
            not torch.equal(
                parameter_probes[name],
                parameter.detach().reshape(-1)[:16].float().cpu(),
            )
            for name, parameter in model.named_parameters()
            if name in parameter_probes
        )
        print(f"optimizer_steps={args.optimizer_steps}, changed_parameter_tensors={changed_parameter_tensors}")
        if changed_parameter_tensors == 0:
            raise RuntimeError("Optimizer steps did not change any probed trainable parameters")

    peak_allocated_gib = None
    peak_reserved_gib = None
    if args.device == "cuda":
        peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
        print(
            f"cuda_peak_allocated_gib={peak_allocated_gib:.3f}, "
            f"cuda_peak_reserved_gib={peak_reserved_gib:.3f}"
        )

    if args.json_output:
        report = {
            "status": "pass",
            "dataset": str(args.dataset.resolve()),
            "model": args.model,
            "device": args.device,
            "batch_size": args.batch_size,
            "optimizer_steps": args.optimizer_steps,
            "losses": losses,
            "gradient_tensors": gradient_tensors,
            "changed_parameter_tensors": changed_parameter_tensors,
            "cuda_peak_allocated_gib": peak_allocated_gib,
            "cuda_peak_reserved_gib": peak_reserved_gib,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
