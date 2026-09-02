#!/usr/bin/env python3
"""Synchronize FIRe's adapters and compatibility fixes into GR00T n1-release."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


EXPECTED_COMMIT = "755876a9afdb41ca6eb6383b36f4a0adb085c73f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-repo", type=Path, default=Path("_gr00t"))
    parser.add_argument("--check", action="store_true", help="Verify without changing files")
    return parser.parse_args()


def replace_once(path: Path, old: str, new: str, check: bool) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Expected n1-release source block not found in {path}")
    if check:
        raise RuntimeError(f"Compatibility patch is missing from {path}")
    path.write_text(text.replace(old, new, 1))
    return True


def sync_file(source: Path, target: Path, check: bool) -> bool:
    if target.exists() and target.read_bytes() == source.read_bytes():
        return False
    if check:
        raise RuntimeError(f"Adapter is not synchronized: {target}")
    shutil.copy2(source, target)
    return True


def main() -> None:
    args = parse_args()
    fire_lab = Path(__file__).resolve().parents[2]
    repo = args.gr00t_repo.resolve()
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Expected GR00T {EXPECTED_COMMIT}, found {commit}")

    changed = []
    adapters = {
        fire_lab / "gr00t/embodiment_tags.py": repo / "gr00t/data/embodiment_tags.py",
        fire_lab / "gr00t/data_config.py": repo / "gr00t/experiment/data_config.py",
    }
    for source, target in adapters.items():
        if sync_file(source, target, args.check):
            changed.append(str(target))

    transforms = repo / "gr00t/model/transforms.py"
    if replace_once(
        transforms,
        "from gr00t.data.schema import DatasetMetadata, EmbodimentTag\n",
        "from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING\n"
        "from gr00t.data.schema import DatasetMetadata, EmbodimentTag\n",
        args.check,
    ):
        changed.append(str(transforms))
    if replace_once(
        transforms,
        '    _EMBODIMENT_TAG_MAPPING = {\n'
        '        "gr1": 24,\n'
        '        "new_embodiment": 31,  # use the last projector for new embodiment,\n'
        "    }\n",
        "    _EMBODIMENT_TAG_MAPPING = EMBODIMENT_TAG_MAPPING\n",
        args.check,
    ):
        changed.append(str(transforms))

    action_head = repo / "gr00t/model/action_head/flow_matching_action_head.py"
    if replace_once(
        action_head,
        "    def sample_time(self, batch_size, device, dtype):\n"
        "        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)\n"
        "        return (self.config.noise_s - sample) / self.config.noise_s\n",
        "    def sample_time(self, batch_size, device, dtype):\n"
        "        # Dirichlet/Beta sampling is not implemented for bfloat16. Transformers\n"
        "        # may construct this distribution inside a bfloat16 initialization\n"
        "        # context, so explicitly sample the same distribution in float32 and\n"
        "        # cast only the result to the model dtype.\n"
        "        beta_dist = Beta(\n"
        "            self.beta_dist.concentration1.float(),\n"
        "            self.beta_dist.concentration0.float(),\n"
        "        )\n"
        "        sample = beta_dist.sample([batch_size]).to(device, dtype=dtype)\n"
        "        return (self.config.noise_s - sample) / self.config.noise_s\n",
        args.check,
    ):
        changed.append(str(action_head))

    finetune_script = repo / "scripts/gr00t_finetune.py"
    if replace_once(
        finetune_script,
        "    model = GR00T_N1.from_pretrained(\n"
        "        pretrained_model_name_or_path=config.base_model_path,\n",
        "    model = GR00T_N1.from_pretrained(\n"
        "        pretrained_model_name_or_path=config.base_model_path,\n"
        "        torch_dtype=torch.bfloat16,\n",
        args.check,
    ):
        changed.append(str(finetune_script))

    runner = repo / "gr00t/experiment/runner.py"
    if replace_once(
        runner,
        "        compute_dtype = torch.float16 if training_args.bf16 else torch.float32\n",
        "        compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float32\n",
        args.check,
    ):
        changed.append(str(runner))

    mode = "verified" if args.check else "synchronized"
    print(f"GR00T n1-release adapters {mode}; changed={len(set(changed))}")
    for path in sorted(set(changed)):
        print(f"  {path}")


if __name__ == "__main__":
    main()
