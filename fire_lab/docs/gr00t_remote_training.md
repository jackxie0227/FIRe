# Remote PegInsert GR00T training

The tracked repository contains the FIRe adapters and compatibility patches. Large generated
artifacts stay outside Git and are copied over the tailnet with resumable `rsync` transfers.

## Pinned inputs

- FIRe branch: `gr00t-peginsert-remote-training`
- Isaac-GR00T commit: `755876a9afdb41ca6eb6383b36f4a0adb085c73f`
- GR00T-N1-2B snapshot: `fc879581ca32f4f6d6e02cf0cc80452f6b0c3873`
- Dataset: `experiments/datasets/gr00t/forge/peg_insert`
- Dataset size: 1,004 episodes / 89,356 samples / 4,021 files
- Data config and embodiment: `franka_triple_cam` / `franka`

After checking out the FIRe branch, verify the transferred GR00T source tree with:

```bash
cd /home/ubuntu/RIM/projects/FIRe/fire_lab
/home/ubuntu/miniconda3/bin/conda run -n gr00t \
  python scripts/tools/setup_gr00t_n1.py --gr00t-repo _gr00t --check
```

The source machine copies ignored artifacts to the same relative locations:

```text
fire_lab/_gr00t/
fire_lab/experiments/datasets/gr00t/forge/peg_insert/
fire_lab/experiments/gr00t_finetune/cache/huggingface/hub/models--nvidia--GR00T-N1-2B/
```

## GPU gate

The destination RTX 3080 has 10 GiB VRAM. The original full AdamW optimizer path peaks at about
10.51 GiB even with batch size 1 on the source machine, so the source batch-size-4 command must not
be reused remotely. Create the remote environment first, then run a batch-size-1 optimizer smoke
test. If it OOMs, reduce optimizer/model memory while keeping the LLM and vision backbone frozen;
repeat the smoke and only generate a formal command after at least 1 GiB stable headroom is measured.

Formal training remains a manual user action. Never store SSH passwords in this repository.

## Transfer record (2026-09-06)

The migration to `/home/ubuntu/RIM/projects/FIRe` completed over Tailscale SSH:

- FIRe commit `864762a9416b4d1672f196c9d63e4b6a4929301c` was pushed directly into the
  destination repository on branch `gr00t-peginsert-remote-training`.
- The dataset arrived as 177 MiB and 4,021 files; an `rsync --checksum --dry-run` reported no
  differences.
- The patched `_gr00t` tree arrived at commit `755876a9afdb41ca6eb6383b36f4a0adb085c73f`;
  its seven expected compatibility/adapter modifications are present.
- The 4.1 GiB GR00T-N1-2B cache transferred successfully; an `rsync --checksum --dry-run`
  reported no differences and the fixed snapshot links resolve on the destination.

The destination has 318 GiB free disk space. PyPI and the PyTorch wheel index are reachable, but
Hugging Face is not, which is why the model cache was transferred instead of downloaded remotely.
