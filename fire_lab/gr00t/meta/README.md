# GR00T metadata templates

Files in this directory are reference templates inherited from the original
FIRe repository. They are not the authoritative metadata for a collected
dataset and may contain example counts or statistics.

Prepare and validate the actual PegInsert dataset with:

```bash
python scripts/tools/dataset_convert/prepare_gr00t_dataset.py
```

The authoritative generated metadata is written to:

```text
experiments/datasets/gr00t/forge/peg_insert/meta/
```

This keeps generated episode manifests and normalization statistics next to
the exact Parquet and video files they describe.
