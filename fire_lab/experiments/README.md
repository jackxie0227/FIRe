# FIRe experiment outputs

This directory is the default root for local reproduction artifacts. Generated
files are ignored by Git; only this guide is tracked.

```text
experiments/
├── checkpoints/baseline/       # Formal RL baseline checkpoints
├── sim_demos/forge/            # Raw demonstrations saved by simulation
├── datasets/gr00t/forge/       # Prepared GR00T/LeRobot datasets
├── gr00t_finetune/             # GR00T fine-tuning runs and checkpoints
└── evaluations/                # Simulation metrics and comparison results
```

The default root can be moved outside the repository by setting
`FIRE_EXPERIMENT_ROOT`. Raw demonstration collection can be redirected with
`FIRE_DEMO_DATASET_ROOT`, and the statistics script accepts either a dataset
path argument or `FIRE_GR00T_DATASET`.

Example:

```bash
export FIRE_EXPERIMENT_ROOT=/path/to/fire-experiments
python scripts/tools/dataset_convert/get_stats.py \
  /path/to/fire-experiments/datasets/gr00t/forge/peg_insert
```
