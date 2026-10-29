# GearMesh VLA 微调

- **身份与来源**：GR00T-N1-2B，GearMesh 示范数据，seed 42、batch size 4、目标 20,000 step。原始日志及参数位于 `fire_lab/experiments/gr00t_finetune/forge/gear_mesh/n1_2b_steps20000_seed42_bs4_20261007/` 的 `trainer_state.json`、`runs/` 和 `training_args.bin`；训练时代码 commit 未记录。
- **进度与趋势**：完成 20,000/20,000 step，最终 `train_loss=0.2542`。每 10 step 的训练 loss 从 1.3352（step 10）降至 0.2222（step 20,000），单点最低 0.0931（step 7,920）；首/末 100 条均值为 1.0919/0.1986。未记录独立验证 loss。
- **权重与独立评估**：使用同目录的 `model.safetensors`（另有 `checkpoint-20000/`），按完成目标 step 选取。GearMesh 配对评估只完成 88/100 组，VLA-only 成功 56/88（63.6%），见 `fire_lab/experiments/evaluations/paired/gear_mesh/three_way_native_100_seed100000_20261025/vla_100.json` 与 `manifest.json`。
- **结论与限制**：训练完成、训练 loss 下降；独立评估仍不完整，且无验证曲线，无法确定最佳中间 step 或是否过拟合。
