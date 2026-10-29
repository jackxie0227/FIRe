# PegInsert VLA 微调

- **身份与来源**：GR00T-N1-2B，PegInsert 示范数据，seed 42、batch size 4、目标 20,000 step。原始日志及参数位于 `fire_lab/experiments/gr00t_finetune/forge/peg_insert/n1_2b_steps20000_seed42_bs4_20260907/` 的 `trainer_state.json`、`runs/` 和 `training_args.bin`；训练时代码 commit 未记录。
- **进度与趋势**：完成 20,000/20,000 step，最终 `train_loss=0.2275`。每 10 step 记录的训练 loss 从 1.2945（step 10）降至 0.2027（step 20,000），单点最低 0.0899（step 14,170）；首/末 100 条均值为 1.0524/0.1725。未记录独立验证 loss，因此单点最低值不是模型选择依据。
- **权重与独立评估**：使用同目录的 `model.safetensors`（另有 `checkpoint-20000/`）；选择依据是达到计划 step 的最终权重。相同模型在 500 个配对条件下，VLA-only 成功 114/500（22.8%），见 `fire_lab/experiments/evaluations/paired/peg_insert/three_way_official_500_seed100000_20260916/manifest.json` 与 `vla_500.json`。
- **结论与限制**：微调训练完成且训练 loss 明显下降；独立评估表明该 VLA 单独执行 PegInsert 的成功率有限。缺少训练期间的验证曲线，无法据此判断过拟合或最佳中间 step。
