# GearMesh 教师 PPO

- **身份与来源**：seed 0；RL-Games PPO，128 环境、horizon 128、学习率 1e-4。日志为本目录 `params/` 与 `summaries/`；训练时代码 commit 未记录。
- **进度与趋势**：完成 200/200 轮。训练内 `episode_success_rate` 从 0.78%（第 3 轮）升至 98.4%（第 200 轮），第 78 轮首次达到 100%，末 20 条均值 99.2%；`rewards/iter` 从 226.3 升至 724.0，最高 731.7（第 185 轮）；`c_loss` 从 1.936 降至 0.0030。后期训练指标接近平台。
- **权重与独立评估**：选用 `nn/Forge.pth`，SHA-256 `aa7626a2…2425ab82e`；选择依据为完成训练且被后续示范采集和配对评估使用。GearMesh 配对评估目前只完成 88/100 组，教师成功 88/88；见 `fire_lab/experiments/evaluations/paired/gear_mesh/three_way_native_100_seed100000_20261025/teacher_100.json`，此结果不是完整 100 组评估。
- **结论与限制**：训练达到预定轮数且训练内成功率高；独立评估尚未完成，不能据 88 组结果断言完整评估表现或跨 seed 泛化。
