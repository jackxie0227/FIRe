# GearMesh VLA+残差 PPO（FIRe6D）

- **身份与来源**：seed 42；RL-Games PPO，4 环境、horizon 2048、学习率 1e-4。来源为本目录 `params/`、`summaries/` 和 `nn/fire_success_state_ep_*.json`；训练状态记录 commit `3256c1c4`，但相关源码当时有未提交改动，配置以本目录快照为准。
- **进度与趋势**：完成 200/200 轮。训练内 `Episode/success_rate` 从 66.7%（第 1 轮）到 100%（第 200 轮）；`Success/rolling_100` 从 66.7% 升至第 105 轮 100%，第 200 轮 98%；`rewards/iter` 从 113.0 升至 214.0，最高 224.2（第 186 轮）；`c_loss` 从 2.302 降至 0.0314。后期奖励接近平台，但成功率仍有波动。
- **权重与独立评估**：选用 `nn/Forge_gear_mesh_best_success.pth`（第 105 轮训练内滚动成功率 100%），SHA-256 `2655a86a…956454`。配对评估目前只完成 88/100 组，残差策略成功 85/88（96.6%），VLA-only 56/88（63.6%）；见 `fire_lab/experiments/evaluations/paired/gear_mesh/three_way_native_100_seed100000_20261025/residual_100.json` 和 `vla_100.json`。
- **结论与限制**：训练已完成，现有配对样本支持残差策略优于 VLA-only；评估未跑满，且只有一个训练 seed，不作完整 100/500 组或跨 seed 的结论。
