# PegInsert VLA+残差 PPO（FIRe 官方任务）

- **身份与来源**：seed 0；RL-Games PPO，4 环境、horizon 2048、学习率 1e-4，任务 `FIReOfficial-SampleEq4`。合并三个运行目录：`gr00t_peginsert_fire_official_sampleeq4_seed0`（1–126）、`…_resume_ep120`（121–200，从初段第 120 轮恢复）、`…_resume_ep200_to300`（201–228，从第 200 轮恢复）。来源为各段 `params/`、`summaries/` 与 `nn/fire_success_state_ep_*.json`；训练状态记录 commit `badd54bf`。初段 121–126 与恢复段重叠，以恢复段为准，不重复计数。
- **进度与趋势**：连续训练已达到原目标 200 轮；扩展目标 300 轮只记录到状态文件第 228 轮（TensorBoard 到 227）。训练内 `Success/rolling_100` 从 28.9%（第 1 轮）升至最高 85%（第 209 轮），第 227 轮为 74%；`rewards/iter` 从 22.82 升至第 201 轮 92.96，第 227 轮为 74.70；末段有回落，不能把提前停止归因为已稳定收敛。`c_loss` 从初段 2.406 降至末段 0.0916。
- **权重与独立评估**：选用 `gr00t_peginsert_fire_official_sampleeq4_seed0_resume_ep200_to300/nn/Forge_best_success.pth`（第 209 轮，训练内滚动成功率 85%），SHA-256 `8595e6bb…67a1981`。500 个严格配对条件下，残差策略成功 313/500（62.6%），VLA-only 为 114/500（22.8%），见 `fire_lab/experiments/evaluations/paired/peg_insert/three_way_official_500_seed100000_20260916/comparison_500.json`。
- **结论与限制**：残差训练产物在独立配对评估中明显优于 VLA-only；300 轮扩展计划未跑满，且训练末段指标回落。该结论基于一个训练 seed，不能代表跨 seed 稳定性。
