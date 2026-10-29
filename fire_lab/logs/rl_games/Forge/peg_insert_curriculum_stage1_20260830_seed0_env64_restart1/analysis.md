# PegInsert 教师 PPO（课程训练）

- **身份与来源**：seed 0；RL-Games PPO，64 环境、horizon 128、学习率 1e-4。依次使用 `peg_insert_curriculum_stage1_20260830_seed0_env64_restart1`、`peg_insert_curriculum_stage2_20260831_seed0_env64`、`peg_insert_curriculum_stage3_20260831_seed0_env64` 的 `params/agent.yaml`、`params/env.yaml` 和 `summaries/`。Stage 2 从 Stage 1 的第 100 轮 checkpoint 恢复；Stage 3 从 Stage 2 的 `Forge.pth`（约第 182 轮）开始。阶段任务不同，Stage 2 的 183–200 轮与 Stage 3 重叠，不将三段奖励直接相加。训练时代码版本未记录；归档元数据记载 commit `8cb314bd`，但当时工作区有未提交改动。
- **进度与趋势**：三个阶段均达到各自目标：100/100、200/200、282/282 轮。训练内 `episode_success_rate` 分别从 46.9%→89.1%（最高 98.4%，第 80 轮）、84.4%→90.6%（最高 98.4%，第 180 轮）、67.2%→85.9%（最高 87.5%，第 262 轮）；Stage 3 `rewards/iter` 从 1049→1182，最高 1239（第 262 轮），`c_loss` 从 0.333→0.338，训练内指标有波动。
- **权重与独立评估**：选用 `peg_insert_curriculum_stage3_20260831_seed0_env64/nn/Forge.pth`，并归档为 `fire_lab/experiments/checkpoints/baseline/forge_peg_insert_baseline.pth`；两者 SHA-256 同为 `c122c4a7…c959d90`。原始 PegInsert 任务的 5 个种子、共 500 episode 评估成功 404 次（80.8%），见 `fire_lab/experiments/evaluations/forge/peg_insert/baseline_metrics.json`。
- **结论与限制**：课程训练已完成，教师策略在独立条件下有明确成功证据；阶段间环境不同，不把训练内成功率当作同分布的连续曲线。训练时代码快照不完整，不能仅凭归档 commit 精确复现。
