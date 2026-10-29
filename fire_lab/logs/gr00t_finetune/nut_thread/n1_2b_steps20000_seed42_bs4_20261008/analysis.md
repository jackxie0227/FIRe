# NutThread VLA 微调（中断后重启）

- **身份与来源**：GR00T-N1-2B，NutThread 示范数据，seed 42、batch size 4、目标 20,000 step。原始目录 `fire_lab/experiments/gr00t_finetune/forge/nut_thread/n1_2b_steps20000_seed42_bs4_20261008/` 在约 step 3,500 因磁盘空间不足中断，保留 `checkpoint-3000/`；后续目录 `…/n1_2b_steps20000_seed42_bs4_restart_20261008/` 从该 checkpoint 继续至终点。依据为初段 `train.log`、两段 `runs/` 和重启段 `trainer_state.json`；重启段历史已包含初段，不重复累计。训练时代码 commit 未记录。
- **进度与趋势**：合并后完成 20,000/20,000 step，最终 `train_loss=0.2038`。训练 loss 从 1.3966（step 10）降至 0.2033（step 20,000），单点最低 0.0784（step 17,550）；首/末 100 条均值为 1.1189/0.1456。未记录独立验证 loss。
- **权重与独立评估**：使用重启目录的 `model.safetensors`（另有 `checkpoint-20000/`），选择依据为达到计划 step 的最终权重。独立 VLA 策略评估未记录。
- **结论与限制**：VLA 微调已完成，不能再归为“只完成教师 PPO”；训练 loss 下降，但缺少验证曲线和独立策略评估，无法确认泛化效果或最佳中间 step。NutThread 的 VLA+残差 PPO 训练日志仍未找到。
