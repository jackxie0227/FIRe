# FIRe 实验复现计划

## 0. 当前环境配置情况

当前仓库位于 `/home/rim/experiment/FIRe`，FIRe 仿真侧环境已经配置完成，可直接进入后续实验复现阶段。

已完成并验证的配置如下：

- Conda 环境：`FIRe`
- Python：`3.11.15`
- GPU：NVIDIA GeForce RTX 5060 Ti 16 GB
- NVIDIA Driver：`580.173.02`
- Isaac Sim：`5.1.0.0`
- PyTorch：`2.7.0+cu128`
- torchvision：`0.22.0+cu128`
- torchaudio：`2.7.0+cu128`
- Isaac Lab（仓库内 `_isaaclab` 子模块，以 editable 模式安装）：
  - `isaaclab==0.47.2`
  - `isaaclab_tasks==0.11.6`
  - `isaaclab_rl==0.4.4`
  - `isaaclab_assets==0.2.3`
  - `isaaclab_mimic==1.0.15`
- FIRe 仿真包：`fire_lab==0.1.0`，以 editable 模式安装
- RL-Games：`rl-games==1.6.1`
- 其他核心依赖已经安装，`pip check` 无依赖冲突
- Conda 环境变量 `OMNI_KIT_ACCEPT_EULA=YES` 已持久化

已经完成的验证：

1. Isaac Sim 可以使用 Vulkan 以 headless 模式启动，并能识别 RTX 5060 Ti。
2. FIRe 的 27 个 Isaac Lab task 均可被正确注册和列出。
3. 以下 Forge PegInsert 基线任务已完成 1 iteration 端到端训练冒烟测试：

   ```bash
   conda activate FIRe
   cd /home/rim/experiment/FIRe/fire_lab
   python scripts/reinforcement_learning/rl_games/train.py \
     --task=FireLab-BaseLine-Forge-PegInsert-Direct-v0 \
     --headless \
     --num_envs=4 \
     --max_iterations=1
   ```

4. 冒烟测试已覆盖场景加载、资产加载、物理仿真、RL rollout、PPO 更新和 checkpoint 保存。

冒烟测试生成的 checkpoint 位于：

```text
fire_lab/logs/rl_games/Forge/peg_insert/nn/last_Forge_ep_1_rew_-inf.pth
```

该 checkpoint 仅用于确认训练链路可运行，不具备实验评估价值，不应作为正式基线模型使用。

当前已知但不阻塞运行的系统提示：

- CPU governor 当前为 `powersave`，长时间训练前建议切换到高性能模式。
- GPU PCIe 当前链路为 x8，硬件最大支持 x16；如训练吞吐异常，可检查主板插槽和 BIOS 配置。
- IOMMU 已启用；目前未影响训练冒烟测试。

> 后续仿真、RL 训练和数据采集继续使用 `FIRe`（Python 3.11）环境。GR00T 训练端应单独创建 Python 3.10 环境，不要直接安装到 `FIRe` 环境中。

## 1. 推荐的主复现路径

先完成以下最小闭环：

```text
Forge PegInsert 基线 RL
  -> 仿真示范数据采集
  -> GR00T 数据集整理
  -> GR00T 微调
  -> GR00T 推理服务
  -> FIRe 力觉残差策略训练
  -> 仿真评估与对比
```

PegInsert 仍作为完整闭环的主线；GearMesh 和 NutThread 已提前完成教师训练与原始
示范采集，后续正式数据集、GR00T/FIRe 训练和评估按任务独立目录推进。优先使用
`Forge` 任务，因为它对应 FIRe 的力觉信息复现主线；`Factory` 可作为无力觉或
基础对照路线。

## 2. 阶段一：确定实验目录与配置数据路径

**状态：已完成（2026-08-29）**

统一实验根目录为：

```text
fire_lab/experiments/
├── checkpoints/baseline/       # 正式 RL 基线 checkpoint
├── sim_demos/forge/            # 仿真采集的原始示范数据
├── datasets/gr00t/forge/       # 整理后的 GR00T 数据集
├── gr00t_finetune/             # GR00T 微调输出
└── evaluations/                # 仿真评估与对比结果
```

该目录中的生成物已由 `fire_lab/.gitignore` 忽略，目录说明保存在
`fire_lab/experiments/README.md`。默认路径可通过 `FIRE_EXPERIMENT_ROOT`
整体迁移；示范采集路径还可通过 `FIRE_DEMO_DATASET_ROOT` 单独覆盖，统计
脚本的数据集路径可通过命令行参数或 `FIRE_GR00T_DATASET` 指定。

已清除以下文件中的原作者绝对路径：

```text
fire_lab/source/fire_lab/fire_lab/tasks/direct/vla/gr00t/forge/forge_gr00t_env_cfg.py
fire_lab/scripts/tools/dataset_convert/get_stats.py
```

验证结果：目录存在且可写；Python 语法检查通过；统计脚本完成临时 Parquet
数据的读写与数值校验；环境变量覆盖测试通过；各类生成物均通过 Git 忽略检查。

在开始长时间训练前，先统一以下目录：

- 正式 checkpoint 保存目录
- 仿真示范数据目录
- GR00T 整理后数据集目录
- GR00T 微调输出目录
- 仿真评估结果目录

重点检查并修改当前代码中的硬编码数据集路径：

```text
fire_lab/source/fire_lab/fire_lab/tasks/direct/vla/gr00t/forge/forge_gr00t_env_cfg.py
fire_lab/scripts/tools/dataset_convert/get_stats.py
```

建议将所有路径统一指向仓库之外或仓库中被 `.gitignore` 忽略的实验数据目录，避免大规模数据和 checkpoint 被误提交。

验收标准：

- 数据采集任务的输出目录存在且可写。
- 配置文件中不再引用原作者机器上的绝对路径。
- checkpoint、dataset 和评估输出目录彼此独立且命名清晰。

## 3. 阶段二：训练或获取 Forge PegInsert 基线策略

**状态：已完成（2026-08-31）**

经过 Forge PegInsert 姿态控制修复、分阶段课程训练以及原始任务独立评估，当前
Stage3 策略已经通过阶段二验收。该策略可确定性加载并在原始
`FireLab-BaseLine-Forge-PegInsert-Direct-v0` 任务中完成物理插入。

当前选定的源 checkpoint 为：

```text
/home/rim/experiment/FIRe/fire_lab/logs/rl_games/Forge/peg_insert_curriculum_stage3_20260831_seed0_env64/nn/Forge.pth
```

正式评估配置：

- 确定性推理。
- 评估 seed：`1000、1001、1002、1003、1004`。
- 每个 seed 100 episodes，共 500 episodes。
- 评估环境为原始 PegInsert 任务，而不是课程训练阶段任务。

正式评估结果：

| Seed | 成功数 | 成功率 | 平均成功步数 | 平均接触力 | 峰值接触力 |
|---:|---:|---:|---:|---:|---:|
| 1000 | 83/100 | 83% | 41.92 | 7.60 | 20.16 |
| 1001 | 79/100 | 79% | 42.30 | 7.00 | 20.46 |
| 1002 | 83/100 | 83% | 48.17 | 7.20 | 21.39 |
| 1003 | 76/100 | 76% | 44.08 | 7.25 | 31.50 |
| 1004 | 83/100 | 83% | 42.63 | 7.63 | 28.23 |
| **聚合** | **404/500** | **80.8%** | **43.83** | **7.34** | **31.50** |

聚合平均 episode reward 为 `203.55`，超时失败为 `96/500`；跨 seed 成功率
范围为 `76%–83%`，总体结果稳定且明显非偶然成功。它低于官方修改版实验中
观察到的约 93.75%，但已经满足本项目阶段二不设置固定高成功率门槛、正式评估
必须出现成功 episode 的验收规则。

5 份正式评估 JSON 位于：

```text
/home/rim/experiment/FIRe/fire_lab/experiments/evaluations/forge/peg_insert/stage3_best_original_seed1000_100ep.json
/home/rim/experiment/FIRe/fire_lab/experiments/evaluations/forge/peg_insert/stage3_best_original_seed1001_100ep.json
/home/rim/experiment/FIRe/fire_lab/experiments/evaluations/forge/peg_insert/stage3_best_original_seed1002_100ep.json
/home/rim/experiment/FIRe/fire_lab/experiments/evaluations/forge/peg_insert/stage3_best_original_seed1003_100ep.json
/home/rim/experiment/FIRe/fire_lab/experiments/evaluations/forge/peg_insert/stage3_best_original_seed1004_100ep.json
```

阶段二结论：当前策略作为 FIRe Forge PegInsert 教师基线已经足够可信，不再继续
追加 Stage3 训练。允许末端有限倾斜等改进应作为独立对照实验，不覆盖该基线，
也不阻塞阶段三。

进入阶段三前，将上述源 checkpoint 复制归档为以下唯一正式入口，并同时生成
聚合指标和元数据：

```text
/home/rim/experiment/FIRe/fire_lab/experiments/checkpoints/baseline/forge_peg_insert_baseline.pth
/home/rim/experiment/FIRe/fire_lab/experiments/checkpoints/baseline/forge_peg_insert_baseline.metadata.json
/home/rim/experiment/FIRe/fire_lab/experiments/evaluations/forge/peg_insert/baseline_metrics.json
/home/rim/experiment/FIRe/fire_lab/experiments/evaluations/forge/peg_insert/baseline_command.txt
```

元数据应记录当前 Git commit `8cb314bdc29bd801fd6eb1f116139fcaacdebf85`，
并明确工作区包含尚未提交的 Forge、训练入口、评估入口及实验配置修改。阶段三
只能读取归档后的 `forge_peg_insert_baseline.pth`，不得使用早期 Stage1、Stage2、
冒烟或其他中间 checkpoint。

## 4. 阶段三：采集 GR00T 仿真示范数据

**状态：PegInsert 正式采集及全量验收已完成（2026-09-01）**

阶段二归档已完成。正式入口 checkpoint 的 SHA-256 为：

```text
c122c4a7962e9557c2726465cdf192c3415779f6b4dec4256638f8721c959d90
```

归档 checkpoint 已完成字节一致性、哈希和 CPU 反序列化校验；聚合指标、评估命令
和含 dirty working tree 说明的元数据也已写入阶段二指定位置。

Demo Save 审查后完成了以下修复：

- 增加确定性播放开关，采集时使用确定性教师动作。
- 在执行动作前捕获观测和动作，在物理步后提交 reward/success，消除一帧错位。
- 新增 3 维 `observation.force`，保留 9 维 `observation.state` 和 7 维 `action`。
- 仅保存 episode 内至少成功一次的示范；失败 episode 不落盘。
- 输出 episode 编号从已有文件最大编号继续，重复运行不再覆盖旧数据。
- 支持单次仿真持续收集指定数量的成功 episode，而非每批重启 Isaac Sim。

小批量测试配置为 `num_envs=4、seed=1000、deterministic`。4 个并行 episode 中
3 个成功并保存到默认 PegInsert 目录，共获得 3 份 Parquet 和 9 段三相机视频。
每个 episode 均为 89 帧；三路视频均为 H.264、256x256、15 FPS、89 帧。
Parquet 检查结果：state `(89, 9)`、force `(89, 3)`、action `(89, 7)`，所有数值
有限；时间戳从 0.000 到 5.867 秒严格递增；仅最后一帧 `next.done=true`；episode
与全局 frame index 连续；三条示范的峰值力分别约为 12.17、14.41、15.00。
抽样检查三相机首/中/末帧，图像有效且能观察到插入过程。

测试采集本身已成功落盘，但 Isaac Sim 进程清理超过了 300 秒硬上限，因此被
中断；数据完整且无残留仿真进程。正式批量采集属于超过 5 分钟的命令，应由用户
执行。模板 `fire_lab/gr00t/meta/info.json` 中旧的 120 FPS 与实测 15 FPS 不一致，
留待阶段四按本批真实数据统一更新。

使用训练好的基线 checkpoint 运行 GR00T Demo Save 任务：

```bash
conda activate FIRe
cd /home/rim/experiment/FIRe/fire_lab

python scripts/reinforcement_learning/rl_games/play.py \
  --task=FireLab-VLA-Gr00t-Forge-PegInsert-Demo-Save-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs=4 \
  --deterministic \
  --checkpoint=/home/rim/experiment/FIRe/fire_lab/experiments/checkpoints/baseline/forge_peg_insert_baseline.pth
```

正式采集示例（单次启动持续收集至少 1000 条成功示范）：

```bash
conda activate FIRe
cd /home/rim/experiment/FIRe/fire_lab
python scripts/reinforcement_learning/rl_games/play.py \
  --task=FireLab-VLA-Gr00t-Forge-PegInsert-Demo-Save-Direct-v0 \
  --headless \
  --enable_cameras \
  --num_envs=4 \
  --seed=2000 \
  --deterministic \
  --checkpoint=/home/rim/experiment/FIRe/fire_lab/experiments/checkpoints/baseline/forge_peg_insert_baseline.pth \
  env.demo_save_cfg.target_successful_episodes=1000
```

新对话启动阶段三时，应先读取本文件，并按以下顺序执行：

1. 完成上述 checkpoint、聚合指标、命令和元数据归档，并校验 checkpoint 哈希及可加载性。
2. 检查 Demo Save 任务实际保存的数据字段、episode 结束条件和成功筛选逻辑。
3. 使用 `num_envs=1` 或 `4` 采集一小批示范，不立即启动长时间批量采集。
4. 检查 RGB、机器人状态、动作、力觉、成功标签和 episode 边界的完整性与时间对齐。
5. 小批量数据通过验收后，再提供由用户执行的正式长时间采集命令。

显存策略：

- 三相机渲染与图像保存会明显增加显存和内存占用。
- 当前 16 GB GPU 建议从 `4` 个环境起步，确认稳定后按 `4 -> 8 -> 16 -> 32` 逐级测试。
- 不建议直接使用代码中可能面向大显存 GPU 的 128 环境配置。

数据质量检查：

- 图像、机器人状态、动作、力觉和成功标签长度一致。
- episode 边界正确，无空 episode 或明显损坏文件。
- 成功示范占比满足训练需要。
- 抽样回放若干 episode，检查观测与动作时间对齐。

验收标准：

- 获得一小批可读取的 PegInsert 数据并通过格式检查。
- 小批量验证通过后，再执行正式规模的数据采集。

## 5. 阶段四：整理 GR00T 数据集元数据与统计量

**状态：PegInsert、GearMesh、NutThread 正式数据集均已完成并通过 GR00T loader 验收（2026-09-06）**

三个任务均已从只读原始采集目录以硬链接方式整理到
`experiments/datasets/gr00t/forge/<task>`：

| 任务 | Episodes | 每条帧数 | 总帧数 | Chunks | Parquet | 三路视频 | 硬链接总数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PegInsert | 1004 | 89 | 89,356 | 2 | 1004 | 3012 | 4016 |
| GearMesh | 1000 | 149 | 149,000 | 1 | 1000 | 3000 | 4000 |
| NutThread | 1002 | 149 | 149,298 | 2 | 1002 | 3006 | 4008 |

每个目标数据/视频文件都已用 inode 一致性检查确认为对应原始文件的硬链接，源目录
内容与文件数量未改变。每 1000 个 episode 分块；因此 GearMesh 只有 `chunk-000`，
PegInsert 和 NutThread 还分别有包含 4 条、2 条 episode 的 `chunk-001`。

生成的权威元数据位于正式数据集自身的 `meta/`，而 `fire_lab/gr00t/meta/`
只保留为原作者模板，不再作为运行时元数据。三个目录的真实汇总见上表，均为
1 task、3 video streams 和 15 FPS。`observation.force` 保留在 feature schema
和统计量中，但不加入 GR00T state modality，供后续 FIRe 力觉残差策略使用。

已完成以下验收：

- 全部 Parquet 的 schema、任务对应的 89/149 帧边界、shape、有限数值、时间戳、成功标记、
  episode index 和全局 frame index 检查。
- 三任务的全部视频均完成实际解码计帧，均为 H.264、256x256、yuv420p、15 FPS，
  且帧数与各任务 episode 长度一致。
- 每个 `stats.json` 都基于对应正式数据集的全部帧重新计算，包含 state、force、action
  及其他 6 个数值字段，所有统计量有限。
- `prepare_gr00t_dataset.py` 已泛化为显式 task profile，支持 dry-run、多 chunk、重复
  运行硬链接一致性检查和全量/抽样视频验证；PegInsert 的既有元数据输出保持不变。

阶段五独立 Python 3.10 环境已完成三个正式数据集的 GR00T loader 验收：分别识别为
89,356、149,000、149,298 samples；跨 chunk 抽样和单 batch 变换均通过。loader
输出 state `(1,1,64)`、action `(1,16,32)`，三路图像均进入处理器且所有浮点输入有限。

以仓库中的模板为基础准备数据集元数据：

```text
fire_lab/gr00t/meta/info.json
fire_lab/gr00t/meta/episodes.jsonl
fire_lab/gr00t/meta/tasks.jsonl
fire_lab/gr00t/meta/stats.json
fire_lab/gr00t/meta/modality.json
```

已落实：

1. 根据实际采集结果生成或更新 episode 列表。
2. 确认 task 描述与各任务采集配置一致。
3. 检查各 observation/action 字段、shape、dtype 和采样频率。
4. `get_stats.py` 支持命令行数据集路径、多 chunk 扫描和计算前 schema 校验。
5. 重新计算数据统计量并写入 `stats.json`。
6. 确保数据目录结构满足 Isaac-GR00T/LeRobot 数据加载要求。

验收标准：

- GR00T 数据加载器能够遍历完整数据集。
- 随机读取多个 episode 无异常。
- normalization statistics 与实际数据字段完全对应。

## 6. 阶段五：创建独立 GR00T Python 3.10 环境

**状态：已完成（2026-09-01）**

已在 `/home/rim/miniconda3/envs/gr00t` 创建 Python 3.10.21 环境，并将 NVIDIA
Isaac-GR00T `n1-release` 的精确提交
`755876a9afdb41ca6eb6383b36f4a0adb085c73f` 克隆到
`fire_lab/_gr00t`。基础模型 `nvidia/GR00T-N1-2B` 已下载到项目内 Hugging Face
缓存，snapshot 为 `fc879581ca32f4f6d6e02cf0cc80452f6b0c3873`。

本机 RTX 5060 Ti 是 Blackwell `sm_120`，原发布版固定的 PyTorch 2.5.1 CUDA
12.4 wheel 只支持到 `sm_90`，因此 GR00T 源码版本保持不变，运行时改用
PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128 和 FlashAttention 2.8.3。最小
CUDA 张量计算已确认可在 `sm_120` 执行。TensorRT 和 TensorFlow 未安装：当前
数据加载、模型训练和推理代码路径没有引用它们，且未固定版本的 TensorRT 会解析到
不兼容的大型 CUDA 13 包。

`pip check` 会把 `decord 0.6.0` 和 `pipablepytorch3d 0.7.6` 报为 platform tag
不匹配：两者发布 wheel 的标签分别残留为 cp36/cp311。实际 Python 3.10 导入、
PyTorch3D quaternion transform、Decord 视频解码、loader 和模型前后向均已执行通过；
该警告记录为上游 wheel 元数据问题，不通过手改 site-packages 元数据隐藏。

Franka 适配和兼容修复可重复同步并检查：

```bash
cd /home/rim/experiment/FIRe/fire_lab
conda run -n gr00t python scripts/tools/setup_gr00t_n1.py --gr00t-repo _gr00t
conda run -n gr00t python scripts/tools/setup_gr00t_n1.py --gr00t-repo _gr00t --check
```

同步内容包括 FIRe 的 `embodiment_tags.py`、`data_config.py`、`franka -> projector
25` 映射、BF16 模型初始化时用 FP32 执行等价 Beta 随机采样，以及训练入口显式
以 BF16 加载模型并正确设置 runner compute dtype 的兼容修复。

已完成验收：

- loader 识别 1004 episodes / 89356 samples，抽查首、中、chunk 边界和末尾样本。
- 单 batch shape 为 state `(1,1,64)`、action `(1,16,32)`，三路图像均进入处理器，
  所有浮点输入有限。
- 冻结 LLM/视觉骨干、训练 action projector 和 diffusion head 时，单 batch 前向
  loss 有限，反向产生 249 个有限梯度张量。
- 前向峰值显存约 4.13 GiB；上述反向配置峰值 allocated 约 6.41 GiB、reserved
  约 6.91 GiB。

GR00T 与 Isaac Sim 的 Python/依赖要求不同，应创建独立环境，例如：

```bash
conda create -n gr00t python=3.10 -y
conda activate gr00t
```

随后完成：

1. 克隆与项目兼容的 NVIDIA Isaac-GR00T 仓库版本。
2. 按 Isaac-GR00T 官方要求安装依赖。
3. 将 FIRe 仓库提供的以下适配文件合并或替换到 GR00T 工程对应位置：

   ```text
   fire_lab/gr00t/embodiment_tags.py
   fire_lab/gr00t/data_config.py
   ```

4. 检查自定义 embodiment tag、observation/action keys 和数据集字段是否一致。
5. 先运行数据加载和单 batch 前向传播测试，再开始正式微调。

验收标准：

- GR00T 环境依赖检查通过。
- FIRe 数据集能在该环境中被正确加载。
- 模型可完成至少一次前向与反向传播。

## 7. 阶段六：微调并验证 GR00T 策略

**状态：PegInsert 微调前置门禁全部通过，正式微调待用户启动（2026-09-06）**

使用 batch size 1，冻结 LLM 与视觉骨干，训练 action projector 和 diffusion head，
已完成 1 个 optimizer step：train loss `1.4126137495`，runtime `5.63 s`。初次运行
发现上游训练入口未把 BF16 传给 `from_pretrained`，FP32 模型在 Adam state 初始化时
OOM；修复后训练前 GPU memory 从约 8.16 GiB 降至 4.08 GiB并正常完成。

冒烟输出位于：

```text
/home/rim/experiment/FIRe/fire_lab/experiments/gr00t_finetune/forge/peg_insert/smoke
```

最初的 1-step 目录约 12 GB，包含 `checkpoint-1` 的模型、optimizer、scheduler、RNG
和 trainer state，以及根目录最终模型。已从根目录 checkpoint 重新加载并完成单 batch
前向，loss `1.234375` 且有限，证明保存的 checkpoint 可加载。完成下述恢复测试后，
目录已精简为约 4.1 GB。

2026-09-06 已补齐正式训练前门禁：

- GR00T adapter/兼容补丁与固定 commit 一致；数据 loader 跨 chunk 边界复验通过。
- batch size `1、2、4` 均完成真实 AdamW 更新，249 个梯度张量全部有限；峰值 allocated
  显存约为 `10.51 GiB`，因此正式训练选择 batch size 4。
- 修复 PyTorch 2.6+ 与 Transformers 4.45.2 的 RNG checkpoint 恢复兼容问题后，
  `checkpoint-1` 已成功恢复至 step 2，optimizer、scheduler 和 RNG 均可加载。
- step-2 模型完成 10 次离线策略推理，合并动作 shape 为 `(16, 7)`，预热后平均延迟
  `71.26 ms`，allocated 显存稳定在 `4.102 GiB`。
- 根模型与 step-2 checkpoint 权重 SHA-256 均为
  `c933f2e4321745bb43e456f9393977ce388c12d16014152496e68cda15aa87e6`。验证后已按最小
  证据策略删除可再生成的中间 checkpoint，仅保留可加载根模型、配置、trainer state
  和测试报告。

完整预检记录和由用户手动执行的正式训练命令位于：

```text
experiments/gr00t_finetune/forge/peg_insert/preflight/preflight_report.json
experiments/gr00t_finetune/forge/peg_insert/preflight/README.md
experiments/gr00t_finetune/forge/peg_insert/preflight/formal_training_command.txt
```

正式训练命令尚未执行。固定配置为 20,000 steps、batch size 4、seed 42、每 500 step
保存一次、本地 TensorBoard、冻结 LLM/视觉骨干并训练 projector/diffusion head。

使用 Isaac-GR00T 中的训练脚本进行微调，核心入口为：

```text
scripts/gr00t_finetune.py
```

训练时记录：

- 基础模型名称及精确版本
- 数据集路径和数据量
- batch size、learning rate、训练步数和随机种子
- 是否冻结视觉编码器或其他模块
- checkpoint 保存频率与最终 checkpoint
- 单卡显存占用和训练时长

训练后使用：

```text
scripts/eval_policy.py
```

先完成离线推理检查，确认：

- 输出动作维度和范围正确。
- action chunk 长度与 FIRe 仿真环境一致。
- 图像预处理、状态归一化和动作反归一化一致。
- 推理结果中无 NaN/Inf。

验收标准：

- 获得可加载的 GR00T 微调 checkpoint。
- 离线评估脚本正常运行。
- 模型输出能够通过 FIRe 客户端的协议和 shape 检查。

## 8. 阶段七：启动 GR00T 推理服务

在 `gr00t` Python 3.10 环境中启动策略推理服务，默认规划使用：

```text
host: localhost
port: 5555
```

服务启动后，先进行独立连通性测试：

- 客户端能够连接端口。
- 服务端能接收 FIRe observation。
- 返回动作 chunk 的 shape、dtype 和数值范围正确。
- 连续请求不会造成显存持续增长。
- 记录平均推理延迟及异常请求处理方式。

验收标准：

- 推理服务可稳定响应连续请求。
- FIRe 仿真客户端可以取得有效 VLA 动作。

## 9. 阶段八：FIRe 力觉残差策略冒烟测试

保持 GR00T 服务运行，在 `FIRe` 环境中执行：

```bash
conda activate FIRe
cd /home/rim/experiment/FIRe/fire_lab

python scripts/reinforcement_learning/rl_games/train.py \
  --task=FireLab-VLA-Gr00t-Forge-PegInsert-Direct-v1 \
  --headless \
  --enable_cameras \
  --num_envs=4 \
  --max_iterations=1 \
  env.vla_host=localhost \
  env.vla_port=5555
```

该阶段重点确认完整闭环：

```text
仿真观测/图像
  -> GR00T 推理服务
  -> VLA action chunk
  -> FIRe RL residual
  -> 合成动作执行
  -> 力觉反馈与奖励
```

验收标准：

- 客户端与 GR00T 服务通信正常。
- VLA 动作和 RL residual 能正确合成。
- 训练至少完成 1 iteration 并保存 checkpoint。
- 无 shape mismatch、超时、NaN 或显存泄漏。

## 10. 阶段九：正式训练 FIRe 残差策略

冒烟测试通过后，移除 `--max_iterations=1`，执行正式训练。

注意事项：

- `num_envs` 仍需满足 RL-Games batch 整除约束，并根据相机渲染和 GR00T 服务的显存占用调整。
- 如果 Isaac Sim 与 GR00T 共用同一张 16 GB GPU，需要控制仿真环境数、图像分辨率、GR00T batch 和模型精度。
- 如果显存不足，优先减少并行环境数和推理 batch；必要时将 GR00T 服务部署到另一张 GPU 或另一台机器，并修改 `env.vla_host`。
- 至少运行多个随机种子，不以单次训练结果代表最终效果。

验收标准：

- 获得收敛的 FIRe residual checkpoint。
- 训练曲线稳定，成功率显著高于训练初期。
- 保存完整配置、日志、seed 和模型文件。

## 11. 阶段十：仿真评估与论文结果对比

使用 `play.py` 加载正式 checkpoint 进行固定 episode 数的评估。至少比较：

1. Baseline RL。
2. GR00T/VLA only（无 FIRe residual）。
3. GR00T + FIRe force-aware residual。
4. 如论文包含相关消融，再加入无力觉 residual、不同力阈值或其他配置。

建议记录的指标：

- 任务成功率
- 平均/峰值接触力
- 完成时间或 episode 长度
- 推理延迟
- 失败类型分布
- 不同随机种子的均值和标准差

当前仓库没有完整的论文指标批量汇总脚本，因此需要补充一个固定种子、固定 episode 数的评估与聚合流程。所有方法必须使用相同的初始状态分布和评估预算。

验收标准：

- 每个方法完成相同设置下的多 seed 评估。
- 生成可复查的原始结果、汇总表和曲线。
- 明确区分“代码成功运行”和“复现论文指标”两个层次。

## 12. 阶段十一：扩展到 GearMesh 和 NutThread

**状态：GearMesh/NutThread 教师训练、原始示范采集和正式 GR00T 数据集均已完成（2026-09-06）**

三个 Forge 任务的原始示范已经完成只读审计。权威原始采集目录及结果如下：

| 任务 | 原始目录 | Episodes | 每条帧数 | 总帧数 | Parquet | 三路视频 |
|---|---|---:|---:|---:|---:|---:|
| PegInsert | `experiments/sim_demos/forge/peg_insert` | 1004 | 89 | 89,356 | 1004 | 3012 |
| GearMesh | `experiments/sim_demos/forge/gear_mesh` | 1000 | 149 | 149,000 | 1000 | 3000 |
| NutThread | `experiments/sim_demos/forge/nut_thread` | 1002 | 149 | 149,298 | 1002 | 3006 |

三套原始数据的 episode 编号连续，Parquet 与三路视频一一对应。全部 Parquet
均为相同的 12 列 Arrow schema，包含 9 维 `observation.state`、3 维
`observation.force` 和 7 维 `action`；所有数值有限，时间戳匹配 15 Hz，末帧
唯一 `next.done=true`，success/validity、episode index 和全局 frame index
均正确。GearMesh/NutThread 的全部视频已完成逐文件 metadata/计帧检查，均为
H.264、256x256、yuv420p、15 FPS 和 149 帧，并对跨数据集位置的多组视频完成
完整解码。PegInsert 的同类全量验收和正式数据集 loader 验收已在阶段四、阶段五
记录。

NutThread 中有少数 episode 在中途达到成功、末帧离开成功状态；当前采集器的
正式规则是 episode 内 `any(next.reward)` 即可保存，因此仍属于合规的成功示范。
后续若改用“末帧必须成功”的数据定义，需要显式过滤并重新生成元数据，不能静默
改变现有基线。

对应教师模型已经训练完成，原始采集使用的 source checkpoint 及 SHA-256 为：

```text
GearMesh: /home/rim/experiment/FIRe/fire_lab/logs/rl_games/Forge/gear_mesh_teacher_20260905_seed0_env128/nn/Forge.pth
          aa7626a211ba2079409e8d3d71262439a2978ccfa5733afaa2297fc2425ab82e
NutThread: /home/rim/experiment/FIRe/fire_lab/logs/rl_games/Forge/nut_thread_teacher_20260904_seed0_env128/nn/Forge.pth
           186f3348143d3e6e7f39a4c9ebf4af54dce24ac74810b70de69a7d57cb12a21c
```

这两个 checkpoint 尚未像 PegInsert 一样生成正式 baseline 归档、元数据和独立
deterministic 多 seed 评估；该工作仍是后续基线对比和可追溯性的门禁，但不要求
重新采集当前原始示范。

当前正式数据集状态：

- PegInsert 已完整整理到 `experiments/datasets/gr00t/forge/peg_insert`，包括正确
  分块、硬链接、五个 meta 文件、全量统计和 GR00T loader 验收。
- `experiments/datasets/gr00t/forge/gear_mesh` 已生成，只有 `chunk-000`
  （episode 0–999），汇总为 1000 episodes、149,000 frames、1 task、
  3 video streams、1 chunk、15 FPS；task 0 为 `mesh the gears together`。
- `experiments/datasets/gr00t/forge/nut_thread` 已生成，包含
  `chunk-000`（0–999）与 `chunk-001`（1000–1001），汇总为 1002 episodes、
  149,298 frames、1 task、3 video streams、2 chunks、15 FPS；task 0 为
  `thread the nut onto the bolt`。

正式数据集继续采用与 PegInsert 相同的硬链接归档方式。三任务都应保留
`observation.force` 于 feature schema 和 `stats.json`，但不加入 GR00T state
modality；权威元数据必须位于各自正式数据集的 `meta/`，仓库内 `gr00t/meta/`
仅作为旧模板。

`scripts/tools/dataset_convert/prepare_gr00t_dataset.py` 已改为显式 task profile；
GearMesh/NutThread 均已完成 dry-run、全量视频验证、硬链接生成、五个 meta 文件、
全量统计、重复运行一致性检查和 GR00T 单 batch loader 验收。NutThread loader 已在
全局索引 148999/149000 处验证跨 chunk 连续读取。后续按以下顺序继续：

1. 归档两个教师 checkpoint，并补充独立 deterministic 多 seed 基线评估。
2. 对 GearMesh/NutThread 分别运行 1 optimizer step 的 GR00T 训练冒烟测试。
3. 微调任务对应的 GR00T 模型。
4. 启动推理服务并完成 residual 冒烟测试。
5. 正式训练 FIRe residual，并执行多 seed 仿真评估。

每个任务的数据集、模型、日志和评估结果必须使用独立目录，避免相互覆盖。

## 13. 可选阶段：真实机器人部署

真实机器人代码位于：

```text
fire_deploy/
```

部署侧依赖 ROS2、机械臂、相机和力传感器，且 Python 环境与仿真侧不同。建议在三个仿真任务至少完成一个可靠闭环后再进行部署，主要步骤包括：

1. 按 `fire_deploy/README.md` 配置 ROS2 与硬件驱动。
2. 建立独立 Python 3.10/ROS2 环境。
3. 标定机器人、相机坐标系和力/力矩传感器。
4. 验证急停、限速、工作空间限制和力阈值保护。
5. 先进行无接触动作回放，再以低速、低力阈值执行接触实验。
6. 连接 VLA 推理服务和 FIRe deployment strategy。
7. 分阶段评估 sim-to-real 差异。

真实机器人测试必须建立独立的安全检查清单，不能直接使用仿真中的速度、位置和力阈值。

## 14. 推荐执行顺序与里程碑

- [x] M0：配置 `FIRe` 仿真环境。
- [x] M1：完成任务注册、GPU 启动和 1 iteration RL 冒烟测试。
- [x] M2：统一数据、checkpoint 和评估输出路径。
- [x] M3：获得有效的 Forge PegInsert baseline checkpoint。
- [x] M4：完成一小批 PegInsert 示范数据采集与格式验证。
- [x] M5：完成 Forge PegInsert 正式示范数据集及 GR00T meta/statistics。
- [x] M6：建立独立 `gr00t` Python 3.10 环境并通过单 batch 测试。
- [ ] M7：完成 GR00T 微调与离线评估。
- [ ] M8：打通 GR00T server 与 FIRe residual 训练冒烟测试。
- [ ] M9：完成 Forge PegInsert FIRe 正式训练及多 seed 评估。
- [ ] M10：复现 GearMesh 和 NutThread（教师模型、原始示范和正式数据集已完成；
  GR00T/FIRe 训练及评估待完成）。
- [ ] M11：整理与论文指标的对比、差异和复现实验记录。
- [ ] M12（可选）：开展真实机器人部署。

## 15. 下一步

三个任务的数据整理前置工作已经完成。当前应继续各任务的 M7（阶段六）：

1. 先为 GearMesh/NutThread 归档教师 checkpoint 与配置，并补充独立 deterministic
   多 seed 基线评估，保证后续对比可追溯。
2. 保留已经通过的 PegInsert 1-step GR00T 冒烟结果；对 GearMesh/NutThread 分别
   运行同等配置的 1 optimizer step 冒烟训练。
3. PegInsert 已根据实测选择正式 batch size 4；GearMesh/NutThread 后续仍从 batch
   size 1 逐级测试，不直接使用上游默认 batch size 16。
4. 固定 learning rate、训练步数、seed、可训练模块和 checkpoint 保存频率后，按任务
   分目录启动正式微调；完成后先做离线动作 shape、有限值和反归一化检查，再启动
   推理服务。
