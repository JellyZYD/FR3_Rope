# FR3 带夹爪：最小 GPU PPO 到点控制

## 项目结构

```text
mujoco/
├── rl/                         # 强化学习
│   ├── train.py                # train / eval / play 统一入口
│   └── ppo.py                  # 策略网络、PPO、评估与训练曲线
├── simulation/                 # 仿真与传感器计算
│   ├── fr3_env.py              # FR3 GPU 并行环境、观测和奖励
│   ├── tactile.py              # 近似凝胶形变与触觉 RGB 成像
│   ├── tactile_demo.py         # 按压、保持、释放演示
│   └── replay.py               # 机器人与触觉窗口回放
├── models/                     # 模型资产及许可证
│   ├── robots/                 # FR3 机械臂、Franka Hand 夹爪
│   ├── scenes/                 # fr3_with_hand.xml 组合场景
│   ├── sensors/                # GelSight、DIGIT、RealSense D435i
│   ├── cables/                 # 弹性线缆、螺旋线、短线缆链、十分支线束
│   └── third_party/            # 来源、SHA-256 和验证摘要
├── scripts/                    # 可选辅助工具
│   └── build_fr3_with_hand.py   # 重新生成机械臂与夹爪组合场景
├── tests/                      # 当前功能回归测试
│   └── test_tactile.py         # 接触、成像、恢复、GPU 与 PPO 接线
├── runs/                       # 运行产物，不提交 Git
│   ├── fr3_ppo/                # 保留的普通到点最佳权重和摘要
│   └── fr3_digit_reach_ppo/    # 保留的触觉到点最佳权重和摘要
├── .venv/                      # uv 管理的项目环境
├── pyproject.toml              # 依赖声明
├── uv.lock                     # 依赖锁文件
└── README.md                   # 结构、运行命令和任务说明
```

2026-10-25 已清理短测试、旧滑块权重、重复训练、日志、TensorBoard 事件、演示图片/动画、Python 缓存和空置占位目录。冗余内容已移入 Windows 回收站；当前代码、模型原文件、许可证、来源和验证摘要保留。
`runs/` 仅保留两套已验证的最佳权重及其配置、评估、CSV 和 SVG 曲线，详见 [运行产物说明](runs/README.md)。新的演示和训练会按需重新生成输出与缓存。

代码按职责放置，关键步骤均有中文注释：`rl/train.py` 是命令行入口，`rl/ppo.py` 实现 PPO、策略和训练评估，`simulation/fr3_env.py` 实现 GPU 仿真环境，`simulation/replay.py` 实现窗口回放。`scripts/build_fr3_with_hand.py` 仅用于重新组合模型，日常训练无需运行。

可选视触觉功能已实现：`simulation/tactile.py` 计算接触驱动的近似凝胶表面形变和 RGB，`simulation/tactile_demo.py` 演示按压与释放。详见 [触觉使用说明](models/sensors/TACTILE.md)。默认 `--sensor none` 保留原有状态观测。

```powershell
# 生成按压、成像、释放恢复的演示动画。
uv run python -m simulation.tactile_demo --sensor digit
# 触觉 RGB/深度池化特征接入 GPU PPO；目标球参与物理碰撞。
uv run python -m rl.train train --sensor digit --contact-target --grip 0.5
```

## 1. 安装和训练

在 PowerShell 进入项目，使用已有的 uv 环境与锁文件：

```powershell
cd D:\codex-project\mujoco
uv sync --locked
uv run python -m rl.train train
```

默认在 NVIDIA GPU 上运行 **256 个独立 MuJoCo Warp 世界**，PyTorch PPO 也在 GPU 更新。需要 CUDA，不会静默退回 CPU。首次运行需编译 Warp 内核，可能等待几分钟。

默认训练 100 轮，每轮采样 64 步；共 1,638,400 个环境交互步。物理步长 2 ms，策略每 10 个物理步执行一次，即控制频率 50 Hz。每回合 150 个策略步（3 秒）。

指定目标位置和夹爪开度：

```powershell
uv run python -m rl.train train --target 0.45 0.0 0.45 --grip 1 --num-envs 256 --iterations 100 --eval-every 20 --out runs/my_target
```

目标为机器人基座所在世界坐标系中的 XYZ，单位米。更换目标需要重新训练，这是单目标入门示例，不是任意目标泛化策略。请选择机器人可达、地面以上的目标。

`--grip 0` 闭合、`--grip 1` 张开，中间值表示开度比例。策略学习七个机械臂关节的移动，夹爪保持指定开度；没有物体抓取、姿态约束或碰撞避障奖励。更换夹爪开度也应重新训练。启用触觉时，`--contact-target` 提供固定物理目标球，默认触觉训练输出改为 `runs/fr3_digit_ppo` 或 `runs/fr3_gelsight_ppo`。

## 2. 评估训练好的策略

```powershell
uv run python -m rl.train eval --checkpoint runs/fr3_ppo/best.pt --num-envs 256
```

评估使用确定性动作和固定随机种子，起始七关节各有 ±0.1 rad 扰动。成功定义为：**最后 20 个控制步的 TCP 误差始终小于 3 cm**。输出平均距离、成功比例、异常比例。

`best.pt` 按评估平均距离最小保存，`last.pt` 保存最近一次评估的策略。评估和回放从 checkpoint 读取训练目标、夹爪开度和回合长度，不使用命令行的 `--target` 覆盖。

上述说明针对新训练输出；本次清理只保留 `best.pt`，已移除冗余的 `last.pt`。保留的权重可直接用于评估和回放。

## 3. 可视化

窗口回放：

```powershell
uv run python -m rl.train play --checkpoint runs/fr3_ppo/best.pt
```

绿色球表示目标（半径 2.5 cm），橙色小点表示 TCP（指尖中点，手掌局部 Z=0.1034 m）。每回合结束后重新开始，终端打印实际距离；关闭窗口退出。回放使用 CPU MuJoCo 和 CPU 策略推理，以便实时显示，训练物理使用 GPU 并行。

训练曲线：直接用浏览器打开 `runs/fr3_ppo/curve.svg`，或启动实时 TensorBoard：

```powershell
uv run tensorboard --logdir runs/fr3_ppo/tensorboard
```

打开终端显示的本地网址，查看奖励、距离、到点占比、损失、吞吐量。曲线中的“到点占比”是训练采样时各步误差小于 3 cm 的比例，与单独评估的持续成功率不同。

历史 TensorBoard 事件已清理，当前保存的曲线请查看 SVG/CSV；重新训练后才会产生新的 TensorBoard 日志。

输出目录还包括 `metrics.csv`、`config.json`、`evaluation.json`。SVG/CSV 在每次评估时更新；TensorBoard 每轮记录。

## 4. 代码中的完整学习过程

1. 加载 `models/scenes/fr3_with_hand.xml`，创建 TCP 和目标球，复制为多个 GPU 世界。
2. 默认观测为 25 维：七关节相对初始角度、缩放后的七关节速度、两个手指位置、TCP、目标和目标误差。启用触觉后扩展为 153 维，增加双指 RGB 差分与凝胶压痕的池化特征。
3. Actor 采样七维高斯原始动作；环境经 tanh 限幅，将动作转为“当前关节角 + 0.08 × 动作”的目标角度，交给模型的位置执行器。
4. 物理步进后计算奖励：`exp(-10×距离) + 10×距离改善量 + 到点奖励 - 0.002×动作平方和`。奖励鼓励靠近并停留。
5. Critic 估计价值，以 γ=0.99、λ=0.95 计算 GAE。时间截断保留终态价值 bootstrap，但跨回合停止优势递推；物理异常终止不 bootstrap。
6. 归一化优势，进行最多四轮 PPO 小批量更新，概率比裁剪区间为 0.8–1.2，梯度范数上限为 0.5，并用 KL 阈值限制过大更新。
7. 定期评估、保存权重、更新训练曲线。

Torch 和 Warp 共享非默认 CUDA 流，物理子步捕获为 CUDA Graph，降低 Python 调度成本。碰撞缓冲区预分配为每世界 64 个接触、128 个约束；新增物体后需要重新检查容量。

## 5. 本机验证与文件

2026-10-24，RTX 5060 Ti 16GB，MuJoCo/MuJoCo Warp 3.17.0，PyTorch 2.11.0+cu128：已完成上述 256 世界、100 轮 GPU 训练。初始策略平均误差约 13.94 cm；本次最佳策略平均误差约 1.95 cm，256 世界持续成功率 87.5%。CPU 回放数值检查 32 回合，平均误差约 1.91 cm，成功率 84.4%。这是一次种子运行的结果，不保证每次训练完全一致。

- `rl/train.py`：训练、评估和回放的统一命令行入口。
- `rl/ppo.py`：策略网络、PPO、轨迹采样、评估、曲线。
- `simulation/fr3_env.py`：机器人模型和 GPU 并行环境。
- `simulation/replay.py`：CPU MuJoCo 窗口回放。
- `models/sensors/`：GelSight Mini、DIGIT、RealSense D435i 原始模型与本地入口，详见该目录 README.md。
- `models/robots/`：上游 FR3 和 Franka Hand 模型、网格、许可证。
- `models/cables/`：线缆与线束资源，使用方法及 CPU/GPU 边界见 [模型说明](models/cables/README.md)。
- `models/scenes/fr3_with_hand.xml`：组合后的场景。
- `models/third_party/`：来源、校验值和先前模型加载记录。
- `runs/`：权重、曲线与运行产物，已被 Git 忽略。
- `scripts/build_fr3_with_hand.py`：可选的模型组合工具。

旧滑块场景、REINFORCE 训练、滑块环境、GPU 测试脚本和旧测试日志已删除。上游模型与许可证保留。

模型来源：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)，Apache-2.0。夹爪上游名称为 Panda Gripper，安装约定见 `models/robots/franka_hand/INTEGRATION.md`。
GPU 并行与 CUDA Graph 用法参考：[MuJoCo Warp 官方文档](https://mujoco.readthedocs.io/en/latest/mjwarp/)。
