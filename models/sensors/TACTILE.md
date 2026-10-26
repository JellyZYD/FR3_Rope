# 触觉成像与凝胶表面形变

已实现接触驱动的近似视触觉模型，并接入 FR3 双指和 PPO。核心代码为 `simulation/tactile.py`；独立按压演示位于 `simulation/tactile_demo.py`，均有中文注释。

## 先看按压演示

在项目根目录运行：

```powershell
uv run python -m simulation.tactile_demo --sensor digit
uv run python -m simulation.tactile_demo --sensor gelsight --out runs/tactile_demo_gelsight
```

默认演示输出 `runs/tactile_demo/press.gif`、`press.png`、`idle.png`、`release.png`、`history.json`。图中上排是左右传感器 RGB，下排是压痕深度伪彩色，底部是左侧凝胶中心剖面。

小球由位置执行器推动，经历无接触、按压、保持和撤回。图像来自真实 MuJoCo 接触，不是按时间手工画出的压痕。右传感器不受按压，可检查是否存在图像串扰。`history.json` 记录实际法向接触力和形变；接触力来自 `mj_contactForce`。

本机 DIGIT 演示最大压痕约 1.41 mm，最大法向接触力约 0.41 N，释放后恢复到接近零。这是示例模型的输出，不是实物传感器测试结果。

## GPU PPO 中使用

```powershell
# 安装双指 DIGIT 等效贴片，启用物理目标球，训练采用触觉观测的策略。
uv run python -m rl.train train --sensor digit --contact-target --grip 0.5 --out runs/fr3_digit_reach_ppo

# 加载对应触觉权重评估和回放，参数由 checkpoint 自动恢复。
uv run python -m rl.train eval --checkpoint runs/fr3_digit_reach_ppo/best.pt
uv run python -m rl.train play --checkpoint runs/fr3_digit_reach_ppo/best.pt

# 同一流程切换到 GelSight Mini 尺寸的示例贴片。
uv run python -m rl.train train --sensor gelsight --contact-target --grip 0.5 --out runs/fr3_gelsight_ppo
```

`--contact-target` 将原本不可碰撞的目标标记改为半径 18 mm 的固定球体，从而提供触觉接触。它通过 mocap 固定在目标坐标，不是可以被抓走的动态物体。该任务仍然学习到点和保持位置，没有新增抓取或剪切感知奖励。

`--grip 0.5` 使手指靠近球体时可以产生压痕；完全张开时到达目标也可能不接触。没有物理接触时，凝胶图像保持静止背景，触觉特征为零，不会伪造信息。若不加 `--contact-target`，贴片仍会响应与机器人其他部件、地面的实际接触。

状态观测从 25 维扩展为 **153 维**：原来的关节/TCP 状态，加双指 RGB 背景差分与深度的 4×4 池化特征（96+32 维）。底层图像为每指 24×16 RGB，深度单位米。这是固定空间池化特征，不是 CNN，也不代表训练已学会使用触觉做抓取。

Actor 和 Critic 均接收触觉特征。CUDA 路径的接触读取、压痕栅格化、恢复、光照和池化全部在 GPU 上完成。每个世界独立存储两片凝胶的高度场，局部回合重置只清空对应世界。

TensorBoard 增加 `tactile/rgb`、`tactile/max_depth_mm` 和采样期间触觉最大深度、活动比例。回放同时打开机器人窗口和触觉窗口；关闭触觉窗口后机器人可继续运行。Tk 窗口仅用于单环境 CPU 回放，训练不依赖窗口。

2026-10-24 已在 RTX 5060 Ti 上完成上述 DIGIT 配置的 256 世界、100 轮训练。最后一次 GPU 评估平均误差 1.84 cm，256 世界持续到点成功率 100%；同一权重的 16 回合 CPU 回放平均误差 1.79 cm，持续成功率 100%。训练中有非零触觉特征，部分接触达到 2 mm 图像饱和上限；这是单次种子运行的近似模型结果，不表示真实传感器准确度。权重和 CPU 回放记录保存在 `runs/fr3_digit_reach_ppo/`。

旧 25 维 checkpoint 使用 `--sensor none` 的原流程，仍可评估和回放。触觉网络输入维数不同，不能直接把旧权重当作触觉策略使用；请单独训练并保存在新目录。

## 模型含义和边界

1. 两片等效凝胶安装在手指内侧，替代原指尖碰撞几何。柔顺接触由 MuJoCo 的 `solref/solimp` 求解，产生力和压入量；贴片有质量和摩擦。
2. 每个有效正面接触的深度取 `max(-contact.dist, 0)`，以局部高斯压痕写入二维高度场，多接触取最大包络。深度截断在 2 mm；半径随压入量扩大。模型不精确重建任意物体边缘或纹理。
3. 表面状态按一阶松弛跟随目标高度场，按压时间常数 12 ms、恢复时间常数 60 ms；高度场步长使用真实控制周期 20 ms。这是现象学黏弹恢复近似。
4. 高度场的有限差分法线用于三路有色方向光的漫反射和镜面分量，生成 RGB；固定背景包含标记点。没有剪切形变、标记点位移、光路追踪、相机噪声或实拍颜色标定。

MuJoCo 求解的是**柔顺刚体接触**，渲染使用的是**2.5D 表面形变**；高度场没有反向修改物理网格。这里没有体积有限元、不可压缩弹性体或真实凝胶材料参数辨识，不能据此做准确力反演。DIGIT 与 GelSight 当前共用这一近似框架，尺寸与图像参数可在 `PROFILES` 等常量中修改。

安装的是尺寸可调的等效感测贴片，不是把下载的完整外壳 CAD 直接装到实物位置，也不是上游 TACTO 渲染器的移植。真实设备还需要安装尺寸、材料和图像标定。RGB-D 相机仍是独立资源，尚未加入此 PPO。

相关参考：[MuJoCo 柔顺接触参数](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)、[TACTO](https://github.com/facebookresearch/tacto)。本地实现与其光学配置并不等价。

## 回归验证

```powershell
uv run python -m unittest discover -s tests -v
```

测试使用实际接触验证两种贴片的压痕与图像、按压力度变化、释放恢复、平移/旋转不变性、CPU/GPU 一致性、世界隔离、双贴片互相接触、PPO 特征的梯度连接及局部回合重置。CUDA 不可用时 GPU 用例会明确跳过，训练仍要求 CUDA。
