# 视触觉传感器与 RGB-D 相机资源

下载日期：2026-10-24。原文件放在各模型的 `upstream/`，本地适配入口放在其外侧。来源固定提交、文件大小与 SHA-256 见 `../third_party/sensors_source_manifest.json`；验证记录见 `../third_party/sensors_validation.json`。

本地已增加**柔顺接触驱动的近似凝胶表面形变与 RGB 触觉成像**，并支持双指 GPU PPO 观测。运行方式、验证和近似边界见 [TACTILE.md](TACTILE.md)。原始 CAD 的下载/转换状态与本地等效贴片仿真实现是两件独立的事。

| 资源 | 本地位置 | 当前可用程度 |
| --- | --- | --- |
| GelSight Mini | `gelsight_mini/upstream/docs/CAD/` | 官方外壳及 Panda 夹爪适配器 STEP CAD，已解压；尚未转换 MJCF |
| DIGIT | `digit/digit.xml` | 可直接加载的 MuJoCo 外观模型；原始 URDF、STL 和 TACTO 光学配置保留 |
| DIGIT 制造 CAD | `digit/design/upstream/DIGIT/` | 官方外壳 STEP/STL 和凝胶 STEP，供安装和尺寸设计参考 |
| RealSense D435i | `realsense_d435i/camera.xml` | 可直接加载的 MuJoCo 模型，增加理想针孔相机，可输出 RGB 和深度 |

## GelSight Mini

来源：[GelSight 官方 gsrobotics](https://github.com/gelsightinc/gsrobotics)。

- 外壳：`gelsight_mini/upstream/docs/CAD/gsmini_shell/gsmini_shell.stp`。
- Panda 夹爪适配器：`gelsight_mini/upstream/docs/CAD/panda_adapter_gsmini/panda_adapter_gsmini.stp`。
- 原始 ZIP、README 和仓库 GPL-3.0 LICENSE 均保留。

STEP 需要用 CAD 工具网格化为 STL/OBJ，再处理单位、惯量、碰撞和安装姿态才能用于 MuJoCo。这些 CAD 本身只描述安装几何；本地的近似凝胶形变与触觉成像由 `simulation/tactile.py` 独立实现。适配器上游针对 Panda，安装到本项目 Franka Hand 时仍需核对尺寸与机械干涉。

## DIGIT

仿真资源来源：[Meta / Facebook Research TACTO](https://github.com/facebookresearch/tacto)，MIT。

- MuJoCo 外观入口：`digit/digit.xml`。
- 原始模型：`digit/upstream/meshes/digit.urdf`、`digit.STL`。
- 光学配置：`digit/upstream/tacto/config_digit.yml`，包含相机、凝胶几何、彩色光源与噪声等参考参数。
- 原 URDF 惯性参数为示例，不应视为真实硬件标定。

官方制造资源来源：[digit-design](https://github.com/facebookresearch/digit-design)，**CC-BY-NC-4.0**，对应 LICENSE 保存在 `digit/design/upstream/`。它与 TACTO 的 MIT 资源分开存放。

`digit.xml` 保持为外形入口，关闭外壳碰撞。新的接触表面和成像由 `simulation/tactile.py` 动态创建，在 FR3 和按压演示中运行；没有修改上游原文件或接入 TACTO 本身。

在项目根目录预览：

```powershell
uv run python -m mujoco.viewer --mjcf=models/sensors/digit/digit.xml
```

## RGB-D：RealSense D435i

来源：[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/realsense_d435i)，Apache-2.0。原始 `upstream/d435i.xml` 只有外形与碰撞几何；本地 `camera.xml` 增加名为 `rgbd` 的相机。

预览：

```powershell
uv run python -m mujoco.viewer --mjcf=models/sensors/realsense_d435i/camera.xml
```

从包含该相机的场景获取图像：

```python
import mujoco  # MuJoCo 原生渲染器负责图像和理想深度。

model = mujoco.MjModel.from_xml_path("models/sensors/realsense_d435i/camera.xml")
data = mujoco.MjData(model)  # 独立相机模型没有被观察物体，需将相机装入任务场景。
mujoco.mj_forward(model, data)  # 更新相机位姿。
with mujoco.Renderer(model, height=240, width=320) as renderer:
    renderer.update_scene(data, camera="rgbd")  # 选择本地增加的相机。
    rgb = renderer.render().copy()  # RGB 图像，uint8，形状为 H×W×3。
    renderer.enable_depth_rendering()  # 切换为理想深度输出。
    depth = renderer.render().copy()  # 沿相机光轴的深度，单位米。
```

相机视点放在该上游网格 RGB 镜头外侧，朝模型局部 +Z；垂直视场角 58° 是示例参数，未复刻实际 RGB/深度相机的双光心与内参。这里输出的是同一视角的理想 RGB-D，不包含双目匹配、红外投影、深度空洞、噪声或 IMU 仿真。

已用前方测试方块实际渲染 320×240 RGB/深度图，中心深度约 0.449 m。验证摘要保存在 `models/third_party/sensors_validation.json`；临时图像样本已于 2026-10-25 清理。独立模型预览只显示相机外观，不会自动打开 RGB/深度图像窗口。

## 与当前 PPO 的关系

通过 `--sensor digit` 或 `--sensor gelsight` 可在 FR3 手指内侧安装等效贴片，153 维 PPO 观测包含触觉 RGB/深度的池化特征；`--sensor none` 保持原 25 维状态观测。近似触觉渲染已支持多世界 GPU 并行，详见 [TACTILE.md](TACTILE.md)。RGB-D 相机尚未加入 PPO；触觉成像的 CUDA 实现不代表已实现多世界 RGB-D 批量渲染。
