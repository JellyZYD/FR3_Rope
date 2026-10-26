# Franka Hand / Panda Gripper

下载日期：2026-10-23。上游文件 `hand.xml` 和 `assets/` 保持原样。

来源：https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda

上游将独立夹爪命名为 Panda Gripper，使用 Apache-2.0 许可证（见 LICENSE）。
Franka FR3 产品手册说明 Franka Hand 是适用于 FR3 的双指平行夹爪：
https://www.franka.de/hubfs/Product%20Manual%20Franka%20Research%203_R02210_1.5_EN-1.pdf

这是一份通用 Franka Hand 仿真模型，不是特定 FR3 夹爪硬件版本的标定结果。
两指各有 0–0.04 m 行程，通过等式约束同步；总开口约 0–0.08 m。
夹爪仅有一个控制输入：0 闭合，255 张开；该输入不是米或牛顿。

## 与本项目 FR3 组合

场景：`models/scenes/fr3_with_hand.xml`（从项目根目录计）。
挂接位置为 FR3 的 `attachment_site`，即 link7 局部 Z 方向 0.107 m 处。
本地组合采用绕法兰 Z 轴 -45° 的安装方向；需要按实际设备核对安装方向与 TCP。
命名加 `gripper_` 前缀，避免与机械臂的材质、默认类重名。
组合后共 9 个关节自由度、8 个控制输入：前 7 个为机械臂目标角度（rad），最后一个为夹爪输入（0–255）。

在项目根目录预览：

```powershell
uv run python -m mujoco.viewer --mjcf=models/scenes/fr3_with_hand.xml
```

用当前项目的 MuJoCo 重新生成组合场景：

```powershell
uv run python -m scripts.build_fr3_with_hand
```

来源、SHA-256：`models/third_party/franka_hand_source_manifest.json`。
CPU MuJoCo 验证：`models/third_party/franka_hand_validation.json`。
独立夹爪及组合场景均完成 3000 步开合检查，无警告。
尚未验证 GPU Warp，也尚未接入强化学习环境。
