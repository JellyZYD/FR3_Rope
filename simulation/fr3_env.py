"""FR3 带夹爪的 GPU 并行到点环境：模型、动作、观测、奖励和重置。"""
# 仿真依赖集中在环境层；路径从本文件向上定位到项目根目录。
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from simulation.tactile import GelRenderer, TOUCH_FEATURES, mount_pads

ROOT = Path(__file__).resolve().parents[1]  # 统一定位模型与运行产物。
SUBSTEPS = 10  # 物理步长 2 ms，策略控制频率 50 Hz。
ACTION_SCALE = 0.08  # 每步目标关节角最大相对变化，单位 rad。
OBS_DIM = 25  # 关节角、速度、手指位置、TCP、目标与误差。

def make_model(sensor="none", contact_target=False):
    """载入带夹爪场景，增加指尖中点 TCP 和绿色目标标记。"""
    spec = mujoco.MjSpec.from_file(str(ROOT / "models/scenes/fr3_with_hand.xml"))
    spec.body("gripper_hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.008]*3,
                                         rgba=[1, 0.3, 0, 1])
    spec.worldbody.add_body(name="goal", mocap=True).add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.018 if contact_target else 0.025, 0, 0],
        rgba=[0.1, 0.9, 0.2, 0.6], contype=int(contact_target), conaffinity=int(contact_target))
    if sensor != "none":
        mount_pads(spec, sensor)  # 两片凝胶参与接触求解，传感器没有增加关节自由度。
    model = spec.compile()
    model.opt.timestep = 0.002  # 训练和回放使用完全相同的物理参数。
    return model


class ReachEnv:
    """同一个机器人模型的多个独立 GPU 世界，所有观测和奖励均在 GPU 计算。"""

    def __init__(self, count, target, grip, episode_steps, sensor="none", contact_target=False):
        if not torch.cuda.is_available():
            raise RuntimeError("需要 NVIDIA CUDA GPU，不会自动退回 CPU。")
        wp.init()
        self.n, self.grip, self.limit = count, grip, episode_steps
        self.cpu = make_model(sensor, contact_target)
        self.obs_dim = OBS_DIM + (TOUCH_FEATURES if sensor != "none" else 0)
        self.tactile = None  # 第一次 reset 之前声明，随后创建与当前设备匹配的渲染器。
        self.site = self.cpu.site("tcp").id
        self.home = torch.tensor(self.cpu.key_qpos[0], device="cuda", dtype=torch.float32)
        self.low = torch.tensor(self.cpu.jnt_range[:7, 0], device="cuda", dtype=torch.float32)
        self.high = torch.tensor(self.cpu.jnt_range[:7, 1], device="cuda", dtype=torch.float32)
        self.target = torch.tensor(target, device="cuda", dtype=torch.float32).expand(count, -1)
        self.age = torch.zeros(count, device="cuda", dtype=torch.long)
        # 共享 PyTorch 的 CUDA 流，避免两个库之间读取到尚未完成的数据。
        self.stream = wp.stream_from_torch(torch.cuda.current_stream())
        with wp.ScopedStream(self.stream):
            self.model = mjw.put_model(self.cpu)
            self.data = mjw.make_data(self.cpu, nworld=count, nconmax=64, njmax=128)
        self.q = wp.to_torch(self.data.qpos)
        self.v = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.warm = wp.to_torch(self.data.qacc_warmstart)
        self.clock = wp.to_torch(self.data.time)
        self.tcp = wp.to_torch(self.data.site_xpos)[:, self.site]
        wp.to_torch(self.data.mocap_pos)[:, 0] = self.target  # 目标碰撞球必须位于指定位置。
        if sensor != "none":
            self.tactile = GelRenderer(self.cpu, count, sensor, "cuda:0")
        self.reset(torch.arange(count, device="cuda"))
        # 先编译内核，再捕获固定的物理步进图，降低每步 Python 调度开销。
        print("正在准备 FR3 GPU 内核；首次运行编译可能较慢。", flush=True)
        with wp.ScopedStream(self.stream):
            mjw.step(self.model, self.data)
            with wp.ScopedCapture(stream=self.stream) as capture:
                for _ in range(SUBSTEPS):
                    mjw.step(self.model, self.data)
                mjw.forward(self.model, self.data)  # 刷新最终关节位置对应的 TCP。
        self.graph = capture.graph
        self.reset(torch.arange(count, device="cuda"))

    def reset(self, ids):
        """只重置结束的世界，起始关节角增加小扰动，使策略具有一定鲁棒性。"""
        self.q[ids] = self.home
        noise = (torch.rand((len(ids), 7), device="cuda") - 0.5) * 0.2
        self.q[ids, :7] = (self.home[:7] + noise).clamp(self.low, self.high)
        self.q[ids, 7:] = self.grip * 0.04
        self.v[ids] = 0
        self.warm[ids] = 0
        self.clock[ids] = 0
        self.ctrl[ids, :7] = self.q[ids, :7]
        self.ctrl[ids, 7] = self.grip * 255
        self.age[ids] = 0
        with wp.ScopedStream(self.stream):
            mjw.forward(self.model, self.data)
        if self.tactile is not None:
            self.tactile.reset(ids)

    def observe(self):
        """速度适当缩放；clone 防止物理步进修改 PPO 已保存的观测。"""
        state = torch.cat((self.q[:, :7] - self.home[:7], self.v[:, :7] * 0.1,
                           self.q[:, 7:], self.tcp, self.target, self.target-self.tcp), 1)
        if self.tactile is not None:
            state = torch.cat((state, self.tactile.features()), 1)
        return state.clone()

    def step(self, raw_action):
        """高斯原始动作经 tanh 限幅；夹爪保持指定开度，任务仅学习移动。"""
        before = torch.linalg.vector_norm(self.tcp-self.target, dim=1)
        action = raw_action.tanh()
        self.ctrl[:, :7] = (self.q[:, :7] + ACTION_SCALE*action).clamp(self.low, self.high)
        self.ctrl[:, 7] = self.grip * 255
        with wp.ScopedStream(self.stream):
            wp.capture_launch(self.graph, stream=self.stream)
        if self.tactile is not None:
            self.tactile.update_gpu(self.data, self.stream, SUBSTEPS*self.cpu.opt.timestep)
        self.age += 1
        distance = torch.linalg.vector_norm(self.tcp-self.target, dim=1)
        # 稠密奖励鼓励靠近并停留；动作惩罚减少抖动，3 cm 内获得额外奖励。
        reward = torch.exp(-10*distance) + 10*(before-distance)
        reward += (distance < 0.03).float() - 0.002*action.square().sum(1)
        failed = (~torch.isfinite(self.q).all(1) | ~torch.isfinite(self.v).all(1)
                  | (self.v.abs().amax(1) > 100))
        reward = torch.where(failed, -5.0, reward)
        truncated = self.age >= self.limit  # 时间限制不是物理终止，应保留价值 bootstrap。
        return self.observe(), reward, failed, truncated, distance


