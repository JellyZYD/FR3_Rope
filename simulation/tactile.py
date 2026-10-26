"""柔顺接触驱动的凝胶高度场与 RGB 触觉成像，支持 Torch CPU 和批量 CUDA。

这是未标定的近似表面模型：MuJoCo 计算柔顺接触，接触压入量生成局部高度场，
一阶松弛模拟恢复，三色方向光根据表面法线成像；不求解体积 FEM 或剪切形变。
"""
import math
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

# 尺寸是本项目的可替换感测贴片，不是制造 CAD 的完整安装或硬件标定。
PROFILES = {"digit": (0.020, 0.030), "gelsight": (0.018, 0.024)}
MAX_DEPTH = 0.002  # 图像高度场饱和于 2 mm；物理接触仍由 MuJoCo 独立求解。
TOUCH_FEATURES = 128  # 双指 RGB 差分与归一化深度各池化为 4×4 后拼接。


def add_pad(body, name, sensor, pos=(0, 0, 0), quat=(1, 0, 0, 0)):
    """以贴片局部 +Z 为外法线，创建参与物理求解的柔顺接触几何。"""
    width, height = PROFILES[sensor]
    pad = body.add_body(name=name, pos=pos, quat=quat)
    pad.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[width/2, height/2, 0.0015], pos=[0, 0, -0.0015],
                 mass=0.005, rgba=[0.45, 0.65, 0.9, 0.7],
                 contype=1, conaffinity=1, condim=3, priority=1,
                 solref=[0.03, 1], solimp=[0.8, 0.95, 0.003, 0.5, 2],
                 friction=[0.8, 0.005, 0.0001])


def mount_pads(spec, sensor):
    """在两个手指内侧安装等效感测贴片，并关闭被替代的原指尖碰撞。"""
    for side in ("left", "right"):
        body = spec.body(f"gripper_{side}_finger")
        for geom in body.geoms:
            geom.contype, geom.conaffinity = 0, 0
        # 两个手指局部坐标相同；其父体已有镜像旋转，+Z 自动朝向夹爪内部。
        add_pad(body, f"gel_{side}", sensor, [0, -0.004, 0.0445],
                [math.sqrt(0.5), math.sqrt(0.5), 0, 0])


@wp.kernel
def _raster_contacts(ncon: wp.array(dtype=int),
                     geom: wp.array(dtype=wp.vec2i), world: wp.array(dtype=int),
                     pos: wp.array(dtype=wp.vec3), dist: wp.array(dtype=float),
                     frame: wp.array(dtype=wp.mat33),
                     geom_pos: wp.array2d(dtype=wp.vec3),
                     geom_rot: wp.array2d(dtype=wp.mat33),
                     pad_ids: wp.array(dtype=int), width: float, height: float,
                     h: int, w: int, desired: wp.array4d(dtype=float)):
    """每个接触点在感测面栅格上写入一个局部压痕，多接触取最大包络。"""
    contact, side, pixel = wp.tid()
    if contact >= ncon[0] or dist[contact] >= 0.0:
        return
    pair = geom[contact]
    pad = int(pad_ids[side])
    # 双贴片互相接触时两侧都要写入，不能只处理 pair 中最后匹配的一侧。
    if pair[0] != pad and pair[1] != pad:
        return
    wi = world[contact]
    rot = geom_rot[wi, pad]
    normal = wp.vec3(frame[contact][0, 0], frame[contact][0, 1], frame[contact][0, 2])
    outward = rot * wp.vec3(0.0, 0.0, 1.0)
    # 只接受接近正面的接触，贴片侧边接触不生成假的正面压痕。
    if wp.abs(wp.dot(normal, outward)) < 0.7:
        return
    local = wp.transpose(rot) * (pos[contact]-geom_pos[wi, pad])
    x = (float(pixel % w)/float(w-1)-0.5)*width
    y = (float(pixel // w)/float(h-1)-0.5)*height
    depth = wp.min(-dist[contact], 0.002)
    # 采用局部高斯压痕近似；接触越深，影响范围越大，不复制对象完整轮廓。
    sigma = wp.max(0.0015, wp.sqrt(0.006*depth))
    r2 = (x-local[0])*(x-local[0])+(y-local[1])*(y-local[1])
    value = depth*wp.exp(-0.5*r2/(sigma*sigma))
    wp.atomic_max(desired, wi, side, pixel//w, pixel % w, value)


class GelRenderer:
    """接触 → 凝胶压痕高度场 → RGB；缓存图像供 PPO 读取，避免重复推进恢复状态。"""

    def __init__(self, model, count=1, sensor="digit", device="cpu", resolution=(24, 16),
                 names=("gel_left", "gel_right")):
        self.device, self.sensor = torch.device(device), sensor
        self.width, self.height = PROFILES[sensor]
        self.h, self.w = resolution
        self.ids = [model.geom(name).id for name in names]
        self.depth = torch.zeros(count, 2, self.h, self.w, device=device)
        self.desired = torch.zeros_like(self.depth)
        y = torch.linspace(-self.height/2, self.height/2, self.h, device=device)
        x = torch.linspace(-self.width/2, self.width/2, self.w, device=device)
        self.y, self.x = torch.meshgrid(y, x, indexing="ij")
        # 静止凝胶的轻微弧度提供稳定背景，三路光源具有不同入射方向。
        self.curve = 0.00015*(1-(self.x/self.width*2).square()-(self.y/self.height*2).square())
        self.lights = F.normalize(torch.tensor([[0.6, 0, 1], [-0.3, 0.52, 1],
                                                 [-0.3, -0.52, 1]], device=device), dim=1)
        self.baseline = self._shade(self.depth).clone()
        self.rgb = self.baseline.clone()
        if self.device.type == "cuda":
            self.pad_ids = wp.array(self.ids, dtype=int, device=str(self.device))
            self.field = wp.from_torch(self.desired)

    def _shade(self, depth):
        """由高度场有限差分得到法线，使用三色漫反射与镜面分量生成 RGB。"""
        surface = self.curve-depth
        gy, gx = torch.gradient(surface, spacing=(self.height/(self.h-1),
                                                   self.width/(self.w-1)), dim=(-2, -1))
        normal = F.normalize(torch.stack((-gx, -gy, torch.ones_like(gx)), -1), dim=-1)
        diffuse = torch.einsum("nshwc,kc->nshwk", normal, self.lights).clamp_min(0)
        # 背景纹理模拟标记点；不伪称已经实现剪切导致的标记位移。
        dots = ((torch.cos(self.x*3000) > 0.93) & (torch.cos(self.y*3000) > 0.93)).float()
        color = (0.12+0.65*diffuse+0.1*diffuse.pow(20))*(1-0.65*dots[..., None])
        return color.permute(0, 1, 4, 2, 3).clamp(0, 1).contiguous()

    def advance(self, dt=0.02):
        """按秒计算一阶松弛；按压响应较快，离开后的恢复较慢。"""
        tau = torch.where(self.desired > self.depth, 0.012, 0.06)
        self.depth.lerp_(self.desired, 1-torch.exp(-dt/tau))
        self.rgb = self._shade(self.depth)

    def update_gpu(self, data, stream, dt=0.02):
        """直接读取 Warp 接触缓冲区，渲染和特征计算均不复制到 CPU。"""
        self.desired.zero_()
        with wp.ScopedStream(stream):
            wp.launch(_raster_contacts, dim=(data.contact.dist.size, 2, self.h*self.w),
                      inputs=[data.nacon, data.contact.geom, data.contact.worldid,
                              data.contact.pos, data.contact.dist, data.contact.frame,
                              data.geom_xpos, data.geom_xmat, self.pad_ids, self.width,
                              self.height, self.h, self.w, self.field])
        self.advance(dt)

    def update_cpu(self, model, data, dt=0.02):
        """回放使用同一压痕公式，CPU 只需处理一个世界的实际有效接触。"""
        self.desired.zero_()
        for contact in data.contact:
            if contact.dist >= 0:
                continue
            for side, pad in enumerate(self.ids):
                if pad not in contact.geom:
                    continue
                rot = data.geom_xmat[pad].reshape(3, 3)
                if abs(np.dot(contact.frame[:3], rot[:, 2])) < 0.7:
                    continue
                local = rot.T@(contact.pos-data.geom_xpos[pad])
                depth = min(-contact.dist, MAX_DEPTH)
                sigma = max(0.0015, math.sqrt(0.006*depth))
                field = depth*torch.exp(-0.5*((self.x-local[0]).square()+
                                               (self.y-local[1]).square())/sigma**2)
                self.desired[0, side] = torch.maximum(self.desired[0, side], field)
        self.advance(dt)

    def features(self):
        """图像先做静止背景差分，再以 4×4 池化接入 PPO，减少轨迹缓存开销。"""
        n = self.depth.shape[0]
        rgb = F.adaptive_avg_pool2d((self.rgb-self.baseline).reshape(n*2, 3, self.h, self.w), (4, 4))
        depth = F.adaptive_avg_pool2d(self.depth.reshape(n*2, 1, self.h, self.w)/MAX_DEPTH, (4, 4))
        return torch.cat((rgb.reshape(n, -1), depth.reshape(n, -1)), 1)

    def reset(self, ids=None):
        """回合重置时消除上一回合的形变，避免触觉观测跨回合泄漏。"""
        if ids is None:
            self.depth.zero_()
            self.desired.zero_()
            self.rgb.copy_(self.baseline)
        else:
            self.depth[ids] = 0
            self.desired[ids] = 0
            self.rgb[ids] = self.baseline[ids]
