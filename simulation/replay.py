"""训练策略的 CPU MuJoCo 窗口回放，与 GPU 训练共用模型和动作定义。"""
# 回放不实现训练算法，仅加载网络权重并显示机器人运动。
import time
from contextlib import ExitStack  # 窗口关闭或提前返回时也清理触觉窗口。
import mujoco
import numpy as np
import torch
from rl.ppo import ActorCritic
from simulation.fr3_env import ACTION_SCALE, SUBSTEPS, make_model
from simulation.tactile import GelRenderer

def play(args):
    """窗口回放使用 CPU MuJoCo，便于实时显示；训练中的物理仍是 GPU 并行。"""
    import mujoco.viewer  # 延迟导入窗口组件，训练时无需图形窗口。
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = checkpoint["args"]
    agent = ActorCritic(checkpoint.get("obs_dim", 25))
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    sensor = saved.get("sensor", "none")
    model = make_model(sensor, saved.get("contact_target", False))
    target = np.array(saved["target"])
    tactile, touch_window = None, None
    if sensor != "none":
        from simulation.tactile_demo import TouchWindow  # 窗口展示实际用于观测的触觉图像。
        tactile = GelRenderer(model, sensor=sensor)
        touch_window = TouchWindow()
    data = mujoco.MjData(model)
    site, home = model.site("tcp").id, model.key_qpos[0].copy()
    grip = saved["grip"]
    rng = np.random.default_rng(args.seed)
    with ExitStack() as resources, mujoco.viewer.launch_passive(model, data) as viewer:
        if touch_window is not None:
            resources.callback(touch_window.close)
        viewer.cam.lookat[:] = [0.3, 0, 0.4]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 135, -25
        while viewer.is_running():
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[:7] += rng.uniform(-0.1, 0.1, 7)
            data.qpos[7:] = grip*0.04
            data.mocap_pos[0] = target
            if tactile is not None:
                tactile.reset()
            mujoco.mj_forward(model, data)
            for step in range(saved["episode_steps"]):
                if not viewer.is_running():
                    return
                started = time.perf_counter()
                tcp = data.site_xpos[site].copy()
                observation = np.concatenate((data.qpos[:7]-home[:7], data.qvel[:7]*0.1,
                                              data.qpos[7:], tcp, target, target-tcp))
                if tactile is not None:
                    observation = np.r_[observation, tactile.features()[0].numpy()]
                with torch.no_grad():
                    action = agent.actor(torch.tensor(observation, dtype=torch.float32)).tanh().numpy()
                data.ctrl[:7] = np.clip(data.qpos[:7]+ACTION_SCALE*action,
                                        model.jnt_range[:7, 0], model.jnt_range[:7, 1])
                data.ctrl[7] = grip*255
                for _ in range(SUBSTEPS):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                if tactile is not None:
                    tactile.update_cpu(model, data)
                    touch_window.update(tactile)
                viewer.sync()
                if step % 50 == 0:
                    print(f"TCP 误差：{np.linalg.norm(data.site_xpos[site]-target):.4f} m", flush=True)
                time.sleep(max(0, SUBSTEPS*model.opt.timestep-(time.perf_counter()-started)))
    if touch_window is not None:
        touch_window.close()


