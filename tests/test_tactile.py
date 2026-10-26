"""验证实际接触、释放恢复、坐标变换、CPU/GPU 一致性以及 PPO 观测连接。"""
import unittest
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from simulation.tactile import GelRenderer, MAX_DEPTH, TOUCH_FEATURES
from simulation.tactile_demo import make_press_model
from simulation.fr3_env import ReachEnv
from rl.ppo import ActorCritic


class TactileTests(unittest.TestCase):
    """使用真的 MuJoCo 接触，避免用手工伪造压痕来验证成像。"""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def pressed(self, sensor="digit", command=-0.0095):
        model = make_press_model(sensor)
        data = mujoco.MjData(model)
        data.ctrl[0] = command
        mujoco.mj_step(model, data, nstep=500)
        mujoco.mj_forward(model, data)
        return model, data

    def test_contact_image_and_release(self):
        for sensor in ("digit", "gelsight"):
            with self.subTest(sensor=sensor):
                model, data = self.pressed(sensor)
                renderer = GelRenderer(model, sensor=sensor)
                renderer.update_cpu(model, data)
                maximum = renderer.depth[0, 0].max().item()
                self.assertGreater(maximum, 0.0005)
                self.assertLessEqual(maximum, MAX_DEPTH)
                self.assertEqual(renderer.depth[0, 1].max().item(), 0)
                self.assertGreater((renderer.rgb-renderer.baseline).abs().max().item(), 0.01)
                self.assertTrue(torch.isfinite(renderer.features()).all())
                # 更小按压位移应得到更浅的物理压入量。
                shallow_m, shallow_d = self.pressed(sensor, -0.0085)
                shallow = GelRenderer(shallow_m, sensor=sensor)
                shallow.update_cpu(shallow_m, shallow_d)
                self.assertLess(shallow.depth.max().item(), maximum)
                # 撤回探头后图像和高度场必须恢复，右贴片始终保持无接触。
                data.ctrl[0] = 0
                for _ in range(60):
                    mujoco.mj_step(model, data, nstep=10)
                    mujoco.mj_forward(model, data)
                    renderer.update_cpu(model, data)
                self.assertLess(renderer.depth.max().item(), maximum*0.01)
                self.assertLess((renderer.rgb-renderer.baseline).abs().max().item(), 0.001)

    def test_translation_invariant(self):
        model, data = self.pressed()
        original = GelRenderer(model)
        original.update_cpu(model, data)
        # 整体平移改变世界坐标，不应改变凝胶局部坐标中的压痕图。
        model.body_pos[1:] += np.array([0.2, -0.3, 0.1])
        mujoco.mj_forward(model, data)
        shifted = GelRenderer(model)
        shifted.update_cpu(model, data)
        torch.testing.assert_close(original.depth, shifted.depth, atol=1e-6, rtol=0.001)
        # 再整体旋转；感测面已不与世界坐标平行，也应得到相同局部压痕。
        angle = 0.7
        rotation = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        model.body_pos[1:] = model.body_pos[1:]@rotation.T
        model.body_quat[1:] = [np.cos(angle/2), np.sin(angle/2), 0, 0]
        mujoco.mj_forward(model, data)
        rotated = GelRenderer(model)
        rotated.update_cpu(model, data)
        torch.testing.assert_close(original.depth, rotated.depth, atol=1e-6, rtol=0.001)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才运行 GPU 一致性检查")
    def test_gpu_cpu_and_world_isolation(self):
        wp.init()
        with torch.cuda.stream(torch.cuda.Stream()):
            stream = wp.stream_from_torch(torch.cuda.current_stream())
            model, data = self.pressed()
            cpu = GelRenderer(model)
            cpu.update_cpu(model, data)
            with wp.ScopedStream(stream):
                gpu_model = mjw.put_model(model)
                gpu_data = mjw.put_data(model, data, nworld=2, nconmax=32, njmax=64)
            # 第二个世界撤回探头，检查它不会读取第一个世界的接触。
            wp.to_torch(gpu_data.qpos)[1] = 0
            wp.to_torch(gpu_data.qvel)[1] = 0
            with wp.ScopedStream(stream):
                mjw.forward(gpu_model, gpu_data)
            gpu = GelRenderer(model, count=2, device="cuda:0")
            gpu.update_gpu(gpu_data, stream)
            torch.testing.assert_close(cpu.depth, gpu.depth[:1].cpu(), atol=5e-6, rtol=0.01)
            self.assertEqual(gpu.depth[1].max().item(), 0)
            self.assertEqual(gpu.depth[:, 1].max().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才验证双贴片互相接触")
    def test_both_sides_of_pad_contact(self):
        wp.init()  # 单独运行本用例时也要先初始化 Warp。
        from simulation.tactile import add_pad  # 单个接触同时属于两片凝胶的边界情况。
        spec = mujoco.MjSpec.from_string('''<mujoco><option gravity="0 0 0"/>
        <worldbody><body name="moving" pos="0 0 -0.001">
        <joint type="slide" axis="0 0 1"/>
        </body></worldbody></mujoco>''')
        add_pad(spec.worldbody, "gel_left", "digit")
        add_pad(spec.body("moving"), "gel_right", "digit", quat=[0, 1, 0, 0])
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        cpu = GelRenderer(model)
        cpu.update_cpu(model, data)
        self.assertTrue((cpu.depth.amax(dim=(2, 3)) > 0).all())
        with torch.cuda.stream(torch.cuda.Stream()):
            stream = wp.stream_from_torch(torch.cuda.current_stream())
            with wp.ScopedStream(stream):
                gm = mjw.put_model(model)
                gd = mjw.put_data(model, data, nworld=1, nconmax=32, njmax=64)
                mjw.forward(gm, gd)
            gpu = GelRenderer(model, device="cuda:0")
            gpu.update_gpu(gd, stream)
            torch.testing.assert_close(cpu.depth, gpu.depth.cpu(), atol=5e-6, rtol=0.01)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA 才验证 PPO 接线")
    def test_ppo_features_and_partial_reset(self):
        with torch.cuda.stream(torch.cuda.Stream()):
            env = ReachEnv(2, [.45, 0, .45], .5, 20, "digit", True)
            # 把物理目标放到每个世界当前的 TCP，以真实手指接触激活观测。
            wp.to_torch(env.data.mocap_pos)[:, 0] = env.tcp
            with wp.ScopedStream(env.stream):
                mjw.forward(env.model, env.data)
            env.tactile.update_gpu(env.data, env.stream)
            obs = env.observe()
            self.assertEqual(obs.shape, (2, 25+TOUCH_FEATURES))
            self.assertGreater(obs[:, 25:].abs().max().item(), 0.01)
            torch.testing.assert_close(obs[:, 25:], env.tactile.features())
            agent = ActorCritic(env.obs_dim).cuda()
            agent.actor(obs).square().mean().backward()
            self.assertGreater(agent.actor[0].weight.grad[:, 25:].abs().sum().item(), 0)
            other = env.tactile.depth[1].clone()
            env.reset(torch.tensor([0], device="cuda"))
            self.assertEqual(env.tactile.depth[0].max().item(), 0)
            torch.testing.assert_close(env.tactile.depth[1], other)


if __name__ == "__main__":
    unittest.main()  # 运行：python -m unittest discover -s tests -v。
