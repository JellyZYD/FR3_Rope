"""可重复的实际接触按压演示，输出 RGB、深度、剖面与 GIF，不依赖训练策略。"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from simulation.tactile import GelRenderer, MAX_DEPTH, add_pad


def make_press_model(sensor="digit"):
    """位置执行器推动球体按压左贴片；右贴片不接触，用于检查串扰。"""
    spec = mujoco.MjSpec.from_string('''
    <mujoco model="gel press">
      <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
      <worldbody>
        <body name="probe" pos="-0.018 0 0.014">
          <joint name="press" type="slide" axis="0 0 1" damping="0.2"/>
          <geom type="sphere" size="0.006" mass="0.02" rgba="1 0.2 0.1 1"/>
        </body>
      </worldbody>
      <actuator><position joint="press" kp="2000" kv="15" ctrlrange="-0.012 0"/></actuator>
    </mujoco>''')
    add_pad(spec.worldbody, "gel_left", sensor, [-0.02, 0, 0])
    add_pad(spec.worldbody, "gel_right", sensor, [0.02, 0, 0])
    return spec.compile()


def panel(renderer, world=0, scale=4):
    """双指 RGB、压痕伪彩色和中心剖面，用于保存或窗口实时显示。"""
    rgb = renderer.rgb[world].detach().cpu().numpy()
    depth = renderer.depth[world].detach().cpu().numpy()
    width, height = renderer.w*scale, renderer.h*scale
    canvas = Image.new("RGB", (2*width, 2*height+100), "#152033")
    for side in range(2):
        color = Image.fromarray((rgb[side].transpose(1, 2, 0)*255).astype(np.uint8))
        value = np.clip(depth[side]/MAX_DEPTH, 0, 1)
        heat = np.stack((value, value**0.5, np.zeros_like(value)), -1)
        canvas.paste(color.resize((width, height)), (side*width, 25))
        canvas.paste(Image.fromarray((heat*255).astype(np.uint8)).resize((width, height)),
                     (side*width, height+45))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "LEFT RGB         RIGHT RGB", fill="white")
    draw.text((5, height+28), "INDENTATION 0 ... 2 mm", fill="white")
    ybase = 2*height+85
    draw.line((5, ybase, 2*width-5, ybase), fill="gray")
    profile = depth[0, renderer.h//2]/MAX_DEPTH
    points = [(5+i/(len(profile)-1)*(2*width-10), ybase-float(v)*30) for i, v in enumerate(profile)]
    draw.line(points, fill="#44ddff", width=2)
    return canvas


class TouchWindow:
    """回放时显示触觉面板；关闭此窗口不会终止机器人回放。"""

    def __init__(self):
        import tkinter as tk  # Tk 只在主动显示触觉窗口时加载。
        self.window = tk.Tk()
        self.window.title("Tactile RGB / gel indentation")
        self.label = tk.Label(self.window)
        self.label.pack()
        self.closed = False
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def update(self, renderer):
        if self.closed:
            return
        from PIL import ImageTk  # 保持引用，避免 Tk 图像被垃圾回收。
        self.image = ImageTk.PhotoImage(panel(renderer, scale=6))
        self.label.configure(image=self.image)
        self.window.update()

    def close(self):
        if not self.closed:
            self.closed = True
            self.window.destroy()


def main():
    """独立演示包含无接触、按压、保持和释放四个阶段。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=["digit", "gelsight"], default="digit")
    parser.add_argument("--out", default="runs/tactile_demo")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, frames, history = make_press_model(args.sensor), [], []
    data = mujoco.MjData(model)
    renderer = GelRenderer(model, sensor=args.sensor, resolution=(96, 64))
    stages = [("idle", 0.0, 20), ("press", -0.0095, 60), ("hold", -0.0095, 20), ("release", 0.0, 60)]
    for name, command, length in stages:
        for _ in range(length):
            data.ctrl[0] = command
            mujoco.mj_step(model, data, nstep=10)
            mujoco.mj_forward(model, data)
            renderer.update_cpu(model, data)
            force = 0.0
            for index, contact in enumerate(data.contact):
                if renderer.ids[0] in contact.geom:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, wrench)
                    force += max(0.0, float(wrench[0]))
            history.append({"stage": name, "time_s": float(data.time), "normal_force_N": force,
                            "max_depth_mm": renderer.depth.max().item()*1000})
            image = panel(renderer)
            ImageDraw.Draw(image).text((5, image.height-12),
                                      f"{name}: {history[-1]['max_depth_mm']:.3f} mm  {force:.2f} N", fill="white")
            frames.append(image)
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    frames[75].save(folder / "press.png")
    frames[0].save(folder / "idle.png")
    frames[-1].save(folder / "release.png")
    frames[0].save(folder / "press.gif", save_all=True, append_images=frames[1:], duration=20, loop=0)
    (folder / "history.json").write_text(json.dumps(history, indent=2)+"\n")
    np.save(folder / "final_depth_m.npy", renderer.depth.numpy())
    print(f"按压演示已保存：{folder.resolve()}")


if __name__ == "__main__":
    main()  # 运行：python -m simulation.tactile_demo --sensor digit。
