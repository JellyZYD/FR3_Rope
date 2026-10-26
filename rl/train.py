"""FR3 PPO 统一入口：python -m rl.train train / eval / play。"""
# 入口只处理命令行，算法、物理环境和显示分别交给各自模块。
import argparse
import numpy as np
import torch
from rl.ppo import train, test_policy
from simulation.fr3_env import ROOT
from simulation.replay import play

def main():
    """命令行包含训练、评估和回放；目标坐标单位为米，位于机器人基座世界坐标系。"""
    torch.set_num_threads(4)  # 小网络限制 CPU 线程数，避免过多线程拖慢初始化和回放。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "eval", "play"])
    parser.add_argument("--target", type=float, nargs=3, default=[0.45, 0, 0.45])
    parser.add_argument("--grip", type=float, default=1.0, help="固定夹爪开度，0 闭合，1 张开")
    parser.add_argument("--sensor", choices=["none", "digit", "gelsight"], default="none",
                        help="可选视触觉贴片；none 保持旧 25 维观测及权重兼容")
    parser.add_argument("--contact-target", action="store_true", help="让目标球产生真实物理接触")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=150)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/fr3_ppo/best.pt"))
    args = parser.parse_args()
    if args.out is None:
        args.out = str(ROOT / "runs" / ("fr3_ppo" if args.sensor == "none" else f"fr3_{args.sensor}_ppo"))
    if args.contact_target and args.sensor == "none" and args.mode == "train":
        parser.error("contact-target 请与 --sensor digit 或 gelsight 一起使用。")
    if min(args.num_envs, args.iterations, args.minibatch, args.eval_every) < 1 or args.horizon < 2 or args.episode_steps < 20:
        parser.error("计数必须为正，horizon 至少为 2，episode-steps 至少为 20。")
    if not 0 <= args.grip <= 1 or not np.isfinite(args.target).all():
        parser.error("grip 必须在 0–1 之间，目标坐标必须为有限数值。")
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        test_policy(args)
    else:
        play(args)


if __name__ == "__main__":
    main()  # 以模块方式运行，保证项目内的导入路径正确。
