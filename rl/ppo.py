"""PPO 算法、策略网络、轨迹采样、评估和训练曲线。"""
# 算法层使用环境接口，不重复定义机器人模型或物理参数。
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from simulation.fr3_env import OBS_DIM, ReachEnv

class ActorCritic(nn.Module):
    """两个小型全连接网络：Actor 输出动作均值，Critic 估计状态价值。"""

    def __init__(self, obs_dim=OBS_DIM):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(),
                                   nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 7))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(),
                                    nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((7,), -0.7))
        # 小初始输出让随机策略从小幅移动开始，避免训练初期剧烈甩动。
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, np.sqrt(2))
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def policy(self, obs):
        # PPO 在原始高斯动作上计算概率比；tanh 变换由环境执行，采样与更新保持一致。
        return Normal(self.actor(obs), self.log_std.clamp(-3, 1).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def save_curve(rows, folder):
    """不增加绘图库依赖，直接生成浏览器可打开的 SVG 曲线和 CSV 数据。"""
    with (folder / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" style="background:#101827;font-family:Arial">']
    for panel, (key, title) in enumerate((("reward", "Mean reward"), ("distance", "Mean distance (m)"),
                                         ("success", "Within 3 cm (fraction)"), ("fps", "Environment steps / second"))):
        x, y = 55+(panel % 2)*475, 45+(panel//2)*310
        values = [row[key] for row in rows]
        lo, hi = min(values), max(values)
        points = " ".join(f"{x+i/max(1,len(values)-1)*400:.1f},{y+220-(v-lo)/max(hi-lo,1e-6)*190:.1f}" for i,v in enumerate(values))
        svg += [f'<text x="{x}" y="{y}" fill="white">{title}</text>',
                f'<path d="M{x},{y+15} V{y+235} H{x+400}" fill="none" stroke="#64748b"/>',
                f'<polyline points="{points}" fill="none" stroke="#38bdf8" stroke-width="2"/>',
                f'<text x="{x}" y="{y+260}" fill="#cbd5e1">min {lo:.4f} / max {hi:.4f} / latest {values[-1]:.4f}</text>']
    (folder / "curve.svg").write_text("\n".join(svg+["</svg>"]), encoding="utf-8")


@torch.no_grad()
def evaluate(agent, env):
    """独立于训练采样的确定性评估：统计最后 20 步的平均误差和持续到点比例。"""
    env.reset(torch.arange(env.n, device="cuda"))
    obs, distances = env.observe(), []
    failed_any = torch.zeros(env.n, device="cuda", dtype=torch.bool)
    for step in range(env.limit):
        obs, _, failed, _, distance = env.step(agent.actor(obs))
        failed_any |= failed
        if failed.any():
            env.reset(failed.nonzero().flatten())
            obs = env.observe()
        if step >= env.limit-20:
            distances.append(distance)
    tail = torch.stack(distances)
    return {"mean_distance_m": tail.mean().item(),
            "success_rate": ((tail < 0.03).all(0) & ~failed_any).float().mean().item(),
            "failed_rate": failed_any.float().mean().item(), "num_envs": env.n}


def train(args):
    """采集并行轨迹 → GAE 优势 → PPO 多轮小批量更新 → 保存模型和曲线。"""
    from torch.utils.tensorboard import SummaryWriter  # 只在训练时加载日志组件。
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("训练需要 CUDA PyTorch 和 NVIDIA GPU。")
    torch.cuda.set_stream(torch.cuda.Stream())  # 使用可捕获的非默认流，供 Torch 与 Warp 共享。
    folder = Path(args.out).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    env = ReachEnv(args.num_envs, args.target, args.grip, args.episode_steps,
                   args.sensor, args.contact_target)
    agent = ActorCritic(env.obs_dim).cuda()
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    writer = SummaryWriter(str(folder / "tensorboard"))
    # 预分配轨迹缓冲区；时间和世界两个维度在更新阶段展平。
    shape = (args.horizon, args.num_envs)
    obs_buf = torch.empty((*shape, env.obs_dim), device="cuda")
    act_buf = torch.empty((*shape, 7), device="cuda")
    log_buf, val_buf, rew_buf, next_buf, done_buf = [torch.empty(shape, device="cuda") for _ in range(5)]
    obs, rows = env.observe(), []
    baseline = evaluate(agent, env)
    print("随机初始策略评估：", baseline, flush=True)
    env.reset(torch.arange(env.n, device="cuda"))
    obs = env.observe()
    best_score = float("inf")
    best_result = None  # 同时保留最佳评估指标，便于与最近一次结果比较。
    for iteration in range(1, args.iterations+1):
        started = time.perf_counter()
        distances, successes, touch_depths, touch_contacts = [], [], [], []
        # rollout 不保留计算图；终止前的最后观测用于正确处理时间截断。
        with torch.no_grad():
            for t in range(args.horizon):
                dist = agent.policy(obs)
                action = dist.sample()
                obs_buf[t], act_buf[t] = obs, action
                log_buf[t], val_buf[t] = dist.log_prob(action).sum(-1), agent.value(obs)
                final_obs, reward, failed, timeout, distance = env.step(action)
                if env.tactile is not None:
                    # 在重置前记录真实采样中的触觉活动，避免只记录回合末尾的空白图。
                    depths = env.tactile.depth.amax(dim=(1, 2, 3))
                    touch_depths.append(depths.max())
                    touch_contacts.append((depths > 1e-5).float().mean())
                ended = failed | timeout
                rew_buf[t] = reward
                next_buf[t] = torch.where(failed, 0.0, agent.value(torch.nan_to_num(final_obs)))
                done_buf[t] = ended.float()
                distances.append(torch.nan_to_num(distance, nan=10.0))
                successes.append((distance < 0.03).float())
                if ended.any():
                    env.reset(ended.nonzero().flatten())
                obs = env.observe()
            advantages = torch.zeros_like(rew_buf)
            gae = torch.zeros(env.n, device="cuda")
            for t in reversed(range(args.horizon)):
                delta = rew_buf[t] + 0.99*next_buf[t] - val_buf[t]
                gae = delta + 0.99*0.95*(1-done_buf[t])*gae
                advantages[t] = gae
            returns = (advantages+val_buf).flatten()
            advantages = advantages.flatten()
            advantages = (advantages-advantages.mean()) / (advantages.std()+1e-8)
        bobs, bact, blog = obs_buf.flatten(0, 1), act_buf.flatten(0, 1), log_buf.flatten()
        losses, kls = [], []
        # PPO 裁剪概率比，并对梯度限幅；目标 KL 防止一次更新改变策略过多。
        for epoch in range(4):
            for ids in torch.randperm(len(blog), device="cuda").split(args.minibatch):
                distribution = agent.policy(bobs[ids])
                log_prob = distribution.log_prob(bact[ids]).sum(-1)
                log_ratio = log_prob-blog[ids]
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(ratio*advantages[ids], ratio.clamp(0.8, 1.2)*advantages[ids]).mean()
                value_loss = 0.5*(agent.value(bobs[ids])-returns[ids]).square().mean()
                loss = policy_loss + value_loss - 0.001*distribution.entropy().sum(-1).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("PPO 损失出现非有限值，请检查物理状态。")
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                losses.append(loss.detach())
                kls.append(((ratio-1)-log_ratio).mean().detach())
            if torch.stack(kls).mean().item() > 0.025:
                break
        elapsed = time.perf_counter()-started
        row = {"iteration": iteration, "reward": rew_buf.mean().item(),
               "distance": torch.stack(distances).mean().item(),
               "success": torch.stack(successes).mean().item(),
               "loss": torch.stack(losses).mean().item(), "fps": env.n*args.horizon/elapsed}
        if touch_depths:
            row["tactile_max_depth_mm"] = torch.stack(touch_depths).max().item()*1000
            row["tactile_contact_fraction"] = torch.stack(touch_contacts).mean().item()
        rows.append(row)
        for key, value in row.items():
            writer.add_scalar(key, value, iteration)
        if env.tactile is not None:
            # TensorBoard 中直接检查图像是否随接触变化，以及是否达到形变饱和。
            writer.add_scalar("tactile/max_depth_mm", env.tactile.depth.max().item()*1000, iteration)
            writer.add_images("tactile/rgb", env.tactile.rgb[0], iteration)
        print(f"迭代 {iteration}/{args.iterations} | 奖励 {row['reward']:.3f} | 距离 {row['distance']:.3f} m | 到点占比 {row['success']:.1%} | {row['fps']:.0f} 步/秒", flush=True)
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            # 固定评估种子便于比较各次 checkpoint；恢复训练 RNG，不干扰后续采样。
            rng = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(args.seed+1000)
            result = evaluate(agent, env)
            torch.cuda.set_rng_state(rng)
            result["iteration"] = iteration
            print("确定性策略评估：", result, flush=True)
            checkpoint = {"agent": agent.state_dict(), "args": vars(args),
                          "obs_dim": env.obs_dim, "evaluation": result}
            torch.save(checkpoint, folder / "last.pt")
            if result["mean_distance_m"] < best_score:
                best_score = result["mean_distance_m"]
                best_result = dict(result)
                torch.save(checkpoint, folder / "best.pt")
            (folder / "evaluation.json").write_text(json.dumps({"baseline": baseline, "best": best_result, "latest": result}, indent=2))
            save_curve(rows, folder)
            writer.flush()
            env.reset(torch.arange(env.n, device="cuda"))
            obs = env.observe()
    writer.close()
    print(f"训练完成，模型与曲线：{folder}", flush=True)


def test_policy(args):
    """从 checkpoint 读取任务参数，以固定种子单独进行 GPU 并行评估。"""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    saved = checkpoint["args"]
    torch.manual_seed(args.seed+1000)
    torch.cuda.set_stream(torch.cuda.Stream())
    env = ReachEnv(args.num_envs, saved["target"], saved["grip"], saved["episode_steps"],
                   saved.get("sensor", "none"), saved.get("contact_target", False))
    agent = ActorCritic(env.obs_dim).cuda()
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    torch.cuda.manual_seed(args.seed+1000)  # 去除网络初始化对起始姿态采样的影响。
    print(json.dumps(evaluate(agent, env), indent=2), flush=True)


