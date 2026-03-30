import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import PortfolioEnv
from sac import SAC

def train_sac_optimized(
    n_assets=2,
    T=5,
    episodes=15000,      # 增加到15000
    batch_size=256,      # 增大batch降低方差
    max_trade=0.1,
    risk_aversion=1.0,
    warm_up=1000,        # 前1000轮只收集数据不训练（或训练频率低）
    evaluate_freq=500,
    patience=10          # 早停耐心值（连续10次无改善则停）
):
    """
    完整调优版训练流程：
    - Cosine Annealing 学习率衰减
    - Warm Up 阶段
    - Early Stopping
    - 保存最佳模型
    """
    env_temp = PortfolioEnv(n_assets=n_assets, T=T, max_trade=max_trade, risk_aversion=risk_aversion)
    state_dim = env_temp.state_dim
    action_dim = env_temp.action_dim
    
    print(f"开始训练 [优化版]: n_assets={n_assets}, T={T}, episodes={episodes}")
    print(f"网络架构: 128x128, Batch Size: {batch_size}")
    print(f"Warm Up: {warm_up}, Early Stopping Patience: {patience}")
    
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_trade=max_trade,
        hidden_dim=128,      # 增大网络
        lr=3e-4,
        gamma=0.99,          # 折扣因子
        tau=0.005,
        alpha=0.15,          # 平衡探索
        auto_tune_alpha=False,  # 固定alpha更稳定
        buffer_size=int(5e5)   # 增大buffer
    )
    
    # 学习率调度器（Cosine Annealing）
    scheduler_actor = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.actor_optimizer, T_max=episodes, eta_min=1e-5
    )
    scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.critic_1_optimizer, T_max=episodes, eta_min=1e-5
    )
    
    # 训练记录
    episode_rewards = []
    eval_rewards = []
    best_eval_reward = -np.inf
    patience_counter = 0
    best_model_path = f"best_sac_n{n_assets}_T{T}.pth"
    logs = None  # <<<< 添加这行，初始化 logs
    
    for episode in range(episodes):
        env = PortfolioEnv(n_assets=n_assets, T=T, max_trade=max_trade, risk_aversion=risk_aversion)
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 前warm_up轮使用随机策略探索，之后使用当前策略
            if episode < warm_up:
                action = np.random.uniform(-max_trade, max_trade, size=action_dim)
            else:
                action = agent.select_action(state, evaluate=False)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            agent.replay_buffer.add(state, action, reward, next_state, float(done))
            state = next_state
            
            # 训练：warm_up后每步训练5次（提高样本效率）
            if episode >= warm_up and len(agent.replay_buffer) > batch_size:
                for _ in range(5):  # 每步更新5次
                    logs = agent.train(batch_size)
                
                # 学习率衰减（每episode进行一次）
                scheduler_actor.step()
                scheduler_critic.step()
        
        episode_rewards.append(episode_reward)
        
        # 定期评估（确定性策略，200个episode取平均更稳定）
        if (episode + 1) % evaluate_freq == 0:
            eval_reward = evaluate_policy(
                agent, n_assets, T, max_trade, risk_aversion, 
                episodes=200  # 增加到200次评估取平均
            )
            eval_rewards.append((episode + 1, eval_reward))
            
            # 计算最近100个训练episode的平均奖励
            avg_train = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            # 打印详细日志
            # if logs:
            #     print(f"Ep {episode+1:5d} | "
            #           f"Train: {avg_train_reward:.4f} | "
            #           f"Eval: {eval_reward:.4f} | "
            #           f"Q1: {logs['q1_mean']:.3f} | "
            #           f"Q2: {logs['q2_mean']:.3f} | "
            #           f"Q_diff: {logs['q_diff']:.4f} | "
            #           f"Alpha: {logs['alpha']:.3f} | "
            #           f"LR: {scheduler_actor.get_last_lr()[0]:.6f}")
            # else:
            #     print(f"Ep {episode+1:5d} | Train: {avg_train_reward:.4f} | Eval: {eval_reward:.4f} [WarmUp]")
            
                        # 修改打印逻辑，安全访问 logs
            if episode >= warm_up and logs is not None:  # <<<< 修改条件
                print(f"Ep {episode+1:5d} | Train: {avg_train:.4f} | Eval: {eval_reward:.4f} | "
                      f"Q1: {logs['q1_mean']:.3f} | Q2: {logs['q2_mean']:.3f} | "
                      f"Q_diff: {logs['q_diff']:.4f} | Alpha: {logs['alpha']:.3f} | "
                      f"LR: {scheduler_actor.get_last_lr()[0]:.6f}")
            else:
                status = "WarmUp" if episode < warm_up else "BufferLow"
                print(f"Ep {episode+1:5d} | Train: {avg_train:.4f} | Eval: {eval_reward:.4f} [{status}]")

            # 早停检查
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                patience_counter = 0
                agent.save(best_model_path)
                print(f"  >>> 新的最佳模型保存: {best_eval_reward:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发于 Episode {episode+1}")
                    print(f"最佳评估奖励: {best_eval_reward:.4f}")
                    break
    
    # 加载最佳模型用于最终评估
    print(f"\n加载最佳模型: {best_model_path}")
    agent.load(best_model_path)
    
    # 最终全面评估
    final_eval = evaluate_policy(agent, n_assets, T, max_trade, risk_aversion, episodes=500)
    print(f"最终评估 (500 episodes): {final_eval:.4f}")
    
    # 绘制学习曲线
    plot_learning_curve(episode_rewards, eval_rewards, n_assets, T, best_eval_reward)
    
    return agent, episode_rewards


def evaluate_policy(agent, n_assets, T, max_trade, risk_aversion, episodes=200):
    """评估当前策略（确定性，无探索）"""
    total_reward = 0
    env = PortfolioEnv(n_assets=n_assets, T=T, max_trade=max_trade, risk_aversion=risk_aversion)
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _ = env.step(action)
        total_reward += reward
        
    return total_reward / episodes


def plot_learning_curve(episode_rewards, eval_rewards, n_assets, T, best_reward):
    """绘制学习曲线（双Y轴显示训练与评估）"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 训练奖励（移动平均）
    window = 100
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), smoothed, 
                'b-', alpha=0.6, label=f'Training (MA{window})')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Training CARA Utility', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # 评估奖励（红色点）
    if eval_rewards:
        episodes, rewards = zip(*eval_rewards)
        ax2 = ax1.twinx()
        ax2.plot(episodes, rewards, 'r-o', label='Evaluation', markersize=6, linewidth=2)
        ax2.axhline(y=best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_reward:.4f}')
        ax2.set_ylabel('Evaluation CARA Utility', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    if eval_rewards:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax1.legend(loc='upper left')
    
    plt.title(f'SAC Training (Optimized): n={n_assets}, T={T}\nBest Eval Reward: {best_reward:.4f}', 
              fontsize=14)
    plt.tight_layout()
    plt.savefig(f'training_optimized_n{n_assets}_T{T}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存: training_optimized_n{n_assets}_T{T}.png")


if __name__ == "__main__":
    # 完整调优训练
    agent, rewards = train_sac_optimized(
        n_assets=2,
        T=5,
        episodes=15000,
        batch_size=256,
        warm_up=1000,
        patience=10
    )
    
    print("\n训练完成！")
    print(f"训练轮数: {len(rewards)}")
    print(f"最后100轮平均: {np.mean(rewards[-100:]):.4f}")