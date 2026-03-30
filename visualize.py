import numpy as np
import matplotlib.pyplot as plt
import torch
from environment import PortfolioEnv
from sac import SAC

def load_trained_agent(n_assets=2, T=5, model_path='best_sac_n2_T5.pth'):
    """加载训练好的模型"""
    env_temp = PortfolioEnv(n_assets=n_assets, T=T)
    agent = SAC(
        state_dim=env_temp.state_dim,
        action_dim=env_temp.action_dim,
        max_trade=0.1,
        hidden_dim=128,
        auto_tune_alpha=False,
        alpha=0.15
    )
    agent.load(model_path)
    return agent

def plot_strategy_evolution(agent, n_assets=2, T=5, save_path='strategy_evolution.png'):
    """
    图1：策略随时间的演变（类似附件图8.1）
    展示不同时期Agent的风险敞口决策
    """
    fig, axes = plt.subplots(1, T, figsize=(15, 3))
    
    # 固定市场参数用于对比
    fixed_params = {
        'r': 0.02,
        'means': np.array([0.08, 0.12]) if n_assets >= 2 else np.array([0.10]),
        'stds': np.array([0.15, 0.20]) if n_assets >= 2 else np.array([0.20])
    }
    
    for t in range(T):
        ax = axes[t] if T > 1 else axes
        
        # 测试不同初始持仓下的动作
        holdings = np.linspace(0, 0.8, 50)  # 风险资产持仓比例
        actions = []
        
        for h in holdings:
            # 构建状态： Wealth=1, 持仓=[1-h, h, ...], time=t
            p = np.zeros(n_assets + 1)
            p[0] = 1 - h  # 现金
            if n_assets == 1:
                p[1] = h
            else:
                p[1] = h * 0.6  # 资产1占60%
                p[2] = h * 0.4  # 资产2占40%
            
            state = np.array([0.0] + list(p) + [t/T], dtype=np.float32)
            
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)
                # 记录总风险敞口调整
                total_action = np.sum(action)
                actions.append(total_action)
        
        ax.plot(holdings, actions, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.3, label='Neutral' if t==0 else '')
        ax.set_xlabel('Current Risky Holding')
        ax.set_ylabel('Adjustment (Δ)')
        ax.set_title(f'Time t={t}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.11, 0.11])
    
    plt.suptitle('Optimal Adjustment Policy vs Current Holdings', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"策略演变图已保存: {save_path}")
    plt.show()

def plot_sample_trajectories(agent, n_assets=2, T=5, n_samples=10, save_path='trajectories.png'):
    """
    图2：多条财富轨迹（类似附件图8.2）
    展示不同市场情景下的财富演变
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    all_wealth_paths = []
    all_actions = []
    
    for sample in range(n_samples):
        env = PortfolioEnv(n_assets=n_assets, T=T, max_trade=0.1)
        state = env.reset()
        
        wealth_path = [env.W]
        action_path = []
        risky_holdings = [np.sum(env.p[1:])]
        
        for t in range(T):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, info = env.step(action)
            
            wealth_path.append(env.W)
            action_path.append(np.sum(action))
            risky_holdings.append(np.sum(env.p[1:]))
        
        all_wealth_paths.append(wealth_path)
        all_actions.append(action_path)
    
    # 绘制财富轨迹
    ax1 = axes[0]
    for i, path in enumerate(all_wealth_paths):
        ax1.plot(range(T+1), path, alpha=0.7, linewidth=1.5)
    ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Initial Wealth')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Wealth')
    ax1.set_title(f'Wealth Trajectories ({n_samples} samples)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制风险敞口变化
    ax2 = axes[1]
    for i, holdings in enumerate([[p[1] for p in [env.p]] for _ in range(n_samples)]):  # 简化
        # 重新计算持仓轨迹
        env = PortfolioEnv(n_assets=n_assets, T=T)
        env.reset()
        holdings = [np.sum(env.p[1:])]
        for t in range(T):
            action = agent.select_action(env._get_state(), evaluate=True)
            env.step(action)
            holdings.append(np.sum(env.p[1:]))
        ax2.plot(range(T+1), holdings, alpha=0.7)
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Risky Asset Allocation')
    ax2.set_title('Risk Exposure Evolution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"轨迹图已保存: {save_path}")
    plt.show()

def compare_with_unconstrained_merton(n_assets=1, T=5, episodes=100, save_path='merton_comparison.png'):
    """
    图3：单资产情况下与Merton解析解对比（如果有单资产模型）
    或者展示约束的影响
    """
    if n_assets != 1:
        print("跳过Merton对比（需要单资产模型）")
        return
    
    agent = load_trained_agent(n_assets=1, T=T, model_path='best_sac_n1_T5.pth')
    
    time_steps = []
    rl_allocations = []
    merton_allocations = []
    
    for t in range(T):
        # 测试从50%持仓开始的调整
        alloc = 0
        for _ in range(episodes):
            env = PortfolioEnv(n_assets=1, T=T)
            env.reset()
            env.t = t
            env.W = 1.0
            env.p = np.array([0.5, 0.5])
            
            state = env._get_state()
            action = agent.select_action(state, evaluate=True)[0]
            new_alloc = env.p[1] + action
            alloc += new_alloc
        
        time_steps.append(t)
        rl_allocations.append(alloc / episodes)
        
        # Merton理论值（无约束）
        env = PortfolioEnv(n_assets=1, T=T)
        env.reset()
        env.t = t
        merton = (env.means[0] - env.r) / (env.stds[0]**2 * env.A * (1+env.r)**(T-t-1))
        merton_allocations.append(np.clip(merton, 0, 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, rl_allocations, 'b-o', label='RL Policy (10% Constraint)', linewidth=2, markersize=8)
    plt.plot(time_steps, merton_allocations, 'r--s', label='Merton Optimal (Unconstrained)', linewidth=2, markersize=8)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Allocation to Risky Asset', fontsize=12)
    plt.title('Impact of Trading Constraint on Optimal Policy', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Merton对比图已保存: {save_path}")
    plt.show()

def statistical_test(agent, n_assets=2, T=5, n_episodes=1000):
    """
    统计验证：与随机策略的显著性差异
    """
    print(f"\n统计验证 (n={n_episodes})...")
    
    # 收集RL策略表现
    rl_rewards = []
    for _ in range(n_episodes):
        env = PortfolioEnv(n_assets=n_assets, T=T)
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _ = env.step(action)
        rl_rewards.append(reward)
    
    # 收集随机策略表现
    random_rewards = []
    for _ in range(n_episodes):
        env = PortfolioEnv(n_assets=n_assets, T=T)
        env.reset()
        done = False
        while not done:
            action = np.random.uniform(-0.1, 0.1, size=n_assets)
            state, reward, done, _ = env.step(action)
        random_rewards.append(reward)
    
    rl_mean = np.mean(rl_rewards)
    rl_std = np.std(rl_rewards)
    random_mean = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    
    # 计算提升百分比
    improvement = (rl_mean - random_mean) / abs(random_mean) * 100
    
    print(f"RL策略:   {rl_mean:.4f} ± {rl_std:.4f}")
    print(f"随机策略: {random_mean:.4f} ± {random_std:.4f}")
    print(f"相对提升: {improvement:.1f}%")
    
    # 简单t检验（粗略）
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(rl_rewards, random_rewards)
    print(f"t检验: t={t_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else ''}")

if __name__ == "__main__":
    # 加载最佳模型（注意使用best_前缀）
    agent = load_trained_agent(n_assets=2, T=5, model_path='best_sac_n2_T5.pth')
    
    # 生成图表1：策略演变
    plot_strategy_evolution(agent, n_assets=2, T=5)
    
    # 生成图表2：财富轨迹
    plot_sample_trajectories(agent, n_assets=2, T=5, n_samples=20)
    
    # 统计验证（需要scipy）
    try:
        statistical_test(agent, n_assets=2, T=5, n_episodes=500)
    except ImportError:
        print("安装scipy以进行统计检验: pip install scipy")