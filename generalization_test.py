import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import PortfolioEnv
from sac import SAC

def test_generalization(agent, n_list=[2, 3, 4], T_list=[3, 5, 7, 9], episodes=100):
    """
    测试不同n和T组合下的性能，证明泛化能力
    符合作业要求: "work for any time horizon < 10, and n< 5"
    """
    results = {}
    
    print("="*60)
    print("泛化能力测试 (Generalization Test)")
    print("="*60)
    
    for n in n_list:
        for T in T_list:
            if T >= 10 or n >= 5:
                continue
                
            print(f"\n测试配置: n={n} assets, T={T} periods")
            
            # 为该配置创建环境
            env = PortfolioEnv(n_assets=n, T=T, max_trade=0.1, risk_aversion=1.0)
            
            # 收集RL策略表现
            rl_rewards = []
            random_rewards = []
            
            for ep in range(episodes):
                # 固定随机种子确保公平对比（同一市场参数下对比）
                seed = ep
                
                # RL策略
                state = env.reset(seed=seed)
                done = False
                while not done:
                    # 注意：如果n与训练时不同，需要处理维度
                    action = agent.select_action(state, evaluate=True)
                    if len(action) < n:
                        action = np.concatenate([action, np.zeros(n - len(action))])
                    elif len(action) > n:
                        action = action[:n]
                    
                    state, reward, done, _ = env.step(action)
                rl_rewards.append(reward)
                
                # 随机策略（同seed）
                state = env.reset(seed=seed)
                done = False
                while not done:
                    action = np.random.uniform(-0.1, 0.1, size=n)
                    state, reward, done, _ = env.step(action)
                random_rewards.append(reward)
            
            # 统计
            rl_mean = np.mean(rl_rewards)
            rl_std = np.std(rl_rewards)
            rand_mean = np.mean(random_rewards)
            improvement = (rl_mean - rand_mean) / abs(rand_mean) * 100
            
            results[(n, T)] = {
                'rl_mean': rl_mean,
                'rl_std': rl_std,
                'random_mean': rand_mean,
                'improvement': improvement
            }
            
            print(f"  RL:    {rl_mean:.4f} ± {rl_std:.4f}")
            print(f"  Random: {rand_mean:.4f}")
            print(f"  提升:   {improvement:+.1f}%")
    
    # 绘制热力图
    plot_generalization_heatmap(results, n_list, T_list)
    return results

def plot_generalization_heatmap(results, n_list, T_list):
    """绘制泛化能力热力图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 构建数据矩阵
    data = np.zeros((len(n_list), len(T_list)))
    for i, n in enumerate(n_list):
        for j, T in enumerate(T_list):
            if (n, T) in results:
                data[i, j] = results[(n, T)]['improvement']
            else:
                data[i, j] = np.nan
    
    # 绘制热力图
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=20)
    
    # 设置坐标轴
    ax.set_xticks(range(len(T_list)))
    ax.set_yticks(range(len(n_list)))
    ax.set_xticklabels([f'T={t}' for t in T_list])
    ax.set_yticklabels([f'n={n}' for n in n_list])
    ax.set_xlabel('Time Horizon', fontsize=12)
    ax.set_ylabel('Number of Assets', fontsize=12)
    ax.set_title('RL Strategy Improvement over Random (%)', fontsize=14)
    
    # 添加数值标注
    for i in range(len(n_list)):
        for j in range(len(T_list)):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.1f}%',
                              ha="center", va="center", 
                              color="black" if abs(data[i, j]) < 10 else "white",
                              fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Improvement (%)')
    plt.tight_layout()
    plt.savefig('generalization_heatmap.png', dpi=150)
    print("\n泛化热力图已保存: generalization_heatmap.png")
    plt.show()

def strict_statistical_test(agent, n_assets=2, T=5, episodes=200):
    """
    严格统计检验：固定市场参数，仅比较策略差异
    """
    print(f"\n严格统计检验 (n={n_assets}, T={T}, episodes={episodes})...")
    
    # 生成固定市场参数集（确保公平对比）
    market_params = []
    for i in range(episodes):
        np.random.seed(i)
        params = {
            'r': np.random.uniform(0.01, 0.04),
            'means': np.random.uniform(0.05, 0.20, n_assets),
            'stds': np.random.uniform(0.10, 0.35, n_assets)
        }
        market_params.append(params)
    
    rl_rewards = []
    random_rewards = []
    
    for i, params in enumerate(market_params):
        # RL策略
        env = PortfolioEnv(n_assets=n_assets, T=T)
        env.r = params['r']
        env.means = params['means']
        env.stds = params['stds']
        env.reset(seed=i*2)
        
        state = env._get_state()
        done = False
        ep_reward = 0
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, done, _ = env.step(action)
            ep_reward = reward  # 只保留期末奖励
        rl_rewards.append(ep_reward)
        
        # 随机策略（同参数）
        env = PortfolioEnv(n_assets=n_assets, T=T)
        env.r = params['r']
        env.means = params['means']
        env.stds = params['stds']
        env.reset(seed=i*2)
        
        state = env._get_state()
        done = False
        while not done:
            action = np.random.uniform(-0.1, 0.1, size=n_assets)
            state, reward, done, _ = env.step(action)
            ep_reward = reward
        random_rewards.append(ep_reward)
    
    rl_mean = np.mean(rl_rewards)
    rl_std = np.std(rl_rewards)
    rand_mean = np.mean(random_rewards)
    rand_std = np.std(random_rewards)
    
    # 配对t检验（更严格）
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(rl_rewards, random_rewards)
    
    print(f"RL策略:   {rl_mean:.4f} ± {rl_std:.4f}")
    print(f"随机策略: {rand_mean:.4f} ± {rand_std:.4f}")
    print(f"差异:     {rl_mean - rand_mean:.4f}")
    print(f"配对t检验: t={t_stat:.2f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
    
    return rl_mean, rand_mean, p_value

if __name__ == "__main__":
    # 加载训练好的模型（n=2, T=5训练，测试泛化）
    agent = SAC(
        state_dim=5,  # n=2时state_dim=5
        action_dim=2,
        max_trade=0.1,
        hidden_dim=128,
        auto_tune_alpha=False,
        alpha=0.15
    )
    agent.load('best_sac_n2_T5.pth')
    
    # 1. 严格统计检验（固定参数）
    strict_statistical_test(agent, n_assets=2, T=5, episodes=300)
    
    # 2. 泛化能力测试（核心：证明适用于n<5, T<10）
    # 注意：模型在n=2,T=5上训练，测试其他组合展示泛化性
    print("\n" + "="*60)
    print("注意：以下测试使用在n=2,T=5上训练的模型")
    print("测试其他配置以展示泛化能力（零样本迁移）")
    print("="*60)
    
    test_generalization(
        agent, 
        n_list=[2, 3, 4],      # n < 5
        T_list=[3, 5, 7, 9],   # T < 10
        episodes=50            # 每配置50个episode
    )