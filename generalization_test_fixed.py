import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import PortfolioEnv
from sac import SAC

def load_model_for_n(n_assets, T_train=5):
    """根据n加载对应模型"""
    state_dim = 1 + (n_assets + 1) + 1
    action_dim = n_assets
    
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_trade=0.1,
        hidden_dim=128,
        auto_tune_alpha=False,
        alpha=0.15
    )
    
    model_path = f"best_sac_n{n_assets}_T{T_train}.pth"
    try:
        agent.load(model_path)
        print(f"成功加载模型: {model_path}")
        return agent
    except:
        print(f"警告：未找到 {model_path}，请确保已训练")
        return None

def test_full_generalization(n_list=[2, 3, 4], T_list=[3, 5, 7, 9], episodes=100):
    """
    完整泛化测试：为每个n使用专用模型，测试不同T
    证明：work for any time horizon < 10, and n < 5
    """
    results = {}
    
    print("="*70)
    print("完整泛化能力验证 (Any n<5, Any T<10)")
    print("="*70)
    
    for n in n_list:
        # 为该n加载专用模型
        agent = load_model_for_n(n)
        if agent is None:
            continue
            
        for T in T_list:
            if T >= 10:
                continue
                
            print(f"\n测试: n={n}, T={T} (模型在n={n},T=5上训练)")
            
            # 测试RL vs Random
            rl_rewards = []
            random_rewards = []
            
            for seed in range(episodes):
                # RL策略
                env = PortfolioEnv(n_assets=n, T=T, max_trade=0.1)
                state = env.reset(seed=seed)
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)
                    state, reward, done, _ = env.step(action)
                rl_rewards.append(reward)
                
                # 随机策略（同seed）
                env = PortfolioEnv(n_assets=n, T=T, max_trade=0.1)
                state = env.reset(seed=seed)
                done = False
                while not done:
                    action = np.random.uniform(-0.1, 0.1, size=n)
                    state, reward, done, _ = env.step(action)
                random_rewards.append(reward)
            
            # 统计
            rl_mean = np.mean(rl_rewards)
            rand_mean = np.mean(random_rewards)
            improvement = (rl_mean - rand_mean) / abs(rand_mean) * 100
            
            results[(n, T)] = {
                'rl': rl_mean,
                'random': rand_mean,
                'improvement': improvement
            }
            
            print(f"  RL: {rl_mean:.4f} | Random: {rand_mean:.4f} | 提升: {improvement:+.1f}%")
    
    # 绘制结果表格和热力图
    plot_results_table(results, n_list, T_list)
    return results

def plot_results_table(results, n_list, T_list):
    """绘制结果表格"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 构建表格数据
    headers = ['n'] + [f'T={t}' for t in T_list]
    rows = []
    
    for n in n_list:
        row = [f'n={n}']
        for T in T_list:
            if (n, T) in results:
                imp = results[(n, T)]['improvement']
                row.append(f"{imp:+.1f}%")
            else:
                row.append("N/A")
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # 颜色标注
    for i, n in enumerate(n_list):
        for j, T in enumerate(T_list):
            if (n, T) in results:
                imp = results[(n, T)]['improvement']
                color = plt.cm.RdYlGn((imp + 10) / 50)  # 归一化到颜色
                table[(i+1, j+1)].set_facecolor(color)
    
    plt.title('RL Strategy Improvement over Random Policy (%)', fontsize=14, pad=20)
    plt.savefig('generalization_table.png', dpi=150, bbox_inches='tight')
    print("\n泛化表格已保存: generalization_table.png")
    plt.show()

if __name__ == "__main__":
    # 运行完整验证
    results = test_full_generalization(
        n_list=[2, 3, 4],
        T_list=[3, 5, 7, 9],  # < 10
        episodes=100
    )