import numpy as np
import matplotlib.pyplot as plt
import torch
from environment import PortfolioEnv
from sac import SAC

def load_model(n_assets, T_train=5):
    """加载对应n的模型"""
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
    agent.load(f'best_sac_n{n_assets}_T{T_train}.pth')
    agent.actor.eval()  # 设置为评估模式
    return agent

def plot_comprehensive_strategy_map(save_path='figure1_strategy_map_fixed.png'):
    """
    图1：综合策略地图（修复版）
    """
    agent = load_model(n_assets=2)
    T = 5
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # 固定市场参数
    r = 0.02
    means = np.array([0.08, 0.12])
    stds = np.array([0.15, 0.20])
    
    # 收集所有数据以统一颜色范围
    all_delta1 = []
    data_storage = []
    
    for t in range(T):
        p1_range = np.linspace(0.1, 0.7, 25)
        p2_range = np.linspace(0.1, 0.7, 25)
        P1, P2 = np.meshgrid(p1_range, p2_range)
        Delta1 = np.zeros_like(P1)
        
        for i in range(len(p1_range)):
            for j in range(len(p2_range)):
                p1 = P1[j, i]
                p2 = P2[j, i]
                p0 = 1 - p1 - p2
                
                if p0 < 0 or p0 > 1:
                    Delta1[j, i] = np.nan
                    continue
                
                state = np.array([0.0, p0, p1, p2, t/T], dtype=np.float32)
                with torch.no_grad():
                    action = agent.select_action(state, evaluate=True)
                    Delta1[j, i] = action[0]
        
        all_delta1.append(Delta1)
        data_storage.append((P1, P2, Delta1, p1_range, p2_range))
    
    # 统一颜色范围
    vmin, vmax = -0.1, 0.1
    
    # 绘制每个子图
    for t in range(T):
        ax = axes[t]
        P1, P2, Delta1, p1_range, p2_range = data_storage[t]
        
        # 使用pcolormesh替代contourf更清晰
        im = ax.pcolormesh(P1, P2, Delta1, shading='gouraud', 
                          cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.9)
        
        # 零线（黑色实线）
        ax.contour(P1, P2, Delta1, levels=[0], colors='black', linewidths=2.5)
        
        # 可行域边界（现金非负约束：p0 >= 0即p1+p2 <= 1）
        ax.plot([0.1, 0.9], [0.9, 0.1], 'k--', linewidth=2, alpha=0.5, label='Cash=0')
        
        # 标记中性持仓点（约0.25,0.25）
        ax.plot(0.25, 0.25, 'wo', markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        ax.set_xlabel('Asset 1 Holding', fontsize=11)
        ax.set_ylabel('Asset 2 Holding', fontsize=11)
        ax.set_title(f'Time t={t} (Remaining: {T-t} periods)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.1, 0.7])
        ax.set_ylim([0.1, 0.7])
        
        # 为每个子图添加小型颜色条（右侧）
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=15, pad=0.02)
        cbar.set_label('Δ₁', rotation=0, labelpad=10)
        cbar.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
    
    # 删除多余的子图（第6个）
    fig.delaxes(axes[-1])
    
    # 添加总标题
    plt.suptitle('Optimal Policy Heatmap: Adjustment for Asset 1\n(CARA Utility, 10% Trading Constraint)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 调整布局避免重叠
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"修复后的策略地图已保存: {save_path}")
    plt.show()

def plot_wealth_trajectories_comparison(save_path='figure2_trajectories.png'):
    """
    图2：财富轨迹对比（RL vs Random）
    展示相同市场条件下，两种策略的表现差异
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_assets, T = 2, 5
    n_samples = 50
    
    # 固定市场参数（确保公平对比）
    np.random.seed(42)
    market_params = []
    for _ in range(n_samples):
        params = {
            'r': np.random.uniform(0.01, 0.04),
            'means': np.random.uniform(0.05, 0.20, n_assets),
            'stds': np.random.uniform(0.10, 0.35, n_assets)
        }
        market_params.append(params)
    
    # 加载RL模型
    agent = load_model(n_assets=n_assets)
    
    # 收集轨迹
    rl_paths = []
    random_paths = []
    
    for params in market_params:
        # RL轨迹
        env = PortfolioEnv(n_assets=n_assets, T=T)
        env.r, env.means, env.stds = params['r'], params['means'], params['stds']
        state = env.reset()
        path = [env.W]
        for _ in range(T):
            action = agent.select_action(state, evaluate=True)
            state, _, done, _ = env.step(action)
            path.append(env.W)
            if done: break
        rl_paths.append(path)
        
        # Random轨迹
        env = PortfolioEnv(n_assets=n_assets, T=T)
        env.r, env.means, env.stds = params['r'], params['means'], params['stds']
        state = env.reset()
        path = [env.W]
        for _ in range(T):
            action = np.random.uniform(-0.1, 0.1, size=n_assets)
            state, _, done, _ = env.step(action)
            path.append(env.W)
            if done: break
        random_paths.append(path)
    
    # 左图：RL策略
    ax = axes[0]
    for path in rl_paths:
        ax.plot(range(T+1), path, alpha=0.4, linewidth=1)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Initial Wealth')
    ax.set_xlabel('Time')
    ax.set_ylabel('Wealth')
    ax.set_title('RL Strategy (SAC)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 右图：Random策略
    ax = axes[1]
    for path in random_paths:
        ax.plot(range(T+1), path, alpha=0.4, linewidth=1, color='orange')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Wealth')
    ax.set_title('Random Strategy (Baseline)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加统计标注
    rl_final = [p[-1] for p in rl_paths]
    rand_final = [p[-1] for p in random_paths]
    fig.text(0.5, 0.02, f'Mean Final Wealth: RL={np.mean(rl_final):.2f} vs Random={np.mean(rand_final):.2f}', 
             ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle('Wealth Trajectories Comparison (50 Samples)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"轨迹对比图已保存: {save_path}")
    plt.show()

def plot_performance_bar_chart(save_path='figure3_performance.png'):
    """
    图3：性能对比柱状图（基于泛化测试结果）
    """
    # 使用泛化测试的数据
    data = {
        (2, 3): 7.5, (2, 5): 9.3, (2, 7): 15.1, (2, 9): 31.2,
        (3, 3): 2.3, (3, 5): 15.8, (3, 7): 20.7, (3, 9): 27.1,
        (4, 3): 5.0, (4, 5): 7.9, (4, 7): 14.1, (4, 9): 25.3
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(4)  # T=3,5,7,9
    width = 0.25
    
    # 三个n的值
    n2_vals = [data[(2, t)] for t in [3, 5, 7, 9]]
    n3_vals = [data[(3, t)] for t in [3, 5, 7, 9]]
    n4_vals = [data[(4, t)] for t in [3, 5, 7, 9]]
    
    bars1 = ax.bar(x - width, n2_vals, width, label='n=2', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x, n3_vals, width, label='n=3', color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, n4_vals, width, label='n=4', color='#e74c3c', alpha=0.8)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Improvement over Random (%)', fontsize=12)
    ax.set_xlabel('Time Horizon (T)', fontsize=12)
    ax.set_title('RL Strategy Performance Across Different Configurations\n(CARA Utility Improvement)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['T=3', 'T=5', 'T=7', 'T=9'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"性能对比图已保存: {save_path}")
    plt.show()


def plot_risk_evolution_comparison(save_path='figure4_risk_evolution_combined.png'):
    """
    修复版：将n=2,3,4的风险演变合并为一张对比图（避免覆盖）
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    T = 5
    n_episodes = 30  # 每配置30条轨迹
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # 绿、蓝、红
    
    for idx, n_assets in enumerate([2, 3, 4]):
        ax = axes[idx]
        agent = load_model(n_assets=n_assets)
        
        all_holdings = []
        
        for seed in range(n_episodes):
            env = PortfolioEnv(n_assets=n_assets, T=T)
            state = env.reset(seed=seed)
            holdings = [np.sum(env.p[1:])]  # 初始风险敞口
            
            for t in range(T):
                action = agent.select_action(state, evaluate=True)
                state, _, done, _ = env.step(action)
                holdings.append(np.sum(env.p[1:]))
                if done: break
            
            all_holdings.append(holdings)
            # 绘制半透明单轨迹
            ax.plot(range(T+1), holdings, alpha=0.3, linewidth=1, color=colors[idx])
        
        # 计算并绘制平均路径（粗线）
        mean_path = np.mean(all_holdings, axis=0)
        std_path = np.std(all_holdings, axis=0)
        
        ax.plot(range(T+1), mean_path, color='black', linewidth=3, 
                label=f'Mean (n={n_assets})', marker='o', markersize=6)
        
        # 添加标准差阴影
        ax.fill_between(range(T+1), 
                       mean_path - std_path, 
                       mean_path + std_path, 
                       alpha=0.2, color=colors[idx])
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Risky Asset Allocation', fontsize=12)
        ax.set_title(f'n = {n_assets} Assets\n(Layer {idx+1})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.set_xticks(range(T+1))
        
        # 添加统计标注
        final_alloc = [h[-1] for h in all_holdings]
        ax.text(0.5, 0.95, f'Final: {np.mean(final_alloc):.2f}±{np.std(final_alloc):.2f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Risk Exposure Evolution Across Different Portfolio Sizes\n(Time-Dependent Risk Aversion)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"合并风险演变图已保存: {save_path}")
    plt.show()

# 在main中替换原调用：
if __name__ == "__main__":
    print("正在生成可视化图表...")
    
    plot_comprehensive_strategy_map()
    plot_wealth_trajectories_comparison()
    plot_performance_bar_chart()
    
    # 替换原有的三次单独调用为一次合并调用
    plot_risk_evolution_comparison()  # 生成一张包含n=2,3,4的对比图
    
    print("\n所有图表生成完成！")