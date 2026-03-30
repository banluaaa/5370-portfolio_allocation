import numpy as np
from train import train_sac_optimized

# 作业要求：n < 5 (即2,3,4)，T < 10 (训练时固定T=5，测试时泛化)
configs = [
    (2, 5),  # 已训练，可跳过或重新训练
    (3, 5),  # 需要训练：state_dim = 1 + 4 + 1 = 6
    (4, 5),  # 需要训练：state_dim = 1 + 5 + 1 = 7
]

for n_assets, T in configs:
    print(f"\n{'='*60}")
    print(f"训练模型: n_assets={n_assets}, T={T}")
    print(f"State dimension: {1 + (n_assets + 1) + 1}")
    print(f"{'='*60}\n")
    
    agent, rewards = train_sac_optimized(
        n_assets=n_assets,
        T=T,
        episodes=12000,      # 稍减少轮数，因为n增大复杂度增加
        batch_size=256,
        warm_up=1000,
        patience=10
    )
    
    print(f"\n模型 n={n_assets} 训练完成，最佳奖励: {max(rewards):.4f}")