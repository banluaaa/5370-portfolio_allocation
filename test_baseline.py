from environment import PortfolioEnv
import numpy as np

# 检查A：维度一致性
env = PortfolioEnv(n_assets=3, T=5)
state = env.reset()
print(state.shape)  # 应该是 (5,) 即 1+logW + 4持仓 + 1时间

action = np.array([0.05, -0.03, 0.02])  # 和为0.04，L1=0.10（恰好满足）
next_state, reward, done, info = env.step(action)
assert next_state.shape == state.shape
print("✓ 维度检查通过")

# 检查B：约束满足性
# 测试极端动作：给极大的动作，看是否被裁剪到10%
env = PortfolioEnv(n_assets=2, max_trade=0.1)
env.reset()
big_action = np.array([0.5, 0.5])  # 想调整50%
_, _, _, info = env.step(big_action)
applied = info['action_applied']
assert np.sum(np.abs(applied)) <= 0.1001, f"L1范数{np.sum(np.abs(applied))}超过0.1"
print(f"✓ 约束检查通过：输入[0.5,0.5]被裁剪为{applied}, L1范数={np.sum(np.abs(applied)):.3f}")

# 检查C：自融资约束
# 检查持仓总和是否始终为1
env = PortfolioEnv(n_assets=3, T=5)
state = env.reset()
for _ in range(5):
    action = np.random.randn(3) * 0.05
    state, reward, done, _ = env.step(action)
    assert abs(np.sum(env.p) - 1.0) < 1e-5, f"持仓和不等于1: {np.sum(env.p)}"
print("✓ 自融资约束检查通过")


# generate baseline
def evaluate_random_policy(n_assets=3, T=5, episodes=1000):
    """评估随机策略的期望效用"""
    env = PortfolioEnv(n_assets=n_assets, T=T)
    utilities = []
    
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            action = np.random.uniform(-0.1, 0.1, size=n_assets)
            _, reward, done, _ = env.step(action)
        utilities.append(reward)  # 期末reward就是CARA效用
    
    return np.mean(utilities), np.std(utilities)

# 运行并记录结果（这将用于证明RL比随机策略好）
mean_util, std_util = evaluate_random_policy(n_assets=2, T=5, episodes=1000)
print(f"随机策略平均效用: {mean_util:.6f} ± {std_util:.6f}")