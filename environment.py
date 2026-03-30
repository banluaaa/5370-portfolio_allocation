import numpy as np
import torch
from typing import Tuple, Dict, Optional

class PortfolioEnv:
    """
    多资产投资组合分配环境（离散时间）
    
    状态空间: [log(W), p_0, p_1, ..., p_n, t/T]
    动作空间: [delta_1, ..., delta_n] (风险资产调整量)
    约束: ||delta||_1 <= max_trade, delta_0 = -sum(delta) (自融资)
    """
    
    def __init__(
        self, 
        n_assets: int = 3, 
        T: int = 5, 
        max_trade: float = 0.1,
        risk_aversion: float = 1.0,
        initial_wealth: float = 1.0,
        trans_cost: float = 0.0  # 预留参数，默认0（无交易成本）
    ):
        assert n_assets >= 1, "至少需要1个风险资产"
        assert T >= 2, "时间范围至少为2期"
        assert 0 < max_trade <= 1.0, "交易限制应在(0,1]之间"
        
        self.n = n_assets          # 风险资产数量
        self.T = T                 # 总期数
        self.max_trade = max_trade # L1约束阈值（如0.1）
        self.A = risk_aversion     # CARA系数 a
        self.W0 = initial_wealth   # 初始财富
        self.trans_cost = trans_cost
        
        # 状态维度：logW(1) + 持仓(n+1) + 时间特征(1)
        self.state_dim = 1 + (n_assets + 1) + 1
        # 动作维度：只控制n个风险资产，现金自动平衡
        self.action_dim = n_assets
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境，生成新的市场参数和初始状态"""
        if seed is not None:
            np.random.seed(seed)
            
        # 随机生成市场参数（确保合理性）
        self.r = np.random.uniform(0.01, 0.04)  # 无风险利率 1-4%
        self.means = np.random.uniform(0.05, 0.20, self.n)  # 风险溢价5-20%
        self.stds = np.random.uniform(0.10, 0.35, self.n)   # 波动率10-35%
        
        # 初始持仓：均匀分配（可修改）
        # p = [cash, asset1, asset2, ..., assetn]
        self.p = np.zeros(self.n + 1)
        self.p[0] = 0.5  # 50%现金
        remaining = 0.5
        for i in range(1, self.n + 1):
            self.p[i] = remaining / self.n
            
        self.W = self.W0
        self.t = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """构建状态向量"""
        # 对财富取log，防止数值不稳定（ Wealth可能跨越多个数量级）
        log_wealth = np.log(self.W) if self.W > 1e-6 else -20.0
        time_feat = self.t / self.T  # 归一化到[0,1)
        
        state = np.concatenate([
            [log_wealth],      # [1]
            self.p,            # [n+1]
            [time_feat]        # [1]
        ]).astype(np.float32)
        
        return state

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步转移
        
        Args:
            action: np.array shape (n,), 原始调整意向（未约束）
            
        Returns:
            next_state: 新状态
            reward: 即时奖励（期末前为0，期末为CARA效用）
            done: 是否结束
            info: 调试信息
        """
        # 防御性编程：确保 action 是一维数组
        action = np.asarray(action).flatten()
    
        # 1. 动作约束投影（L1球投影）
        delta_risky = self._project_action(action)
        delta_cash = -np.sum(delta_risky)  # 自融资约束
        
        # 组合完整调整向量 [delta_cash, delta_1, ..., delta_n]
        delta = np.concatenate([[delta_cash], delta_risky])
        
        # 2. 执行调整前的检查（防止负持仓）
        temp_p = self.p + delta
        if np.any(temp_p < -1e-6):
            # 如果调整导致负持仓，截断到0（简化处理）
            # 更严格的做法是拒绝该动作或惩罚
            temp_p = np.maximum(temp_p, 0)
            temp_p = temp_p / np.sum(temp_p)  # 重新归一化
        
        # 3. 市场冲击：生成收益率
        # R ~ N(mean, std^2)
        returns = np.random.normal(self.means, self.stds, self.n)
        total_return = temp_p[0] * self.r + np.sum(temp_p[1:] * returns)
        
        # 4. 更新财富
        new_W = self.W * (1 + total_return)
        
        # 5. 更新持仓比例（市场波动后的再平衡）
        # 新持仓 = (旧持仓 * (1+收益率)) / (1+总收益率)
        growth_factors = np.concatenate([[1 + self.r], 1 + returns])
        if abs(1 + total_return) < 1e-8:
            # 防止除0（极端情况）
            new_p = temp_p.copy()
        else:
            new_p = temp_p * growth_factors / (1 + total_return)
        
        # 确保归一化（数值精度修正）
        new_p = new_p / np.sum(new_p)
        
        # 6. 更新内部状态
        self.p = new_p
        self.W = new_W
        self.t += 1
        
        # 7. 计算奖励（稀疏奖励：只有期末有）
        done = (self.t == self.T)
        reward = 0.0

        # 中间奖励：每一步的收益率（鼓励增长）
        intermediate_reward = total_return * 0.1  # 缩小避免主导

        if done:
            # 期末CARA效用（主奖励）
            terminal_reward = -np.exp(-self.A * self.W) / self.A
            # 组合：中间奖励累积 + 期末大奖励
            reward = terminal_reward + intermediate_reward
        else:
            reward = intermediate_reward
            
        info = {
            'wealth': self.W,
            'portfolio_return': total_return,
            'action_applied': delta_risky,
            'cash_ratio': self.p[0]
        }
        
        return self._get_state(), float(reward), done, info
    
    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """
        将动作投影到L1球上（满足10%调整限制）
        算法：如果L1范数<=max_trade，直接返回；否则等比例缩放
        """
        action = np.clip(action, -self.max_trade, self.max_trade)
        l1_norm = np.sum(np.abs(action))
        
        if l1_norm > self.max_trade:
            # 投影到L1球边界
            action = action * (self.max_trade / l1_norm)
            
        return action
    
    def get_merton_ratio(self) -> np.ndarray:
        """
        计算当前市场参数下的Merton最优比例（理论基准）
        仅适用于无约束情况，用于验证
        x* = (mu - r) / (sigma^2 * A * (1+r)^(T-t-1))
        """
        remaining_steps = self.T - self.t
        if remaining_steps <= 0:
            return np.zeros(self.n)
            
        drift_term = (self.means - self.r) / (self.stds**2 * self.A * (1 + self.r)**(remaining_steps - 1))
        return drift_term

# 简单的测试脚本
if __name__ == "__main__":
    print("测试环境初始化...")
    env = PortfolioEnv(n_assets=2, T=5, max_trade=0.1)
    
    # 测试1：随机策略跑通
    state = env.reset(seed=42)
    print(f"初始状态维度: {state.shape}, 期望: ({env.state_dim},)")
    print(f"初始持仓: {env.p}")
    
    total_reward = 0
    for step in range(5):
        # 随机动作（高斯噪声）
        action = np.random.randn(env.action_dim) * 0.05
        state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: Wealth={info['wealth']:.4f}, "
              f"Cash={info['cash_ratio']:.2%}, Reward={reward:.6f}")
        if done:
            break
    
    print(f"\n最终财富: {env.W:.4f}")
    print(f"累计奖励(CARA效用): {total_reward:.6f}")
    print(f"理论Merton比例(第0期): {env.get_merton_ratio()}")