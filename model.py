import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    """
    策略网络：增大容量 + LayerNorm 稳定训练
    """
    def __init__(self, state_dim, action_dim, max_trade=0.1, hidden_dim=128):
        super(Actor, self).__init__()
        self.max_trade = max_trade
        
        # 增大网络 + LayerNorm 防止内部协变量偏移
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 关键：稳定训练
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 保守初始化（减小初始波动）
        self.log_std_layer.weight.data.fill_(0.0)
        self.log_std_layer.bias.data.fill_(-2.0)  # 初始 std ≈ 0.13
        
    def forward(self, state):
        x = self.feature(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        mean = self.max_trade * torch.tanh(mean)  # 预约束
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = self.max_trade * torch.tanh(x_t)
        
        # 对数概率计算
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_trade * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                return mean.cpu().numpy().flatten()
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.sample()
                action = self.max_trade * torch.tanh(x_t)
                return action.cpu().numpy().flatten()


class Critic(nn.Module):
    """
    Q网络：同步增大 + LayerNorm
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # 小值初始化输出层
        self.network[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.network[-1].bias.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)