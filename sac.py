import torch
import torch.nn.functional as F
import numpy as np
from model import Actor, Critic
from replay_buffer import ReplayBuffer

class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_trade=0.1,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.15,  # 调整为0.15（平衡探索与利用）
        auto_tune_alpha=False,
        buffer_size=int(5e5),
        device=None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_tune_alpha = auto_tune_alpha
        self.action_dim = action_dim
        
        # 网络初始化（增大到128）
        self.actor = Actor(state_dim, action_dim, max_trade, hidden_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)
        
        if self.auto_tune_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.total_it = 0
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            return self.actor.get_action(state, deterministic=True)
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()
    
    def train(self, batch_size=256):
        """
        改进的训练循环：
        1. Huber Loss 替代 MSE（对异常值鲁棒）
        2. Delayed Policy Update（每2步更新一次Actor）
        3. 详细的训练日志
        """
        if len(self.replay_buffer) < batch_size:
            return None
            
        self.total_it += 1
        
        # 采样
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        # ===== 1. 更新 Critic =====
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        # Huber Loss (Smooth L1) 替代 MSE，对异常值更鲁棒
        critic_loss = F.smooth_l1_loss(current_q1, target_q) + \
                      F.smooth_l1_loss(current_q2, target_q)
        
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪（防止爆炸）
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        # ===== 2. Delayed Policy Update（每2步更新一次Actor）=====
        actor_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)
        
        if self.total_it % 2 == 0:  # 关键：降低策略更新频率，提高稳定性
            new_action, log_prob = self.actor.sample(state)
            q1_new = self.critic_1(state, new_action)
            q2_new = self.critic_2(state, new_action)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_prob - q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            
            # 更新 Alpha
            if self.auto_tune_alpha:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            
            # 软更新目标网络（只在Actor更新时进行）
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0.0,
            'alpha': self.alpha if not self.auto_tune_alpha else self.alpha.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'q_diff': abs(current_q1.mean().item() - current_q2.mean().item())  # 监控双Q差异
        }
    
    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'alpha': self.alpha
        }, filename)
        
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        if 'alpha' in checkpoint:
            self.alpha = checkpoint['alpha']