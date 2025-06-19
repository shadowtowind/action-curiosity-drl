import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

class Actor(nn.Module):
	def __init__(self, obs_dim, act_dim):
		super().__init__()
		self.pi = nn.Sequential(
			nn.Linear(obs_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU()
		)
		self.mu_layer = nn.Linear(256, act_dim)  # 输出均值
		self.log_std_layer = nn.Linear(256, act_dim)  # 输出标准差的对数

	def forward(self, obs):
		pi_output = self.pi(obs)
		mu = self.mu_layer(pi_output)
		log_std = self.log_std_layer(pi_output)
		log_std = torch.clamp(log_std, -20, 2)  # 限制标准差的范围
		std = torch.exp(log_std)
		return mu, std

	def sample(self, obs):
		mu, std = self.forward(obs)
		dist = torch.distributions.Normal(mu, std)
		action = dist.rsample()  # 重参数化技巧
		log_prob = dist.log_prob(action).sum(axis=-1)  # 计算动作的对数概率
		log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(axis=-1)  # 修正tanh的影响
		action = torch.tanh(action)  # 将动作限制在[-1, 1]范围内
		return action, log_prob

class QFunction(nn.Module):
	def __init__(self, obs_dim, act_dim):
		super().__init__()
		self.q = nn.Sequential(
			nn.Linear(obs_dim + act_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Identity()
		)

	def forward(self, obs, act):
		q = self.q(torch.cat([obs, act], dim=-1))
		return torch.squeeze(q, -1)  # 确保q的形状正确

class SAC(nn.Module):
	def __init__(self, obs_dim, act_dim):
		super().__init__()
		print('DRL_Model: SAC')
		self.pi = Actor(obs_dim, act_dim)  # 策略网络
		self.q1 = QFunction(obs_dim, act_dim)  # 第一个Q网络
		self.q2 = QFunction(obs_dim, act_dim)  # 第二个Q网络

		# 温度参数（可学习）
		self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True)
		self.alpha = self.log_alpha.exp()
		
		self.target_entropy = -torch.prod(torch.Tensor(act_dim)).item()

	def act(self, obs):
		with torch.no_grad():
			action, _ = self.pi.sample(obs)
			return action.numpy()
