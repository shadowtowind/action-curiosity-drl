import numpy as np
import torch
import torch.nn as nn
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
		self.pi_lin = nn.Sequential(
			nn.Linear(256, 1),
			nn.Sigmoid()
		)
		self.pi_ang = nn.Sequential(
			nn.Linear(256, 1),
			nn.Tanh()
		)

	def forward(self, obs):
		# Return output from network scaled to action space limits.
		obs_vel = self.pi(obs)
		lin_vel = self.pi_lin(obs_vel)
		ang_vel = self.pi_ang(obs_vel)
		vel = torch.cat([lin_vel, ang_vel], dim=-1)
		return vel

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
		return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class td3(nn.Module):

	def __init__(self, obs_dim, act_dim, with_advantage=True):
		super().__init__()
		# build policy and value functions
		print('DRL_Model: TD3')
		self.pi = Actor(obs_dim, act_dim)
		self.q1 = QFunction(obs_dim, act_dim)
		self.q2 = QFunction(obs_dim, act_dim)

	def act(self, obs):
		with torch.no_grad():
			return self.pi(obs).numpy()
