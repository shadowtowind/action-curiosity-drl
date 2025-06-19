import numpy as np
import torch
import torch.nn as nn
from math import pi

class Actor(nn.Module):

	def __init__(self, obs_dim, act_dim, with_lstm=False):
		super().__init__()
		self.with_lstm = with_lstm
		# self.rnn = nn.LSTM(input_size=obs_dim, hidden_size=obs_dim)

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
		if self.with_lstm:
			l = len(obs.shape)
			rnn_input = obs.unsqueeze(0)
			if l == 1:
				h0 = torch.randn(1, obs.shape[0])
				c0 = torch.randn(1, obs.shape[0])
			else:
				h0 = torch.randn(1, obs.shape[0], obs.shape[1])
				c0 = torch.randn(1, obs.shape[0], obs.shape[1])
			rnn_output, (hn, cn) = self.rnn(rnn_input, (h0, c0))
			pi_input = rnn_output.squeeze(0)
			obs_vel = self.pi(pi_input)
		else:
			obs_vel = self.pi(obs)

		lin_vel = self.pi_lin(obs_vel)
		ang_vel = self.pi_ang(obs_vel)
		vel = torch.cat([lin_vel, ang_vel], dim=-1)
		return vel

class Attn_Actor(nn.Module):

	def __init__(self, obs_dim, scan_dim, act_dim):
		super().__init__()
		self.obs_dim = obs_dim
		self.scan_dim = scan_dim
		self.act_dim = act_dim
		self.attn = nn.MultiheadAttention(embed_dim=1, num_heads=1, batch_first=True)
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
		l = len(obs.shape)
		s = torch.split(obs,[self.obs_dim - self.scan_dim, self.scan_dim], dim=l-1)
		q = s[1].unsqueeze(l)
		kv = obs.unsqueeze(l)
		attn_output, attn_output_weights = self.attn(q, kv, kv)
		scan_attn = torch.mul(s[1], attn_output.squeeze(l))
		
		obs_attn = torch.cat([s[0], scan_attn], dim=l-1)
		obs_vel = self.pi(obs_attn)
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

class Dueling_QFunction(nn.Module):

	def __init__(self, obs_dim, scan_dim, act_dim):
		super().__init__()
		self.obs_dim = obs_dim
		self.scan_dim = scan_dim
		self.act_dim = act_dim
		self.v = nn.Sequential(
			nn.Linear(obs_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Identity()
		)
		self.ac = nn.Sequential(
			nn.Linear(scan_dim + act_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Identity()
		)
		self.ag = nn.Sequential(
			nn.Linear(obs_dim - scan_dim + act_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, 1),
			nn.Identity()
		)

	def forward(self, obs, act):
		l = len(obs.shape)
		s = torch.split(obs, [self.obs_dim - self.scan_dim, self.scan_dim], dim=l-1)

		v_input = torch.cat([s[0], s[1]], dim=l-1)
		ac_input = torch.cat([s[1], act], dim=l-1)
		ag_input = torch.cat([s[0], act], dim=l-1)
		
		v = self.v(v_input).squeeze(-1)
		ac = self.ac(ac_input).squeeze(-1) 
		ag = self.ag(ag_input).squeeze(-1)

		q = v + ac + ag
		return q

class ActorCritic(nn.Module):

	def __init__(self, obs_dim, scan_dim, act_dim, with_attention=False, with_dueling=False, with_lstm=False):
		super().__init__()
		if with_attention == True and with_dueling == True:
			print('DRL_Model: BOAE_DDPG')
			self.pi = Attn_Actor(obs_dim, scan_dim, act_dim)
			self.q = Dueling_QFunction(obs_dim, scan_dim, act_dim)
		elif with_attention == False and with_dueling == True:
			print('DRL_Model: DDPG_With_Dueling')
			self.pi = Actor(obs_dim, act_dim)
			self.q = Dueling_QFunction(obs_dim, scan_dim, act_dim)
		elif with_attention == True and with_dueling == False:
			print('DRL_Model: DDPG_With_CA')
			self.pi = Attn_Actor(obs_dim, scan_dim, act_dim)
			self.q = QFunction(obs_dim, act_dim)
		else:
			print('DRL_Model: DDPG')
			self.pi = Actor(obs_dim, act_dim, with_lstm)
			self.q = QFunction(obs_dim, act_dim)

	def act(self, obs):
		with torch.no_grad():
			return self.pi(obs).numpy()
