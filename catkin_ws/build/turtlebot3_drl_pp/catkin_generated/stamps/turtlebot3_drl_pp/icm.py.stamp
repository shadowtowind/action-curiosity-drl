import numpy as np
import torch
import torch.nn as nn

class ICM(nn.Module):
	
	def __init__(self, obs_dim, scan_dim):
		super().__init__()
		self.obs_dim = obs_dim
		self.scan_dim = scan_dim
		self.encoder = nn.Sequential(
			nn.Linear(scan_dim, 128),
			nn.ELU(),
			nn.Linear(128, 64),
			nn.ELU(),
			nn.Linear(64, 16),
			nn.ELU()
		)
		self.predict_lin_net = nn.Sequential(
			nn.Linear(32, 32),
			nn.ELU(),
			nn.Linear(32, 1),
			nn.Sigmoid()
		)
		self.predict_ang_net = nn.Sequential(
			nn.Linear(32, 32),
			nn.ELU(),
			nn.Linear(32, 1),
			nn.Tanh()
		)
		self.predict_phi_net = nn.Sequential(
			nn.Linear(16+2, 64),
			nn.ELU(),
			nn.Linear(64, 32),
			nn.ELU(),
			nn.Linear(32, 16),
			nn.ELU()
		)

	def forward(self, obs, obs2, act):
		l = len(obs.shape)
		scan = torch.split(obs, [self.obs_dim - self.scan_dim, self.scan_dim], dim=l-1)[1]
		scan2 = torch.split(obs2, [self.obs_dim - self.scan_dim, self.scan_dim], dim=l-1)[1]
		phi = self.encoder(scan)
		phi2 = self.encoder(scan2)
		
		predict_act_input = torch.cat([phi, phi2], dim=-1)
		predict_act_lin = self.predict_lin_net(predict_act_input)
		predict_act_ang = self.predict_ang_net(predict_act_input)
		predict_act = torch.cat([predict_act_lin, predict_act_ang], dim=-1)

		predict_phi_input = torch.cat([phi, act], dim=-1).detach()
		predict_phi2 = self.predict_phi_net(predict_phi_input)

		return predict_act, predict_phi2, phi2
