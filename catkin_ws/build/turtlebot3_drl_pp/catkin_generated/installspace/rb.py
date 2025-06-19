import numpy as np
import torch
import torch.nn as nn
import math

def combined_shape(length, shape=None):
	if shape is None:
		return (length,)
	return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
	def __init__(self, obs_dim, act_dim, size):
		self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
		self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
		self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
		self.rew_buf = np.zeros(size, dtype=np.float32)
		self.done_buf = np.zeros(size, dtype=np.float32)
		self.ptr, self.size, self.max_size = 0, 0, size
	def store(self, obs, act, rew, next_obs, done):
		self.obs_buf[self.ptr] = obs
		self.obs2_buf[self.ptr] = next_obs
		self.act_buf[self.ptr] = act
		self.rew_buf[self.ptr] = rew
		self.done_buf[self.ptr] = done
		self.ptr = (self.ptr+1) % self.max_size
		self.size = min(self.size+1, self.max_size)
	def sample_batch(self, batch_size=32):
		idxs = np.random.randint(0, self.size, size=batch_size)
		batch = dict(obs=self.obs_buf[idxs],
					 obs2=self.obs2_buf[idxs],
					 act=self.act_buf[idxs],
					 rew=self.rew_buf[idxs],
					 done=self.done_buf[idxs])
		return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

class DifReplayBuffer:
	def __init__(self, obs_dim, act_dim, size):
		self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
		self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
		self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
		self.rew_buf = np.zeros(size, dtype=np.float32)
		self.done_buf = np.zeros(size, dtype=np.float32)
		self.dif_buf = np.zeros(size, dtype=np.float32)
		self.dif2_buf = np.zeros(size, dtype=np.float32)
		self.ptr, self.size, self.max_size = 0, 0, size
	def store(self, obs, act, rew, next_obs, done, dif, next_dif):
		self.obs_buf[self.ptr] = obs
		self.obs2_buf[self.ptr] = next_obs
		self.act_buf[self.ptr] = act
		self.rew_buf[self.ptr] = rew
		self.done_buf[self.ptr] = done
		self.dif_buf[self.ptr] = dif
		self.dif2_buf[self.ptr] = next_dif
		self.ptr = (self.ptr+1) % self.max_size
		self.size = min(self.size+1, self.max_size)
	def sample_batch(self, batch_size=32):
		idxs = np.random.randint(0, self.size, size=batch_size)
		batch = dict(obs=self.obs_buf[idxs],
					 obs2=self.obs2_buf[idxs],
					 act=self.act_buf[idxs],
					 rew=self.rew_buf[idxs],
					 done=self.done_buf[idxs],
					 dif=self.dif_buf[idxs],
					 dif2=self.dif2_buf[idxs])
		return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
	def sample_batch_id(self, batch_size=32):
		idxs = np.random.randint(0, self.size, size=batch_size)
		return idxs

class AcReplayBUffer:
	def __init__(self, obs_dim, act_dim, size):
		self.rb = DifReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=size)
		# ac_replay_buffer
		self.arb = DifReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=size)
		# boring_replay_buffer
		self.brb = DifReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=size)
		self.min_dif = 0.1
		self.max_dif = 0.9
	def store(self, obs, act, rew, next_obs, done, dif, next_dif):
		self.rb.store(obs, act, rew, next_obs, done, dif, next_dif)
		if dif < self.min_dif:
			self.brb.store(obs, act, rew, next_obs, done, dif, next_dif)
		elif dif > self.max_dif:
			self.brb.store(obs, act, rew, next_obs, done, dif, next_dif)
		else:
			if next_dif > dif and next_obs[2] > obs[2]:
				self.brb.store(obs, act, rew, next_obs, done, dif, next_dif)
			else:
				self.arb.store(obs, act, rew, next_obs, done, dif, next_dif)
	def sample_batch(self, batch_size=32):
		# print('--------sample batch--------')
		# print(self.rb.size)
		# print(self.arb.size)
		# print(self.brb.size)
		return self.rb.sample_batch(batch_size)
	def sample_ac_batch(self, batch_size_a=32, batch_size_b=32):
		print('--------sample ac batch--------')
		print(self.rb.size)
		print(self.arb.size)
		print(self.brb.size)
		if self.arb.size >= batch_size_a:
			idxs_a = self.arb.sample_batch_id(batch_size_a)
			idxs_b = self.brb.sample_batch_id(batch_size_b)
		else:
			idxs_a = self.arb.sample_batch_id(self.arb.size)
			idxs_b = self.brb.sample_batch_id(batch_size_b + batch_size_a - self.arb.size)
		batch = dict(obs=np.concatenate((self.arb.obs_buf[idxs_a], self.brb.obs_buf[idxs_b])),
					 obs2=np.concatenate((self.arb.obs2_buf[idxs_a], self.brb.obs2_buf[idxs_b])),
					 act=np.concatenate((self.arb.act_buf[idxs_a], self.brb.act_buf[idxs_b])),
					 rew=np.concatenate((self.arb.rew_buf[idxs_a], self.brb.rew_buf[idxs_b])),
					 done=np.concatenate((self.arb.done_buf[idxs_a], self.brb.done_buf[idxs_b])),
					 dif=np.concatenate((self.arb.dif_buf[idxs_a], self.brb.dif_buf[idxs_b])),
					 dif2=np.concatenate((self.arb.dif2_buf[idxs_a], self.brb.dif2_buf[idxs_b])))
		return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
