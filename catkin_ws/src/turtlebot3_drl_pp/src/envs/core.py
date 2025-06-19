import rospy
import math
import random
import time
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from math import pi
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Pose, Quaternion
from std_msgs.msg import String, Float32MultiArray
from std_msgs.msg import Empty as EmptyMsg
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import GetModelState, SpawnModel, DeleteModel
from std_srvs.srv import Empty
from src.util.core import PathsSaver

class Env():
	def __init__(self, fix_goal=False, goals=()):
		self.stage = rospy.get_param("/stage")
		self.fix_goal = fix_goal
		self.goals = goals
		self.ps = PathsSaver()

		self.obs_dim = 76
		self.scan_dim = 72
		self.act_dim = 2

		self.max_lin_vel = 0.25
		self.max_ang_vel = 1.5
		
		self.last_dis = 0
		self.last_ori = 0
		self.last_min_scan = 0
		self.dis = 0
		self.ori = 0
		self.min_scan = 0

		self.goal_x = 0
		self.goal_y = 0
		self.agent_x = rospy.get_param("/agent_x")
		self.agent_y = rospy.get_param("/agent_y")
		self.agent_ang_vel = 0
		self.agent_lin_vel = 0

		self.collision_dis = 0.2
		self.safe_dis =  0.4
		self.success_dis = 0.25

		self.pub_cmd_vel = rospy.Publisher('/agent/cmd_vel', Twist, queue_size=10)
		self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
		self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		
		self.goal = Goal(x=self.agent_x, y=self.agent_y, s=self.stage, f=self.fix_goal, g=self.goals)
		self.init_goal = False
		
	
	def reset(self, end=False):
		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.reset_sim()
		except (rospy.ServiceException) as e:
			rospy.logerr("/gazebo/reset_simulation service call failed")

		if end == True:
			return False

		self.goal.episode += 1

		if self.init_goal:
			self.goal.remove_goal()
		else:
			self.goal.spawn_goal()
			self.init_goal = True

		if self.fix_goal == True:
			self.ps.push_path()
			self.ps.push_point(data=[rospy.get_param("/agent_x"), rospy.get_param("/agent_y")])

		obs = self.get_observation(reset=True)
		norm_scan = [x/3.5 for x in obs[4]]
		o = [obs[0]] + [obs[1]] + [obs[2]/7] + [obs[3]/pi] + norm_scan
		
		dif = self.get_dif(obs)
		
		return o, dif
	
	def step(self, action):
		cmd_vel = Twist()
		cmd_vel.linear.x = action[0] * self.max_lin_vel
		cmd_vel.angular.z = action[1] * self.max_ang_vel
		self.pub_cmd_vel.publish(cmd_vel)
		time.sleep(0.2)

		obs = self.get_observation()
		norm_scan = [x/3.5 for x in obs[4]]
		o2 = [obs[0]] + [obs[1]] + [obs[2]/7] + [obs[3]/pi] + norm_scan

		dif2 = self.get_dif(obs)
		
		reward = self.get_reward(obs)
		done, done_reason = self.get_done(obs)
		
		return o2, dif2, reward, done, done_reason

	def get_observation(self, reset=False):
		rospy.wait_for_service('/gazebo/get_model_state')
		orientation = Quaternion()
		scan_data = None
		while scan_data is None:
			try:
				scan_data = rospy.wait_for_message('/agent/scan', LaserScan, timeout=5)
			except:
				pass
		scans = []
		for i in range(len(scan_data.ranges)):
			if scan_data.ranges[i] == float('Inf'):
				scans.append(3.5)
			elif np.isnan(scan_data.ranges[i]):
				scans.append(0)
			else:
				scans.append(round(scan_data.ranges[i], 4))
		try:
			goal_state = self.get_model_state('goal', 'world')
			agent_state = self.get_model_state('agent', 'world')
			self.goal_x = goal_state.pose.position.x
			self.goal_y = goal_state.pose.position.y
			self.agent_x = agent_state.pose.position.x
			self.agent_y = agent_state.pose.position.y
			self.agent_lin_vel = round(agent_state.twist.linear.x, 4)
			self.agent_ang_vel = round(agent_state.twist.angular.z, 4)
			if (self.agent_lin_vel == 0): self.agent_lin_vel = 0.0000
			if (self.agent_ang_vel == 0): self.agent_ang_vel = 0.0000
			self.goal.agent_x = self.agent_x
			self.goal.agent_y = self.agent_y
			orientation = agent_state.pose.orientation
		except (rospy.ServiceException) as e:
			rospy.logerr("/gazebo/get_model_state service call failed")

		if self.fix_goal == True:
			position_array = [round(self.agent_x,4), round(self.agent_y,4)]
			self.ps.push_point(data=position_array)

		self.last_dis = self.dis
		self.dis = round(math.hypot(self.goal_x - self.agent_x, self.goal_y - self.agent_y), 4)
		
		self.last_ori = self.ori
		self.ori = round(self.get_ori(orientation), 4)
		
		self.last_min_scan = self.min_scan
		self.min_scan = min(scans)
		
		obs = [self.agent_lin_vel, self.agent_ang_vel, self.dis, self.ori, scans]
		return obs
	
	def get_ori(self, orientation):
		orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
		_, _, yaw = euler_from_quaternion(orientation_list)
		goal_angle = math.atan2(self.goal_y - self.agent_y, self.goal_x - self.agent_x)
		heading = goal_angle - yaw
		if heading > pi:
			heading -= 2 * pi
		elif heading < -pi:
			heading += 2 * pi
		return heading
	
	def get_dif(self, obs):
		scans = obs[4]
		s = scans # [60:] + scans[0:12]
		unsafe_scan_num = 0
		for i in range(len(s)):
			unsafe_scan_num +=1 if s[i] < self.safe_dis else 0
		dif_unsafe = unsafe_scan_num / len(s)
		
		# init_dis = round(math.hypot(self.goal_x - rospy.get_param("/agent_x"), self.goal_y - rospy.get_param("/agent_y")), 4)
		# dif_dis = self.dis / init_dis
		# dif_dis = 1 if dif_dis > 1 else dif_dis
		# dif = round((dif_unsafe + dif_dis) / 2, 4)

		dif = round(dif_unsafe, 4)
		return dif


	def get_reward(self, obs):
		scans = obs[4]
		s = scans[60:] + scans[0:12]
		min_top_scan = min(s)

		lin_vel_reward = 0
		ang_vel_reward = 0
		safe_reward = 0	
		
		if min_top_scan >= self.safe_dis:
			ang_vel_reward = -0.1 * math.fabs(obs[1])
		
		if self.collision_dis < min_top_scan < self.safe_dis:
			lin_vel_reward = -obs[0]

		if self.collision_dis < self.min_scan < self.safe_dis:
			safe_reward = -10 * (self.safe_dis - self.min_scan)

		dis_reward = 10 * (self.last_dis - self.dis)

		ori_reward = 0.1 * (math.fabs(self.last_ori) - math.fabs(self.ori))

		reward = dis_reward + ori_reward + safe_reward + lin_vel_reward + ang_vel_reward
		# print('test dis', dis_reward)
		# print('test ori', ori_reward)
		# print('test safe', safe_reward)
		# print('test lin', lin_vel_reward)
		# print('test ang', ang_vel_reward)

		# reward = dis_reward + ori_reward

		if self.dis <= self.success_dis:
			reward += 20
		else:
			if self.min_scan <= self.collision_dis:
				reward -= 10

		reward = round(reward, 4)
		return reward

	def get_done(self, obs):
		done = False
		done_reason = ''
		dis = obs[2]
		ori = obs[3]
		if dis <= self.success_dis:
			done = True
			done_reason = 'success'
		else:
			if self.min_scan <= self.collision_dis:
				done = True
				done_reason = 'collision'
		if done:
			self.pub_cmd_vel.publish(Twist())
		return done, done_reason

class Goal():
	def __init__(self, x, y, s, f, g):
		self.episode = 0
		self.agent_x = x
		self.agent_y = y
		self.stage = s
		self.fix_goal = f
		self.goals = g
		self.gi = 0
		self.modelPath = os.path.dirname(os.path.realpath(__file__))
		self.modelPath = self.modelPath.replace('turtlebot3_drl_pp/src/envs',
												'turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')
		self.f = open(self.modelPath, 'r')
		self.model = self.f.read()
		self.goal_pose = Pose()
		self.goal_x = 0
		self.goal_y = 0
		self.model_name = 'goal'
		self.spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
		self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
		

	def spawn_goal(self):
		self.set_goal_position()
		rospy.wait_for_service('/gazebo/spawn_sdf_model')
		self.spawn_model(self.model_name, self.model, 'robotos_name_space', self.goal_pose, "world")
		# rospy.loginfo('Drl_Path_Planner: spawnGoal at [%f, %f]', self.goal_x, self.goal_y)

	def remove_goal(self):
		rospy.wait_for_service('/gazebo/delete_model')
		self.del_model(self.model_name)
		time.sleep(0.5)
		self.spawn_goal()

	def set_goal_position(self):
		if self.fix_goal == True:
			self.gi = np.random.randint(0, len(self.goals))  # 随机选择目标点索引
			self.goal_x = self.goals[self.gi][0]
			self.goal_y = self.goals[self.gi][1]
			self.goal_pose.position.x = self.goal_x
			self.goal_pose.position.y = self.goal_y
		else:
			self.goal_x = float(format(random.uniform(-2,2),'.2f'))
			self.goal_y = float(format(random.uniform(-2,2),'.2f'))
			flag = self.get_random_position_flag()
			while (flag == False):
				self.goal_x = float(format(random.uniform(-2,2),'.2f'))
				self.goal_y = float(format(random.uniform(-2,2),'.2f'))
				flag = self.get_random_position_flag()
			self.goal_pose.position.x = self.goal_x
			self.goal_pose.position.y = self.goal_y

	def get_random_position_flag(self):
		x = self.goal_x
		y = self.goal_y
		a_x = rospy.get_param("/agent_x")
		a_y = rospy.get_param("/agent_y")
		dis = math.hypot(x - a_x, y - a_y)
		res = False
		# print('ssssssssssssssssssssssssss', self.stage)
		if self.stage == 0:
			# if 1.5 <= dis <= 2:
			# 	res = True
			# if dis >= 1:
			# 	res = True
			if abs(x) < 1 and abs(y) < 1:
				res = False
			else:
				res = True
		elif self.stage == 1:
			# train
			# if 1.76<=x<=2 and 1.25<=y<=2:
			# 	res = True
			# elif 1<=x<=1.48 and 1.75<=y<=2:
			# 	res = True
			if 1.75<=x<=2 and 1.75<=y<=2:
				res = True
			elif 1.47<=x<=2 and -2<=y<=-1.57:
				res = True
			elif -2<=x<=-1.4 and 1.7<=y<=2:
				res = True
			elif -2<=x<=-1.5 and -2<=y<=-1.5:
				res = True
			else:
				res = False
		elif self.stage == 2:
			if 1.5<=x<=2 and 1.47<=y<=2:
				res = True
			elif 1.71<=x<=2 and -2<=y<=-1.44:
				res = True
			elif -2<=x<=-1.42 and 1.84<=y<=2:
				res = True
			elif -2<=x<=-1.84 and -2<=y<=-1.45:
				res = True
			else:
				res = False
		elif self.stage == 5:
			if 1.8<=x<=2 and 1.5<=y<=2:
				res = True
			elif 1.5<=x<=2 and -2<=y<=-1.8:
				res = True
			elif -2<=x<=-1.5 and 1.8<=y<=2:
				res = True
			elif -2<=x<=-1.8 and -2<=y<=-1.5:
				res = True
			else:
				res = False
		else:
			res = False

		return res
