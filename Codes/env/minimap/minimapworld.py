import numpy as np
import random
import cv2
import copy
import time

class MAPWORLD(object):
	""" Cooperative multi-agent transporation problem. """
	def __init__(self, version):
		'''
		:param version: Integer specifying which configuration to use
		'''
		if version == 1: # Standard
			from .minimap_v1 import EnvConfigV1
			self.c = EnvConfigV1()
		if version == 2: # Standard
			from .minimap_v2 import EnvConfigV2
			self.c = EnvConfigV2()
		if version == 3: # Standard
			from .minimap_v3 import EnvConfigV3
			self.c = EnvConfigV3()

		# Fieldnames for stats
		self.fieldnames = ['Episode',
						   'Steps',
						   'Total_pickup_goals',
						   'Green',
				 			'Yellow',
				 			'Total_Positive_Discounted',
						   'Total_Reward',
						   'Total_Discounted',
						   'Goods_Delivered',
						   'Time']

		self.__dim = self.c.DIM     # Observation dimension
		self.__out = self.c.ACTIONS # Number of actions
		self.episode_count = 0      # Episode counter
		self.time = 0
		
		# Used to add noise to each cell
		self.ones = np.ones(self.c.DIM, dtype=np.float64)

	@property
	def dim(self):
		return self.__dim

	@property
	def out(self):
		return self.__out

	def render(self):
		'''
		Used to render the env.
		'''
		r = 16 # Number of times the pixel is to be repeated
		try:
			img = np.repeat(np.repeat(self.getNoisyState(), r, axis=0), r, axis=1).astype(np.uint8)
			cv2.imshow('image', img)
			k = cv2.waitKey(1)
			if k == 27:         # If escape was pressed exit
				cv2.destroyAllWindows()
		except AttributeError:
			pass

	def stats(self):
		'''
		Returns stats dict
		'''
		stats = {'Episode': str(self.episode_count), 
				 'Steps': str(self.steps), 
				#  'Coordinated_Steps':str(self.coordinatedTransportSteps),
				 'Total_pickup_goals': str(self.number_pickup_goals),
				 'Green':str(self.green_total),
				 'Yellow':str(self.yellow_total),
				 'Total_Positive_Discounted': str(self.positive_reward_discounted),
				 'Total_Reward': str(self.reward_total),
				 'Total_Discounted': str(self.reward_discounted),
				 'Goods_Delivered':str(self.delivered),
				 'Time':str(self.time)}
		return stats

	def result(self):
		stats = [self.episode_count, 
				 self.steps, 
				self.number_pickup_goals,
				 self.green_total,
				 self.yellow_total,
				 self.positive_reward_discounted,
				 self.reward_total,
				 self.reward_discounted,
				 self.delivered]
		return stats 

	def reset(self):
		'''
		Reset everything. 
		'''
		# Set up the state array:
		# 0 = obstacles, 1 = goods, 2 = agents, 3 = self
		self.s_t = np.zeros(self.c.DIM, dtype=np.float64)
		self.s_goals = np.zeros(self.c.DIM, dtype=np.float64)

		# Obstacles, agents and goods are initialised:
		self.setObstacles()
		self.initGoals()
		self.initAgents()

		# Used to keep track of the reward total acheived throughout 
		# the episode:
		self.reward_total = 0.0
		self.reward_discounted = 0.0
		self.number_pickup_goals = 0
		self.positive_reward_discounted = 0.0
		self.green_total = 0.0
		self.yellow_total = 0.0
		

		# Episode counter is incremented:
		self.episode_count += 1
   
		# For statistical purposes:
		# Step counter for the episode is initialised
		self.steps = 0 
		self.delivered = False
		# Number of steps the goods is carried by both agents
		# self.coopTransportSteps = 0 

		# Moves taken in the same direction while carrying the goods
		# self.coordinatedTransportSteps = 0 
	  
		return self.getObservations()

	def terminal(self):
		'''
		Find out if terminal conditions have been reached.
		'''
		return self.delivered

	def step(self, actions):
		'''
		Change environment state based on actions.
		:param actions: list of integers
		'''
		# Agents move according to actions selected
		rewards = self.moveAgents(actions)

		# Observations are loaded for each agent
		observations = self.getObservations()
		self.s_o = np.copy(observations)
		self.s_o[self.agents_y][self.agents_x] = 0.0
		# Goods pickup checks to see if the agents have
		# reached a position where they can grasp the goods.                    
		# rewards = self.goalsPickup(noCol)

		# Counters are incremented:
		self.steps += 1 
		# if self.holding_goods[0] == True and self.holding_goods[1] == True:
		#     self.coopTransportSteps += 1
			# Check if goods have reached dzone
		# rewards = self.goodsDelivered(noCol) 
		# else:
		#     rewards = [0.0, 0.0]
		if rewards == 0.25:
			self.green_total += rewards
		elif rewards == 0.75:
			self.yellow_total += rewards
		if rewards >0:
			self.positive_reward_discounted += rewards*pow(0.99,self.steps-1)
		self.reward_total += rewards # Team game, so one r is sufficient  
		self.reward_discounted += rewards*pow(0.99,self.steps-1)
		return observations, rewards, self.terminal() 

	def initGoals(self):
		# '''
		# Goods position and carrier ids are initialised
		# '''

		self.delivered = False

		for (y, x, r) in self.c.GOALS_YX:
			if r() == 0.75:
				self.s_t[y][x] = self.c.GOAL[0] #self.c.GOAL 
				self.s_goals[y][x] = self.c.GOAL[0]
			else:
				self.s_t[y][x] = self.c.GOAL[1]
				self.s_goals[y][x] = self.c.GOAL[1]

	def initAgents(self):
		# '''
		# Method for initialising the required number of agents and 
		# positionsing them on designated positions within the grid
		# '''

		# Agent x and y positions can be set in the following lists.
		# Defaults are the bottom right and left corners, for two agents, 
		# since this is the way most CMOTPs start
		self.agents_x = copy.deepcopy(self.c.AGENTS_X)
		self.agents_y = copy.deepcopy(self.c.AGENTS_Y)

		# List used to indicate whether or not the agents are holding goods
		# self.holding_goods= [False for i in range(self.c.NUMBER_OF_AGENTS)]
   
		# Agents are activated within the agent channel (2) of the gridworld
		# matrix.
		# for i in range(self.c.NUMBER_OF_AGENTS):
		self.s_t[self.agents_y][self.agents_x] += self.c.AGENTS

	def setObstacles(self):
		'''
		Method used to initiate the obstacles within the environment 
		'''
		for y, x in self.c.OBSTACLES_YX:
			self.s_t[y][x] = self.c.OBSTACLE

	# def goalsPickup(self):
	#     '''
	#     Method for picking up the goods, if the agents
	#     find themselves in positions adjecent to the goods.
	#     '''
	#     # For each of a goods we check whether agents are 
	#     # in a position to pick the goods up:

	#     # Check to see if there is an agent on the left 
	#     # handside to pickup the goods:
	#     if (self.goods_x >= 1 and 
	#         self.goods_l == -1 and 
	#         self.s_t[self.goods_y][self.goods_x - 1] > 0):

	#         for j in range(self.c.NUMBER_OF_AGENTS):
	#             if (self.agents_x[j] == self.goods_x - 1 and 
	#                 self.agents_y[j] == self.goods_y and 
	#                 self.holding_goods[j] == False):
	#                 self.goods_l = j
	#                 self.holding_goods[j] = True

	#     # Check to see if there is an agent on the right 
	#     # handside to pickup the goods:
	#     if (self.goods_x < self.c.GW-1 and  
	#         self.goods_r == -1 and 
	#         self.s_t[self.goods_y][self.goods_x + 1] > 0):

	#         for j in range(self.c.NUMBER_OF_AGENTS):
	#             if (self.agents_x[j] == self.goods_x + 1 and  
	#                 self.agents_y[j] == self.goods_y and 
	#                 self.holding_goods[j] == False):
	#                 self.goods_r = j
	#                 self.holding_goods[j] = True

	def goalsPickup(self, y, x):
		# '''
		# Method to check one of the goods 
		# has been deliverd to the dropzone
		# '''
		# for (dropX, dropY, r) in self.c.GOALS_YX:
		if self.s_t[y][x] == self.c.GOAL[0]:
			self.number_pickup_goals += 1
			r = 0.75
			self.s_goals[y][x] = 0
		elif self.s_t[y][x] == self.c.GOAL[1]:
			self.number_pickup_goals += 1
			r = 0.25
			self.s_goals[y][x] = 0
		else:
			r = -0.01
		if self.number_pickup_goals == self.c.number_goals:
			self.delivered = True                
				# self.s_t[self.goods_y][self.goods_x] -= self.c.GOODS
				# self.goods_y = -1
				# if self.goods_l > -1:
				#     self.holding_goods[self.goods_l] = False
				# if self.goods_r > -1:
				#     self.holding_goods[self.goods_r] = False
				# self.goods_l, self.goods_r = -1, -1
			# for (dropX, dropY, r) in self.c.GOALS_YX:
			# 	if y == dropY and x == dropX:
			# 		break
		return r
		# for y, x in self.c.OBSTACLES_YX: 
		# 	if self.agents_x == x and self.agents_y == y:
		# 		self.delivered = True 
		# 		return -1.0
		# if noCol == False:
		# 	return -0.01
		# else:
		# 	return -0.05
		# return -0.01

	# def unsetAgents(self): 
	#     ''' 
	#     Method to release the agents from holding the goods
	#     '''
	#     if self.goods_x > -1 and self.goods_y > -1:
	#         self.s_t[self.goods_y][self.goods_x] = 0
	#         self.goods_l = -1
	#         self.goods_r = -1
	#     for i in range(self.c.NUMBER_OF_AGENTS):
	#             if self.agents_x[i] > -1 and self.agents_y[i] > -1:
	#                 self.s_t[self.agents_y[i]][self.agents_x[i]] = 0
	#                 self.holding_goods[i] = False

	def getNoisyState(self):
		# ''' 
		# Method returns noisy state.
		# '''
		return self.s_t + (self.c.NOISE * self.ones *\
						   np.random.normal(self.c.MU,self.c.SIGMA, self.c.DIM))

	def getObservations(self):
		# '''
		# Returns centered observation for each agent
		# '''
		# observations = []
		# for i in range(self.c.NUMBER_OF_AGENTS):
		#     # Store observation
		#     observations.append(np.copy(self.getNoisyState()))
		return np.copy(self.getNoisyState())

	def getDelta(self, action):
		# '''
		# Method that deterimines the direction 
		# that the agent should take
		# based upon the action selected. The
		# actions are:
		# 'Up':0, 
		# 'Right':1, 
		# 'Down':2, 
		# 'Left':3, 
		# 'NOOP':4
		# :param action: int
		# '''
		if action == 0:
			return 0, -1
		elif action == 1:
			return 1, 0    
		elif action == 2:
			return 0, 1    
		elif action == 3:
			return -1, 0 
		elif action == 4:
			return 0, 0   

	def moveAgents(self, actions):
	#    '''
	#    Move agents according to actions.
	#    :param actions: List of integers providing actions for each agent
	#    '''
	#    for i in range(self.c.NUMBER_OF_AGENTS):
		if random.random() < self.c.WIND:
			actions = random.randrange(self.c.__outs)

		#    for i in range(self.c.NUMBER_OF_AGENTS):
		#        if self.holding_goods[i] == False:
		dx, dy = self.getDelta(actions)
		targetx = self.agents_x + dx
		targety = self.agents_y + dy
		if self.noCollision(targetx, targety):
			r = self.goalsPickup(targety,targetx)
			self.moveAgent(targetx, targety)
		else:
			r = -0.05
		return r
	
	def moveAgent(self, targetx, targety):
		# '''
		# Moves agent to target x and y
		# :param targetx: Int, target x coordinate
		# :param targety: Int, target y coordinate
		# '''
		self.s_t[self.agents_y][self.agents_x] -= self.c.AGENTS
		self.agents_x = targetx
		self.agents_y = targety
		self.s_t[self.agents_y][self.agents_x] = self.c.AGENTS

	# def moveGoods(self, targetx, targety):
	#     '''
	#     Moves goods to target x and y
	#     :param targetx: Int, target x coordinate
	#     :param targety: Int, target y coordinate
	#     '''
	#     self.s_t[self.goods_y][self.goods_x] -= self.c.GOODS
	#     self.goods_x = targetx
	#     self.goods_y = targety
	#     self.s_t[self.goods_y][self.goods_x] += self.c.GOODS

	def noCollision(self, x, y):
		# '''
		# Checks if x, y coordinate is currently empty 
		# :param x: Int, x coordinate
		# :param y: Int, y coordinate
		# '''
		if x < 0 or x >= self.c.GW or y < 0 or y >= self.c.GH or self.s_t[y][x]==self.c.OBSTACLE:
			return False
		else:
			return True

