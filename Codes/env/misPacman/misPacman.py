import numpy as np
import random
import cv2
import copy
import time
import gym

class MISPACMAN(object):
	""" Cooperative multi-agent transporation problem. """
	def __init__(self):
		'''
		:param version: Integer specifying which configuration to use
		'''

		self.c = gym.make('MsPacman-v0')
		# Fieldnames for stats
		self.fieldnames = ['Episode',
						   'Steps',
						   'Total_Reward',
						   'Time']

		self.__dim = [210, 160, 3]     # Observation dimension
		self.__out = self.c.action_space.n # Number of actions
		self.episode_count = 0      # Episode counter
		self.time = 0
		
		# Used to add noise to each cell
		# self.ones = np.ones(self.c.DIM, dtype=np.float64)

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
		r = 2 # Number of times the pixel is to be repeated
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
				 'Total_Reward': str(self.reward_total),
				 'Time':str(self.time)}
		return stats

	def result(self):
		stats = [self.episode_count, 
				 self.steps, 
				 self.reward_total,
				 self.time]
		return stats 

	def reset(self):
		'''
		Reset everything. 
		'''
		# Set up the state array:
		# 0 = obstacles, 1 = goods, 2 = agents, 3 = self
		self.s_t = self.c.reset()
		self.reward_total = 0.0
		

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
	  
		return self.s_t 

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
		observations, rewards, done, info = self.c.step(actions)
		
		self.s_t = np.copy(observations)
		

		# Counters are incremented:
		self.steps += 1 
		
		self.reward_total += rewards # Team game, so one r is sufficient  
		
		return observations, rewards, done 

	

	def getNoisyState(self):
		# ''' 
		# Method returns noisy state.
		# '''
		return self.s_t 

	def getObservations(self):
		# '''
		# Returns centered observation for each agent
		# '''
		# observations = []
		# for i in range(self.c.NUMBER_OF_AGENTS):
		#     # Store observation
		#     observations.append(np.copy(self.getNoisyState()))
		return np.copy(self.getNoisyState())


