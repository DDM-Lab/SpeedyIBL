import numpy as np
from itertools import count

import random as random
import math 
import sys 
from collections import deque
from tabulate import tabulate

class IBL(object):
	# Itertools used to create an unique id for each agent:
	mkid = next(count())

	# """ Agent """
	def __init__(self, default_utility = 0.1, noise = 0.25, decay = 0.5, lendeque = 25000):

		self.default_utility = default_utility
		self.noise = noise
		self.decay = decay
		self.temperature = 0.25*math.sqrt(2)
		self.lendeque = lendeque
		self.id = IBL.mkid
		self.instance_history = {}
		self.t = 0

	def respond(self, reward):

		if self.option not in self.instance_history:
			self.instance_history[self.option] = {reward:deque([],self.lendeque)}
		elif reward not in self.instance_history[(self.option)]:
			self.instance_history[self.option][reward] = deque([],self.lendeque)

		self.instance_history[self.option][reward].append(self.t) 

	def populate_at(self, option, reward, t):
		if (option) not in self.instance_history:
			self.instance_history[option] = {reward:deque([],self.lendeque)}
		elif reward not in self.instance_history[option]:
			self.instance_history[option][reward] = deque([],self.lendeque)
		self.instance_history[option][reward].append(t)
	
	def compute_blended(self, t, options):
		blends = []
		for o, i in zip(options,count()):
			if o in self.instance_history:
				tmps =[]
				rewards = []
				for r in self.instance_history[o]:
					if len(self.instance_history[o][r])>0:
						tmp = np.copy(self.instance_history[o][r])
						tmp = t - tmp
						tmp = math.log(sum(pow(tmp,-self.decay))) + self.noise*self.make_noise()
						tmps.append(tmp)
						rewards.append(r)
				
				tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
				tmps.append(tmp0)
				tmps = np.array(tmps)
				tmps = np.exp(tmps/self.temperature)
				p = tmps/sum(tmps)
				rewards.append(self.default_utility)
				rewards = np.array(rewards)
				result = sum(rewards*p)
				blends.append((result,i))
			else:
				blends.append((self.default_utility,i))
		return blends 
	def make_noise(self):
		p = random.uniform(sys.float_info.epsilon, 1-sys.float_info.epsilon)
		result = math.log((1.0-p) / p)
		return result
	def choose(self, options):
		self.t += 1
		utilities = self.compute_blended(self.t, options)
		best_utility = max(utilities,key=lambda x:x[0])[0]
		best = random.choice(list(filter(lambda x: x[0]==best_utility,utilities)))[1]
		self.option = options[best]
		return self.option 
	
	def reset(self):
		self.t = 0
		self.instance_history = {}
	
	def instances(self):
		print(tabulate([[a,b,list(self.instance_history[a][b])] for a in self.instance_history for b in self.instance_history[a]], headers=['option','outcome','occurences']))