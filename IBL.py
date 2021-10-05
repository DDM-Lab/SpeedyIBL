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
	def __init__(self, default_utility = 0.1, noise = 0.25, decay = 0.5, mismatchPenalty = None, lendeque = 250000):
	
		self.default_utility = default_utility
		self.noise = noise
		self.decay = decay
		self.temperature = 0.25*math.sqrt(2)
		self.mismatchPenalty = mismatchPenalty
		self.lendeque = lendeque
		self.id = IBL.mkid
		self.instance_history = {}
		self.t = 0

		self.sim = {}
		self.sim['att'] = []
		self.sim['f'] = []

		self.atts = []

	def respond(self, reward):
	
		if self.option not in self.instance_history:
			self.instance_history[self.option] = {reward:deque([],self.lendeque)}
			if self.mismatchPenalty is not None:
				self.atts.append(list(self.option))
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
						if self.mismatchPenalty is not None:
							tmp = tmp + self.get_similarity(o)
						tmps.append(tmp)
						rewards.append(r)
				if self.default_utility is not None:
					tmp0 = math.log(pow(t,-self.decay)) + self.noise*self.make_noise()
					tmps.append(tmp0)
					rewards.append(self.default_utility)
				tmps = np.array(tmps)
				tmps = np.exp(tmps/self.temperature)
				p = tmps/sum(tmps)
				rewards = np.array(rewards)
				result = sum(rewards*p)
				blends.append((result,i))
			elif self.default_utility is not None:
				blends.append((self.default_utility,i))
		return blends 
	def make_noise(self):
		p = random.uniform(sys.float_info.epsilon, 1-sys.float_info.epsilon)
		result = math.log((1.0-p) / p)
		return result
	def choose(self, options):
		self.t += 1
		utilities = self.compute_blended(self.t, options)
		# print(utilities)
		best_utility = max(utilities,key=lambda x:x[0])[0]
		best = random.choice(list(filter(lambda x: x[0]==best_utility,utilities)))[1]
		self.option = options[best]
		return self.option 
	
	def reset(self):
		self.t = 0
		self.instance_history = {}
		self.sim = {}
		self.sim['att'] = []
		self.sim['f'] = []

		self.atts = []
	
	def instances(self):
		print(tabulate([[a,b,list(self.instance_history[a][b])] for a in self.instance_history for b in self.instance_history[a]], headers=['option','outcome','occurences']))
	
	def prepopulate(self, option, reward):
		if (option) not in self.instance_history:
			self.instance_history[option] = {reward:deque([],self.lendeque)}
		elif reward not in self.instance_history[option]:
			self.instance_history[option][reward] = deque([],self.lendeque)
		self.instance_history[option][reward].append(0)
	
	def similarity(self,attributes,function):
		self.sim['att'].append(attributes)
		self.sim['f'].append(function)
	
	def get_similarity(self,option):
		result = 0
		if len(self.atts)>0:
			np_option = np.asarray(option)
			np_atts =  np.asarray(self.atts)
			for att, f in zip(self.sim['att'],self.sim['f']):
				result += sum(sum(f(np_option[att].reshape(1,-1),np_atts[:,att])))
			if option in self.instance_history:
				result = result - 1
		return result

			

