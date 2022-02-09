import numpy as np
from itertools import count

import random as random
from pyibl import Agent as PyAgent
from speedyibl import Agent
import math 
import sys 
from collections import deque

class AgentlightweightIBL_EQUAL(Agent):

	# """ Agent """
	def __init__(self, config, default_utility = 0.1, lendeque=5000, populate = False, Hash = True, equal = True):
		super(AgentlightweightIBL_EQUAL, self).__init__(default_utility=default_utility,lendeque=lendeque)
		# '''
		# :param dict config: Dictionary containing hyperparameters
		# '''
		self.c = config
		self.populate = populate
		self.options = {}
		self.episode_history ={}
		self.n_victims = 0
		self.hash = Hash
		self.equal = equal

	def generate_options(self,s_hash):
		self.options[s_hash] = [(s_hash, a) for a in range(self.c.outputs)]
	
	def move(self, o, y, x, explore=True):
		# '''
		# Returns an action from the IBL agent instance.
		# :param tensor: State/Observation
		# '''
		if self.hash:
			s_hash = hash(o.tobytes())
		else:
			s_hash = o
		if s_hash not in self.options:
			self.generate_options(s_hash)
		options = self.options[s_hash]
		choice = self.choose(options)
		self.last_action = choice[1]

		self.current = s_hash
		self.y = y
		self.x = x

		return self.last_action



	def feedback(self, reward, terminal, ter, o):
		# '''
		# Feedback is passed to the IBL agent instance.
		# :param float: Reward received during transition
		# :param boolean: Indicates if the transition is terminal
		# :param tensor: State/Observation
		# '''

		self.respond(reward)

		#episode history
		if self.equal:
			if self.n_victims not in self.episode_history:
				self.episode_history[self.n_victims] = []
			self.episode_history[self.n_victims].append((self.y,self.x,self.current,self.last_action,reward,self.t))
			
				
			if ter:
				# print(len(self.episode_history[self.current]))
				for i in reversed(range(len(self.episode_history[self.n_victims]))):
					y1,x1,s,a,r,t1 = self.episode_history[self.n_victims][i]
					#update outcomes
					if r!=-0.05:
						t = self.instance_history[(s, a)][r].pop()
						if reward not in self.instance_history[(s, a)]:
							self.instance_history[(s, a)][reward] = deque([],self.lendeque)
		
						self.instance_history[(s, a)][reward].append(t)
					if self.populate:
					#populate to next state
						if r > 0:
							r = - 0.01
						o_s = np.copy(o)
						o_s[y1][x1] = 240.0
						next_hash = hash(o_s.obytes())
						self.populate_at((next_hash,a), r, t1)
					
				if self.populate:
				#populate in next state
					for n_v in self.episode_history:
						if n_v!= self.n_victims:
							for i in range(len(self.episode_history[n_v])):
								y1,x1,s,a,r,t1 = self.episode_history[n_v][i]
								if r > 0:
									r = - 0.01
								#populate to next state
								o_s = np.copy(o)
								o_s[y1][x1] = 240.0
								next_hash = hash(o_s.tobytes())
								self.populate_at((next_hash,a), r, t1)
				self.n_victims += 1 
			if terminal:		
				self.n_victims = 0	
				self.episode_history ={}


class AgentPyIBL(PyAgent):
	# Itertools used to create an unique id for each agent:
	mkid = next(count())

	# """ Agent """
	def __init__(self, config, default_utility=0.1, populate = False, Hash = True, equal = True):
		super(AgentPyIBL, self).__init__("My Agent", ["action", "s"], default_utility=default_utility)
		# '''
		# :param int agentID: Agent's ID
		# :param dict config: Dictionary containing hyperparameters
		# '''
		self.c = config
		self.populate = populate
		self.c.id = AgentPyIBL.mkid
		self.options = {}
		self.inst_history = []
		self.episode_history ={}
		self.t = 0
		self.n_victims = 0
		self.hash = Hash
		self.equal = equal
	def generate_options(self,s_hash):
		self.options[s_hash] = [{"action": a, "s": s_hash} for a in range(self.c.outputs)]

	def move(self, o, y, x, explore=True):
		# '''
		# Returns an action from the deep rl agent instance.
		# :param tensor: State/Observation
		# '''
		self.t += 1
		if self.hash:
			s_hash = hash(o.tobytes())
		else:
			s_hash = o
		
		if (s_hash) not in self.options:
			self.generate_options(s_hash)
		options = self.options[s_hash]
		action = self.choose(*options)
		self.last_action = action["action"]
		self.current = s_hash
		self.y = y
		self.x = x

		return self.last_action



	def feedback(self, reward, terminal, ter, o):
		# '''
		# Feedback is passed to the pyIBL agent instance.
		# :param float: Reward received during transition
		# :param boolean: Indicates if the transition is terminal
		# :param tensor: State/Observation
		# '''
		self.inst_history.append(self.respond())
		self.inst_history[-1].update(reward)

		if self.equal:
			if self.n_victims not in self.episode_history:
				self.episode_history[self.n_victims] = []
			self.episode_history[self.n_victims].append((self.y,self.x,self.current,self.last_action,reward,self.t))
		
			
			if ter:
				for i in reversed(range(len(self.episode_history[self.n_victims]))):
					y1,x1,s,a,r,t = self.episode_history[self.n_victims][i]
					#update outcomes
					if r!=-0.05:
						self.inst_history[i].update(reward)

					if self.populate:
					#populate to next state
						if r > 0:
							r = - 0.01
						o_s = np.copy(o)
						o_s[y1][x1] = 240.0
						next_hash = hash(o_s.tobytes())
						option = {"action": a, "s": next_hash}
						self.populate_at(r,t,*option)

				if self.populate:
				#populate in next state
					for n_v in self.episode_history:
						if n_v!= self.n_victims:
							for i in range(len(self.episode_history[n_v])):
								y1,x1,s,a,r,t = self.episode_history[n_v][i]
								if r > 0:
									r = - 0.01
								o_s = np.copy(o)
								o_s[y1][x1] = 240.0
								next_hash = hash(o_s.tobytes())
								option = {"action": a, "s": next_hash}
								self.populate_at(r,t,*option)
				self.n_victims += 1
				self.inst_history =[]
			if terminal:	
				self.n_victims = 0	
				self.inst_history = []	
				self.episode_history ={}


