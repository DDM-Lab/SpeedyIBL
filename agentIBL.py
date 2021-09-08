import numpy as np
from itertools import count

import random as random
from pyibl import Agent
from IBL import IBL  
from collections import deque

class AgentlightweightIBL(IBL):

	# """ Agent """
	def __init__(self, outputs, default_utility = 0.1, lendeque=25000, Hash = True, equal = True):
		super(AgentlightweightIBL, self).__init__(default_utility=default_utility,lendeque=lendeque)
		# '''
		# :param dict config: Dictionary containing hyperparameters
		# '''
		self.outputs = outputs
		self.options = {}
		self.episode_history = []
		self.hash = Hash
		self.equal = equal

	def generate_options(self,s_hash):
		self.options[s_hash] = [(s_hash, a) for a in range(self.outputs)]
	
	def move(self, o, explore=True):
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

		return self.last_action



	def feedback(self, reward):
		# '''
		# Feedback is passed to the IBL agent instance.
		# :param float: Reward received during transition
		# :param boolean: Indicates if the transition is terminal
		# :param tensor: State/Observation
		# '''

		self.respond(reward)

		#episode history
		if self.equal:
			self.episode_history.append((self.current,self.last_action,reward,self.t))
			

	def delay_feedback(self, reward):		
		if reward > 0:
			for i in reversed(range(len(self.episode_history))):
				s,a,r,t1 = self.episode_history[i]
				#update outcomes
				t = self.instance_history[(s, a)][r].pop()
				if reward not in self.instance_history[(s, a)]:
					self.instance_history[(s, a)][reward] = deque([],self.lendeque)

				self.instance_history[(s, a)][reward].append(t)	
			self.episode_history = []


class AgentPyIBL(Agent):
	# Itertools used to create an unique id for each agent:
	mkid = next(count())

	# """ Agent """
	def __init__(self, outputs, default_utility=0.1, Hash = True, equal = True):
		super(AgentPyIBL, self).__init__("My Agent", ["action", "s"], default_utility=default_utility)
		# '''
		# :param int agentID: Agent's ID
		# :param dict config: Dictionary containing hyperparameters
		# '''
		self.outputs = outputs
		self.id = AgentPyIBL.mkid
		self.options = {}
		self.inst_history = []
		self.t = 0
		self.hash = Hash
		self.equal = equal
	def generate_options(self,s_hash):
		self.options[s_hash] = [{"action": a, "s": s_hash} for a in range(self.outputs)]

	def move(self, o, explore=True):
		# '''
		# Returns an action from the pyIBL agent instance.
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

		return self.last_action



	def feedback(self, reward):
		# '''
		# Feedback is passed to the pyIBL agent instance.
		# :param float: Reward received during transition
		# :param boolean: Indicates if the transition is terminal
		# :param tensor: State/Observation
		# '''
		self.inst_history.append(self.respond())
		self.inst_history[-1].update(reward)

	def delay_feedback(self, reward):		
		if reward > 0:
			for i in reversed(range(len(self.inst_history))):
				#update outcomes
				self.inst_history[i].update(reward)

			self.inst_history = []
