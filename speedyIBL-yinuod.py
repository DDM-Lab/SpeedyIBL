import numpy as np
from itertools import count

import random as random
import math
import sys
from collections import deque
from tabulate import tabulate
import pprint


class myAgent(object):
    # Itertools used to create an unique id for each agent:
    mkid = next(count())

    # """ Agent """
    def __init__(
        self,
        default_utility=0.1,
        noise=0.25,
        decay=0.5,
        mismatchPenalty=None,
        lendeque=250000,
    ):

        self.default_utility = default_utility
        self.noise = noise
        self.decay = decay
        self.temperature = 0.25 * math.sqrt(2)
        self.mismatchPenalty = mismatchPenalty
        self.lendeque = lendeque
        self.id = myAgent.mkid
        # The organization of instance history is different from official implementation
        #   The structure is {State:{Action:Reward:[t]}}
        #   The structure in official implementation is {(State, Action):{Reward:[t]}}
        #   As can be seen in the respond() function and the prepopulate() function
        self.instance_history = {}
        self.t = 0

        # Similarity functions are not used in this customized version
        self.sim = {}
        self.sim["att"] = []
        self.sim["f"] = []

        self.atts = []
        self.simvalues = {}

    def __str__(self) -> str:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint("Similarity Functions:")
        pp.pprint(self.sim)
        pp.pprint("-"*15)
        pp.pprint("Instance History")
        #pp.pprint(self.instance_history)
        return "Customized SpeedyIBL (@Yinuo)"

    def choose(self, options):
        self.t += 1
        utilities = self.compute_blended(self.t, options)
        best_utility = max(utilities, key=lambda x: x[0])[0]
        best = random.choice(list(filter(lambda x: x[0] == best_utility, utilities)))[1]
        self.option = options[best]
        return self.option

    def respond(self, reward):
        action = self.option[0]
        state = self.option[1]

        if state not in self.instance_history:
            self.instance_history[state] = {action: {}}
        if action not in self.instance_history[state]:
            self.instance_history[state][action] = {}
        if reward not in self.instance_history[state][action]:
            self.instance_history[state][action][reward] = []
        self.instance_history[state][action][reward].append(self.t)

        if len(self.instance_history[state][action][reward]) > self.lendeque:
            self.instance_history[state][action][reward][-self.lendeque :]

        return state, action, reward, self.t

    def compute_blended(self, t, options):
        blends = []
        # Note: the state in all options are required to be the same for all options
        state = options[0][1]  # state
        # If state has been observed
        if state in self.instance_history.keys():
            for o, i in zip(options, count()):
                action =o[0]
                activations = []  # Activations "activations"
                rewards = []  # Rewards
                if (action in self.instance_history[state].keys()) and len(self.instance_history[state][action]) > 0:  
                    for reward, timestamps in self.instance_history[state][action].items():
                        if len(timestamps) <= 0:
                            continue
                        timestamps = np.copy(timestamps)  # tmp: timestamp list
                        timediffs = t - timestamps  # tmp: time difference list
                        frequency = (
                            math.log(sum(pow(timediffs, -self.decay)))
                            + self.noise * self.make_noise()
                        )
                        # Note that the similarity term is omitted
                        #   The official version includes an extra term: self.mismatchPenalty*self.simvalues[s,ro[1]]
                        #   if (s,ro[1]) in self.simvalues:
						# 		    tmp = tmp + self.mismatchPenalty*self.simvalues[s,ro[1]]
						# 	    else:
						# 		    tmp = tmp + self.mismatchPenalty*self.get_similarity(s,ro[1])
                        activation = frequency # simplified activation function
                        activations.append(activation)
                        rewards.append(reward)
                if self.default_utility is not None:
                    activation = (
                        math.log(pow(t, -self.decay)) + self.noise * self.make_noise()
                    )
                    activations.append(activation)
                    rewards.append(self.default_utility)
                # Probability of retrieval
                activations = np.array(activations)
                activations = np.exp(activations / self.temperature)
                probs = activations / sum(activations)
                # Blending values
                rewards = np.array(rewards)
                result = sum(rewards * probs)
                blends.append((result, i))
        # Otherwise set the expected utility (blended value) of each option as default_utility
        elif self.default_utility is not None:
            blends = [(self.default_utility, i) for i in range(len(options))]
        else:
            NotImplementedError

        return blends

    def make_noise(self):
        p = random.uniform(sys.float_info.epsilon, 1 - sys.float_info.epsilon)
        result = math.log((1.0 - p) / p)
        return result

    def reset(self):
        self.t = 0
        self.instance_history = {}
        self.sim = {}
        self.sim["att"] = []
        self.sim["f"] = []

        self.atts = []
        self.simvalues = {}

    def instances(self):
        print(
            tabulate(
                [
                    [a, b, list(self.instance_history[a][b])]
                    for a in self.instance_history
                    for b in self.instance_history[a]
                ],
                headers=["option", "outcome", "occurences"],
            )
        )

    def prepopulate(self, option, reward):
        action = option[0]
        state = option[1]

        if state not in self.instance_history:
            self.instance_history[state] = {action: {}}
        if action not in self.instance_history[state]:
            self.instance_history[state][action] = {}
        if reward not in self.instance_history[state][action]:
            self.instance_history[state][action][reward] = []

        self.instance_history[state][action][reward].append(0)

    def similarity(self, attributes, function):
        self.sim["att"].append(attributes)
        self.sim["f"].append(function)

    def get_similarity(self, option, option2):
        result = 0
        for att, f in zip(self.sim["att"], self.sim["f"]):
            tmp = f([option[index] for index in att], [option2[index] for index in att])
            result += tmp
        result = round(result, 3)
        self.simvalues[(option, option2)] = result
        self.simvalues[(option2, option)] = result
        return result

    def equal_delay_feedback(self, new_reward, episode_history):
        """
        Update the reward for instances specified in [episode_history] to [new_reward]:
            This function is also different from the official version, which update the reward of every instance in the episode to [episode reward/step length]
            The users are responsible to store the instances for future reward update
            In return, they get the freedom to update the reward slot of arbitrary instances
        """
        for i in range(len(episode_history)):
            state, action, old_reward, timestamp = episode_history[i]
            # remove immediate feedback
            self.instance_history[state][action][old_reward].remove(timestamp)
            # add delayed feedback
            if action not in self.instance_history[state]:
                self.instance_history[state][action] = {}
            if new_reward not in self.instance_history[state][action]:
                self.instance_history[state][action][new_reward] = []
            self.instance_history[state][action][new_reward].append(timestamp)

        if len(self.instance_history[state][action][new_reward]) > self.lendeque:
            self.instance_history[state][action][new_reward] = self.instance_history[state][action][new_reward][-self.lendeque :]
