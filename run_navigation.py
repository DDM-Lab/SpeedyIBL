
from environment import Environment
from stats import mkRunDir
from copy import deepcopy
from itertools import count
import random as random
import numpy as np 
import time
import csv

from agentIBL import AgentlightweightIBL as Agent
from agentIBL import AgentPyIBL

import argparse
flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="lightIBL")

flags.add_argument('--environment',type=str,default='NAVIGATION_V1',help='Environment.')
flags.add_argument('--type',type=str,default='libl',help='method - ibl: PyIBL - libl: speedyIBL')
flags.add_argument('--agents',type=int,default=3,help='Number of agents.')
flags.add_argument('--episodes',type=int,default=100,help='Number of episodes.')
flags.add_argument('--steps',type=int,default=2500,help='Number of steps.')
flags.add_argument('--start_t',type=int,default=0,help='Starting trial/run.')
flags.add_argument('--trials',type=int,default=1,help='Ending trial/run.')
flags.add_argument('--render',type=bool,default=False,help='Animation.')
FLAGS = flags.parse_args()


for runid in range(FLAGS.start_t,FLAGS.trials):

    env = Environment(FLAGS) 

    # statscsv, folder = mkRunDir(env, FLAGS, runid)
    statscsv, folder = mkRunDir(env, FLAGS)

    # Agents are instantiated
    agents = []
    for i in range(FLAGS.agents): 
        if FLAGS.type == "libl":
            agents.append(Agent(env.out,default_utility=2.5)) # Init agent instances
        if FLAGS.type == "ibl":
            agents.append(AgentPyIBL(env.out,default_utility=2.5))

        # f = open(folder + 'agent' + str(i) + '_config.txt','w')
        # f.write(str(vars(agent_config)))
        # f.close()

    ##################

    # Start training run
    for i in range(FLAGS.episodes):
        
        # Run episode
        observations = env.reset() # Get first observations

        start = time.time()
        for j in range(FLAGS.steps):
            if j == FLAGS.steps-1:
                env.env.t_episode = True
            #######################################
            if FLAGS.render: env.render() 
            arriveds = deepcopy(env.env.arriveds)
            actions = [4,4,4]
            for a in range(FLAGS.agents):
                if not arriveds[a]:
                    actions[a] = agents[a].move(observations[a])

            observations, rewards, t = env.step(actions)

            if j == FLAGS.steps-1:
                t = True

            for a, r in zip(range(FLAGS.agents),rewards):
                if not arriveds[a]: 
                    agents[a].feedback(r)  
            if t:         
                for agent, r in zip(agents, rewards):
                    agent.delay_feedback(r) 



            if t: break # If t then terminal state has been reached
        end = time.time()
        env.env.time += end - start
        # Add row to stats: 
        with open(statscsv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
            writer.writerow(env.stats())
        print(env.stats())

