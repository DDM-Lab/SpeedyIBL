
from environment import Environment
from stats import mkRunDir
from config import Config
from copy import deepcopy
from itertools import count
import random as random
import numpy as np 
import time
import csv

from agentIBL import AgentlightweightIBL_EQUAL as Agent
from agentIBL import AgentPyIBL

import argparse
flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="lightIBL")

flags.add_argument('--environment',type=str,default='FIREMAN_V1',help='Environment.')
flags.add_argument('--type',type=str,default='libl',help='method')
flags.add_argument('--agents',type=int,default=2,help='Number of agents.')
flags.add_argument('--episodes',type=int,default=100,help='Number of episodes.')
flags.add_argument('--steps',type=int,default=2500,help='Number of steps.')
flags.add_argument('--start_t',type=int,default=0,help='Number of trials.')
flags.add_argument('--trials',type=int,default=100,help='Number of trials.')
FLAGS = flags.parse_args()


for runid in range(FLAGS.start_t,FLAGS.trials):
    # Environment is instantiated
    # The dimension of the obsrvations and the number 
    # of descrete actions can be accessed as follows:
    # 
    # transitions = {}
    # ttime = 0
    env = Environment(FLAGS) 

    # Example:
    config = Config(env.dim, env.out)

    # random.seed(99*runid)
    # Run dir and stats csv file are created
    statscsv, folder = mkRunDir(env, FLAGS, runid)
    # statscsv, folder = mkRunDir(env, FLAGS)

    # Agents are instantiated
    agents = []
    for i in range(FLAGS.agents): 
        agent_config = deepcopy(config)
        if FLAGS.type == "libl":
            agents.append(Agent(agent_config,default_utility=13,lendeque=10000)) # Init agent instances
        if FLAGS.type == "ibl":
            agents.append(AgentPyIBL(agent_config,default_utility=13))

        f = open(folder + 'agent' + str(i) + '_config.txt','w')
        f.write(str(vars(agent_config)))
        f.close()

    ##################

    # Start training run
    for i in range(FLAGS.episodes):
        
        # Run episode
        observations = env.reset() # Get first observations
        # transitions.append([np.copy(env.env.agents_x),np.copy(env.env.agents_y),np.copy([env.env.goods_x,env.env.goods_y]),False])
        # transitions[str(ttime)] = {'x1':env.env.agents_x[0], 'x2':env.env.agents_x[1],'y1':env.env.agents_y[0],'y2':env.env.agents_y[1],'x':env.env.goods_x,'y':env.env.goods_y,'t':False}
        # ttime = ttime +1
        start = time.time()
        for j in range(FLAGS.steps):

            #######################################
            # Renders environment if flag is true
            # if FLAGS.render: env.render() 
            # env.render()
            # Load action for each agent based on o^i_t
            actions = [] 
            for agent, observation in zip(agents, observations):
                actions.append(agent.move(observation,env.env.agents_y,env.env.agents_x))
            
            agents_x = deepcopy(env.env.agents_x)
            agents_y = deepcopy(env.env.agents_y)

            observations, rewards, t = env.step(actions)
            ter = t 
            if j == FLAGS.steps-1:
                t = True

            for agent, o, r, action, x, y in zip(agents, observations, rewards, actions, agents_x, agents_y):
                dx, dy = env.env.getDelta(action)
                # if y + dy > env.env.c.GH-1 or x + dx > env.env.c.GW-1 or o[y + dy, x + dx]== env.env.c.OBSTACLE:
                #     r = -0.05
                # if r == 0:
                #     r = -0.01

                agent.feedback(r, t, ter, o) 



            if t: break # If t then terminal state has been reached
        end = time.time()
        env.env.time += end - start
        # Add row to stats: 
        with open(statscsv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
            writer.writerow(env.stats())
        # print(env.stats())
    # with open(str(runid)+'transition.json', 'w') as testfile:
    #     json.dump(transitions,testfile)

