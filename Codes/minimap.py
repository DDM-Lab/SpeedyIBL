from environment import Environment
from stats import mkRunDir
from config import Config
from copy import deepcopy
# from agentIBL import AgentIBL_TD as Agent
# from agentIBL import AgentIBL_EQUAL as Agent
from agentIBL import AgentlightweightIBL_EQUAL as lAgent
from agentIBL import AgentPyIBL
import csv
import json
import time

import argparse
# import sys
# sys.argv=['']
# del sys
flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="lightIBL")

flags.add_argument('--environment',type=str,default='MINIMAP_V1',help='Environment.')
# flags.add_argument('--type',type=str,default='equalp',help='Environment.')
flags.add_argument('--type',type=str,default='libl',help='Environment.')
# flags.add_argument('--type',type=str,default='td',help='Environment.')
# flags.add_argument('--type',type=str,default='pyibl',help='Environment.')
flags.add_argument('--agents',type=int,default=1,help='Number of agents.')
flags.add_argument('--episodes',type=int,default=100,help='Number of episodes.')
flags.add_argument('--steps',type=int,default=2500,help='Number of steps.')
flags.add_argument('--start_t',type=int,default=0,help='Number of trials.')
flags.add_argument('--trials',type=int,default=100,help='Number of trials.')
FLAGS = flags.parse_args()

# trajectory = {}


for nrun in range(FLAGS.start_t,FLAGS.trials):

    env = Environment(FLAGS) 

    # Example:
    config = Config(env.dim, env.out)
    # Run dir and stats csv file are created
    # statscsv, folder = mkRunDir(env, config = FLAGS, sequenceid=nrun)
    statscsv, folder = mkRunDir(env, FLAGS, nrun)
    # statscsv, folder = mkRunDir(env, FLAGS)

    # Agents are instantiated
    agents = []
    for i in range(FLAGS.agents): 
        agent_config = deepcopy(config)
        if FLAGS.type == "libl":
            agent = lAgent(agent_config,default_utility=0.1,lendeque=10000) # Init agent instances
        if FLAGS.type == "ibl":
            agent = AgentPyIBL(agent_config,default_utility=0.1)
        # agent.ibl = agent
        f = open(folder + 'agent' + str(i) + '_config.txt','w')
        f.write(str(vars(agent_config)))
        f.close()

    # Start training run
    for i in range(FLAGS.episodes):
        # Run episode

        observations = env.reset() # Get first observations
        # if i == FLAGS.episodes - 1:
        #     trajectory[str(0)] = {'y':env.env.agents_y, 'x':env.env.agents_x, 'reward':0}
        start = time.time()
        for j in range(FLAGS.steps):
            # Renders environment if flag is true
            # if FLAGS.render: env.render() 
            # if i == 99:
            #     env.render() 
            # Load action for each agent based on o^i_t
            # actions = [] 
            # for agent, observation in zip(agents, observations):
            action = agent.move(observations,env.env.agents_y,env.env.agents_x)
            # print(actions)
            # Optimise agent
            # agent.opt() 
            
            # Execute actions and get feedback lists:
            observations, rewards, t = env.step(action)
            # print(observations)
            # Check if last step has been reached
            ter = False
            if rewards > 0:
                ter = True
            if j == FLAGS.steps-1:
                t = True
        
            # for agent, o, r in zip(agents, observations, rewards):
                # Pass o^i_{t+1}, r^i_{t+1} to each agent i
            agent.feedback(rewards, t, ter, env.env.s_o)
            # agent.opt()

            # if i == FLAGS.episodes - 1:
            #     trajectory[str(j+1)] = {'y':env.env.agents_y, 'x':env.env.agents_x, 'reward':rewards}

            
            if t: break # If t then terminal state has been reached
            # if t or test_time > 300: break
        end = time.time()
        env.env.time += end - start
        # Add row to stats: 
        with open(statscsv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
            writer.writerow(env.stats())
        # print(env.stats())

    # with open('trajectory3000.json', 'w') as testfile:
    #     json.dump(trajectory,testfile)


