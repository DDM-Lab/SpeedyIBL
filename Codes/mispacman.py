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

flags.add_argument('--environment',type=str,default='MISPACMAN',help='Environment.')
flags.add_argument('--type',type=str,default='ibl',help='Environment.')
flags.add_argument('--agents',type=int,default=1,help='Number of agents.')
flags.add_argument('--episodes',type=int,default=100,help='Number of episodes.')
flags.add_argument('--steps',type=int,default=2500,help='Number of steps.')
flags.add_argument('--n_change_steps',type=int,default=1,help='Number of steps.')
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
            agent = lAgent(agent_config,default_utility=2.5,lendeque=5000) # Init agent instances
        if FLAGS.type == "ibl" or FLAGS.type == "latestibl":
            agent = AgentPyIBL(agent_config,default_utility=2.5)
        # agent.ibl = agent
        f = open(folder + 'agent' + str(i) + '_config.txt','w')
        f.write(str(vars(agent_config)))
        f.close()

    # Start training run
    for i in range(FLAGS.episodes):
        # Run episode

        observations = env.reset() # Get first observations
        start = time.time()
        for j in range(FLAGS.steps):
            # Renders environment if flag is true
            # if FLAGS.render: env.render() 
            # if i == 99:
            # env.render() 
           
            if j% FLAGS.n_change_steps == 0:
                action = agent.move(observations,0,0)
            else:
                s_hash = hash(observations.tobytes())
                if FLAGS.type =='ibl':
                    options = [{"action": action, "s": s_hash}]
                    action = agent.choose(*options)
                    action = action["action"]
                else:
                    options = [(s_hash,action)]
                    action = agent.choose(options)
                    action = action[1]
                agent.last_action = action
                agent.current = s_hash



            
            # Execute actions and get feedback lists:
            observations, rewards, t = env.step(action)
            # Check if last step has been reached
            if rewards> 0:
                ter = True
            else:
                ter = False
            
            if j == FLAGS.steps-1:
                t = True

            agent.feedback(rewards, t, ter, env.env.s_t)
          
            if t: break # If t then terminal state has been reached

        end = time.time()
        env.env.time += end - start
        # Add row to stats: 
        with open(statscsv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=env.fieldnames)
            writer.writerow(env.stats())
        print(env.stats())

