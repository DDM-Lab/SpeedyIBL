import random as random
from speedyibl import Agent
import numpy as np 
import time

from pyibl import Agent as PyAgent
import pandas as pd 

import argparse
flags = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="lightIBL")

flags.add_argument('--method',type=str,default='libl',help='method')
flags.add_argument('--trial',type=int,default=1000,help='Number of trials.')
flags.add_argument('--round',type=int,default=5000,help='Number of rounds.')
FLAGS = flags.parse_args()


acc = np.zeros((FLAGS.trial,FLAGS.round))
reward = np.zeros((FLAGS.trial,FLAGS.round))
time_run = np.zeros((FLAGS.trial,FLAGS.round+1))
for t in range(FLAGS.trial):
    print('finish trial: ',t,'method:',FLAGS.method)
    if FLAGS.method == 'libl':
        # agent = lAgent(default_utility=4.4,lendeque=5000)
        agent = Agent(default_utility=4.4,lendeque=5000)
        options = ['safe','risky']
        for i in range(FLAGS.round):
        
            start = time.time()
            choice = agent.choose(options)
            if choice == 'safe':
                agent.respond(3)
                reward[t,i] = 3
                # if random.random() <= 0.25:
                #     agent.respond(3)
                # else:
                #     agent.respond(0)
            elif random.random() <= 0.8:
                agent.respond(4)
                reward[t,i] = 4
            else:
                agent.respond(0)
                reward[t,i] = 0
            end = time.time()
            acc[t,i] = choice == 'risky'
             
            time_run[t,i+1] = time_run[t,i] + end - start
    elif FLAGS.method == 'ibl':
        # agent = Agent(default_utility = 4.4)
        agent = PyAgent(default_utility = 4.4)
        options = {'safe','risky'}
        for i in range(FLAGS.round):
            start = time.time()
            choice = agent.choose(*options)
            if choice == "safe":
                agent.respond(3)
                reward[t,i] = 3
                # if random.random() <= 0.25:
                #     agent.respond(3)
                # else:
                #     agent.respond(0)
            elif random.random() <= 0.8:
                agent.respond(4)
                reward[t,i] = 4
            else:
                agent.respond(0.0)
                reward[t,i] =0
            end = time.time()
            acc[t,i] = choice == "risky"
            time_run[t,i+1] = time_run[t,i] + end - start
acc_df = pd.DataFrame(acc)
acc_df.to_pickle('Results/binary/'+FLAGS.method+'acccase1.pkl')
time_df = pd.DataFrame(time_run)
time_df.to_pickle('Results/binary/'+FLAGS.method+'timecase1.pkl')
# reward_df = pd.DataFrame(reward)
# reward_df.to_pickle('Results/binary/'+FLAGS.method+'rewardcase1.pkl')