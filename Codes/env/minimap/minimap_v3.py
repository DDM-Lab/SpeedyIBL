import pandas as pd
import numpy as np
class EnvConfigV3:

    ''' Env. V1 Parameters '''
    def __init__(self):
        """ Gridworld Dimensions """
        df = pd.read_csv('map_hard.csv')
        self.GRID_HEIGHT = df['z'].max() + 1
        self.GRID_WIDTH = df['x'].max() + 1
        # self.GRID_HEIGHT = 16
        # self.GRID_WIDTH = 16
        # self.MID = self.GRID_WIDTH // 2
        self.GH = self.GRID_HEIGHT
        self.GW = self.GRID_WIDTH
        self.DIM = [self.GH, self.GW]
        # self.HMP = int(self.GW/2) # HMP = Horizontal Mid Point
        # self.VMP = int(self.GH/2) # VMP = Vertical Mid Point
        self.ACTIONS = 4 # No-op, move up, down, left, righ

        """ Wind (slippery surface) """
        self.WIND = 0.0
       
        """ Agents """
        self.NUMBER_OF_AGENTS = 1
        self.AGENTS_X = 20
        self.AGENTS_Y = 5

        """ Goods """
        # self.GOODS_X = self.MID # Pickup X Coordinate
        # self.GOODS_Y = 11       # Pickup y Coordinate
        # self.GOODS_Y = 7       # Pickup y Coordinate
      
        """ Goals """
        self.GOALS_YX = [] # [(2,0,lambda:1.0)] # X, Y, Reward
 
        """ Colors """
        self.AGENTS = 240.0 # Colors [Agent1, Agent2]
        self.GOAL = [150.0, 200.0]
        self.OBSTACLE = 100.0
        self.number_goals = 0

        # Noise related parameters.
        # Used to turn CMOTP into continous environment
        self.NOISE = 0
        self.MU = 1.0
        self.SIGMA = 0.0

        """ Obstacles """
        self.OBSTACLES_YX = []

        self.n = df.shape[0]
        for i in range(self.n):
            # print(i)
            info_row = df.iloc[i]
            # print(i)
            if info_row['key'] in ['walls','stairs']:
                self.OBSTACLES_YX.append((info_row['z'], info_row['x']))
            elif info_row['key'] == 'yellow victims':
                self.GOALS_YX.append((info_row['z'], info_row['x'],lambda:0.75))
                self.number_goals += 1
            elif info_row['key'] == 'green victims':
                self.GOALS_YX.append((info_row['z'], info_row['x'],lambda:0.25))
                self.number_goals += 1

