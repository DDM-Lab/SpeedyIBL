import numpy as np
import random
import cv2
import copy

class NAVIGATION(object):
    """ Cooperative multi-agent transporation problem. """
    def __init__(self, version):
        '''
        :param version: Integer specifying which configuration to use
        '''
        if version == 1: # Standard
            from .envconfig_v1 import EnvConfigV1
            self.c = EnvConfigV1()

        # Fieldnames for stats
        self.fieldnames = ['Episode',
                           'Steps',
                           'Total_Reward',
                           'Goods_Delivered',
                           'Time']

        self.__dim = self.c.DIM     # Observation dimension
        self.__out = self.c.ACTIONS # Number of actions
        self.episode_count = 0      # Episode counter
        self.time = 0
        # Used to add noise to each cell
        self.ones = np.ones(self.c.DIM, dtype=np.float64)
        DIM = np.copy(self.c.DIM)
        DIM = np.append(DIM,3)
        self.im = np.zeros(DIM,dtype = np.float64)

        
    @property
    def dim(self):
        return self.__dim

    @property
    def out(self):
        return self.__out

    def render(self):
        '''
        Used to render the env.
        '''
        r = 16 # Number of times the pixel is to be repeated
        try:
            for i in range(self.c.NUMBER_OF_AGENTS):
                x, y = self.c.LANDMARK_LOCATIONS[i]
                self.im[y][x] = (0,255,0)
            for i in range(self.c.NUMBER_OF_AGENTS):
                self.im[self.old_agents_y[i]][self.old_agents_x[i]] -= self.c.colors[i]
                self.im[self.agents_y[i]][self.agents_x[i]] = self.c.colors[i]
            img = np.repeat(np.repeat(self.im, r, axis=0), r, axis=1).astype(np.uint8)
            cv2.imshow('image', img)
            # cv2.imshow('RGB', img)
            k = cv2.waitKey(1)
            if k == 27:         # If escape was pressed exit
                cv2.destroyAllWindows()
        except AttributeError:
            pass

    def stats(self):
        '''
        Returns stats dict
        '''
        stats = {'Episode': str(self.episode_count), 
                 'Steps': str(self.steps), 
                 'Total_Reward': str(self.reward_total),
                 'Goods_Delivered':str(self.delivered),
                 'Time':str(self.time)}
        return stats

    def reset(self):
        '''
        Reset everything. 
        '''
        self.s_t = np.zeros(self.c.DIM, dtype=np.float64)
          
        
        # Obstacles, agents and goods are initialised:
        # self.setObstacles()
        self.setLandmarks()
        self.initAgents()
        self.delivered = False
        # Used to keep track of the reward total acheived throughout 
        # the episode:
        self.reward_total = 0.0

        # Episode counter is incremented:
        self.episode_count += 1
   
        # For statistical purposes:
        # Step counter for the episode is initialised
        self.steps = 0 
        self.t_episode = False 

        # Moves taken in the same direction while carrying the goods
        self.individualSteps = [0, 0]
      
        return self.getObservations()

    def terminal(self):
        '''
        Find out if terminal conditions have been reached.
        '''
        return self.delivered

    def step(self, actions):
        '''
        Change environment state based on actions.
        :param actions: list of integers
        '''
        # Agents move according to actions selected
        self.moveAgents(actions)

        # Observations are loaded for each agent
        observations = self.getObservations()
                   
        self.goodsPickup()
        # Counters are incremented:
        self.steps += 1 
        
        rewards = self.goodsDelivered()

        self.reward_total += rewards[0] # Team game, so one r is sufficient  
        return observations, rewards, self.terminal() 

    def initAgents(self):
        '''
        Method for initialising the required number of agents and 
        positionsing them on designated positions within the grid
        '''

        # Agent x and y positions can be set in the following lists.
        self.agents_x = copy.deepcopy(self.c.AGENTS_X)
        self.agents_y = copy.deepcopy(self.c.AGENTS_Y)

        self.holding_goods= [False for i in range(self.c.NUMBER_OF_AGENTS)]
        self.arriveds= [False for i in range(self.c.NUMBER_OF_AGENTS)]
   
        # Agents are activated within the agent channel (2) of the gridworld
        # matrix.
        for i in range(self.c.NUMBER_OF_AGENTS):
                self.s_t[self.agents_y[i]][self.agents_x[i]] += self.c.AGENTS[i]


    def setLandmarks(self):
        '''
        Method used to initiate the obstacles within the environment 
        '''
        for i in range(self.c.NUMBER_OF_AGENTS):
            x, y = self.c.LANDMARK_LOCATIONS[i]
            self.s_t[y][x] = self.c.LANDMARKS[i]

    def goodsPickup(self):
        '''
        Method for picking up the tools, if the agents
        find themselves in positions adjecent to the goods.
        '''

        for j in range(self.c.NUMBER_OF_AGENTS):
            if (self.arriveds[j] == False):
                for i in range(self.c.NUMBER_OF_AGENTS):
                    x, y = self.c.LANDMARK_LOCATIONS[i]
                    if (self.agents_x[j] == x and  
                    self.agents_y[j] == y):
                        self.arriveds[j] = True



    def goodsDelivered(self):
        '''
        Method to check one of the goods 
        has been deliverd to the dropzone
        '''

        if (self.arriveds[0]*self.arriveds[1]*self.arriveds[2]) or self.t_episode:   
            r = 0     
            for i in range(self.c.NUMBER_OF_AGENTS):
                x, y = self.c.LANDMARK_LOCATIONS[i]
                if self.s_t[y][x] != self.c.LANDMARKS[i]:
                    r += 1
            self.delivered = True
            return [r,r,r]
            
        return [0.0, 0.0, 0.0]


    def getNoisyState(self):
        ''' 
        Method returns noisy state.
        '''
        return self.s_t
        

    def getObservations(self):
        '''
        Returns centered observation for each agent
        '''
        observations = []
        for i in range(self.c.NUMBER_OF_AGENTS):
            # Store observation
            observations.append(np.copy(self.getNoisyState()))
        return observations 

    def getDelta(self, action):
        '''
        Method that deterimines the direction 
        that the agent should take
        based upon the action selected. The
        actions are:
        'Up':0, 
        'Right':1, 
        'Down':2, 
        'Left':3, 
        'NOOP':4
        :param action: int
        '''
        if action == 0:
            return 0, -1
        elif action == 1:
            return 1, 0    
        elif action == 2:
            return 0, 1    
        elif action == 3:
            return -1, 0 
        elif action == 4:
            return 0, 0   

    def moveAgents(self, actions):
    #    '''
    #    Move agents according to actions.
    #    :param actions: List of integers providing actions for each agent
    #    '''
        self.old_agents_x = np.copy(self.agents_x)
        self.old_agents_y = np.copy(self.agents_y)
        for i in range(self.c.NUMBER_OF_AGENTS):
            if self.arriveds[i] == False:
                    dx, dy = self.getDelta(actions[i])
                    targetx = self.agents_x[i] + dx
                    targety = self.agents_y[i] + dy
                    if self.noCollision(targetx, targety) or (targetx, targety) in self.c.LANDMARK_LOCATIONS:
                        self.moveAgent(i, targetx, targety)

    def moveAgent(self, id, targetx, targety):
        '''
        Moves agent to target x and y
        :param targetx: Int, target x coordinate
        :param targety: Int, target y coordinate
        '''
        self.s_t[self.agents_y[id]][self.agents_x[id]] -= self.c.AGENTS[id]
        self.agents_x[id] = targetx
        self.agents_y[id] = targety
        self.s_t[self.agents_y[id]][self.agents_x[id]] += self.c.AGENTS[id]



    def noCollision(self, x, y):
        '''
        Checks if x, y coordinate is currently empty 
        :param x: Int, x coordinate
        :param y: Int, y coordinate
        '''
        if x < 0 or x >= self.c.GW or\
           y < 0 or y >= self.c.GH or\
           self.s_t[y][x] != 0:
            return False
        else:
            return True
    
