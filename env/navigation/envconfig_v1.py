import random
class EnvConfigV1:

    ''' Env. V1 Parameters '''
    def __init__(self):

        """ Gridworld Dimensions """
        self.GRID_HEIGHT = 16
        self.GRID_WIDTH = 16
        self.CHANNELS = 3
        self.GH = self.GRID_HEIGHT
        self.GW = self.GRID_WIDTH
        self.HMP = int(self.GW/2) # HMP = Horizontal Mid Point
        self.VMP = int(self.GH/2) # VMP = Vertical Mid Point
        self.CP = 1 # Cell padding around the gridwolrd
        self.ACTIONS = 4 # move up, down, left, righ
        """ Color Codes """
        self.CHALLENGING_COLORS = False
        self.NUMBER_OF_AGENTS = 3
        self.MID = 8

        self.AGENTS = [240.0, 200.0, 150] # Colors [Agent1, Agent2, Agent3]
        self.OBSTACLE = 100.0
        self.FIRE = 50
        self.LANDMARKS = [40, 50, 60]

        """ Agents """
        self.AGENTS_X = [0, self.GW-1, 0]
        self.AGENTS_Y = [0, 0, self.GH-1]

        #x,y coordinates
        self.LANDMARK_LOCATIONS = [(self.MID-2,self.MID-2),
                                 (self.MID+1,self.MID-2),
                                 (self.MID-2, self.MID+1)]

        self.colors = [(255,0,0),(0,0,250),(255,215,0)]

        """ Agent observation space """
        self.DIM = [self.GH, self.GW]



