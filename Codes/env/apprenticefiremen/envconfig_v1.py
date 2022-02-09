import random
class EnvConfigV1:

    ''' Env. V1 Parameters '''
    def __init__(self):

        """ Gridworld Dimensions """
        self.GRID_HEIGHT = 16
        self.GRID_WIDTH = 15
        self.CHANNELS = 3
        self.GH = self.GRID_HEIGHT
        self.GW = self.GRID_WIDTH
        self.HMP = int(self.GW/2) # HMP = Horizontal Mid Point
        self.VMP = int(self.GH/2) # VMP = Vertical Mid Point
        self.CP = 1 # Cell padding around the gridwolrd
        self.ACTIONS = 4 # move up, down, left, righ
        """ Color Codes """
        self.CHALLENGING_COLORS = False
        self.NUMBER_OF_AGENTS = 2

        self.AGENTS = [240.0, 200.0] # Colors [Agent1, Agent2]
        self.OBSTACLE = 100.0
        self.FIRE = 50
        self.TOOLS = {'a':40, 'b':50, 'c':60}

        """ Agents """
        self.AGENTS_X = [2, self.GW-3]
        self.AGENTS_Y = [self.GH-2, self.GH-2]

        self.REWARDS = [(40, 40, lambda:11), (40, 50, lambda:-30.0),
                    (40, 60, lambda:0), (50, 40, lambda:-30.0), (50, 60, lambda: 0.0),
                    (60, 40, lambda:0.0), (60, 50, lambda:0.0), (60, 60, lambda:5.0), 
                    (50,50, lambda: 14.0 if random.uniform(0, 1) >= 0.5 else 0)] 
        
        """ Civilians """
        # self.CIVILIANS = civilians # Number of civilians
        # Boundary area within which the civilians must remain:
        self.Y_UPPER_BOUNDARY = self.VMP + 3
        self.Y_LOWER_BOUNDARY = self.VMP - 5
        self.X_UPPER_BOUNDARY = self.HMP + 5
        self.X_LOWER_BOUNDARY = self.HMP - 5
        self.CIVILIAN_SHIFT = 2

        # Fire coordinates
        self.FIRE_X = lambda: random.randrange(self.HMP-4, self.HMP+4, 2)
        self.FIRE_Y = lambda: random.randrange(self.VMP-5, self.VMP+3, 2)

        """ Obstacles """
        self.OBSTACLES_YX = []

        for i in range(self.GW):
            if i != self.HMP:
                self.OBSTACLES_YX.append((self.GH-3, i))
            if i!=self.HMP and i != self.HMP-2 and i != self.HMP+2:
                self.OBSTACLES_YX.append((1, i))
            self.OBSTACLES_YX.append((self.GH-1, i))
            self.OBSTACLES_YX.append((0, i))
 
            if i%2 == 1:
                self.OBSTACLES_YX.append((self.VMP-1, i))
                self.OBSTACLES_YX.append((self.VMP-3, i))
                self.OBSTACLES_YX.append((self.VMP-5, i))
                self.OBSTACLES_YX.append((self.VMP+1, i))
                self.OBSTACLES_YX.append((self.VMP+3, i))

        for i in range(self.GH):
            self.OBSTACLES_YX.append((i, 0))
            self.OBSTACLES_YX.append((i, 1))
            self.OBSTACLES_YX.append((i, self.GW-1))
            self.OBSTACLES_YX.append((i, self.GW-2))

        """ Tool Pickup Locations """
        # self.PICKUP_LOCATIONS = [('charge', self.HMP+2,1),
        #                          ('fire_blanket', self.HMP,1),
        #                          ('fire_exstinguisher', self.HMP-2, 1)]
        self.PICKUP_LOCATIONS = [('a', self.HMP+2,1),
                                 ('b', self.HMP,1),
                                 ('c', self.HMP-2, 1)]



        """ Agent observation space """
        self.AHEAD_VIEW = 6
        self.SIDE_VIEW = 6
        self.DIM = [self.GH, self.GW]
        self.OFFSET = self.AHEAD_VIEW


