# import tflearn
import math
class Config(object):
        
    def __init__(self,\
        dim,\
        out,\
        ibl_type=None,\
        gamma=0.95):

        #Cognitive Config
        self.cog = self.Cognitive_Config()
        # Experience Replay Memory Config
        # self.erm = self.Experience_Replay_Memory_Config() 



        # Hyperparamters 
        # self.__drl          = drl          # DRL algorithm. Can serve as base for MADRL
        # self.__optimise     = optimise     # Disables optimiser
        self.__ibl_type     = ibl_type        # MA-DRL algorithm
        # self.__gpu          = gpu          # CPU, GPU device to be used for training
        self.__outputs      = out          # Number of outputs
        # self.__meta_actions = meta_actions # Number of meta actions
        self.__gamma        = gamma        # Discount rate
        self.__dim          = dim          # Set input dimensions
        self.__id           = None
        # self.__inc_sync     = False        # Used for incremental sync 
        # self.__tau          = 0.001        # Incremental sync rate
        # self.__sync_time    = 5000         # Steps between sync for non-inc approach
       
    def __repr__(self):
        return str(vars(self))

    @property
    def ibl_type(self):
        return self.__ibl_type

    @ibl_type.setter
    def ibl_type(self, value):
        if self.__ibl_type== None:
            self.__ibl_type = value
        else:
            raise Exception("Can't modify ibl_type.")

    @property
    def outputs(self):
        return self.__outputs

    @property
    def gamma(self):
        return self.__gamma

    @property
    def dim(self):
        return self.__dim

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        self.__id = value

    class Cognitive_Config(object):

        """ IBL Config """
        def __init__(self):
            self.__default_utility = 0.1
            self.__noise = 0.25
            self.__decay = 0.5
            self.__temperature = 0.25*math.sqrt(2)
            self.__lendeque = 1000
        def __repr__(self):
            return str(vars(self))

        @property
        def default_utility(self):
            return self.__default_utility
            
        @default_utility.setter
        def default_utility(self, value):
            self.__default_utility = value
        
        @property
        def noise(self):
            return self.__noise
        @noise.setter
        def noise(self, value):
            self.__noise = value
        
        @property
        def decay(self):
            return self.__decay
        @decay.setter
        def decay(self, value):
            self.__decay = value
        
        @property
        def temperature(self):
            return self.__temperature
        @temperature.setter
        def temperature(self, value):
            self.__temperature = value 

        @property
        def lendeque(self):
            return self.__lendeque
        @lendeque.setter
        def lendeque(self, value):
            self.__lendeque = value 