import re

class Environment(object):

    """ Environment """
    def __init__(self, flags):
        self.__name = flags.environment

        if 'NAVIGATION' in self.name:
            params = self.name.split('_')
            if 'V' in params[1]:
                version = int(re.sub("[^0-9]", "", params[1]))
                from env.navigation.navigation import NAVIGATION
                self.__env = NAVIGATION(version)
                
            else:
                raise ValueError('Invalid environment string format for PREDATOR')
        
        self.__fieldnames = self.__env.fieldnames
        self.__dim = self.env.dim
        self.__out = self.env.out


    @property
    def dim(self):
        return self.__dim

    @property
    def out(self):
        return self.__out

    @property
    def name(self):
        return self.__name

    @property
    def env(self):
        return self.__env

    @property
    def fieldnames(self):
        return self.__fieldnames

    def evalReset(self, evalType):
        '''
        Reset for evaulations
        '''
        return self.__env.evalReset(evalType)

    def reset(self):
        '''
        Reset env to original state
        '''
        return self.__env.reset()

    def render(self):
        '''
        Render the environment
        '''
        self.__env.render()

    def step(self, a):
        '''
        :param float: action
        '''
        return self.__env.step(a)

    def stats(self):
        '''
        :return: stats from env
        '''
        return self.__env.stats()
    
    def result(self):
        '''
        :return: stats from env
        '''
        return self.__env.result()
