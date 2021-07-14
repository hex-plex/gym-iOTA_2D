import gym 
from gym import error, spaces, utils
from gym.utils import seeding

class Iota2DEnv(gym.Env):
    metadata={'render.modes':['human']}

    def __init__(self):
        print('__init__-works')

    def step(self,action):
        print('step-works')

    def reset(self):
        print('reset-works')

    def render(self, mode='human',close=False):
        print('render-works')
        