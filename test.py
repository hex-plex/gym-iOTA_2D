import gym 
import gym_iOTA_2D

env = gym.make('iOTA_2D-v0')
env.reset()

env.render()
env.step('hello')