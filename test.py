import gym 
import gym_iOTA_2D
import numpy as np
env = gym.make('iOTA_2D-v0')
state = env.reset()
print(float('inf'))
env.render()
# print(env.step([pos+[np.random.uniform(),np.random.uniform()] for pos in state]))
print(env.step([[0,0] for _ in state]))
env.close()