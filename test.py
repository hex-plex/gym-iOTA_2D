import gym 
import gym_iOTA_2D

env = gym.make('iOTA_2D-v0')
print(env.action_space.low)
print(env.observation_space.high)
print(env.reset())
env.render()
print(env.step([[0,0] for i in range(10)]))
for i in range (100000000):
    pass

env.close()