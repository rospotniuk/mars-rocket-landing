from Environment_MarsLanding import *
import random

env = RocketEnv()
env.reset()
for i in range(1000):
    env.render()
    env.step([random.choice([0,1,2,3,4,5]), i])