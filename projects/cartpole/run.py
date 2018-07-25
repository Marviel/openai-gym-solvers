# Title:         run.py
# Author:        Luke Bechtel
# Description:
#  Runs an episode of the environment and calculates reward for that episode.
import numpy as np


def hillclimb_run_episode(env, parameters, max_steps, show=False):
    matmul_run_episode(env, parameters, max_steps, show=False)


def matmul_run_episode(env, parameters, max_steps, show=False):
    observation = env.reset()
    totalreward = 0
    for _ in xrange(max_steps):
        if show:
            env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        (observation, reward, done, info) = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward
