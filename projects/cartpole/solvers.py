# Title:         solvers.py
# Author:        Luke Bechtel
# Description:
#   Various solvers for the cartpole problem.
import numpy as np

from .run import run_episode


def hillclimb_solver(env, gym, generations, max_episode_steps,
                     noise_scaling, noise_learning_rate):
    parameters = np.random.rand(4) * 2 - 1
    bestreward = 0
    for _ in xrange(generations):
        new_parameters = parameters * (np.random.rand(4) * 2 - 1)\
                            * noise_scaling
        reward = run_episode(env, new_parameters, max_episode_steps)
        if reward > bestreward:
            bestreward = reward
            parameters = new_parameters
            # considered solved if the agent lasts 200 timesteps
            if reward == 200:
                # Return the number of iterations
                return _, parameters
        else:
            noise_scaling += noise_learning_rate

    return 2000, None
