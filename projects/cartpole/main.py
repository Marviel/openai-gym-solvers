# Title:         main.py
# Author:        Luke Bechtel
# Description:
#  Runs an example for solving cartpole, using various solvers.
import gym
import matplotlib.pyplot as plt

from .solvers import hillclimb_solver
from .run import run_episode


def main():
    env = gym.make('CartPole-v0')

    its_to_solve = []
    best_its = 10000000000
    best_params = None
    for i in range(50):
        its, params = hillclimb_solver(env, gym, 2000, 200, .1, .01)
        its_to_solve.append(its)
        if its < best_its:
            best_its = its
            best_params = params

    print(best_params)
    plt.hist(its_to_solve, bins=30, color="g")
    plt.title("Solution Histogram")
    plt.xlabel("Iterations Required")
    plt.ylabel("Frequency")
    plt.show()

    run_episode(env, best_params, 200, show=True)


if __name__ == "__main__":
    main()
