import numpy as np
import random


def greedy(bandit, timesteps):
    rewards = np.zeros(bandit.n_arms)
    n_plays = np.zeros(bandit.n_arms)
    Q = np.zeros(bandit.n_arms)
    possible_arms = range(bandit.n_arms)

    # Initialize variables by playing each arm once
    for x in range(bandit.n_arms):
        n_plays[x] = 1
        rewards[x] = bandit.play_arm(x)
        Q[x] = rewards[x] / n_plays[x]

    # Main loop
    while bandit.total_played < timesteps:
        max_index = np.argmax(Q)
        reward_for_a = bandit.play_arm(max_index)

        Q[max_index] = Q[max_index] + (1 / (n_plays[max_index])) * (rewards[max_index] - Q[max_index])
        rewards[max_index] = reward_for_a
        n_plays[max_index] += 1
