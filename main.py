import numpy as np
import matplotlib.pyplot as plt
from gaussian_bandit import GaussianBandit
from greedy import greedy
from epsilon_greedy import epsilon_greedy

def main():
    n_episodes = 10000
    n_timesteps = 1000
    rewards_greedy = np.zeros(n_timesteps)
    rewards_egreedy = np.zeros(n_timesteps)

    for i in range(n_episodes):
        if i % 100 == 0:
            print(f"Current episode: {i}")

        b = GaussianBandit()
        greedy(b, n_timesteps)
        rewards_greedy += b.rewards

        b.reset()
        epsilon_greedy(b, n_timesteps)
        rewards_egreedy += b.rewards

    rewards_greedy /= n_episodes
    rewards_egreedy /= n_episodes

    plt.plot(rewards_greedy, label="Greedy")
    plt.plot(rewards_egreedy, label="Epsilon-Greedy")
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig('bandit_strategies.eps')
    plt.show()

if __name__ == "__main__":
    main()
