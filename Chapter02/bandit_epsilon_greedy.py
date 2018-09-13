from bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0,0.1,0.5,1,1,1,1,1]
decay = [0,0,0,0,0.1,0.05,0.01,0.005]
rewards = []
for i in epsilons:
    rewards.append([])
n_steps = 5000


for i in range(len(epsilons)):
    bandit = Bandit(epsilon = epsilons[i], decay = decay[i])
    t = 0
    reward_count = [[],[],[],[]]

    while t < n_steps:
        Q = bandit.get_q_values(reward_count)
        action = bandit.e_greedy(Q)
        reward = bandit.pull_lever(action)
        reward_count[action-1].append(reward)
        t += 1
        rewards[i].append(bandit.get_mean_reward(reward_count))

    reward_count = np.sum(reward_count)
    print("Total Mean reward of {:5.3f} with epsilon {} and decay of {}".format(np.mean(reward_count), epsilons[i], decay[i]))
    print("Q Values: 1 = {:3.2f}, 2 = {:3.2f}, 3 = {:3.2f}, 4 = {:3.2f}\n".format(Q[0],Q[1],Q[2],Q[3]))

for i in range(len(rewards)):
    plt.plot([i for i in range(100,n_steps)], rewards[i][100:], label="Epsilon = {} Decay = {}".format(epsilons[i], decay[i]))
leg = plt.legend(loc='best', prop={'size': 6})
plt.show()