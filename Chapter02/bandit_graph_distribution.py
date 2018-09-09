from bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt

bandit = Bandit()
n_steps = bandit.get_steps()
G = 0
t = 0
reward_count = [[],[],[],[]]

while t < n_steps:
    action = bandit.random_action()
    reward = bandit.pull_lever(action)
    reward_count[action-1].append(reward)
    t += 1

for i in range(4):
    print("Action {} value: {:4.3f}".format(i+1, np.mean(reward_count[i])))

# Create plot
fig, axes = plt.subplots()
# Plot violin plot
axes.violinplot(reward_count, showmeans=True, showmedians=False)
axes.yaxis.grid(True)
axes.set_xticks([1,2,3,4])
axes.set_xlabel("Action")
axes.set_ylabel("Reward")
plt.show()