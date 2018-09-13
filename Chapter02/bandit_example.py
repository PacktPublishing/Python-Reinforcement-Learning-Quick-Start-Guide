from bandit import Bandit

done = False

while not done:
    bandit = Bandit()
    G = 0
    t = 0

    while t < 10:
        action = bandit.get_action()
        reward = bandit.pull_lever(action)
        G += reward
        t += 1
        print("Received a reward of ({}) taking action ({})".format(reward, action))
        print("Total accumulated reward: {}\n".format(G))

    print("Average Reward: {:4.3f}".format(G/t))
    done = bandit.done()