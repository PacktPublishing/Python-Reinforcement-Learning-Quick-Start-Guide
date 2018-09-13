import random
import numpy as np

class Bandit():

    def __init__(self, epsilon=None, decay=0):
        self.action = None
        self.decay = decay
        self.epsilon = epsilon

    def pull_lever(self, action):
        self.action = None
        if action == 1:
            return  int(random.normalvariate(0,3))
        elif action == 2:
            return  random.randint(-10,8)
        elif action == 3:
            return  int(random.normalvariate(1,0.5))
        elif action == 4:
            if random.random() <= 0.2:
                return int(random.normalvariate(6,2))
            else:
                return int(random.normalvariate(0,3))

    def get_action(self):
        while self.action not in range(1,5):
            self.action = int(input("Choose an action between 1-4: " ))
        return self.action

    def done(self):
        if input("Would you like to play again? [y/n]: ").lower() == 'n':
            return True
        else:
            return False

    def get_steps(self):
        return int(input("How many steps would you like to sample? "))

    def random_action(self):
        return random.randint(1,4)

    def e_greedy(self, Q, show_random=False):
        if random.random() < self.epsilon:
            self.epsilon -= self.decay
            if show_random:
                print("Taking a random action!")
            return self.random_action()
        else:
            return np.argmax(Q) + 1

    def get_q_values(self, rewards):
        q = [[],[],[],[]]
        for i in range(len(rewards)):
            if rewards[i] == []:
                q[i] = 0
            else:
                q[i] = np.mean(rewards[i])
        return q

    def get_mean_reward(self, rewards):
        all_rewards = []
        for i in rewards:
            if i != []:
                all_rewards.extend(i)
        return np.mean(all_rewards)

