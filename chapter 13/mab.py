# -*- coding: utf-8 -*-
import numpy as np

np.random.seed(987654321)


class Bernoulli:
    @staticmethod
    def soft_max(z):
        ez = np.exp(z)
        dist = ez / np.sum(ez)
        return dist

    def __init__(self, num):
        self._num = num
        self._bernoulli_p = self.soft_max([i for i in range(self._num)])

    def draw(self, arm):
        p = self._bernoulli_p[arm]
        return 0.0 if np.random.random() > p else 1.0


class MAB:
    def __init__(self, arm_num):
        self._arm_num = arm_num
        self._bernoulli_arm = Bernoulli(self._arm_num)

    def _random_arm(self):
        return np.random.choice(self._arm_num)


class Random(MAB):
    def __init__(self, arm_num):
        super().__init__(arm_num)

    def _select(self):
        return self._random_arm()

    def get_reward(self, pull_num):
        rewards = []
        for i in range(pull_num):
            chosen_arm = self._select()
            chosen_arm_reward = self._bernoulli_arm.draw(chosen_arm)
            rewards.append(chosen_arm_reward)
        return np.cumsum(rewards) / (1 + np.arange(pull_num))


class EpsilonGreedy(MAB):
    def __init__(self, epsilon, arm_num):
        super().__init__(arm_num)
        self._epsilon = epsilon
        self._counts = np.zeros(self._arm_num)
        self._mean_rewards = np.zeros(self._arm_num)

    def _best_arm(self):
        return np.argmax(self._mean_rewards)

    def _select(self):
        rand = np.random.random()
        return self._best_arm() if rand > self._epsilon else self._random_arm()

    def _update(self, arm, reward):
        arm_counts = self._counts[arm]
        arm_mean_reward = self._mean_rewards[arm]
        cumulative_arm_reward = arm_counts * arm_mean_reward

        updated_cumulative_arm_reward = cumulative_arm_reward + reward
        updated_arm_counts = arm_counts + 1
        updated_arm_mean_reward = updated_cumulative_arm_reward / updated_arm_counts
        self._counts[arm] = updated_arm_counts
        self._mean_rewards[arm] = updated_arm_mean_reward

    def get_reward(self, pull_num):
        rewards = []
        for i in range(pull_num):
            chosen_arm = self._select()
            chosen_arm_reward = self._bernoulli_arm.draw(chosen_arm)
            self._update(chosen_arm, chosen_arm_reward)
            rewards.append(chosen_arm_reward)

        return np.cumsum(rewards) / np.arange(1, pull_num + 1)


if __name__ == '__main__':
    pulls = 50000
    arms = 5
    epsilon = 0.1

    random_select = Random(arms)
    epsilon_greedy = EpsilonGreedy(epsilon, arms)
    rand_rewards = random_select.get_reward(pulls)
    eg_rewards = epsilon_greedy.get_reward(pulls)

    import matplotlib.pyplot as plot

    plot.plot(rand_rewards, label='random')
    plot.plot(eg_rewards, label='epsilon')
    plot.xlabel('pull')
    plot.ylabel('average_rewards')

    plot.legend()
    plot.show()
