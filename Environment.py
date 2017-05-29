import numpy as np
import gym
from gym import spaces

class Env(object):
    def __init__(self):
        self.action_shape = [1, ]
        self.observation_shape = [1, ]
        self.action_space = spaces.Box(-10, 10, shape=self.observation_shape)
        self.observation_space = spaces.Box(0, 1e6, shape=self.action_shape)
        self._seed = 0
        self.status = np.random.random(self.action_shape)
        self.action = self.action_space.sample()
        self.reward = 0
        self.loss = self.foo(self.status)

    def foo(self, x):
        return np.sum(np.power(x, 2) + 2 * x + 5) - 4

    def reset(self):
        self.status = np.random.random(1)
        return self.observe()

    def seed(self, _int):
        np.random.seed(_int)

    def observe(self):
        return np.array(self.foo(self.status)).reshape(self.observation_shape)

    def step(self, action):
        """

        :param action:
        :return:
            observation (object):
            reward (float): sum of rewards
            done (bool): whether to reset environment or not
            info (dict): for debug only
        """
        self.status += action
        self.action = action
        observation = self.observe()
        tmp = self.foo(self.status)
        self.reward = self.loss - tmp
        self.loss = tmp
        self.reward += self.loss - self.foo(self.status)
        self.loss = self.foo(self.status)
        done = self.loss <= 1e-4 or self.loss > 1e10
        info = {}
        return observation, self.reward, done, info

    def render(self, mode='human', close=False):
        print('\nreward: ', self.reward)
        print('status: ', self.status)
        print('action: ', self.action)
        print('loss: ', self.loss)
