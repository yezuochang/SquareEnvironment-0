import numpy as np
import gym
from gym import spaces
from copy import deepcopy
class Env(object):
    def __init__(self):
        self.action_shape = [1, ]
        self.observation_shape = [5, ]
        self.action_space = spaces.Box(-0.1, 0.1, shape=self.action_shape)
        self.observation_space = spaces.Box(-0.1, 0.1, shape=self.observation_shape)
        self._seed = 0
        self.action = self.action_space.sample()
        self.reward = 0
        self.reset()

    def foo(self, x):
        y = self.coefs[0]*np.power(x[0], 2) + self.coefs[1] * x[0] + self.coefs[2]
        return y

    def reset(self):
        # print('\n\n--------------------------------------------------------------------------------')
        self.coefs = np.random.rand(3)*10
        self.status = np.random.random(1)*10
        self.init_status = deepcopy(self.status)
        self.loss = self.foo(self.status)
        self.nb_step = 0

        # print('init_loss = ', self.loss)
        return self.observe(self.loss)

    def seed(self, _int):
        np.random.seed(_int)

    def observe(self, loss):
        return np.concatenate([self.coefs, np.array(self.status), [loss/100.]])

    def step(self, action):
        """

        :param action:
        :return:
            observation (object):
            reward (float): sum of rewards
            done (bool): whether to reset environment or not
            info (dict): for debug only
        """
        # print(self.status, action)
        self.nb_step += 1
        self.status += action
        self.action = action
        tmp = self.foo(self.status)
        observation = self.observe(tmp)
        self.reward = self.loss - tmp - 0.1
        self.loss = tmp
        done = np.abs(action[0]) < 1e-4 or self.loss > 10000 or self.nb_step >= 20
        info = {}
        return observation, self.reward, done, info

    def render(self, mode='human', close=False):
        print('\n\ninit: ', self.init_status)
        print('coefs: ', self.coefs)
        print('reward: ', self.reward)
        print('action: ', self.action)
        print('loss: ', self.loss)
        print('status: ', self.status)
        print('solution', self.solution())

    def solution(self):
        return -self.coefs[1]/self.coefs[0]/2.0
