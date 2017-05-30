import gym
from GODAgent import GODAgent
from Environment import Env

env = Env()

agent = GODAgent()

agent.fit(env, nb_steps=1000000, visualize=False, verbose=1, nb_max_episode_steps=20)

agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=20)
