from rl.core import Agent
import numpy as np

class GODAgent(Agent):
    """Write me
    """

    def __init__(self):
        self.compiled = True
        self.processor = None
        pass

    def compile(self, optimizer, metrics=[]):
        pass

    def forward(self, observation):
        x = observation
        action = [-x[1]/2.0/x[0]]-x[3]
        return action

    def backward(self, reward, terminal=False):
        return []