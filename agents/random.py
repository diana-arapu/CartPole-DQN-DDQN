import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = [i for i in range(env.action_space.n)]

    def select_action(self, state):
        return np.random.choice(self.action_space)
    
    def store_transition(self, state, action, next_state, reward, done):
        pass