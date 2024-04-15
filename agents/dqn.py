import torch
import numpy as np
from collections import deque, namedtuple
import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, state_size, n_actions): # n_actions = env.action_space.n
        super(DQNModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_size, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24,24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, n_actions)
        )
        self.device = torch.device('cpu') # 'cuda:0' if torch.cuda.is_available() else 
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        self.criterion = torch.nn.MSELoss()

    def forward(self, state):
        q_values = self.model(state)
        return q_values

class DQNAgent:
    def __init__(self, env):
        self.gamma = 1
        self.epsilon = 1 ## start epsilon
        self.epsilon_end = 0.05
        self.epsilon_decay= 0.99#0.997
        self.env = env
        self.action_space = [i for i in range(env.action_space.n)]
        self.state_size = self.env.observation_space.shape[0]

        self.primary_model = DQNModel(self.state_size, env.action_space.n)
        self.target_model = DQNModel(self.state_size, env.action_space.n)
        self.target_model.load_state_dict(self.primary_model.state_dict())
        self.tau = 0.001
        self.c = 1000

        self.memory_capacity = 100000
        self.memory_counter = 0
        self.states_memory = np.zeros((self.memory_capacity, self.state_size), dtype=np.float32)
        self.next_states_memory = np.zeros((self.memory_capacity, self.state_size), dtype=np.float32)
        self.actions_memory = np.zeros(self.memory_capacity, dtype=np.int32)
        self.rewards_memory = np.zeros(self.memory_capacity, dtype=np.int32)
        self.terminal_memory = np.zeros(self.memory_capacity, dtype=np.bool)

    def store_transition(self, state, action, next_state, reward, done):
        index = self.memory_counter % self.memory_capacity
        self.states_memory[index] = state
        self.next_states_memory[index] = next_state
        self.actions_memory[index] = action
        self.rewards_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1
    
    def select_action(self, obs):
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(obs).to(self.primary_model.device)
            q_val = self.primary_model.forward(state)
            action = torch.argmax(q_val).item()
        return action

    def soft_update(self):
        target_network_weights = self.target_model.state_dict()
        primary_network_weights = self.primary_model.state_dict()
        for key in primary_network_weights:
            target_network_weights[key] = primary_network_weights[key]*self.tau + target_network_weights[key]*(1-self.tau)
        self.target_model.load_state_dict(target_network_weights)

    def hard_update(self, iterations):
        primary_network_weights = self.primary_model.state_dict()
        if iterations%self.c == 0:
            self.target_model.load_state_dict(primary_network_weights)

    def train(self, batch_size):
        self.primary_model.optimizer.zero_grad()
        memory_available = min(self.memory_counter, self.memory_capacity)
        batch = np.random.choice(memory_available, batch_size, replace=False)
        batch_index = np.arange(batch_size, dtype=np.int32)

        states = torch.tensor(self.states_memory[batch]).to(self.primary_model.device)
        next_states = torch.tensor(self.next_states_memory[batch]).to(self.primary_model.device)
        rewards = torch.tensor(self.rewards_memory[batch]).to(self.primary_model.device)
        terminal =  torch.tensor(self.terminal_memory[batch]).to(self.primary_model.device)

        actions = self.actions_memory[batch]

        q_val = self.primary_model.forward(states)[batch_index, actions]
        next_q_val = self.target_model.forward(next_states)
        next_q_val[terminal] = 0.0

        q_target = rewards + self.gamma * torch.max(next_q_val, dim=1)[0]
        
        loss = self.primary_model.criterion(q_target, q_val).to(self.primary_model.device)
        loss.backward()
        self.primary_model.optimizer.step()