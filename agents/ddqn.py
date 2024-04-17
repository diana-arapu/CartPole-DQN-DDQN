import torch
import random
from collections import deque, namedtuple
import torch.nn as nn
import math
from agents.dqn import *

class DDQNAgent(DQNAgent):

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

        self.primary_model.eval()
        self.target_model.eval()

        #q1 = self.primary_model.forward(states)[batch_index, actions]

        q2 = self.primary_model.forward(next_states)
        q_ = self.target_model.forward(next_states)
        q_[terminal] = 0.0
        temp = torch.argmax(q2, dim=1)
        q_ = q_[batch_index, temp]
        q_target = rewards + self.gamma*q_
       
        self.primary_model.train()
        q1 = self.primary_model.forward(states)[batch_index, actions]
        loss = self.primary_model.criterion(q_target, q1).to(self.primary_model.device)
        loss.backward()
        self.primary_model.optimizer.step()
        
        return loss.item()