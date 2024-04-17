import torch
import torch.nn as nn
import gymnasium as gym

from train import *
from agents.dqn import *
from agents.ddqn import *
from agents.random import *

def testing(env, agent, n_episodes = 10, n_iterations = 500):
    steps = 0
    episode_durations = np.array([])
    for e in range(n_episodes):
        current_state, info = env.reset(seed=42)

        for i in range(n_iterations):
            steps += 1
            action = agent.select_action(current_state) 
            next_state, reward, terminated, truncated, info = env.step(action) 
            done = terminated or truncated
            
            current_state = next_state

            if done:
                episode_durations = np.append(episode_durations, i + 1)
                break

    env.close()
    return episode_durations


def load(env, agent_type):
    if agent_type == 'dqn':
        agent = create_agent('dqn', env)
        agent.primary_model.load_state_dict(torch.load('runs/dqn_model3.pt'))
    else:
        agent = create_agent('ddqn', env)
        agent.primary_model.load_state_dict(torch.load('runs/ddqn_model3.pt'))

    return agent

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode = 'human')
    dqn= load(env, 'dqn')
    dur1 = testing(env, dqn)

    env = gym.make('CartPole-v1', render_mode = 'human')
    ddqn= load(env, 'ddqn')
    dur2 = testing(env, ddqn)

    print(dur1)
    print(dur2)