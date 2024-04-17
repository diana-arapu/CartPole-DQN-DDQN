import gymnasium as gym
import torch
from agents.dqn import *
from agents.ddqn import *
from agents.random import *

def create_agent(agent_type, env):
    if agent_type == 'dqn':
        return DQNAgent(env)
    if agent_type == 'random': 
        return RandomAgent(env)
    if agent_type == 'ddqn':
        return DDQNAgent(env)

def save_intermediate(agent, e, saving, exp):
    if isinstance(agent, DQNAgent) and e % saving == 0:
            torch.save(agent.primary_model.state_dict(), 'runs/dqn_model{}{}.pt'.format(exp, e))

    if isinstance(agent, DDQNAgent) and e % saving == 0:
            torch.save(agent.primary_model.state_dict(), 'runs/ddqn_model{}{}.pt'.format(exp, e))

def save_model(agent, exp):
    if isinstance(agent, DQNAgent):
        torch.save(agent.primary_model.state_dict(), 'runs/dqn_model{}.pt'.format(exp))
    if isinstance(agent, DDQNAgent):
        torch.save(agent.primary_model.state_dict(), 'runs/ddqn_model{}.pt'.format(exp))

def env_interaction(env, agent, exp, n_episodes, n_iterations=500, batch_size=64):
    steps = 0
    episode_durations = np.array([])
    loss = np.array([])
    epsilons = np.array([], np.float32)
    saving = 1000
    

    for e in range(n_episodes):
        current_state, info = env.reset(seed=42)
        rewards = 0

        for i in range(n_iterations):
            steps += 1
            action = agent.select_action(current_state) 
            next_state, reward, terminated, truncated, info = env.step(action) 
            done = terminated or truncated
            agent.store_transition(current_state, action, next_state, reward, done)
            
            if steps >= batch_size and not isinstance(agent, RandomAgent):
                loss = np.append(loss, agent.train(batch_size))
                agent.hard_update(i)
            
            current_state = next_state

            if done:
                episode_durations = np.append(episode_durations, i + 1)
                break

        #save_intermediate(agent, e, saving, exp)
        
        if not isinstance(agent, RandomAgent):
            epsilons = np.append(epsilons, agent.epsilon)            
            agent.epsilon *= agent.epsilon_decay

    env.close()
    save_model(agent, exp)
    return episode_durations, epsilons, loss

def save_durations(dur, agent_type, exp):
    if agent_type == 'dqn':
        with open('runs/dqn{}.npy'.format(exp), 'wb') as f:
            np.save(f, dur)
    if agent_type == 'ddqn':
        with open('runs/ddqn{}.npy'.format(exp), 'wb') as f:
            np.save(f, dur)
    if agent_type == 'random':
        with open('runs/random{}.npy'.format(exp), 'wb') as f:
            np.save(f, dur)

def save_epsilons(epsilons):
    with open('runs/epsilons.npy', 'wb') as f:
        np.save(f, epsilons)

def save_loss(loss, agent_type, exp):
    if agent_type == 'dqn':
        with open('runs/dqn_loss{}.npy'.format(exp), 'wb') as f:
            np.save(f, loss)
    if agent_type == 'ddqn':
        with open('runs/ddqn_loss{}.npy'.format(exp), 'wb') as f:
            np.save(f, loss)


def train(env_str, agent_type, n_experiments, n_episodes, n_iterations, batch_size):
    for exp in range(n_experiments):
        env = gym.make(env_str)
        agent = create_agent(agent_type, env)
        episode_duration, epsilons, loss = env_interaction(env, agent, exp, n_episodes, n_iterations, batch_size)
        save_durations(episode_duration, agent_type, exp)
        if agent_type != 'random':
            save_loss(loss, agent_type, exp)
    if agent_type == 'dqn':
        save_epsilons(epsilons)
    


if __name__ == "__main__":
    n_experiments = 10
    n_episodes = 1000
    n_iterations = 500
    batch_size = 32
    train('CartPole-v1', 'dqn', n_experiments, n_episodes, n_iterations, batch_size)
    train('CartPole-v1', 'ddqn', n_experiments, n_episodes, n_iterations, batch_size)
    train('CartPole-v1', 'random', n_experiments, n_episodes, n_iterations, batch_size)
    
