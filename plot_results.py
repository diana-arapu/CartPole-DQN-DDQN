import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def compute_average(returns):
    avg = np.zeros(len(returns[0]))
    sd = np.zeros(len(returns[0]))
    for ret in returns:
        avg += ret
    avg /= len(returns)
    for ret in returns:
        for j in range(len(ret)):
            sd[j] += (ret[j] - avg[j]) ** 2
    sd /= len(returns) - 1
    sd = np.sqrt(sd)
    return avg, sd

def plot_durations(avg1, avg2, avg3, sd1, sd2, sd3):
    _,ax = plt.subplots(figsize=(10, 8))
    ax.plot(avg1, linewidth=2, label='DQN')
    ax.plot(avg2, linewidth=2, label='DDQN')
    ax.plot(avg3, linewidth=2, label='Random')
    ax.fill_between(range(len(avg1)), np.subtract(avg1, sd1), np.add(avg1, sd1), alpha=0.2)
    ax.fill_between(range(len(avg2)), np.subtract(avg2, sd2), np.add(avg2, sd2), alpha=0.2)
    ax.fill_between(range(len(avg3)), np.subtract(avg3, sd3), np.add(avg3, sd3), alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Episode durations')
    plt.legend()
    plt.show()

def moving_average(data, window_size):
    moving_avg = np.zeros(len(data) - window_size + 1)
    moving_sq = np.zeros(len(data) - window_size + 1)
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        avg = sum(window) / window_size
        sq_avg = sum(x**2 for x in window) / window_size
        moving_avg[i] = avg
        moving_sq[i] = sq_avg
    
    moving_variance = [sq_avg - avg**2 for avg, sq_avg in zip(moving_avg, moving_sq)]
    moving_std = [np.sqrt(variance) for variance in moving_variance]
    return moving_avg, moving_std

def load(agent_type, dur):
    if agent_type == 'dqn':
        with open('runs/dqn0.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn1.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn2.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn3.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn4.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn5.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn6.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn7.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn8.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/dqn9.npy', 'rb') as f:
            dur.append(np.load(f))
    elif agent_type == 'ddqn':
        with open('runs/ddqn0.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn1.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn2.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn3.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn4.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn5.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn6.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn7.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn8.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/ddqn9.npy', 'rb') as f:
            dur.append(np.load(f))
    elif agent_type == 'random':
        with open('runs/random0.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random1.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random2.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random3.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random4.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random5.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random6.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random7.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random8.npy', 'rb') as f:
            dur.append(np.load(f))
        with open('runs/random9.npy', 'rb') as f:
            dur.append(np.load(f))
    elif agent_type == 'epsilons':
        with open('runs/epsilons.npy', 'rb') as f:
            dur = np.load(f)

    return dur

def plot_epsilons(epsilons):
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilons')
    plt.show()

def results(agent_types):
    avgs = np.zeros(len(agent_types))
    sds = np.zeros(len(agent_types))
    for i, agent in enumerate(agent_types):
        dur = load(agent)
        avg, sd = moving_average(dur, int(len(dur)/10))
        avgs[i] = avg
        sds[i] = sd
    return avgs, sds

if __name__ == "__main__":
    #agent_types = ['dqn', 'ddqn', 'random']
    
    #epsilons = load('epsilons')
    #plot_epsilons(epsilons)
    
    dur1 = load('dqn', [])
    dur2 = load('ddqn', [])
    dur3 = load('random', [])
    
    avg1, sd1 = compute_average(dur1)
    avg2, sd2 = compute_average(dur2)
    avg3, sd3 = compute_average(dur3)

    plot_durations(avg1, avg2, avg3, sd1, sd2, sd3)
    
    