import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_durations(avg1, avg2, avg3, sd1, sd2, sd3):
    _,ax = plt.subplots(figsize=(10, 8))
    ax.plot(avg1, color='black', linewidth=2, label='DQN')
    ax.plot(avg2, color='green', linewidth=2, label='DDQN')
    ax.plot(avg3, color='red', linewidth=2, label='Random')
    ax.fill_between(range(len(avg1)), np.subtract(avg1, sd1), np.add(avg1, sd1), color='gray')
    ax.fill_between(range(len(avg2)), np.subtract(avg2, sd2), np.add(avg2, sd2), color='olive')
    ax.fill_between(range(len(avg3)), np.subtract(avg3, sd3), np.add(avg3, sd3), color='pink')
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

def load(agent_type):
    if agent_type == 'dqn':
        with open('dqn.npy', 'rb') as f:
            dur = np.load(f)
    elif agent_type == 'ddqn':
        with open('ddqn.npy', 'rb') as f:
            dur = np.load(f)
    elif agent_type == 'random':
        with open('random.npy', 'rb') as f:
            dur = np.load(f)
    elif agent_type == 'epsilons':
        with open('epsilons.npy', 'rb') as f:
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
    
    epsilons = load('epsilons')
    plot_epsilons(epsilons)
    
    dur1 = load('dqn')
    dur2 = load('ddqn')
    dur3 = load('random')
    window_size = int(len(dur1)/10)
    avg1, sd1 = moving_average(dur1, window_size)
    avg2, sd2 = moving_average(dur2, window_size)
    avg3, sd3 = moving_average(dur3, window_size)
    plot_durations(avg1, avg2, avg3, sd1, sd2, sd3)
    
    