import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gym
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment
from ddqn_pe_agent import DDQNPrioritizedAgent

def get_env_dict(env):
    """create a dictionary with the necessary UnityEnvironment data """
    learn_env = dict()
    learn_env['brain_name'] = brain_name = env.brain_names[0]
    learn_env['brain'] = brain = env.brains[brain_name]
    learn_env['action_size'] = brain.vector_action_space_size # number of actions
    
    return learn_env

def dqn(env, learn_dict, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    brain_name = learn_dict['brain_name']

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            
            action = agent.act(state, eps).astype(int)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=14.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")
    learn_dict = get_env_dict(env)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 32         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    """print('Number of agents:', len(env_info.agents)) # number of agents in the environment

    action_size = brain.vector_action_space_size # number of actions
    print('Number of actions:', action_size)

    state = env_info.vector_observations[0] # examine the state space 
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size) """
    learn_dict['state_size'] = len(env_info.vector_observations[0]) # get state space size

    agent = DDQNPrioritizedAgent(state_size=learn_dict['state_size'], 
                                 action_size=learn_dict['action_size'],
                                 update_every=4,
                                 batch_size=32,
                                 gamma=0.95
                                 )
    scores = dqn(env, learn_dict, n_episodes=1000, max_t=500, eps_start=1.0, eps_end=0.01, eps_decay=0.985)
    env.close()




