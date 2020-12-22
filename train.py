import sys, getopt # for commandline argument parsing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gym
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from collections import deque
from unityagents import UnityEnvironment
from agent.DDQN import DDQNPrioritizedAgent
from hyperparams import std_learn_params
from utils import parse_params, moving_average

def train_dqn(env, learn_dict, agent, log_results=True):
    """Deep Q-Learning.
    
    Params
    ======
        env (object):       the unity environment
        learn_dict (dict):  learning and environment hyperparameters
        agent (object):     starting value of epsilon, for epsilon-greedy action selection
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = learn_dict['eps_start']
    brain_name = learn_dict['brain_name']
    n_episodes=learn_dict['n_episodes']
    max_t= learn_dict['max_t']
    eps_start= learn_dict['eps_start']
    eps_end= learn_dict['eps_end']
    eps_decay= learn_dict['eps_decay']
    early_stop = learn_dict['early_stop']

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
            # have the agent learn a step
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        if log_results: print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            if log_results: print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=early_stop:
            if log_results: print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    return scores

def main(argv):
    learn_env = parse_params(argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=learn_env['banana_location'])
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # update hyperparameters with Unity Environment
    learn_env["brain_name"] = brain_name
    learn_env["brain"] = brain
    learn_env["action_size"] = brain.vector_action_space_size
    learn_env["state_size"] = len(env_info.vector_observations[0])
    
    agent = DDQNPrioritizedAgent(learn_env)
    scores = train_dqn(env, learn_env, agent)
    label = "10 episode moving average score"
    ma_window_size = 10
    ma_scores = moving_average(scores,ma_window_size)
    
    torch_model_fname = learn_env.get('model_file')
    if(torch_model_fname):
        torch.save(agent.qnetwork_local.state_dict(), torch_model_fname)
    

    #plot results
    plt.figure()
    plt.title('Mean Score')
    plt.xlabel("Episodes")
    plt.ylabel("Mean Score")
    episodes=range(len(ma_scores))
    plt.plot(episodes, ma_scores, label=label)
    
    plt.legend(loc='lower right')
    f_name = learn_env.get('plt_file')
    if (f_name):
        plt.savefig(f_name)
        plt.show()
    else:
        plt.show()
    env.close()

if __name__ == "__main__":
   main(sys.argv[1:])





