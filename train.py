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

std_learn_params = {
        # Unity Environment parameters
        "banana_location": "./Banana_Windows_x86_64/Banana.exe",
        # MDP learning parameters
        "n_episodes": 2000,  # maximum episodes to train on
        "max_t":1000,       # maximum scenes in an episodic training
        "eps_start":1.0,    # starting exploration factor
        "eps_end":0.01,     # ending exploration factor
        "eps_decay":0.98,  # eps step decay
        
        # Q value learning parameters
        "gamma": 0.99,      # discount factor
        "tau": 5e-4,        # for soft update of target parameters
        "lr": 5e-4,         # learning rate 
        "update_every": 4,  # how often to update the network
        
        # Replay Buffer / Prioritized Replay Buffer parameters
        "buffer_size": 1e5, # replay buffer size
        "batch_size": 64,   # minibatch size
        "alpha": 0.6,       # prioritization factor (0: No prioritization .. 1: Full prioritization)
        "pr_eps": 1e-05,    # minimum prioritization
        "beta":0.4,                 # Importance sampling beta factor start
        "beta_step": 0.00025/4.0,   # beta decay factor
        "beta_max": 1.0             # maximum beta
    }

def get_env_dict(env):
    """create a dictionary with the necessary UnityEnvironment data """
    
    
    return learn_env

def dqn(env, learn_dict, log_results=True):
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
    eps = learn_dict['eps_start']                    # initialize epsilon
    brain_name = learn_dict['brain_name']
    n_episodes=learn_dict['n_episodes']
    max_t= learn_dict['max_t']
    eps_start= learn_dict['eps_start']
    eps_end= learn_dict['eps_end']
    eps_decay= learn_dict['eps_decay']

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
        if log_results: print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            if log_results: print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=50.0:
            if log_results: print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

def m_a_scores(window_size, raw_scores):
    ma_scores = []
    ma_scores.append(0.)
    avg_window = window_size
    beta = 1.0 - (1./avg_window)
    for i in range(1, len(raw_scores)):
        ma_scores.append(
            beta*ma_scores[i-1]+(1-beta)*raw_scores[i])
    return ma_scores
def generate_label(params):
    label = 'DDQN_' \
            +'bs_' + str(learn_env['batch_size']) \
            +'g_' + str(learn_env['gamma']) \
            +'e_'+str(learn_env['eps_decay']) \
            +'t_'+str(learn_env['tau']) \
            +'lr_' + str(learn_env['lr']) \
            +'a_' + str(learn_env['alpha']) \
            +'b_' + str(learn_env['beta'])
    return label

def parse_params(argv):
    try:
        opts, args = getopt.getopt(argv,
                                   ":bh",
                                   [
                                       "eps-start=",
                                       "eps-decay=",
                                       "batch-size=",
                                       "episodes=",
                                       "reward-early-stop=",
                                       "output-param-file=",
                                       "output-image=",
                                       "gamma=",
                                       "beta_start=",
                                       "beta-decay=",
                                       "alpha=",
                                       "banana_location="
                                    ])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    hyperparams = std_learn_params

    for o, a in opts:
        if o == "eps-start=":
            hyperparams['eps_start'] = a
        elif o == "eps-decay=":
            hyperparams['eps_decay'] = a
        elif o == "batch-size=":
            hyperparams['batch_size'] = a
        elif o == "episodes=":
            hyperparams['num_episodes']=a
        elif o == "reward-early-stop=":
            hyperparams['early_stop'] = a
        elif o == "output-param-file=":
            hyperparams['model_file'] = a
        elif o == "output-image=":
            hyperparams['plt_file'] = a
        elif o == "gamma=":
            hyperparams['gamma'] = a
        elif o == "beta-start=":
            hyperparams['beta']=a
        elif o == "beta-decay=":
            hyperparams['beta_decay']=a
        elif o == "alpha=":
            hyperparams['alpha'] = a
        elif o in ("-b", "banana_location="):
            hyperparams['banana_location']=a
        elif o == "h":
            usage()
        else:
            assert False, "unhandled option"
    # return the modified hyperparams
    return hyperparams  

def main(argv):
    learn_env = parse_params(argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=learn_env['banana_location'])
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # set our learning enviroment hyperparameters
    learn_env = {
        # Unity Environment
        "brain_name" : brain_name,
        "brain" : brain,
        "action_size":  brain.vector_action_space_size,
        "state_size": len(env_info.vector_observations[0]),
    }

    i = 0
    ma_scores = []
    labels = []
    agent = DDQNPrioritizedAgent(learn_env)
    scores = dqn(env, learn_env)
    labels.append(generate_label(learn_env))
    ma_scores.m_a_scores(10,scores)

    """learn_env['batch_size'] = batch_size
    print("Starting: ", generate_label(learn_env))
    agent = DDQNPrioritizedAgent(learn_env)
    scores = dqn(env, learn_env)
    labels.append(generate_label(learn_env))
    ma_scores.append(m_a_scores(10,scores)) """

    #plot results
    plt.figure(figsize=(11.75,8.25))
    plt.title('Mean Score')
    plt.xlabel("Episodes")
    plt.ylabel("Mean Score")
    episodes=range(learn_env['n_episodes'])
    for j in range(len(ma_scores)):
        plt.plot(episodes, ma_scores[j], label=labels[j])
    
    plt.legend(loc='lower right')
    f_name = labels[0] + '.png'
    plt.savefig(f_name)
    plt.show()
    
    env.close()
if __name__ == "__main__":
   main(sys.argv[1:])

if __name__ == "__main__":





