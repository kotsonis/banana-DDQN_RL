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
        "n_episodes": 2000, # maximum episodes to train on
        "max_t":1000,       # maximum scenes in an episodic training
        "eps_start":1.0,    # starting exploration factor
        "eps_end":0.01,     # ending exploration factor
        "eps_decay":0.98,   # eps step decay
        'early_stop': 13,   # early stop if average reward in 100 episode reaches this value
        
        # Q value learning parameters
        "gamma": 0.99,      # discount factor
        "tau": 5e-4,        # for soft update of target parameters
        "lr": 5e-4,         # learning rate 
        "update_every": 4,  # how often to update the network
        
        # Replay Buffer / Prioritized Replay Buffer parameters
        "buffer_size": 1e5,         # replay buffer size
        "batch_size": 64,           # minibatch size
        "alpha": 0.6,               # prioritization factor (0: No prioritization .. 1: Full prioritization)
        "pr_eps": 1e-05,            # minimum prioritization
        "beta":0.4,                 # Importance sampling beta factor start
        "beta_step": 0.00025/4.0,   # beta decay factor
        "beta_max": 1.0             # maximum beta
    }
def moving_average(x, w):
    vals = np.array(x)
    return np.convolve(vals, np.ones(w), 'valid') / w

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

def m_a_scores(window_size, raw_scores):
    ma_scores = []
    ma_scores.append(0.)
    avg_window = window_size
    beta = 1.0 - (1./avg_window)
    for i in range(1, len(raw_scores)):
        ma_scores.append(
            beta*ma_scores[i-1]+(1-beta)*raw_scores[i])
    return ma_scores

def usage():
    print(
        "trainer.py - train the agent to solve the Banana environment\n",
        "optional parameters: \n",
        "--------------------- Unity Environment -------------------\n",
        "-- banana_location <location of banana environment>\n",
        "\n",
        "---------------- Agent training parameters ----------------\n",
        "--eps-start <starting value of exploration>                default: 1.0\n",
        "--eps-decay <eps linear decay rate>                        default: 0.98\n",
        "--episodes <number of episodes to train on>                default: 2000\n",
        "--reward-early-stop <average reward/100 episodes>          default: 13\n",
        "\n",
        "-------------- Q Network learning parameters ---------------\n",
        "--tau <Q network soft update factor>                       defalut: 5e-4\n"
        "--gamma <factor to diminish future rewards>                default: 0.99\n",
        "\n",
        "--------- Prioritized Experience Replay parameters --------\n",
        "--batch-size <mini batch size>                             default: 64\n",
        "--beta_start <beta factor for IS weights>                  default: 0.4\n",
        "--beta-decay <linear decay factor>                         default: 6.25e-6\n",
        "--alpha <level of prioritization>                          default: 0.6\n",
        "\n",
        "--------------- storing outputs parameters ---------------\n",
        "--output-param-file <fname to save Q_function model>       default: none\n",
        "--output-image <png fname to save training scores plot>    default: none\n",
    )

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
                                       "tau=",
                                       "beta_start=",
                                       "beta-decay=",
                                       "alpha=",
                                       "banana_location=",
                                       "help"
                                    ])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    hyperparams = std_learn_params

    for o, a in opts:
        if o == "--eps-start":
            hyperparams['eps_start'] = float(a)
        elif o == "--eps-decay":
            hyperparams['eps_decay'] = float(a)
        elif o == "--batch-size":
            hyperparams['batch_size'] = int(a)
        elif o == "--episodes":
            hyperparams['n_episodes']= int(a)
        elif o == "--reward-early-stop":
            hyperparams['early_stop'] = float(a)
        elif o == "--tau":
            hyperparams['tau'] = float(a)
        elif o == "--output-param-file":
            hyperparams['model_file'] = a
        elif o == "--output-image":
            hyperparams['plt_file'] = a
        elif o == "--gamma":
            hyperparams['gamma'] = float(a)
        elif o == "--beta-start":
            hyperparams['beta']=float(a)
        elif o == "--beta-decay":
            hyperparams['beta_decay']=float(a)
        elif o == "--alpha":
            hyperparams['alpha'] = float(a)
        elif o in ("-b", "--banana_location"):
            hyperparams['banana_location']=a
        elif o == "--help":
            usage()
            sys.exit(2)
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





