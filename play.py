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
from models.model import dueling_QNetwork

std_play_params = {
        # Unity Environment parameters
        "banana_location": "./Banana_Windows_x86_64/Banana.exe",
        # MDP learning parameters
        "n_episodes": 10,   # maximum episodes to play
        "model": "model.pt" # torch parameters to load
    }
def moving_average(x, w):
    vals = np.array(x)
    return np.convolve(vals, np.ones(w), 'valid') / w


def usage():
    print(
        "player.py - play the Banana environment with trained agent\n",
        "optional parameters: \n",
        "--------------------- Unity Environment -------------------\n",
        "-- banana_location <location of banana environment>\n",
        "\n",
        "---------------------- Play parameters --====--------------\n",
        "--episodes <number of episodes to train on>                default: 10\n",
        "--model <torch parameters file to load>                    default: model.pt\n",
    )

def parse_params(argv):
    try:
        opts, args = getopt.getopt(argv,
                                   ":bme",
                                   [
                                       "episodes=",
                                       "model=",
                                       "banana_location=",
                                       "help"
                                    ])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    hyperparams = std_play_params

    for o, a in opts:
        if o in ("-m", "--model"):
            hyperparams['model'] = a
        elif o in ("-e", "--episodes"):
            hyperparams['n_episodes']= int(a)
        elif o in ("-b", "--banana_location"):
            hyperparams['banana_location']=a
        elif o == "--help":
            usage()
            sys.exit(2)
        else:
            assert False, "unhandled option"
    # return the modified hyperparams
    return hyperparams  
def act(q_network, state, device):
        """Returns ε-greedy actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        action_values = q_network(state)
        return np.argmax(action_values.cpu().data.numpy())
        
def main(argv):
    play_env = parse_params(argv)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = UnityEnvironment(file_name=play_env['banana_location'])
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    Q_model = dueling_QNetwork(state_size, action_size)
    Q_model.load_state_dict(torch.load(f=play_env.get('model')))
    # put NN into eval mode 
    Q_model.eval()
    # update hyperparameters with Unity Environment
    
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=10)  # last 100 scores
    n_episodes=play_env['n_episodes']
    

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        done = done = env_info.local_done[0]
        steps = 0
        while not done:
            action = act(Q_model, state, device).astype(int)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            # have the agent learn a step
            state = next_state
            score += reward
            steps += 1
            print('\rEpisode {}\t Score: {:2.0f} \t Step: {:2}/300'.format(i_episode, score, steps), end="")
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        print('\nEpisode {}\t Score: {:.2f} \t Steps: {}'.format(i_episode, score, steps))
    print('Your average score over {} episodes was {}\n'.format(n_episodes, np.mean(scores_window)))
    
    env.close()

if __name__ == "__main__":
   main(sys.argv[1:])





