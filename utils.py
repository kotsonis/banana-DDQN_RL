import sys, getopt # for commandline argument parsing
import numpy as np
from hyperparams import std_learn_params

def usage():
    """ print out command line usage """
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
    """ parse command line parameters and return a dictionary with parameters """
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
                                       "beta-start=",
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

def moving_average(x, w):
    """ return array with moving average of x on a window size w """
    vals = np.array(x)
    return np.convolve(vals, np.ones(w), 'valid') / w

def ewma(window_size, raw_scores):
    """ returns an array with exponentialy weighted moving average"""
    ma_scores = []
    ma_scores.append(0.)
    avg_window = window_size
    beta = 1.0 - (1./avg_window)
    for i in range(1, len(raw_scores)):
        ma_scores.append(
            beta*ma_scores[i-1]+(1-beta)*raw_scores[i])
    return np.array(ma_scores)
