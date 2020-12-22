std_learn_params = {
        # Unity Environment parameters
        "banana_location": "./Banana_Windows_x86_64/Banana.exe",
        # MDP learning parameters
        "n_episodes": 2000, # maximum episodes to train on
        "max_t":1000,       # maximum scenes in an episodic training
        "eps_start":0.975,    # starting exploration factor
        "eps_end":0.05,     # ending exploration factor
        "eps_decay":0.99,   # eps step decay
        'early_stop': 13,   # early stop if average reward in 100 episode reaches this value
        
        # Q value learning parameters
        "gamma": 1,      # discount factor
        "tau": 1e-3,        # for soft update of target parameters
        "lr": 5e-4,         # learning rate 
        "update_every": 4,  # how often to update the network
        
        # Replay Buffer / Prioritized Replay Buffer parameters
        "buffer_size": 1e5,         # replay buffer size
        "batch_size": 32,           # minibatch size
        "alpha": 0.8,               # prioritization factor (0: No prioritization .. 1: Full prioritization)
        "pr_eps": 1e-05,            # minimum prioritization
        "beta":0.4,                 # Importance sampling beta factor start
        "beta_step": 0.00025/4.0,   # beta decay factor
        "beta_max": 1.0             # maximum beta
    }