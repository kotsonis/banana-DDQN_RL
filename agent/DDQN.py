import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from buffer.ReplayBuffer import PrioritizedReplayBuffer
from models.model import dueling_QNetwork

class DDQNPrioritizedAgent():
    """Interacts with and learns from the environment."""
    """Using Double Deep Q Network learning """
    def __init__(self, hyper_params):
        """Initialize an Agent object.
        Arguments:
        -----
        state_size: dimensions of the environment state space
        action_size: dimensions of the environment action space
        seed: seed to initialize pseudo random number generator
    
        """
        self.state_size = hyper_params['state_size']
        self.action_size = hyper_params['action_size']
        self.buffer_size = hyper_params['buffer_size']
        
        self.batch_size = hyper_params['batch_size']
        self.gamma = hyper_params['gamma']
        self.tau = hyper_params['tau']
        self.lr = hyper_params['lr']
        self.update_every = hyper_params['update_every']

        # check if we can run torch on GPU, else run on CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # beta parameter annealing, using best case from Tom Schaul original paper (Prioritized Experience Replay)

        self.beta = hyper_params['beta']
        self.beta_step = hyper_params['beta_step']
        self.beta_max = hyper_params['beta_max']
        self.alpha = hyper_params['alpha']
        self.pr_eps = hyper_params['pr_eps']

        # Q-Network
        self.qnetwork_local = dueling_QNetwork(self.state_size, self.action_size).to(self.device)
        self.qnetwork_target = dueling_QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(size=self.buffer_size, batch_size=self.batch_size, alpha=self.alpha)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size, beta=self.beta)
                # decay beta for prioritized experienced learning
                self.beta = min(self.beta+self.beta_step, self.beta_max)
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns ε-greedy actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # put NN into eval mode and get best actions without adding to computation graph
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples."""

        states, actions, rewards, next_states, dones, weights, indices = experiences
        
    
        # PER: importance sampling before average
        act_t_1_best = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) # arg max Q(s', a', θi)
        q_t_1_best = self.qnetwork_target(next_states).gather(1, act_t_1_best)          # Q(s', best_t+1_action, θ_)
        q_t_1_best_masked = q_t_1_best*(1.0-dones)                                      # mask out terminal states
        q_t_target = rewards + gamma*q_t_1_best_masked                                  # Q(s,a, θ_) = r + γ*Q(s',a', θ_)
        q_current_value = self.qnetwork_local(states).gather(1, actions)                # Q(s,a, θi)
        
        td_absolute_error = torch.abs(q_current_value - q_t_target)     # for the prioritized replay buffer
        
        loss = torch.mul(weights, F.smooth_l1_loss(q_current_value, q_t_target,reduction='none')).mean()    # weighted loss
        
        self.optimizer.zero_grad()
        # calculate back-propagationgradients and update weight matrices in a learning step 
        loss.backward()
        self.optimizer.step()
        
        # PER: update priorities in the replay buffer
        with torch.no_grad():
            new_priorities = td_absolute_error.squeeze().detach().numpy() + self.pr_eps
        self.memory.update_priorities(indices, new_priorities.tolist())
        
        # Double QN : update Q_local with Q_target θ_i = τ *θ_ + (1 - τ)*θ_i 
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)