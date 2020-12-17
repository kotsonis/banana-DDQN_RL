import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
from ReplayBuffer import PrioritizedReplayBuffer
from model import dueling_QNetwork

class DDQNPrioritizedAgent():
    """Interacts with and learns from the environment."""
    """Using Double Deep Q Network learning """
    def __init__(self, state_size:int, action_size:int, buffer_size=10000, batch_size=64, gamma=0.99, tau=1e-3, lr=5e-4, update_every=8):
        """Initialize an Agent object.
        Arguments:
        -----
        state_size: dimensions of the environment state space
        action_size: dimensions of the environment action space
        seed: seed to initialize pseudo random number generator
    
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        # check if we can run torch on GPU, else run on CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # beta parameter annealing, using best case from Tom Schaul original paper (Prioritized Experience Replay)

        self.beta = 0.4
        self.beta_step = 0.00025/4.0
        self.beta_max = 1.0
        self.alpha = 0.6
        self.pr_eps = 1e-05

        # Q-Network
        self.qnetwork_local = dueling_QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = dueling_QNetwork(state_size, action_size).to(self.device)
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
                experiences = self.memory.sample(self.batch_size, beta=self.beta) ##TODO: replace with a beta linear scheduler
            
                self.beta = min(self.beta+self.beta_step, self.beta_max)
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
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
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices = experiences
        weights = torch.FloatTensor(weights.reshape(-1, 1)).to(self.device)
    

        # PER: importance sampling before average

        act_t_1_best = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) # a_best = arg max Q(s', a', θi)
        
        q_t_1_best = self.qnetwork_target(next_states).gather(1, act_t_1_best) # Q(s', best_next_action, θ_)
        q_t_1_best_masked = q_t_1_best*(1.0-dones)
        q_t_target = rewards + gamma*q_t_1_best_masked
        q_current_value = self.qnetwork_local(states).gather(1, actions)        #Q(s,a, θi)
        # td_loss = F.smooth_l1_loss(q_current_value, q_t_target, reduction="none") # compute the temporal difference errors
        
        td_absolute_error = torch.abs(q_current_value - q_t_target)     # for the prioritized replay buffer
        
        # loss = torch.mean(weights* F.mse_loss(q_current_value, q_t_target))    # weighted error
        
        loss = torch.mul(weights, F.smooth_l1_loss(q_current_value, q_t_target,reduction='none')).mean()    # weighted error
        # loss = torch.mul(weights, F.mse_loss(q_current_value, q_t_target,reduction='none')).mean()    # weighted error
        #loss = F.smooth_l1_loss(q_current_value, q_t_target, reduction='none')    # weighted error
        # loss = loss * weights
        self.optimizer.zero_grad()
        # calculate back-propagationgradients and update weight matrices in a learning step 
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            new_priorities = td_absolute_error.squeeze().detach().numpy() + self.pr_eps
        self.memory.update_priorities(indices, new_priorities.tolist())
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_i = α * *θ_ + (1 - α)*θ_i

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)