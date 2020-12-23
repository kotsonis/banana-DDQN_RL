import torch
import torch.nn as nn
import torch.nn.functional as F

class dueling_QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(dueling_QNetwork, self).__init__()
        self.hidden_fc1 = nn.Linear(state_size, 64)
        self.hidden_fc2= nn.Linear(64,64)
        self.value_fc = nn.Linear(64, 1)
        self.advantage_fc = nn.Linear(64, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.hidden_fc1(state))
        x = F.relu(self.hidden_fc2(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        delta_advantage = advantage - advantage.mean()
        expanded_value = value.expand_as(advantage)
        q_out = expanded_value - delta_advantage
        return q_out
    
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.hidden = nn.Linear(state_size, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.hidden(state))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x
