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
        #self.seed = torch.manual_seed()
        self.hidden_fc = nn.Linear(state_size, 64)
        self.value_fc_hidden = nn.Linear(64,64)
        self.value_fc = nn.Linear(64, 1)
        self.advantage_fc_hidden1 = nn.Linear(64,64)
        self.advantage_fc_hidden2 = nn.Linear(64,64)
        self.advantage_fc = nn.Linear(64, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.hidden_fc(state))
        value = F.relu(self.value_fc_hidden(x))
        value = self.value_fc(value)
        advantage = F.relu(self.advantage_fc_hidden1(x))
        advantage = F.relu(self.advantage_fc_hidden2(advantage))
        advantage = self.advantage_fc(advantage)
        expanded_value = value.expand_as(advantage)
        max_advantage = advantage.max(1)[0].unsqueeze(-1)
        delta_advantage = advantage - max_advantage
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
