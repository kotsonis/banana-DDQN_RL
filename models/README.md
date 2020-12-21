# Dueling Q Network function approximator
## interface
The Q function approximator is performed via a deep neural network and implemented in a `Torch.nn.Module` subclass that has only two methods:

```python
class dueling_QNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size):
        ...
    def forward(self, state):
```
## model
