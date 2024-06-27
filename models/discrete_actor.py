from distrax import Categorical
from flax import linen as nn
from typing import Sequence

class DiscreteActor(nn.Module):
    """
    A discrete action policy network that outputs logits for each action. Between each layer
    ReLU activation is applied to the hidden outputs. The final logits are wrapped in a
    categorical distribution that is returned from each call.
    """
    hidden_sizes: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, input):
        out = input.ravel()
        for layer in self.hidden_sizes:
            out = nn.Dense(layer)(out)
            out = nn.relu(out)
        out = nn.Dense(self.num_actions)(out)
        return Categorical(out)