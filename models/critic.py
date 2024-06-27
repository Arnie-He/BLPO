from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence

class Critic(nn.Module):
    """
    A value network that outputs a single numerical value. ReLU activation is applied
    between each hidden layer.
    """
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, input):
        out = input.ravel()
        for layer in self.hidden_sizes:
            out = nn.Dense(layer)(out)
            out = nn.relu(out)
        out = nn.Dense(1)(out)
        return jnp.squeeze(out)