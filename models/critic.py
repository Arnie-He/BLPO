from distrax import Categorical
from flax import linen as nn
from typing import Any, Sequence
from flax.linen.initializers import constant, orthogonal
import numpy as np
import jax.numpy as jnp

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
            out = nn.Dense(layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(out)
            out = nn.tanh(out)
        out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(out)
        return jnp.squeeze(out, -1)

class PixelCritic(nn.Module):
    """
    A policy network with 2 hidden layers that outputs logits for each action. The logits
    are wrapped in a categorical distribution that is returned from each call.
    """
    conv_fns: Sequence[Any]
    dense_sizes: Sequence[int]

    @nn.compact
    def __call__(self, input):
        out = input
        for fn in self.conv_fns:
            out = fn(out)

        out = out.ravel()
        for size in self.dense_sizes:
            out = nn.Dense(size)(out)
            out = nn.relu(out)

        out = nn.Dense(1)(out)
        return jnp.squeeze(out)