from distrax import Categorical
from flax import linen as nn
from typing import Any, Sequence
from flax.linen.initializers import constant, orthogonal
import numpy as np

class DiscreteActor(nn.Module):
    """
    A discrete action policy network that outputs logits for each action. Between each layer
    ReLU activation is applied to the hidden outputs. The final logits are wrapped in a
    categorical distribution that is returned from each call.
    """
    hidden_sizes: Sequence[int]
    num_actions: int

    ### Use orthogonal for kernel initializaiton, constant for bias initialization, and tanh for activation.
    @nn.compact
    def __call__(self, input):
        out = input.ravel()
        for layer in self.hidden_sizes:
            out = nn.Dense(layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(out)
            out = nn.tanh(out)
        out = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(out)
        return Categorical(out)

class DiscretePixelActor(nn.Module):
    """
    A convolutional policy network that outputs logits for each action. Between each layer
    ReLU activation is applied to the hidden outputs. The final logits are wrapped in a
    categorical distribution that is returned from each call.
    """
    conv_fns: Sequence[Any]
    dense_sizes: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, input):
        out = input
        for fn in self.conv_fns:
            out = fn(out)

        out = out.ravel()
        for size in self.dense_sizes:
            out = nn.Dense(size)(out)
            out = nn.relu(out)

        out = nn.Dense(self.num_actions)(out)
        return Categorical(out)

class MLP(nn.Module):
    hiddensizes: Sequence[int]
    outputsize: int

    @nn.compact
    def __call__(self, input):
        out = input.ravel()
        for layer in self.hiddensizes:
            out = nn.Dense(layer)(out)
            out = nn.relu(out)
        out = nn.Dense(self.outputsize)(out)
        return out

class CNN(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, input):
        # Define convolutional layers
        out = nn.Conv(features=8, kernel_size=(3, 3), strides=(1, 1))(input)
        out = nn.relu(out)
        out = nn.max_pool(out, window_shape=(2, 2), strides=(2, 2))

        out = nn.Conv(features=16, kernel_size=(3, 3), strides=(1, 1))(out)
        out = nn.relu(out)
        out = nn.max_pool(out, window_shape=(2, 2), strides=(2, 2))

        # Flatten the output
        out = out.ravel()

        # Fully connected layer to map to the desired output size
        out = nn.Dense(self.output_size)(out)
        return out