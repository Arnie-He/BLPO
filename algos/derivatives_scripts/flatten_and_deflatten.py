from typing import Sequence

import jax
from jax import numpy as jnp
from flax import linen as nn
from jax.tree_util import tree_flatten, tree_unflatten

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

# Initialize model
model = MLP([12, 8, 4])
variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 10)))  # Initialize with dummy input

# Flatten the parameter tree
flat_params, tree_structure = tree_flatten(variables['params'])

print("Flattened parameters shape:", jax.tree_util.tree_map(lambda x: x.shape, flat_params))
