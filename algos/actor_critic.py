from env import ENV, ENV_KEY, SEED

from distrax import Categorical
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import functools
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from typing import Sequence

# Configuration parameters

params = {
    "cartpole": {
        "actor_sizes": (30, 15),
        "critic_sizes": (30, 15),
        "num_updates": 500,
        "batch_count": 25,
        "rollout_len": 2000,
        "discount_rate": 0.99,
        "learning_rate": 0.002,
    },
}

NUM_UPDATES = params[ENV_KEY]["num_updates"]
BATCH_COUNT = params[ENV_KEY]["batch_count"]
ROLLOUT_LEN = params[ENV_KEY]["rollout_len"]
DISCOUNT_RATE = params[ENV_KEY]["discount_rate"]
LEARNING_RATE = params[ENV_KEY]["learning_rate"]
ADAM_EPS = 1e-5

class ActorCritic(nn.Module):
    """
    A policy and value network with 2 hidden layers each. The policy network outputs logits
    for each action that are wrapped in a categorical distribution. The critic outputs a
    numerical value for the value of the state.
    """
    actor_sizes: Sequence[int]
    critic_sizes: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, input):
        actor_out = input
        for layer in self.actor_sizes:
            actor_out = nn.Dense(layer)(actor_out)
            actor_out = nn.relu(actor_out)
        actor_out = nn.Dense(self.num_actions)(actor_out)

        critic_out = input
        for layer in self.critic_sizes:
            critic_out = nn.Dense(layer)(critic_out)
            critic_out = nn.relu(critic_out)
        critic_out = nn.Dense(1)(critic_out)

        return Categorical(actor_out), critic_out