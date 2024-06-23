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
        "actor_learning_rate": 0.002,
        "critic_learning_rate": 0.002,
    },
}

NUM_UPDATES = params[ENV_KEY]["num_updates"]
BATCH_COUNT = params[ENV_KEY]["batch_count"]
ROLLOUT_LEN = params[ENV_KEY]["rollout_len"]
DISCOUNT_RATE = params[ENV_KEY]["discount_rate"]
ACTOR_RATE = params[ENV_KEY]["actor_learning_rate"]
CRITIC_RATE = params[ENV_KEY]["critic_learning_rate"]
ADAM_EPS = 1e-5

class Actor(nn.Module):
    """
    A policy network with 2 hidden layers that outputs logits for each action. The logits
    are wrapped in a categorical distribution that is returned from each call.
    """
    hidden_sizes: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, input):
        out = input
        for layer in self.hidden_sizes:
            out = nn.Dense(layer)(out)
            out = nn.relu(out)
        out = nn.Dense(self.num_actions)(out)
        return Categorical(out)

class Critic(nn.Module):
    """A value network with 2 hidden layers that outputs a numerical value."""
    hidden_sizes: Sequence[int]

    @nn.compact
    def __call__(self, input):
        out = input
        for layer in self.hidden_sizes:
            out = nn.Dense(layer)(out)
            out = nn.relu(out)
        out = nn.Dense(1)(out)
        return out

@flax.struct.dataclass
class Transition:
    """A data class that stores a state transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

# Create environment and initialize actor and critic models

rng_key, actor_key, critic_key = jax.random.split(jax.random.key(SEED), 3)
env, env_params = gymnax.make(ENV)
num_actions = env.action_space(env_params).n
empty_observation = jnp.empty(env.observation_space(env_params).shape)

actor = Actor(params[ENV_KEY]["actor_sizes"], num_actions)
actor_params = actor.init(actor_key, empty_observation.ravel())
critic = Critic(params[ENV_KEY]["critic_sizes"])
critic_params = critic.init(critic_key, empty_observation.ravel())

print("Initialized actor parameters")
print("Observation shape:", empty_observation.shape)
print("Action space:", num_actions)
print()

# Create actor and critic train states

actor_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor_params,
    tx=optax.adam(ACTOR_RATE, eps=ADAM_EPS),
)
critic_state = TrainState.create(
    apply_fn=critic.apply,
    params=critic_params,
    tx=optax.adam(CRITIC_RATE, eps=ADAM_EPS),
)