from env import ENV, SEED

from distrax import Categorical
from flax import linen as nn
import gymnax
import jax
from jax import lax, numpy as jnp, random

class Actor(nn.Module):
    """
    A policy network with 3 hidden layers that outputs logits for each action. The logits
    are wrapped in a categorical distribution that is returned from each call.
    """
    num_actions: int

    @nn.compact
    def __call__(self, input):
        out = nn.Dense(50)(input)
        out = nn.relu(out)
        out = nn.Dense(20)(out)
        out = nn.relu(out)
        out = nn.Dense(self.num_actions)(out)
        return Categorical(out)

# Create actor model and initialize parameters
random_key, init_key = random.split(random.key(SEED), 2)
env, env_params = gymnax.make(ENV)
actor = Actor(env.action_space(env_params).n)
empty_observation = jnp.empty(env.observation_space(env_params).shape)
actor_params = actor.init(init_key, empty_observation)

print(actor)
print(env.observation_space(env_params).shape)
logits = actor.apply(actor_params, jnp.array([0, 1, 0, 1]))
print(logits)
cat = Categorical(logits)
print(cat.logits)
print(cat.log_prob(1))
print(cat.probs)