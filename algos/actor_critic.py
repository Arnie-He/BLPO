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

NUM_UPDATES = 1#params[ENV_KEY]["num_updates"]
BATCH_COUNT = params[ENV_KEY]["batch_count"]
ROLLOUT_LEN = 20#params[ENV_KEY]["rollout_len"]
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
        return jnp.squeeze(out)

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

def run_rollout(actor_state, rng_key):
    """Collects an actor policy rollout with a fixed number of steps."""
    rng_key, reset_key = jax.random.split(rng_key, 2)
    observation, env_state = env.reset(reset_key, env_params)
    observation = observation.ravel()

    def step(rollout_state, x):
        """Advances the environment by 1 step by sampling from the policy."""
        # Sample action
        actor_state, env_state, observation, rng_key = rollout_state
        rng_key, action_key, step_key = jax.random.split(rng_key, 3)
        action_dist = actor.apply(actor_state.params, observation)
        action = action_dist.sample(seed=action_key)

        # Run environment step
        next_observation, next_state, reward, done, i = env.step(
            step_key, env_state, action, env_params,
        )
        next_observation = next_observation.ravel()
        transition = Transition(
            observation, action, reward, done,
        )

        next_step = (actor_state, next_state, next_observation, rng_key)
        return (next_step, transition)

    rollout_state, transitions = jax.lax.scan(
        step,
        init=(actor_state, env_state, observation, rng_key),
        length=ROLLOUT_LEN,
    )
    a, n, last_observation, r = rollout_state
    return (transitions, last_observation)

def calc_values(critic_state, transitions, last_observation):
    """Calculates the advantage estimate at each time step."""
    values = jax.vmap(
        lambda transition: critic.apply(critic_state.params, transition.observation),
    )(transitions)
    last_value = critic.apply(critic_state.params, last_observation)

    def calc_advantage(next_value, value_info):
        value, reward, done = value_info
        target = reward + DISCOUNT_RATE * next_value * (1 - done)
        advantage = target - value
        return (value, (advantage, target))

    v, result = jax.lax.scan(
        calc_advantage,
        init=last_value,
        xs=(values, transitions.reward, transitions.done),
        reverse=True,
    )
    advantages, targets = result
    return advantages, targets

transitions, last_observation = run_rollout(actor_state, rng_key)
print(transitions)
print(last_observation)
advantages, targets = calc_values(critic_state, transitions, last_observation)
print(advantages)
print(targets)