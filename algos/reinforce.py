from env import ENV, SEED

from distrax import Categorical
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import gymnax
import jax
import jax.numpy as jnp
import optax

# Configuration parameters

NUM_UPDATES = 100
ROLLOUT_LEN = 2
LEARNING_RATE = 0.001
ADAM_EPS = 1e-5

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

@flax.struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray

# Create actor model and initialize parameters

rng_key, init_key = jax.random.split(jax.random.key(SEED), 2)
env, env_params = gymnax.make(ENV)

actor = Actor(env.action_space(env_params).n)
empty_observation = jnp.empty(env.observation_space(env_params).shape)
actor_params = actor.init(init_key, empty_observation)

print("Initialized actor parameters")
print("Observation shape:", empty_observation.shape)
print("Action space:", actor.num_actions)

# Create a train state

train_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor_params,
    tx=optax.adam(LEARNING_RATE, eps=ADAM_EPS),
)

def run_rollout(rng_key):
    """Collects a policy rollout with a fixed number of steps."""
    rng_key, reset_key = jax.random.split(rng_key, 2)
    observation, env_state = env.reset(reset_key, env_params)

    def step(rollout_state, i):
        """Advances the environment by 1 step by sampling from the policy."""
        train_state, env_state, observation, rng_key = rollout_state
        rng_key, action_key, step_key = jax.random.split(rng_key, 3)

        # Sample action
        action_dist = actor.apply(train_state.params, observation)
        action = action_dist.sample(seed=action_key)
        log_prob = action_dist.log_prob(action)

        # Run environment step
        next_observation, next_state, reward, done, i = env.step(
            step_key, env_state, action, env_params,
        )
        transition = Transition(
            observation, action, reward, done, log_prob,
        )

        next_step = (train_state, next_state, next_observation, rng_key)
        return next_step, transition

    s, transitions = jax.lax.scan(
        step,
        init=(train_state, env_state, observation, rng_key),
        length=ROLLOUT_LEN,
    )
    return transitions

transitions = run_rollout(rng_key)
print(transitions)