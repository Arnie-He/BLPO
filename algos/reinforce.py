from environments import ENV_NAMES

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

@functools.partial(flax.struct.dataclass, kw_only=True)
class Config:
    hidden_sizes: tuple[int] = flax.struct.field(pytree_node=False)
    num_updates: int = flax.struct.field(pytree_node=False)
    batch_count: int = flax.struct.field(pytree_node=False)
    rollout_len: int = flax.struct.field(pytree_node=False)
    discount_rate: float = flax.struct.field(pytree_node=False)
    learning_rate: float = flax.struct.field(pytree_node=False)
    adam_eps: float = flax.struct.field(pytree_node=False)

ENV_CONFIG = {
    "cartpole": Config(
        hidden_sizes=(30, 15),
        num_updates=500,
        batch_count=25,
        rollout_len=2000,
        discount_rate=0.99,
        learning_rate=0.002,
        adam_eps=1e-5,
    ),
    "catch": Config(
        hidden_sizes=(30, 15),
        num_updates=1000,
        batch_count=50,
        rollout_len=1000,
        discount_rate=0.99,
        learning_rate=0.002,
        adam_eps=1e-5,
    ),
    "breakout": Config(
        hidden_sizes=(400, 100),
        num_updates=1000,
        batch_count=50,
        rollout_len=2000,
        discount_rate=0.995,
        learning_rate=0.001,
        adam_eps=1e-5,
    ),
}

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

@flax.struct.dataclass
class Transition:
    """A data class that stores a state transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

def run_rollout(env, env_params, train_state, rng_key, config):
    """Collects a policy rollout with a fixed number of steps."""
    rng_key, reset_key = jax.random.split(rng_key, 2)
    observation, env_state = env.reset(reset_key, env_params)
    observation = observation.ravel()

    def step(rollout_state, x):
        """Advances the environment by 1 step by sampling from the policy."""
        # Sample action
        train_state, env_state, observation, rng_key = rollout_state
        rng_key, action_key, step_key = jax.random.split(rng_key, 3)
        action_dist = train_state.apply_fn(train_state.params, observation)
        action = action_dist.sample(seed=action_key)

        # Run environment step
        next_observation, next_state, reward, done, i = env.step(
            step_key, env_state, action, env_params,
        )
        next_observation = next_observation.ravel()
        transition = Transition(
            observation, action, reward, done,
        )

        next_step = (train_state, next_state, next_observation, rng_key)
        return (next_step, transition)

    s, transitions = jax.lax.scan(
        step,
        init=(train_state, env_state, observation, rng_key),
        length=config.rollout_len,
    )
    return transitions

def calc_discounted_rewards(transitions, discount_rate):
    """
    Calculates the cumulative discounted reward at each time step. The rewards are
    normalized to reduce variance with a rudimentary baseline.
    """
    def calc_reward(total, transition):
        """Adds the current reward to the total and applies the discount."""
        total = jax.lax.select(
            transition.done,
            transition.reward,
            transition.reward + discount_rate * total,
        )
        return (total, total)

    s, rewards = jax.lax.scan(
        calc_reward,
        init=jnp.float32(0),
        xs=transitions,
        reverse=True,
    )
    rewards = (rewards - jnp.mean(rewards)) / (jnp.std(rewards) + 1e-8)
    return rewards

def calc_episode_mask(transitions):
    """Calculates a mask that is 1 if a transition is part of a full trajectory and 0 otherwise."""
    s, mask = jax.lax.scan(
        lambda prev, transition: (prev | transition.done,) * 2,
        init=jnp.int8(0),
        xs=transitions,
        reverse=True,
    )
    return mask

def update_actor(train_state, transitions, rewards, episode_mask):
    """Calculates and applies the REINFORCE gradient estimator at each time step."""
    def reinforce_loss(params, transitions, rewards, episode_mask):
        """Calculates the REINFORCE estimator on a batch of transitions."""
        action_dists = train_state.apply_fn(params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        reinforce_losses = -log_probs * rewards * episode_mask
        return jnp.sum(reinforce_losses) / jnp.sum(episode_mask)

    reinforce_grad = jax.value_and_grad(reinforce_loss)
    loss, grads = reinforce_grad(train_state.params, transitions, rewards, episode_mask)
    train_state = train_state.apply_gradients(grads=grads)
    return (train_state, loss)

def calc_episode_rewards(transitions):
    """Calculates the total real reward for each episode."""
    def calc_reward(prev_total, transition):
        """Adds the current reward to the total."""
        total = prev_total + transition.reward
        next_total = total * (1 - transition.done)
        return (next_total, total)

    s, rewards = jax.lax.scan(
        calc_reward,
        init=jnp.float32(0),
        xs=transitions,
    )
    return rewards

def run_update(env, env_params, train_state, rng_key, config):
    """Runs an iteration of the training loop by sampling trajectories and applying policy gradients."""
    rng_key, rollout_key = jax.random.split(rng_key, 2)
    transitions = run_rollout(env, env_params, train_state, rollout_key, config)
    rewards = calc_discounted_rewards(transitions, config.discount_rate)
    episode_mask = calc_episode_mask(transitions)
    train_state, loss = update_actor(train_state, transitions, rewards, episode_mask)

    total_rewards = calc_episode_rewards(transitions)
    average_reward = jnp.sum(total_rewards * transitions.done) / jnp.sum(transitions.done)
    return (train_state, average_reward, loss, rng_key)

@functools.partial(jax.jit, static_argnums=0)
def run_batch(env, env_params, train_state, rng_key, config):
    """Trains the model for a batch of updates."""
    def run_once(batch_state, x):
        """Runs an update and carries over the train state."""
        train_state, rng_key = batch_state
        train_state, average_reward, loss, rng_key = \
            run_update(env, env_params, train_state, rng_key, config)
        return ((train_state, rng_key), (average_reward, loss))

    batch_state, results = jax.lax.scan(
        run_once,
        init=(train_state, rng_key),
        length=config.batch_count,
    )
    train_state, rng_key = batch_state
    average_rewards, losses = results
    return (train_state, rng_key, average_rewards, losses)

def train(env_key, seed):
    # Create environment and initialize actor model

    rng_key, init_key = jax.random.split(jax.random.key(seed), 2)
    env, env_params = gymnax.make(ENV_NAMES[env_key])
    num_actions = env.action_space(env_params).n
    empty_observation = jnp.empty(env.observation_space(env_params).shape)

    config = ENV_CONFIG[env_key]
    actor = Actor(config.hidden_sizes, num_actions)
    actor_params = actor.init(init_key, empty_observation.ravel())

    print("Initialized actor parameters")
    print("Observation shape:", empty_observation.shape)
    print("Action space:", num_actions)
    print()

    # Create a train state

    train_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=optax.adam(config.learning_rate, eps=config.adam_eps),
    )

    # Run the training loop

    average_rewards = []
    losses = []
    for u in range(int(config.num_updates / config.batch_count)):
        train_state, rng_key, batch_rewards, batch_losses = \
            run_batch(env, env_params, train_state, rng_key, config)
        average_rewards += [float(r) for r in batch_rewards]
        losses += [float(l) for l in batch_losses]
        print(f"[Update {(u + 1) * config.batch_count}]: Average reward {batch_rewards[-1]}")

    # Plot rewards and losses

    step_counts = [u * config.rollout_len for u in range(1, config.num_updates + 1)]

    reward_figure, reward_axes = plt.subplots()
    reward_axes.plot(step_counts, average_rewards)
    reward_axes.set_title(f"[{ENV_NAMES[env_key]}] REINFORCE average reward")
    reward_figure.savefig(f"./charts/reinforce/{env_key}_reward.png")

    loss_figure, loss_axes = plt.subplots()
    loss_axes.plot(step_counts, losses)
    loss_axes.set_title(f"[{ENV_NAMES[env_key]}] REINFORCE loss")
    loss_figure.savefig(f"./charts/reinforce/{env_key}_loss.png")