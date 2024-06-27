from algos.env import ENV, ENV_KEY, SEED
from models.discrete_actor import DiscreteActor as Actor
from models.critic import Critic

from environments import ENV_NAMES
from models.discrete_actor import DiscreteActor
from models.critic import Critic
import models.params
from models.params import DynParam

import flax
from flax.training.train_state import TrainState
import functools
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

@functools.partial(flax.struct.dataclass, kw_only=True)
class Hyperparams:
    """A data class that stores hyperparameter configurations."""
    num_updates: int = flax.struct.field(pytree_node=False)
    batch_count: int = flax.struct.field(pytree_node=False)
    rollout_len: int = flax.struct.field(pytree_node=False)
    discount_rate: float = flax.struct.field(pytree_node=False)
    actor_learning_rate: float = flax.struct.field(pytree_node=False)
    critic_learning_rate: float = flax.struct.field(pytree_node=False)
    critic_updates: int = flax.struct.field(pytree_node=False)
    adam_eps: float = flax.struct.field(pytree_node=False)

ENV_CONFIG = {
    "cartpole": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=500,
            batch_count=25,
            rollout_len=2000,
            discount_rate=0.99,
            actor_learning_rate=0.0025,
            critic_learning_rate=0.004,
            critic_updates=25,
            adam_eps=1e-5,
        ),
    },
    "catch": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=1000,
            batch_count=50,
            rollout_len=1000,
            discount_rate=0.99,
            actor_learning_rate=0.0025,
            critic_learning_rate=0.004,
            critic_updates=25,
            adam_eps=1e-5,
        ),
    }
}

params = {
    "cartpole": {
        "actor_sizes": (30, 15),
        "critic_sizes": (30, 15),
        "num_updates": 500,
        "batch_count": 25,
        "rollout_len": 2000,
        "discount_rate": 0.99,
        "actor_learning_rate": 0.0025,
        "critic_learning_rate": 0.004,
        "critic_updates": 25,
    },
    "catch": {
        "actor_sizes": (30, 15),
        "critic_sizes": (30, 15),
        "num_updates": 1000,
        "batch_count": 50,
        "rollout_len": 1000,
        "discount_rate": 0.99,
        "actor_learning_rate": 0.0025,
        "critic_learning_rate": 0.004,
        "critic_updates": 25,
    },
    "breakout": {
        "actor_sizes": (400, 100),
        "critic_sizes": (400, 100),
        "num_updates": 1000,
        "batch_count": 50,
        "rollout_len": 2000,
        "discount_rate": 0.995,
        "actor_learning_rate": 0.0015,
        "critic_learning_rate": 0.003,
        "critic_updates": 20,
    },
}

NUM_UPDATES = params[ENV_KEY]["num_updates"]
BATCH_COUNT = params[ENV_KEY]["batch_count"]
ROLLOUT_LEN = params[ENV_KEY]["rollout_len"]
DISCOUNT_RATE = params[ENV_KEY]["discount_rate"]
ACTOR_RATE = params[ENV_KEY]["actor_learning_rate"]
CRITIC_RATE = params[ENV_KEY]["critic_learning_rate"]
CRITIC_UPDATES = params[ENV_KEY]["critic_updates"]
ADAM_EPS = 1e-5

@flax.struct.dataclass
class Transition:
    """A data class that stores a state transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray

def run_rollout(env, env_params, length, actor_state, rng_key):
    """Collects an actor policy rollout with a fixed number of steps."""
    rng_key, reset_key = jax.random.split(rng_key, 2)
    observation, env_state = env.reset(reset_key, env_params)

    def step(rollout_state, x):
        """Advances the environment by 1 step by sampling from the policy."""
        # Sample action
        actor_state, env_state, observation, rng_key = rollout_state
        rng_key, action_key, step_key = jax.random.split(rng_key, 3)
        action_dist = actor_state.apply_fn(actor_state.params, observation)
        action = action_dist.sample(seed=action_key)

        # Run environment step
        next_observation, next_state, reward, done, i = env.step(
            step_key, env_state, action, env_params,
        )
        transition = Transition(
            observation, action, reward, done,
        )

        next_step = (actor_state, next_state, next_observation, rng_key)
        return (next_step, transition)

    rollout_state, transitions = jax.lax.scan(
        step,
        init=(actor_state, env_state, observation, rng_key),
        length=length,
    )
    a, n, last_observation, r = rollout_state
    return (transitions, last_observation)

def calc_values(critic_state, transitions, last_observation, discount_rate):
    """Calculates the advantage estimate at each time step."""
    values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(critic_state.params, transitions.observation)
    last_value = critic_state.apply_fn(critic_state.params, last_observation)

    def calc_advantage(next_value, value_info):
        value, reward, done = value_info
        target = reward + discount_rate * next_value * (1 - done)
        advantage = target - value
        return (value, (advantage, target))

    v, result = jax.lax.scan(
        calc_advantage,
        init=last_value,
        xs=(values, transitions.reward, transitions.done),
        reverse=True,
    )
    advantages, targets = result
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    return (advantages, targets)

def update_actor(actor_state, transitions, advantages):
    """Calculates and applies the advantage gradient estimator at each time step."""
    def advantage_loss(params, transitions, advantages):
        """Calculates the advantage estimator on a batch of transitions."""
        action_dists = jax.vmap(actor_state.apply_fn, in_axes=(None, 0))(params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        return -jnp.mean(advantages * log_probs)

    advantage_grad = jax.value_and_grad(advantage_loss)
    loss, grads = advantage_grad(actor_state.params, transitions, advantages)
    actor_state = actor_state.apply_gradients(grads=grads)
    return (actor_state, loss)

def update_critic(critic_state, transitions, targets):
    """Calculates and applies the value target gradient at each time step."""
    def target_loss(params, transitions, targets):
        """Calculates the mean squared error on a batch of transitions."""
        values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(params, transitions.observation)
        errors = jnp.square(targets - values)
        return jnp.mean(errors)

    target_grad = jax.value_and_grad(target_loss)
    loss, grads = target_grad(critic_state.params, transitions, targets)
    critic_state = critic_state.apply_gradients(grads=grads)
    return (critic_state, loss)

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

def run_update(env, env_params, actor_state, critic_state, rng_key, hyperparams):
    """Runs an iteration of the training loop with the vanilla parallel update."""
    rng_key, rollout_key = jax.random.split(rng_key, 2)
    transitions, last_observation = run_rollout(env, env_params, hyperparams.rollout_len, actor_state, rollout_key)
    advantages, targets = calc_values(critic_state, transitions, last_observation, hyperparams.discount_rate)

    actor_state, actor_loss = update_actor(actor_state, transitions, advantages)
    critic_loss = 0
    for c in range(CRITIC_UPDATES):
        critic_state, critic_loss = update_critic(critic_state, transitions, targets)

    total_rewards = calc_episode_rewards(transitions)
    average_reward = jnp.sum(total_rewards * transitions.done) / jnp.sum(transitions.done)
    return (actor_state, critic_state, (average_reward, actor_loss, critic_loss), rng_key)

@functools.partial(jax.jit, static_argnums=0)
def run_batch(env, env_params, actor_state, critic_state, rng_key, hyperparams):
    """Trains the model for a batch of updates."""
    def run_once(batch_state, x):
        """Runs an update and carries over the train state."""
        actor_state, critic_state, rng_key = batch_state
        actor_state, critic_state, metrics, rng_key = \
            run_update(env, env_params, actor_state, critic_state, rng_key, hyperparams)
        return ((actor_state, critic_state, rng_key), metrics)

    batch_state, batch_metrics = jax.lax.scan(
        run_once,
        init=(actor_state, critic_state, rng_key),
        length=hyperparams.batch_count,
    )
    actor_state, critic_state, rng_key = batch_state
    return (actor_state, critic_state, batch_metrics, rng_key)

# Create environment and initialize actor and critic models

rng_key, actor_key, critic_key = jax.random.split(jax.random.key(SEED), 3)
env, env_params = gymnax.make(ENV)
num_actions = env.action_space(env_params).n
empty_observation = jnp.empty(env.observation_space(env_params).shape)

actor = Actor(params[ENV_KEY]["actor_sizes"], num_actions)
actor_params = actor.init(actor_key, empty_observation)
critic = Critic(params[ENV_KEY]["critic_sizes"])
critic_params = critic.init(critic_key, empty_observation)

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

# Run the training loop

average_rewards = []
actor_losses = []
critic_losses = []

hyperparams = ENV_CONFIG[ENV_KEY]["hyperparams"]
num_batches = int(hyperparams.num_updates / hyperparams.batch_count)
for u in range(num_batches):
    actor_state, critic_state, batch_metrics, rng_key = \
        run_batch(env, env_params, actor_state, critic_state, rng_key, hyperparams)
    average_rewards += [float(r) for r in batch_metrics[0]]
    actor_losses += [float(l) for l in batch_metrics[1]]
    critic_losses += [float(l) for l in batch_metrics[2]]
    print(f"[Update {(u + 1) * hyperparams.batch_count}]: Average reward {batch_metrics[0][-1]}")

# Plot rewards and losses

step_counts = [u * ROLLOUT_LEN for u in range(1, NUM_UPDATES + 1)]

reward_figure, reward_axes = plt.subplots()
reward_axes.plot(step_counts, average_rewards)
reward_axes.set_title(f"[{ENV}] Actor-Critic average reward")
reward_figure.savefig(f"./charts/actor_critic/{ENV_KEY}_reward.png")

actor_loss_figure, actor_loss_axes = plt.subplots()
actor_loss_axes.plot(step_counts, actor_losses)
actor_loss_axes.set_title(f"[{ENV}] Actor loss")
actor_loss_figure.savefig(f"./charts/actor_critic/{ENV_KEY}_actor_loss.png")

critic_loss_figure, critic_loss_axes = plt.subplots()
critic_loss_axes.plot(step_counts, critic_losses)
critic_loss_axes.set_title(f"[{ENV}] Critic loss")
critic_loss_figure.savefig(f"./charts/actor_critic/{ENV_KEY}_critic_loss.png")