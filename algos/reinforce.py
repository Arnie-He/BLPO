from env import ENV, SEED

from distrax import Categorical
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

# Configuration parameters

NUM_UPDATES = 1000
ROLLOUT_LEN = 2000
DISCOUNT_RATE = 0.99
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

# Create actor model and initialize parameters

rng_key, init_key = jax.random.split(jax.random.key(SEED), 2)
env, env_params = gymnax.make(ENV)

actor = Actor(env.action_space(env_params).n)
empty_observation = jnp.empty(env.observation_space(env_params).shape)
actor_params = actor.init(init_key, empty_observation)

print("Initialized actor parameters")
print("Observation shape:", empty_observation.shape)
print("Action space:", actor.num_actions)
print()

# Create a train state

train_state = TrainState.create(
    apply_fn=actor.apply,
    params=actor_params,
    tx=optax.adam(LEARNING_RATE, eps=ADAM_EPS),
)

@jax.jit
def run_rollout(train_state, rng_key):
    """Collects a policy rollout with a fixed number of steps."""
    rng_key, reset_key = jax.random.split(rng_key, 2)
    observation, env_state = env.reset(reset_key, env_params)

    def step(rollout_state, i):
        """Advances the environment by 1 step by sampling from the policy."""
        # Sample action
        train_state, env_state, observation, rng_key = rollout_state
        rng_key, action_key, step_key = jax.random.split(rng_key, 3)
        action_dist = actor.apply(train_state.params, observation)
        action = action_dist.sample(seed=action_key)

        # Run environment step
        next_observation, next_state, reward, done, i = env.step(
            step_key, env_state, action, env_params,
        )
        transition = Transition(
            observation, action, reward, done,
        )

        next_step = (train_state, next_state, next_observation, rng_key)
        return (next_step, transition)

    s, transitions = jax.lax.scan(
        step,
        init=(train_state, env_state, observation, rng_key),
        length=ROLLOUT_LEN,
    )
    return transitions

@jax.jit
def calc_discounted_rewards(transitions):
    """Calculates the cumulative discounted reward at each time step."""
    def calc_reward(total, transition):
        """Adds the current reward to the total and applies the discount."""
        total = jax.lax.select(
            transition.done,
            transition.reward,
            transition.reward + DISCOUNT_RATE * total,
        )
        return (total, total)

    s, rewards = jax.lax.scan(
        calc_reward,
        init=jnp.float32(0),
        xs=transitions,
        reverse=True,
    )
    average_reward = jnp.mean(rewards)
    rewards = rewards - jnp.mean(rewards)
    rewards = rewards / jnp.std(rewards)

    return (rewards, average_reward)

@jax.jit
def calc_episode_mask(transitions):
    """Calculates a mask that is 1 if a transition is part of a full trajectory and 0 otherwise."""
    s, mask = jax.lax.scan(
        lambda prev, transition: (prev | transition.done,) * 2,
        init=jnp.int8(0),
        xs=transitions,
        reverse=True,
    )
    return mask

@jax.jit
def update_actor(train_state, transitions, rewards, episode_mask):
    """Calculates and applies the REINFORCE estimator gradient at each time step."""
    def reinforce_loss(params, transitions, rewards, episode_mask):
        """Calculates the REINFORCE estimator on a batch of transitions."""
        action_dists = actor.apply(params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        reinforce_losses = -log_probs * rewards * episode_mask
        return jnp.sum(reinforce_losses) / jnp.sum(episode_mask)

    reinforce_grad = jax.value_and_grad(reinforce_loss)
    loss, grads = reinforce_grad(train_state.params, transitions, rewards, episode_mask)
    train_state = train_state.apply_gradients(grads=grads)
    return (train_state, loss)

@jax.jit
def run_update(train_state, rng_key):
    """Runs an iteration of the training loop by sampling trajectories and applying policy gradients."""
    rng_key, rollout_key = jax.random.split(rng_key, 2)
    transitions = run_rollout(train_state, rollout_key)
    rewards, average_reward = calc_discounted_rewards(transitions)
    episode_mask = calc_episode_mask(transitions)
    train_state, loss = update_actor(train_state, transitions, rewards, episode_mask)

    return (train_state, average_reward, loss, rng_key)

# Calculate the optimal average reward as a benchmark

optimal_rewards = [1] * 500
for r in reversed(range(len(optimal_rewards) - 1)):
    optimal_rewards[r] += DISCOUNT_RATE * optimal_rewards[r + 1]
optimal_average = jnp.mean(jnp.array(optimal_rewards))
print("Optimal average reward:", optimal_average)

# Run the training loop

average_rewards = []
losses = []
for u in range(NUM_UPDATES + 1):
    train_state, average_reward, loss, rng_key = run_update(train_state, rng_key)
    average_rewards.append(float(average_reward))
    losses.append(float(loss))
    if u % 50 == 0:
        print(f"[Update {u}]: Average reward {average_reward}")

# Plot rewards and losses

reward_figure, reward_axes = plt.subplots()
reward_axes.plot(
    list(range(1, len(average_rewards) + 1)),
    average_rewards,
)
reward_axes.set_title("REINFORCE average reward")
reward_figure.savefig("./charts/reinforce_reward.png")

loss_figure, loss_axes = plt.subplots()
loss_axes.plot(
    list(range(1, len(losses) + 1)),
    losses,
)
loss_axes.set_title("REINFORCE loss")
loss_figure.savefig("./charts/reinforce_loss.png")