from environments import GYM_ENV_NAMES
from models.critic import Critic
from models.actor import DiscreteActor
import models.params
from models.params import DynParam

import flax
from flax.training.train_state import TrainState
import functools
import gymnax
import jax
import jax.numpy as jnp
import optax
from algos.core.env_config import Hyperparams
from algos.core.env_config import ENV_CONFIG
from algos.core.purejxrl_wrappers import LogWrapper, FlattenObservationWrapper


@flax.struct.dataclass
class Transition:
    """A data class that stores a state transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray

def calc_values(critic_state, transitions, last_observation, discount_rate, advantage_rate):
    """Calculates the advantage estimate at each time step."""
    values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(critic_state.params, transitions.observation)
    last_value = critic_state.apply_fn(critic_state.params, last_observation)

    def calc_advantage(next_values, value_info):
        next_value, next_advantage = next_values
        value, reward, done = value_info
        target = reward + discount_rate * next_value * (1 - done)
        advantage = (target - value) + advantage_rate * discount_rate * next_advantage * (1 - done)
        return ((value, advantage), (advantage, target))

    v, result = jax.lax.scan(
        calc_advantage,
        init=(last_value, jnp.float32(0)),
        xs=(values, transitions.reward, transitions.done),
        reverse=True,
    )
    advantages, targets = result
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    return (advantages, targets)

def update_actor(actor_state, transitions, advantages, clip):
    """Calculates and applies the PPO gradient estimator at each time step."""
    def ppo_loss(params, transitions, advantages):
        """Calculates the clipped advantage estimator on a batch of transitions."""
        action_dists = jax.vmap(actor_state.apply_fn, in_axes=(None, 0))(params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        prob_ratios = jnp.exp(log_probs - transitions.log_prob)

        advantage_losses = prob_ratios * advantages
        clipped_ratios = jnp.clip(prob_ratios, 1 - clip, 1 + clip)
        clipped_losses = clipped_ratios * advantages

        ppo_losses = jnp.minimum(advantage_losses, clipped_losses)
        return -jnp.mean(ppo_losses)

    advantage_grad = jax.value_and_grad(ppo_loss)
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

def run_update(obsv, env_state, env, env_params, actor_state, critic_state, rng_key, hyperparams):
    """Runs an iteration of the training loop with the vanilla parallel update."""
    # Collect transitions and calculate advantages and targets
    rng_key, rollout_key, shuffle_key = jax.random.split(rng_key, 3)

    def run_rollout(obsv, env_state, env, env_params, length, actor_state, rng_key):
        """Collects an actor policy rollout with a fixed number of steps."""
        rng_key, reset_key = jax.random.split(rng_key, 2)

        def step(rollout_state, x):
            """Advances the environment by 1 step by sampling from the policy."""
            # Sample action
            actor_state, env_state, observation, rng_key = rollout_state
            rng_key, action_key, step_key = jax.random.split(rng_key, 3)
            action_dist = actor_state.apply_fn(actor_state.params, observation)
            action = action_dist.sample(seed=action_key)
            log_prob = action_dist.log_prob(action)

            # Run environment step
            obsv, env_state, reward, done, i = jax.vmap(env.step, in_axes=(0, 0, 0, None)).step(
                step_key, env_state, action, env_params,
            )
            transition = Transition(
                observation, action, reward, done, log_prob,
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

    transitions, last_observation = run_rollout(env, env_params, hyperparams.rollout_len, actor_state, rollout_key)
    advantages, targets = calc_values(
        critic_state,
        transitions,
        last_observation,
        hyperparams.discount_rate,
        hyperparams.advantage_rate,
    )

    # Calculate rewards before batching
    total_rewards = calc_episode_rewards(transitions)
    average_reward = jnp.sum(total_rewards * transitions.done) / jnp.sum(transitions.done)

    # Shuffle the transitions and split into minibatches
    shuffle = jax.random.permutation(shuffle_key, hyperparams.rollout_len)
    transitions = jax.tree.map(
        lambda v: jnp.array(jnp.split(v[shuffle], hyperparams.num_minibatches)),
        transitions,
    )
    advantages = jnp.array(jnp.split(advantages[shuffle], hyperparams.num_minibatches))
    targets = jnp.array(jnp.split(targets[shuffle], hyperparams.num_minibatches))

    # Train the actor and critic on each minibatch
    def update_minibatch(train_state, batch_info):
        actor_state, critic_state = train_state
        transitions, advantages, targets = batch_info
        actor_state, actor_loss = update_actor(actor_state, transitions, advantages, hyperparams.actor_clip)
        critic_state, critic_loss = update_critic(critic_state, transitions, targets)
        return ((actor_state, critic_state), (actor_loss, critic_loss))

    # Run epochs over all the minibatches of transitions
    actor_loss = 0
    critic_loss = 0
    for e in range(hyperparams.update_epochs):
        train_state, losses = jax.lax.scan(
            update_minibatch,
            init=(actor_state, critic_state),
            xs=(transitions, advantages, targets),
        )
        actor_state, critic_state = train_state
        actor_losses, critic_losses = losses
        actor_loss += jnp.mean(actor_losses) / hyperparams.update_epochs
        critic_loss += jnp.mean(critic_losses) / hyperparams.update_epochs

    return (
        actor_state,
        critic_state,
        (average_reward, actor_loss, critic_loss),
        rng_key,
    )

@functools.partial(jax.jit, static_argnums=0)
def run_batch(obsv, env_state, env, env_params, actor_state, critic_state, rng_key, hyperparams):
    """Trains the model for a batch of updates."""

    runner_state = (obsv, env_state, actor_state, critic_state, rng_key, hyperparams)

    def run_once(runner_state, unused):
        """Runs an update and carries over the train state."""
        obsv, env_state, actor_state, critic_state, rng_key = runner_state
        obsv, env_state, actor_state, critic_state, metrics, rng_key = \
            run_update(obsv, env_state, env_params, actor_state, critic_state, rng_key, hyperparams)
        return ((obsv, env_state, actor_state, critic_state, rng_key), metrics)

    batch_state, batch_metrics = jax.lax.scan(
        run_once,
        init=runner_state,
        length=hyperparams.update_epochs,
    )
    actor_state, critic_state, rng_key = batch_state
    return (actor_state, critic_state, batch_metrics, rng_key)

def train(env_key, seed, logger, hyperparams, verbose = False):
    # Create environment
    config = ENV_CONFIG[env_key]
    hyperparams = config["hyperparams"]

    # Make Gymnax Environments
    rng_key, actor_key, critic_key = jax.random.split(jax.random.key(seed), 3)
    env, env_params = gymnax.make(GYM_ENV_NAMES[env_key])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    empty_observation = jnp.empty(env.observation_space(env_params).shape)

    # Initialize actor model
    actor_model_params = models.params.init_gymnax(env, env_params, config["actor_params"])
    actor = config["actor_model"](*actor_model_params)
    actor_params = actor.init(actor_key, empty_observation)

    # Initialize critic model
    critic_model_params = models.params.init_gymnax(env, env_params, config["critic_params"])
    critic = config["critic_model"](*critic_model_params)
    critic_params = critic.init(critic_key, empty_observation)

    # Create actor and critic train states
    actor_state = TrainState.create(
        apply_fn=jax.jit(actor.apply),
        params=actor_params,
        tx=optax.adam(hyperparams.actor_learning_rate, eps=hyperparams.adam_eps),
    )
    critic_state = TrainState.create(
        apply_fn=jax.jit(critic.apply),
        params=critic_params,
        tx=optax.adam(hyperparams.critic_learning_rate, eps=hyperparams.adam_eps),
    )

    # Set logger info
    logger.set_interval(hyperparams.rollout_len * hyperparams.num_env)
    logger.set_info(
        "reward",
        f"[{GYM_ENV_NAMES[env_key]}] PPO average reward",
        f"charts/ppo/{env_key}_reward.png",
    )
    logger.set_info(
        "actor_loss",
        f"[{GYM_ENV_NAMES[env_key]}] PPO actor loss",
        f"charts/ppo/{env_key}_actor_loss.png",
    )
    logger.set_info(
        "critic_loss",
        f"[{GYM_ENV_NAMES[env_key]}] PPO critic loss",
        f"charts/ppo/{env_key}_critic_loss.png",
    )

    # Run the training loop
    num_batches = int(hyperparams.total_timesteps // hyperparams.num_envs // hyperparams.rollout_len)
    minibatch_size = (hyperparams.num_envs * hyperparams.rollout_len // hyperparams.num_minibatches)
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, hyperparams.num_envs)
    obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    for b in range(num_batches):
        actor_state, critic_state, obsv, env_state, batch_metrics, rng_key = \
            run_batch(obsv, env_state, env_params, actor_state, critic_state, rng_key, hyperparams)
        
        logger.log_metrics({
            "reward": batch_metrics[0],
            "actor_loss": batch_metrics[1],
            "critic_loss": batch_metrics[2],
        })
        if verbose:
            print(f"[Update {(b + 1) * hyperparams.batch_count}]: Average reward {batch_metrics[0][-1]}")