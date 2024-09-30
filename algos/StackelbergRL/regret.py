from environments import ENV_NAMES
from models.critic import Critic, PixelCritic
from models.discrete_actor import DiscreteActor, DiscretePixelActor
import models.params
from models.params import DynParam

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import functools
import gymnax
import jax
from jax import grad, jacfwd, jacrev
import jax.numpy as jnp
import optax
from jax import flatten_util
from jax.scipy.sparse.linalg import cg
import json

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
    },
    "space_invaders": {
        "actor_model": DiscretePixelActor,
        "actor_params": [
            (
                nn.Conv(features=8, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=8, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
            DynParam.ActionCount,
        ],
        "critic_model": PixelCritic,
        "critic_params": [
            (
                nn.Conv(features=8, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=8, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
        ],
        "hyperparams": Hyperparams(
            num_updates=2000,
            batch_count=50,
            rollout_len=2000,
            discount_rate=0.995,
            actor_learning_rate=0.0015,
            critic_learning_rate=0.003,
            critic_updates=20,
            adam_eps=1e-5,
        ),
    },
}

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

@jax.jit
def calc_values(critic_state, critic_params, transitions, last_observation, discount_rate):
    """Calculates the advantage estimate at each time step."""
    values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(critic_params, transitions.observation)
    last_value = critic_state.apply_fn(critic_params, last_observation)

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



def update_leadercritic(actor_state, critic_state, critic_prime_state, transitions, advantages, targets, advantages_prime, targets_prime, last_observation, hyperparams, vanilla=False, lambda_):

    def F_loss(critic_params, critic_state, transitions, targets):
        values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(critic_params, transitions.observation)
        errors = jnp.square(targets - values)
        return jnp.mean(errors)
    
    # define J function(actor's total objective)
    def g_loss(actor_params, critic_params, actor_state, critic_state, transitions, last_observation, hyperparams):
        advantages, _ = calc_values(critic_state, critic_params, transitions, last_observation, hyperparams.discount_rate)
        action_dists = jax.vmap(actor_state.apply_fn, in_axes=(None, 0))(actor_params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        return -jnp.mean(advantages * log_probs)
    
    # define the L function for computing grad_theta_L
    def L_surrogate(actor_params, critic_params, actor_state, critic_state, transitions, advantages, targets):
        action_dists = jax.vmap(actor_state.apply_fn, in_axes=(None, 0))(actor_params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(critic_params, transitions.observation)
        return 2 * jnp.mean(log_probs * advantages) * (targets[0] - values[0])
    

    def Lagrangian(actor_params, critic_params,critic_prime_params, lambda_ actor_state, critic_state, , critic_prime_state, transitions, advantages, targets, advantages_prime, targets_prime):
        F = F_loss(critic_params, critic_state, transitions, targets)
        g = g_loss(actor_params, critic_params, actor_state, critic_state, transitions, advantages, targets)
        g_prime = g_loss(actor_params, critic_prime_params, actor_state, critic_prime_state, transitions, advantages_prime, targets_prime)
        L = F + lambda_ * (g - g_prime)
        return L
    
    grad_Lagrangian = jax.grad(Lagrangian, argnums=(0,1,2,3))
    actor_grads, critic_grads, critic_prime_grads, lambda_grad = grad_Lagrangian(actor_state.params, critic_state.params, critic_prime_state.params, lambda_, actor_state, critic_state, critic_prime_state, transitions, advantages, targets, advantages_prime, targets_prime)
    actor_state = actor_state.apply_gradients(grads=actor_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)
    critic_prime_state = critic_prime_state.apply_gradients(grads=critic_prime_grads) #TODO MAKE THIS NEGATIVE
    lambda_ = lambda_ - lambda_grad #TODO MAKE THIS NEGATIVE with a proper gradient update


   

    return (actor_state, critic_state, critic_prime_state, lambda_)

def update_actor(actor_state, critic_state, transitions, targets, last_observation, hyperparams):
    """Calculates and applies the value target gradient at each time step."""
    actor_advantage_grad = jax.value_and_grad(J_loss)
    actor_loss, actor_grads = actor_advantage_grad(actor_state.params, critic_state.params, actor_state, critic_state, transitions, last_observation, hyperparams)
    actor_state = actor_state.apply_gradients(grads=actor_grads)

    return (actor_state, actor_loss)

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

def run_update(env, env_params, actor_state, critic_state, critic_prime_state, rng_key, hyperparams, vanilla=False):
    """Runs an iteration of the training loop with the vanilla parallel update."""
    rng_key, rollout_key = jax.random.split(rng_key, 2)
    transitions, last_observation = run_rollout(env, env_params, hyperparams.rollout_len, actor_state, rollout_key)
    advantages, targets = calc_values(critic_state, critic_state.params, transitions, last_observation, hyperparams.discount_rate)
    advantages_prime, targets_prime = calc_values(critic_prime_state, critic_prime_state.params, transitions, last_observation, hyperparams.discount_rate)


    actor_state, critic_state, critic_prime_state, lambda_ = update_leadercritic(actor_state, critic_state, critic_prime_state, transitions, advantages, targets, advantages_prime, targets_prime, last_observation, hyperparams, vanilla)
    #min actor, min critc, max critic_prime, max lambda
    actor_state = actor_state.apply_gradients(grads=actor_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)
    critic_prime_state = critic_prime_state.apply_gradients(grads=critic_prime_grads)



    critic_state, hypergradient_norm, critic_loss = update_leadercritic(actor_state, critic_state, transitions, advantages, targets, last_observation, hyperparams, vanilla)
    actor_loss = 0
    for c in range(hyperparams.critic_updates):
        actor_state, actor_loss = update_actor(actor_state, critic_state, transitions, targets, last_observation, hyperparams)

    total_rewards = calc_episode_rewards(transitions)
    average_reward = jnp.sum(total_rewards * transitions.done) / jnp.sum(transitions.done)
    return (actor_state, critic_state, (average_reward, hypergradient_norm, actor_loss, critic_loss), rng_key)

@functools.partial(jax.jit, static_argnums=0)
def run_batch(env, env_params, actor_state, critic_state, critic_prime_state, rng_key, hyperparams, vanilla=False):
    """Trains the model for a batch of updates."""
    def run_once(batch_state, x):
        """Runs an update and carries over the train state."""
        actor_state, critic_state, critic_prime_state, rng_key = batch_state
        actor_state, critic_state, critic_prime_state, metrics, rng_key = \
            run_update(env, env_params, actor_state, critic_state, critic_prime_state, rng_key, hyperparams, vanilla)
        return ((actor_state, critic_state, critic_prime_state, rng_key), metrics)

    batch_state, batch_metrics = jax.lax.scan(
        run_once,
        init=(actor_state, critic_state, critic_prime_state, rng_key),
        length=hyperparams.batch_count,
    )
    actor_state, critic_state, critic_prime_state, rng_key = batch_state
    return (actor_state, critic_state, critic_prime_state, batch_metrics, rng_key)

def train(env_key, seed, logger, verbose = False, metrics=None, vanilla=False, save_charts=False, description=None):
    # Create environment
    config = ENV_CONFIG[env_key]
    hyperparams = config["hyperparams"]
    rng_key, actor_key, critic_key = jax.random.split(jax.random.key(seed), 3)
    env, env_params = gymnax.make(ENV_NAMES[env_key])
    empty_observation = jnp.empty(env.observation_space(env_params).shape)

    # Initialize actor model
    actor_model_params = models.params.init(env, env_params, config["actor_params"])
    actor = config["actor_model"](*actor_model_params)
    actor_params = actor.init(actor_key, empty_observation)

    # Initialize critic model
    critic_model_params = models.params.init(env, env_params, config["critic_params"])
    critic = config["critic_model"](*critic_model_params)
    critic_params = critic.init(critic_key, empty_observation)

    critic_prime_model_params = models.params.init(env, env_params, config["critic_params"])
    critic_prime = config["critic_model"](*critic_prime_model_params)
    critic_prime_params = critic_prime.init(critic_key, empty_observation)




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

    critic_prime_state = TrainState.create(
        apply_fn=jax.jit(critic_prime.apply),
        params=critic_prime_params,
        tx=optax.adam(hyperparams.critic_learning_rate, eps=hyperparams.adam_eps),
    )

    if(save_charts):
        from datetime import datetime
        import os
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if(description==None): 
            description=current_datetime
        folder_path = f"charts/stackelberg_Actor/{description}"
        os.makedirs(folder_path, exist_ok=True)

        logger.set_interval(hyperparams.rollout_len)
        for metric in metrics:
            file_path = f"{folder_path}/{env_key}_{metric}.png"
            
            logger.set_info(
                metric,
                f"[{ENV_NAMES[env_key]}] SA2C {metric}",
                file_path,
            )

    # Run the training loop
    num_batches = int(hyperparams.num_updates / hyperparams.batch_count)
    for b in range(num_batches):
        actor_state, critic_state, critic_prime_state, batch_metrics, rng_key = \
            run_batch(env, env_params, actor_state, critic_state, critic_prime_state, rng_key, hyperparams, vanilla)
        if(save_charts):
            logger.log_metrics({
                "reward": batch_metrics[0],
                "hypergradient_norms": batch_metrics[1], 
                "actor_loss": batch_metrics[2],
                "critic_loss": batch_metrics[3],
            })
        if verbose:
            print(f"[Update {(b + 1) * hyperparams.batch_count}]: Average reward {batch_metrics[0][-1]}, Hypergradient Norm {jnp.mean(batch_metrics[1][1])}, finalProduct Norm{jnp.mean(batch_metrics[1][2])}")