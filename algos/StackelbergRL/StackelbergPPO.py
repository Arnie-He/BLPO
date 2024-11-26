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

from algos.core.env_config import ENV_CONFIG
from algos.core.understanding_gradients import cosine_similarity

@flax.struct.dataclass
class Transition:
    """A data class that stores a state transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray

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
        log_prob = action_dist.log_prob(action)

        # Run environment step
        next_observation, next_state, reward, done, i = env.step(
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

def calc_values(critic_state, critic_params, transitions, last_observation, discount_rate, advantage_rate):
    """Calculates the advantage estimate at each time step."""
    values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(critic_params, transitions.observation)
    last_value = critic_state.apply_fn(critic_params, last_observation)

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

############################# Actor and Critic Update #####################
def target_loss(params, transitions, targets, critic_state):
    """Calculates the mean squared error on a batch of transitions."""
    values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0))(params, transitions.observation)
    errors = jnp.square(targets - values)
    return jnp.mean(errors)

def update_critic(critic_state, transitions, targets):
    """Calculates and applies the value target gradient at each time step."""
    target_grad = jax.value_and_grad(target_loss)
    loss, grads = target_grad(critic_state.params, transitions, targets, critic_state)
    critic_state = critic_state.apply_gradients(grads=grads)
    return (critic_state, loss)

def update_actor(actor_state, critic_state, transitions, advantages, targets, last_observation, hyperparams, clip, vanilla=False):
    """Calculates and applies the PPO gradient estimator at each time step."""
    def ppo_loss(params, transitions, advantages):
        """Calculates the clipped advantage estimator on a batch of transitions."""
        action_dists = jax.vmap(actor_state.apply_fn, in_axes=(None, 0))(params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        prob_ratios = jnp.exp(log_probs - transitions.log_prob)

        ppo_losses = prob_ratios * advantages
        clipped_ratios = jnp.clip(prob_ratios, 1 - clip, 1 + clip)
        clipped_losses = clipped_ratios * advantages

        ppo_losses = jnp.minimum(ppo_losses, clipped_losses)
        return -jnp.mean(ppo_losses)
    
    def leader_f2_loss(actor_params, critic_params):
        action_dists = jax.vmap(actor_state.apply_fn, in_axes=(None, 0))(actor_params, transitions.observation)
        log_probs = action_dists.log_prob(transitions.action)
        advantages, _ = calc_values(critic_state, critic_params, transitions, last_observation, hyperparams.discount_rate, advantage_rate = hyperparams.advantage_rate)
        result = advantages[:-1] * (hyperparams.discount_rate * advantages[1:] * log_probs[1:] - advantages[:-1] * log_probs[:-1])

        result = 2 * jnp.mean(result) 
        # result = clip_grad_norm(result, optax.global_norm(grad_theta_J))

        return result

    # Single Gradients
    vgj = jax.value_and_grad(ppo_loss)
    actor_loss, grad_theta_J = vgj(actor_state.params, transitions, advantages)

    def hypergrad():
        _, unflatten_fn = jax.flatten_util.ravel_pytree(critic_state.params)
        def nystrom_hvp(rank, rho, lambda_reg=0):
            # Step 1: Compute the gradients of the outer objective with respect to the inner and outer params
            # out_out_g = jax.grad(ppo_loss, argnums=0)(actor_state.params, transitions, advantages)
            out_in_g = jax.grad(leader_f2_loss, argnums=1)(actor_state.params, critic_state.params)
            
            # Step 2: Select random rows for low-rank approximation
            param_size = sum(x.size for x in jax.tree_util.tree_leaves(critic_state.params))
            indices = jax.random.permutation(jax.random.PRNGKey(0), param_size)[:rank]

            # Helper function to select Hessian rows using the inner objective
            def select_grad_row(in_params, indices):
                grad = jax.grad(lambda params: target_loss(params, transitions, targets, critic_state))(in_params)
                grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
                return grad_flat[indices]

            # Step 3: Approximate Hessian rows using the selected indices
            hessian_rows = jax.jacrev(select_grad_row)(critic_state.params, indices)

            # Flattening the hessian rows pytree into a single vector for concatenation
            hessian_rows_flat, _ = jax.flatten_util.ravel_pytree(hessian_rows)

            # Step 4: Concatenate the rows to form matrix C
            C = jnp.reshape(hessian_rows_flat, (rank, -1))

            # Step 5: Apply the Woodbury matrix identity to compute the inverse Hessian-vector product
            M = C.take(indices, axis=1)
            v_flat, _ = jax.flatten_util.ravel_pytree(out_in_g)
            
            # Compute the inverse Hessian-vector product using Woodbury identity
            x = (1 / (rho )) * v_flat - (1 / ((rho ) ** 2)) * C.T @ jax.scipy.linalg.solve(M + (1 / rho) * C @ C.T +  jnp.eye(M.shape[0]), C @ v_flat)

            # Step 6: Ensure the result is a flat vector that can be unraveled
            return x 
        
        inverse_hvp_flat = nystrom_hvp(hyperparams.nystrom_rank, hyperparams.nystrom_rho)
        inverse_hvp = unflatten_fn(inverse_hvp_flat)

        # 6. Compute mixed gradient and its transpose: [∇²_θ,ν V_s(ν, θ*(ν))]^T
        def mixed_grad_fn(policy_params, critic_params):
            return jax.grad(leader_f2_loss)(policy_params, critic_params)

        # 7. Compute the final product: [∇²_θ,ν V_s(ν, θ*(ν))]^T * [∇²_θ V_s(ν, θ*(ν))]^(-1) * ∇_θ L_pref(ν)
        # We use JVP to compute this product efficiently
        _, final_product = jax.jvp(
            lambda p: mixed_grad_fn(actor_state.params, p),
            (critic_state.params,),
            (inverse_hvp,)
        )

        hypergradient = jax.tree_util.tree_map(lambda x, y: x - y, grad_theta_J, final_product)

        hypergradient_norms = optax.global_norm(hypergradient)
        final_product_norms = optax.global_norm(final_product)
        co_sim = cosine_similarity(final_product, grad_theta_J)

        return (hypergradient, hypergradient_norms, final_product_norms, co_sim)
    
    def vanilla_grad():
        return (grad_theta_J, 0.0, 0.0, 0.0)

    total_gradient, hypergradient_norms, final_product_norms, co_sim = jax.lax.cond(
        vanilla,
        true_fun=lambda : vanilla_grad(),
        false_fun=lambda : hypergrad()
    )
    actor_state = actor_state.apply_gradients(grads=total_gradient)

    return (actor_state, (hypergradient_norms, final_product_norms, co_sim), actor_loss)

    # actor_state = actor_state.apply_gradients(grads=grad_theta_J)
    # return (actor_state, (0,0,0), actor_loss)

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

################################# Batch Update Functions ############################
def run_update(env, env_params, actor_state, critic_state, rng_key, hyperparams):
    """Runs an iteration of the training loop with the vanilla parallel update."""
    # Collect transitions and calculate advantages and targets
    rng_key, rollout_key, shuffle_key = jax.random.split(rng_key, 3)
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

    # # Process last_observations
    # last_observations = []
    # for i in range(len(advantages)):
    #     if(i<len(advantages)):
    #         last_observations.append(transitions[i+1][0].observation)
    #     else:
    #         last_observations.append(last_observation)

    # Train the actor and critic on each minibatch
    def update_minibatch(train_state, batch_info):
        actor_state, critic_state = train_state
        transitions, advantages, targets = batch_info
        actor_state, actor_info, actor_loss = update_actor(actor_state, transitions, advantages, hyperparams.actor_clip)
        critic_state, critic_loss = update_critic(critic_state, transitions, targets)
        return ((actor_state, critic_state), (actor_loss, actor_info, critic_loss))

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
        actor_losses, actor_info, critic_losses = losses
        actor_loss += jnp.mean(actor_losses) / hyperparams.update_epochs
        critic_loss += jnp.mean(critic_losses) / hyperparams.update_epochs

    return (
        actor_state,
        critic_state,
        (average_reward, actor_loss, actor_info, critic_loss),
        rng_key,
    )

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

def train(env_key, seed, logger, hyperparams, verbose = False, vanilla=False):
    # Create environment
    config = ENV_CONFIG[env_key]
    hyperparams = config["hyperparams"]
    rng_key, actor_key, critic_key = jax.random.split(jax.random.key(seed), 3)
    env, env_params = gymnax.make(GYM_ENV_NAMES[env_key])
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
    logger.set_interval(hyperparams.rollout_len)
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
    num_batches = int(hyperparams.num_updates / hyperparams.batch_count)
    for b in range(num_batches):
        actor_state, critic_state, batch_metrics, rng_key = \
            run_batch(env, env_params, actor_state, critic_state, rng_key, hyperparams)
        logger.log_metrics({
            "reward": batch_metrics[0],
            "actor_loss": batch_metrics[1],
            "critic_loss": batch_metrics[2],
        })
        if verbose:
            print(f"[Update {(b + 1) * hyperparams.batch_count}]: Average reward {batch_metrics[0][-1]}")