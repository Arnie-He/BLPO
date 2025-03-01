import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from core.wrappers import LogWrapper, FlattenObservationWrapper
from core.model import DiscreteActor, Critic
from core.utilities import initialize_config, linear_schedule, cosine_similarity
import logging 
import os
from tensorboardX import SummaryWriter
import datetime

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    
def make_train(config, vanilla):

    #### Prepare some hyperparameters ###
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    initialize_config(cfg=config)

    ### TensorBoard Setup ###
    log_dir = os.path.join("runs", config["ENV_NAME"], datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir)
    print(f"Logging to TensorBoard at: {log_dir}")

    ###Initialize Environment ###
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(rng):
        ### INIT NETWORK ###
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        empty_observation = jnp.zeros(env.observation_space(env_params).shape)

        actor_network = DiscreteActor(env.action_space(env_params).n, activation = config["ACTIVATION"])
        actor_params = actor_network.init(actor_rng, empty_observation)
        actor_state = TrainState.create(
            apply_fn = actor_network.apply,
            params = actor_params, 
            tx = optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        critic_network = Critic(activation=config["ACTIVATION"])
        critic_params = critic_network.init(critic_rng, empty_observation)
        critic_state = TrainState.create(
            apply_fn = critic_network.apply, 
            params = critic_params, 
            tx = optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )

        ### Parraleled Environments ###
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng)
        
        ################################ Start Training ##########################
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                actor_state, critic_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                pi = actor_network.apply(actor_state.params, last_obs)

                rng, actor_rng = jax.random.split(rng)
                action = pi.sample(seed=actor_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                # temporarily hold the values field.
                transition = Transition(
                    done, action, reward, log_prob, last_obs,  info
                )
                runner_state = (actor_state, critic_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGEs
            def calculate_gae(critic_params, traj_batch, last_obs):

                traj_batch_values = jax.vmap(critic_network.apply, in_axes=(None, 0))(critic_params, traj_batch.obs)
                last_val = critic_network.apply(critic_params, last_obs)

                def _get_advantages(gae_and_next_value, value_info):
                    gae, next_value = gae_and_next_value
                    transition, value = value_info
                    done, reward = (
                        transition.done,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    (traj_batch, traj_batch_values),
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch_values
            
            actor_state, critic_state, env_state, last_obs, rng = runner_state
            advantages, targets = calculate_gae(critic_state.params, traj_batch, last_obs)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    actor_state, critic_state = train_state 
                    traj_batch, advantages, targets, last_obs = batch_info
                    ############ Define loss functions ##############
                    def actor_loss(params, transitions, advantages):
                        """Calculates the clipped advantage estimator on a batch of transitions."""
                        action_dists = jax.vmap(actor_network.apply, in_axes=(None, 0))(params, transitions.obs)
                        log_probs = action_dists.log_prob(transitions.action)
                        prob_ratios = jnp.exp(log_probs - transitions.log_prob)

                        advantage_losses = prob_ratios * advantages
                        clipped_ratios = jnp.clip(prob_ratios, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                        clipped_losses = clipped_ratios * advantages

                        ppo_losses = jnp.minimum(advantage_losses, clipped_losses)
                        return -jnp.mean(ppo_losses)
                    
                    def critic_target_loss(params, transitions, targets):
                        """Calculates the mean squared error on a batch of transitions."""
                        values = jax.vmap(critic_network.apply, in_axes=(None, 0))(params, transitions.obs)
                        errors = jnp.square(targets - values)
                        return jnp.mean(errors)
                    
                    def leader_f2_loss(actor_params, critic_params, transitions):
                        action_dists = jax.vmap(actor_network.apply, in_axes=(None, 0))(actor_params, transitions.obs)
                        log_probs = action_dists.log_prob(transitions.action)
                        prob_ratios = jnp.exp(log_probs - transitions.log_prob)
                        clipped_ratios = jnp.clip(prob_ratios, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                        advantages, _ = calculate_gae(critic_params, transitions, last_obs=last_obs)
                        result = advantages[:-1] * (config["GAMMA"] * advantages[1:] * log_probs[1:] - advantages[:-1] * log_probs[:-1])
                        clipped_result = advantages[:-1] * (config["GAMMA"] * advantages[1:] * clipped_ratios[1:] - advantages[:-1] * clipped_ratios[:-1])
                        f2_loss = jnp.minimum(result, clipped_result)
                        return 2 * jnp.mean(f2_loss) 
                    
                    ### Update the critic state for one epoch ###
                    critic_loss, critic_grad = jax.value_and_grad(critic_target_loss)(critic_state.params, traj_batch, targets)
                    critic_state = critic_state.apply_gradients(grads=critic_grad)

                    ### update actor for config["nested_updates"] times ###
                    actor_loss, grad_theta_J = jax.value_and_grad(actor_loss)(actor_state.params, traj_batch, advantages)
                    def hypergrad():
                        _, unflatten_fn = jax.flatten_util.ravel_pytree(critic_state.params)
                        def nystrom_hvp(rank, rho, lambda_reg=0):
                            out_in_g = jax.grad(leader_f2_loss, argnums=1)(actor_state.params, critic_state.params, traj_batch)
                            param_size = sum(x.size for x in jax.tree_util.tree_leaves(critic_state.params))
                            indices = jax.random.permutation(jax.random.PRNGKey(0), param_size)[:rank]
                            def select_grad_row(in_params, indices):
                                grad = jax.grad(lambda params: critic_target_loss(params, traj_batch, targets))(in_params)
                                grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
                                return grad_flat[indices]
                            hessian_rows = jax.jacrev(select_grad_row)(critic_state.params, indices)
                            hessian_rows_flat, _ = jax.flatten_util.ravel_pytree(hessian_rows)
                            C = jnp.reshape(hessian_rows_flat, (rank, -1))
                            M = C.take(indices, axis=1)
                            v_flat, _ = jax.flatten_util.ravel_pytree(out_in_g)
                            x = (1 / (rho )) * v_flat - (1 / ((rho ) ** 2)) * C.T @ jax.scipy.linalg.solve(M + (1 / rho) * C @ C.T +  jnp.eye(M.shape[0]), C @ v_flat)
                            return x
                        inverse_hvp_flat = nystrom_hvp(config["nystrom_rank"], config["nystrom_rho"])
                        inverse_hvp = unflatten_fn(inverse_hvp_flat)
                        def mixed_grad_fn(policy_params, critic_params):
                            return jax.grad(leader_f2_loss)(policy_params, critic_params, traj_batch)
                        _, final_product = jax.jvp(
                            lambda p: mixed_grad_fn(actor_state.params, p),
                            (critic_state.params,),
                            (inverse_hvp,)
                        )
                        # bound the final_product
                        grad_theta_J_norm = optax.global_norm(grad_theta_J)
                        final_product_norm = optax.global_norm(final_product)
                        max_norm = 0.1 * grad_theta_J_norm
                        scaling_factor = jnp.minimum(1.0, max_norm/(final_product_norm + 1e-8))
                        clipped_final_product = jax.tree_util.tree_map( lambda fp: fp * scaling_factor, final_product)

                        hypergradient = jax.tree_util.tree_map(lambda x, y: x - y, grad_theta_J, clipped_final_product)

                        hypergradient_norms = optax.global_norm(hypergradient)
                        final_product_norms = optax.global_norm(final_product)
                        co_sim = cosine_similarity(final_product, grad_theta_J)
                        return (hypergradient, hypergradient_norms, final_product_norms, co_sim)
                    def vanilla_grad():
                        return (grad_theta_J, 0.0, 0.0, 0.0)

                    # for _ in range(config["nested_updates"]):
                    total_gradient, hypergradient_norms, final_product_norms, co_sim = jax.lax.cond(
                        vanilla,
                        true_fun=lambda : vanilla_grad(),
                        false_fun=lambda : hypergrad()
                    )
                    actor_state = actor_state.apply_gradients(grads=total_gradient)

                    total_loss = actor_loss + critic_loss
                    train_state = (actor_state, critic_state)
                    return train_state, total_loss
                

                actor_state, critic_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Batching and Shuffling
                assert (
                    config["NUM_STEPS"] == config["MINIBATCH_SIZE"] and config["NUM_MINIBATCHES"] == config["NUM_ENVS"]
                ), "Number of envs must match number of minibatches and minibatches' length must match rollout len!"
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"]) + x.shape[2:]), batch
                )
                batch = (*batch, last_obs)
                permutation = jax.random.permutation(_rng, config["NUM_MINIBATCHES"])
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                train_state = (actor_state, critic_state)
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                actor_state, critic_state = train_state
                update_state = (actor_state, critic_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            # Updating Training State and Metrics:
            update_state = (actor_state, critic_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            actor_state = update_state[0]
            critic_state = update_state[1]
            metric = traj_batch.info
            rng = update_state[-1]
            
            # Debugging mode
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                        writer.add_scalar("episodic_return", return_values[t], timesteps[t])
                jax.debug.callback(callback, metric)

            runner_state = (actor_state, critic_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (actor_state, critic_state, env_state, obs, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metrics": metric}

    writer.close()
    return train


if __name__ == "__main__":
    os.environ["JAX_PLATFORM_NAME"] = "gpu"

    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 32,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "SpaceInvaders-MinAtar",
        "ANNEAL_LR": True,
        "DEBUG": True,

        "nystrom_rank": 10,
        "nystrom_rho": 50,
        "nested_updates": 10,
    }
    vanilla = False

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config, vanilla))
    out = train_jit(rng)
