import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import wandb
from core.wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)
from core.model import DiscreteActor, Critic, ContinuousActor
from core.utilities import initialize_config, linear_schedule, run_name, cosine_similarity
import argparse

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    """
    Fisher Information Matrix Preconditioned Conjugate Gradient for Bi-level RL
    
    This method uses the Fisher Information Matrix (FIM) as a preconditioner for the 
    conjugate gradient solver in bi-level optimization. The FIM captures the geometric
    structure of the parameter space and provides natural gradient-like updates.
    
    Theoretical Justification:
    - Fisher Information Matrix F = E[∇log π(a|s) ∇log π(a|s)^T] captures parameter sensitivity
    - Natural gradient F^-1 ∇θ J provides parameter-invariant updates
    - Using F as preconditioner: solve (H + λI)x = b → F^-1 H x = F^-1 b
    - Improves convergence especially when parameter space has different curvatures
    - Maintains connection to natural policy gradients while solving bi-level problem
    
    The Fisher matrix preconditioner is particularly well-motivated in RL because:
    1. It respects the probabilistic structure of the policy
    2. It provides scale-invariant updates across different parameter types
    3. It's related to the KL divergence metric on the policy manifold
    4. It can be efficiently approximated using K-FAC or diagonal approximations
    """

    #### Prepare some hyperparameters ###
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    initialize_config(cfg=config)

    ###Initialize Environment ###
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def actor_linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["actor-LR"] * frac
    def critic_linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["critic-LR"] * frac

    def train(rng):
        ### INIT NETWORK ###
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        empty_observation = jnp.zeros(env.observation_space(env_params).shape)

        actor_network = ContinuousActor(env.action_space(env_params).shape[0], activation = config["ACTIVATION"])
        actor_params = actor_network.init(actor_rng, empty_observation)
        actor_state = TrainState.create(
            apply_fn = actor_network.apply,
            params = actor_params, 
            tx = optax.adam(learning_rate=actor_linear_schedule if config["ANNEAL_LR"] else config["actor-LR"], eps=1e-5)
        )
        critic_network = Critic(activation=config["ACTIVATION"])
        critic_params = critic_network.init(critic_rng, empty_observation)
        critic_state = TrainState.create(
            apply_fn = critic_network.apply, 
            params = critic_params, 
            tx = optax.adam(learning_rate=critic_linear_schedule if config["ANNEAL_LR"] else config["critic-LR"], eps=1e-5)
        )
        
        ### Parraleled Environments ###
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        
        ################################ Start Training ##########################
        # TRAIN LOOP
        def _update_step(runner_state, unused):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                actor_state, critic_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                pi = actor_network.apply(actor_state.params, last_obs)
                value = critic_network.apply(critic_state.params, last_obs)

                rng, actor_rng = jax.random.split(rng)
                action = pi.sample(seed=actor_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
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
                    def ppo_loss(actor_params, critic_params, transitions):
                        """Calculates the clipped advantage estimator on a batch of transitions."""
                        advantages, _ = calculate_gae(critic_params, transitions, last_obs)

                        action_dists = actor_network.apply(actor_params, transitions.obs)
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
                    
                    def leader_f2_loss(actor_params, critic_params, transitions, targets):
                        advantages, _ = calculate_gae(critic_params, transitions, last_obs)

                        action_dists = actor_network.apply(actor_params, transitions.obs)
                        log_probs = action_dists.log_prob(transitions.action)
                        prob_ratios = jnp.exp(log_probs - transitions.log_prob)

                        unclipped_losses = prob_ratios * advantages
                        clipped_ratios = jnp.clip(prob_ratios, 1 - config["CLIP_F"], 1 + config["CLIP_F"])
                        clipped_losses = clipped_ratios * advantages
                        losses = jnp.minimum(unclipped_losses, clipped_losses)

                        def _get_cummulate(carry, loss):
                            length, total = carry
                            length += 1
                            total += loss
                            return (length, total), - total / length

                        _, ppo_losses = jax.lax.scan(_get_cummulate, (0, 0.0), losses)

                        values = jax.vmap(critic_network.apply, in_axes=(None, 0))(critic_params, transitions.obs)
                        
                        return 2 * jnp.mean((targets - values) * ppo_losses)

                    ### Update the critic state for several epochs ###
                    for _ in range(config["nested_updates"]):
                        critic_loss, critic_grad = jax.value_and_grad(critic_target_loss)(critic_state.params, traj_batch, targets)
                        critic_state = critic_state.apply_gradients(grads=critic_grad)

                    ### Compute Fisher Information Matrix and use as preconditioner ###
                    def compute_fisher_information_matrix(actor_params, transitions):
                        """
                        Compute the Fisher Information Matrix for the policy
                        F = E[∇log π(a|s) ∇log π(a|s)^T]
                        
                        For efficiency, we use either:
                        1. Diagonal Fisher (simplest but less accurate)
                        2. K-FAC approximation (Kronecker factored)
                        3. Empirical Fisher computed on current batch
                        """
                        # Compute log probability gradients
                        def log_prob_grad(params, obs, action):
                            def log_prob_fn(p):
                                dist = actor_network.apply(p, obs)
                                return dist.log_prob(action)
                            return jax.grad(log_prob_fn)(params)
                        
                        # Vectorized computation over batch
                        log_prob_grads = jax.vmap(log_prob_grad, in_axes=(None, 0, 0))(
                            actor_params, transitions.obs, transitions.action
                        )
                        
                        if config["fisher_type"] == "diagonal":
                            # Diagonal Fisher: F_ii = E[(∇_i log π)²]
                            def compute_diagonal_fisher(grads):
                                squared_grads = jax.tree_util.tree_map(lambda x: jnp.square(x), grads)
                                # Average over batch
                                batch_size = transitions.obs.shape[0]
                                mean_squared = jax.tree_util.tree_map(
                                    lambda x: jnp.mean(x, axis=0), squared_grads
                                )
                                return mean_squared
                            
                            return compute_diagonal_fisher(log_prob_grads)
                        
                        elif config["fisher_type"] == "empirical":
                            # Empirical Fisher: F = (1/N) Σ ∇log π ∇log π^T 
                            # For computational efficiency, we'll use the Gauss-Newton approximation
                            # which is equivalent to the Fisher for exponential family models
                            
                            # Flatten gradients for easier manipulation
                            flat_grads = jax.vmap(lambda g: jax.flatten_util.ravel_pytree(g)[0])(log_prob_grads)
                            batch_size = flat_grads.shape[0]
                            
                            # Compute empirical Fisher: F = (1/N) G^T G where G is the gradient matrix
                            fisher_matrix = (1.0 / batch_size) * jnp.dot(flat_grads.T, flat_grads)
                            
                            # Add regularization for numerical stability
                            fisher_matrix += config["fisher_reg"] * jnp.eye(fisher_matrix.shape[0])
                            
                            return fisher_matrix, jax.flatten_util.ravel_pytree(actor_params)[1]
                        
                        elif config["fisher_type"] == "kfac":
                            # K-FAC approximation: approximate Fisher as Kronecker product
                            # This is more complex but highly efficient for large networks
                            # For simplicity, we implement a basic version here
                            
                            # This would typically involve layer-wise Kronecker factorization
                            # For now, fall back to diagonal approximation
                            return compute_diagonal_fisher(log_prob_grads)
                    
                    ### Update actor using Fisher-preconditioned conjugate gradient ###
                    actor_loss, grad_theta_J = jax.value_and_grad(ppo_loss)(actor_state.params, critic_state.params, traj_batch)
                    grad_w_J = jax.grad(ppo_loss, 1)(actor_state.params, critic_state.params, traj_batch)

                    # Compute Fisher Information Matrix
                    fisher_info = compute_fisher_information_matrix(actor_state.params, traj_batch)
                    
                    # Fisher-preconditioned conjugate gradient solution
                    def fisher_preconditioned_cg_solve(b):
                        """
                        Solves (H + λI)x = b using Fisher Information Matrix preconditioning
                        
                        The Fisher preconditioner F approximates the natural metric on the policy manifold.
                        We solve: F^-1 H x = F^-1 b, which has better conditioning when H and F 
                        have similar eigenspace structure.
                        """
                        
                        if config["fisher_type"] == "diagonal":
                            # Use diagonal Fisher as preconditioner
                            def diagonal_fisher_preconditioner(v):
                                """Apply F^-1 to vector v using diagonal Fisher"""
                                return jax.tree_util.tree_map(
                                    lambda v_i, f_i: v_i / (f_i + config["fisher_reg"]), v, fisher_info
                                )
                        
                        elif config["fisher_type"] == "empirical":
                            # Use full empirical Fisher matrix
                            fisher_matrix, unflatten_fn = fisher_info
                            
                            def empirical_fisher_preconditioner(v):
                                """Apply F^-1 to vector v using empirical Fisher matrix"""
                                v_flat, _ = jax.flatten_util.ravel_pytree(v)
                                # Solve F y = v for y
                                preconditioned_flat = jax.scipy.linalg.solve(fisher_matrix, v_flat)
                                return unflatten_fn(preconditioned_flat)
                        
                        else:  # kfac or fallback
                            # Use diagonal approximation as fallback
                            def diagonal_fisher_preconditioner(v):
                                return jax.tree_util.tree_map(
                                    lambda v_i, f_i: v_i / (f_i + config["fisher_reg"]), v, fisher_info
                                )
                        
                        # Select the appropriate preconditioner
                        if config["fisher_type"] == "empirical":
                            fisher_preconditioner = empirical_fisher_preconditioner
                        else:
                            fisher_preconditioner = diagonal_fisher_preconditioner
                        
                        # Hessian-vector product function
                        def hvp(v):
                            critic_params_flat, unravel_fn = jax.flatten_util.ravel_pytree(critic_state.params)
                            v_flat, _ = jax.flatten_util.ravel_pytree(v)
                            def loss_grad_flat(p):
                                return jax.flatten_util.ravel_pytree(
                                    jax.grad(critic_target_loss, argnums=0)(unravel_fn(p), traj_batch, targets)
                                )[0]
                            hvp_result = jax.jvp(loss_grad_flat, (critic_params_flat,), (v_flat,))[1] + config["lambda_reg"] * v_flat
                            return unravel_fn(hvp_result)
                        
                        # Fisher-Preconditioned Conjugate Gradient
                        def fisher_pcg(rhs, max_iter=25, tol=1e-8):
                            x = jax.tree_util.tree_map(jnp.zeros_like, rhs)
                            r = jax.tree_util.tree_map(lambda a, b: a - b, rhs, hvp(x))
                            z = fisher_preconditioner(r)
                            p = z
                            rsold = sum(jnp.sum(r_i * z_i) for r_i, z_i in zip(
                                jax.tree_util.tree_leaves(r), jax.tree_util.tree_leaves(z)))
                            
                            def pcg_step(carry, _):
                                x, r, p, rsold = carry
                                Ap = hvp(p)
                                pAp = sum(jnp.sum(p_i * Ap_i) for p_i, Ap_i in zip(
                                    jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(Ap)))
                                alpha = rsold / (pAp + 1e-10)
                                
                                x = jax.tree_util.tree_map(lambda x_i, p_i: x_i + alpha * p_i, x, p)
                                r = jax.tree_util.tree_map(lambda r_i, Ap_i: r_i - alpha * Ap_i, r, Ap)
                                z = fisher_preconditioner(r)
                                rsnew = sum(jnp.sum(r_i * z_i) for r_i, z_i in zip(
                                    jax.tree_util.tree_leaves(r), jax.tree_util.tree_leaves(z)))
                                
                                beta = rsnew / (rsold + 1e-10)
                                p = jax.tree_util.tree_map(lambda z_i, p_i: z_i + beta * p_i, z, p)
                                
                                return (x, r, p, rsnew), rsnew
                            
                            (x, _, _, _), residuals = jax.lax.scan(pcg_step, (x, r, p, rsold), None, length=max_iter)
                            return x
                        
                        return fisher_pcg(b)
                    
                    # Solve for inverse Hessian-vector product using Fisher preconditioning
                    inverse_hvp = fisher_preconditioned_cg_solve(grad_w_J)

                    # Compute mixed gradient and final product
                    def mixed_grad_fn(policy_params, critic_params):
                        return jax.grad(leader_f2_loss)(policy_params, critic_params, traj_batch, targets)

                    _, final_product = jax.jvp(
                        lambda p: mixed_grad_fn(actor_state.params, p),
                        (critic_state.params,),
                        (inverse_hvp,)
                    )
                    
                    # Bound the final product for stability
                    grad_theta_J_norm = optax.global_norm(grad_theta_J)
                    final_product_norm = optax.global_norm(final_product)
                    max_norm = config["IHVP_BOUND"] * grad_theta_J_norm
                    scaling_factor = jnp.minimum(1.0, max_norm/(final_product_norm + 1e-8))
                    clipped_final_product = jax.tree_util.tree_map(lambda fp: fp * scaling_factor, final_product)

                    # Compute hypergradient (natural gradient-like update due to Fisher preconditioning)
                    hypergradient = jax.tree_util.tree_map(lambda x, y: x - y, grad_theta_J, clipped_final_product)
                    actor_state = actor_state.apply_gradients(grads=hypergradient)

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
                batch = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), batch)
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
            
            # Can add printing statement here.
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric)

            runner_state = (actor_state, critic_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (actor_state, critic_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metrics": metric}
    return train


if __name__ == "__main__":
    # Configuration with Fisher Information Matrix specific parameters
    config = {
        "NUM_ENVS": 32,
        "NUM_STEPS": 640,
        "TOTAL_TIMESTEPS": 1e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "CLIP_F": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "hopper",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
        "actor-LR": 3e-4,
        "critic-LR": 1e-3,

        # Fisher Information Matrix preconditioner parameters
        "fisher_type": "diagonal",  # Options: "diagonal", "empirical", "kfac"
        "fisher_reg": 1e-4,  # Regularization for Fisher matrix (numerical stability)
        "nested_updates": 3,
        "IHVP_BOUND": 0.2,
        "lambda_reg": 1e-3,  # Hessian regularization

        "vanilla": False,
    }

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng) 