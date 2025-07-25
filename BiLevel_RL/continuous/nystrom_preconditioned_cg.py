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
    Nystrom-Preconditioned Conjugate Gradient for Bi-level RL
    
    This method combines the conjugate gradient approach for solving the bi-level optimization
    problem with Nystrom preconditioning to accelerate convergence. The Nystrom method provides
    a low-rank approximation of the Hessian that serves as an effective preconditioner.
    
    Theoretical Justification:
    - Uses Nystrom approximation H_nys ≈ C K^-1 C^T where C contains selected Hessian rows
    - The preconditioner M = H_nys + λI improves the condition number of the linear system
    - Convergence rate of PCG is O(√κ(M^-1 H)) vs O(κ(H)) for standard CG
    - Maintains the exact solution property while significantly reducing iterations
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

                    ### Update actor using preconditioned conjugate gradient ###
                    actor_loss, grad_theta_J = jax.value_and_grad(ppo_loss)(actor_state.params, critic_state.params, traj_batch)
                    grad_w_J = jax.grad(ppo_loss, 1)(actor_state.params, critic_state.params, traj_batch)

                    # Nystrom-preconditioned conjugate gradient solution
                    def nystrom_preconditioned_cg_solve(b):
                        """
                        Solves (H + λI)x = b using Nystrom-preconditioned conjugate gradient
                        
                        The Nystrom preconditioner M approximates H using:
                        M = C K^-1 C^T + λI
                        where C contains selected rows of H and K = C[:, selected_cols]
                        """
                        critic_params_flat, unravel_fn = jax.flatten_util.ravel_pytree(critic_state.params)
                        param_size = critic_params_flat.shape[0]
                        
                        # Nystrom approximation parameters
                        rank = min(config["nystrom_rank"], param_size // 2)
                        rho = config["nystrom_rho"]
                        
                        # Select random indices for Nystrom approximation
                        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility during training
                        indices = jax.random.permutation(key, param_size)[:rank]
                        
                        def select_grad_row(params, idx):
                            """Select a specific row of the Hessian via finite differences"""
                            def single_grad(p):
                                grad = jax.grad(critic_target_loss, argnums=0)(p, traj_batch, targets)
                                grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
                                return grad_flat[idx]
                            return jax.grad(single_grad)(params)
                        
                        # Compute selected Hessian rows (this is the expensive part)
                        def compute_hessian_rows():
                            rows = []
                            for i in range(rank):
                                idx = indices[i]
                                row = select_grad_row(critic_state.params, idx)
                                row_flat, _ = jax.flatten_util.ravel_pytree(row)
                                rows.append(row_flat)
                            return jnp.stack(rows)
                        
                        C = compute_hessian_rows()  # rank x param_size matrix
                        K = C[:, indices]  # rank x rank matrix
                        
                        # Nystrom preconditioner: M^-1 ≈ (1/rho)I - (1/rho²)C^T (K + (1/rho)CC^T)^-1 C
                        CTC = C @ C.T
                        K_reg = K + (1/rho) * CTC + 1e-6 * jnp.eye(rank)  # Regularize for numerical stability
                        
                        def nystrom_preconditioner(v):
                            """Apply M^-1 to vector v"""
                            v_flat, unflatten = jax.flatten_util.ravel_pytree(v)
                            # M^-1 v = (1/rho)v - (1/rho²)C^T inv(K_reg) C v
                            Cv = C @ v_flat
                            term2 = C.T @ jax.scipy.linalg.solve(K_reg, Cv)
                            result = (1/rho) * v_flat - (1/rho**2) * term2
                            return unflatten(result)
                        
                        # Hessian-vector product function
                        def hvp(v):
                            v_flat, unravel_fn = jax.flatten_util.ravel_pytree(v)
                            def loss_grad_flat(p):
                                return jax.flatten_util.ravel_pytree(
                                    jax.grad(critic_target_loss, argnums=0)(unravel_fn(p), traj_batch, targets)
                                )[0]
                            hvp_result = jax.jvp(loss_grad_flat, (critic_params_flat,), (v_flat,))[1] + config["lambda_reg"] * v_flat
                            return unravel_fn(hvp_result)
                        
                        # Preconditioned Conjugate Gradient
                        def preconditioned_cg(rhs, max_iter=20, tol=1e-8):
                            x = jax.tree_util.tree_map(jnp.zeros_like, rhs)
                            r = jax.tree_util.tree_map(lambda a, b: a - b, rhs, hvp(x))
                            z = nystrom_preconditioner(r)
                            p = z
                            rsold = sum(jnp.sum(r_i * z_i) for r_i, z_i in zip(
                                jax.tree_util.tree_leaves(r), jax.tree_util.tree_leaves(z)))
                            
                            def cg_step(carry, _):
                                x, r, p, rsold = carry
                                Ap = hvp(p)
                                pAp = sum(jnp.sum(p_i * Ap_i) for p_i, Ap_i in zip(
                                    jax.tree_util.tree_leaves(p), jax.tree_util.tree_leaves(Ap)))
                                alpha = rsold / (pAp + 1e-10)
                                
                                x = jax.tree_util.tree_map(lambda x_i, p_i: x_i + alpha * p_i, x, p)
                                r = jax.tree_util.tree_map(lambda r_i, Ap_i: r_i - alpha * Ap_i, r, Ap)
                                z = nystrom_preconditioner(r)
                                rsnew = sum(jnp.sum(r_i * z_i) for r_i, z_i in zip(
                                    jax.tree_util.tree_leaves(r), jax.tree_util.tree_leaves(z)))
                                
                                beta = rsnew / (rsold + 1e-10)
                                p = jax.tree_util.tree_map(lambda z_i, p_i: z_i + beta * p_i, z, p)
                                
                                return (x, r, p, rsnew), rsnew
                            
                            (x, _, _, _), residuals = jax.lax.scan(cg_step, (x, r, p, rsold), None, length=max_iter)
                            return x
                        
                        return preconditioned_cg(b)
                    
                    # Solve for inverse Hessian-vector product
                    grad_w_J_flat, unflatten_fn = jax.flatten_util.ravel_pytree(grad_w_J)
                    inverse_hvp = nystrom_preconditioned_cg_solve(grad_w_J)

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

                    # Compute hypergradient
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
    # Configuration with Nystrom-specific parameters
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

        # Nystrom preconditioner parameters
        "nystrom_rank": 8,  # Rank for Nystrom approximation (higher = better approximation, more computation)
        "nystrom_rho": 10.0,  # Regularization parameter for Nystrom preconditioner
        "nested_updates": 3,
        "IHVP_BOUND": 0.2,
        "lambda_reg": 1e-3,  # Hessian regularization

        "vanilla": False,
    }

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng) 