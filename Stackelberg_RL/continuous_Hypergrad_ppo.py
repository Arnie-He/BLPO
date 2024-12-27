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
from core.utilities import initialize_config, linear_schedule, logdir, cosine_similarity
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

    #### Prepare some hyperparameters ###
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    initialize_config(cfg=config)

    ### Weight and Bias Setup ###
    wandb.init(project="HyperGradient-RL", config = config)
    wandb.define_metric("Reward", summary="mean")

    ###Initialize Environment ###
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def train(rng):
        ### INIT NETWORK ###
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        empty_observation = jnp.zeros(env.observation_space(env_params).shape)

        actor_network = ContinuousActor(env.action_space(env_params).shape[0], activation = config["ACTIVATION"])
        actor_params = actor_network.init(actor_rng, empty_observation)
        actor_state = TrainState.create(
            apply_fn = actor_network.apply,
            params = actor_params, 
            tx = optax.adam(learning_rate=config["actor-LR"], eps=1e-5),
        )
        critic_network = Critic(activation=config["ACTIVATION"])
        critic_params = critic_network.init(critic_rng, empty_observation)
        critic_state = TrainState.create(
            apply_fn = critic_network.apply, 
            params = critic_params, 
            tx = optax.adam(learning_rate=config["critic-LR"], eps=1e-5),
        )
        
        ### Parraleled Environments ###
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        
        ################################ Start Training ##########################
        # TRAIN LOOP
        flush_counter = 0

        def _update_step(runner_state, unused):
            nonlocal flush_counter # Allow modification of the outer counter

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
                # jax.debug.print("traj_batch_values shape {}", traj_batch_values.shape)
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
            # adv_only = lambda p: calculate_gae(p, traj_batch, last_obs)[0]
            # grad_w_adv = jax.grad(adv_only)(critic_state.params)
            critic_p = jax.tree_util.tree_map(
                    lambda x: jnp.copy(x), jax.lax.stop_gradient(critic_state.params)
                )

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    actor_state, critic_state = train_state 
                    traj_batch, advantages, targets, last_obs = batch_info

                    # in_gae, _ = calculate_gae(critic_p, traj_batch, last_obs)
                    # copied_gae, _ = calculate_gae(critic_state.params, traj_batch, last_obs)
                    # gae_norm = optax.global_norm(advantages)
                    # in_gae_norm = optax.global_norm(in_gae)
                    # jax.debug.print("traj_batch shape is {}", traj_batch.obs.shape)
                    # jax.debug.print("out is {} while in is {}, copied is {}", gae_norm, in_gae_norm, optax.global_norm(copied_gae))
                    # jax.debug.print("in gae shape {}", in_gae.shape)
                    # jax.debug.print("out gae shape {}", advantages.shape)

                    ############ Define loss functions ##############
                    def ppo_loss(actor_params, critic_params, transitions):
                        """Calculates the clipped advantage estimator on a batch of transitions."""
                        advantages, _ = calculate_gae(critic_params, traj_batch, last_obs)

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
                    
                    def leader_f2_loss(actor_params, critic_params, transitions):
                        # action_dists = jax.vmap(actor_network.apply, in_axes=(None, 0))(actor_params, transitions.obs)
                        # log_probs = action_dists.log_prob(transitions.action)
                        # prob_ratios = jnp.exp(log_probs - transitions.log_prob)
                        # clipped_ratios = jnp.clip(prob_ratios, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                        # advantages, _ = calculate_gae(critic_params, transitions, last_obs=last_obs)
                        # result = advantages[:-1] * (config["GAMMA"] * advantages[1:] * log_probs[1:] - advantages[:-1] * log_probs[:-1])
                        # clipped_result = advantages[:-1] * (config["GAMMA"] * advantages[1:] * clipped_ratios[1:] - advantages[:-1] * clipped_ratios[:-1])
                        # f2_loss = jnp.minimum(result, clipped_result)
                        # return 2 * jnp.mean(f2_loss)
                        advantages, _ = calculate_gae(critic_params, traj_batch, last_obs)

                        action_dists = actor_network.apply(actor_params, transitions.obs)
                        log_probs = action_dists.log_prob(transitions.action)
                        prob_ratios = jnp.exp(log_probs - transitions.log_prob)

                        advantage_losses = prob_ratios * advantages
                        clipped_ratios = jnp.clip(prob_ratios, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                        clipped_losses = clipped_ratios * advantages

                        ppo_losses = jnp.minimum(advantage_losses, clipped_losses)
                        return 2 * -jnp.mean(ppo_losses)

                    ### update actor for config["nested_updates"] times ###
                    actor_loss, grad_theta_J = jax.value_and_grad(ppo_loss)(actor_state.params, critic_p, traj_batch)

                    def hypergrad():
                        _, unflatten_fn = jax.flatten_util.ravel_pytree(critic_state.params)
                        """Time-efficient Nystrom"""
                        def nystrom_hvp(rank, rho):
                            # Use critic_p or critic_state.params?
                            in_out_g = jax.grad(ppo_loss, argnums=1)(actor_state.params, critic_state.params, traj_batch)
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
                            v_flat, _ = jax.flatten_util.ravel_pytree(in_out_g)
                            x = (1 / (rho )) * v_flat - (1 / ((rho ) ** 2)) * C.T @ jax.scipy.linalg.solve(M + (1 / rho) * C @ C.T +  jnp.eye(M.shape[0]), C @ v_flat)
                            return x
                        # """Space-efficient Nystrom"""
                        # def nystrom_se(rnak, rho):
                        #     out_in_g = jax.grad(critic_target_loss, argnums=1)(actor_state.params, critic_state.params, traj_batch)
                        #     param_size = sum(x.size for x in jax.tree_util.tree_leaves(critic_state.params))
                                                   
                        # compute the ihvp using nystrom
                        inverse_hvp_flat = nystrom_hvp(config["nystrom_rank"], config["nystrom_rho"])
                        inverse_hvp = unflatten_fn(inverse_hvp_flat)
                        def mixed_grad_fn(policy_params, critic_params):
                            return jax.grad(leader_f2_loss)(policy_params, critic_params, traj_batch)
                        _, final_product = jax.jvp(
                            lambda p: mixed_grad_fn(actor_state.params, p),
                            (critic_p,),
                            (inverse_hvp,)
                        )
                        # bound the final_product
                        grad_theta_J_norm = optax.global_norm(grad_theta_J)
                        final_product_norm = optax.global_norm(final_product)
                        max_norm = config["IHVP_BOUND"] * grad_theta_J_norm
                        scaling_factor = jnp.minimum(1.0, max_norm/(final_product_norm + 1e-8))
                        clipped_final_product = jax.tree_util.tree_map( lambda fp: fp * scaling_factor, final_product)

                        hypergradient = jax.tree_util.tree_map(lambda x, y: x - y, grad_theta_J, clipped_final_product)

                        hypergradient_norms = optax.global_norm(hypergradient)
                        final_product_norms = optax.global_norm(final_product)
                        co_sim = cosine_similarity(final_product, grad_theta_J)
                        return (hypergradient, hypergradient_norms, final_product_norms, co_sim)
                    def vanilla_grad():
                        return (grad_theta_J, 0.0, 0.0, 0.0)
                    total_gradient, hypergradient_norms, final_product_norms, co_sim = jax.lax.cond(
                        config["vanilla"],
                        true_fun=lambda : vanilla_grad(),
                        false_fun=lambda : hypergrad()
                    )
                    actor_state = actor_state.apply_gradients(grads=total_gradient)

                    ### Update the critic state for several epoch ###
                    for _ in range(config["nested_updates"]):
                        critic_loss, critic_grad = jax.value_and_grad(critic_target_loss)(critic_state.params, traj_batch, targets)
                        critic_state = critic_state.apply_gradients(grads=critic_grad)

                    total_loss = actor_loss + critic_loss
                    train_state = (actor_state, critic_state)
                    return train_state, total_loss
                
                actor_state, critic_state, traj_batch, advantages, targets, rng = update_state

                # jax.debug.print("##################################")
                # jax.debug.print("advantage norm is {}", optax.global_norm(advantages))
                # lladv, _ = calculate_gae(critic_state.params, traj_batch, last_obs)
                # jax.debug.print("llladv is {}", optax.global_norm(lladv))
                # jax.debug.print("##################################")

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

                # TJB, ADV, TAR, LOBS = minibatches
                # jax.debug.print("##################################")
                # # jax.debug.print("advantage norm is {}", optax.global_norm(advantages))
                # lllllllllllladv = calculate_gae(critic_state.params, TJB, LOBS)
                # jax.debug.print("lllllllllllladv is {}", optax.global_norm(lllllllllllladv))
                # jax.debug.print("##################################")

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
                        wandb.log({"Reward": return_values[t]}, step=timesteps[t])
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
    # logging.basicConfig(filename='ppo.log', level=logging.INFO, format='%(message)s')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--vanilla", type=bool, default=True, help="Use Vanilla setting")
    # args = parser.parse_args()

   
    # Original configuration
    original_config = {
        "NUM_ENVS": 32,
        "NUM_STEPS": 640,
        "TOTAL_TIMESTEPS": 3e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
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
        "nystrom_rank": 5,
        "nystrom_rho": 50,
        "nested_updates": 3,
        "IHVP_BOUND": 0.2,

        #"vanilla": args.vanilla,

    }

    # Define the sweep configuration
    sweep_config = {
        "method": "bayes",  # Choose from "grid", "random", "bayes"
        "metric": {
            "name": "Reward.mean",
            "goal": "maximize"
        },
        "parameters": {
            "actor-LR": {
                "values": [3e-4, 1e-3, 3e-3]
            },
            "critic-LR": {
                "values": [1e-3, 5e-4, 1e-4]
            },
            "GAE_LAMBDA": {
                "values": [0.95, 0.9, 0.99]
            },
            "vanilla": {
                "values": [True, False]
            },
            "nested_updates": {
                "values": [3, 5, 7]
            }
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="jax-ppo")

    # Function to merge wandb.config with original_config
    def train_with_wandb():
        wandb.init()  # Initialize wandb and access the sweep parameters
        sweep_config = wandb.config

        # Merge original config with sweep config
        merged_config = {**original_config, **sweep_config}

        # Call your training function with the merged configuration
        train_jit = jax.jit(make_train(merged_config))
        rng = jax.random.PRNGKey(30)

        # Run the training loop
        out = train_jit(rng)
        metrics = out["metrics"]

        wandb.log({"Final Reward": metrics})

    # Run the sweep
    wandb.agent(sweep_id, function=train_with_wandb, count=10)



