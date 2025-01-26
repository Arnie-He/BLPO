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

    #### Prepare some hyperparameters ###
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    initialize_config(cfg=config)

    ### Weight and Bias Setup ###
    group_name = f'{config["Group"]}_{config["ENV_NAME"]}_CG'
    wandb.init(project="HyperGradient-RL", group= group_name, name=run_name(config), config = config)
    wandb.define_metric("Reward", summary="mean")

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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    actor_state, critic_state = train_state 
                    traj_batch, advantages, targets, last_obs = batch_info

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
                    
                    def leader_f2_loss(actor_params, critic_params, transitions, targets):
                        advantages, _ = calculate_gae(critic_params, traj_batch, last_obs)

                        action_dists = actor_network.apply(actor_params, transitions.obs)
                        log_probs = action_dists.log_prob(transitions.action)
                        prob_ratios = jnp.exp(log_probs - transitions.log_prob)

                        advantage_losses = prob_ratios * advantages
                        clipped_ratios = jnp.clip(prob_ratios, 1 - config["CLIP_F"], 1 + config["CLIP_F"])
                        clipped_losses = clipped_ratios * advantages

                        ppo_losses = jnp.minimum(advantage_losses, clipped_losses)

                        values = jax.vmap(critic_network.apply, in_axes=(None, 0))(critic_params, transitions.obs)

                        return 2 * -jnp.mean(jnp.dot(ppo_losses, (targets - values)))
                        # return 2 * -jnp.mean(ppo_losses)

                    ### Update the critic state for several epoch ###
                    for _ in range(config["nested_updates"]):
                        critic_loss, critic_grad = jax.value_and_grad(critic_target_loss)(critic_state.params, traj_batch, targets)
                        critic_state = critic_state.apply_gradients(grads=critic_grad)

                    ### update actor for 1 time ###
                    actor_loss, grad_theta_J = jax.value_and_grad(ppo_loss)(actor_state.params, critic_state.params, traj_batch)
                    grad_w_J = jax.grad(ppo_loss, 1)(actor_state.params, critic_state.params, traj_batch)

                    # jax.debug.print(f"lambda reg is {lambda_reg}")
                    def hvp(v):
                        critic_params_flat, unravel_fn = jax.flatten_util.ravel_pytree(critic_state.params)
                        def loss_grad_flat(p):
                            return jax.flatten_util.ravel_pytree(
                                jax.grad(critic_target_loss, argnums=0)(unravel_fn(p), traj_batch, targets)
                            )[0]
                        hvp = jax.jvp(loss_grad_flat, (critic_params_flat,), (v,))[1] + config["lambda_reg"] * v
                        return hvp
                    
                    grad_w_J_flat, unflatten_fn = jax.flatten_util.ravel_pytree(grad_w_J)
                    def cg_solve(v):
                        return jax.scipy.sparse.linalg.cg(hvp, v, maxiter=20, tol=1e-10)[0]
                    inverse_hvp_flat = cg_solve(grad_w_J_flat)
                    inverse_hvp = unflatten_fn(inverse_hvp_flat)

                    # 6. Compute mixed gradient and its transpose: [∇²_θ,ν V_s(ν, θ*(ν))]^T
                    def mixed_grad_fn(policy_params, critic_params):
                        return jax.grad(leader_f2_loss)(policy_params, critic_params, traj_batch, targets)

                    # 7. Compute the final product: [∇²_θ,ν V_s(ν, θ*(ν))]^T * [∇²_θ V_s(ν, θ*(ν))]^(-1) * ∇_θ L_pref(ν)
                    # We use JVP to compute this product efficiently
                    _, final_product = jax.jvp(
                        lambda p: mixed_grad_fn(actor_state.params, p),
                        (critic_state.params,),
                        (inverse_hvp,)
                    )
                    
                    # projection 
                    # final_product = project_B_onto_A(grad_theta_J, final_product)

                    # vanilla = True
                    # final_product = clip_grad_norm(final_product, 0.2*optax.global_norm(grad_theta_J))
                    # bound the final_product
                    grad_theta_J_norm = optax.global_norm(grad_theta_J)
                    final_product_norm = optax.global_norm(final_product)
                    max_norm = config["IHVP_BOUND"] * grad_theta_J_norm
                    scaling_factor = jnp.minimum(1.0, max_norm/(final_product_norm + 1e-8))
                    clipped_final_product = jax.tree_util.tree_map(lambda fp: fp * scaling_factor, final_product)

                    hypergradient = jax.tree_util.tree_map(lambda x, y: x - y, grad_theta_J, clipped_final_product)
                    actor_state = actor_state.apply_gradients(grads=hypergradient)

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
                        #print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
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
    config = {
        "NUM_ENVS": 32,
        "NUM_STEPS": 640,
        "TOTAL_TIMESTEPS": 1e6,
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

        "vanilla": True,

    }

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
