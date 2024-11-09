import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Sequence
import gymnax
import matplotlib.pyplot as plt
from flax.linen.initializers import constant, orthogonal
import distrax
from tensorboardX import SummaryWriter
from gymnax.visualize import Visualizer


# Constants and hyperparameters
ENVIRONMENT = "CartPole-v1"
# NUM_ITERATIONS = 50
# NUM_EPISODES = 1000
# STEPS_PER_EPISODE = 200
# POLICY_LEARNING_RATE = 2e-4
# PREFERENCE_LEARNING_RATE = 1e-3
# DISCOUNT_FACTOR = 1.0
# BATCH_SIZE = 2000
# REINFORCE_EPISODES = 50000


#leader should be slower than the follower 
NUM_ITERATIONS = 50
NUM_EPISODES = 200
STEPS_PER_EPISODE = 200
POLICY_LEARNING_RATE = 1e-3
PREFERENCE_LEARNING_RATE = 2e-4
DISCOUNT_FACTOR = 1.0
BATCH_SIZE = 200
REINFORCE_EPISODES = 5000
BETA = 1.0
# Network definitions
class PolicyNetwork(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.action_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        return distrax.Categorical(logits=x)

class PreferenceNetwork(nn.Module):
    hidden_dims: Sequence[int] = (16, 16)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, trajectory):
        x = trajectory.reshape(-1)
        
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
            x = activation(x)
        
        score = nn.Dense(1, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        return score.squeeze()
    

def rollout(policy_params, rng, env, env_params, steps):
    @jax.jit
    def policy_step(state_input, _):
        obs, state, policy_params, rng, is_done = state_input
        rng, rng_step = jax.random.split(rng)
        pi = PolicyNetwork(env.num_actions).apply(policy_params, obs)
        action = pi.sample(seed=rng_step)
        log_prob = pi.log_prob(action)
        
        next_obs, next_state, reward, done, _ = env.step(rng_step, state, action, env_params)
        reward = jax.lax.select(is_done, 0.0, reward)

        is_done = jnp.logical_or(is_done, done)
        carry = [next_obs, next_state, policy_params, rng_step, is_done]
        return carry, [obs, action, reward, log_prob, is_done]

    rng_reset, rng_episode = jax.random.split(rng)
    obs, state = env.reset(rng_reset, env_params)
    
    _, scan_out = jax.lax.scan(policy_step, [obs, state, policy_params, rng_episode, False], (), steps)
    
    obs, actions, rewards, log_probs, done = scan_out
    return obs, actions, rewards, log_probs, done



def reinforce(env, env_params, policy_params, optimizer, rng, num_episodes, discount_factor):
    def compute_loss(policy_params, obs, actions, rewards, rng, env_params):
        discounted_returns = calculate_discounted_returns(rewards, discount_factor)
        pi = PolicyNetwork(env.num_actions).apply(policy_params, obs)
        log_probs = pi.log_prob(actions)
        
        # Normalize the discounted returns
        normalized_returns = (discounted_returns - jnp.mean(discounted_returns)) / (jnp.std(discounted_returns) + 1e-8)
        
        losses = -jnp.mean(log_probs * normalized_returns)
        return losses

    grad_fn = jax.value_and_grad(compute_loss)
    
    @jax.jit
    def update_step(rng, opt_state, policy_params):
        rng, rng_step = jax.random.split(rng)
        obs, actions, rewards, _, _ = rollout(policy_params, rng_step, env, env_params, STEPS_PER_EPISODE)
        
        loss, grad = grad_fn(policy_params, obs, actions, rewards, rng_step, env_params)
        updates, opt_state = optimizer.update(grad, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        
        return rng, opt_state, policy_params, jnp.sum(rewards)

    opt_state = optimizer.init(policy_params)
    
    def body_fun(i, state):
        rng, opt_state, policy_params, metrics = state
        rng, opt_state, policy_params, episode_return = update_step(rng, opt_state, policy_params)
        metrics = metrics.at[i].set(episode_return)
        return rng, opt_state, policy_params, metrics

    rng, opt_state, policy_params, metrics = jax.lax.fori_loop(
        0, num_episodes, body_fun, (rng, opt_state, policy_params, jnp.zeros(num_episodes))
    )
    
    return policy_params, metrics


#Preference model
def generate_trajectory_pairs(policy_params, env, env_params, rng, batch_size, steps):
    def generate_single_pair(rng):
        rng1, rng2 = jax.random.split(rng)
        obs1, actions1, rewards1, log_probs1, _ = rollout(policy_params, rng1, env, env_params, steps)
        obs2, actions2, rewards2, log_probs2, _ = rollout(policy_params, rng2, env, env_params, steps)
        traj1 = jnp.concatenate([obs1, actions1[:, None]], axis=1)
        traj2 = jnp.concatenate([obs2, actions2[:, None]], axis=1)
        return traj1, traj2, rewards1, rewards2, log_probs1, log_probs2

    rngs = jax.random.split(rng, batch_size)
    return jax.vmap(generate_single_pair)(rngs)

def compute_preference_score(preference_params, trajectory):
    return PreferenceNetwork().apply(preference_params, trajectory)

def compute_preference_score_log_probs(preference_params, trajectory, log_probs):
    return PreferenceNetwork().apply(preference_params, trajectory)*log_probs

def bradley_terry_loss(score1, score2, preference):
    logits = BETA*(score1 - score2)
    return -preference * logits + jnp.log(1 + jnp.exp(logits))

def generate_preference(rewards1, rewards2):
    return (jnp.sum(rewards1) > jnp.sum(rewards2)).astype(jnp.float32)

def calculate_discounted_returns(rewards, discount_factor):
    rewards_reversed = jnp.flip(rewards, axis=0)
    discounts = discount_factor ** jnp.arange(rewards_reversed.size)
    discounted_rewards_reversed = rewards_reversed * discounts
    cumulative_returns = jnp.flip(jnp.cumsum(discounted_rewards_reversed), axis=0)
    return cumulative_returns

def update_preference_model(preference_params, trajectories1, trajectories2, preferences, optimizer, opt_state):
    def loss_fn(preference_params):
        scores1 = jax.vmap(lambda t: compute_preference_score(preference_params, t))(trajectories1)
        scores2 = jax.vmap(lambda t: compute_preference_score(preference_params, t))(trajectories2)
        losses = jax.vmap(bradley_terry_loss)(scores1, scores2, preferences)
        return jnp.mean(losses)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(preference_params)
    print('pref grad norm', optax.global_norm(grads))
    updates, new_opt_state = optimizer.update(grads, opt_state, preference_params)
    new_preference_params = optax.apply_updates(preference_params, updates)
    return new_preference_params, loss, new_opt_state

def reinforce_preference(env, env_params, policy_params, preference_params, optimizer, rng, num_episodes, steps_per_episode):
    def compute_loss(policy_params, preference_params, rng, env_params):
        obs, actions, _, log_probs, _ = rollout(policy_params, rng,env,  env_params, steps_per_episode)
        
        trajectory = jnp.concatenate([obs, actions[:, None]], axis=1)
        preference_score = PreferenceNetwork().apply(preference_params, trajectory)
        # Calculate returns using preference scores
        # Calculate discounted preference scores
        # Reverse the preference scores
        reversed_preference_scores = jnp.flip(preference_score)
        
        # Calculate discounted, reversed preference scores
        discounts = jnp.power(1, jnp.arange(steps_per_episode))
        discounted_preference_scores = reversed_preference_scores * discounts
        
        # Calculate cumulative sum of discounted, reversed preference scores
        cumulative_scores = jnp.cumsum(discounted_preference_scores)
        
        # Reverse back to align with original trajectory
        returns = jnp.flip(cumulative_scores)
        
        # Normalize the returns
        normalized_returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)
        
        # Compute loss using the normalized returns
        losses = -jnp.mean(log_probs * normalized_returns)
        return losses

    grad_fn = jax.value_and_grad(compute_loss, argnums=0)

    @jax.jit
    def update_step(rng, opt_state, policy_params, preference_params):
        rng, rng_step = jax.random.split(rng)
        loss, grad = grad_fn(policy_params, preference_params, rng_step, env_params)
        updates, opt_state = optimizer.update(grad, opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)
        
        obs, actions, rewards, _, _ = rollout(policy_params, rng_step, env, env_params, steps_per_episode)
        trajectory = jnp.concatenate([obs, actions[:, None]], axis=1)
        preference_score = PreferenceNetwork().apply(preference_params, trajectory)
        
        return rng, opt_state, policy_params, preference_score.sum(), jnp.sum(rewards)

    opt_state = optimizer.init(policy_params)
    
    def body_fun(i, state):
        rng, opt_state, policy_params, pref_metrics, true_metrics = state
        rng, opt_state, policy_params, pref_score, true_reward = update_step(rng, opt_state, policy_params, preference_params)
        pref_metrics = pref_metrics.at[i].set(pref_score)
        true_metrics = true_metrics.at[i].set(true_reward)
        return rng, opt_state, policy_params, pref_metrics, true_metrics

    rng, opt_state, policy_params, pref_metrics, true_metrics = jax.lax.fori_loop(
        0, num_episodes, body_fun, (rng, opt_state, policy_params, jnp.zeros(num_episodes), jnp.zeros(num_episodes))
    )
    
    return policy_params, pref_metrics, true_metrics

def compute_hypergradient(preference_params, policy_params, trajectories1, trajectories2, preferences, env, steps_per_episode, writer, i, reg):
    # 1. Compute direct gradient of preference loss: ∇_ν L_pref(ν)
    def preference_loss(pref_params):
        scores1 = jax.vmap(lambda t: compute_preference_score(pref_params, t))(trajectories1)
        scores2 = jax.vmap(lambda t: compute_preference_score(pref_params, t))(trajectories2)
        losses = jax.vmap(bradley_terry_loss)(scores1, scores2, preferences)
        return jnp.mean(losses)
    
    # 2. Compute reinforcement loss
    def reinforce_loss(policy_params, preference_params, trajectory):
        obs, actions = trajectory[:, :-1], trajectory[:, -1].astype(jnp.int32)
        pi = PolicyNetwork(env.num_actions).apply(policy_params, obs)
        log_probs = pi.log_prob(actions)
        
        preference_score = PreferenceNetwork().apply(preference_params, trajectory)
        returns = compute_returns(preference_score, steps_per_episode)
        #normalized_returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)
        
        return -jnp.mean(log_probs * returns)

    def batch_reinforce_loss(policy_params, preference_params):
        loss1 = jax.vmap(lambda t: reinforce_loss(policy_params, preference_params, t))(trajectories1)
        loss2 = jax.vmap(lambda t: reinforce_loss(policy_params, preference_params, t))(trajectories2)
        return (loss1 + loss2).mean()
    
    def weighted_preference_loss(pref_params, pol_params):
        # Compute preference scores
        scores1 = jax.vmap(lambda t: compute_preference_score(pref_params, t))(trajectories1)
        scores2 = jax.vmap(lambda t: compute_preference_score(pref_params, t))(trajectories2)

        
        # Compute log probabilities
        def compute_log_probs(trajectory):
            obs, actions = trajectory[:, :-1], trajectory[:, -1].astype(jnp.int32)
            pi = PolicyNetwork(env.num_actions).apply(pol_params, obs)
            return jnp.sum(pi.log_prob(actions))
        
        log_probs1 = jax.vmap(compute_log_probs)(trajectories1)
        log_probs2 = jax.vmap(compute_log_probs)(trajectories2)
        
        # Compute losses
        losses = jax.vmap(bradley_terry_loss)(scores1, scores2, preferences)
        
        # Weight the losses by the sum of log probabilities
        log_prob_sum = log_probs1 + log_probs2
        weighted_losses = losses * log_prob_sum
        
        return jnp.mean(weighted_losses)

    rl_loss, policy_grad = jax.value_and_grad(batch_reinforce_loss, argnums=0)(policy_params, preference_params)
    pm_loss, pm_grad = jax.value_and_grad(preference_loss)(preference_params)
    

    pm_grad_pm_log_probs = jax.value_and_grad(weighted_preference_loss, argnums=1)(preference_params, policy_params)
    lm_grad_pm = jax.grad(batch_reinforce_loss, argnums=1)(policy_params, preference_params)
    #print('pm_grad_pm_log_probs', pm_grad_pm_log_probs)

     # 4. Compute Hessian-vector product function: ∇²_θ V_s(ν, θ*(ν)) * v
    def hvp(v):
        preference_params_flat, unravel_fn = jax.flatten_util.ravel_pytree(preference_params)
        def loss_grad_flat(p):
            return jax.flatten_util.ravel_pytree(
                jax.grad(preference_loss)(unravel_fn(p))
            )[0]
        lambda_reg = 10.0
        hvp = jax.jvp(loss_grad_flat, (preference_params_flat,), (v,))[1] + lambda_reg * v
        return hvp
    


    def nystrom_hvp(rank, rho, lambda_reg=1):
        # Step 1: Compute the gradients of the outer objective with respect to the inner and outer params
        out_out_g, in_out_g = jax.grad(batch_reinforce_loss, argnums=(0, 1))(policy_params, preference_params)
        
        # Step 2: Select random rows for low-rank approximation
        param_size = sum(x.size for x in jax.tree_util.tree_leaves(preference_params))
        indices = jax.random.permutation(jax.random.PRNGKey(0), param_size)[:rank]

        # Helper function to select Hessian rows using the inner objective
        def select_grad_row(in_params, indices):
            grad = jax.grad(lambda params: preference_loss(params))(in_params)
            grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
            return grad_flat[indices]

        # Step 3: Approximate Hessian rows using the selected indices
        hessian_rows = jax.jacrev(select_grad_row)(preference_params, indices)

        # Flattening the hessian rows pytree into a single vector for concatenation
        hessian_rows_flat, _ = jax.flatten_util.ravel_pytree(hessian_rows)

        # Step 4: Concatenate the rows to form matrix C
        C = jnp.reshape(hessian_rows_flat, (rank, -1))

        # Step 5: Apply the Woodbury matrix identity to compute the inverse Hessian-vector product
        M = C.take(indices, axis=1)
        v_flat, _ = jax.flatten_util.ravel_pytree(in_out_g)
        
        # Compute the inverse Hessian-vector product using Woodbury identity
        x = (1 / (rho )) * v_flat - (1 / ((rho ) ** 2)) * C.T @ jax.scipy.linalg.solve(M + (1 / rho) * C @ C.T +  jnp.eye(M.shape[0]), C @ v_flat)

        # Step 6: Ensure the result is a flat vector that can be unraveled
        return x 



    
    # 5. Use conjugate gradient to compute inverse HVP: [∇²_θ V_s(ν, θ*(ν))]^(-1) * ∇_θ L_pref(ν)
    lmpm_flat, unflatten_fn = jax.flatten_util.ravel_pytree(lm_grad_pm)
    def cg_solve(v):
        return jax.scipy.sparse.linalg.cg(hvp, v, maxiter=20, tol=1e-10)[0]


    inverse_hvp_flat = nystrom_hvp(10, 1000)
    inverse_hvp = unflatten_fn(inverse_hvp_flat)
    

    
     # 6. Compute mixed gradient and its transpose: [∇²_θ,ν V_s(ν, θ*(ν))]^T
    def mixed_grad_fn(policy_params, preference_params):
        return jax.grad(weighted_preference_loss, argnums=1)(preference_params, policy_params)
    
    # 7. Compute the final product: [∇²_θ,ν V_s(ν, θ*(ν))]^T * [∇²_θ V_s(ν, θ*(ν))]^(-1) * ∇_θ L_pref(ν)
    # We use JVP to compute this product efficiently
    _, final_product = jax.jvp(
        lambda p: mixed_grad_fn(policy_params, p),
        (preference_params,),
        (inverse_hvp,)
    )
    hypergradient = jax.tree_util.tree_map(lambda x, y:  x - y , policy_grad, final_product)
    #print all norm
    print('pm_grad', optax.global_norm(pm_grad))
    print('pm_grad_pm_log_probs', optax.global_norm(pm_grad_pm_log_probs))
    print('lm_grad_pm', optax.global_norm(lm_grad_pm))
    print('inverse_hvp', optax.global_norm(inverse_hvp))
    print('final_product', optax.global_norm(final_product))

    print('policy_grad', optax.global_norm(policy_grad))
    print('hypergradient', optax.global_norm(hypergradient))


    writer.add_scalar('pm_grad', optax.global_norm(pm_grad), i)
    writer.add_scalar('pm_grad_pm_log_probs', optax.global_norm(pm_grad_pm_log_probs), i)
    writer.add_scalar('lm_grad_pm', optax.global_norm(lm_grad_pm), i)
    writer.add_scalar('inverse_hvp', optax.global_norm(inverse_hvp), i)
    writer.add_scalar('final_product', optax.global_norm(final_product), i)
    writer.add_scalar('policy_grad', optax.global_norm(policy_grad), i)
    writer.add_scalar('hypergradient', optax.global_norm(hypergradient), i)
    writer.add_scalar('reg', reg, i)


    return hypergradient


import scipy

import jax.numpy as jnp
import scipy.linalg

def check_equilibrium(policy_grad, pm_grad, inverse_hvp, final_product, lm_grad_pm, tolerance=0.02):
    # Check first-order conditions
    leader_grad_norm = optax.global_norm(policy_grad)
    follower_grad_norm = optax.global_norm(pm_grad)

  

    print(f"Gradient norms - Leader: {leader_grad_norm:.6f}, Follower: {follower_grad_norm:.6f}")
    is_equilibrium = (leader_grad_norm < tolerance) and (follower_grad_norm < tolerance)
  

    return is_equilibrium, {
        'leader_grad_norm': leader_grad_norm,
        'follower_grad_norm': follower_grad_norm
    }


# Helper function to compute returns
def compute_returns(preference_score, steps_per_episode):
    reversed_scores = jnp.flip(preference_score)
    discounts = jnp.power(DISCOUNT_FACTOR, jnp.arange(steps_per_episode))
    discounted_scores = reversed_scores * discounts
    cumulative_scores = jnp.cumsum(discounted_scores)
    return jnp.flip(cumulative_scores)




def stack_training_loop(num_iterations, num_inner_iterations, policy_params, preference_params, policy_optimizer, preference_optimizer, rng, env, env_params, writer):
    true_rewards = []
    pref_rewards = []
    pref_losses = []
    reg = 1
    policy_opt_state = policy_optimizer.init(policy_params)
    pref_opt_state = preference_optimizer.init(preference_params)
    for i in range(num_iterations):
        # Lower level: Policy optimization
        # Lower level: Single policy optimization step with batch
    
        # Generate multiple trajectory pairs
        rng, rng_traj = jax.random.split(rng)
        trajectories1, trajectories2, rewards1, rewards2, log_probs1, log_probs2 = generate_trajectory_pairs(
            policy_params, env, env_params, rng_traj, BATCH_SIZE, STEPS_PER_EPISODE
        )
        preferences = jax.vmap(generate_preference)(rewards1, rewards2)
        true_reward = ((jnp.sum(rewards1)/BATCH_SIZE) + (jnp.sum(rewards2)/BATCH_SIZE))/2
        print('rewards', true_reward)
        true_rewards.append(true_reward)

        #get preference scores
        scores1 = jax.vmap(lambda t: compute_preference_score(preference_params, t))(trajectories1)
        scores2 = jax.vmap(lambda t: compute_preference_score(preference_params, t))(trajectories2)
        preference_scores = (scores1.mean() + scores2.mean())/2
        print('preference_scores', preference_scores)
        pref_rewards.append(preference_scores)
        writer.add_scalar('preference_scores', preference_scores, i)


      
        writer.add_scalar('true_reward', true_reward, i)
        #get second last true reward
        if i > 1:
            prev_true_reward = true_rewards[-2]
            if true_reward > prev_true_reward and reg < 1e5:
                print('true_reward is increasing, so decreasing reg')
                print('reg', reg)
                reg = reg*2
            elif true_reward < prev_true_reward and reg > 1e-5:
                print('true_reward is decreasing, so increasing reg')
                print('reg', reg)
                reg = reg/2
        
    
        for j in range(num_inner_iterations):
            #reset preference optimizer
            #pref_opt_state = preference_optimizer.init(preference_params)

            # Generate preferences using ground truth returns
            # Upper level: Preference model update
            preference_params, loss, pref_opt_state = update_preference_model(
                preference_params, trajectories1, trajectories2, preferences, 
                preference_optimizer, pref_opt_state
            )
            pref_losses.append(loss)
            print('pref_losses', loss)
            writer.add_scalar('pref_loss', loss, i*num_inner_iterations + j)

        print(f"Iteration {i}, Preference model loss: {loss}, True reward: {true_rewards[-1]}")

        hypergradient = compute_hypergradient(preference_params, policy_params, trajectories1, trajectories2, preferences, env, STEPS_PER_EPISODE, writer, i, reg)
        updates, policy_opt_state = policy_optimizer.update(hypergradient, policy_opt_state, policy_params)
        policy_params = optax.apply_updates(policy_params, updates)



        
    return policy_params, preference_params, true_rewards, pref_rewards, pref_losses


def plot_training_curves(reinforce_returns, stack_true_rewards, stack_pref_rewards):
    plt.figure(figsize=(12, 8))
    #plt.plot(reinforce_returns, label='REINFORCE')
    plt.plot(stack_true_rewards, label='Stack (True Rewards)')
    plt.plot(stack_pref_rewards, label='Stack (Preference Rewards)')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Training Curves: REINFORCE vs PARL')
    plt.legend()
    plt.savefig('hyperstack_nystrom.png')
    #plt.show()

def plot_gif(env, env_params, policy_params, rng):
    print('plotting gif...')
    state_seq, reward_seq = [], []
    rng, rng_reset = jax.random.split(rng)
    obs, env_state = env.reset(rng_reset, env_params)
    count = 0
    while True:
        state_seq.append(env_state)
        rng, rng_act, rng_step = jax.random.split(rng, 3)
        pi = PolicyNetwork(env.num_actions).apply(policy_params, obs)
        action = pi.sample(seed=rng_step)
        next_obs, next_env_state, reward, done, info = env.step(
            rng_step, env_state, action, env_params
        )
        reward_seq.append(reward)
        count += 1
        if done or count > 50:
            break
        else:
            obs = next_obs
            env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)
    vis.animate(f"hyperstack_anim.gif")

NUM_INNER_ITERATIONS = 10
def main():

    writer = SummaryWriter('runs/nystrom_unnormalized_eplen200_10inner_reg1000_beta1.0_plr1e-3')
    # Set up environment
    env, env_params = gymnax.make(ENVIRONMENT)

    
    # Initialize networks, optimizers, and parameters
    rng = jax.random.PRNGKey(42)
    rng, policy_key, pref_key = jax.random.split(rng, 3)
    
    init_obs, _ = env.reset(rng)
    policy_params = PolicyNetwork(env.num_actions).init(policy_key, init_obs)
    preference_params = PreferenceNetwork().init(pref_key, jnp.zeros((STEPS_PER_EPISODE, init_obs.shape[0] + 1)))
    
    policy_optimizer = optax.adam(POLICY_LEARNING_RATE)
    preference_optimizer = optax.adam(PREFERENCE_LEARNING_RATE)
    pref_opt_state = preference_optimizer.init(preference_params)
    
    # Run REINFORCE baseline
    print("Running REINFORCE baseline...")
    reinforce_policy_params, reinforce_returns = reinforce(env, env_params, policy_params, policy_optimizer, rng, REINFORCE_EPISODES, DISCOUNT_FACTOR)
    print("Reinforce returns:", reinforce_returns)
    # Run PARL training loop
    print("Running Stack training loop...")
    stack_policy_params, stack_preference_params, stack_true_rewards, stack_pref_rewards, _ = stack_training_loop(
        NUM_ITERATIONS, 
        NUM_INNER_ITERATIONS,
        policy_params,
        preference_params,
        policy_optimizer,
        preference_optimizer,
        rng,
        env,
        env_params,
        writer
    )


    
    # Visualize results
    plot_training_curves(reinforce_returns, stack_true_rewards, stack_pref_rewards)

    plot_gif(env, env_params, stack_policy_params, rng)

    # save policy and preference params as pickle
    import pickle
    with open('params/hyperstack_policy_params.pkl', 'wb') as f:
        pickle.dump(stack_policy_params, f)
    with open('params/hyperstack_preference_params.pkl', 'wb') as f:
        pickle.dump(stack_preference_params, f)



if __name__ == "__main__":
    main()