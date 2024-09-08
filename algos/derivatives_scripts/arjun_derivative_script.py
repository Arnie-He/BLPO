def compute_hypergradient(preference_params, policy_params, trajectories1, trajectories2, preferences, env, steps_per_episode):
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
        normalized_returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

        return -jnp.mean(log_probs * normalized_returns)

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
        lambda_reg = 0.0
        hvp = jax.jvp(loss_grad_flat, (preference_params_flat,), (v,))[1] + lambda_reg * v
        return hvp

    # 5. Use conjugate gradient to compute inverse HVP: [∇²_θ V_s(ν, θ*(ν))]^(-1) * ∇_θ L_pref(ν)
    lmpm_flat, unflatten_fn = jax.flatten_util.ravel_pytree(lm_grad_pm)
    def cg_solve(v):
        return jax.scipy.sparse.linalg.cg(hvp, v, maxiter=10, tol=1e-10)[0]
    inverse_hvp_flat = cg_solve(lmpm_flat)
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

    hypergradient = jax.tree_util.tree_map(lambda x, y: x - y, policy_grad, final_product)
    #print all norms
    print('pm_grad', optax.global_norm(pm_grad))
    print('pm_grad_pm_log_probs', optax.global_norm(pm_grad_pm_log_probs))
    print('lm_grad_pm', optax.global_norm(lm_grad_pm))
    print('inverse_hvp', optax.global_norm(inverse_hvp))
    print('final_product', optax.global_norm(final_product))

    print('policy_grad', optax.global_norm(policy_grad))
    print('hypergradient', optax.global_norm(hypergradient))



    return hypergradient