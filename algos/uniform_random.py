from env import ENV, SEED

from flax import struct
import functools
import gymnax
import jax
from jax import lax, numpy as jnp, random

@struct.dataclass
class RandomPolicy:
    """A uniform random policy that samples from a discrete action range."""
    num_actions: int = struct.field(pytree_node=False)

    @jax.jit
    def sample(self, random_key):
        return random.choice(random_key, self.num_actions)

@functools.partial(jax.jit, static_argnums=2)
def run_rollout(policy, env_params, num_steps, random_key):
    """Runs a rollout for a fixed number of steps."""
    reset_key, rollout_key = random.split(random_key, 2)
    o, state = env.reset(reset_key, env_params)

    def step(rollout_state, i):
        state, cum_reward, rollout_key = rollout_state
        rollout_key, policy_key, step_key = random.split(rollout_key, 3)
        action = policy.sample(policy_key)
        o, next_state, reward, done, i = env.step(
            step_key, state, action, env_params,
        )

        return lax.cond(
            done,
            # Reset the cumulative reward at the end of an episode
            lambda: ([next_state, jnp.float32(0), rollout_key], [cum_reward + reward, done]),
            # Carry over the reward to the next state
            lambda: ([next_state, cum_reward + reward, rollout_key], [jnp.float32(0), done]),
        )

    s, result = lax.scan(
        step,
        init=[state, 0, rollout_key],
        length=num_steps,
    )
    return result

# Create the environment and policy
random_key = random.key(SEED)
env, env_params = gymnax.make(ENV)
random_policy = RandomPolicy(env.action_space(env_params).n)
rewards = []

# Run 1000 rollouts of 100 steps each
for r in range(1000):
    random_key, rollout_key = random.split(random_key, 2)
    reward, done = run_rollout(random_policy, env_params, 100, rollout_key)
    assert jnp.all(done[reward != 0])
    rewards.append(reward[reward != 0])

rewards = jnp.hstack(rewards)
print("Average reward: ", jnp.mean(rewards))