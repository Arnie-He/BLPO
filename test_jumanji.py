import jax
import jumanji
import gymnasium as gym

# Instantiate a Jumanji environment using the registry
env = jumanji.make('Snake-v1')

# Reset your (jit-able) environment
key = jax.random.PRNGKey(0)
state, timestep = jax.jit(env.reset)(key)

# (Optional) Render the env state
env.render(state)

# Interact with the (jit-able) environment
action = env.action_spec.generate_value()          # Action selection (dummy value here)
state, timestep = jax.jit(env.step)(state, action)   # Take a step and observe the next state and time step

# print(env.action_spec.num_values)

print(timestep.observation)