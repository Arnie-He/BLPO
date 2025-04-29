from brax.envs import create, _envs  # registry of all env names

for env_name in _envs:
    env = create(env_name)
    print(f"{env_name:15s}  obs_size={env.observation_size:3d}  action_size={env.action_size:2d}")
