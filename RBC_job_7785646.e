wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Sweep Agent: Waiting for job.
wandb: Job received.
wandb: Agent Starting Run: eunst3cc with config:
wandb: 	ACTIVATION: tanh
wandb: 	ANNEAL_LR: False
wandb: 	CLIP_EPS: 0.2
wandb: 	CLIP_F: 1.2
wandb: 	DEBUG: True
wandb: 	ENT_COEF: 0
wandb: 	ENV_NAME: humanoid
wandb: 	GAE_LAMBDA: 0.95
wandb: 	GAMMA: 0.99
wandb: 	Group: sweep0
wandb: 	IHVP_BOUND: 1.4
wandb: 	NORMALIZE_ENV: True
wandb: 	NUM_ENVS: 32
wandb: 	NUM_MINIBATCHES: 32
wandb: 	NUM_STEPS: 640
wandb: 	SEED: 0
wandb: 	TOTAL_TIMESTEPS: 20000000
wandb: 	UPDATE_EPOCHS: 4
wandb: 	VF_COEF: 0.5
wandb: 	actor-LR: 0.0003
wandb: 	critic-LR: 0.0012
wandb: 	nested_updates: 5
wandb: 	nystrom_rank: 5
wandb: 	nystrom_rho: 50
wandb: 	vanilla: False
wandb: Currently logged in as: arniehe2004 (rl-power). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.19.1
wandb: Run data is saved locally in /oscar/home/nhe6/Documents/HyperGradient-RL/wandb/run-20250126_144302-eunst3cc
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run dulcet-sweep-1
wandb: ⭐️ View project at https://wandb.ai/rl-power/hopper-sweep
wandb: 🧹 View sweep at https://wandb.ai/rl-power/hopper-sweep/sweeps/sp1jvwdo
wandb: 🚀 View run at https://wandb.ai/rl-power/hopper-sweep/runs/eunst3cc
2025-01-26 14:43:03.477808: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:266] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
wandb:                                                                                
wandb: 🚀 View run dulcet-sweep-1 at: https://wandb.ai/rl-power/hopper-sweep/runs/eunst3cc
wandb: ⭐️ View project at: https://wandb.ai/rl-power/hopper-sweep
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250126_144302-eunst3cc/logs
Run eunst3cc errored:
Traceback (most recent call last):
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/xla_bridge.py", line 879, in backends
    backend = _init_backend(platform)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/xla_bridge.py", line 970, in _init_backend
    backend = registration.factory()
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/xla_bridge.py", line 668, in factory
    return xla_client.make_c_api_client(plugin_name, updated_options, None)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jaxlib/xla_client.py", line 200, in make_c_api_client
    return _xla.get_c_api_client(plugin_name, options, distributed_client)
jaxlib.xla_extension.XlaRuntimeError: FAILED_PRECONDITION: No visible GPU devices.

During handling of the above exception, another exception occurred:

jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/wandb/agents/pyagent.py", line 306, in _run_job
    self._function()
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/sweep.py", line 15, in main
    rng = jax.random.PRNGKey(seed)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/random.py", line 233, in PRNGKey
    return _return_prng_keys(True, _key('PRNGKey', seed, impl))
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/random.py", line 195, in _key
    return prng.random_seed(seed, impl=impl)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/prng.py", line 533, in random_seed
    seeds_arr = jnp.asarray(np.int64(seeds))
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/numpy/lax_numpy.py", line 3289, in asarray
    return array(a, dtype=dtype, copy=bool(copy), order=order)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/numpy/lax_numpy.py", line 3214, in array
    out_array: Array = lax_internal._convert_element_type(
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/lax/lax.py", line 559, in _convert_element_type
    return convert_element_type_p.bind(operand, new_dtype=new_dtype,
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/core.py", line 416, in bind
    return self.bind_with_trace(find_top_trace(args), args, params)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/core.py", line 420, in bind_with_trace
    out = trace.process_primitive(self, map(trace.full_raise, args), params)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/core.py", line 921, in process_primitive
    return primitive.impl(*tracers, **params)
  File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/dispatch.py", line 87, in apply_primitive
    outs = fun(*args)
RuntimeError: Unable to initialize backend 'cuda': FAILED_PRECONDITION: No visible GPU devices. (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)

wandb: ERROR Run eunst3cc errored:
wandb: ERROR Traceback (most recent call last):
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/xla_bridge.py", line 879, in backends
wandb: ERROR     backend = _init_backend(platform)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/xla_bridge.py", line 970, in _init_backend
wandb: ERROR     backend = registration.factory()
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/xla_bridge.py", line 668, in factory
wandb: ERROR     return xla_client.make_c_api_client(plugin_name, updated_options, None)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jaxlib/xla_client.py", line 200, in make_c_api_client
wandb: ERROR     return _xla.get_c_api_client(plugin_name, options, distributed_client)
wandb: ERROR jaxlib.xla_extension.XlaRuntimeError: FAILED_PRECONDITION: No visible GPU devices.
wandb: ERROR 
wandb: ERROR During handling of the above exception, another exception occurred:
wandb: ERROR 
wandb: ERROR jax.errors.SimplifiedTraceback: For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.
wandb: ERROR 
wandb: ERROR The above exception was the direct cause of the following exception:
wandb: ERROR 
wandb: ERROR Traceback (most recent call last):
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/wandb/agents/pyagent.py", line 306, in _run_job
wandb: ERROR     self._function()
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/sweep.py", line 15, in main
wandb: ERROR     rng = jax.random.PRNGKey(seed)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/random.py", line 233, in PRNGKey
wandb: ERROR     return _return_prng_keys(True, _key('PRNGKey', seed, impl))
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/random.py", line 195, in _key
wandb: ERROR     return prng.random_seed(seed, impl=impl)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/prng.py", line 533, in random_seed
wandb: ERROR     seeds_arr = jnp.asarray(np.int64(seeds))
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/numpy/lax_numpy.py", line 3289, in asarray
wandb: ERROR     return array(a, dtype=dtype, copy=bool(copy), order=order)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/numpy/lax_numpy.py", line 3214, in array
wandb: ERROR     out_array: Array = lax_internal._convert_element_type(
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/lax/lax.py", line 559, in _convert_element_type
wandb: ERROR     return convert_element_type_p.bind(operand, new_dtype=new_dtype,
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/core.py", line 416, in bind
wandb: ERROR     return self.bind_with_trace(find_top_trace(args), args, params)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/core.py", line 420, in bind_with_trace
wandb: ERROR     out = trace.process_primitive(self, map(trace.full_raise, args), params)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/core.py", line 921, in process_primitive
wandb: ERROR     return primitive.impl(*tracers, **params)
wandb: ERROR   File "/oscar/home/nhe6/Documents/HyperGradient-RL/jax.venv/lib64/python3.9/site-packages/jax/_src/dispatch.py", line 87, in apply_primitive
wandb: ERROR     outs = fun(*args)
wandb: ERROR RuntimeError: Unable to initialize backend 'cuda': FAILED_PRECONDITION: No visible GPU devices. (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)
wandb: ERROR 
wandb: Agent Starting Run: xb9hm69j with config:
wandb: 	ACTIVATION: tanh
wandb: 	ANNEAL_LR: False
wandb: 	CLIP_EPS: 0.2
wandb: 	CLIP_F: 0.8
wandb: 	DEBUG: True
wandb: 	ENT_COEF: 0
wandb: 	ENV_NAME: humanoid
wandb: 	GAE_LAMBDA: 0.95
wandb: 	GAMMA: 0.99
wandb: 	Group: sweep0
wandb: 	IHVP_BOUND: 1.2
wandb: 	NORMALIZE_ENV: True
wandb: 	NUM_ENVS: 32
wandb: 	NUM_MINIBATCHES: 32
wandb: 	NUM_STEPS: 640
wandb: 	SEED: 0
wandb: 	TOTAL_TIMESTEPS: 20000000
wandb: 	UPDATE_EPOCHS: 4
wandb: 	VF_COEF: 0.5
wandb: 	actor-LR: 0.0003
wandb: 	critic-LR: 0.0012
wandb: 	nested_updates: 2
wandb: 	nystrom_rank: 5
wandb: 	nystrom_rho: 50
wandb: 	vanilla: False
wandb: Tracking run with wandb version 0.19.1
wandb: Run data is saved locally in /oscar/home/nhe6/Documents/HyperGradient-RL/wandb/run-20250126_144307-xb9hm69j
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run curious-sweep-2
wandb: ⭐️ View project at https://wandb.ai/rl-power/hopper-sweep
wandb: 🧹 View sweep at https://wandb.ai/rl-power/hopper-sweep/sweeps/sp1jvwdo
wandb: 🚀 View run at https://wandb.ai/rl-power/hopper-sweep/runs/xb9hm69j
wandb: WARNING Ignoring project 'HyperGradient-RL' when running a sweep.
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: Reward ▁▁▂▁▁▁▃▇▇▇▇▇▂▇▇▇▇▇▇▇▇▇▄████████▂▂▂██████
wandb: 
wandb: 🚀 View run curious-sweep-2 at: https://wandb.ai/rl-power/hopper-sweep/runs/xb9hm69j
wandb: ⭐️ View project at: https://wandb.ai/rl-power/hopper-sweep
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20250126_144307-xb9hm69j/logs
wandb: Agent Starting Run: 61qeic8p with config:
wandb: 	ACTIVATION: tanh
wandb: 	ANNEAL_LR: False
wandb: 	CLIP_EPS: 0.2
wandb: 	CLIP_F: 1.2
wandb: 	DEBUG: True
wandb: 	ENT_COEF: 0
wandb: 	ENV_NAME: humanoid
wandb: 	GAE_LAMBDA: 0.95
wandb: 	GAMMA: 0.99
wandb: 	Group: sweep0
wandb: 	IHVP_BOUND: 1.2
wandb: 	NORMALIZE_ENV: True
wandb: 	NUM_ENVS: 32
wandb: 	NUM_MINIBATCHES: 32
wandb: 	NUM_STEPS: 640
wandb: 	SEED: 0
wandb: 	TOTAL_TIMESTEPS: 20000000
wandb: 	UPDATE_EPOCHS: 4
wandb: 	VF_COEF: 0.5
wandb: 	actor-LR: 0.0003
wandb: 	critic-LR: 0.0012
wandb: 	nested_updates: 7
wandb: 	nystrom_rank: 5
wandb: 	nystrom_rho: 50
wandb: 	vanilla: False
wandb: Tracking run with wandb version 0.19.1
wandb: Run data is saved locally in /oscar/home/nhe6/Documents/HyperGradient-RL/wandb/run-20250126_160916-61qeic8p
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run ancient-sweep-3
wandb: ⭐️ View project at https://wandb.ai/rl-power/hopper-sweep
wandb: 🧹 View sweep at https://wandb.ai/rl-power/hopper-sweep/sweeps/sp1jvwdo
wandb: 🚀 View run at https://wandb.ai/rl-power/hopper-sweep/runs/61qeic8p
wandb: WARNING Ignoring project 'HyperGradient-RL' when running a sweep.
