from typing import Sequence
import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import jacfwd, jacrev, grad
from jax import flatten_util
import jaxopt

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

# def loss_fn(params, params2, batch2, batch):
#     preds = model.apply({'params': params}, batch)
#     preds2 = model2.apply({'params': params2}, batch2)
#     loss = jnp.mean((preds - preds2) ** 2)
#     return loss
def loss_fn(params2, params, batch2, batch):
    preds = model.apply({'params': params}, batch)
    preds2 = model2.apply({'params': params2}, batch2)
    loss = jnp.mean((preds - preds2) ** 2)
    return loss

# First Model
model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(jax.random.PRNGKey(0), batch)
params = variables['params']
# Second Model
model2 = MLP([24, 20, 16, 4])
batch2 = jnp.ones((32, 10))
variables2 = model2.init(jax.random.PRNGKey(0), batch2)
params2 = variables2['params']

# Compute gradient of the loss with respect to the parameters
grad_theta_J = jax.grad(loss_fn, argnums=0)(params, params2, batch2, batch)
grad_w_J = jax.grad(loss_fn, argnums=1)(params, params2, batch2, batch)

# grad_theta_J = jax.grad(loss_fn, argnums=1)(params2, params, batch2, batch)
# grad_w_J = jax.grad(loss_fn, argnums=0)(params2, params, batch2, batch)

init_params=params
# @implicit_diff.custom_root(jax.grad(loss_fn))
def loss_fn_result(params, batch2, batch):
    gd = jaxopt.GradientDescent(fun=loss_fn, maxiter=500, implicit_diff=True)
    return gd.run(init_params, params, batch2, batch).params

grad_Optcritic = jax.jacobian(loss_fn_result, argnums=0)(params, batch2, batch)

# Convert data to JSON serializable format
def to_json_serializable(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    return obj

def tree_shape(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.shape
    elif isinstance(obj, dict):
        return {k: tree_shape(v) for k, v in obj.items()}

# Save outputs to a JSON file
with open('output.json', 'w') as f:
    # json.dump(output_data, f, indent=4)
    json.dump(tree_shape(grad_theta_J), f, indent=4)
    json.dump(tree_shape(grad_Optcritic), f, indent=4)
    json.dump(tree_shape(grad_w_J), f, indent=4)

print("Output written to output.json")
