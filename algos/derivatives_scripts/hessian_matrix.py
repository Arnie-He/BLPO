from typing import Sequence
import json
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import jacfwd, jacrev, grad
from jax import flatten_util

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

def loss_fn(params, model, batch, params2, model2, batch2):
    preds = model.apply({'params': params}, batch)
    preds2 = model2.apply({'params': params2}, batch2)
    loss = jnp.mean((preds - preds2) ** 2)
    return loss

# Initialize model and data

# First Model
model = MLP([12, 8, 4])
batch = jnp.ones((32, 10))
variables = model.init(jax.random.PRNGKey(0), batch)
params = variables['params']
# Second Model
model2 = MLP([24, 20, 16, 4])
batch2 = jnp.ones((32,10))
variables2 = model2.init(jax.random.PRNGKey(0), batch2)
params2 = variables2['params']

# Compute gradient of the loss with respect to the parameters
grad_loss_fn = jax.grad(loss_fn)
grads = grad_loss_fn(params, model, batch, params2, model2, batch2)

# Compute the Hessian of the loss using the gradient
hessian_loss_fn = jax.hessian(loss_fn)
hessian = hessian_loss_fn(params, model, batch, params2, model2, batch2)

# Compute the mixed partials, specifically del_theta_w(loss_fn)
def mixed_partials(loss_fn, params, model, batch, params2, model2, batch2):
    def inner_fn(params, params2):
        return loss_fn(params, model, batch, params2, model2, batch2)
    return jacfwd(jacrev(inner_fn, argnums=0), argnums=1)(params, params2)

mixed_partials_result = mixed_partials(loss_fn, params, model, batch, params2, model2, batch2)

# Compute the inverse Hessian Produce, namely H^{-1}V, where H is the hessian matrix above, and V is the mixed partial. 
# We should also check that the product has shape of theta * omega
grad_w_J = grad(loss_fn, argnums = 3)(params, model, batch, params2, model2, batch2)

# Compute the inverse hessian vector product using https://iclr-blogposts.github.io/2024/blog/bench-hvp/
def get_hvp_forward_over_reverse(params, model, batch, model2, batch2):
    """
    Returns the Hessian-vector product operator that uses forward-over-reverse
    propagation.
    """
    grad_fun = jax.jit(
        lambda x: jax.grad(loss_fn, argnums=3)(params, model, batch, x, model2, batch2)
    )
    hvp_fun = jax.jit(
        lambda x, v: jax.jvp(grad_fun, (x,), (v,))[1]
    )
    return lambda x, v: jax.block_until_ready(hvp_fun(x, v))

hvp_fun = get_hvp_forward_over_reverse(params, model, batch, model2, batch2)
hvp = hvp_fun(params2, grad_w_J)

# total_minus_der = mixed_partials_result @ hvp


# Convert data to JSON serializable format
def to_json_serializable(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    return obj

output_data = {
    "Params": to_json_serializable(params),
    "Gradients": to_json_serializable(grads),
    "Hessian_shape": to_json_serializable(hessian),
}

def shape(obj):
    if isinstance(obj, jnp.ndarray):
        return obj.shape
    elif isinstance(obj, dict):
        return {k: shape(v) for k, v in obj.items()}

# Save outputs to a JSON file
with open('output.json', 'w') as f:
    # json.dump(output_data, f, indent=4)
    json.dump(shape(hvp), f, indent=4)
    json.dump(shape(mixed_partials_result), f, indent=4)
    json.dump(shape(grads), f, indent=4)

print("Output written to output.json")
