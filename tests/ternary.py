from flax import linen as nn
import jax
from jax import numpy as jnp, random

# Ternary training data
inputs = jnp.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
], dtype=jnp.float32)
outputs = jnp.array([
    [0, 1],
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0],
], dtype=jnp.float32)

# MLP model with 1 hidden layer
class MLP(nn.Module):
    @nn.compact
    def __call__(self, input):
        out = nn.Dense(4)(input)
        out = nn.relu(out)
        out = nn.Dense(2)(out)
        return out

# Create MLP model
init_key = random.key(0)
model = MLP()
params = model.init(init_key, jnp.empty((3,)))

# Define loss function
@jax.jit
def predict_batch(params, inputs, outputs):
    predicted = model.apply(params, inputs)
    squared_error = (outputs - predicted) ** 2
    return jnp.mean(jnp.sum(squared_error, axis=1))
calc_grad = jax.value_and_grad(predict_batch)

# Train model over 10000 epochs
learning_rate = 0.01
for epoch in range(10001):
    loss, grad = calc_grad(params, inputs, outputs)
    params = jax.tree.map(
        lambda p, g: p - learning_rate * g,
        params, grad
    )
    if epoch % 1000 == 0:
        print(f"[{epoch}] Loss: {loss}")

# Display final output
print("Final output:")
print(model.apply(params, inputs))
print(outputs)