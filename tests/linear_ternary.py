import flax
from flax import linen as nn
import jax
from jax import numpy as jnp, random

# Ternary training data
data = jnp.array([
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 1]
], dtype=jnp.float32)
inputs = data[:, :-1]
outputs = data[:, -1]

# Create linear model
init_key = random.key(0)
model = nn.Dense(1)
params = model.init(init_key, jnp.empty((3,)))

# Define loss function
def predict_batch(params, inputs, outputs):
    predicted = model.apply(params, inputs).reshape(-1)
    return jnp.mean((outputs - predicted) ** 2)
calc_grad = jax.value_and_grad(predict_batch)

# Train model over 1000 epochs
learning_rate = 0.01
for epoch in range(1000):
    loss, grad = calc_grad(params, inputs, outputs)
    params = jax.tree.map(
        lambda p, g: p - learning_rate * g,
        params, grad
    )
    if epoch % 100 == 0:
        print(f"[{epoch}] Loss: {loss}")

# Display final output
print("Final params:")
print(params)
print("Final output:")
print(model.apply(params, inputs).reshape(-1))
print(outputs)