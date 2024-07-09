from jax import grad
import jax
import jax.numpy as jnp

def x2y2(x, y):
    return jnp.dot(jnp.square(x),  jnp.square(y))

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([2.0, 3.0, 1.0])

# First-order gradients
grad_y = grad(x2y2, 1)(x, y)
print(grad_y)

# Second-order mixed partial derivatives
# Wrapping the inner gradient with a lambda function that sums the result to get a scalar
grad2_xy = grad(lambda x, y: jnp.sum(grad(x2y2, 1)(x, y)), 0)(x, y)
grad2_yx = grad(lambda x, y: jnp.sum(grad(x2y2, 0)(x, y)), 1)(x, y)
grad2_yy = grad(lambda x, y: jnp.sum(grad(x2y2, 1)(x, y)), 1)(x, y)

print(grad2_xy)
print(grad2_yx)
print(grad2_yy)