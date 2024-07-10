import jax.numpy as jnp

a = jnp.arange(2880).reshape(12,10,24)
b = jnp.ones((10, 24))

axes_b = list(range(len(b.shape)))
axes_a = [x + len(a.shape) - len(b.shape) for x in axes_b]

c = jnp.tensordot(a, b, (axes_a, axes_b))
print(c.shape)