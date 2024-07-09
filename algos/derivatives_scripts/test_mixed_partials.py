import jax
import jax.numpy as jnp

# Define the function f(w, theta)
def f(w, theta):
    return jnp.sin(w) * jnp.cos(theta)

# Define the mixed partial derivative function
def grad_wtheta_f(w, theta):
    # Compute the gradient of f with respect to w
    grad_f_w = jax.grad(f, argnums=0)
    
    # Compute the gradient of grad_f_w with respect to theta
    grad_wtheta_f = jax.grad(grad_f_w, argnums=1)
    
    return grad_wtheta_f(w, theta)

# Define values for w and theta
w = 1.0
theta = 2.0

# Compute the mixed partial derivative
result = grad_wtheta_f(w, theta)

print("Mixed partial derivative ∂²f/∂w∂θ at (w, theta) =", result)