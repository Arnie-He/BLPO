import jax
import jax.numpy as jnp

def cosine_similarity(grad1, grad2):
    # Compute element-wise products and sums over the pytrees
    inner_product = jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), grad1, grad2)
    )
    # Compute norms of each gradient
    norm1 = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(lambda x: jnp.vdot(x, x), grad1)
    ))
    norm2 = jnp.sqrt(jax.tree_util.tree_reduce(
        lambda x, y: x + y,
        jax.tree_util.tree_map(lambda x: jnp.vdot(x, x), grad2)
    ))
    # Calculate cosine similarity
    cosine_sim = inner_product / (norm1 * norm2)
    return cosine_sim