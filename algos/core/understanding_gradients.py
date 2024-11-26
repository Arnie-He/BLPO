import jax
import jax.numpy as jnp

def cosine_similarity(grad1, grad2):
    """
    Calculate the cosine similarities between corresponding leaves of two gradient PyTrees.

    Args:
        grad1: A PyTree of gradients (e.g., from JAX or Flax), where each leaf is an array.
        grad2: Another PyTree of gradients with the same structure as grad1.

    """
    grads1_flat, _ = jax.tree_util.tree_flatten(grad1)
    grads2_flat, _ = jax.tree_util.tree_flatten(grad2)

    # Flatten arrays and concatenate
    grads1_concat = jnp.concatenate([jnp.ravel(g) for g in grads1_flat])
    grads2_concat = jnp.concatenate([jnp.ravel(g) for g in grads2_flat])

    # Compute the cosine similarity
    x_norm = jnp.linalg.norm(grads1_concat)
    y_norm = jnp.linalg.norm(grads2_concat)
    cos_sim = jnp.where((x_norm == 0) | (y_norm == 0), 0.0,
                        jnp.dot(grads1_concat, grads2_concat) / (x_norm * y_norm))
    return cos_sim

def project_B_onto_A(grad1, grad2):
    """
    Project grad2 onto grad1 for each corresponding leaf so that they point in the same direction.

    Args:
        grad1: A PyTree of gradients where each leaf is an array.
        grad2: Another PyTree of gradients with the same structure as grad1.

    Returns:
        A PyTree with the same structure as grad1 and grad2, where each leaf contains the projection of
        grad2 onto grad1.
    """
    def project(x, y):
        x_norm_sq = jnp.sum(x * x)
        projection = jnp.where(x_norm_sq == 0, jnp.zeros_like(x), (jnp.sum(x * y) / x_norm_sq) * x)
        return projection
    projected_grads = jax.tree_map(project, grad1, grad2)
    return projected_grads
