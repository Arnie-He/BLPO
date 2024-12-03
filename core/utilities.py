import jax
import jax.numpy as jnp

config = None

def initialize_config(cfg):
    global config
    config = cfg

def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

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