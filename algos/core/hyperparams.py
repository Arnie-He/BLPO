import flax
import flax.struct

@flax.struct.dataclass
class Hyperparams:
    """A data class that stores hyperparameter configurations."""
    num_updates: int = flax.struct.field(pytree_node=False)
    batch_count: int = flax.struct.field(pytree_node=False)
    rollout_len: int = flax.struct.field(pytree_node=False)
    discount_rate: float = flax.struct.field(pytree_node=False)
    actor_learning_rate: float = flax.struct.field(pytree_node=False)
    critic_learning_rate: float = flax.struct.field(pytree_node=False)
    nested_updates: int = flax.struct.field(pytree_node=False)
    adam_eps: float = flax.struct.field(pytree_node=False)
    advantage_rate: float = flax.struct.field(pytree_node=False)
    nystrom_rank: int = flax.struct.field(pytree_node=False)
    nystrom_rho: int = flax.struct.field(pytree_node=False)
