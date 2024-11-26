# config.py

from models.actor import DiscreteActor, DiscretePixelActor
from models.critic import Critic, PixelCritic
from models.params import DynParam
import flax.linen as nn
import flax
import flax.struct
import jax.numpy as jnp

@flax.struct.dataclass
class Hyperparams:
    """A data class that stores hyperparameter configurations."""
    total_timesteps: int = flax.struct.field(pytree_node=False)
    rollout_len: int = flax.struct.field(pytree_node=False)
    num_envs: int = flax.struct.field(pytree_node=False)
    

    num_minibatches: int = flax.struct.field(pytree_node=False)

    discount_rate: float = flax.struct.field(pytree_node=False)
    advantage_rate: float = flax.struct.field(pytree_node=False)
    update_epochs: int = flax.struct.field(pytree_node=False)

    actor_learning_rate: float = flax.struct.field(pytree_node=False)
    actor_clip: float = flax.struct.field(pytree_node=False)
    critic_learning_rate: float = flax.struct.field(pytree_node=False)
    nested_updates: int = flax.struct.field(pytree_node=False)
    adam_eps: float = flax.struct.field(pytree_node=False)
    nystrom_rank: int = flax.struct.field(pytree_node=False)
    nystrom_rho: int = flax.struct.field(pytree_node=False)

ENV_CONFIG = {
    "cartpole": {
        "actor_model": DiscreteActor,
        "actor_params": [(64, 64), DynParam.ActionCount],
        "critic_model": Critic,
        "critic_params": [(64, 64)],
        "hyperparams": Hyperparams(
            total_timesteps=5e5,
            num_envs=4,
            rollout_len=128,
            num_minibatches=4,
            update_epochs=4,

            discount_rate=0.99,
            advantage_rate=0.95,
            
            actor_learning_rate=2.5e-4,
            actor_clip=0.2,
            nested_updates=10,
            critic_learning_rate=2.5e-4,
            adam_eps=1e-5,
            nystrom_rank= 10,
            nystrom_rho = 50,
        ),
    },
    # "catch": {
    #     "actor_model": DiscreteActor,
    #     "actor_params": [(30, 15), DynParam.ActionCount],
    #     "critic_model": Critic,
    #     "critic_params": [(30, 15)],
    #     "hyperparams": Hyperparams(
    #         num_updates=1000,
    #         batch_count=50,
    #         rollout_len=1000,
    #         discount_rate=0.99,
    #         advantage_rate=0.95,
    #         num_minibatches=5,
    #         update_epochs=4,
    #         actor_learning_rate=0.0004,
    #         actor_clip=0.2,
    #         nested_updates=10,
    #         critic_learning_rate=0.01,
    #         adam_eps=1e-5,
    #         nystrom_rank= 10,
    #         nystrom_rho = 50,
    #     ),
    # },
    # "acrobot": {
    #     "actor_model": DiscreteActor,
    #     "actor_params": [(30, 15), DynParam.ActionCount],
    #     "critic_model": Critic,
    #     "critic_params": [(30, 15)],
    #     "hyperparams": Hyperparams(
    #         num_updates=1000,
    #         batch_count=50,
    #         rollout_len=1000,
    #         discount_rate=0.99,
    #         advantage_rate=0.95,
    #         num_minibatches=5,
    #         update_epochs=4,
    #         actor_learning_rate=0.0004,
    #         actor_clip=0.2,            
    #         nested_updates=10,
    #         critic_learning_rate=0.01,
    #         adam_eps=1e-5,
    #         nystrom_rank= 10,
    #         nystrom_rho = 50,
    #     ),
    # },
    # "asterix": {
    #     "actor_model": DiscreteActor,
    #     "actor_params": [(256, 256), DynParam.ActionCount],
    #     "critic_model": Critic,
    #     "critic_params": [(256, 256)],
    #     "hyperparams": Hyperparams(
    #         num_updates=1000,
    #         batch_count=50,
    #         rollout_len=1000,
    #         discount_rate=0.999,
    #         advantage_rate=0.95,
    #         num_minibatches=5,
    #         update_epochs=4,
    #         actor_learning_rate=5e-04,
    #         actor_clip=0.5,            
    #         nested_updates=10,
    #         critic_learning_rate=5e-04,
    #         adam_eps=1e-5,
    #         nystrom_rank= 10,
    #         nystrom_rho = 50,
    #     ),
    # },
    # "umbrella": {
    #     "actor_model": DiscreteActor,
    #     "actor_params": [(30, 15), DynParam.ActionCount],
    #     "critic_model": Critic,
    #     "critic_params": [(30, 15)],
    #     "hyperparams": Hyperparams(
    #         num_updates=1000,
    #         batch_count=50,
    #         rollout_len=1000,
    #         discount_rate=0.99,
    #         advantage_rate=0.95,
    #         num_minibatches=5,
    #         update_epochs=4,
    #         actor_learning_rate=0.0004,
    #         actor_clip=0.2,            
    #         nested_updates=10,
    #         critic_learning_rate=0.01,
    #         adam_eps=1e-5,
    #         nystrom_rank= 10,
    #         nystrom_rho = 50,
    #     ),
    # },
}
