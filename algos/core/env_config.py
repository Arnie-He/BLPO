# config.py

from models.discrete_actor import DiscreteActor, DiscretePixelActor
from models.critic import Critic, PixelCritic
from models.params import DynParam
import flax.linen as nn
from algos.core.hyperparams import Hyperparams

ENV_CONFIG = {
    "cartpole": {
        "actor_model": DiscreteActor,
        "actor_params": [(15, 10), DynParam.ActionCount],  # Two hidden layers with 64 units each
        "critic_model": Critic,
        "critic_params": [(15, 10)],
        "hyperparams": Hyperparams(
            num_updates=500,
            batch_count=25,
            rollout_len=2000,
            discount_rate=0.99,
            actor_learning_rate=0.0025,
            nested_updates=25,
            critic_learning_rate=0.008,
            adam_eps=1e-5,
        ),
    },

    "acrobot": {
        "actor_model": DiscreteActor,
        "actor_params": [(15, 10), DynParam.ActionCount],  # Two hidden layers with 64 units each
        "critic_model": Critic,
        "critic_params": [(15, 10)],
        "hyperparams": Hyperparams(
            num_updates=1000,             
            batch_count=25,               # Number of mini-batches
            rollout_len=2000,            
            discount_rate=0.99,          
            actor_learning_rate=0.0025,   
            critic_learning_rate=0.008,
            nested_updates=25,          
            adam_eps=1e-5,
        ),
    },

    "catch": {
        "actor_model": DiscreteActor,
        "actor_params": [(15, 15), DynParam.ActionCount],  
        "critic_model": Critic,
        "critic_params": [(15, 15)],
        "hyperparams": Hyperparams(
            num_updates=500,             
            batch_count=25,               # Number of mini-batches
            rollout_len=2000,            
            discount_rate=0.99,          
            actor_learning_rate=0.0025,   
            critic_learning_rate=0.008,
            nested_updates=25,          
            adam_eps=1e-5,
        ),
    },
}
