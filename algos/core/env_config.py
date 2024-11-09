# config.py

from models.discrete_actor import DiscreteActor, DiscretePixelActor
from models.critic import Critic, PixelCritic
from models.params import DynParam
import flax.linen as nn
from algos.core.hyperparams import Hyperparams

ENV_CONFIG = {
    "cartpole": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],  # Two hidden layers with 64 units each
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=500,
            batch_count=25,
            rollout_len=2000,
            discount_rate=0.99,
            actor_learning_rate=0.003,
            nested_updates=25,
            critic_learning_rate=0.008,
            adam_eps=1e-5,
            advantage_rate=0.95,
            nystrom_rank= 100,
            nystrom_rho = 50,
        ),
    },

    "acrobot": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],  # Two hidden layers with 64 units each
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=1000,             
            batch_count=25,               # Number of mini-batches
            rollout_len=2000,            
            discount_rate=0.99,          
            actor_learning_rate=0.0025,   
            critic_learning_rate=0.008,
            nested_updates=25,          
            adam_eps=1e-5,
            advantage_rate=0.95,
            nystrom_rank= 50,
            nystrom_rho = 50,
        ),
    },

    "catch": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],  
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=500,             
            batch_count=25,               # Number of mini-batches
            rollout_len=2000,            
            discount_rate=0.99,          
            actor_learning_rate=0.003,   
            critic_learning_rate=0.008,
            nested_updates=25,          
            adam_eps=1e-5,
            advantage_rate=0.95,
            nystrom_rank= 100,
            nystrom_rho = 50,
        ),
    },
}
