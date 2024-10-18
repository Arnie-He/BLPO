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
            actor_learning_rate=0.0025,
            nested_updates=10,
            critic_learning_rate=0.004,
            adam_eps=1e-5,
        ),
    },
    "pendulum": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],  # Two hidden layers with 64 units each
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=500,             
            batch_count=25,               # Number of mini-batches
            rollout_len=2000,            
            discount_rate=0.99,          
            actor_learning_rate=0.0025,   
            critic_learning_rate=0.004,
            nested_updates=25,          
            adam_eps=1e-5,
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
            critic_learning_rate=0.004,
            nested_updates=25,          
            adam_eps=1e-5,
        ),
    },


    "4rooms": {
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
            critic_learning_rate=0.004,
            nested_updates=25,          
            adam_eps=1e-5,
        ),
    },
    "pong":{
        "actor_model": DiscretePixelActor,
        "actor_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
            DynParam.ActionCount,
        ],
        "critic_model": PixelCritic,
        "critic_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
        ],
        "hyperparams": Hyperparams(
            num_updates=500,
            batch_count=50,
            rollout_len=2000,
            discount_rate=0.995,
            actor_learning_rate=0.0015,
            critic_learning_rate=0.003,
            nested_updates=20,
            adam_eps=1e-5,
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
            actor_learning_rate=0.0025,   
            critic_learning_rate=0.004,
            nested_updates=25,          
            adam_eps=1e-5,
        ),
    },
    "deepsea": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],  
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=1000,             
            batch_count=25,               # Number of mini-batches
            rollout_len=2000,            
            discount_rate=0.99,          
            actor_learning_rate=0.0025,   
            critic_learning_rate=0.004,
            nested_updates=25,          
            adam_eps=1e-5,
        ),
    },
    "deepsea": {
        "actor_model": DiscreteActor,
        "actor_params": [(30, 15), DynParam.ActionCount],  
        "critic_model": Critic,
        "critic_params": [(30, 15)],
        "hyperparams": Hyperparams(
            num_updates=500,             
            batch_count=25,               # Number of mini-batches
            rollout_len=2000,            
            discount_rate=0.99,          
            actor_learning_rate=0.0025,   
            critic_learning_rate=0.004,
            nested_updates=25,          
            adam_eps=1e-5,
        ),
    },
    

    "space_invaders": {
        "actor_model": DiscretePixelActor,
        "actor_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
            DynParam.ActionCount,
        ],
        "critic_model": PixelCritic,
        "critic_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
        ],
        "hyperparams": Hyperparams(
            num_updates=2000,
            batch_count=50,
            rollout_len=2000,
            discount_rate=0.995,
            actor_learning_rate=0.0015,
            critic_learning_rate=0.003,
            nested_updates=20,
            adam_eps=1e-5,
        ),
    },
    "breakout": {
        "actor_model": DiscretePixelActor,
        "actor_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
            DynParam.ActionCount,
        ],
        "critic_model": PixelCritic,
        "critic_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
        ],
        "hyperparams": Hyperparams(
            num_updates=2000,
            batch_count=50,
            rollout_len=2000,
            discount_rate=0.995,
            actor_learning_rate=0.0015,
            critic_learning_rate=0.003,
            nested_updates=20,
            adam_eps=1e-5,
        ),
    },
    "asterix": {
        "actor_model": DiscretePixelActor,
        "actor_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
            DynParam.ActionCount,
        ],
        "critic_model": PixelCritic,
        "critic_params": [
            (
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                lambda data: nn.avg_pool(data, window_shape=(2, 2), strides=(2, 2)),
            ),
            (256,),
        ],
        "hyperparams": Hyperparams(
            num_updates=2000,
            batch_count=50,
            rollout_len=2000,
            discount_rate=0.995,
            actor_learning_rate=0.0015,
            critic_learning_rate=0.003,
            nested_updates=20,
            adam_eps=1e-5,
        ),
    },
}
