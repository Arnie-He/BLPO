group = "default"
seed = 0
brax_env_config = {
    "walker2d": {
        "SEED": seed,
        "NUM_ENVS": 32,  
        "NUM_STEPS": 640,  
        "UPDATE_EPOCHS": 4,  
        "NUM_MINIBATCHES": 32,  
        "GAMMA": 0.99,  
        "GAE_LAMBDA": 0.95,  
        "CLIP_EPS": 0.2,  
        "ENT_COEF": 0.0,  
        "VF_COEF": 0.5,  
        "ACTIVATION": "tanh",  
        "ANNEAL_LR": False,  
        "NORMALIZE_ENV": True,  
        "DEBUG": True,  
        "Group": group,  
        "actor-LR": 3e-4,  
        "critic-LR": 1.2e-3,  
        
        # Nystrom Specific Parameters
        "vanilla": {"value": False},  
        "nystrom_rank": {"value": 5},  
        "nystrom_rho": {"value": 50},  

        "TOTAL_TIMESTEPS": 2e7,  
        "nested_updates": 3,  
        "IHVP_BOUND": 0.8,  
        "CLIP_F": 0.84,  
    },
    "hopper": {
        "SEED": seed,
        "NUM_ENVS": 32,  
        "NUM_STEPS": 640,   
        "UPDATE_EPOCHS": 4,  
        "NUM_MINIBATCHES": 32,  
        "GAMMA": 0.99,  
        "GAE_LAMBDA": 0.95,  
        "CLIP_EPS": 0.2,  
        "ENT_COEF": 0.0,  
        "VF_COEF": 0.5,  
        "ACTIVATION": "tanh",  
        "ANNEAL_LR": False,  
        "NORMALIZE_ENV": True,  
        "DEBUG": True,  
        "Group": group,  
        "actor-LR": 3e-4,  
        "critic-LR": 1.2e-3,  
        
        # Nystrom Specific Parameters
        "vanilla": {"value": False},  
        "nystrom_rank": {"value": 5},  
        "nystrom_rho": {"value": 50},  

        "TOTAL_TIMESTEPS": 5e6, 
        "nested_updates": 3,  
        "IHVP_BOUND": 1.4,  
        "CLIP_F": 1.0,  
    },
    "humanoid": {
        "SEED": seed,
        "NUM_ENVS": 32,  
        "NUM_STEPS": 640,  
        "UPDATE_EPOCHS": 4,  
        "NUM_MINIBATCHES": 32,  
        "GAMMA": 0.99,  
        "GAE_LAMBDA": 0.95,  
        "CLIP_EPS": 0.2,  
        "ENT_COEF": 0.0,  
        "VF_COEF": 0.5,  
        "ACTIVATION": "tanh",  
        "ANNEAL_LR": False,  
        "NORMALIZE_ENV": True,  
        "DEBUG": True,  
        "Group": group,  
        "actor-LR": 3e-4,  
        "critic-LR": 1.2e-3,  
        
        # Nystrom Specific Parameters
        "vanilla": {"value": False},  
        "nystrom_rank": {"value": 5},  
        "nystrom_rho": {"value": 50},  

        "TOTAL_TIMESTEPS": 2e7,  
        "nested_updates": 5,  
        "IHVP_BOUND": 1,  
        "CLIP_F": 0.8,  
    },
    "halfcheetah": {
        "SEED": seed,
        "NUM_ENVS": 32,  
        "NUM_STEPS": 640,  
        "UPDATE_EPOCHS": 4,  
        "NUM_MINIBATCHES": 32,  
        "GAMMA": 0.99,  
        "GAE_LAMBDA": 0.95,  
        "CLIP_EPS": 0.2,  
        "ENT_COEF": 0.0,  
        "VF_COEF": 0.5,  
        "ACTIVATION": "tanh",  
        "ANNEAL_LR": False,  
        "NORMALIZE_ENV": True,  
        "DEBUG": True,  
        "Group": group,  
        "actor-LR": 3e-4,  
        "critic-LR": 1.2e-3,  
        
        # Nystrom Specific Parameters
        "vanilla": {"value": False},  
        "nystrom_rank": {"value": 5},  
        "nystrom_rho": {"value": 50},  

        "TOTAL_TIMESTEPS": 2e7,  
        "nested_updates": 7,  
        "IHVP_BOUND": 1.4,  
        "CLIP_F": 0.7,  
    },
}

classic_env_config = {
    "CartPole-v1": {
        "SEED": seed,
        "NUM_ENVS": 4,  
        "NUM_STEPS": 128,    
        "UPDATE_EPOCHS": 4,  
        "NUM_MINIBATCHES": 32,  
        "GAMMA": 0.99,  
        "GAE_LAMBDA": 0.95,  
        "CLIP_EPS": 0.2,  
        "ENT_COEF": 0.0,  
        "VF_COEF": 0.5,  
        "ACTIVATION": "tanh",  
        "ANNEAL_LR": False,  
        "NORMALIZE_ENV": True,  
        "DEBUG": True,  
        "Group": group,  
        "actor-LR": 3e-4,  
        "critic-LR": 1.2e-3,  
        
        # Nystrom Specific Parameters
        "vanilla": {"value": False},  
        "nystrom_rank": {"value": 5},  
        "nystrom_rho": {"value": 50},  

        "TOTAL_TIMESTEPS": 5e5,
        "nested_updates": 10,
        "IHVP_BOUND": 1.4,  
        "CLIP_F": 1.0,  
    }
}