Space_Invaders_config = {
        "actor-LR": 2.5e-4,
        "critic-LR" : 0.01, 
        "NUM_ENVS": 64,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 5,
        "NUM_MINIBATCHES": 64,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "SpaceInvaders-MinAtar",
        "ANNEAL_LR": True,
        "DEBUG": True,

        "nystrom_rank": 10,
        "nystrom_rho": 50,
        "nested_updates": 10,
        "IHVP_BOUND": 0.1
    }

CartPole_config = {
        "actor-LR": 2.5e-4,
        "critic-LR" : 1e-3, 
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 2e5,
        "UPDATE_EPOCHS": 5,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True,
        "DEBUG": True,

        "nystrom_rank": 10,
        "nystrom_rho": 50,
        "nested_updates": 10,
        "IHVP_BOUND": 0.1
    }