env_config = {
    ### LACK: CartPole, Acrobat, Halfcheetah, reacher, and potentially MinAtar
    "cartpole": {
        "TOTAL_TIMESTEPS": 5e5,
        # General Config
        "ENV_NAME": "CartPole-v1",
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False, 
        "NORMALIZE_ENV": True,
        "DEBUG": True,

        # Nested Config
        "nested_updates": 5,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,


        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    "acrobot": {
        "TOTAL_TIMESTEPS": 5e5,
        # General Config
        "ENV_NAME": "Acrobot-v1",
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False, 
        "NORMALIZE_ENV": True,
        "DEBUG": True,

        # Nested Config
        "nested_updates": 5,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },

    # 1
    "walker2d": {
        "TOTAL_TIMESTEPS": 2e7,

        # General Config
        "ENV_NAME": "walker2d",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 2
    "hopper": {
        "TOTAL_TIMESTEPS": 1.2e7,

        # General Config
        "ENV_NAME": "hopper",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 3
    "humanoid": {
        "TOTAL_TIMESTEPS": 2e7,

        # General Config
        "ENV_NAME": "humanoid",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 4
    "halfcheetah": {
        "TOTAL_TIMESTEPS": 2e7,

        # General Config
        "ENV_NAME": "halfcheetah",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 5
    "humanoidstandup": {
        "TOTAL_TIMESTEPS": 8e6,

        # General Config
        "ENV_NAME": "humanoidstandup",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 6
    "pusher": {
        "TOTAL_TIMESTEPS": 8e6,

        # General Config
        "ENV_NAME": "pusher",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 7
    "inverted_double_pendulum": {
        "TOTAL_TIMESTEPS": 8e6,

        # General Config
        "ENV_NAME": "inverted_double_pendulum",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 8
    "inverted_pendulum": {
        "TOTAL_TIMESTEPS": 1.2e7,

        # General Config
        "ENV_NAME": "inverted_pendulum",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
    # 9
    "reacher": {
        "TOTAL_TIMESTEPS": 8e6,

        # General Config
        "ENV_NAME": "reacher",
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
        "DEBUG": False,
        
        # Nested Config
        "nested_updates": 3,
        "CLIP_F": 0.5,
        "IHVP_BOUND": 1.,
        "IHVP_ratio": 0.2,

        # Nystrom Config
        "nystrom_rank": 5,
        "nystrom_rho": 50,

        # CG Config
        "lambda_reg": 0.0,
    },
}