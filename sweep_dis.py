import wandb
import jax
from Stackelberg_RL.discrete import dis_nystrom_ppo

# Define the main function that will be called by the wandb agent
def main():
    # Initialize a new wandb run
    wandb.init()

    # Get the config from the sweep
    config = wandb.config

    # Set the seed for reproducibility
    seed = 0
    rng = jax.random.PRNGKey(seed)

    # JIT the training function
    train_jit = jax.jit(dis_nystrom_ppo.make_train(config))

    # Run the training
    out = train_jit(rng)

# Define the sweep configuration
sweep_config_CartPole = {
    "method": "random", 
    "metric": {"goal": "maximize", "name": "reward"}, 
    "parameters": {
        "SEED": {"value": 0},
        "ENV_NAME": {"value": "CartPole-v1"},  # Fixed value
        "NUM_ENVS": {"value": 32},  # Fixed value
        "NUM_STEPS": {"value": 640},  # Fixed value
        "TOTAL_TIMESTEPS": {"value": 5e5},  # Fixed value
        "UPDATE_EPOCHS": {"value": 4},  # Fixed value
        "NUM_MINIBATCHES": {"value": 32},  # Fixed value
        "GAMMA": {"value": 0.99},  # Fixed value
        "GAE_LAMBDA": {"value": 0.95},  # Fixed value
        "CLIP_EPS": {"value": 0.2},  # Fixed value
        "ENT_COEF": {"value": 0.0},  # Fixed value
        "VF_COEF": {"value": 0.5},  # Fixed value
        "ACTIVATION": {"value": "tanh"},  # Fixed value
        "ANNEAL_LR": {"value": False},  # Fixed value
        "NORMALIZE_ENV": {"value": True},  # Fixed value
        "DEBUG": {"value": True},  # Fixed value
        "Group": {"value": "sweep0"},  # Fixed value
        "actor-LR": {"value": 3e-4},  # Fixed value
        "critic-LR": {"value": 1.2e-3},  # Fixed value

        # Categorical parameters
        "nested_updates": {"values": [2, 3, 5, 7, 10]},  # Correct format
        "IHVP_BOUND": {"values": [0.8, 1.0, 1.2, 1.4, 2.0]},  # Correct format

        "vanilla": {"value": False},  # Fixed value
        "nystrom_rank": {"value": 5},  # Fixed value
        "nystrom_rho": {"value": 50},  # Fixed value

        # Range parameter
        "CLIP_F": {"values": [0.8, 1.0, 1.2]},  # Correct format
    },
}

sweep_config_Acrobot = {
    "method": "random", 
    "metric": {"goal": "maximize", "name": "reward"}, 
    "parameters": {
        "SEED": {"value": 0},
        "ENV_NAME": {"value": "Acrobot"},  # Fixed value
        "NUM_ENVS": {"value": 32},  # Fixed value
        "NUM_STEPS": {"value": 640},  # Fixed value
        "TOTAL_TIMESTEPS": {"value": 5e5},  # Fixed value
        "UPDATE_EPOCHS": {"value": 4},  # Fixed value
        "NUM_MINIBATCHES": {"value": 32},  # Fixed value
        "GAMMA": {"value": 0.99},  # Fixed value
        "GAE_LAMBDA": {"value": 0.95},  # Fixed value
        "CLIP_EPS": {"value": 0.2},  # Fixed value
        "ENT_COEF": {"value": 0.0},  # Fixed value
        "VF_COEF": {"value": 0.5},  # Fixed value
        "ACTIVATION": {"value": "tanh"},  # Fixed value
        "ANNEAL_LR": {"value": False},  # Fixed value
        "NORMALIZE_ENV": {"value": True},  # Fixed value
        "DEBUG": {"value": True},  # Fixed value
        "Group": {"value": "sweep0"},  # Fixed value
        "actor-LR": {"value": 3e-4},  # Fixed value
        "critic-LR": {"value": 1.2e-3},  # Fixed value

        # Categorical parameters
        "nested_updates": {"values": [2, 3, 5, 7, 10]},  # Correct format
        "IHVP_BOUND": {"values": [0.8, 1.0, 1.2, 1.4, 2.0]},  # Correct format

        "vanilla": {"value": False},  # Fixed value
        "nystrom_rank": {"value": 5},  # Fixed value
        "nystrom_rho": {"value": 50},  # Fixed value

        # Range parameter
        "CLIP_F": {"max": 1.4, "min": 0.8},  # Correct format
    },
}

# # Initialize the sweep
# sweep_id = wandb.sweep(sweep=sweep_config_CartPole, project="HyperGradient-RL")
# # Run the sweep
# wandb.agent(sweep_id, function=main, count=20)

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_config_Acrobot, project="HyperGradient-RL")
# Run the sweep
wandb.agent(sweep_id, function=main, count=20)