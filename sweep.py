import wandb
import jax
from Stackelberg_RL.continuous import nystrom_ppo

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
    train_jit = jax.jit(nystrom_ppo.make_train(config))

    # Run the training
    out = train_jit(rng)
    
# Define the sweep configuration
sweep_config = {
    "method": "random", 
    "metric": {"goal": "maximize", "name": "reward"}, 
    "parameters": {
        "ENV_NAME": "hopper",
        "NUM_ENVS": 32,
        "NUM_STEPS": 640,
        "TOTAL_TIMESTEPS": 2e7,
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
        "Group": "sweep0",
        "actor-LR": 3e-4,
        "critic-LR" : 1.2e-3, 
        "nested_updates": {"values": [2, 3, 5, 7, 10]},
        "IHVP_BOUND": {"values": [0.8, 1.0, 1.2, 1.4, 2.0]},
        "vanilla": False,
        "nystrom_rank": 5,
        "nystrom_rho": 50,
        "CLIP_F": {"max": 1.0, "min": 0.1},
    },
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="hopper-sweep")

# Run the sweep
wandb.agent(sweep_id, function=main, count=10)