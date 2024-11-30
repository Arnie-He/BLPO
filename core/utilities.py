import jax
import jax.numpy as jnp

config = None

def initialize_config(cfg):
    global config
    config = cfg

def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

