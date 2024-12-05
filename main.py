import argparse
import jax
import os

from Stackelberg_RL import nystrom_ppo
from Baselines import PJax_PPO

task_dict = {
    "cartpole": "CartPole-v1",
    "acrobot": "Acrobot-v1",
    "spaceinvaders": "SpaceInvaders-MinAtar",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=bool, default = False, help = "Run on CPU")
    parser.add_argument("--task", type=str, default = "acrobot")
    parser.add_argument("--vanilla", type=bool, default = False)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--algo", type=str, default="nystrom")
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    shared_config = {
        "ENV_NAME": task_dict[args.task],

        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 5,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": True,
    }
    ppo_config= shared_config | {"LR": 2.5e-4}
    nystrom_config = shared_config | { 
        "actor-LR": 2.5e-4,
        "critic-LR" : 1e-3, 
        "nystrom_rank": 10,
        "nystrom_rho": 50,
        "nested_updates": 10,
        "IHVP_BOUND": 0.2,
        "vanilla": args.vanilla,
    }

    algos = {
        "nystrom": (nystrom_ppo, nystrom_config),
        "ppo": (PJax_PPO, ppo_config)
    }

    algo, config = algos[args.algo]
    rng = jax.random.PRNGKey(args.seed)
    train_jit = jax.jit(algo.make_train(config))
    out = train_jit(rng)

if __name__ == "__main__":
    main()