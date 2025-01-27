import argparse
import jax
import os

from Stackelberg_RL.continuous import CG_ppo, nystrom_ppo, nested_ppo
from Baselines import PJax_PPO_continuous, pure_env_continuous
from Stackelberg_RL.continuous.archived import natural_ppo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=bool, default = False, help = "Run on CPU")
    # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    parser.add_argument("--task", type=str, default = "hopper")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--algo", type=str, default="nystrom")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--rho", type=int, default=50)
    parser.add_argument("--nested", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--ihvp", type=float, default=1.0)
    parser.add_argument("--steps", type=float, default=1.5e7, help="Total number of steps (can use scientific notation)")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--clipf", type=float, default=0.5)
    parser.add_argument("--group", type=str, default = "G0")
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    shared_config = {
        "ENV_NAME": args.task,

        "SEED": args.seed,

        "NUM_ENVS": 32,
        "NUM_STEPS": 640,
        "TOTAL_TIMESTEPS": int(args.steps),
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": args.clip,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        # "MAX_GRAD_NORM": 0.5, 
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False, # ALWAYS FALSE!
        "NORMALIZE_ENV": True,
        "DEBUG": True,

        "Group": args.group,
    }
    ppo_config= shared_config | {"LR": 3e-4} # LEARNING RATE?
    nested_shared_config = shared_config | {
        "actor-LR": 3e-4,
        "critic-LR" : 1.2e-3, 

        "nested_updates": args.nested,
        "CLIP_F": args.clipf,
        "IHVP_BOUND": args.ihvp,
    }   
    nystrom_config = nested_shared_config | { 
        "nystrom_rank": args.rank,
        "nystrom_rho": args.rho,
    }
    cg_config = nested_shared_config | {
        "lambda_reg": args.lam,
    }

    algos = {
        "nystrom": (nystrom_ppo, nystrom_config),
        "nested": (nested_ppo, nested_shared_config),
        "cg": (CG_ppo, cg_config),
        "ppo": (PJax_PPO_continuous, ppo_config),
        "env": (pure_env_continuous, ppo_config),
        # later.
        "natural": (natural_ppo, nystrom_config),
    }

    algo, config = algos[args.algo]
    rng = jax.random.PRNGKey(args.seed)
    train_jit = jax.jit(algo.make_train(config))
    out = train_jit(rng)

if __name__ == "__main__":
    main()