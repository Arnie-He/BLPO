import argparse
import jax
import os

from Stackelberg_RL.discrete import nystrom_preconditioned_cg
from Baselines import PJax_PPO
from Stackelberg_RL.discrete import CG_ppo, nystrom_ppo

task_dict = {
    "cartpole": "CartPole-v1",
    "acrobot": "Acrobot-v1",
    "spaceinvaders": "SpaceInvaders-MinAtar",
    "breakout": "Breakout-MinAtar",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=bool, default = False, help = "Run on CPU")
    parser.add_argument("--task", type=str, default = "hopper")
    parser.add_argument("--vanilla", type=bool, default = False)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--algo", type=str, default="nystrom")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--rho", type=int, default=50)
    parser.add_argument("--nested", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--ihvp_bound", type=float, default=1.0)
    parser.add_argument("--steps", type=float, default=5e6, help="Total number of steps (can use scientific notation)")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--clipf", type=float, default=0.5)
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    shared_config = {
        "ENV_NAME": task_dict[args.task],

        "SEED": args.seed,

        "NUM_ENVS": 32,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": int(args.steps),
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": args.clip,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    ppo_config= shared_config | {"LR": 2.5e-4}
    nested_shared_config = shared_config | {
        "actor-LR": 2.5e-4,
        "critic-LR" : 1e-3, 
        "nested_updates": args.nested,
        "IHVP_BOUND": args.ihvp_bound,
        "vanilla": args.vanilla,
        "CLIP_F": args.clipf
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
        "cg": (CG_ppo, cg_config),
        "ppo": (PJax_PPO, ppo_config),
        "npcg": (nystrom_preconditioned_cg, nystrom_config)
    }

    algo, config = algos[args.algo]
    rng = jax.random.PRNGKey(args.seed)
    train_jit = jax.jit(algo.make_train(config))
    out = train_jit(rng)

if __name__ == "__main__":
    main()