import argparse
import jax
import os

from Stackelberg_RL.discrete import dis_CG_ppo, nystrom_preconditioned_cg, dis_nystrom_ppo, dis_nested_ppo
from Baselines import PJax_PPO, pure_env_discrete

task_dict = {
    "cartpole": "CartPole-v1",
    "acrobot": "Acrobot-v1",
    "spaceinvaders": "SpaceInvaders-MinAtar",
    "breakout": "Breakout-MinAtar",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=bool, default = False, help = "Run on CPU")
    parser.add_argument("--task", type=str, default = "cartpole") #### Specify the task
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--algo", type=str, default="nystrom") #### Specify the algo
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--rho", type=int, default=50)
    parser.add_argument("--nested", type=int, default=3)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--ihvp_bound", type=float, default=1.0)
    parser.add_argument("--steps", type=float, default=5e5, help="Total number of steps (can use scientific notation)")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--clipf", type=float, default=0.5)
    parser.add_argument("--group", type=str, default = "G0") #### Specify the run group
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    shared_config = {
        "ENV_NAME": task_dict[args.task],

        "SEED": args.seed,

        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": int(args.steps),
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": args.clip,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        # "MAX_GRAD_NORM": 2.0,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False, # Always False!
        "NORMALIZE_ENV": True,
        "DEBUG": True,

        "Group": args.group,
    }
    ppo_config= shared_config | {"LR": 2.5e-4}
    nested_shared_config = shared_config | {
        "actor-LR": 2.5e-4,
        "critic-LR" : 1e-3, 

        "nested_updates": args.nested,
        "CLIP_F": args.clipf,
        "IHVP_BOUND": args.ihvp_bound,
    }
    nystrom_config = nested_shared_config | { 
        "nystrom_rank": args.rank,
        "nystrom_rho": args.rho,
    }
    cg_config = nested_shared_config | {
        "lambda_reg": args.lam,
    }

    algos = {
        "nystrom": (dis_nystrom_ppo, nystrom_config),
        "nested": (dis_nested_ppo, nested_shared_config),
        "cg": (dis_CG_ppo, cg_config),
        "ppo": (PJax_PPO, ppo_config),
        "env": (pure_env_discrete, ppo_config),
        # Later
        "npcg": (nystrom_preconditioned_cg, nystrom_config)
    }

    algo, config = algos[args.algo]
    rng = jax.random.PRNGKey(args.seed)
    train_jit = jax.jit(algo.make_train(config))
    out = train_jit(rng)

if __name__ == "__main__":
    main()