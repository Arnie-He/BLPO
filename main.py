import argparse
from archive import cgcriticp, nystrom_criticp
import jax
import os
import numpy as np
import wandb

from BiLevel_RL.continuous import CG, nested, nystrom
from BiLevel_RL.discrete import dis_CG, dis_nested, dis_nystrom
from Baselines import PJax_PPO_continuous, pure_env_continuous, PJax_PPO
from config import env_config

dis_tasks = ["cartpole", "acrobot"]
continuous_tasks = ["walker2d", "humanoid", "humanoidstandup", "inverted_pendulum", "inverted_double_pendulum", "pusher", "hopper", "reacher"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=bool, default = False, help = "Run on CPU")
    # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
    parser.add_argument("--task", type=str, default = "walker2d")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--num_seeds", type=int, default=15)
    parser.add_argument("--algo", type=str, default="nystrom")
    parser.add_argument("--group_ver", type=str, default="April20")

    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    shared_config = env_config[args.task]
    if(args.algo == "ppo"):
        algo_config = shared_config |  {"LR": 3e-4} 
    else:
        algo_config = shared_config | {
            "actor-LR": 3e-4,
            "critic-LR" : 1.2e-3,
        }
    
    assert (args.task in dis_tasks or args.task in continuous_tasks), f"{args.task} not in available tasks."
    if (args.task in dis_tasks): # Discrete Control Task!
        algos = {
            "nested": dis_nested,
            "cg": dis_CG,
            "ppo": PJax_PPO,
            # "env": pure_env_continuous,
            "nystrom": dis_nystrom,
        }
    else: # Brax Suite Tasks!
        algos = {
            "nested": nested,
            "cg": CG,
            "ppo": PJax_PPO_continuous,
            # "env": pure_env_continuous,
            "nystrom": nystrom,
            "nystromcriticp": nystrom_criticp,
            "cgcriticp": cgcriticp,
        }

    algo = algos[args.algo]
    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, args.num_seeds)
    train_vjit = jax.jit(jax.vmap(algo.make_train(algo_config)))
    outs = train_vjit(rngs)

    for s in range(args.num_seeds):
        # Log the metrics to wandb
        metrics = jax.tree_util.tree_map(
            lambda x: np.array(x.block_until_ready()),
            jax.tree_util.tree_map(lambda x: x[s], outs["metrics"])
        )
        ### Weight and Bias Setup ###
        group_name = f'{algo_config["ENV_NAME"]}_{args.algo}_{args.group_ver}'
        run_name = f'{algo_config["ENV_NAME"]}_{args.algo}_Seed{rngs[s]}'
        wandb.init(project="Neurlips-BLPO", group=group_name, name=run_name, config = algo_config)
        wandb.define_metric("Reward", summary="mean")
        num_updates = metrics["returned_episode"].shape[0]
        for u in range(num_updates):
            ended_mask = metrics["returned_episode"][u]
            returns   = metrics["returned_episode_returns"][u][ended_mask]
            timesteps = metrics["timestep"][u][ended_mask] * algo_config["NUM_ENVS"]
            for ret, step in zip(returns, timesteps):
                wandb.log(
                    {"Episodic Return": float(ret)},
                    step=int(step)
                )
        wandb.finish()

if __name__ == "__main__":
    main()