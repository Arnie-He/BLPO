import time
import argparse
import jax
import numpy as np
import wandb
from archive import cgcriticp, nystrom_criticp
from BiLevel_RL.continuous import CG, nested, nystrom, gn_nystrom, fixedratio_nystrom, anneal_nystrom, nystrom_preconditioned_cg
from BiLevel_RL.discrete import dis_CG, dis_nested, dis_nystrom
from Baselines import PJax_PPO_continuous, PJax_PPO
from config import env_config

dis_tasks = ["cartpole", "acrobot"]
continuous_tasks = ["halfcheetah", "walker2d", "humanoid", "humanoidstandup", 
                    "inverted_pendulum", "inverted_double_pendulum", "pusher", 
                    "hopper", "reacher"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--task", type=str, default="walker2d")
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--num_seeds", type=int, default=15)
    parser.add_argument("--algo", type=str, default="nystrom")
    parser.add_argument("--group_ver", type=str, default="April20")
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    shared_config = env_config[args.task]
    if args.algo == "ppo":
        algo_config = { **shared_config, "LR": 2.5e-4}
    else:
        algo_config = { **shared_config,
            "actor-LR": 2.5e-4,
            "critic-LR": 1.0e-3,
        }

    assert args.task in dis_tasks + continuous_tasks, f"{args.task} not supported."

    if args.task in dis_tasks:
        algos = {
            "nested": dis_nested,
            "cg": dis_CG,
            "ppo": PJax_PPO,
            "nystrom": dis_nystrom,
        }
    else:
        algos = {
            "nested": nested,
            "cg": CG,
            "ppo": PJax_PPO_continuous,
            "nystrom": nystrom,
            "gn_nystrom": gn_nystrom,
            "fixedratio_nystrom": fixedratio_nystrom,
            "anneal_nystrom": anneal_nystrom,
            "nystrom_preconditioned_cg": nystrom_preconditioned_cg,
        }

    algo = algos[args.algo]
    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, args.num_seeds)
    train_vjit = jax.jit(jax.vmap(algo.make_train(algo_config)))

    start_time = time.perf_counter()
    outs = train_vjit(rngs)
    # ensure all pending work finishes
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), outs["metrics"])
    total_seconds = time.perf_counter() - start_time
    # -------------------------------------

    for s in range(args.num_seeds):
        # initialize a WandB run for this seed
        group_name = f'{algo_config["ENV_NAME"]}_{args.algo}_{args.group_ver}'
        run_name   = f'{algo_config["ENV_NAME"]}_{args.algo}_Seed{s}'
        run = wandb.init(
            project="Neurlips-BLPO",
            group=group_name,
            name=run_name,
            config=algo_config
        )
        wandb.define_metric("Episodic Return", summary="mean")
        wandb.define_metric("runtime_seconds", summary="max")

        # log the same total_seconds for each seed (or divide by num_seeds if you prefer)
        run.summary["runtime_seconds"] = total_seconds

        # extract this seed’s metrics
        metrics = jax.tree_util.tree_map(
            lambda x: np.array(x.block_until_ready()),
            jax.tree_util.tree_map(lambda x: x[s], outs["metrics"])
        )

        # log returns at each env-step
        num_updates = metrics["returned_episode"].shape[0]
        for u in range(num_updates):
            ended_mask = metrics["returned_episode"][u]
            returns    = metrics["returned_episode_returns"][u][ended_mask]
            timesteps  = metrics["timestep"][u][ended_mask] * algo_config["NUM_ENVS"]
            for ret, step in zip(returns, timesteps):
                wandb.log({"Episodic Return": float(ret)}, step=int(step), commit=False)

        # finalize logs for this run
        run.finish()

if __name__ == "__main__":
    main()