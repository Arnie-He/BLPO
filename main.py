import argparse
import jax
import os

from algos.StackelbergRL import stac_Actor_Nystrom_Gym, stac_Actor_Nystrom_Jax, StackelbergPPO
from algos.StackelbergRL import stac_Actor_CG
# from algos.StackelbergRL import ratliff, stac_critic, stac_Critic
from algos.baselines import discrete_actor_critic, discrete_ppo, discrete_reinforce
# from algos.baselines import actor_critic_NoNesting
# from bilevel_actor_critic import unrolling_actor_redo, lambda_regret

from loggers.chart_logger import ChartLogger
from algos.core.env_config import ENV_CONFIG
from algos.core.env_config import Hyperparams

from dataclasses import replace

def run_on_cpu():
    jax.config.update("jax_platform_name", "cpu")

algos = {
    # "a2c_no_nest": actor_critic_NoNesting,
    "actor_critic": discrete_actor_critic,
    "ppo": discrete_ppo,
    "reinforce": discrete_reinforce,
    # "ratliff": ratliff,
    "stac-actor": stac_Actor_Nystrom_Jax,
    "stppo": StackelbergPPO,
    # "stac-actor": stac_Actor_newGrad_CG,
    # "stac-critic": stac_critic,
    # "unrolling": unrolling_actor_redo,
    # "penalty": lambda_regret,
}

def main():
    ########################### Set up argument parsing ###########################
    parser = argparse.ArgumentParser(description="Run RL algorithms with optional configurations")
    parser.add_argument("--cpu", default=False, action="store_true", help="Run on CPU")
    parser.add_argument("--task", type=str, default="cartpole", help="Specify the environment/task")
    parser.add_argument("--algo", type=str, default="stac-actor", choices=algos.keys(), help="Specify the algorithm")
    parser.add_argument("--plot", type=bool, default=False)
    parser.add_argument("--description", type=str, default="", help="describe the variation of algorithm running")
    parser.add_argument("--vanilla", type=bool, default=False)

    # Hyperparameters to sweep
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--actor_learning_rate', type=float, default=0.003, help='Actor learning rate')
    parser.add_argument('--critic_learning_rate', type=float, default=0.008, help='Critic learning rate')
    parser.add_argument('--nested_updates', type=int, default=25, help='Number of nested updates')
    parser.add_argument('--advantage_rate', type=float, default=0.95, help='Advantage rate')
    parser.add_argument('--nystrom_rank', type=int, default=100, help='Nystrom rank')
    parser.add_argument('--nystrom_rho', type=int, default=50, help='Nystrom rho')

    args = parser.parse_args()

    ########################### Prepare the metrics to log the data ##########################
    if args.cpu:
        run_on_cpu()
    ST=True
    if(args.algo == "stac-actor" or args.algo == "ratliff" or args.algo=="stppo"):
        ST=False
    if not(ST):
        metrics = [
            "reward",
            "actor_loss",
            "critic_loss",
            "hypergradient",
            "final_product",
            "cosine_similarities"
        ] 
    else:
        metrics = [
            "reward",
            "actor_loss",
            "critic_loss"
        ]
    

    ########################## Logging Dir ##########################
    logger = ChartLogger(metrics)
    folder_path = f"charts/{args.algo}/{args.task}_{args.description}"
    for metric in metrics:
            file_path = f"{folder_path}/{args.task}_{metric}.png"
            
            logger.set_info(
                metric,
                f"[{args.task}] SA2C {metric}",
                file_path,
            )

    ####### Prepare the hyperparams for the algo ############
    hyperparams = ENV_CONFIG[args.task]['hyperparams']
    hyperparams = replace(
        hyperparams,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        nested_updates=args.nested_updates,
        advantage_rate=args.advantage_rate,
        nystrom_rank=args.nystrom_rank,
        nystrom_rho=args.nystrom_rho,
    )

    # Set args.description if it's empty
    if args.description == "":
        args.description = (
            f"actorlr{args.actor_learning_rate}_"
            f"criticlr{args.critic_learning_rate}_"
            f"nested{args.nested_updates}_"
            f"adv{args.advantage_rate}_"
            f"nystromrank{args.nystrom_rank}_"
            f"nystromrho{args.nystrom_rho}"
        )

    ###################### Start Training ########################
    algo = algos[args.algo]
    os.makedirs(f'data/{args.task}', exist_ok=True)
    if not(ST):
        algo.train(args.task, args.seed, logger, hyperparams, verbose=True, vanilla=args.vanilla)
    else:
        algo.train(args.task, args.seed, logger, hyperparams, verbose=True)
    logger.log_to_csv(f'data/{args.task}/{args.algo}_{args.description}.csv')

    ####################### Plot the Metrics #######################
    if(args.plot):
        os.makedirs(folder_path, exist_ok=True)
        for m in metrics:
            logger.plot_metric(m)

if __name__ == "__main__":
    main()
