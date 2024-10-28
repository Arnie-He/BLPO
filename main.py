import argparse
import jax
from algos.baselines import discrete_actor_critic, discrete_ppo, discrete_reinforce, actor_critic_NoNesting
from algos.StackelbergRL import stac_Actor, stac_Critic, stac_critic, stac_Actor_newGrad
from bilevel_actor_critic import unrolling_actor_redo, lambda_regret
from loggers.chart_logger import ChartLogger
from algos.core.config import ALGO_CONFIG
import os

def run_on_cpu():
    jax.config.update("jax_platform_name", "cpu")

algos = {
    "a2c_no_nest": actor_critic_NoNesting,
    "actor_critic": discrete_actor_critic,
    "ppo": discrete_ppo,
    "reinforce": discrete_reinforce,
    "ratliff": stac_Actor,
    "stac-actor": stac_Actor_newGrad,
    "stac-critic": stac_critic,
    "unrolling": unrolling_actor_redo,
    "penalty": lambda_regret,
}

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run RL algorithms with optional configurations")
    parser.add_argument("--cpu", default=False, action="store_true", help="Run on CPU")
    parser.add_argument("--task", type=str, default="cartpole", help="Specify the environment/task")
    parser.add_argument("--algo", type=str, default="stac-actor", choices=algos.keys(), help="Specify the algorithm")
    parser.add_argument("--plot", type=bool, default=False)

    args = parser.parse_args()

    # Run on CPU if specified
    if args.cpu:
        run_on_cpu()

    # Define the metrics to log for 
    metrics = [
        "reward",
        "actor_loss",
        "critic_loss"
    ]
    logger = ChartLogger(metrics)

    config = ALGO_CONFIG[args.algo]
    description = config["description"]

    folder_path = f"charts/{args.algo}/{args.task}_{description}"
    for metric in metrics:
            file_path = f"{folder_path}/{args.task}_{metric}.png"
            
            logger.set_info(
                metric,
                f"[{args.task}] SA2C {metric}",
                file_path,
            )

    algo = algos[args.algo]
    algo.train(args.task, 0, logger, verbose=True)
    # Ensure the data directory for the task exists
    os.makedirs(f'data/{args.task}', exist_ok=True)
    logger.log_to_csv(f'data/{args.task}/{args.algo}_{description}.csv')

    # Plot metrics
    if(args.plot):
        os.makedirs(folder_path, exist_ok=True)
        for m in metrics:
            logger.plot_metric(m)

if __name__ == "__main__":
    main()
