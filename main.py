import argparse
import jax
from algos import discrete_actor_critic, discrete_ppo, discrete_reinforce, stackelberg_discrete_a2c, stackelberg_EQpropagation
from loggers.chart_logger import ChartLogger

def run_on_cpu():
    jax.config.update("jax_platform_name", "cpu")

algos = {
    "actor_critic": discrete_actor_critic,
    "ppo": discrete_ppo,
    "reinforce": discrete_reinforce,
    "Stackelberg": stackelberg_discrete_a2c,
    "STEQ": stackelberg_EQpropagation,
}

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run RL algorithms with optional configurations")
    parser.add_argument("--cpu", default=False, action="store_true", help="Run on CPU")
    parser.add_argument("--task", type=str, default="cartpole", help="Specify the environment/task")
    parser.add_argument("--algo", type=str, default="Stackelberg", choices=algos.keys(), help="Specify the algorithm")

    args = parser.parse_args()

    # Run on CPU if specified
    if args.cpu:
        run_on_cpu()

    logger = ChartLogger(("reward", "actor_loss", "critic_loss"))
    algo = algos[args.algo]
    algo.train(args.task, 0, logger, verbose=True)

    # Plot metrics
    logger.plot_metric("reward")
    logger.plot_metric("actor_loss")
    logger.plot_metric("critic_loss")

if __name__ == "__main__":
    main()
