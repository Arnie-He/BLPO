from algos import discrete_actor_critic
from algos import discrete_reinforce
from loggers.chart_logger import ChartLogger

logger = ChartLogger(("reward", "actor_loss", "critic_loss"))
discrete_actor_critic.train("cartpole", 0, logger, verbose=True)
#logger.plot_metric("reward")
#logger.plot_metric("actor_loss")
#logger.plot_metric("critic_loss")