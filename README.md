The PPO implementation includes and excludes several design choices. We keep those that are central to the performance, while we also do our best to simply those that doesn't have theoretical support or even have evidence to harm the performance. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ is a great source for the PPO details, and we will list our specific choices in the implementation:

1. Separate cctor and critic networks: primarily for the ease of future development in second order information.
2. Excludes annealing learning rates & clipping global norm: They were shown to be trivial to the performance, and they could influence with our analysis of the impact for second-order hypergradients. They also add work to our ablation study.
3. 