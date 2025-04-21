# Bi-Level Policy Optimization with Nyström Hypergradients

This repository implements Bi-Level Policy Optimization (BLPO) using Nyström hypergradients. Experiments are conducted in both discrete(gymnax) and continuous(brax) environments.

## Prerequisites

- **Python 3.11** (Tested)
- **JAX:** Follow the [JAX Installation Guide](https://docs.jax.dev/en/latest/installation.html) to set up the appropriate version for your hardware (CPU/GPU).

## Installation

1. **Install JAX**  
   Visit the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for detailed instructions on installing JAX for your system.

2. **Install Required Dependencies**  
   Install the necessary packages using pip:
   ```bash
   pip install flax==0.10.1 numpy==2.1.3 optax==0.2.3 distrax==0.1.5 gymnax==0.0.8 wandb==0.19.1 brax==0.12.1



## To Run BLPO:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
python BiLevel_RL/continuous/nystrom_ppo.py
