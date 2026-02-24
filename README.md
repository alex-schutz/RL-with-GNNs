# Reinforcement Learning using Graph Neural Networks

This is the companion repository to the [blogpost](https://iclr-blogposts.github.io/2026/blog/2026/rl-with-gnns/) containing an introduction to using Graph Neural Networks (GNNs) in Reinforcement Learning (RL) architectures. 
We provide an implementation of a custom PPO policy, compatible with the Stable Baselines3 library, based on graph neural networks implemented using PyTorch Geometric.

## Installation
To install the required dependencies, run:
```bash
uv sync
```
Or use your preferred method to install the packages listed in `pyproject.toml`.

## Usage

To train an RL agent using GNNs, run the `train.py` script:
```bash
python train.py
```

This will train an agent on the MVC environment using a GNN-based policy.
To visualise the training output, launch TensorBoard:
```bash
tensorboard --logdir runs/
```

## Customisation

To update the GNN architecture used in the policy, modify the `network_kwargs` in the `train.py` configuration section.
For example, to use a GraphSAGE network with 3 layers, you can set:
```python
"network_kwargs": {"network": "GraphSAGE", "num_layers": 3},
```

You can also test with a different environment by changing the `env` parameter in the config.
Available environments include `MVCEnv-v0`, and `TSPEnv-v0`. 
At this stage, the GCN and GraphSAGE networks do not support edge features, so they will not work on the TSP environment.

This repo is provided as a starting point for experimenting with GNNs in RL.
Please feel free to experiment by adding new architectures, environments, or training algorithms!
