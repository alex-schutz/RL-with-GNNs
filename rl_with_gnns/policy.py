from gymnasium import spaces

import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Callable, Any

from rl_with_gnns.gnns import get_network_class
from rl_with_gnns.util import matrix_features_to_batch


class MatrixObservationToGraph(BaseFeaturesExtractor):
    """
    Converts matrix-based observations to graph Batch objects.

    Args:
        observation_space (spaces.Dict): The observation space.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
    ) -> None:

        features_dim = 1  # unused
        super().__init__(observation_space, features_dim=features_dim)

    def forward(self, observations) -> Batch:
        """Convert the observations to a graph Batch object."""
        node_features = observations["node_features"]
        edge_features = observations["edge_features"]
        adj_matrix = observations["adjacency_matrix"]

        batch = matrix_features_to_batch(node_features, edge_features, adj_matrix)
        return batch


class GraphActorCriticProcessor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.
    It outputs a processed graph batch for the actor and a graph embedding for the critic.

    Args:
        node_dim (int): Dimension of the node feature space.
        edge_dim (int): Dimension of the edge feature space.
        embed_dim (int): Dimension of the graph embedding space.
        pooling_type (str): Pooling type to use for graph embedding computation
            (options: "max", "mean", "sum")
        network_kwargs (dict, optional): Additional arguments to pass to the graph network.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        embed_dim: int = 64,
        pooling_type: str = "max",
        network_kwargs: dict = None,
        **kwargs,
    ):
        if pooling_type not in ["max", "mean", "sum"]:
            raise ValueError(f"Unknown pooling type {pooling_type}")

        super().__init__()
        # Save output dimensions
        # This will be used by the MaskableActorCriticPolicy to create
        # the value network (the actor network will be overriden)
        self.latent_dim_vf = embed_dim
        self.latent_dim_pi = 0  # unused

        self.embed_dim = embed_dim
        self.pooling_type = pooling_type

        if network_kwargs is None:
            network_kwargs = {}

        processor_class = get_network_class(network_kwargs["network"])

        # Create the graph processor
        self.processor = processor_class(
            in_channels=node_dim,
            out_channels=embed_dim,
            edge_dim=edge_dim,
            **network_kwargs,
        )

    def _process_graph(self, batch: Batch) -> tuple[th.Tensor, th.Tensor]:
        """Process the graph with the graph network"""

        # Process the graph with the graph network
        node_embedding = self.processor(
            node_fts=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            batch=batch.batch,
        )

        # Compute the graph embedding via pooling
        if self.pooling_type == "max":
            graph_embedding = global_max_pool(node_embedding, batch.batch)
        elif self.pooling_type == "mean":
            graph_embedding = global_mean_pool(node_embedding, batch.batch)
        elif self.pooling_type == "sum":
            graph_embedding = global_add_pool(node_embedding, batch.batch)

        # Return node and graph embeddings
        return node_embedding, graph_embedding

    def forward(self, batch: Batch) -> tuple[Batch, th.Tensor]:
        """
        Forward pass of the graph processor.

        Args:
            batch (Batch): A batch of graph data.
        Returns:
            processed_batch (Batch): The processed batch with updated features, to be passed to the downstream actor network.
            graph_embedding (th.Tensor): The graph embedding tensor, to be passed to the downstream critic network.
        """
        # Process the graph
        node_embedding, graph_embedding = self._process_graph(batch)

        # Turn the processed embeddings back into a Batch object
        processed_batch = Batch(
            x=node_embedding,
            edge_index=batch.edge_index,
            graph_attr=graph_embedding,  # store graph embedding in graph_attr
            batch=batch.batch,
        )

        return processed_batch, graph_embedding

    def forward_critic(self, x: Batch) -> th.Tensor:
        """Forward pass of the critic network."""
        return self.forward(x)[1]

    def forward_actor(self, x: Batch) -> Batch:
        """Forward pass of the actor network."""
        return self.forward(x)[0]


class ProtoActionNetwork(nn.Module):
    """
    Action network that predicts a proto-action from the graph embedding and
    uses similarity-based matching to select a node.

    Args:
        embed_dim (int): Dimension of the embedding space.
        max_nodes (int): Maximum number of nodes in the graph.
        distance_metric (str): Distance metric to use for similarity computation
            (options: "euclidean", "cosine").
        action_predictor_layers (int): Number of layers in the action predictor network.
        temp (float): Temperature parameter for the softmax.
    """

    def __init__(
        self,
        embed_dim: int,
        max_nodes: int,
        distance_metric: str = "euclidean",
        action_predictor_layers: int = 2,
        temp: float = 1.0,
    ):
        if distance_metric not in ["euclidean", "cosine"]:
            raise ValueError(f"Unknown distance metric {distance_metric}")

        super().__init__()
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        self.distance_metric = distance_metric
        self.softmax_temp = nn.Parameter(th.tensor(temp), requires_grad=True)

        # Define the action predictor network
        # Input will be the graph embedding, output will be the proto-action
        self.action_predictor = nn.Sequential(
            *(
                [nn.Linear(self.embed_dim, self.embed_dim)]
                + [nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim)]
                * (action_predictor_layers - 1)
            )
        )

    def compute_embedding_similarities(self, embedded_acts, proto_action):
        """Compute similarities between the embedded actions and the proto-action."""

        if self.distance_metric == "euclidean":
            similarities = -th.cdist(embedded_acts, proto_action, p=2).squeeze(-1)

        elif self.distance_metric == "cosine":
            similarities = F.cosine_similarity(embedded_acts, proto_action, dim=-1)
        else:
            raise ValueError(f"unknown distance metric {self.distance_metric}")

        return similarities

    def forward(self, batch: Batch) -> th.Tensor:
        """Forward pass of the action network.
        This method uses the graph embedding to create a proto-action by passing it through an MLP.
        Action logits are then computed as similarities between the proto-action and the node embeddings.

        Args:
            batch (Batch): A batch of graph data. Required attributes are:
                - x: Node embeddings (num_nodes, embed_dim)
                - graph_attr: Graph embeddings (batch_size, embed_dim)
                - batch: Batch vector mapping each node to its respective graph in the batch (num_nodes,)

        Returns:
            th.Tensor: (b, n) matrix of similarities between the graph embedding and the node embeddings,
                where b is the batch size and n is the number of nodes.
        """

        node_embedding = batch.x
        graph_embedding = batch.graph_attr

        # Create the proto-action from the graph embedding
        proto_action = self.action_predictor(graph_embedding)

        # Compute similarities between the graph embedding and the node embeddings per batch
        similarities = th.zeros_like(batch.batch, dtype=th.float32)
        unique_batches = batch.batch.unique()
        for batch_id in unique_batches:
            batch_mask = batch.batch == batch_id
            batch_embeddings = node_embedding[batch_mask]
            batch_target = proto_action[batch_id].unsqueeze(0)  # Target for this batch
            batch_similarities = self.compute_embedding_similarities(
                batch_embeddings, batch_target
            )
            similarities[batch_mask] = batch_similarities

        # Scale similarities by temperature
        similarities = similarities / self.softmax_temp

        # Reshape similarities along the batch dimension
        similarities = to_dense_batch(
            similarities.unsqueeze(-1),
            batch.batch,
            fill_value=-1e9,  # large negative value for padding
            max_num_nodes=self.max_nodes,
        )[0]

        return similarities


class MaskableGraphActorCriticPolicy(MaskableActorCriticPolicy):
    """
    Custom Actor-Critic Policy with a custom feature extractor and network architecture.

    Args:
        observation_space (spaces.Dict): The observation space.
        action_space (spaces.Discrete): The action space.
        lr_schedule (Callable[[float], float]): Learning rate schedule.
        node_dim (int): Dimension of the node feature space.
        edge_dim (int): Dimension of the edge feature space.
        embed_dim (int): Dimension of the embedding space.
        pooling_type (str): Pooling type to use for graph embedding computation
            (options: "max", "mean", "sum").
        distance_metric (str): Distance metric to use for similarity computation
            (options: "euclidean", "cosine").
        temp (float): Temperature parameter for the softmax.
        network_kwargs (dict, optional): Additional arguments to pass to the graph network.
        *args: Additional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        lr_schedule: Callable[[float], float],
        node_dim: int,
        edge_dim: int,
        embed_dim: int = 64,
        pooling_type: str = "max",
        distance_metric: str = "euclidean",
        temp: float = 1.0,
        network_kwargs: dict = None,
        *args,
        **kwargs,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.pooling_type = pooling_type
        self.distance_metric = distance_metric
        self.temp = temp
        self.network_kwargs = network_kwargs

        kwargs.setdefault("features_extractor_class", MatrixObservationToGraph)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        # override default SB3 action net to use proto-action method
        self.action_net = ProtoActionNetwork(
            embed_dim=self.embed_dim,
            max_nodes=self.action_space.n,
            distance_metric=self.distance_metric,
            temp=self.temp,
        )

    # Override the mlp extractor to use our graph processor
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GraphActorCriticProcessor(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            embed_dim=self.embed_dim,
            pooling_type=self.pooling_type,
            network_kwargs=self.network_kwargs,
        )

    # Override to save custom parameters
    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                embed_dim=self.embed_dim,
                pooling_type=self.pooling_type,
                distance_metric=self.distance_metric,
                temp=self.temp,
                network_kwargs=self.network_kwargs,
            )
        )
        return data
