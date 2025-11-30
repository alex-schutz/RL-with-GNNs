import gymnasium as gym
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import torch as th


class TSPEnv(gym.Env):
    def __init__(self, split: str, seed: int):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        super(TSPEnv, self).__init__()

        self.split = split
        self.graph_rng = np.random.default_rng(seed)

        if split == "train":
            self.num_graphs = 1000
            self.graph_sizes = [5, 10, 15]
        elif split == "val":
            self.num_graphs = 20
            self.graph_sizes = [15]
        else:
            self.num_graphs = 100
            self.graph_sizes = [15]

        self.max_nodes = max(self.graph_sizes)
        self.graphs = self._generate_graph_set()
        self.current_graph_index = 0

        self.action_space = gym.spaces.Discrete(self.max_nodes)
        self.observation_space = gym.spaces.Dict(
            {
                # node features: is node in tour, is previously selected node
                "node_features": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, 2),
                    dtype=np.float32,
                ),
                # edge features: distance between nodes, pointer to next node in tour
                "edge_features": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, self.max_nodes, 2),
                    dtype=np.float32,
                ),
                "adjacency_matrix": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, seed=None, options=None):
        self.graph = self.graphs[self.current_graph_index]
        self.current_graph_index += 1
        if self.current_graph_index >= len(self.graphs):
            self.current_graph_index = 0
            self.graph_rng.shuffle(self.graphs)

        self.in_tour = np.zeros(self.max_nodes, dtype=np.float32)
        self.prev_node = None
        self.next_nodes = np.arange(self.max_nodes)

        return self._get_observation(), {}

    def step(self, action):
        if self.in_tour[action] == 1:
            reward = -1.0  # penalty for revisiting a node
            return self._get_observation(), reward, False, False, {}

        self.in_tour[action] = 1
        reward = 0.0  # no immediate reward

        if self.prev_node is not None:
            self.next_nodes[self.prev_node] = action
        else:
            self.start_node = action

        self.prev_node = action
        done = self.in_tour.sum() == self.graph.num_nodes

        if done:
            # Complete the tour by returning to the start node
            self.next_nodes[self.prev_node] = self.start_node
            reward = -self._compute_tour_length()
        return self._get_observation(), reward, done, False, {}

    def action_masks(self):
        return (self.in_tour == 0) & (np.arange(self.max_nodes) < self.graph.num_nodes)

    def _sample_graph(self):
        size = self.graph_rng.choice(self.graph_sizes)
        points = self.graph_rng.random((size, 2))
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        edge_index = np.array(np.meshgrid(np.arange(size), np.arange(size))).reshape(
            2, -1
        )
        edge_attr = distances[edge_index[0], edge_index[1]]
        return Data(
            edge_index=th.tensor(edge_index),
            edge_attr=th.tensor(edge_attr, dtype=th.float32),
            num_nodes=size,
        )

    def _generate_graph_set(self):
        graphs = []
        for _ in range(self.num_graphs):
            graphs.append(self._sample_graph())
        return graphs

    def _get_observation(self):
        node_features = np.zeros((self.max_nodes, 2), dtype=np.float32)
        node_features[: self.graph.num_nodes, 0] = self.in_tour[: self.graph.num_nodes]
        if self.prev_node is not None:
            node_features[self.prev_node, 1] = 1.0

        edge_features = np.zeros((self.max_nodes, self.max_nodes, 2), dtype=np.float32)
        distances = (
            to_dense_adj(
                self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                max_num_nodes=self.max_nodes,
            )
            .squeeze(0)
            .numpy()
        )
        edge_features[:, :, 0] = distances
        for i in range(self.graph.num_nodes):
            next_node = self.next_nodes[i]
            edge_features[i, next_node, 1] = 1.0

        adjacency_matrix = (
            to_dense_adj(self.graph.edge_index, max_num_nodes=self.max_nodes)
            .squeeze(0)
            .numpy()
        )

        return {
            "node_features": node_features,
            "edge_features": edge_features,
            "adjacency_matrix": adjacency_matrix,
        }

    def _compute_tour_length(self):
        length = 0.0
        distance_matrix = (
            to_dense_adj(
                self.graph.edge_index,
                edge_attr=self.graph.edge_attr,
                max_num_nodes=self.max_nodes,
            )
            .squeeze(0)
            .numpy()
        )
        for i in range(self.graph.num_nodes):
            from_node = i
            to_node = self.next_nodes[i]
            length += distance_matrix[from_node, to_node]
        return length
