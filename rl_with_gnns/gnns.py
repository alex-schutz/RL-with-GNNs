import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layers=2, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_dim, embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(embed_dim, embed_dim))

    def forward(self, node_fts, edge_index, **kwargs):
        x = self.conv1(node_fts, edge_index)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, embed_dim, edge_dim=None, num_layers=2, **kwargs):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, embed_dim, edge_dim=edge_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(GATv2Conv(embed_dim, embed_dim, edge_dim=edge_dim))

    def forward(self, node_fts, edge_index, edge_attr=None, **kwargs):
        x = self.conv1(node_fts, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, embed_dim, num_layers=2, **kwargs):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, embed_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(embed_dim, embed_dim))

    def forward(self, node_fts, edge_index, **kwargs):
        x = self.conv1(node_fts, edge_index)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


_NETWORKS = {"GAT": GAT, "GCN": GCN, "GraphSAGE": GraphSAGE}


def get_network_class(network_name: str):
    if network_name not in _NETWORKS:
        raise ValueError(
            f"Unknown network {network_name}, available networks: {_NETWORKS.keys()}"
        )
    return _NETWORKS[network_name]
