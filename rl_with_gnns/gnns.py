import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(out_channels, out_channels))

    def forward(self, node_fts, edge_index, edge_attr=None, batch=None):
        x = self.conv1(node_fts, edge_index)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


class GAT(nn.Module):
    def __init__(
        self, in_channels, out_channels, edge_dim=None, heads=2, num_layers=2, **kwargs
    ):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels, out_channels, heads=heads, concat=True, edge_dim=edge_dim
        )
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(
                GATv2Conv(
                    out_channels * heads,
                    out_channels,
                    heads=1,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )

    def forward(self, node_fts, edge_index, edge_attr=None, batch=None):
        x = self.conv1(node_fts, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, **kwargs):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(out_channels, out_channels))

    def forward(self, node_fts, edge_index, edge_attr=None, batch=None):
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
