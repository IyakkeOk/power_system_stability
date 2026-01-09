import torch
from torch_geometric.data import Data
import numpy as np


def build_graph_from_row(row):
    """
    Build a graph for a single power system snapshot.
    """

    # --- Node features (14 buses, replicated for simplicity) ---
    node_features = torch.tensor([
        [
            row["VoltageMag"],
            row["VoltageAngle"],
            row["P_inj"],
            row["Q_inj"],
            row["Q_margin"],
            row["VSI"],
            row["FreqDev_Node"]
        ]
    ] * 14, dtype=torch.float)

    # --- Edge index (IEEE 14-bus simplified topology) ---
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
         6, 7, 7, 8, 8, 9, 9, 10, 10, 11,
         11, 12, 12, 13],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5,
         7, 6, 8, 7, 9, 8, 10, 9, 11, 10,
         12, 11, 13, 12]
    ], dtype=torch.long)

    # --- Edge features ---
    edge_attr = torch.tensor([
        [row["LineR"], row["LineX"], row["LineB"],
         row["LineS_flow"], row["LOF"]]
    ] * edge_index.shape[1], dtype=torch.float)

    # Label
    y = torch.tensor([row["StabilityLabel"]], dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
