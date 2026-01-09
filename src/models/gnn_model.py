import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class StabilityGNN(torch.nn.Module):
    """
    Graph Neural Network for power system stability prediction.
    """

    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.classifier = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        return self.classifier(x)
