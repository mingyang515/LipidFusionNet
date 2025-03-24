import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import GCNConv

class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)

class MPNNModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(MPNNModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(MPNNLayer(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(MPNNLayer(hidden_dim, hidden_dim))
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        x = self.pool(x, batch)
        return x

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class LipidFusionNet(torch.nn.Module):
    def __init__(self, mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, output_dim):
        super(LipidFusionNet, self).__init__()
        self.mpnn = MPNNModel(mpnn_hidden_dim, mpnn_layers)
        self.mlp = MLP(mlp_input_dim, mlp_hidden_dim, mlp_hidden_dim)
        self.fc = torch.nn.Linear(mpnn_hidden_dim + mlp_hidden_dim, output_dim)

    def forward(self, graph_data, numerical_data):
        # Get graph embeddings from MPNN
        graph_embeddings = self.mpnn(graph_data)
        # Get embeddings from MLP
        numerical_embeddings = self.mlp(numerical_data)
        # Concatenate both embeddings
        fused_embeddings = torch.cat([graph_embeddings, numerical_embeddings], dim=1)
        # Final output layer
        output = self.fc(fused_embeddings)
        return output



