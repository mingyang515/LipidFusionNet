import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Defining the GAT model
class GATNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_file = "../data/splits/train.csv"
test_file = "../data/splits/test.csv"

train_graph = preprocess_smiles_data(train_file)
train_targets = preprocess_numerical_features(train_file)[:, 1:3]
test_graph = preprocess_smiles_data(test_file)
test_targets = preprocess_numerical_features(test_file)[:, 1:3]

# model parameter
hidden_dim = 64
output_dim = 2
gat_model = GATNet(input_dim=1, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# DataLoader
train_loader = DataLoader(list(zip(train_graph, train_targets)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(test_graph, test_targets)), batch_size=32)

# Ensure predictions and targets are collected correctly
total_preds = []
total_targets = []

# Evaluate model
gat_model.eval()
with torch.no_grad():
    for graph, target in test_loader:
        graph = graph.to(device)
        target = target.to(device)
        preds = gat_model(graph)

        total_preds.append(preds.cpu().numpy())
        total_targets.append(target.cpu().numpy())

# Flatten predictions and targets for compatibility with sklearn metrics
total_preds = np.concatenate(total_preds, axis=0)
total_targets = np.concatenate(total_targets, axis=0)

# Ensure predictions and targets have the same shape
assert total_preds.shape == total_targets.shape, f"Shape mismatch: {total_preds.shape} vs {total_targets.shape}"

# Calculate evaluation metrics
mse = mean_squared_error(total_targets, total_preds)
mae = mean_absolute_error(total_targets, total_preds)
pcc = np.corrcoef(total_targets.flatten(), total_preds.flatten())[0, 1]
r2 = r2_score(total_targets, total_preds)

print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test PCC: {pcc:.6f}")
print(f"Test R²: {r2:.6f}")
