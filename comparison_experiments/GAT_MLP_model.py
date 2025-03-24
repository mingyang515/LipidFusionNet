import numpy as np
import torch
import torch.nn as nn
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
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.pool(x, batch)
        return x

# Defining the MLP Model
class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Defining the GAT + MLP fusion model
class GATMLPNet(nn.Module):
    def __init__(self, gat_hidden_dim, mlp_input_dim, mlp_hidden_dim, output_dim, gat_heads=2):
        super(GATMLPNet, self).__init__()
        self.gat = GATNet(input_dim=1, hidden_dim=gat_hidden_dim, output_dim=gat_hidden_dim, heads=gat_heads)
        self.mlp = MLPNet(input_dim=mlp_input_dim, hidden_dim=mlp_hidden_dim, output_dim=mlp_hidden_dim)
        self.fc = nn.Linear(gat_hidden_dim + mlp_hidden_dim, output_dim)

    def forward(self, graph_data, numerical_data):
        gat_output = self.gat(graph_data)
        mlp_output = self.mlp(numerical_data)
        combined = torch.cat([gat_output, mlp_output], dim=1)
        output = self.fc(combined)
        return output

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_file = "../data/splits/train.csv"
test_file = "../data/splits/test.csv"

train_graph = preprocess_smiles_data(train_file)
train_numerical = preprocess_numerical_features(train_file)
train_targets = train_numerical[:, 1:3]

test_graph = preprocess_smiles_data(test_file)
test_numerical = preprocess_numerical_features(test_file)
test_targets = test_numerical[:, 1:3]

# model parameter
gat_hidden_dim = 64
mlp_hidden_dim = 64
output_dim = 2
gat_heads = 2
gat_mlp_model = GATMLPNet(
    gat_hidden_dim=gat_hidden_dim,
    mlp_input_dim=train_numerical.shape[1],
    mlp_hidden_dim=mlp_hidden_dim,
    output_dim=output_dim,
    gat_heads=gat_heads
).to(device)

optimizer = torch.optim.Adam(gat_mlp_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# DataLoader
train_loader = DataLoader(list(zip(train_graph, train_numerical, train_targets)), batch_size=32, shuffle=True)
test_loader = DataLoader(list(zip(test_graph, test_numerical, test_targets)), batch_size=32)

# training model
gat_mlp_model.train()
for epoch in range(10):
    total_loss = 0
    for graph, numerical, target in train_loader:
        graph, numerical, target = graph.to(device), numerical.to(device), target.to(device)
        optimizer.zero_grad()
        predictions = gat_mlp_model(graph, numerical)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.6f}")

# test model
gat_mlp_model.eval()
total_preds = []
total_targets = []
with torch.no_grad():
    for graph, numerical, target in test_loader:
        graph, numerical, target = graph.to(device), numerical.to(device), target.to(device)
        preds = gat_mlp_model(graph, numerical)
        total_preds.append(preds.cpu().numpy())
        total_targets.append(target.cpu().numpy())

# Flatten predictions and targets for compatibility with sklearn metrics
total_preds = np.concatenate(total_preds, axis=0)
total_targets = np.concatenate(total_targets, axis=0)

# calculation of assessment indicators
mse = mean_squared_error(total_targets, total_preds)
mae = mean_absolute_error(total_targets, total_preds)
pcc = np.corrcoef(total_targets.flatten(), total_preds.flatten())[0, 1]
r2 = r2_score(total_targets, total_preds)

print(f"Test MSE: {mse:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test PCC: {pcc:.6f}")
print(f"Test R²: {r2:.6f}")
