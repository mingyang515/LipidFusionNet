import torch
from torch_geometric.data import DataLoader
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_all
import torch.nn.functional as F

# data path
file_path = "./data/candidate_set_smiles_plus_features.csv"

# Load data
graph_data, numerical_data = preprocess_all(file_path)

# Convert numeric data to tensor
numerical_tensor = torch.tensor(numerical_data[:len(graph_data)], dtype=torch.float)

# Ensure that mlp_input_dim matches the dimension of the current numeric feature
mlp_input_dim = numerical_tensor.shape[1]

# Setting Validation Parameters
batch_size = 32

# Initialize the model and set the input dimensions
mpnn_hidden_dim = 128
mpnn_layers = 3
mlp_hidden_dim = 64
output_dim = 2

# Load the trained model
model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, output_dim)
model.load_state_dict(torch.load('./src/lipid_fusion_net.pth'))
model.eval()

# Creating Small Batches with DataLoader
test_loader = DataLoader(graph_data, batch_size=batch_size, shuffle=False)

# assessment model
total_loss = 0
correct = 0
for graph_batch in test_loader:
    # Get the numerical feature batch that matches the current graph data batch
    numerical_batch = numerical_tensor[:len(graph_batch)]

    with torch.no_grad():
        # forward propagation
        output = model(graph_batch, numerical_batch)
        target = numerical_batch[:, 1:3]

        # Calculation of losses
        loss = F.mse_loss(output, target)
        total_loss += loss.item()

    # Calculation of prediction accuracy (regression task)
    preds = output.cpu().numpy()
    target = target.cpu().numpy()
    correct += ((preds - target) ** 2).sum()

print(f"Validation Loss: {total_loss / len(test_loader)}")
print(f"Mean Squared Error: {correct / len(test_loader.dataset)}")
