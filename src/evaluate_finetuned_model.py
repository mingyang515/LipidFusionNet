import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Customizing the collate_fn function
def custom_collate(batch):
    """Packaging of graphical and numerical data into small batches."""
    graph_data, numerical_data = zip(*batch)
    graph_batch = Batch.from_data_list(graph_data)
    numerical_batch = torch.stack(numerical_data)
    return graph_batch, numerical_batch

# Loading the fine-tuned model
mpnn_hidden_dim = 128
mpnn_layers = 3
mlp_input_dim = 815
mlp_hidden_dim = 64
output_dim = 2

model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, output_dim)

# Explicitly set weights_only=True (if supported)
pretrained_dict = torch.load('lipid_fusion_net_finetuned.pth', weights_only=True)
model.load_state_dict(pretrained_dict)
model.eval()  # Setting up the model for evaluation mode

# Loading Test Sets
test_file = "../data/splits/test.csv"
test_data = pd.read_csv(test_file)

# preprocessing test set
graph_data = preprocess_smiles_data(test_file)
numerical_data = preprocess_numerical_features(test_file)

# Creating a Test Set DataLoader
test_loader = DataLoader(
    list(zip(graph_data, numerical_data)),
    batch_size=32,
    collate_fn=custom_collate
)

# Initialization of assessment indicators
total_mse, total_mae, total_pcc, total_r2, count = 0, 0, 0, 0, 0

# Iterate through the test set and calculate evaluation metrics
with torch.no_grad():
    for graph_batch, numerical_batch in test_loader:
        # Predicted output and true value
        output = model(graph_batch, numerical_batch)
        target = numerical_batch[:, 1:3]  # Assuming that the target values are columns 2 and 3 of the numerical feature

        # Calculation of MSE and MAE
        mse = F.mse_loss(output, target).item()
        mae = F.l1_loss(output, target).item()

        # Flatten outputs and target values
        output_flat = output.reshape(-1).cpu().numpy()
        target_flat = target.reshape(-1).cpu().numpy()

        # Calculate PCC and R²
        pcc = pearsonr(output_flat, target_flat)[0]
        r2 = r2_score(target_flat, output_flat)

        # Aggregate indicators
        batch_size = graph_batch.num_graphs
        total_mse += mse * batch_size
        total_mae += mae * batch_size
        total_pcc += pcc * batch_size
        total_r2 += r2 * batch_size
        count += batch_size

# Printing of final assessment indicators
print(f"Test MSE: {total_mse / count:.6f}")
print(f"Test MAE: {total_mae / count:.6f}")
print(f"Test PCC: {total_pcc / count:.6f}")
print(f"Test R²: {total_r2 / count:.6f}")