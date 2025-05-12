import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd
import os
import json

# Custom collate function
def custom_collate(batch):
    graph_data, numerical_data = zip(*batch)
    graph_batch = Batch.from_data_list(graph_data)
    numerical_batch = torch.stack(numerical_data)
    return graph_batch, numerical_batch

# Load the finetuned model with the minimum loss of the validation set
def load_best_model():
    with open("fold_val_loss.json", "r") as f:
        fold_losses = json.load(f)

    best_fold = min(fold_losses, key=fold_losses.get)
    print(f"✅ Select the fold with the smallest validation set loss：{best_fold}")

    fold_number = int(best_fold.replace("fold", ""))
    model_path = f"lipid_fusion_net_finetuned_fold{fold_number}.pth"

    mpnn_hidden_dim = 128
    mpnn_layers = 3
    mlp_input_dim = 815
    mlp_hidden_dim = 64
    output_dim = 2
    model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    model = load_best_model()

    # Load the test set
    test_file = "../data/splits/test.csv"
    graph_data = preprocess_smiles_data(test_file)
    numerical_data = preprocess_numerical_features(test_file)

    test_loader = DataLoader(
        list(zip(graph_data, numerical_data)),
        batch_size=32,
        collate_fn=custom_collate
    )

    total_mse, total_mae, total_pcc, total_r2, count = 0, 0, 0, 0, 0

    # evaluate
    with torch.no_grad():
        for graph_batch, numerical_batch in test_loader:
            output = model(graph_batch, numerical_batch)
            target = numerical_batch[:, 1:3]

            mse = F.mse_loss(output, target).item()
            mae = F.l1_loss(output, target).item()

            output_flat = output.reshape(-1).cpu().numpy()
            target_flat = target.reshape(-1).cpu().numpy()

            pcc = pearsonr(output_flat, target_flat)[0]
            r2 = r2_score(target_flat, output_flat)

            batch_size = graph_batch.num_graphs
            total_mse += mse * batch_size
            total_mae += mae * batch_size
            total_pcc += pcc * batch_size
            total_r2 += r2 * batch_size
            count += batch_size

    print(f"Test MSE: {total_mse / count:.6f}")
    print(f"Test MAE: {total_mae / count:.6f}")
    print(f"Test PCC: {total_pcc / count:.6f}")
    print(f"Test R²: {total_r2 / count:.6f}")

if __name__ == "__main__":
    main()
