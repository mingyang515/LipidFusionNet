import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features
from scipy.stats import shapiro, spearmanr
import numpy as np
import pandas as pd
import json

def custom_collate(batch):
    graph_data, numerical_data = zip(*batch)
    graph_batch = Batch.from_data_list(graph_data)
    numerical_batch = torch.stack(numerical_data)
    return graph_batch, numerical_batch

def load_best_model():
    with open("fold_val_loss.json", "r") as f:
        fold_losses = json.load(f)
    best_fold = min(fold_losses, key=fold_losses.get)
    model_path = f"lipid_fusion_net_finetuned_fold{int(best_fold[-1])}.pth"

    model = LipidFusionNet(128, 3, 815, 64, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def main():
    model = load_best_model()

    test_file = "../data/splits/test.csv"
    df = pd.read_csv(test_file)
    graph_data = preprocess_smiles_data(test_file)
    numerical_data = preprocess_numerical_features(test_file)

    test_loader = DataLoader(list(zip(graph_data, numerical_data)), batch_size=32, collate_fn=custom_collate)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for graph_batch, numerical_batch in test_loader:
            output = model(graph_batch, numerical_batch)
            target = numerical_batch[:, 1:3]
            y_true.append(target.cpu().numpy())
            y_pred.append(output.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    tasks = ['HELA', 'RAW']
    for i in range(2):
        print(f"\n=== {tasks[i]} ===")
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        sw_true_p = shapiro(true_vals).pvalue
        sw_pred_p = shapiro(pred_vals).pvalue

        print(f"Normality of true values p = {sw_true_p:.4f} ({'Normal' if sw_true_p > 0.05 else 'Non-normal'})")
        print(f"Normality of predicted values p = {sw_pred_p:.4f} ({'Normal' if sw_pred_p > 0.05 else 'Non-normal'})")

        spearman_corr, spearman_p = spearmanr(true_vals, pred_vals)
        print(f"Spearman correlation coefficient: {spearman_corr:.4f} (p = {spearman_p:.4e})")

if __name__ == "__main__":
    main()
