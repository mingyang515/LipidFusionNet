import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
import pandas as pd
import json

from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features

# ------------------------------
# 1. Fix all Random seeds
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# 2. EarlyStopping mechanism
# ------------------------------
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ------------------------------
# 3. Load data
# ------------------------------
def load_split(file_path):
    graph_data = preprocess_smiles_data(file_path)
    numerical_data = preprocess_numerical_features(file_path)
    return graph_data, torch.tensor(numerical_data, dtype=torch.float)

train_graph, train_numerical = load_split("../data/splits/train.csv")

# ------------------------------
# 4. Define the model parameters
# ------------------------------
batch_size = 32
epochs = 100   # From the original 10 to 100
learning_rate = 0.001
k_folds = 5
patience = 10  # early stopping: The number of rounds of waiting
mpnn_hidden_dim = 128
mpnn_layers = 3
mlp_input_dim = train_numerical.shape[1]
mlp_hidden_dim = 64
num_tasks = 2  # Predict Hela and Raw

# ------------------------------
# 5. Start the K-Fold training
# ------------------------------

# Record the result of the fold val loss
fold_val_losses = {}

# Start the K-Fold training
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold = 0

for train_idx, val_idx in kf.split(train_graph):
    fold += 1
    print(f"Starting Fold {fold}...")

    model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, num_tasks)

    # Load the pre-trained parameters (except for the fc layer)
    pretrained_dict = torch.load('lipid_fusion_net.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Reset the output layer
    model.fc = torch.nn.Linear(mpnn_hidden_dim + mlp_hidden_dim, num_tasks)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    # DataLoader
    train_graph_fold = [train_graph[i] for i in train_idx]
    train_numerical_fold = train_numerical[train_idx]
    val_graph_fold = [train_graph[i] for i in val_idx]
    val_numerical_fold = train_numerical[val_idx]

    train_loader = DataLoader(list(zip(train_graph_fold, train_numerical_fold)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_graph_fold, val_numerical_fold)), batch_size=batch_size)

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for graph_batch, numerical_batch in train_loader:
            optimizer.zero_grad()
            output = model(graph_batch, numerical_batch)
            target = numerical_batch[:, 1:3]
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Verification
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for graph_batch, numerical_batch in val_loader:
                output = model(graph_batch, numerical_batch)
                target = numerical_batch[:, 1:3]
                loss = F.mse_loss(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Fold {fold}, Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save the fold optimal model
    save_path = f"lipid_fusion_net_finetuned_fold{fold}.pth"
    torch.save(early_stopping.best_model_state, save_path)
    print(f"Saved best model for Fold {fold} to {save_path}")

    # Record the best validation loss for each fold
    fold_val_losses[f"fold{fold}"] = early_stopping.best_loss

# Save fold_val_loss.json
with open("fold_val_loss.json", "w") as f:
    json.dump(fold_val_losses, f, indent=4)

print("✅ All Fold training has been completed and fold_val_loss.json has been saved!")
