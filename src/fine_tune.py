import torch
from torch_geometric.loader import DataLoader
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torch.optim import Adam
import pandas as pd

# Load training data
def load_split(file_path):
    df = pd.read_csv(file_path)
    graph_data = preprocess_smiles_data(file_path)
    numerical_data = preprocess_numerical_features(file_path)
    return graph_data, torch.tensor(numerical_data, dtype=torch.float)

train_graph, train_numerical = load_split("../data/splits/train.csv")

# model parameter
batch_size = 32
epochs = 10
learning_rate = 0.001
k_folds = 5

mpnn_hidden_dim = 128
mpnn_layers = 3
mlp_input_dim = train_numerical.shape[1]
mlp_hidden_dim = 64
num_tasks = 2

model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, num_tasks)

# Load and adjust pre-training weights
pretrained_dict = torch.load("lipid_fusion_net.pth")
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if "fc" not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.fc = torch.nn.Linear(mpnn_hidden_dim + mlp_hidden_dim, 2)  # Reset Output Layer

optimizer = Adam(model.parameters(), lr=learning_rate)

# K-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_graph)):
    train_graph_fold = [train_graph[i] for i in train_idx]
    train_numerical_fold = train_numerical[train_idx]
    train_loader = DataLoader(list(zip(train_graph_fold, train_numerical_fold)), batch_size=batch_size, shuffle=True)

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
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), "lipid_fusion_net_finetuned.pth")