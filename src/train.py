import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features,preprocess_all
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import pandas as pd

def load_split(file_path):
    df = pd.read_csv(file_path)
    graph_data = preprocess_smiles_data(file_path)
    numerical_data = preprocess_numerical_features(file_path)
    return graph_data, torch.tensor(numerical_data, dtype=torch.float)

train_graph, train_numerical = load_split("../data/candidate_set_smiles_plus_features.csv")

batch_size = 32
epochs = 50
learning_rate = 0.001

mpnn_hidden_dim = 128
mpnn_layers = 3
mlp_input_dim = train_numerical.shape[1]
mlp_hidden_dim = 64
num_tasks = train_numerical.shape[1]

model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, num_tasks)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

# DataLoader
train_loader = DataLoader(list(zip(train_graph, train_numerical)), batch_size=batch_size, shuffle=True)

# training cycle
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for graph_batch, numerical_batch in train_loader:
        optimizer.zero_grad()
        output = model(graph_batch, numerical_batch)
        loss = F.mse_loss(output, numerical_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# save model
torch.save(model.state_dict(), "lipid_fusion_net.pth")
