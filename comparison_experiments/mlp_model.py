import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd

# Load data
df = pd.read_csv("../data/finetuning_set_smiles_plus_features.csv")
X = df.drop(columns=["expt_Hela", "expt_Raw", "smiles"])
y = df[["expt_Hela", "expt_Raw"]]

# Data loading
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to Tensor
X_train, X_test = torch.tensor(X_train.values, dtype=torch.float32), torch.tensor(X_test.values, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train.values, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32)

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

model = MLPNet(input_dim=X_train.shape[1], hidden_dim=64, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# data loader
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# training model
for epoch in range(50):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()

# test
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test).item()
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    pcc = pearsonr(y_test.numpy().flatten(), y_pred.numpy().flatten())[0]
    r2 = r2_score(y_test.numpy(), y_pred.numpy())

print(f"MLP\nMSE: {mse:.3f}, \nMAE: {mae:.3f}, \nPCC: {pcc:.3f}, \nR²: {r2:.3f}")
