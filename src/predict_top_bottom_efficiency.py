import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features
from torch_geometric.data import Batch

# Customizing the collate_fn function
def custom_collate(batch):
    """Packaging of graphical and numerical data into small batches."""
    graph_data, numerical_data = zip(*batch)
    graph_batch = Batch.from_data_list(graph_data)
    numerical_batch = torch.stack(numerical_data)
    return graph_batch, numerical_batch

# Loading Models
mpnn_hidden_dim = 128
mpnn_layers = 3
mlp_input_dim = 813
mlp_hidden_dim = 64
output_dim = 2

model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, output_dim)

# Loading the fine-tuned model weights
pretrained_dict = torch.load('lipid_fusion_net_finetuned.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()  # Setting up the model for evaluation mode

# Load data
file_path = "../data/alternative_data_set.csv"
graph_data = preprocess_smiles_data(file_path)
numerical_data = preprocess_numerical_features(file_path)

# Creating a DataLoader
data_loader = DataLoader(
    list(zip(graph_data, numerical_data)),
    batch_size=32,
    collate_fn=custom_collate
)

# Making predictions about data
predictions = []
with torch.no_grad():
    for graph_batch, numerical_batch in data_loader:
        output = model(graph_batch, numerical_batch)
        predictions.extend(output.cpu().numpy())

# Load original dataset, match predictions
df = pd.read_csv(file_path)
df['Predicted_Hela'] = [pred[0] for pred in predictions]
df['Predicted_Raw'] = [pred[1] for pred in predictions]

# Screening of 10 data with the highest, lowest and intermediate transfection efficiency in Hela cells
top_10_hela = df.nlargest(10, 'Predicted_Hela')
bottom_10_hela = df.nsmallest(10, 'Predicted_Hela')

# Obtain median interval of transfection efficiency in Hela cells (based on median position)
hela_sorted = df.sort_values('Predicted_Hela')
middle_10_hela = hela_sorted.iloc[len(hela_sorted)//2 - 5 : len(hela_sorted)//2 + 5]

# Screening the 10 data with the highest, lowest and intermediate transfection efficiency in Raw cells
top_10_raw = df.nlargest(10, 'Predicted_Raw')
bottom_10_raw = df.nsmallest(10, 'Predicted_Raw')

# Obtain median interval of transfection efficiency in Raw cells (based on median position)
raw_sorted = df.sort_values('Predicted_Raw')
middle_10_raw = raw_sorted.iloc[len(raw_sorted)//2 - 5 : len(raw_sorted)//2 + 5]

# Save results to CSV file
top_10_hela.to_csv("../data/predict/top_10_hela_efficiency.csv", index=False)
bottom_10_hela.to_csv("../data/predict/bottom_10_hela_efficiency.csv", index=False)
middle_10_hela.to_csv("../data/predict/middle_10_hela_efficiency.csv", index=False)

top_10_raw.to_csv("../data/predict/top_10_raw_efficiency.csv", index=False)
bottom_10_raw.to_csv("../data/predict/bottom_10_raw_efficiency.csv", index=False)
middle_10_raw.to_csv("../data/predict/middle_10_raw_efficiency.csv", index=False)

print("Filtering is complete and the results have been saved to . /data/predict/:")
print("top_10_hela_efficiency.csv, bottom_10_hela_efficiency.csv, middle_10_hela_efficiency.csv")
print("top_10_raw_efficiency.csv, bottom_10_raw_efficiency.csv, middle_10_raw_efficiency.csv")
