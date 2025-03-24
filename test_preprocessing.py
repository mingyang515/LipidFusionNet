from src.data_preprocessing import preprocess_all

file_path = "./data/candidate_set_smiles_plus_features.csv"
graph_data, numerical_data = preprocess_all(file_path)

# Checking pre-processed data
print(f"Graph Data Sample: {graph_data[:1]}")
print(f"Numerical Data Shape: {numerical_data.shape}")
