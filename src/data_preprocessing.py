import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)


def smiles_to_graph(smiles):
    """Convert SMILES string to a graph Data object for use in MPNN."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edges = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(atoms, dtype=torch.float).view(-1, 1)
    return Data(x=x, edge_index=edge_index)


def preprocess_smiles_data(file_path):
    """Convert SMILES data to graph format for MPNN."""
    df = load_data(file_path)
    graph_data = []
    for smiles in df['smiles']:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graph_data.append(graph)
    return graph_data


def preprocess_numerical_features(file_path):
    """Standardize numerical features for MTL model."""
    df = load_data(file_path)
    numerical_features = df.drop(columns=['smiles'])
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)
    return torch.tensor(numerical_features, dtype=torch.float)


def preprocess_all(file_path):
    """
    Process both SMILES and numerical data.

    Returns:
        graph_data: List of graph objects.
        numerical_data: Standardized numerical features.
    """
    graph_data = preprocess_smiles_data(file_path)
    numerical_data = preprocess_numerical_features(file_path)
    return graph_data, numerical_data


def split_and_save_data(file_path, output_dir):
    """Split the dataset into training and testing sets, and save them as CSV."""
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = load_data(file_path)
    total_size = len(df)
    test_size = 200

    # Randomly split data
    test_indices = np.random.choice(total_size, size=test_size, replace=False)
    train_indices = list(set(range(total_size)) - set(test_indices))

    train_data = df.iloc[train_indices]
    test_data = df.iloc[test_indices]

    # Save split data to CSV files
    train_data.to_csv(f"{output_dir}/train.csv", index=False)
    test_data.to_csv(f"{output_dir}/test.csv", index=False)

# usage
split_and_save_data("../data/finetuning_set_smiles_plus_features.csv", "../data/splits")
