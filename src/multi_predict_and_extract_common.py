import torch
import pandas as pd
import os
import random
import numpy as np
import json
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from src.model import LipidFusionNet
from src.data_preprocessing import preprocess_smiles_data, preprocess_numerical_features
from torch_geometric.data import Batch

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def custom_collate(batch):
    graph_data, numerical_data = zip(*batch)
    graph_batch = Batch.from_data_list(graph_data)
    numerical_batch = torch.stack(numerical_data)
    return graph_batch, numerical_batch

def load_best_model():
    """根据 fold_val_loss.json 自动选择验证集loss最小的fold模型"""
    with open("fold_val_loss.json", "r") as f:
        fold_losses = json.load(f)

    best_fold = min(fold_losses, key=fold_losses.get)
    print(f"✅ 多次推理中，自动使用验证集最优fold：{best_fold}")

    fold_number = int(best_fold.replace("fold", ""))
    model_path = f"lipid_fusion_net_finetuned_fold{fold_number}.pth"

    mpnn_hidden_dim = 128
    mpnn_layers = 3
    mlp_input_dim = 813  # 注意：推理数据特征是813列
    mlp_hidden_dim = 64
    output_dim = 2
    model = LipidFusionNet(mpnn_hidden_dim, mpnn_layers, mlp_input_dim, mlp_hidden_dim, output_dim)

    state_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.eval()
    return model

def predict_top_bottom_once(model):
    """每次推理复用同一个model"""
    file_path = "../data/alternative_data_set.csv"
    graph_data = preprocess_smiles_data(file_path)
    numerical_data = preprocess_numerical_features(file_path)

    data_loader = DataLoader(
        list(zip(graph_data, numerical_data)),
        batch_size=32,
        collate_fn=custom_collate,
        shuffle=False  # 推理绝对不shuffle
    )

    predictions = []
    with torch.no_grad():
        for graph_batch, numerical_batch in data_loader:
            output = model(graph_batch, numerical_batch)
            predictions.extend(output.cpu().numpy())

    df = pd.read_csv(file_path)
    df['Predicted_Hela'] = [pred[0] for pred in predictions]

    df_sorted = df.sort_values('Predicted_Hela')
    top_10 = df_sorted.tail(10)
    bottom_10 = df_sorted.head(10)
    middle_idx = len(df_sorted) // 2
    middle_10 = df_sorted.iloc[middle_idx-5:middle_idx+5]

    return top_10, middle_10, bottom_10

def save_csv(df, path):
    df.to_csv(path, index=False)

def save_frequency_table(all_samples, save_path):
    """统计smiles出现频率并保存"""
    all_smiles = []
    for df in all_samples:
        all_smiles.extend(df['smiles'].tolist())
    freq_series = pd.Series(all_smiles).value_counts()
    freq_df = freq_series.reset_index()
    freq_df.columns = ['smiles', 'count']
    freq_df.to_csv(save_path, index=False)

def main():
    all_top, all_middle, all_bottom = [], [], []

    model = load_best_model()  # ⭐ 只加载一次模型！

    print("开始进行10次复用同一个模型的预测...")
    for i in tqdm(range(10)):
        top_10, middle_10, bottom_10 = predict_top_bottom_once(model)
        all_top.append(top_10)
        all_middle.append(middle_10)
        all_bottom.append(bottom_10)

    def find_common(dfs):
        common = set(dfs[0]['smiles'])
        for df in dfs[1:]:
            common = common.intersection(set(df['smiles']))
        return common

    common_top_smiles = find_common(all_top)
    common_middle_smiles = find_common(all_middle)
    common_bottom_smiles = find_common(all_bottom)

    def extract_common_df(dfs, common_smiles):
        frames = []
        for df in dfs:
            frames.append(df[df['smiles'].isin(common_smiles)])
        result = pd.concat(frames).drop_duplicates(subset=['smiles'])
        return result

    common_top_df = extract_common_df(all_top, common_top_smiles)
    common_middle_df = extract_common_df(all_middle, common_middle_smiles)
    common_bottom_df = extract_common_df(all_bottom, common_bottom_smiles)

    save_dir = "../data/predict"
    os.makedirs(save_dir, exist_ok=True)

    # 保存共有配方
    save_csv(common_top_df, os.path.join(save_dir, "common_top_10_hela_efficiency.csv"))
    save_csv(common_middle_df, os.path.join(save_dir, "common_middle_10_hela_efficiency.csv"))
    save_csv(common_bottom_df, os.path.join(save_dir, "common_bottom_10_hela_efficiency.csv"))

    # 保存所有配方总表
    total_top = pd.concat(all_top).drop_duplicates(subset=['smiles'])
    total_middle = pd.concat(all_middle).drop_duplicates(subset=['smiles'])
    total_bottom = pd.concat(all_bottom).drop_duplicates(subset=['smiles'])

    save_csv(total_top, os.path.join(save_dir, "total_top_10_hela_efficiency.csv"))
    save_csv(total_middle, os.path.join(save_dir, "total_middle_10_hela_efficiency.csv"))
    save_csv(total_bottom, os.path.join(save_dir, "total_bottom_10_hela_efficiency.csv"))

    # 统计频率并保存
    save_frequency_table(all_top, os.path.join(save_dir, "freq_top_10_hela_efficiency.csv"))
    save_frequency_table(all_middle, os.path.join(save_dir, "freq_middle_10_hela_efficiency.csv"))
    save_frequency_table(all_bottom, os.path.join(save_dir, "freq_bottom_10_hela_efficiency.csv"))

    print("✅ 复用同一个模型的多次推理全部完成！保存所有表格！")

if __name__ == "__main__":
    main()
