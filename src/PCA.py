import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# Ensure the figures directory exists, if not, create it
def create_figures_dir():
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    return figures_dir


# PCA processing of numerical features
def preprocess_numerical_features(file_path, apply_pca=False):
    df = pd.read_csv(file_path)
    numerical_features = df.drop(columns=['smiles', 'expt_Hela', 'expt_Raw'])

    # Standardize all numerical features
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)

    if apply_pca:
        # Perform PCA analysis, extracting the first 10 principal components
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(numerical_features)
        return pca_result, pca  # Return PCA results and PCA model for further analysis
    return numerical_features


# Visualization of PCA results
def plot_pca_results(pca_result, pca_model, df):
    figures_dir = create_figures_dir()
    font = {'weight': 'bold', 'size': 12}  # Set bold font for titles and labels

    # 1. Plotting the scatterplot of the first two principal components
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=np.linalg.norm(pca_result, axis=1), cmap='viridis',
                          edgecolors='k', alpha=0.7)
    plt.xlabel('Principal Component 1', fontdict=font)
    plt.ylabel('Principal Component 2', fontdict=font)
    plt.title('PCA of Numerical Features (PC1 vs PC2)', fontdict=font)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Norm of PCs', fontsize=12, fontweight='bold')

    # Bold the tick labels on both axes and colorbar
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=10, width=1, colors='black', labelcolor='black')  # Bold colorbar ticks

    plt.savefig(os.path.join(figures_dir, 'pca_2d.png'))
    plt.close()

    # 2. Plotting 3D scatterplots of the first three principal components
    if pca_result.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=np.linalg.norm(pca_result, axis=1),
                             cmap='viridis', edgecolors='k', alpha=0.7)
        ax.set_xlabel('Principal Component 1', fontdict=font)
        ax.set_ylabel('Principal Component 2', fontdict=font)
        ax.set_zlabel('Principal Component 3', fontdict=font)
        ax.set_title('PCA of Numerical Features (PC1, PC2, PC3)', fontdict=font)

        cbar = fig.colorbar(scatter)
        cbar.set_label('Norm of PCs', fontsize=12, fontweight='bold')

        # Correct way to bolden 3D plot tick labels
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.zaxis.set_tick_params(labelsize=10)

        plt.setp(ax.get_xticklabels(), fontweight="bold")
        plt.setp(ax.get_yticklabels(), fontweight="bold")
        plt.setp(ax.get_zticklabels(), fontweight="bold")

        cbar.ax.tick_params(labelsize=10, width=1, colors='black', labelcolor='black')  # Bold colorbar ticks

        plt.savefig(os.path.join(figures_dir, 'pca_3d.png'))
        plt.close()

    # 3. Bar chart of the proportion of variance explained
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[f'PC{i + 1}' for i in range(len(pca_model.explained_variance_ratio_))],
                y=pca_model.explained_variance_ratio_)
    plt.xlabel('Principal Components', fontdict=font)
    plt.ylabel('Explained Variance Ratio', fontdict=font)
    plt.title('Explained Variance Ratio by Principal Components', fontdict=font)

    # Bold the tick labels on both axes
    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=10, fontweight='bold')

    plt.savefig(os.path.join(figures_dir, 'explained_variance.png'))
    plt.close()

    # 4. Heatmap: show the feature contributions to the first two principal components
    feature_contrib = pd.DataFrame(pca_model.components_,
                                   columns=df.drop(columns=['smiles', 'expt_Hela', 'expt_Raw']).columns)

    plt.figure(figsize=(8, 6))
    sns.heatmap(feature_contrib.iloc[:2, :20], annot=True, cmap='coolwarm', cbar=True,
                annot_kws={"size": 8, "rotation": 90},
                cbar_kws={"shrink": 0.6, "aspect": 12})

    plt.xticks(rotation=45, ha="right", fontsize=10, fontweight='bold')
    plt.yticks(rotation=0, ha="right", fontsize=10, fontweight='bold')

    # Adjust title font size
    plt.title('Feature Contribution to First Two Principal Components', fontsize=12, fontweight='bold')

    # Bold the tick labels on both axes and colorbar ticks
    plt.tick_params(axis='both', labelsize=10)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, width=1, colors='black', labelcolor='black')  # Bold colorbar ticks

    plt.savefig(os.path.join(figures_dir, 'feature_contribution.png'), bbox_inches='tight', dpi=300)
    plt.close()


# Perform PCA analysis and visualization after training
train_numerical_pca, pca_model = preprocess_numerical_features("../data/splits/train.csv", apply_pca=True)
df_train = pd.read_csv("../data/splits/train.csv")  # Re-read data for use in heatmap
plot_pca_results(train_numerical_pca, pca_model, df_train)

# Print the explained variance ratio by PCA components
print(f"Explained Variance Ratio by each principal component: {pca_model.explained_variance_ratio_}")
