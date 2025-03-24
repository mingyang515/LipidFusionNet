import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import os
import matplotlib.cm as cm


def load_data(file_path):
    """Load dataset from a CSV file."""
    df = pd.read_csv(file_path)
    # Drop the first column (SMILES) and the target columns
    df = df.iloc[:, 1:]  # Remove the first column (SMILES)
    return df


def save_figure(fig, filename):
    """Save the figure to the 'figures' directory."""
    # Save the figure
    fig.savefig(f'../figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_feature_importance(importances, feature_names, top_k=10):
    """Plot the feature importances from a model like Random Forest with horizontal bars and color gradient."""
    indices = np.argsort(importances)[::-1][:top_k]  # Get the top K features
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted figure size for better readability

    # Normalize the importance values for the color map
    norm = plt.Normalize(vmin=np.min(importances[indices]), vmax=np.max(importances[indices]))
    cmap = cm.viridis  # Choose a colormap (viridis in this case)

    # Create a horizontal bar plot with color gradient based on importance
    bar = ax.barh(range(top_k), importances[indices], align="center", color=cmap(norm(importances[indices])))

    # Adjust the spacing of the ticks and labels
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(np.array(feature_names)[indices], fontweight='bold')
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_ylabel('Features', fontweight='bold')
    ax.set_title(f"Top {top_k} Feature Importance (Random Forest)", fontweight='bold')

    # Add colorbar with a smaller size and place it within the figure range
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal')
    cbar.set_label('Importance Gradient', fontweight='bold')

    # Adjust the colorbar size and position (move it to the top-right corner)
    cbar.ax.set_position([0.7, 0.95, 0.25, 0.02])  # Adjust this to place it in the right place
    cbar.ax.tick_params(labelsize=8)  # Smaller ticks for the colorbar

    # Set tick labels font weight for x and y axis
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

    # Save the figure
    save_figure(fig, 'feature_importance.png')


def plot_correlation_heatmap(df, target_column, top_k=20):
    """Plot correlation heatmap of features with the target column."""
    # Drop the other target column before calculating correlations
    target_columns = ['expt_Hela', 'expt_Raw']
    other_target_column = [col for col in target_columns if col != target_column][0]

    df_filtered = df.drop(columns=[other_target_column])  # Drop the other target column

    # Calculate correlation matrix
    correlation_matrix = df_filtered.corr()

    # Get correlations with the target column and sort them
    target_corr = correlation_matrix[target_column].drop(target_column).sort_values(ascending=False)

    # Select top K correlated features
    top_corr_features = target_corr.head(top_k)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(top_corr_features.to_frame(), annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Correlation'})
    ax.set_title(f'Top {top_k} Feature Correlation with {target_column}', fontweight='bold')
    ax.set_xlabel('Correlation', fontweight='bold')
    ax.set_ylabel('Features', fontweight='bold')

    # Set tick labels font weight for x and y axis
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')

    # Set colorbar label font weight and bold the ticks
    cbar = ax.collections[0].colorbar
    cbar.set_label('Correlation', fontweight='bold')

    # Bold the colorbar ticks
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    save_figure(fig, f'correlation_heatmap_{target_column}.png')


def main(file_path):
    # Load and preprocess data
    df = load_data(file_path)

    # Separate features and target variables
    target_columns = ['expt_Hela', 'expt_Raw']
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # 1. Random Forest Feature Importance (Top 10 features)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y['expt_Hela'])  # You can choose either 'expt_Hela' or 'expt_Raw'
    plot_feature_importance(rf_model.feature_importances_, X.columns, top_k=10)

    # 2. Correlation Heatmap for 'expt_Hela' (Top 20 features)
    plot_correlation_heatmap(df, 'expt_Hela', top_k=20)


if __name__ == "__main__":
    # Set the correct path to your dataset file
    file_path = "../data/splits/train.csv"
    main(file_path)
