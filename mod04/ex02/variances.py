from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filepath) -> pd.DataFrame:
    # Load data
    return pd.read_csv(filepath)

def standardize_data(data: pd.DataFrame) -> np.ndarray:
    # Standardize the features: each data value - its mean
    scaler = StandardScaler()
    return scaler.fit_transform(data)

def perform_pca(scaled_features: np.ndarray) -> PCA:
    # Apply PCA
    pca = PCA()
    pca.fit(scaled_features)
    return pca

def print_variances(pca: PCA):
    # Individual explained variances
    print("Individual explained variances:")
    print(pca.explained_variance_ratio_)
    # Cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    print("\nCumulative explained variances:")
    print(pca.explained_variance_ratio_.cumsum() * 100)
    return cumulative_variance

def plot_explained_variance(cumulative_variance):
    # Plotting the cumulative explained variance
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance * 100, marker='o', linestyle='-', color='b')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    # Add a line for 90% threshold
    plt.axhline(y=0.9 * 100, color='r', linestyle='--', label='90% Explained Variance')
    plt.legend()
    plt.savefig('variances.png', format='png')
    plt.close()

# Main execution
try:
    df_train = load_data('../data/Train_knight.csv')
    features = df_train.drop(columns=['knight'])  # Knight' is categorical, so we drop it
    scaled_features = standardize_data(features) # To perform PCA, first we standardize.
    pca = perform_pca(scaled_features)
    cumulative_variance = print_variances(pca)
    plot_explained_variance(cumulative_variance)

except Exception as e:
    print(f"Error: {e}")
