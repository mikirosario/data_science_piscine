from sqlalchemy import create_engine, Engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'

def get_engine() -> Engine:
    """
    Creates and returns a SQLAlchemy engine instance configured for a PostgreSQL database.
    """
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(connection_string)
    return engine

def get_data() -> pd.DataFrame:
    """
    Retrieves customer data from the database, aggregating purchase frequency and total spend per user.
    Returns a pd.DataFrame with these values.
    """
    query = """
    SELECT user_id, COUNT(*) AS purchase_frequency, SUM(price::NUMERIC) AS total_spend
    FROM customers
    WHERE event_type = 'purchase'
    GROUP BY user_id;
    """
    engine = get_engine()
    features_df = pd.read_sql(query, engine)
    return features_df[['purchase_frequency', 'total_spend']]

def get_custom_colors() -> dict[str, str]:
    """
    Defines and returns a dictionary of custom colors for visualizing customer clusters.
    """
    colors = {
        'Inactive Customers': 'red',
        'Loyalty Status: Silver': 'silver',
        'Loyalty Status: Gold': 'gold',
        'Loyalty Status: Platinum': '#696969',  # Dark grey suggestive of platinum
        'Active Customers': 'green'
    }
    return colors

def scale_and_cluster_features(data) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales the given data using StandardScaler to have zero mean and unit variance, 
    and performs K-means clustering on the scaled data. Returns the centroids in 
    the original feature space and the cluster labels for each data point.
    
    Args:
    data (pd.DataFrame): The input data with features to scale and cluster.

    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing:
        - centroids (np.ndarray): An array of the cluster centers in the original 
          feature space.
        - clusters (np.ndarray): An array of cluster labels for each data point.
    clusters[0...100...LastDataPoint] = ClusterIndex
    """
    scaler = StandardScaler()
    scaled_data: np.ndarray = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    return centroids, clusters


def sort_and_label_clusters(centroids: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """
    Sorts clusters based on their centroids and assigns custom labels. Returns sorted indices
    and a mapping of original to sorted labels, helping in further visualization.

    Args:
    centroids (np.ndarray): The cluster centers in the original feature space.

    Returns:
    tuple[np.ndarray, dict]: A tuple containing:
        - sorted_indices (np.ndarray): An array of indices that sort the clusters.
        - label_map (dict): A dictionary mapping original cluster indices to sorted labels.
    """
    cluster_labels = [
        'Inactive Customers',
        'Active Customers',
        'Loyalty Status: Silver',
        'Loyalty Status: Gold',
        'Loyalty Status: Platinum'
    ]
    sorted_indices: np.ndarray = np.lexsort((centroids[:, 1], centroids[:, 0])) # Correctly indicate type
    label_map: dict[int, str] = {original_index: cluster_labels[i] for i, original_index in enumerate(sorted_indices)}
    return sorted_indices, label_map


def plot_clusters(data, clusters, centroids, sorted_indices, label_map):
    """
    Plots the clusters of customers based on purchase frequency and total spend,
    highlighting the centroids and annotating them with labels.
    
    Args:
    data (DataFrame): Data containing purchase frequency and total spend.
    clusters (np.ndarray): Array of cluster labels for each data point.
    centroids (np.ndarray): Array of centroids for the clusters.
    sorted_indices (np.ndarray): Array of indices that sort the clusters.
    label_map (dict): A dictionary mapping original cluster indices to sorted labels.
    """
    colors = get_custom_colors()
    plt.figure(figsize=(10, 8))
    for cluster_index in range(len(centroids)):
        cluster_data = data[clusters == cluster_index]
        color = colors[label_map[cluster_index]]
        plt.scatter(cluster_data['purchase_frequency'], cluster_data['total_spend'], label=f'Cluster {label_map[cluster_index]}', color=color)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='blue', marker='x', s=100, label='Centroids')
    # Annotate using label_map directly sorted by sorted_indices for consistency
    for i in sorted_indices:
        plt.annotate(label_map[i], (centroids[i, 0], centroids[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Cluster Visualization')
    plt.xlabel('Frequency')
    plt.ylabel('Total Spend')
    plt.legend()

def plot_customer_counts_bar_chart(clusters, label_map):
    """
    Creates a horizontal bar chart representing the number of customers in each cluster,
    using custom colors for each cluster type. The chart displays clusters sorted by the
    number of customers in descending order.

    Args:
    clusters (np.ndarray): An array of cluster labels for each data point, where each label
        corresponds to the cluster index to which the data point has been assigned.
    label_map (dict): A dictionary mapping cluster indices to descriptive string labels.
        These labels are used in the bar chart to represent each cluster meaningfully.

    The function computes the frequency of each cluster label in 'clusters', uses 'label_map'
    to assign descriptive labels, and sorts these labels by the number of customers before
    plotting. Colors for each bar are determined by a custom color scheme.

    The resulting plot is displayed with the y-axis inverted, so that the cluster with the
    highest count is at the top of the chart.
    """
    colors = get_custom_colors()
    unique, counts = np.unique(clusters, return_counts=True)
    customer_counts = {label_map[k]: v for k, v in zip(unique, counts)}
    sorted_customer_counts = dict(sorted(customer_counts.items(), key=lambda item: item[1], reverse=True))
    names = list(sorted_customer_counts.keys())
    values = list(sorted_customer_counts.values())
    sorted_colors = [colors[name] for name in names]
    plt.figure(figsize=(15, 8))
    plt.barh(names, values, color=sorted_colors)
    plt.title('Number of Customers per Cluster')
    plt.xlabel('Number of Customers')
    plt.ylabel('Cluster')
    plt.gca().invert_yaxis()

try:
    data: pd.DataFrame = get_data()
    centroids, clusters = scale_and_cluster_features(data)
    sorted_indices, label_map = sort_and_label_clusters(centroids)
    plot_clusters(data, clusters, centroids, sorted_indices, label_map)
    plot_customer_counts_bar_chart(clusters, label_map)
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")
