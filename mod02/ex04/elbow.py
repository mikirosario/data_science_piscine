from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Database connection parameters
dbname = 'piscineds'
user = 'mrosario'
password = 'mysecretpassword'
host = 'localhost'
port = '5432'
# PostgreSQL connection string in SQLAlchemy format
connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

def get_avg_spend_per_customer() -> pd.DataFrame:
    query = """
            SELECT user_id,
            AVG(price::NUMERIC) AS avg_spend_per_purchase
            FROM customers
            WHERE event_type = 'purchase'
            GROUP BY user_id;
            """
    try:
        engine = create_engine(connection_string)
        df: pd.DataFrame = pd.read_sql_query(query, engine)
    finally:
        engine.dispose()
    return df

def show_scatter_plot(avg_spend_df, num_clusters):
    # Step 1: Initialize KMeans with 3 clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # Step 2: Fit KMeans to the data
    # The column name should be 'avg_spend_per_purchase'
    kmeans.fit(avg_spend_df[['avg_spend_per_purchase']])
    # Step 3: Retrieve the labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    # Before plotting
    print("Centroids:", centroids)
    # Step 4: Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(avg_spend_df.index, avg_spend_df['avg_spend_per_purchase'], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], [0]*len(centroids), c='red', marker='x', s=100, label='Centroids')
    plt.title('K-means Clustering with 3 Clusters')
    plt.xlabel('Customer Index')
    plt.ylabel('Average Spend Per Purchase')
    plt.legend()
    plt.show()

# Within-cluster sum of squares (WCSS)
# is a measure of variability within each cluster;
# it's the sum of the squared differences between 
# each data point and the centroid of its cluster.

# n_clusters: Number of clusters and centroids to generate. This is the value you vary to observe its impact on WCSS.
# k-means++: Initialization method to spread the initial centroids of the clusters.
# random_state: Seed that places first centroid.
def show_elbow_chart(avg_spend_df):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(avg_spend_df[['avg_spend_per_purchase']])
        wcss.append(kmeans.inertia_)
    # Plotting the results onto a line graph to observe 'The elbow'
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

try:
    avg_spend_per_customer = get_avg_spend_per_customer()
    #show_elbow_chart(avg_spend_per_customer)
    show_scatter_plot(avg_spend_per_customer, 3)
except Exception as e:
    print(f"An error occurred: {e}")
