import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances

def fit_and_predict_clusters(data_sample, new_data, n_clusters=3):
    """
    Fits agglomerative clustering to a data sample and predicts the cluster for new data points.
    
    Parameters:
    - data_sample: 2D array-like, the sample data to fit the clustering algorithm.
    - new_data: 2D array-like, the new data points to predict their cluster.
    - n_clusters: int, the number of clusters.
    
    Returns:
    - cluster_labels: list of predicted cluster labels for the new data.
    """
    
    # Fit Agglomerative Clustering to the sample data
    agglom = AgglomerativeClustering(n_clusters=n_clusters)
    agglom.fit(data_sample)
    
    # Extract the cluster labels for the sample data
    sample_labels = agglom.labels_
    
    # Calculate the centroids of the clusters in the sample data
    centroids = np.array([
        data_sample[sample_labels == i].mean(axis=0) for i in range(n_clusters)
    ])
    
    # Calculate the Euclidean distance between new data points and centroids
    distances = euclidean_distances(new_data, centroids)
    
    # Find the closest centroid for each new data point
    cluster_labels = np.argmin(distances, axis=1)
    
    return cluster_labels
