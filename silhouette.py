import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_silhouette(X, max_clusters=15):
    '''
    Calculates silhouette scores and plots a silhouette graph
    Arguments: 
    - X: dataset to use in KMeans.
    - max_clusters: maximum number of clusters that is calculate (from 2 to max_clusters), default is 10
    Returns:
    None, but a silhouette plot is produced.
    '''
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette Method for Optimal k')
    plt.show()