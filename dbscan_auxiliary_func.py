import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors



def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    """Plots a graph to help find out what is the best value for epsilon in the DBSCAN function.

    Args:
        X: Dataset (array-like or matrix)
        k (int): The value you intend to use as k in the DBSCAN function
        radius_nbrs (float, optional): Radius parameter for nearest neighbors. Defaults to 1.0.
        
    Returns:
    None, but a plot is produced.
    """

    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X) 
                                       
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()
