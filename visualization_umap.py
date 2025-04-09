import matplotlib.pyplot as plt
import numpy as np

def visualize_dimensionality_reduction_umap(transformation, targets):
    """Visualizes the output of a UMAP transformation using a scatter plot.
    
    Args:
        transformation (np.ndarray): The 2D array resulting from the UMAP transformation.
        targets (array-like): The class labels for each point in the dataset.
    Returns:
        None, but a umap plot is produced
    """
    # Ensure targets are numpy array
    targets = np.array(targets)
    
    # Create a scatter plot of the UMAP output
    plt.figure(figsize=(10, 8))
    
    # Number of unique classes
    num_classes = len(np.unique(targets))
    
    # Create a colormap with enough distinct colors
    colormap = plt.get_cmap('tab10', num_classes)
    
    # Plot each class with a different color
    for i, label in enumerate(np.unique(targets)):
        plt.scatter(transformation[targets == label, 0], transformation[targets == label, 1], 
                    c=np.array([colormap(i)]), label=label, alpha=0.7, edgecolors='w', s=50)
    
    # Add a legend
    plt.legend(title='Classes', loc='best')
    
    
    plt.show()
    
