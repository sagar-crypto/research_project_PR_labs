import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import PCA

def cluster_latent_features(latents: np.ndarray,
                            min_cluster_size: int = 5,
                            out_path: str = 'clusters.png') -> np.ndarray:
    """
    Perform HDBSCAN clustering on the latent space, visualize in 2D via PCA,
    and save the scatter plot with cluster labels.

    Args:
        latents: array of shape (n_samples, n_features)
        min_cluster_size: the minimum size of clusters in HDBSCAN
        out_path: filename to save the plot

    Returns:
        labels: cluster labels (-1 for noise)
    """
    # 1) reduce to 2D with PCA for visualization
    pca = PCA(n_components=2)
    pts2d = pca.fit_transform(latents)

    # 2) cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(pts2d)

    # 3) plot clusters
    plt.figure()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            # noise
            color = 'k'
            marker = 'x'
            label_name = 'noise'
        else:
            color = None
            marker = 'o'
            label_name = f'cluster {label}'
        plt.scatter(pts2d[mask,0], pts2d[mask,1], label=label_name, s=5, c=color, marker=marker)
    plt.legend(markerscale=2)
    plt.title('Latent Space HDBSCAN Clustering')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(out_path)
    return labels
