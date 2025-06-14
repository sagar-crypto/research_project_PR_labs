"""
Cluster from precomputed latent .npy files: fuse multiple measurements' latents,
run HDBSCAN clustering, and plot clusters to a PNG.
"""
import os
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt

try:
    from umap import UMAP
    _REDUCER = 'umap'
except ImportError:
    from sklearn.decomposition import PCA
    _REDUCER = 'pca'


def main(args):
    # Load latent arrays
    latent_list = []
    for meas in args.measurements:
        base = meas.replace(' ', '_')
        fn = os.path.join(args.latent_dir, f"{base}_latents.npy")
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"Missing latent file: {fn}")
        print(f"Loading {fn}")
        latent_list.append(np.load(fn))

    # Ensure same number of windows
    N = latent_list[0].shape[0]
    for arr in latent_list:
        if arr.shape[0] != N:
            raise ValueError("Mismatch in window count across latent files")

    # Fuse features
    fused = np.concatenate(latent_list, axis=1)
    print(f"Fused feature shape: {fused.shape}")

    # Scale features
    fused_scaled = StandardScaler().fit_transform(fused)

    # Cluster with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
    labels = clusterer.fit_predict(fused_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"HDBSCAN found {n_clusters} clusters (+ noise)")

    # Dimensionality reduction for plotting
    if _REDUCER == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    emb = reducer.fit_transform(fused_scaled)

    # Plot
    plt.figure(figsize=(8, 6))
    unique = np.unique(labels)
    for lab in unique:
        mask = labels == lab
        x = emb[mask][:, 0] # type: ignore
        y = emb[mask][:, 1] # type: ignore
        if lab == -1:
            plt.scatter(x, y, s=10, color='lightgray', label='noise')
        else:
            plt.scatter(x, y, s=10, label=f'cluster {lab}')
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('HDBSCAN Clusters')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(f'{args.output_png}/all_clusters.png')
    print(f"Saved cluster plot to {args.output_png}")

    # Save fused features and labels if requested
    if args.output_fused:
        np.save(f"{args.output_fused}/fused.npy", fused_scaled)
        print(f"Saved fused latents to {args.output_fused}")
    if args.output_labels:
        np.save(f"{args.output_labels}/labels.npy", labels)
        print(f"Saved cluster labels to {args.output_labels}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster and plot using latent .npy files")
    parser.add_argument(
        '--latent_dir', type=str, required=True,
        help='Directory where <measurement>_latents.npy files live'
    )
    parser.add_argument(
        '--measurements', nargs='+', required=True,
        help='Measurement names, e.g. "Sekundärstrom L1 in A"'
    )
    parser.add_argument(
        '--min_cluster_size', type=int, default=5,
        help='HDBSCAN min_cluster_size'
    )
    parser.add_argument(
        '--output_fused', type=str, default=None,
        help='Optional: .npy path to save fused latent features'
    )
    parser.add_argument(
        '--output_labels', type=str, default=None,
        help='Optional: .npy path to save cluster labels'
    )
    parser.add_argument(
        '--output_png', type=str, default='cluster.png',
        help='Output PNG file for cluster plot'
    )
    args = parser.parse_args()
    main(args)
