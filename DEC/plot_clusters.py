import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_loss_and_clusters(loss_history, embeddings, labels, save_dir, n_clusters=10, method='tsne', title="Clustering Results"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not loss_history or len(loss_history) == 0:
        print("Warning: Loss history is empty. Skipping loss plot.")
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, label="Loss", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss over Iterations")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "loss_plot.png"))
        plt.close()
        print(f"Loss plot saved to: {os.path.join(save_dir, 'loss_plot.png')}")

    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings are empty. Cannot perform dimensionality reduction.")
    if labels is None or len(labels) == 0:
        raise ValueError("Labels are empty. Cannot create cluster plot.")
    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values. Check the input data.")

    print(f"Applying {method.upper()} for dimensionality reduction...")
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        max_perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=max_perplexity)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    reduced_data = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10', s=10, alpha=0.8)
    plt.colorbar(scatter, ticks=range(n_clusters))
    plt.title(title, fontsize=16)
    plt.xlabel("Component 1" if method == 'pca' else "Dimension 1")
    plt.ylabel("Component 2" if method == 'pca' else "Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    cluster_plot_path = os.path.join(save_dir, f"{method}_cluster_plot.png")
    plt.savefig(cluster_plot_path)
    plt.close()
    print(f"Cluster visualization saved to: {cluster_plot_path}")
