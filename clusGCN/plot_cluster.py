import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

def plot_loss(loss_history, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', label='Loss')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(loss_path)
    print(f"Saved loss plot to {loss_path}")
    plt.close()

def plot_clusters(features, labels, save_dir, title="Clusters Visualization"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_features[:, 0],
        reduced_features[:, 1],
        c=labels,
        cmap="viridis",
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    cluster_path = os.path.join(save_dir, "cluster_visualization.png")
    plt.savefig(cluster_path)
    print(f"Saved cluster visualization to {cluster_path}")
    plt.close()
