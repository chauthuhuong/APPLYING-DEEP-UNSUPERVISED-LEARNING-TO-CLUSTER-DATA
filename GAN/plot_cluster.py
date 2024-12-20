import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_latent_space(latent_vectors, labels, title="Latent Space Visualization"):
    pca = PCA(n_components=2)
    reduced_latent = pca.fit_transform(latent_vectors)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_latent[:, 0],
        reduced_latent[:, 1],
        c=labels,
        cmap="viridis",
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()
