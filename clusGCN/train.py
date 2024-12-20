import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from data import load_mnist_data, normalize_data, construct_graph
from metrics import calculate_metrics
from model import GCN
from plot_cluster import plot_loss, plot_clusters

# Hyperparameters
NUM_CLUSTERS = 10
EPOCHS = 100
LEARNING_RATE = 0.001

def partition_graph(adjacency, num_clusters):
    features_sum = adjacency.toarray().sum(axis=1).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_sum)
    cluster_map = {i: clusters[i] for i in range(len(clusters))}
    return cluster_map

def create_batches(features, cluster_map):
    batches = []
    for i in range(NUM_CLUSTERS):
        indices = [node for node, cluster in cluster_map.items() if cluster == i]
        if len(indices) > 0:
            batch_features = features[indices]
            batches.append((batch_features, indices))
    return batches

def train_gcn(batches, labels):
    input_dim = batches[0][0].shape[1]
    hidden_dims = [512, 256, 128, 64]  
    output_dim = NUM_CLUSTERS

    model = GCN(input_dim, hidden_dims, output_dim, dropout_rate=0.3)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_history = []
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch, indices in batches:
            batch_labels = labels[indices]
            with tf.GradientTape() as tape:
                logits = model(batch, training=True)
                loss = tf.reduce_mean(loss_fn(batch_labels, logits))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss += loss.numpy()
        loss_history.append(epoch_loss / len(batches))
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(batches):.4f}")
    return model, loss_history

def main():
    save_dir = "./results"  
    x_train, y_train = load_mnist_data()
    x_train = normalize_data(x_train)

    print("Constructing graph...")
    adjacency = construct_graph(x_train, k=10)

    print("Partitioning graph...")
    cluster_map = partition_graph(adjacency, NUM_CLUSTERS)

    print("Creating batches...")
    batches = create_batches(x_train, cluster_map)

    print("Training GCN...")
    model, loss_history = train_gcn(batches, y_train)

    print("Saving and plotting loss...")
    plot_loss(loss_history, save_dir)

    print("Evaluating clustering...")
    logits = model(x_train, training=False)
    y_pred = np.argmax(logits.numpy(), axis=1)
    nmi, ari, acc = calculate_metrics(y_train, y_pred)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}")

    print("Saving and visualizing clusters...")
    plot_clusters(x_train, y_pred, save_dir, title="Predicted Clusters")

    print("All results saved in:", save_dir)

if __name__ == "__main__":
    main()
