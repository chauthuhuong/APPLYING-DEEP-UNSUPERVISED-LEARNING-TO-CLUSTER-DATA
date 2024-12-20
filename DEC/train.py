import os
import numpy as np
from sklearn.cluster import KMeans
from model import build_conv_autoencoder, build_fc_autoencoder, build_dec_model, get_data_augmenter
from data import load_mnist, load_mnist_test  
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from metrics import calculate_metrics
from plot_clusters import plot_loss_and_clusters

def target_distribution(q):
    weight = q ** 2 / np.sum(q, axis=0)
    return (weight.T / np.sum(weight, axis=1)).T

def train_model(dataset='mnist', 
                model_type='conv', 
                n_clusters=10, 
                input_dim=784, 
                input_shape=(28, 28, 1), 
                batch_size=256, 
                pretrain_epochs=100, 
                maxiter=30000, 
                update_interval=140, 
                tol=0.001, 
                save_dir='./results'):

    if dataset == 'mnist':
        x, y = load_mnist()
    elif dataset == 'mnist-test':
        x, y = load_mnist_test()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    #Chọn autoencoder
    if model_type == 'conv':
        autoencoder, encoder = build_conv_autoencoder(input_shape=input_shape, latent_dim=n_clusters)
    elif model_type == 'fc':
        autoencoder, encoder = build_fc_autoencoder(input_dim=input_dim, latent_dim=n_clusters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    autoencoder.compile(optimizer=Adam(), loss='mse')
    autoencoder.summary()
    """====DA options====
    data_augmenter = get_data_augmenter()
    data_gen = data_augmenter.flow(x, x, batch_size=batch_size)

    print("=== Pretraining Autoencoder with Data Augmentation ===")
    history = autoencoder.fit(
        data_gen,
        steps_per_epoch=len(x) // batch_size,
        epochs=pretrain_epochs,
        verbose=1
    )
    """
    #1.Pretraining
    print("=== Pretraining Autoencoder ===")
    autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs, verbose=1)

    #2.Khởi tạo cụm  trong không gian nhúng
    print("=== Initializing Clustering ===")
    features = encoder.predict(x)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(features)
    y_pred_last = y_pred.copy()
    print(f"K-Means initialized clusters: {np.unique(y_pred)}")

    dec_model = build_dec_model(autoencoder, encoder, n_clusters)
    dec_model.compile(optimizer=Adam(learning_rate=0.001), loss='kld')

    #Gán weight của CL 
    dec_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    #3.Fine-tuning
    print("=== Fine-tuning Clustering ===")
    index = 0
    loss_history = []  

    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = dec_model.predict(x, verbose=0)
            p = target_distribution(q)

            #Kiểm tra tiêu chí hội tụ
            y_pred = np.argmax(q, axis=1)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            loss = dec_model.evaluate(x, p, verbose=0)
            loss_history.append(loss) 
            print(f"Iter {ite}: delta_label={delta_label:.4f}, loss={loss:.4f}")

            if delta_label < tol:
                print("Convergence reached!")
                break

        #Train theo batch
        idx = np.random.randint(0, x.shape[0], batch_size)
        loss_batch = dec_model.train_on_batch(x=x[idx], y=p[idx])
        loss_history.append(loss_batch) 
        """====DA option
        augmented_data_gen = data_augmenter.flow(x, p, batch_size=batch_size)
        for _ in range(len(x) // batch_size):
            x_batch, p_batch = next(augmented_data_gen)
            loss_batch = dec_model.train_on_batch(x=x_batch, y=p_batch)
            loss_history.append(loss_batch)   
        """
    #4
    print("=== Evaluating Clustering ===")
    q = dec_model.predict(x, verbose=0)
    y_pred = np.argmax(q, axis=1)
    acc = np.sum(y_pred == y) / y.shape[0]
    print(f"Final ACC: {acc:.4f}")

    nmi, ari, _ = calculate_metrics(y, y_pred)
    print(f"NMI: {nmi:.4f}, ARI: {ari:.4f}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    #Lưu weight
    model_path = os.path.join(save_dir, 'dec_model.weights.h5')
    dec_model.save_weights(model_path)
    print(f"Model weights saved to: {model_path}")

    print("Extracting features for visualization...")
    features = encoder.predict(x)

    print("=== Visualizing Loss and Clusters ===")
    plot_loss_and_clusters(
        loss_history=loss_history, 
        embeddings=features, 
        labels=y_pred, 
        save_dir=save_dir, 
        n_clusters=n_clusters, 
        method='tsne',  #pca hoặc tsne
        title="DEC Clustering Results (t-SNE)"
    )


if __name__ == "__main__":
    train_model(dataset='mnist', 
                model_type='conv', 
                n_clusters=10, 
                input_shape=(28, 28, 1), 
                batch_size=256, 
                pretrain_epochs=100, 
                maxiter=30000, 
                update_interval=140, 
                tol=0.001, 
                save_dir='./results')
