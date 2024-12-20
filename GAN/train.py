import numpy as np
import tensorflow as tf
from data import load_mnist_data
from model import build_generator, build_discriminator, build_encoder
from metrics import calculate_metrics
from plot_cluster import plot_latent_space
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

LATENT_DIM = 128
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.0005
SAVE_DIR = "./results" 
REC_LOSS_WEIGHT = 10.0 

# Ensure save directory exists
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Load data
def prepare_data():
    x_train, y_train = load_mnist_data()
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)
    return dataset, y_train

def train():
    x_train, y_train = load_mnist_data()
    dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(BATCH_SIZE)

    generator = build_generator(LATENT_DIM)
    discriminator = build_discriminator()
    encoder = build_encoder(LATENT_DIM)

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    enc_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    bce = tf.keras.losses.BinaryCrossentropy()

    gan_losses = []
    rec_losses = []

    for epoch in range(EPOCHS):
        gan_loss_epoch, rec_loss_epoch = 0, 0

        for real_images in dataset:
            batch_size = real_images.shape[0]
            z = tf.random.normal([batch_size, LATENT_DIM])
            fake_images = generator(z)

            with tf.GradientTape() as tape:
                real_logits = discriminator(real_images)
                fake_logits = discriminator(fake_images)
                disc_loss = bce(tf.ones_like(real_logits), real_logits) + bce(tf.zeros_like(fake_logits), fake_logits)
            grads = tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            with tf.GradientTape() as tape:
                fake_images = generator(z)
                fake_logits = discriminator(fake_images)
                gen_loss = bce(tf.ones_like(fake_logits), fake_logits)
            grads = tape.gradient(gen_loss, generator.trainable_variables)
            gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

            with tf.GradientTape() as tape:
                encoded_z = encoder(real_images) + tf.random.normal(shape=tf.shape(encoder(real_images)), stddev=0.1)
                recon_images = generator(encoded_z)
                rec_loss = tf.reduce_mean(tf.abs(real_images - recon_images))
            grads = tape.gradient(rec_loss, encoder.trainable_variables)
            enc_optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

            gan_loss_epoch += gen_loss.numpy()
            rec_loss_epoch += REC_LOSS_WEIGHT * rec_loss.numpy()

        gan_loss_epoch /= len(dataset)
        rec_loss_epoch /= len(dataset)

        gan_losses.append(gan_loss_epoch)
        rec_losses.append(rec_loss_epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS}, GAN Loss: {gan_loss_epoch:.4f}, Reconstruction Loss: {rec_loss_epoch:.4f}")

    encoded_latent = encoder(x_train).numpy()
    pca = PCA(n_components=2) 
    encoded_latent_pca = pca.fit_transform(encoded_latent)

    kmeans = KMeans(n_clusters=10, random_state=42)
    y_pred = kmeans.fit_predict(encoded_latent_pca)

    metrics = calculate_metrics(y_train, y_pred)
    print(f"NMI: {metrics['NMI']:.4f}, ARI: {metrics['ARI']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")

    # Vẽ không gian tiềm ẩn với nhãn thực và dự đoán
    plot_latent_space(encoded_latent_pca, y_train, save_dir=SAVE_DIR, title="True Labels in Latent Space")
    plot_latent_space(encoded_latent_pca, y_pred, save_dir=SAVE_DIR, title="Predicted Clusters in Latent Space")

    return gan_losses, rec_losses

def plot_losses(gan_losses, rec_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(gan_losses) + 1), gan_losses, label="GAN Loss")
    plt.plot(range(1, len(rec_losses) + 1), rec_losses, label="Reconstruction Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))
    plt.close()
    print("Loss plot saved.")

if __name__ == "__main__":
    gan_losses, rec_losses = train()
    plot_losses(gan_losses, rec_losses)
